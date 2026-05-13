# ==================== Analytical H-Scale Calibration ====================
"""
Finds the optimal diffusion scale factor s that minimises vol-space RMSE
without any gradient training.

Why this works
--------------
sigma_mod(s) ≈ s * sigma_mod(s=1)   (approximately linear in s)

The OLS solution is therefore analytic:
    s_opt = sum(sigma_mod_i * sigma_mkt_i) / sum(sigma_mod_i^2)

The scale is applied via the `diffusion_scale` argument of
simulate_latent_paths(), so reconstruction (which uses the ODE,
not the simulation) is completely unaffected.

Output
------
- Console table: per-(expiry, tenor) vol comparison at s=1 and s=s_opt
- Sweep plot: vol MAE vs diffusion_scale
- Calibrated checkpoint with H weights scaled by s_opt (for consistency)
"""

import math
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code import config
config.confirm_variant()

from Code.utils import helpers as H_utils
from Code.load_swapdata import my_data
from Code.model.full_model_stable import FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import swap_rate_torch, forward_swap_rate_torch
from Code.Simulation.simulate_model import simulate_latent_paths, compute_discount_paths

print("Active variant:", config.VARIANT)
device = torch.device("cpu")

# ================================================================
# Settings
# ================================================================

LATENT_DIM   = 4
N_PATHS      = 2048     # more paths → lower MC error on each vol estimate
DT           = 1 / 12   # monthly steps
CCY_FILTER   = "EUR"
USE          = "bbg"

# Warm-start checkpoint (reconstruction-only, clean baseline)
PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ================================================================
# Load model
# ================================================================

model = FullModel(latent_dim=LATENT_DIM).to(device)
raw = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
model.eval()
print(f"Loaded warm-start: {PRETRAIN_CKPT}")

# ================================================================
# Load data
# ================================================================

meta, X_tensor, *_ = my_data(use=USE)
X_tensor = X_tensor.float()

meta_eur = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_eur    = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0

dates_swap = set(pd.to_datetime(meta_eur["as_of_date"]).dt.normalize())
df_vol = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {
    pd.Timestamp(row["as_of_date"]).normalize(): i
    for i, row in meta_eur.iterrows()
}

print(f"EUR obs: {len(meta_eur)} dates, {len(df_vol)} swaption vol targets")

# ================================================================
# Core: price one swaption given a diffusion_scale
# ================================================================

sqrt_2pi = math.sqrt(2.0 * math.pi)

@torch.no_grad()
def price_swaption(date, expiry, tenor, diffusion_scale=1.0):
    """
    Returns (sigma_mod_bp, sigma_mkt_bp, path_finite_frac) or None on failure.
    """
    if date not in date_to_idx:
        return None
    idx = date_to_idx[date]
    xb  = X_eur[idx:idx+1].to(device)
    z0  = model.encoder(xb)

    n_steps = max(12, int(round(expiry / DT)))
    dt_eff  = expiry / n_steps

    # Antithetic variates
    half     = N_PATHS // 2
    z_paths, r_paths, _, _ = simulate_latent_paths(
        model, z0, n_paths=half, n_steps=n_steps, dt=dt_eff,
        device=device, diffusion_scale=diffusion_scale,
    )
    z_anti, r_anti, _, _ = simulate_latent_paths(
        model, z0, n_paths=half, n_steps=n_steps, dt=dt_eff,
        device=device, diffusion_scale=diffusion_scale,
    )

    z_T = torch.cat([z_paths[:, -1, :], z_anti[:, -1, :]], dim=0)
    r_full = torch.cat([r_paths, r_anti], dim=0)

    disc_paths = compute_discount_paths(r_full, dt_eff)
    D_T = disc_paths[:, -1]

    # Two-pass: probe then grad-free decode on survivors
    finite_mask = torch.isfinite(z_T).all(dim=1) & torch.isfinite(D_T)
    n_finite = int(finite_mask.sum())
    if n_finite < 32:
        return None

    z_keep = z_T[finite_mask]
    D_keep = D_T[finite_mask]

    _, aux_T = model.decode_from_z(z_keep, tau=None, return_aux=True)
    P_T = aux_T["P_full"]

    fa_ok = torch.isfinite(P_T).all(dim=1)
    if int(fa_ok.sum()) < 32:
        return None

    F_T, A_T = swap_rate_torch(P_T[fa_ok], tenor=tenor)
    D_use     = D_keep[fa_ok]

    fa_finite = torch.isfinite(F_T) & torch.isfinite(A_T)
    if int(fa_finite.sum()) < 32:
        return None
    F_T = F_T[fa_finite]; A_T = A_T[fa_finite]; D_use = D_use[fa_finite]

    # Time-0 anchor
    _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
    F0, A0 = forward_swap_rate_torch(aux0["P_full"][0], expiry, tenor)
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 0):
        return None

    payoff = A_T * torch.relu(F_T - F0)
    V_MC   = (D_use * payoff).mean()
    if not torch.isfinite(V_MC):
        return None

    sigma_mod_bp = float(V_MC) * sqrt_2pi / (A0 * math.sqrt(expiry)) * 10_000.0
    path_frac    = n_finite / N_PATHS
    return sigma_mod_bp, path_frac


# ================================================================
# Step 1: compute baseline vols at diffusion_scale = 1
# ================================================================

print("\n=== Step 1: pricing at diffusion_scale = 1 ===")

records = []
combos  = df_vol[["as_of_date", "option_maturity", "swap_tenor", "market_vol"]].drop_duplicates()

for _, row in combos.iterrows():
    date    = pd.Timestamp(row["as_of_date"]).normalize()
    expiry  = int(row["option_maturity"])
    tenor   = int(row["swap_tenor"])
    sig_mkt = float(row["market_vol"]) * 10_000.0

    result = price_swaption(date, expiry, tenor, diffusion_scale=1.0)
    if result is None:
        continue
    sig_mod, pfrac = result
    if not math.isfinite(sig_mod) or sig_mod <= 0:
        continue

    records.append({
        "date":       date.date(),
        "expiry":     expiry,
        "tenor":      tenor,
        "sigma_mkt":  round(sig_mkt,  1),
        "sigma_mod1": round(sig_mod,  1),
        "ratio":      round(sig_mod / sig_mkt, 3) if sig_mkt > 0 else None,
        "path_frac":  round(pfrac, 3),
    })
    ratio_str = f"ratio={sig_mod/sig_mkt:.2f}  " if sig_mkt > 0 else ""
    print(f"  {expiry}x{tenor}  mkt={sig_mkt:.0f}bp  mod={sig_mod:.0f}bp  "
          f"{ratio_str}paths={pfrac*100:.0f}%")

df_base = pd.DataFrame(records)
print(f"\nPriced {len(df_base)} / {len(combos)} swaptions at s=1")
df_base.to_csv(os.path.join(OUT_DIR, "baseline_vols_s1.csv"), index=False)

if df_base.empty:
    print("ERROR: no swaptions priced. Check data / checkpoint path.")
    sys.exit(1)

# ================================================================
# Step 2: analytic OLS for s_opt
# ================================================================

sig_mod_arr = df_base["sigma_mod1"].values
sig_mkt_arr = df_base["sigma_mkt"].values

# sigma_mod(s) ≈ s * sigma_mod_1  →  OLS: s_opt = Σ(mod*mkt) / Σ(mod²)
s_ols = float(np.dot(sig_mod_arr, sig_mkt_arr) / np.dot(sig_mod_arr, sig_mod_arr))
print(f"\nOLS scale factor  s_opt = {s_ols:.4f}")
print(f"Median ratio mkt/mod  = {np.median(sig_mkt_arr/sig_mod_arr):.4f}")

# ================================================================
# Step 3: sweep s and compute predicted vol MAE (using linearity)
# ================================================================

s_grid = np.linspace(max(0.01, s_ols * 0.1), s_ols * 3.0, 200)
mae_grid = [np.mean(np.abs(s * sig_mod_arr - sig_mkt_arr)) for s in s_grid]

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.plot(s_grid, mae_grid, lw=1.5, color="steelblue")
ax.axvline(s_ols, color="firebrick", ls="--", lw=1.2, label=f"s_opt={s_ols:.3f}")
ax.axvline(1.0,   color="grey",     ls=":",  lw=1.0, label="s=1 (baseline)")
ax.set_xlabel("diffusion_scale  s"); ax.set_ylabel("Vol MAE (bp)")
ax.set_title("Vol MAE vs diffusion_scale (predicted via linearity)")
ax.grid(True, alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "vol_mae_sweep.png"), dpi=200)
plt.close(fig)
print("Saved sweep plot.")

# ================================================================
# Step 4: verify at s_opt by actually re-running MC
# ================================================================

print(f"\n=== Step 4: verifying at diffusion_scale = {s_ols:.4f} ===")

verify_records = []
for _, row in df_base.iterrows():
    date   = pd.Timestamp(row["date"])
    expiry = int(row["expiry"])
    tenor  = int(row["tenor"])
    sig_mkt = float(row["sigma_mkt"])

    result = price_swaption(date, expiry, tenor, diffusion_scale=s_ols)
    if result is None:
        continue
    sig_mod_cal, pfrac = result
    if not math.isfinite(sig_mod_cal):
        continue
    verify_records.append({
        "date":           date.date(),
        "expiry":         expiry,
        "tenor":          tenor,
        "sigma_mkt":      round(sig_mkt,         1),
        "sigma_mod_s1":   round(row["sigma_mod1"],1),
        "sigma_mod_scal": round(sig_mod_cal,      1),
        "err_s1_bp":      round(row["sigma_mod1"] - sig_mkt, 1),
        "err_cal_bp":     round(sig_mod_cal       - sig_mkt, 1),
    })

df_cal = pd.DataFrame(verify_records)

if not df_cal.empty:
    mae_s1  = df_cal["err_s1_bp"].abs().mean()
    mae_cal = df_cal["err_cal_bp"].abs().mean()
    rmse_s1  = (df_cal["err_s1_bp"]  ** 2).mean() ** 0.5
    rmse_cal = (df_cal["err_cal_bp"] ** 2).mean() ** 0.5

    print(f"\n{'':>25}  {'s=1':>10}  {'s=s_opt':>10}")
    print(f"{'Vol MAE (bp)':>25}  {mae_s1:>10.1f}  {mae_cal:>10.1f}")
    print(f"{'Vol RMSE (bp)':>25}  {rmse_s1:>10.1f}  {rmse_cal:>10.1f}")
    print()

    # Per-(expiry, tenor) summary
    pivot = df_cal.groupby(["expiry", "tenor"]).agg(
        mkt_bp    = ("sigma_mkt",      "mean"),
        mod_s1    = ("sigma_mod_s1",   "mean"),
        mod_cal   = ("sigma_mod_scal", "mean"),
        err_s1    = ("err_s1_bp",      lambda x: x.abs().mean()),
        err_cal   = ("err_cal_bp",     lambda x: x.abs().mean()),
    ).reset_index()

    print(f"{'Exp':>4} {'Ten':>4}  {'mkt(bp)':>8}  "
          f"{'mod s=1':>8}  {'err s=1':>8}  "
          f"{'mod cal':>8}  {'err cal':>8}")
    print("-" * 65)
    for _, r in pivot.iterrows():
        print(f"{int(r.expiry):>4} {int(r.tenor):>4}  {r.mkt_bp:>8.0f}  "
              f"{r.mod_s1:>8.0f}  {r.err_s1:>8.0f}  "
              f"{r.mod_cal:>8.0f}  {r.err_cal:>8.0f}")

    df_cal.to_csv(os.path.join(OUT_DIR, "calibrated_vols.csv"), index=False)
    pivot.to_csv(os.path.join(OUT_DIR, "calibrated_vols_summary.csv"), index=False)

# ================================================================
# Step 5: reconstruction RMSE before and after applying s_opt to weights
# ================================================================

print("\n=== Step 5: reconstruction RMSE with H weights scaled by s_opt ===")

@torch.no_grad()
def rmse_bps_all(mdl):
    outs = []
    mdl.eval()
    for i in range(0, X_tensor.shape[0], 256):
        xb = X_tensor[i:i+256].to(device)
        outs.append(mdl(xb).detach().cpu())
    S_hat = torch.cat(outs)
    mask  = torch.isfinite(X_tensor).all(1) & torch.isfinite(S_hat).all(1)
    rmse_per_ccy = H_utils.rmse_bps_per_currency_paper(
        X_tensor[mask], S_hat[mask],
        meta.loc[mask.numpy()].reset_index(drop=True),
    )
    return rmse_per_ccy

rmse_before = rmse_bps_all(model)
print(f"RMSE before H scaling  — avg: {rmse_before.mean():.2f} bp")
for ccy, v in rmse_before.items():
    print(f"  {ccy}: {v:.2f} bp")

# Scale last linear layer of H by s_opt
import copy
model_cal = copy.deepcopy(model)
_last = None
for m in model_cal.H.modules():
    if isinstance(m, torch.nn.Linear):
        _last = m
if _last is not None:
    with torch.no_grad():
        _last.weight.mul_(s_ols)
        if _last.bias is not None:
            _last.bias.mul_(s_ols)
    print(f"\nScaled H last linear layer by s_opt={s_ols:.4f}")

rmse_after = rmse_bps_all(model_cal)
print(f"RMSE after  H scaling  — avg: {rmse_after.mean():.2f} bp")
for ccy, v in rmse_after.items():
    print(f"  {ccy}: {v:.2f} bp")

# ================================================================
# Step 6: save calibrated checkpoint
# ================================================================

ckpt_path = os.path.join(OUT_DIR, f"checkpoint_dim{LATENT_DIM}_hscale_{s_ols:.4f}.pt")
torch.save({
    "model_state_dict": model_cal.state_dict(),
    "model_config":     {"latent_dim": LATENT_DIM},
    "latent_dim":       LATENT_DIM,
    "variant":          config.VARIANT,
    "h_diffusion_scale": s_ols,
    "rmse_before_bp":   float(rmse_before.mean()),
    "rmse_after_bp":    float(rmse_after.mean()),
    "vol_mae_s1_bp":    float(mae_s1)    if not df_cal.empty else None,
    "vol_mae_cal_bp":   float(mae_cal)   if not df_cal.empty else None,
}, ckpt_path)
print(f"\nSaved calibrated checkpoint: {ckpt_path}")

# ================================================================
# Step 7: save summary JSON
# ================================================================

summary = {
    "s_ols":               round(s_ols, 6),
    "n_swaptions_priced":  len(df_base),
    "vol_mae_s1_bp":       round(float(mae_s1),    2) if not df_cal.empty else None,
    "vol_mae_cal_bp":      round(float(mae_cal),   2) if not df_cal.empty else None,
    "vol_rmse_s1_bp":      round(float(rmse_s1),   2) if not df_cal.empty else None,
    "vol_rmse_cal_bp":     round(float(rmse_cal),  2) if not df_cal.empty else None,
    "recon_rmse_before_bp":round(float(rmse_before.mean()), 2),
    "recon_rmse_after_bp": round(float(rmse_after.mean()),  2),
}
with open(os.path.join(OUT_DIR, "calibration_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary:", json.dumps(summary, indent=2))
