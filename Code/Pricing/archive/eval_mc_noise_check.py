# ==================== Monte Carlo Noise Diagnostic ====================
"""
High-path evaluation of the Constant MPR model to separate structural
pricing error from Monte Carlo estimator noise.

Question: Is the 41 bp MAE real model error, or partly noise from 512 paths?

Method:
  - Evaluate Constant MPR with N_PATHS_EVAL paths (5 000 or 10 000)
  - Repeat with 3 fixed seeds
  - Report mean ± std of MAE across seeds at cell, expiry, and overall level

Decision rule after running:
  MAE drops to 25-30 bp  -> part of 41 bp was MC noise, evaluate finals with more paths
  MAE stays ~40 bp       -> error is structural, move to vol-surface model
  1Y >> 5Y/10Y           -> expiry-dependent vol scaling is the right next step

Output -> Figures/pricing/mc_noise_check/
"""

import math, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

from Code.load_swapdata import my_data
from Code.model.full_model_price import FullModelPrice as FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import swap_rate_torch, forward_swap_rate_torch
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable
from Code.model.sigma_matrix import L_from_sigmas_rhos

# ── settings ───────────────────────────────────────────────────────────────────
LATENT_DIM    = 4
N_PATHS_EVAL  = 10000      # 10 000 paths for reliable MC noise estimate
SEEDS         = [123, 456, 789]
TRAIN_FRAC    = 0.70
CCY_FILTER    = "EUR"
RATE_CLIP     = 0.50
ANNUITY_MAX   = 50.0

EXPIRY_VALS = [1, 5, 10]
TENOR_VALS  = [1, 5, 10]

PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt")
CMPR_CKPT     = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_constant_mpr", "ep1000",
                              "checkpoint_constant_mpr_ep1000.pt")
OUT_DIR       = os.path.join(PROJECT_ROOT, "Figures", "pricing",
                              f"mc_noise_check_{N_PATHS_EVAL}paths")
os.makedirs(OUT_DIR, exist_ok=True)

device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

print(f"High-path MC noise check")
print(f"  N_PATHS_EVAL = {N_PATHS_EVAL}")
print(f"  Seeds        = {SEEDS}")
print(f"  Output       = {OUT_DIR}")

# ── Constant MPR module ────────────────────────────────────────────────────────
class ConstantMPR(nn.Module):
    def __init__(self, kp_module, h_module, latent_dim):
        super().__init__()
        self.kp            = kp_module
        self.h             = h_module
        self.latent_dim    = latent_dim
        self.lambda_0      = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma_vec = nn.Parameter(torch.full((latent_dim,), -1.8))

    def forward(self, z_t):
        with torch.no_grad():
            k_base       = self.kp(z_t)
            sigmas, rhos = self.h(z_t)
            L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_0.unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    @property
    def sigma_vec(self):
        return self.log_sigma_vec.exp()

# ── load base model ────────────────────────────────────────────────────────────
print("\nLoading base model ...")
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

lm = ConstantMPR(model.K, model.H, LATENT_DIM).to(device)
print(f"Loading Constant MPR checkpoint: {CMPR_CKPT}")
raw_lm   = torch.load(CMPR_CKPT, map_location=device, weights_only=False)
lm_state = raw_lm.get("lm_state_dict", raw_lm)
lm.load_state_dict(lm_state)
lm.eval()

l0 = lm.lambda_0.detach().cpu().numpy()
sv = lm.sigma_vec.detach().cpu().numpy()
print(f"lambda_0  = {l0.round(4)}  (||.||={np.linalg.norm(l0):.4f})")
print(f"sigma_vec = {sv.round(4)}  (mean={sv.mean():.4f})")

# ── data ───────────────────────────────────────────────────────────────────────
meta, X_tensor, *_ = my_data(use="bbg")
X_tensor = X_tensor.float()
meta_eur = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_eur    = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0
df_vol = df_vol[df_vol["option_maturity"].isin(EXPIRY_VALS)].copy()

dates_swap  = set(pd.to_datetime(meta_eur["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_eur.iterrows()
}

all_dates   = sorted(df_vol["as_of_date"].unique())
n_train     = int(len(all_dates) * TRAIN_FRAC)
train_dates = set(all_dates[:n_train])
test_dates  = set(all_dates[n_train:])
print(f"\nEUR: {len(meta_eur)} curve dates, {len(all_dates)} vol dates")
print(f"Train: {n_train}  Test: {len(test_dates)}")

combos = (df_vol[["as_of_date", "option_maturity", "swap_tenor", "market_vol"]]
          .drop_duplicates().sort_values("as_of_date"))
print(f"Combos to price: {len(combos)}")

# ── pricing function ───────────────────────────────────────────────────────────
def price_cell(date, expiry, tenor, seed):
    if date not in date_to_idx:
        return None
    idx = date_to_idx[date]
    xb  = X_eur[idx:idx + 1].to(device)

    with torch.no_grad():
        z0 = model.encoder(xb)
        _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
        P0 = aux0["P_full"][0]

    max_idx = P0.shape[0] - 1
    if expiry + tenor > max_idx:
        return None
    F0, A0 = forward_swap_rate_torch(P0, expiry, tenor)
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 1e-6):
        return None

    dt_eff  = min(1.0 / 12.0, expiry / 10.0)
    n_steps = max(12, int(round(expiry / dt_eff)))
    half    = N_PATHS_EVAL // 2

    torch.manual_seed(seed)
    with torch.no_grad():
        eps_half = torch.randn(half, n_steps, LATENT_DIM, device=device)
        z_T, D_T = simulate_to_expiry_differentiable(
            model, z0, n_steps=n_steps, dt=dt_eff,
            n_paths=N_PATHS_EVAL, eps=eps_half,
            k_override=lm,
            sigma_scale=lm.sigma_vec,
            antithetic=True,
            freeze_H=True,
        )

    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 32:
        return None
    z_k, D_k = z_T[ok], D_T[ok]

    with torch.no_grad():
        _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True)
    P_T = aux_T["P_full"]
    dok = torch.isfinite(P_T).all(1)
    if dok.sum() < 32:
        return None

    F_T, A_T = swap_rate_torch(P_T[dok], tenor=tenor)
    D_k = D_k[dok]
    sane = (torch.isfinite(F_T) & torch.isfinite(A_T)
            & (F_T > -RATE_CLIP) & (F_T < RATE_CLIP)
            & (A_T > 1e-6) & (A_T < ANNUITY_MAX))
    if sane.sum() < 32:
        return None
    F_T, A_T, D_k = F_T[sane], A_T[sane], D_k[sane]

    V_pay = float((D_k * A_T * torch.relu(F_T - F0)).mean())
    V_rec = float((D_k * A_T * torch.relu(F0 - F_T)).mean())
    if not (math.isfinite(V_pay) and math.isfinite(V_rec)):
        return None

    sigma_str_bp    = (V_pay + V_rec) * 0.5 * sqrt_2pi / (A0 * math.sqrt(expiry)) * 1e4
    forward_bias_bp = (V_pay - V_rec) / A0 * 1e4
    path_frac       = float(ok.float().mean())

    return {
        "sigma_str_bp":    sigma_str_bp,
        "forward_bias_bp": forward_bias_bp,
        "path_frac":       path_frac,
    }

# ── run over all seeds ─────────────────────────────────────────────────────────
all_results = {}   # seed -> DataFrame

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"Seed {seed}  ({N_PATHS_EVAL} paths)")
    print(f"{'='*60}")
    t0   = time.time()
    rows = []

    for counter, (_, row) in enumerate(combos.iterrows()):
        date   = pd.Timestamp(row["as_of_date"]).normalize()
        expiry = int(row["option_maturity"])
        tenor  = int(row["swap_tenor"])
        mkt_bp = float(row["market_vol"]) * 1e4

        result = price_cell(date, expiry, tenor, seed)
        if counter % 200 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / max(counter, 1) * (len(combos) - counter)
            print(f"  {counter}/{len(combos)}  {date.date()}  {expiry}Yx{tenor}Y  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        if result is None:
            continue

        split = "train" if date in train_dates else "test"
        rows.append({
            "date":            date,
            "expiry":          expiry,
            "tenor":           tenor,
            "split":           split,
            "mkt_bp":          mkt_bp,
            "sigma_str_bp":    result["sigma_str_bp"],
            "vol_error_bp":    result["sigma_str_bp"] - mkt_bp,
            "forward_bias_bp": result["forward_bias_bp"],
            "path_frac":       result["path_frac"],
        })

    df_seed = pd.DataFrame(rows)
    all_results[seed] = df_seed
    df_seed.to_csv(os.path.join(OUT_DIR, f"per_cell_{N_PATHS_EVAL}paths_seed{seed}.csv"), index=False)

    # Quick summary
    print(f"\n  Seed {seed} summary:")
    print(f"  Overall MAE : {df_seed['vol_error_bp'].abs().mean():.1f} bp")
    print(f"  Train MAE   : {df_seed[df_seed['split']=='train']['vol_error_bp'].abs().mean():.1f} bp")
    print(f"  Test MAE    : {df_seed[df_seed['split']=='test']['vol_error_bp'].abs().mean():.1f} bp")
    print(f"  Fwd bias    : {df_seed['forward_bias_bp'].mean():+.1f} bp")
    print(f"  Path finite : {df_seed['path_frac'].mean()*100:.1f}%")
    for e in EXPIRY_VALS:
        sub = df_seed[df_seed["expiry"] == e]
        print(f"  {e}Y expiry MAE: {sub['vol_error_bp'].abs().mean():.1f} bp")
    elapsed_total = time.time() - t0
    print(f"  Runtime: {elapsed_total/60:.1f} min")

# ── cross-seed summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"CROSS-SEED SUMMARY  ({N_PATHS_EVAL} paths per seed)")
print(f"{'='*60}")

def cross_seed_stats(metric_fn):
    vals = [metric_fn(all_results[s]) for s in SEEDS if s in all_results]
    return np.mean(vals), np.std(vals), vals

# Overall MAE
mean_mae, std_mae, seed_maes = cross_seed_stats(
    lambda df: df["vol_error_bp"].abs().mean())
print(f"\nOverall MAE:  {mean_mae:.1f} ± {std_mae:.1f} bp  "
      f"(seeds: {[f'{v:.1f}' for v in seed_maes]})")

# Compare with 512-path result
print(f"Constant MPR with 512 paths: 41.4 bp")
print(f"Difference (high-path - 512-path): {mean_mae - 41.4:+.1f} bp")
if mean_mae < 35:
    print("  -> CONCLUSION: Part of the 41 bp was MC noise. Evaluate finals with more paths.")
elif mean_mae < 45:
    print("  -> CONCLUSION: Error is structural. 512 paths was sufficient.")
else:
    print("  -> CONCLUSION: High-path eval is worse — check for systematic issues.")

# Test MAE
mean_test, std_test, _ = cross_seed_stats(
    lambda df: df[df["split"]=="test"]["vol_error_bp"].abs().mean())
print(f"\nTest MAE:     {mean_test:.1f} ± {std_test:.1f} bp")

# By expiry
print(f"\nMAE by expiry:")
for e in EXPIRY_VALS:
    mean_e, std_e, vals_e = cross_seed_stats(
        lambda df, e=e: df[df["expiry"]==e]["vol_error_bp"].abs().mean())
    print(f"  {e}Y expiry:  {mean_e:.1f} ± {std_e:.1f} bp  "
          f"(seeds: {[f'{v:.1f}' for v in vals_e]})")

# By cell
print(f"\nMAE by cell (mean ± std across {len(SEEDS)} seeds):")
print(f"{'Cell':>9}  {'MAE mean':>9}  {'MAE std':>8}  {'Test MAE':>9}")
print("-" * 45)
for e in EXPIRY_VALS:
    for t in TENOR_VALS:
        mean_c, std_c, _ = cross_seed_stats(
            lambda df, e=e, t=t: df[
                (df["expiry"]==e) & (df["tenor"]==t)
            ]["vol_error_bp"].abs().mean())
        mean_test_c, _, _ = cross_seed_stats(
            lambda df, e=e, t=t: df[
                (df["expiry"]==e) & (df["tenor"]==t) & (df["split"]=="test")
            ]["vol_error_bp"].abs().mean())
        print(f"  {e}Yx{t}Y  {mean_c:>9.1f}  {std_c:>8.2f}  {mean_test_c:>9.1f}")

# ── figure: MAE comparison bar chart ──────────────────────────────────────────
# Build arrays for plotting
cells      = [f"{e}Yx{t}Y" for e in EXPIRY_VALS for t in TENOR_VALS]
mae_means  = []
mae_stds   = []
mae_512    = [97.1, 77.5, 65.9, 34.6, 44.7, 29.2, 14.1, 17.4, 14.2]  # from eval_constant_mpr

for e in EXPIRY_VALS:
    for t in TENOR_VALS:
        mean_c, std_c, _ = cross_seed_stats(
            lambda df, e=e, t=t: df[
                (df["expiry"]==e) & (df["tenor"]==t)
            ]["vol_error_bp"].abs().mean())
        mae_means.append(mean_c)
        mae_stds.append(std_c)

x     = np.arange(len(cells))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, mae_512,    width, label="512 paths (training eval)",
               color="#9ca3af", alpha=0.9)
bars2 = ax.bar(x + width/2, mae_means,  width, label=f"{N_PATHS_EVAL} paths (mean of {len(SEEDS)} seeds)",
               color="#2563eb", alpha=0.9, yerr=mae_stds, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("ATM Vol MAE (bp)", fontsize=10)
ax.set_title(f"Constant MPR: Per-Cell MAE — 512 paths vs {N_PATHS_EVAL} paths\n"
             f"Error bars = ±1 std across {len(SEEDS)} seeds",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.axhline(0, color="black", lw=0.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"fig_mc_noise_check_{N_PATHS_EVAL}paths.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: fig_mc_noise_check.png")

# ── final verdict ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"VERDICT")
print(f"{'='*60}")
print(f"  512-path MAE:        41.4 bp")
print(f"  {N_PATHS_EVAL}-path MAE (mean):  {mean_mae:.1f} bp  (std={std_mae:.2f})")
print(f"  Seed-to-seed std:    {std_mae:.2f} bp  <- MC noise estimate")
print()
noise_frac = std_mae / mean_mae * 100
print(f"  MC noise is ~{noise_frac:.0f}% of mean MAE")
if std_mae < 3:
    print("  -> 512 paths is sufficient for structural assessment.")
    print("  -> Remaining error is structural model limitation.")
else:
    print(f"  -> Seed-to-seed variation is {std_mae:.1f} bp — worth using more paths for final tables.")

print(f"\nAll outputs written to: {OUT_DIR}")
