# ==================== Forward-Adjusted Diffusion-Scale Calibration ====================
"""
Corrects for the forward bias in K before calibrating the diffusion scale.

The problem
-----------
The standard calibration uses:
    sigma_mod = V_pay * sqrt(2pi) / (A0 * sqrt(T_e))   [ATM formula]

This formula is only valid when the option is at-the-money for the MODEL,
i.e. when K = F_model = E^{Q_A}[S_T]. But the model's K was identified
from reconstruction loss, not the martingale condition, so F_model != F_0.

When F_model < F_0 (our case: bias -18 to -694 bp), the payer is OTM
relative to the model's distribution. The ATM formula then understates the
model's true vol, and s* is calibrated to a distorted number.

The fix (inspired by Mercurio et al. 2025)
------------------------------------------
From the same MC run, compute F_model via put-call parity:
    F_model = F_0 + (V_pay - V_rec) / A0

Then re-price the payer at K = F_model (the model's own forward):
    V_pay_adj = mean( D_T * A_T * max(S_T - F_model, 0) )

The ATM formula is now valid:
    sigma_adj = V_pay_adj * sqrt(2pi) / (A0 * sqrt(T_e))  =  sigma_true

sigma_true is the model's true volatility, fully decoupled from the
forward bias. In Bachelier, vol is strike-independent, so sigma_true at
K=F_model is directly comparable to sigma_mkt at K=F_0.

Protocol
--------
1. Price ALL swaptions at s=1 — payer AND receiver — to get F_model and
   sigma_adj per (date, expiry, tenor). Cache as fwd_adj_vols_s1.csv.
2. Expiry-level OLS on sigma_adj (3 parameters, same as before).
3. Re-price OOS dates at adjusted scale. Report MAE on sigma_adj vs sigma_mkt.
4. Compare to baseline expiry-level result (72 bp OOS).

Outputs (to dim4_stable_hscale/forward_adjusted/)
-------------------------------------------------
    fwd_adj_vols_s1.csv         sigma_adj per (date, expiry, tenor) at s=1
    fwd_adj_scales.json         calibrated expiry-level scales
    fwd_adj_results.csv         per-(date, cell) errors
    tab_fwd_adj_comparison.tex  LaTeX table: forward-adjusted vs baseline
    fig_fwd_adj_mae.png         MAE heatmap comparison
"""

import math
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

from Code import config
from Code.load_swapdata import my_data
from Code.model.full_model_stable import FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import swap_rate_torch
from Code.Simulation.simulate_model import simulate_latent_paths, compute_discount_paths

# ================================================================
# Settings
# ================================================================

LATENT_DIM  = 4
N_PATHS     = 2048
DT          = 1 / 12
CCY_FILTER  = "EUR"
USE         = "bbg"
SEED        = 42
TRAIN_FRAC  = 0.70

RATE_CLIP   = 0.50
ANNUITY_MAX = 50.0

CKPT_STABLE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
HSCALE_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
OUT_DIR = os.path.join(HSCALE_DIR, "forward_adjusted")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_CSV = os.path.join(OUT_DIR, "fwd_adj_vols_s1.csv")

torch.manual_seed(SEED)
np.random.seed(SEED)
device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

# ================================================================
# Load model and data
# ================================================================

print(f"Repo root: {PROJECT_ROOT}")
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(CKPT_STABLE, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
model.eval()
print(f"Loaded: {os.path.basename(CKPT_STABLE)}")

meta, X_tensor, *_ = my_data(use=USE)
X_tensor  = X_tensor.float()
meta_eur  = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_eur     = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0
dates_swap  = set(pd.to_datetime(meta_eur["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_eur.iterrows()
}
combos = df_vol[["as_of_date","option_maturity","swap_tenor","market_vol"]].drop_duplicates()
print(f"EUR: {len(meta_eur)} dates, {len(combos)} vol targets")

all_dates   = sorted(df_vol["as_of_date"].unique())
n_train     = int(len(all_dates) * TRAIN_FRAC)
train_dates = set(all_dates[:n_train])
test_dates  = set(all_dates[n_train:])
print(f"Train: {len(train_dates)} dates  OOS: {len(test_dates)} dates")

# ================================================================
# MC pricing helper — forward-adjusted vol
# ================================================================

@torch.no_grad()
def price_forward_adjusted(date, expiry, tenor, s):
    """
    Returns dict with sigma_adj, sigma_atm, forward_bias_bp, F0, F_model, n_surv
    or None on failure.

    sigma_adj : model vol extracted at K = F_model (correct ATM for the model)
    sigma_atm : model vol extracted at K = F_0    (current/biased approach)
    """
    if date not in date_to_idx:
        return None
    idx = date_to_idx[date]
    xb  = X_eur[idx:idx+1].to(device)
    z0  = model.encoder(xb)

    n_steps = max(12, int(round(expiry / DT)))
    dt_eff  = expiry / n_steps
    half    = N_PATHS // 2

    z1,r1,_,_ = simulate_latent_paths(model, z0, n_paths=half,
                                       n_steps=n_steps, dt=dt_eff,
                                       device=device, diffusion_scale=s)
    z2,r2,_,_ = simulate_latent_paths(model, z0, n_paths=half,
                                       n_steps=n_steps, dt=dt_eff,
                                       device=device, diffusion_scale=s)
    z_T   = torch.cat([z1[:,-1,:], z2[:,-1,:]], dim=0)
    r_all = torch.cat([r1, r2], dim=0)
    D_T   = compute_discount_paths(r_all, dt_eff)[:,-1]

    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 32:
        return None
    z_k, D_k = z_T[ok], D_T[ok]

    _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True)
    P_T = aux_T["P_full"]
    dok = torch.isfinite(P_T).all(1)
    if dok.sum() < 32:
        return None
    F_T, A_T = swap_rate_torch(P_T[dok], tenor=tenor)
    D_k = D_k[dok]

    fa = torch.isfinite(F_T) & torch.isfinite(A_T)
    F_T, A_T, D_k = F_T[fa], A_T[fa], D_k[fa]

    sane = (F_T > -RATE_CLIP) & (F_T < RATE_CLIP) & (A_T > 1e-6) & (A_T < ANNUITY_MAX)
    if sane.sum() < 32:
        return None
    F_T, A_T, D_k = F_T[sane], A_T[sane], D_k[sane]

    _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
    F0, A0 = swap_rate_torch(aux0["P_full"], tenor=tenor)
    F0, A0 = float(F0[0]), float(A0[0])
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 0):
        return None

    # --- Current approach: payer at K=F0, ATM formula ---
    V_pay = float((D_k * A_T * torch.relu(F_T - F0)).mean())
    V_rec = float((D_k * A_T * torch.relu(F0 - F_T)).mean())
    if not (math.isfinite(V_pay) and math.isfinite(V_rec)):
        return None

    sigma_atm = V_pay * sqrt_2pi / (A0 * math.sqrt(expiry)) * 10_000.0

    # --- Forward-adjusted approach: price at K=F_model ---
    # Step 1: estimate model forward from put-call parity
    F_model = F0 + (V_pay - V_rec) / A0

    # Step 2: re-price payer at K=F_model using the SAME paths
    V_pay_adj = float((D_k * A_T * torch.relu(F_T - F_model)).mean())
    if not math.isfinite(V_pay_adj) or V_pay_adj <= 0:
        return None

    # Step 3: extract vol with ATM formula — now valid since K=F_model
    sigma_adj = V_pay_adj * sqrt_2pi / (A0 * math.sqrt(expiry)) * 10_000.0

    forward_bias_bp = (F_model - F0) * 10_000.0
    n_surv = int(sane.sum()) / N_PATHS

    return {
        "sigma_adj":        sigma_adj,
        "sigma_atm":        sigma_atm,
        "forward_bias_bp":  forward_bias_bp,
        "F0":               F0,
        "F_model":          F_model,
        "n_surv":           n_surv,
    }

# ================================================================
# Step 1 — Price all swaptions at s=1
# ================================================================

if os.path.exists(CACHE_CSV):
    print(f"\nLoading cached s=1 results from {CACHE_CSV}")
    df_s1 = pd.read_csv(CACHE_CSV)
    df_s1["date"] = pd.to_datetime(df_s1["date"])
else:
    print("\n=== Step 1: Pricing payer + receiver at s=1 ===")
    rows    = []
    counter = 0

    for _, row in combos.iterrows():
        date    = pd.Timestamp(row["as_of_date"]).normalize()
        expiry  = int(row["option_maturity"])
        tenor   = int(row["swap_tenor"])
        mkt_vol = float(row["market_vol"]) * 10_000.0

        result = price_forward_adjusted(date, expiry, tenor, s=1.0)
        counter += 1
        if counter % 100 == 0:
            print(f"  Priced {counter} ({date.date()} {expiry}Yx{tenor}Y)")

        if result is None:
            continue

        split = "train" if date in train_dates else "test"
        rows.append({
            "date":             date,
            "expiry":           expiry,
            "tenor":            tenor,
            "split":            split,
            "sigma_mkt":        mkt_vol,
            "sigma_adj":        result["sigma_adj"],
            "sigma_atm":        result["sigma_atm"],
            "forward_bias_bp":  result["forward_bias_bp"],
            "F0":               result["F0"],
            "F_model":          result["F_model"],
            "n_surv":           result["n_surv"],
        })

    df_s1 = pd.DataFrame(rows)
    df_s1.to_csv(CACHE_CSV, index=False)
    print(f"Saved {CACHE_CSV} ({len(df_s1)} rows)")

# ================================================================
# Step 2 — Compare sigma_adj vs sigma_atm at s=1
# ================================================================

print("\n=== Step 2: Forward bias correction effect ===")
print(f"\n{'Cell':>10}  {'bias (bp)':>10}  {'sigma_atm(s=1)':>15}  {'sigma_adj(s=1)':>15}  {'sigma_mkt':>10}")
print("-"*68)

expiry_vals = sorted(df_s1["expiry"].unique().astype(int))
tenor_vals  = sorted(df_s1["tenor"].unique().astype(int))

for e in expiry_vals:
    for t in tenor_vals:
        sub = df_s1[(df_s1["expiry"]==e) & (df_s1["tenor"]==t)]
        if len(sub) == 0:
            continue
        print(f"  {e}Yx{t}Y  "
              f"  {sub['forward_bias_bp'].mean():>+10.1f}  "
              f"  {sub['sigma_atm'].mean():>15.1f}  "
              f"  {sub['sigma_adj'].mean():>15.1f}  "
              f"  {sub['sigma_mkt'].mean():>10.1f}")

# ================================================================
# Step 3 — Per-expiry OLS calibration on sigma_adj
# ================================================================

print("\n=== Step 3: Expiry-level OLS on forward-adjusted vols ===")

df_valid = df_s1[df_s1["sigma_adj"].notna() & (df_s1["sigma_mkt"] > 0) & (df_s1["sigma_adj"] > 0)].copy()
df_train = df_valid[df_valid["split"] == "train"]

expiry_scales_adj = {}
print(f"\n{'Expiry':>8}  {'s*_adj':>8}  {'s*_atm':>8}  {'R²_adj':>8}  {'R²_atm':>8}")
print("-"*50)

# Also compute baseline ATM scales for comparison
expiry_scales_atm = {}

for e in expiry_vals:
    sub = df_train[df_train["expiry"] == e]

    adj  = sub["sigma_adj"].values.astype(float)
    atm  = sub["sigma_atm"].values.astype(float)
    mkt  = sub["sigma_mkt"].values.astype(float)

    s_adj = float(np.dot(adj, mkt) / np.dot(adj, adj))
    s_atm = float(np.dot(atm, mkt) / np.dot(atm, atm))

    r2_adj = 1 - np.sum((s_adj*adj - mkt)**2) / np.sum((mkt - mkt.mean())**2)
    r2_atm = 1 - np.sum((s_atm*atm - mkt)**2) / np.sum((mkt - mkt.mean())**2)

    expiry_scales_adj[e] = s_adj
    expiry_scales_atm[e] = s_atm
    print(f"  {e}Y      {s_adj:>8.4f}  {s_atm:>8.4f}  {r2_adj:>8.3f}  {r2_atm:>8.3f}")

with open(os.path.join(OUT_DIR, "fwd_adj_scales.json"), "w") as f:
    json.dump({str(k): float(v) for k, v in expiry_scales_adj.items()}, f, indent=2)

# ================================================================
# Step 4 — Evaluate OOS: re-price at calibrated scales
# ================================================================

print("\n=== Step 4: OOS evaluation at calibrated scales ===")

# Linearity: sigma_mod(s) ≈ s * sigma_mod(1)
# For the adjusted approach: model vol at scale s ≈ s * sigma_adj(s=1)
# MAE metric: |s* * sigma_adj(1) - sigma_mkt|

rows_eval = []
for _, row in df_valid.iterrows():
    e   = int(row["expiry"])
    t   = int(row["tenor"])
    spl = row["split"]

    # Forward-adjusted
    s_adj = expiry_scales_adj.get(e, float("nan"))
    pred_adj = s_adj * row["sigma_adj"]
    err_adj  = abs(pred_adj - row["sigma_mkt"])

    # Baseline ATM
    s_atm = expiry_scales_atm.get(e, float("nan"))
    pred_atm = s_atm * row["sigma_atm"]
    err_atm  = abs(pred_atm - row["sigma_mkt"])

    rows_eval.append({
        "date":     row["date"],
        "expiry":   e,
        "tenor":    t,
        "split":    spl,
        "sigma_mkt": row["sigma_mkt"],
        "pred_adj": pred_adj,
        "pred_atm": pred_atm,
        "err_adj":  err_adj,
        "err_atm":  err_atm,
    })

df_eval = pd.DataFrame(rows_eval)
df_eval.to_csv(os.path.join(OUT_DIR, "fwd_adj_results.csv"), index=False)

# ================================================================
# Step 5 — Summary table
# ================================================================

print("\n=== Step 5: Results ===")

for split_name, split_key in [("Train", "train"), ("OOS", "test"), ("All", None)]:
    if split_key is not None:
        sub = df_eval[df_eval["split"] == split_key]
    else:
        sub = df_eval
    mae_adj = sub["err_adj"].mean()
    mae_atm = sub["err_atm"].mean()
    print(f"\n--- {split_name} ---")
    print(f"  Forward-adjusted MAE:  {mae_adj:.1f} bp")
    print(f"  Baseline ATM MAE:      {mae_atm:.1f} bp")
    print(f"  Improvement:           {mae_atm - mae_adj:+.1f} bp")

# Per-cell OOS
print("\n--- Per-cell OOS ---")
print(f"\n{'Cell':>10}  {'Adj MAE':>10}  {'ATM MAE':>10}  {'Improvement':>12}")
print("-"*48)

cell_adj = {}
cell_atm = {}
oos = df_eval[df_eval["split"] == "test"]
for e in expiry_vals:
    for t in tenor_vals:
        sub = oos[(oos["expiry"]==e) & (oos["tenor"]==t)]
        if len(sub) == 0:
            continue
        adj_mae = sub["err_adj"].mean()
        atm_mae = sub["err_atm"].mean()
        cell_adj[(e,t)] = adj_mae
        cell_atm[(e,t)] = atm_mae
        label = f"{e}Yx{t}Y"
        print(f"  {label:>8}  {adj_mae:>10.1f}  {atm_mae:>10.1f}  {atm_mae-adj_mae:>+12.1f}")

# ================================================================
# Step 6 — LaTeX table
# ================================================================

lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(
    r"\caption{Per-cell OOS vol MAE (bp): baseline expiry-level OLS (payer at $K=F_0$, "
    r"ATM formula) versus forward-adjusted calibration (payer re-priced at $K=F_{\mathrm{model}}$, "
    r"where $F_{\mathrm{model}} = F_0 + (V_{\mathrm{pay}}-V_{\mathrm{rec}})/\mathcal{A}_0$). "
    r"The forward adjustment decouples the model's true volatility from the forward bias "
    r"in $\mathcal{K}$, producing a cleaner vol signal for calibration.}"
)
lines.append(r"\label{tab:fwd_adj_comparison}")
lines.append(r"\begin{tabular}{@{}ccrrrr@{}}")
lines.append(r"\toprule")
lines.append(r"\textbf{Exp} & \textbf{Ten} & \textbf{Baseline (bp)} & "
             r"\textbf{Fwd-adj (bp)} & \textbf{Improvement (bp)} \\")
lines.append(r"\midrule")

for i, e in enumerate(expiry_vals):
    for j, t in enumerate(tenor_vals):
        key = (e, t)
        if key not in cell_adj:
            continue
        adj = cell_adj[key]
        atm = cell_atm[key]
        imp = atm - adj
        sign = "+" if imp >= 0 else ""
        lines.append(f"{e} & {t} & {atm:.0f} & {adj:.0f} & ${sign}{imp:.0f}$ \\\\")
    if i < len(expiry_vals) - 1:
        lines.append(r"\addlinespace[2pt]")

oos_mae_adj = oos["err_adj"].mean()
oos_mae_atm = oos["err_atm"].mean()
overall_imp = oos_mae_atm - oos_mae_adj
sign = "+" if overall_imp >= 0 else ""
lines.append(r"\midrule")
lines.append(
    rf"\textbf{{Overall}} & & {oos_mae_atm:.0f} & {oos_mae_adj:.0f} & ${sign}{overall_imp:.0f}$ \\"
)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex_path = os.path.join(OUT_DIR, "tab_fwd_adj_comparison.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved {tex_path}")

# ================================================================
# Step 7 — Figure: side-by-side MAE heatmaps
# ================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, cell_dict, title in zip(
    axes,
    [cell_atm, cell_adj],
    ["Baseline (payer at $K=F_0$)", "Forward-adjusted ($K=F_{\\mathrm{model}}$)"]
):
    grid = np.full((3, 3), np.nan)
    for i, e in enumerate(expiry_vals):
        for j, t in enumerate(tenor_vals):
            if (e, t) in cell_dict:
                grid[i, j] = cell_dict[(e, t)]

    vmax = max(
        max(v for v in cell_atm.values() if not np.isnan(v)),
        max(v for v in cell_adj.values() if not np.isnan(v))
    )
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="MAE (bp)")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"{t}Y" for t in tenor_vals])
    ax.set_yticklabels([f"{e}Y" for e in expiry_vals])
    ax.set_xlabel("Swap tenor")
    ax.set_ylabel("Option expiry")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9)

plt.suptitle("OOS vol MAE (bp): baseline vs forward-adjusted", fontsize=11)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "fig_fwd_adj_mae.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved {fig_path}")

print(f"\nDone. All outputs in: {OUT_DIR}")
