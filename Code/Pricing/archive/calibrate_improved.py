# ==================== Improved Diffusion-Scale Calibration ====================
"""
Four improvements over the per-cell OLS calibration.

ROOT CAUSES OF RESIDUAL ERROR
------------------------------
(A) Overfitting: per-cell OLS has 9 free parameters on 105 training dates.
    The 1Yx10Y cell (s*=0.098) severely overfits: 23 bp train -> 144 bp OOS.

(B) Drift floor: the model's physical-measure drift K(z) systematically
    moves z towards equilibrium z*, changing the swap rate even at s->0.
    This creates a 'free' implied vol floor even with no diffusion.
    For 1Yx1Y the floor may exceed market vol, making the cell unpriceble
    within the diffusion-scaling framework.

(C) Temporal instability: the optimal scale s* varies across market regimes
    (low-rate QE era vs 2022-23 hiking cycle). A fixed s* cannot track this.

(D) Linearity breakdown: sigma_mod(s) ≈ s * sigma_mod(1) fails for small s
    due to the drift floor in (B). The quadratic approximation is more accurate:
    sigma_mod(s)^2 ≈ sigma_drift^2 + s^2 * sigma_diff^2.

APPROACHES IMPLEMENTED
-----------------------
1. RIDGE   — per-cell ridge regression with leave-one-year-out CV on lambda.
             s_ridge*(e,t;λ) = [Σ(mod1·mkt) + λ·s_prior] / [Σ(mod1²) + λ]
             Fixes 1Yx10Y overfitting by pulling extreme s* towards s_global.

2. DRIFT   — price at s=0.01 (near-zero diffusion) to measure the drift floor
             sigma_drift(e,t,d). Identifies cells where sigma_drift < sigma_mkt
             (priceable) vs sigma_drift >= sigma_mkt (fundamental limitation).
             For priceable cells: quadrature calibration
             sigma_mod(s)^2 = sigma_drift^2 + s^2*sigma_diff^2 -> s* analytic.

3. ADAPTIVE — latent-state-conditioned scale using the encoder's z_t.
             s_t(e,t) = alpha + beta @ z_t,  fitted on training dates by
             ridge OLS.  Applied OOS using only the current swap curve (no vol).

4. DAILY   — lower bound: per-date recalibration using today's observed vol.
             s_t(e,t) = sigma_mkt_t / sigma_mod1_t (clipped).
             Prices OOS at this date-specific scale. Measures cost of
             using fixed vs daily-refreshed scale.

Outputs (all to dim4_stable_hscale/improved/)
---------------------------------------------
  ridge_scales.json         nine ridge scale factors
  drift_floor.csv           per-(date,cell) sigma at s=0.01
  adaptive_params.json      per-cell regression weights
  improved_results.csv      per-(date,cell,method) vol errors
  improved_summary.json     aggregate MAE/RMSE per method
  tab_comparison.tex        LaTeX comparison table (all methods)
  fig_comparison.png        side-by-side MAE heatmaps
"""

import math
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
S_DRIFT     = 0.01     # near-zero diffusion for drift floor measurement
S_CLIP_LO   = 0.005   # minimum allowed adaptive / daily scale
S_CLIP_HI   = 2.0     # maximum allowed adaptive / daily scale

CKPT_STABLE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
HSCALE_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
PER_CELL_DIR = os.path.join(HSCALE_DIR, "per_cell")
OUT_DIR      = os.path.join(HSCALE_DIR, "improved")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device    = torch.device("cpu")
sqrt_2pi  = math.sqrt(2.0 * math.pi)

# ================================================================
# Load model
# ================================================================

model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(CKPT_STABLE, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
model.eval()
print(f"Loaded: {os.path.basename(CKPT_STABLE)}")

# ================================================================
# Load market data
# ================================================================

meta, X_tensor, *_ = my_data(use=USE)
X_tensor   = X_tensor.float()
meta_eur   = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_eur      = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0

dates_swap  = set(pd.to_datetime(meta_eur["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_eur.iterrows()
}

combos = df_vol[["as_of_date", "option_maturity", "swap_tenor",
                  "market_vol"]].drop_duplicates()
print(f"EUR: {len(meta_eur)} dates, {len(combos)} vol targets")

# Train/test split
all_dates   = sorted(df_vol["as_of_date"].unique())
n_train     = int(len(all_dates) * TRAIN_FRAC)
train_dates = set(all_dates[:n_train])
test_dates  = set(all_dates[n_train:])
print(f"Train: {len(train_dates)} dates  OOS: {len(test_dates)} dates")

# ================================================================
# Load cached baselines
# ================================================================

BASE_CSV    = os.path.join(HSCALE_DIR, "baseline_vols_s1.csv")
PERCELL_CSV = os.path.join(PER_CELL_DIR, "per_cell_results.csv")

df_base = pd.read_csv(BASE_CSV)
df_base["date"] = pd.to_datetime(df_base["date"])
print(f"Loaded s=1 baseline: {len(df_base)} rows")

df_pcell = pd.read_csv(PERCELL_CSV)
df_pcell["date"] = pd.to_datetime(df_pcell["date"])
print(f"Loaded per-cell results: {len(df_pcell)} rows")

# Load per-cell OLS scales for reference
with open(os.path.join(PER_CELL_DIR, "per_cell_scales.json")) as f:
    ols_scales_raw = json.load(f)
ols_scales = {tuple(int(x) for x in k.split("x")): v
              for k, v in ols_scales_raw.items()}

# Load global calibration summary
with open(os.path.join(HSCALE_DIR, "calibration_summary.json")) as f:
    cal_summary = json.load(f)
S_GLOBAL = float(cal_summary["s_ols"])
print(f"Global OLS scale: s_global={S_GLOBAL:.4f}")

# ================================================================
# Helper: MC pricing at a given (date, expiry, tenor, s)
# ================================================================

@torch.no_grad()
def price_at_scale(date, expiry, tenor, s):
    if date not in date_to_idx:
        return None
    idx  = date_to_idx[date]
    xb   = X_eur[idx:idx+1].to(device)
    z0   = model.encoder(xb)
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
    ok    = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 32:
        return None
    z_k, D_k = z_T[ok], D_T[ok]
    _,aux_T = model.decode_from_z(z_k, tau=None, return_aux=True)
    P_T = aux_T["P_full"]
    dok = torch.isfinite(P_T).all(1)
    if dok.sum() < 32:
        return None
    F_T, A_T = swap_rate_torch(P_T[dok], tenor=tenor)
    D_k = D_k[dok]
    fa  = torch.isfinite(F_T) & torch.isfinite(A_T)
    if fa.sum() < 32:
        return None
    F_T, A_T, D_k = F_T[fa], A_T[fa], D_k[fa]
    _,aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
    F0,A0  = swap_rate_torch(aux0["P_full"], tenor=tenor)
    F0,A0  = float(F0[0]), float(A0[0])
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 0):
        return None
    V_MC = (D_k * A_T * torch.relu(F_T - F0)).mean()
    if not torch.isfinite(V_MC):
        return None
    sig_mod = float(V_MC) * sqrt_2pi / (A0 * math.sqrt(expiry)) * 10_000.0
    pfrac   = int(ok.sum()) / N_PATHS
    return sig_mod, pfrac


# ================================================================
# Step 1 — RIDGE calibration (leave-one-year-out CV for lambda)
# ================================================================
print("\n" + "="*60)
print("Step 1: Ridge per-cell calibration (LOO-year CV)")
print("="*60)

df_valid = df_base[
    df_base["sigma_mod1"].notna() & (df_base["sigma_mkt"] > 0)
].copy()
df_valid["split"] = df_valid["date"].apply(
    lambda d: "train" if pd.Timestamp(d) in train_dates else "test")
df_train_only = df_valid[df_valid["split"] == "train"].copy()
df_train_only["year"] = df_train_only["date"].dt.year

cells = sorted(df_valid[["expiry","tenor"]].drop_duplicates()
               .itertuples(index=False), key=lambda x: (x.expiry, x.tenor))

# LOO-year CV for lambda per cell
lambda_grid = np.logspace(-3, 4, 50)   # 0.001 .. 10000

def ridge_scale(mod1, mkt, lam, s_prior):
    num = np.dot(mod1, mkt) + lam * s_prior
    den = np.dot(mod1, mod1) + lam
    return num / den if den > 0 else s_prior

ridge_scales  = {}
ridge_lambdas = {}
print(f"\n{'Cell':>8}  {'s_ols':>7}  {'s_ridge':>8}  {'lambda':>10}")
print("-" * 42)
for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub  = df_train_only[(df_train_only["expiry"]==e) & (df_train_only["tenor"]==t)]
    if sub.empty:
        ridge_scales[(e,t)] = ols_scales.get((e,t), S_GLOBAL)
        continue
    mod1  = sub["sigma_mod1"].values.astype(float)
    mkt   = sub["sigma_mkt"].values.astype(float)
    years = sub["year"].values

    # OLS (lambda=0)
    s_ols = float(np.dot(mod1, mkt) / np.dot(mod1, mod1))

    # LOO-year CV
    unique_years = sorted(set(years))
    cv_errors = []
    for lam in lambda_grid:
        fold_maes = []
        for yr in unique_years:
            mask_val = years == yr
            mask_tr  = ~mask_val
            if mask_tr.sum() == 0 or mask_val.sum() == 0:
                continue
            s_r = ridge_scale(mod1[mask_tr], mkt[mask_tr], lam, S_GLOBAL)
            pred = s_r * mod1[mask_val]
            fold_maes.append(np.mean(np.abs(pred - mkt[mask_val])))
        cv_errors.append(np.mean(fold_maes) if fold_maes else np.inf)

    lam_opt = lambda_grid[int(np.argmin(cv_errors))]
    s_r     = ridge_scale(mod1, mkt, lam_opt, S_GLOBAL)
    ridge_scales[(e,t)]  = s_r
    ridge_lambdas[(e,t)] = lam_opt
    print(f"  {e}Yx{t}Y  {s_ols:>7.4f}  {s_r:>8.4f}  {lam_opt:>10.4f}")

# Save ridge scales
ridge_serialisable = {f"{e}x{t}": float(v) for (e,t),v in ridge_scales.items()}
with open(os.path.join(OUT_DIR, "ridge_scales.json"), "w") as f:
    json.dump(ridge_serialisable, f, indent=2)
print("Saved ridge_scales.json")


# ================================================================
# Step 2 — DRIFT FLOOR: price all dates at s=0.01
# ================================================================
print("\n" + "="*60)
print(f"Step 2: Drift floor measurement (s={S_DRIFT})")
print("="*60)

DRIFT_CSV = os.path.join(OUT_DIR, "drift_floor.csv")

if os.path.isfile(DRIFT_CSV):
    print(f"Loading cached drift floor from {DRIFT_CSV}")
    df_drift = pd.read_csv(DRIFT_CSV)
    df_drift["date"] = pd.to_datetime(df_drift["date"])
else:
    drift_rows = []
    n_done = 0
    for _, row in combos.iterrows():
        date   = pd.Timestamp(row["as_of_date"]).normalize()
        expiry = int(row["option_maturity"])
        tenor  = int(row["swap_tenor"])
        sig_mkt = float(row["market_vol"]) * 10_000.0

        res = price_at_scale(date, expiry, tenor, S_DRIFT)
        if res is not None:
            sig_drift, _ = res
            n_done += 1
        else:
            sig_drift = float("nan")

        split = "train" if date in train_dates else "test"
        drift_rows.append({
            "date":      date.date(),
            "expiry":    expiry,
            "tenor":     tenor,
            "split":     split,
            "sigma_mkt": round(sig_mkt, 1),
            "sigma_drift": round(sig_drift, 1) if math.isfinite(sig_drift) else None,
        })
        if n_done % 100 == 0:
            print(f"  Drift floor: {n_done} done ({date.date()} {expiry}Yx{tenor}Y)")

    df_drift = pd.DataFrame(drift_rows)
    df_drift["date"] = pd.to_datetime(df_drift["date"])
    df_drift.to_csv(DRIFT_CSV, index=False)
    print(f"Saved drift_floor.csv ({n_done} priced)")

# Report drift floor per cell vs market vol
print(f"\n{'Cell':>8}  {'avg drift (bp)':>14}  {'avg mkt (bp)':>13}  {'priceable?':>11}")
print("-"*54)
priceable_cells = set()
for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub_tr = df_drift[(df_drift["expiry"]==e) & (df_drift["tenor"]==t)
                      & (df_drift["split"]=="train") & df_drift["sigma_drift"].notna()]
    avg_drift = sub_tr["sigma_drift"].mean() if not sub_tr.empty else float("nan")
    avg_mkt   = sub_tr["sigma_mkt"].mean()   if not sub_tr.empty else float("nan")
    priceable = avg_drift < avg_mkt
    if priceable:
        priceable_cells.add((e, t))
    flag = "YES" if priceable else "NO (floor > mkt)"
    print(f"  {e}Yx{t}Y  {avg_drift:>14.1f}  {avg_mkt:>13.1f}  {flag:>16}")


# ================================================================
# Step 3 — ADAPTIVE calibration using encoder latent state z_t
# ================================================================
print("\n" + "="*60)
print("Step 3: Adaptive z_t-conditioned calibration")
print("="*60)

# Encode all training dates -> z_t matrix
print("  Encoding training dates...")
date_to_z = {}
with torch.no_grad():
    for date, idx in date_to_idx.items():
        xb = X_eur[idx:idx+1].to(device)
        z  = model.encoder(xb).squeeze(0).cpu().numpy()
        date_to_z[date] = z

# For each cell, fit ridge linear model:
#   sigma_mkt_t  ~  (alpha + beta @ z_t) * sigma_mod1_t
# Equivalently (with X = sigma_mod1 * [1, z]):
#   sigma_mkt_t  ~  X_t @ w   where w = [alpha, beta[0..3]]
# Use sklearn Ridge with alpha chosen by LOO-year CV

adaptive_params = {}
print(f"\n{'Cell':>8}  {'lam_opt':>10}  {'train_R2':>10}  {'alpha':>8}  {'||beta||':>9}")
print("-"*56)

# LOO-year ridge for each cell.
# Model:  sigma_mkt_t  ~  X_t @ w   where X_t = sigma_mod1_t * [1, z_t]
# w = [alpha, beta[0..3]],  s_t = alpha + beta @ z_t
# Penalty: lambda * ||w||^2  (L2 on all weights including alpha)
# Lambda chosen by LOO-year CV.  No sklearn scaling needed — direct closed form.

lambda_adap_grid = np.logspace(3, 11, 60)  # appropriate for features O(500 bp)

for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub = df_train_only[(df_train_only["expiry"]==e) & (df_train_only["tenor"]==t)
                        & df_train_only["sigma_mod1"].notna()
                        & (df_train_only["sigma_mkt"] > 0)]

    rows_z, rows_m, row_years = [], [], []
    for _, row in sub.iterrows():
        d = pd.Timestamp(row["date"])
        if d not in date_to_z:
            continue
        z_t = date_to_z[d]
        m1  = float(row["sigma_mod1"])
        mk  = float(row["sigma_mkt"])
        # Feature: sigma_mod1 * [1, z[0], z[1], z[2], z[3]]
        feat = m1 * np.concatenate([[1.0], z_t])
        rows_z.append(feat)
        rows_m.append(mk)
        row_years.append(d.year)

    if len(rows_z) < 10:
        w_fallback = np.zeros(1 + LATENT_DIM)
        w_fallback[0] = ols_scales.get((e, t), S_GLOBAL)
        adaptive_params[(e, t)] = {"w": w_fallback.tolist()}
        continue

    Xmat  = np.array(rows_z)    # (n, 5)
    yvec  = np.array(rows_m)    # (n,)
    years = np.array(row_years)
    unique_years = sorted(set(years))

    # LOO-year CV for lambda
    cv_errs = []
    for lam in lambda_adap_grid:
        fold_maes = []
        for yr in unique_years:
            val_mask = years == yr
            tr_mask  = ~val_mask
            if tr_mask.sum() < 5 or val_mask.sum() == 0:
                continue
            A  = Xmat[tr_mask].T @ Xmat[tr_mask] + lam * np.eye(Xmat.shape[1])
            b  = Xmat[tr_mask].T @ yvec[tr_mask]
            w  = np.linalg.solve(A, b)
            pred = Xmat[val_mask] @ w
            fold_maes.append(np.mean(np.abs(pred - yvec[val_mask])))
        cv_errs.append(np.mean(fold_maes) if fold_maes else np.inf)

    lam_opt = float(lambda_adap_grid[int(np.argmin(cv_errs))])

    # Fit on full training data with lam_opt
    A      = Xmat.T @ Xmat + lam_opt * np.eye(Xmat.shape[1])
    b      = Xmat.T @ yvec
    w_orig = np.linalg.solve(A, b)

    # In-sample R²
    pred_tr = Xmat @ w_orig
    ss_res  = np.sum((pred_tr - yvec)**2)
    ss_tot  = np.sum((yvec - yvec.mean())**2)
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    adaptive_params[(e, t)] = {"w": w_orig.tolist()}
    beta_norm = float(np.linalg.norm(w_orig[1:]))
    alpha_val = float(w_orig[0])
    print(f"  {e}Yx{t}Y  {lam_opt:>10.1f}  {train_r2:>10.3f}  {alpha_val:>8.4f}  {beta_norm:>9.4f}")

# Save adaptive parameters
adap_serialisable = {
    f"{e}x{t}": {"w": v["w"]} for (e, t), v in adaptive_params.items()
}
with open(os.path.join(OUT_DIR, "adaptive_params.json"), "w") as f:
    json.dump(adap_serialisable, f, indent=2)
print("Saved adaptive_params.json")


# ================================================================
# Step 4 — MC pricing on OOS dates for RIDGE, ADAPTIVE, DAILY
# ================================================================
print("\n" + "="*60)
print("Step 4: MC pricing on OOS dates for all improved methods")
print("="*60)

oos_combos = combos[combos["as_of_date"].isin(test_dates)].copy()
print(f"  {len(oos_combos)} OOS (date, cell) combinations to price")

# Also merge in sigma_mod1 for the daily recalibration scale
df_base_oos = df_base[df_base["date"].isin([pd.Timestamp(d) for d in test_dates])].copy()

result_rows = []
n_priced    = 0

for _, row in oos_combos.iterrows():
    date    = pd.Timestamp(row["as_of_date"]).normalize()
    expiry  = int(row["option_maturity"])
    tenor   = int(row["swap_tenor"])
    sig_mkt = float(row["market_vol"]) * 10_000.0

    # sigma_mod1 for this date/cell (from cache)
    base_row = df_base[
        (df_base["date"] == date) &
        (df_base["expiry"] == expiry) &
        (df_base["tenor"]  == tenor)
    ]
    if base_row.empty or pd.isna(base_row["sigma_mod1"].iloc[0]):
        sig_mod1 = float("nan")
    else:
        sig_mod1 = float(base_row["sigma_mod1"].iloc[0])

    # OLS scale (from per_cell_results.csv) — reuse existing MC price
    pcell_row = df_pcell[
        (df_pcell["date"]   == date) &
        (df_pcell["expiry"] == expiry) &
        (df_pcell["tenor"]  == tenor)
    ]
    if not pcell_row.empty and pd.notna(pcell_row["sigma_cal"].iloc[0]):
        sig_ols = float(pcell_row["sigma_cal"].iloc[0])
    else:
        sig_ols = float("nan")

    # --- RIDGE scale ---
    s_ridge = ridge_scales.get((expiry, tenor), S_GLOBAL)

    # --- ADAPTIVE scale ---
    params = adaptive_params.get((expiry, tenor), None)
    if params and date in date_to_z:
        z_t = date_to_z[date]
        w   = np.array(params["w"])
        s_adap = float(w[0] + w[1:] @ z_t)
        s_adap = float(np.clip(s_adap, S_CLIP_LO, S_CLIP_HI))
    else:
        s_adap = ols_scales.get((expiry, tenor), S_GLOBAL)

    # --- DAILY scale (using today's observed vol surface) ---
    if math.isfinite(sig_mod1) and sig_mod1 > 0 and sig_mkt > 0:
        s_daily = float(np.clip(sig_mkt / sig_mod1, S_CLIP_LO, S_CLIP_HI))
    else:
        s_daily = ols_scales.get((expiry, tenor), S_GLOBAL)

    # --- Run MC pricing (3 passes: ridge, adaptive, daily) ---
    def safe_price(s):
        res = price_at_scale(date, expiry, tenor, s)
        if res is None or not math.isfinite(res[0]):
            return float("nan")
        return round(res[0], 1)

    # Only reprice ridge if scale differs from OLS by > 0.002
    s_ols_fixed = ols_scales.get((expiry, tenor), S_GLOBAL)
    if abs(s_ridge - s_ols_fixed) > 0.002:
        sig_ridge = safe_price(s_ridge)
    else:
        # use existing OLS MC price (same scale)
        sig_ridge = sig_ols

    sig_adap  = safe_price(s_adap)
    sig_daily = safe_price(s_daily)

    n_priced += 3
    if n_priced % 100 == 0:
        print(f"  Priced {n_priced} (current: {date.date()} {expiry}Yx{tenor}Y)")

    result_rows.append({
        "date":       date.date(),
        "expiry":     expiry,
        "tenor":      tenor,
        "sigma_mkt":  round(sig_mkt, 1),
        "sigma_mod1": round(sig_mod1, 1) if math.isfinite(sig_mod1) else None,
        # scales used
        "s_ols":   round(s_ols_fixed, 5),
        "s_ridge": round(s_ridge, 5),
        "s_adap":  round(s_adap, 5),
        "s_daily": round(s_daily, 5),
        # model vols
        "sig_ols":   round(sig_ols, 1) if math.isfinite(sig_ols) else None,
        "sig_ridge": round(sig_ridge, 1) if math.isfinite(sig_ridge) else None,
        "sig_adap":  round(sig_adap, 1) if math.isfinite(sig_adap) else None,
        "sig_daily": round(sig_daily, 1) if math.isfinite(sig_daily) else None,
    })

df_res = pd.DataFrame(result_rows)
df_res["date"] = pd.to_datetime(df_res["date"])

# Compute abs errors for each method
for method in ["ols", "ridge", "adap", "daily"]:
    df_res[f"ae_{method}"] = (df_res[f"sig_{method}"] - df_res["sigma_mkt"]).abs()

df_res.to_csv(os.path.join(OUT_DIR, "improved_results.csv"), index=False)
print(f"\nSaved improved_results.csv ({len(df_res)} rows)")


# ================================================================
# Step 5 — Summary statistics
# ================================================================
print("\n" + "="*60)
print("Step 5: OOS MAE/RMSE summary")
print("="*60)

# Overall OOS MAE per method
summary = {}
methods = [("ols", "OLS (per-cell)"),
           ("ridge", "Ridge"),
           ("adap", "Adaptive z_t"),
           ("daily", "Daily recal.")]

print(f"\n{'Method':<20}  {'OOS MAE':>9}  {'OOS RMSE':>10}")
print("-"*44)
for key, label in methods:
    col = f"ae_{key}"
    valid = df_res[df_res[f"sig_{key}"].notna() & (df_res["sigma_mkt"] > 0)]
    mae  = float(valid[col].mean()) if not valid.empty else float("nan")
    rmse = float((valid[col]**2).mean()**0.5) if not valid.empty else float("nan")
    summary[key] = {"label": label, "oos_mae": round(mae,1), "oos_rmse": round(rmse,1)}
    print(f"  {label:<18}  {mae:>9.1f}  {rmse:>10.1f}")

# Per-cell OOS breakdown
print(f"\n{'Cell':>8}  {'OLS':>7} {'Ridge':>7} {'Adaptive':>9} {'Daily':>7}")
print("-"*46)
cell_summary = {}
for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub = df_res[(df_res["expiry"]==e) & (df_res["tenor"]==t)]
    vals = {}
    for key, _ in methods:
        col = f"ae_{key}"
        valid = sub[sub[f"sig_{key}"].notna() & (sub["sigma_mkt"] > 0)]
        vals[key] = float(valid[col].mean()) if not valid.empty else float("nan")
    cell_summary[f"{e}x{t}"] = vals
    row_str = "  " + "Yx".join([str(e),str(t)]) + "Y"
    for key, _ in methods:
        v = vals[key]
        row_str += f"  {v:>7.1f}" if math.isfinite(v) else "      --"
    print(row_str)

# Save
summary["per_cell"] = cell_summary
with open(os.path.join(OUT_DIR, "improved_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved improved_summary.json")


# ================================================================
# Step 6 — LaTeX comparison table
# ================================================================
print("\n" + "="*60)
print("Step 6: LaTeX table")
print("="*60)

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{OOS vol MAE (bp) per calibration method for each "
    r"(expiry, tenor) cell. "
    r"\emph{OLS}: fixed per-cell OLS scale. "
    r"\emph{Ridge}: ridge-regularised OLS ($\lambda$ from leave-one-year-out CV). "
    r"\emph{Adaptive}: scale $s_t = \alpha + \beta \cdot z_t$ fitted on training dates, "
    r"applied using only the encoded swap curve at each OOS date. "
    r"\emph{Daily}: scale $s_t = \sigma_{\mathrm{mkt},t}/\sigma_{\mathrm{mod},t}(1)$ "
    r"refitted each OOS date using the current vol surface (lower bound).}",
    r"\label{tab:improved_comparison}",
    r"\begin{tabular}{@{}ccrrrrr@{}}",
    r"\toprule",
    r"\textbf{Exp} & \textbf{Ten} & "
    r"\textbf{OLS} & \textbf{Ridge} & "
    r"\textbf{Adaptive} & \textbf{Daily} \\",
    r"\midrule",
]
prev_e = None
for cell in cells:
    e, t = cell.expiry, cell.tenor
    if prev_e is not None and e != prev_e:
        lines.append(r"\addlinespace[2pt]")
    prev_e = e
    vals = cell_summary.get(f"{e}x{t}", {})
    def fmt(key):
        v = vals.get(key, float("nan"))
        return "--" if not math.isfinite(v) else str(int(round(v)))
    lines.append(f"{e} & {t} & {fmt('ols')} & {fmt('ridge')} "
                 f"& {fmt('adap')} & {fmt('daily')} \\\\")
lines.append(r"\midrule")
# Overall row
def ofmt(key):
    v = summary.get(key, {}).get("oos_mae", float("nan"))
    return "--" if not math.isfinite(v) else str(int(round(v)))
lines.append(r"\textbf{Overall} & & " +
             f"{ofmt('ols')} & {ofmt('ridge')} & {ofmt('adap')} & {ofmt('daily')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
tex_path = os.path.join(OUT_DIR, "tab_comparison.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved {tex_path}")


# ================================================================
# Step 7 — Figures
# ================================================================
print("\n" + "="*60)
print("Step 7: Figures")
print("="*60)

expiry_vals = sorted(df_res["expiry"].dropna().unique().astype(int))
tenor_vals  = sorted(df_res["tenor"].dropna().unique().astype(int))

def make_pivot(df, col):
    return (
        df[df[col].notna() & (df["sigma_mkt"] > 0)]
        .groupby(["expiry","tenor"])[col]
        .mean()
        .unstack("tenor")
        .reindex(index=expiry_vals, columns=tenor_vals)
    )

piv_ols   = make_pivot(df_res, "ae_ols")
piv_ridge = make_pivot(df_res, "ae_ridge")
piv_adap  = make_pivot(df_res, "ae_adap")
piv_daily = make_pivot(df_res, "ae_daily")

vmax = float(max(
    piv_ols.max().max(),
    piv_ridge.max().max(),
    piv_adap.max().max(),
    50.0
))

fig = plt.figure(figsize=(16, 4), dpi=150)
gs  = gridspec.GridSpec(1, 4, figure=fig, hspace=0.3)
titles = ["OLS (per-cell)", "Ridge", "Adaptive z_t", "Daily recal. (lower bound)"]
pivots = [piv_ols, piv_ridge, piv_adap, piv_daily]

for ax_idx, (piv, title) in enumerate(zip(pivots, titles)):
    ax = fig.add_subplot(gs[0, ax_idx])
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(len(tenor_vals)))
    ax.set_xticklabels([f"{c}Y" for c in tenor_vals], fontsize=7)
    ax.set_yticks(range(len(expiry_vals)))
    ax.set_yticklabels([f"{r}Y" for r in expiry_vals], fontsize=7)
    ax.set_xlabel("Tenor", fontsize=8)
    if ax_idx == 0:
        ax.set_ylabel("Expiry", fontsize=8)
    ax.set_title(title, fontsize=8, pad=3)
    for i in range(len(expiry_vals)):
        for j in range(len(tenor_vals)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                        fontsize=7, color="black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="MAE (bp)" if ax_idx==3 else "", fraction=0.04, pad=0.04)

fig.suptitle("OOS vol MAE (bp) by calibration method", fontsize=10, y=1.01)
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "fig_comparison.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved fig_comparison.png")

# Scatter: adaptive pred vs market (OOS only)
valid_adap = df_res[df_res["sig_adap"].notna() & (df_res["sigma_mkt"] > 0)]
if not valid_adap.empty:
    fig2, ax2 = plt.subplots(figsize=(5,5), dpi=150)
    sc = ax2.scatter(valid_adap["sigma_mkt"], valid_adap["sig_adap"],
                     c=valid_adap["expiry"], cmap="viridis", alpha=0.5, s=15)
    lim = max(valid_adap["sigma_mkt"].max(), valid_adap["sig_adap"].max()) * 1.05
    ax2.plot([0,lim],[0,lim],"k--",lw=0.8)
    plt.colorbar(sc, ax=ax2, label="Expiry (Y)")
    ax2.set_xlabel("Market vol (bp)")
    ax2.set_ylabel("Model vol — adaptive (bp)")
    ax2.set_title("Adaptive calibration: model vs market (OOS)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "fig_scatter_adaptive.png"), dpi=200)
    plt.close(fig2)
    print("Saved fig_scatter_adaptive.png")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
