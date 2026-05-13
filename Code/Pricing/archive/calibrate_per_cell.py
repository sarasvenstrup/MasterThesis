# ==================== Per-Cell Diffusion-Scale Calibration ====================
"""
Extends the single global diffusion-scale calibration (calibrate_H_scale.py)
to fit one scale factor per (expiry, tenor) cell.

Motivation
----------
A single global scale s corrects the average level of model vols but cannot
fix the shape mismatch: the ratio sigma_mod(1) / sigma_mkt varies from ~11x
at 1Yx1Y to ~4x at 10Yx10Y.  Nine per-cell factors s*(e,t) remove this
shape constraint, giving the best possible correction within the
diffusion-scaling framework.

Protocol
--------
1. Load the EUR ATM vol surface and price all swaptions at s=1.
   (Reuses baseline_vols_s1.csv if already present; reproduced otherwise.)
2. Time-series split: first TRAIN_FRAC of dates (chronological) -> train,
   remainder -> held-out OOS test.
3. Per-cell OLS on TRAINING dates:
       s*(e,t) = sum_{d in train}[ sigma_mod1(e,t,d) * sigma_mkt(e,t,d) ]
              /  sum_{d in train}[ sigma_mod1(e,t,d)^2 ]
4. Re-price all swaptions using their cell-specific scale (Monte Carlo).
   This is the honest check: the linearity assumption is approximate, so
   actual MC prices may differ from s*(e,t) * sigma_mod1(e,t,d).
5. Compute in-sample and OOS MAE / RMSE per cell and overall.
6. Compare to: single global scale (from calibration_summary.json)
   and to the naive no-calibration baseline (s=1).

Outputs (all to dim4_stable_hscale/per_cell/)
---------------------------------------------
  per_cell_scales.json        nine scale factors s*(e,t)
  per_cell_results.csv        per-(date, expiry, tenor) MC prices + errors
  per_cell_summary.json       aggregate train / OOS MAE and RMSE
  fig_per_cell_heatmap_train.png
  fig_per_cell_heatmap_oos.png
  fig_per_cell_scatter.png    mod vs mkt, train vs test coloured
  fig_scales_heatmap.png      heatmap of the 9 fitted scale factors
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

import os as _os
_os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

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
TRAIN_FRAC  = 0.70      # first 70% of dates (by time) for calibration

CKPT_STABLE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
HSCALE_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
OUT_DIR = os.path.join(HSCALE_DIR, "per_cell")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

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
# Load data
# ================================================================

meta, X_tensor, *_ = my_data(use=USE)
X_tensor = X_tensor.float()

meta_eur = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_eur    = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0

dates_swap  = set(pd.to_datetime(meta_eur["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_eur.iterrows()
}

combos = df_vol[["as_of_date", "option_maturity", "swap_tenor", "market_vol"]].drop_duplicates()
print(f"EUR: {len(meta_eur)} dates, {len(combos)} vol targets")

# ================================================================
# Train / test split  (chronological by unique date)
# ================================================================

all_dates  = sorted(df_vol["as_of_date"].unique())
n_train    = int(len(all_dates) * TRAIN_FRAC)
train_dates = set(all_dates[:n_train])
test_dates  = set(all_dates[n_train:])

print(f"\nSplit: {len(train_dates)} train dates "
      f"({min(train_dates).date()} – {max(train_dates).date()})")
print(f"       {len(test_dates)} test  dates "
      f"({min(test_dates).date()} – {max(test_dates).date()})")

# ================================================================
# Step 1 — price at s = 1 (or reload from cache)
# ================================================================

BASE_CSV = os.path.join(HSCALE_DIR, "baseline_vols_s1.csv")

if os.path.isfile(BASE_CSV):
    print(f"\nLoading cached s=1 prices from {BASE_CSV}")
    df_base = pd.read_csv(BASE_CSV)
    df_base["date"] = pd.to_datetime(df_base["date"])
else:
    print("\n=== Step 1: pricing at s=1 ===")

    @torch.no_grad()
    def _price(date, expiry, tenor, s=1.0):
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
        z_T  = torch.cat([z1[:,-1,:], z2[:,-1,:]], dim=0)
        r_all = torch.cat([r1, r2], dim=0)
        D_T  = compute_discount_paths(r_all, dt_eff)[:,-1]
        ok   = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
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
        return sig_mod, int(ok.sum()) / N_PATHS

    rows = []
    for _, row in combos.iterrows():
        date   = pd.Timestamp(row["as_of_date"]).normalize()
        expiry = int(row["option_maturity"])
        tenor  = int(row["swap_tenor"])
        sig_mkt = float(row["market_vol"]) * 10_000.0
        res = _price(date, expiry, tenor, s=1.0)
        if res is None:
            sig_mod1, pfrac = float("nan"), 0.0
        else:
            sig_mod1, pfrac = res
        rows.append({"date": date.date(), "expiry": expiry, "tenor": tenor,
                     "sigma_mkt": round(sig_mkt, 1),
                     "sigma_mod1": round(sig_mod1, 1) if math.isfinite(sig_mod1) else None,
                     "path_frac": round(pfrac, 3)})
    df_base = pd.DataFrame(rows)
    df_base["date"] = pd.to_datetime(df_base["date"])
    df_base.to_csv(BASE_CSV, index=False)
    print(f"Saved {BASE_CSV}")

# ================================================================
# Step 2 — per-cell OLS on TRAINING dates
# ================================================================

print("\n=== Step 2: per-cell OLS calibration on training dates ===")

# Only use rows where both sigma_mod1 and sigma_mkt are valid (mkt > 0)
df_valid = df_base[
    df_base["sigma_mod1"].notna() &
    (df_base["sigma_mkt"] > 0)
].copy()
df_valid["split"] = df_valid["date"].apply(
    lambda d: "train" if pd.Timestamp(d) in train_dates else "test"
)

cells = sorted(df_valid[["expiry","tenor"]].drop_duplicates()
               .itertuples(index=False), key=lambda x: (x.expiry, x.tenor))

scales = {}
print(f"\n{'Cell':>8}  {'s*':>8}  {'n_train':>8}  {'R²':>6}")
print("-" * 38)
for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub = df_valid[(df_valid["expiry"]==e) & (df_valid["tenor"]==t)
                   & (df_valid["split"]=="train")]
    if sub.empty:
        scales[(e, t)] = 1.0
        continue
    mod1 = sub["sigma_mod1"].values.astype(float)
    mkt  = sub["sigma_mkt"].values.astype(float)
    s_opt = float(np.dot(mod1, mkt) / np.dot(mod1, mod1))
    # R² of the linear regression through the origin
    ss_res = np.sum((s_opt * mod1 - mkt) ** 2)
    ss_tot = np.sum((mkt - mkt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    scales[(e, t)] = s_opt
    print(f"  {e}Yx{t}Y  {s_opt:>8.4f}  {len(sub):>8}  {r2:>6.3f}")

# Save scale factors
scales_serialisable = {f"{e}x{t}": float(v) for (e,t), v in scales.items()}
with open(os.path.join(OUT_DIR, "per_cell_scales.json"), "w") as f:
    json.dump(scales_serialisable, f, indent=2)
print(f"\nSaved scale factors -> per_cell_scales.json")

# ================================================================
# Step 3 — re-price all swaptions at cell-specific scales (MC)
# ================================================================

print("\n=== Step 3: MC re-pricing at per-cell scales ===")

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

result_rows = []
n_ok = 0
for _, row in combos.iterrows():
    date   = pd.Timestamp(row["as_of_date"]).normalize()
    expiry = int(row["option_maturity"])
    tenor  = int(row["swap_tenor"])
    sig_mkt = float(row["market_vol"]) * 10_000.0
    s_cell  = scales.get((expiry, tenor), 1.0)

    res = price_at_scale(date, expiry, tenor, s_cell)
    if res is None:
        sig_mod, pfrac = float("nan"), 0.0
    else:
        sig_mod, pfrac = res
        if math.isfinite(sig_mod) and sig_mod > 0:
            n_ok += 1
        else:
            sig_mod = float("nan")

    split = "train" if date in train_dates else "test"
    result_rows.append({
        "date":       date.date(),
        "expiry":     expiry,
        "tenor":      tenor,
        "split":      split,
        "sigma_mkt":  round(sig_mkt, 1),
        "sigma_cal":  round(sig_mod, 1) if math.isfinite(sig_mod) else None,
        "s_cell":     round(s_cell, 5),
        "path_frac":  round(pfrac, 3),
    })

print(f"Priced {n_ok}/{len(combos)} swaptions at per-cell scales")

df_res = pd.DataFrame(result_rows)
df_res["date"] = pd.to_datetime(df_res["date"])

# Merge in sigma_mod1 from df_base
df_base_merge = df_base[["date","expiry","tenor","sigma_mod1"]].copy()
df_base_merge["date"] = pd.to_datetime(df_base_merge["date"])
df_res = df_res.merge(df_base_merge, on=["date","expiry","tenor"], how="left")

df_res["err_s1_bp"]  = df_res["sigma_mod1"] - df_res["sigma_mkt"]
df_res["err_cal_bp"] = df_res["sigma_cal"]  - df_res["sigma_mkt"]

df_res.to_csv(os.path.join(OUT_DIR, "per_cell_results.csv"), index=False)

# ================================================================
# Step 4 — aggregate MAE / RMSE, train vs OOS
# ================================================================

print("\n=== Step 4: MAE / RMSE summary ===")

def agg(sub):
    valid = sub[sub["err_cal_bp"].notna()]
    valid_s1 = sub[sub["err_s1_bp"].notna()]
    return {
        "n":         len(valid),
        "mae_s1":    round(valid_s1["err_s1_bp"].abs().mean(),  1) if not valid_s1.empty else float("nan"),
        "mae_cal":   round(valid["err_cal_bp"].abs().mean(),     1) if not valid.empty   else float("nan"),
        "rmse_s1":   round((valid_s1["err_s1_bp"]**2).mean()**0.5, 1) if not valid_s1.empty else float("nan"),
        "rmse_cal":  round((valid["err_cal_bp"]**2).mean()**0.5,    1) if not valid.empty   else float("nan"),
    }

summary = {}

print(f"\n{'Cell':>8}  s*(e,t)  "
      f"{'MAE_s1':>7} {'MAE_train':>10} {'MAE_oos':>8}  "
      f"{'RMSE_s1':>8} {'RMSE_oos':>9}")
print("-" * 72)
for cell in cells:
    e, t = cell.expiry, cell.tenor
    s = scales[(e, t)]
    sub_all   = df_res[(df_res["expiry"]==e) & (df_res["tenor"]==t)]
    sub_train = sub_all[sub_all["split"]=="train"]
    sub_test  = sub_all[sub_all["split"]=="test"]
    a_all   = agg(sub_all)
    a_train = agg(sub_train)
    a_test  = agg(sub_test)
    summary[f"{e}x{t}"] = {"s": s, "train": a_train, "oos": a_test, "all": a_all}
    print(f"  {e}Yx{t}Y  {s:>7.4f}  "
          f"{a_all['mae_s1']:>7.1f} {a_train['mae_cal']:>10.1f} {a_test['mae_cal']:>8.1f}  "
          f"{a_all['rmse_s1']:>8.1f} {a_test['rmse_cal']:>9.1f}")

# Overall
for split_name, split_key in [("train","train"),("OOS","test"),("all","all")]:
    sub = df_res if split_key == "all" else df_res[df_res["split"]==split_key]
    a   = agg(sub)
    summary[f"overall_{split_key}"] = a
    print(f"\nOverall {split_name}:  MAE_s1={a['mae_s1']:.1f}bp  "
          f"MAE_cal={a['mae_cal']:.1f}bp  RMSE_cal={a['rmse_cal']:.1f}bp")

# Compare to single global scale
global_summary_path = os.path.join(HSCALE_DIR, "calibration_summary.json")
if os.path.isfile(global_summary_path):
    with open(global_summary_path) as f:
        g = json.load(f)
    print(f"\nSingle global scale: MAE={g.get('vol_mae_cal_bp','?')}bp  "
          f"RMSE={g.get('vol_rmse_cal_bp','?')}bp  (s={g.get('s_ols','?')})")

with open(os.path.join(OUT_DIR, "per_cell_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ================================================================
# Figures
# ================================================================

expiry_vals = sorted(df_res["expiry"].dropna().unique().astype(int))
tenor_vals  = sorted(df_res["tenor"].dropna().unique().astype(int))


def mae_pivot(sub, col="err_cal_bp"):
    pivot_data = (
        sub[sub[col].notna()]
        .groupby(["expiry","tenor"])[col]
        .apply(lambda x: x.abs().mean())
        .unstack("tenor")
        .reindex(index=expiry_vals, columns=tenor_vals)
    )
    return pivot_data


def plot_heatmap(pivot, title, fname, vmax=None):
    fig, ax = plt.subplots(figsize=(max(4, len(tenor_vals)),
                                    max(3, len(expiry_vals))), dpi=150)
    vmax = vmax or max(50.0, float(pivot.abs().max().max()))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}Y" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r}Y" for r in pivot.index])
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(title)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="MAE (bp)")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


# Scales heatmap
scales_arr = np.full((len(expiry_vals), len(tenor_vals)), np.nan)
for i, e in enumerate(expiry_vals):
    for j, t in enumerate(tenor_vals):
        scales_arr[i, j] = scales.get((e, t), np.nan)
fig, ax = plt.subplots(figsize=(max(4, len(tenor_vals)),
                                 max(3, len(expiry_vals))), dpi=150)
im = ax.imshow(scales_arr, cmap="Blues_r", aspect="auto",
               vmin=0, vmax=float(np.nanmax(scales_arr)))
ax.set_xticks(range(len(tenor_vals)))
ax.set_xticklabels([f"{t}Y" for t in tenor_vals])
ax.set_yticks(range(len(expiry_vals)))
ax.set_yticklabels([f"{e}Y" for e in expiry_vals])
ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
ax.set_title("Per-cell diffusion scale factors $s^*(e,t)$")
for i in range(len(expiry_vals)):
    for j in range(len(tenor_vals)):
        v = scales_arr[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if v < 0.3 else "black")
plt.colorbar(im, ax=ax, label="Scale $s^*(e,t)$")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_scales_heatmap.png"), dpi=200)
plt.close(fig)
print("Saved fig_scales_heatmap.png")

# Train / OOS MAE heatmaps (shared colour scale)
piv_train = mae_pivot(df_res[df_res["split"]=="train"])
piv_oos   = mae_pivot(df_res[df_res["split"]=="test"])
piv_s1    = mae_pivot(df_res, col="err_s1_bp")
vmax_shared = float(max(piv_train.max().max(), piv_oos.max().max(), 1))

plot_heatmap(piv_s1,   "Vol MAE (bp) — no calibration (s=1)",
             os.path.join(OUT_DIR, "fig_mae_s1.png"), vmax=float(piv_s1.max().max()))
plot_heatmap(piv_train, "Vol MAE (bp) — per-cell calibration (train)",
             os.path.join(OUT_DIR, "fig_mae_train.png"), vmax=vmax_shared)
plot_heatmap(piv_oos,   "Vol MAE (bp) — per-cell calibration (OOS test)",
             os.path.join(OUT_DIR, "fig_mae_oos.png"),   vmax=vmax_shared)
print("Saved MAE heatmaps")

# Scatter: model vs market, colour = train/test
valid_all = df_res[df_res["sigma_cal"].notna() & df_res["sigma_mkt"].notna()
                   & (df_res["sigma_mkt"] > 0)]
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
for split, colour, marker in [("train","steelblue","o"), ("test","firebrick","^")]:
    sub = valid_all[valid_all["split"]==split]
    ax.scatter(sub["sigma_mkt"], sub["sigma_cal"], alpha=0.4, s=12,
               color=colour, marker=marker, label=split)
lim = max(valid_all["sigma_mkt"].max(), valid_all["sigma_cal"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("Market vol (bp)"); ax.set_ylabel("Model vol — per-cell cal. (bp)")
ax.set_title("Per-cell calibration: model vs market")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_scatter.png"), dpi=200)
plt.close(fig)
print("Saved fig_scatter.png")

# ================================================================
# LaTeX tables
# ================================================================

# 1. Scale-factor table
lines = [r"\begin{table}[H]", r"\centering",
         r"\caption{Per-cell OLS diffusion scale factors $s^*(e,t)$ fitted on "
         r"the training window. Each factor corrects the model-vol level for that "
         r"(expiry, tenor) cell; long-expiry cells require a larger scale than "
         r"short-expiry cells, reflecting the model's over-prediction of "
         r"short-expiry volatility.}",
         r"\label{tab:per_cell_scales}",
         r"\begin{tabular}{@{}r" + "r" * len(tenor_vals) + r"@{}}",
         r"\toprule",
         r"\textbf{Exp\textbackslash Ten} & " +
         " & ".join(r"\textbf{" + f"{t}Y" + r"}" for t in tenor_vals) + r" \\",
         r"\midrule"]
prev_e = None
for i, e in enumerate(expiry_vals):
    if prev_e is not None:
        lines.append(r"\addlinespace[2pt]")
    prev_e = e
    vals = [f"{scales.get((e,t), float('nan')):.3f}" for t in tenor_vals]
    lines.append(f"{e}Y & " + " & ".join(vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(OUT_DIR, "tab_per_cell_scales.tex"), "w") as f:
    f.write("\n".join(lines))

# 2. In-sample vs OOS MAE table
lines2 = [r"\begin{table}[H]", r"\centering",
          r"\caption{Per-cell vol MAE (bp): before calibration ($s=1$), "
          r"in-sample (train), and out-of-sample (OOS test). "
          r"Train window: first \SI{70}{\percent} of dates chronologically. "
          r"OOS window: remaining \SI{30}{\percent}.}",
          r"\label{tab:per_cell_mae}",
          r"\begin{tabular}{@{}ccrrr@{}}",
          r"\toprule",
          r"\textbf{Exp} & \textbf{Ten} & "
          r"\textbf{MAE $s=1$ (bp)} & "
          r"\textbf{MAE train (bp)} & "
          r"\textbf{MAE OOS (bp)} \\",
          r"\midrule"]
prev_e = None
for cell in cells:
    e, t = cell.expiry, cell.tenor
    if prev_e is not None and e != prev_e:
        lines2.append(r"\addlinespace[2pt]")
    prev_e = e
    info = summary.get(f"{e}x{t}", {})
    mae_s1  = info.get("all",   {}).get("mae_s1",  float("nan"))
    mae_tr  = info.get("train", {}).get("mae_cal", float("nan"))
    mae_oos = info.get("oos",   {}).get("mae_cal", float("nan"))
    def fmt(v):
        return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(int(round(v)))
    lines2.append(f"{e} & {t} & {fmt(mae_s1)} & {fmt(mae_tr)} & {fmt(mae_oos)} \\\\")
lines2.append(r"\midrule")
ov_all   = summary.get("overall_all",   {})
ov_train = summary.get("overall_train", {})
ov_oos   = summary.get("overall_test",  {})
lines2.append(r"\textbf{Overall} & & " +
              f"{fmt(ov_all.get('mae_s1'))} & {fmt(ov_train.get('mae_cal'))} & "
              f"{fmt(ov_oos.get('mae_cal'))} \\\\")
lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(OUT_DIR, "tab_per_cell_mae.tex"), "w") as f:
    f.write("\n".join(lines2))

print("\nSaved LaTeX tables.")
print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
