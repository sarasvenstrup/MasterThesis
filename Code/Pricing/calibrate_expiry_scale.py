# ==================== Expiry-Level Calibration ====================
"""
Fits one diffusion scale per EXPIRY row (3 parameters: s_1Y, s_5Y, s_10Y)
instead of one per (expiry, tenor) cell (9 parameters).

Motivation
----------
The per-cell OLS suffers from temporal regime overfitting for the 1Yx10Y cell
(23 bp train -> 144 bp OOS). Ridge regression cannot fix this because LOO-year CV
within the training data doesn't expose the problem.

The expiry-level calibration pools all tenors together for a given expiry,
reducing 9 parameters to 3. Since the diffusion scale physically affects
the SIMULATION (length T_e years), it should depend on expiry, not tenor.
Tenor affects the payoff calculation (bond sum) but not the SDE horizon.

This gives:
  - 3 expiry-specific scales: s_1Y, s_5Y, s_10Y
  - All three estimated on training dates, evaluated on OOS test

Protocol
--------
1. Load baseline_vols_s1.csv (cached sigma_mod1 per date/cell)
2. Per-expiry OLS on training dates (pool all tenors within expiry):
       s*(e) = sum_{d,t} [ sigma_mod1(e,t,d) * sigma_mkt(e,t,d) ]
             / sum_{d,t} [ sigma_mod1(e,t,d)^2 ]
3. Reprice OOS dates at expiry-specific scales (one scale per expiry row)
4. Compare to per-cell OLS (9 params) and global scale (1 param)

Outputs (to dim4_stable_hscale/expiry_scale/)
----------------------------------------------
  expiry_scales.json        three scale factors
  expiry_results.csv        per-(date,cell) errors
  expiry_summary.json       aggregate MAE/RMSE
  tab_expiry_comparison.tex LaTeX table
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

LATENT_DIM = 4
N_PATHS    = 2048
DT         = 1 / 12
CCY_FILTER = "EUR"
USE        = "bbg"
SEED       = 42
TRAIN_FRAC = 0.70

CKPT_STABLE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
HSCALE_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
OUT_DIR = os.path.join(HSCALE_DIR, "expiry_scale")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device   = torch.device("cpu")
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
# Load cached baseline (sigma_mod1 at s=1)
# ================================================================

BASE_CSV = os.path.join(HSCALE_DIR, "baseline_vols_s1.csv")
df_base  = pd.read_csv(BASE_CSV)
df_base["date"] = pd.to_datetime(df_base["date"])
print(f"Loaded s=1 baseline: {len(df_base)} rows")

# ================================================================
# MC pricing helper
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
    return sig_mod, int(ok.sum()) / N_PATHS

# ================================================================
# Step 1 — Per-expiry OLS (pool tenors)
# ================================================================
print("\n=== Step 1: Per-expiry OLS (pooling tenors) ===")

df_valid = df_base[
    df_base["sigma_mod1"].notna() & (df_base["sigma_mkt"] > 0)
].copy()
df_valid["split"] = df_valid["date"].apply(
    lambda d: "train" if pd.Timestamp(d) in train_dates else "test")
df_train = df_valid[df_valid["split"] == "train"]

expiry_vals = sorted(df_valid["expiry"].unique().astype(int))
tenor_vals  = sorted(df_valid["tenor"].unique().astype(int))

expiry_scales = {}
print(f"\n{'Expiry':>8}  {'s*(e)':>8}  {'n_train':>8}  {'R²':>6}")
print("-"*38)
for e in expiry_vals:
    sub  = df_train[df_train["expiry"] == e]
    mod1 = sub["sigma_mod1"].values.astype(float)
    mkt  = sub["sigma_mkt"].values.astype(float)
    s_e  = float(np.dot(mod1, mkt) / np.dot(mod1, mod1))
    ss_res = np.sum((s_e * mod1 - mkt)**2)
    ss_tot = np.sum((mkt - mkt.mean())**2)
    r2     = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    expiry_scales[e] = s_e
    print(f"  {e}Y      {s_e:>8.4f}  {len(sub):>8}  {r2:>6.3f}")

scales_ser = {f"{e}": float(v) for e, v in expiry_scales.items()}
with open(os.path.join(OUT_DIR, "expiry_scales.json"), "w") as f:
    json.dump(scales_ser, f, indent=2)
print("Saved expiry_scales.json")

# ================================================================
# Step 2 — MC pricing at expiry-specific scales
# ================================================================
print("\n=== Step 2: MC re-pricing at expiry-level scales ===")

cells = sorted(
    df_valid[["expiry","tenor"]].drop_duplicates().itertuples(index=False),
    key=lambda x: (x.expiry, x.tenor)
)

result_rows = []
n_ok = 0
for _, row in combos.iterrows():
    date    = pd.Timestamp(row["as_of_date"]).normalize()
    expiry  = int(row["option_maturity"])
    tenor   = int(row["swap_tenor"])
    sig_mkt = float(row["market_vol"]) * 10_000.0
    s_e     = expiry_scales.get(expiry, 0.15)

    res = price_at_scale(date, expiry, tenor, s_e)
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
        "date":      date.date(),
        "expiry":    expiry,
        "tenor":     tenor,
        "split":     split,
        "sigma_mkt": round(sig_mkt, 1),
        "sigma_exp": round(sig_mod, 1) if math.isfinite(sig_mod) else None,
        "s_expiry":  round(s_e, 5),
        "path_frac": round(pfrac, 3),
    })
    if n_ok % 100 == 0 and n_ok > 0:
        print(f"  Priced {n_ok} ({date.date()} {expiry}Yx{tenor}Y)")

print(f"Priced {n_ok}/{len(combos)} swaptions at expiry-level scales")

df_res = pd.DataFrame(result_rows)
df_res["date"] = pd.to_datetime(df_res["date"])
df_res["ae"]   = (df_res["sigma_exp"] - df_res["sigma_mkt"]).abs()
df_res.to_csv(os.path.join(OUT_DIR, "expiry_results.csv"), index=False)

# Merge in sigma_mod1 for reference
df_base_merge = df_base[["date","expiry","tenor","sigma_mod1"]].copy()
df_res = df_res.merge(df_base_merge, on=["date","expiry","tenor"], how="left")

# ================================================================
# Step 3 — Summary
# ================================================================
print("\n=== Step 3: MAE / RMSE summary ===")

def agg(sub, col="ae"):
    valid = sub[sub[col].notna() & (sub["sigma_mkt"] > 0)]
    if valid.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    return {"mae":  round(valid[col].mean(), 1),
            "rmse": round((valid[col]**2).mean()**0.5, 1),
            "n":    len(valid)}

summary = {}
print(f"\n{'Cell':>8}  {'s_expiry':>9}  {'MAE_s1':>7}  {'MAE_train':>10}  {'MAE_oos':>8}")
print("-"*52)
for cell in cells:
    e, t = cell.expiry, cell.tenor
    s_e  = expiry_scales.get(e, 0.15)
    sub_all   = df_res[(df_res["expiry"]==e) & (df_res["tenor"]==t)]
    sub_train = sub_all[sub_all["split"]=="train"]
    sub_test  = sub_all[sub_all["split"]=="test"]
    # s=1 error from df_base
    sub_base  = df_base[(df_base["expiry"]==e) & (df_base["tenor"]==t) &
                        df_base["sigma_mod1"].notna() & (df_base["sigma_mkt"]>0)]
    mae_s1 = float((sub_base["sigma_mod1"] - sub_base["sigma_mkt"]).abs().mean()) if not sub_base.empty else float("nan")
    a_tr = agg(sub_train)
    a_ts = agg(sub_test)
    summary[f"{e}x{t}"] = {"s": s_e, "train": a_tr, "oos": a_ts}
    print(f"  {e}Yx{t}Y  {s_e:>9.4f}  {mae_s1:>7.1f}  "
          f"{a_tr['mae']:>10.1f}  {a_ts['mae']:>8.1f}")

for split_name, split_key in [("train","train"),("OOS","test"),("all","all")]:
    sub = df_res if split_key=="all" else df_res[df_res["split"]==split_key]
    a   = agg(sub)
    summary[f"overall_{split_key}"] = a
    print(f"\nOverall {split_name}:  MAE={a['mae']:.1f}bp  RMSE={a['rmse']:.1f}bp")

with open(os.path.join(OUT_DIR, "expiry_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Saved expiry_summary.json")

# ================================================================
# Step 4 — LaTeX table
# ================================================================

# Load per-cell OLS summary for comparison
pcell_summary_path = os.path.join(HSCALE_DIR, "per_cell", "per_cell_summary.json")
with open(pcell_summary_path) as f:
    pcell_sum = json.load(f)

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Per-cell OOS vol MAE (bp): expiry-level calibration "
    r"(3 parameters) vs per-cell OLS (9 parameters). "
    r"The expiry-level scale pools all tenors within an expiry row, "
    r"reducing the parameter count and eliminating the within-row "
    r"overfitting that occurs for the 1Y$\times$10Y cell.}",
    r"\label{tab:expiry_comparison}",
    r"\begin{tabular}{@{}ccrrrrr@{}}",
    r"\toprule",
    r"\textbf{Exp} & \textbf{Ten} & "
    r"\textbf{$s^*(e)$} & "
    r"\textbf{OLS train} & \textbf{OLS OOS} & "
    r"\textbf{Exp train} & \textbf{Exp OOS} \\",
    r"\midrule",
]
prev_e = None
for cell in cells:
    e, t = cell.expiry, cell.tenor
    if prev_e is not None and e != prev_e:
        lines.append(r"\addlinespace[2pt]")
    prev_e = e
    s_e = expiry_scales.get(e, 0.15)
    info = summary.get(f"{e}x{t}", {})
    pcell_info = pcell_sum.get(f"{e}x{t}", {})
    ols_tr  = pcell_info.get("train",{}).get("mae_cal", float("nan"))
    ols_oos = pcell_info.get("oos",  {}).get("mae_cal", float("nan"))
    exp_tr  = info.get("train",{}).get("mae", float("nan"))
    exp_oos = info.get("oos",  {}).get("mae", float("nan"))
    def fmt(v):
        return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(int(round(v)))
    lines.append(f"{e} & {t} & {s_e:.3f} & {fmt(ols_tr)} & {fmt(ols_oos)} "
                 f"& {fmt(exp_tr)} & {fmt(exp_oos)} \\\\")
lines.append(r"\midrule")
ov_tr_ols  = pcell_sum.get("overall_train",{}).get("mae_cal", float("nan"))
ov_oos_ols = pcell_sum.get("overall_test", {}).get("mae_cal", float("nan"))
ov_tr_exp  = summary.get("overall_train",{}).get("mae", float("nan"))
ov_oos_exp = summary.get("overall_test", {}).get("mae", float("nan"))
def fmt(v):
    return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(int(round(v)))
lines.append(r"\textbf{Overall} & & & " +
             f"{fmt(ov_tr_ols)} & {fmt(ov_oos_ols)} & {fmt(ov_tr_exp)} & {fmt(ov_oos_exp)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(OUT_DIR, "tab_expiry_comparison.tex"), "w") as f:
    f.write("\n".join(lines))
print("Saved tab_expiry_comparison.tex")

# ================================================================
# Step 5 — Heatmap figure
# ================================================================

def mae_pivot(sub, col="ae"):
    return (
        sub[sub[col].notna() & (sub["sigma_mkt"] > 0)]
        .groupby(["expiry","tenor"])[col]
        .mean()
        .unstack("tenor")
        .reindex(index=expiry_vals, columns=tenor_vals)
    )

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
for ax, split, title in [(axes[0],"train","Train MAE (expiry-level)"),
                          (axes[1],"test", "OOS MAE (expiry-level)")]:
    sub  = df_res[df_res["split"]==split]
    piv  = mae_pivot(sub)
    vmax = float(piv.max().max()) if not piv.empty else 300
    im   = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto",
                     vmin=0, vmax=max(50, vmax))
    ax.set_xticks(range(len(tenor_vals)))
    ax.set_xticklabels([f"{c}Y" for c in tenor_vals])
    ax.set_yticks(range(len(expiry_vals)))
    ax.set_yticklabels([f"{r}Y" for r in expiry_vals])
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(title)
    for i in range(len(expiry_vals)):
        for j in range(len(tenor_vals)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="MAE (bp)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_expiry_mae.png"), dpi=200)
plt.close(fig)
print("Saved fig_expiry_mae.png")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
