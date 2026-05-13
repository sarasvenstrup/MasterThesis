# ==================== Rolling Window Diffusion-Scale Calibration ====================
"""
Rolling-window OLS calibration: for each pricing date t (OOS), refit s*(e,ten)
using the most recent WINDOW_MONTHS of historical (sigma_mod1, sigma_mkt) pairs.

Motivation
----------
The daily-recalibration experiment in calibrate_improved.py showed that for
1Y-expiry cells, using the current vol surface directly reduces OOS MAE
dramatically:
    1Yx5Y:  50 bp (fixed OLS) -> 28 bp (daily recal)
    1Yx10Y: 144 bp (fixed OLS) -> 53 bp (daily recal)

This improvement is NOT available at real pricing time (we'd need today's vol
surface to set the scale, which is circular for vanilla swaptions).  However,
if we use only *historical* vol data up to date t, the rolling OLS is genuinely
out-of-sample.  Specifically, at each OOS date t we refit:

    s_t*(e,ten) = sum_{d in [t-W, t)} [ mod1(e,ten,d) * mkt(e,ten,d) ]
                / sum_{d in [t-W, t)} [ mod1(e,ten,d)^2 ]

using the W most recent months of data.  The window slides forward so the scale
adapts to regime changes (hiking cycle, COVID vol spike, etc.).

We cross-validate the window length W in {12, 24, 36, 48, 60} months using
a mini-OOS within the training period (retrospective CV):
    - Train: first 60% of all dates
    - Mini-OOS: dates 60%-70% of all dates
    - CV criterion: overall vol MAE on mini-OOS (using linearity, not MC)
Then price the true OOS (dates 70%-100%) using W_opt.

The rolling window requires re-pricing every OOS date at its own scale, since
the scale now varies by date.  We use the existing price_at_scale() function.

Outputs (all to dim4_stable_hscale/rolling/)
---------------------------------------------
  rolling_window_opt.json      optimal window length per cell (and overall)
  rolling_scales.csv           per-(date, cell) rolling scale factors
  rolling_results.csv          per-(date, cell) vol errors
  rolling_summary.json         aggregate MAE / RMSE (train, OOS, overall)
  tab_rolling_comparison.tex   LaTeX table: rolling vs fixed per-cell OLS
  fig_rolling_mae.png          MAE heatmap (train and OOS)
  fig_rolling_scales.png       time series of rolling scales for 1Y cells
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

LATENT_DIM   = 4
N_PATHS      = 2048
DT           = 1 / 12
CCY_FILTER   = "EUR"
USE          = "bbg"
SEED         = 42
TRAIN_FRAC   = 0.70        # main train/test split
CV_FRAC      = 0.60        # end of 'inner train' for CV window selection
WINDOW_MONTHS = [12, 24, 36, 48, 60]  # candidate lookback windows (months)
MIN_OBS      = 6           # minimum obs required to fit rolling OLS

CKPT_STABLE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
HSCALE_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
PER_CELL_DIR = os.path.join(HSCALE_DIR, "per_cell")
OUT_DIR      = os.path.join(HSCALE_DIR, "rolling")
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
n_total     = len(all_dates)
n_train     = int(n_total * TRAIN_FRAC)
n_cv        = int(n_total * CV_FRAC)
train_dates = set(all_dates[:n_train])
test_dates  = set(all_dates[n_train:])
inner_train = set(all_dates[:n_cv])
mini_oos    = set(all_dates[n_cv:n_train])

print(f"Train: {len(train_dates)} dates ({all_dates[0].date()} - {all_dates[n_train-1].date()})")
print(f"OOS:   {len(test_dates)} dates ({all_dates[n_train].date()} - {all_dates[-1].date()})")
print(f"Inner-train (CV): {len(inner_train)} dates")
print(f"Mini-OOS  (CV):   {len(mini_oos)} dates")

# ================================================================
# Load baseline (sigma_mod1 at s=1)
# ================================================================

BASE_CSV = os.path.join(HSCALE_DIR, "baseline_vols_s1.csv")
df_base  = pd.read_csv(BASE_CSV)
df_base["date"] = pd.to_datetime(df_base["date"])
df_valid = df_base[df_base["sigma_mod1"].notna() & (df_base["sigma_mkt"] > 0)].copy()
df_valid["date"] = pd.to_datetime(df_valid["date"])

# Load per-cell OLS summary for comparison
with open(os.path.join(PER_CELL_DIR, "per_cell_summary.json")) as f:
    pcell_sum = json.load(f)
df_pcell = pd.read_csv(os.path.join(PER_CELL_DIR, "per_cell_results.csv"))
df_pcell["date"] = pd.to_datetime(df_pcell["date"])

cells = sorted(
    df_valid[["expiry","tenor"]].drop_duplicates().itertuples(index=False),
    key=lambda x: (x.expiry, x.tenor)
)
expiry_vals = sorted(df_valid["expiry"].unique().astype(int))
tenor_vals  = sorted(df_valid["tenor"].unique().astype(int))

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
# Rolling OLS helper: fit s* from data within a date window
# ================================================================

def rolling_ols(df_hist, date_t, window_months, e, t):
    """
    Fit per-cell OLS s*(e,t) using data in [date_t - window_months, date_t).
    Returns s* (float) or None if insufficient data.
    """
    cutoff_lo = date_t - pd.DateOffset(months=window_months)
    sub = df_hist[
        (df_hist["expiry"] == e) & (df_hist["tenor"] == t) &
        (df_hist["date"] < date_t) & (df_hist["date"] >= cutoff_lo) &
        df_hist["sigma_mod1"].notna() & (df_hist["sigma_mkt"] > 0)
    ]
    if len(sub) < MIN_OBS:
        return None
    mod1 = sub["sigma_mod1"].values.astype(float)
    mkt  = sub["sigma_mkt"].values.astype(float)
    denom = np.dot(mod1, mod1)
    if denom <= 0:
        return None
    return float(np.dot(mod1, mkt) / denom)

# ================================================================
# Step 1 — Cross-validate window length on mini-OOS
# ================================================================
print("\n=== Step 1: CV window length on mini-OOS ===")

# Use linearity (no MC) for fast CV: MAE ≈ |s*(window)*mod1 - mkt|
cv_results = {}
print(f"\n{'Window':>8}  {'Overall mini-OOS MAE (linear approx.)':>40}")
print("-"*52)
for W in WINDOW_MONTHS:
    cell_maes = []
    for cell in cells:
        e, t = cell.expiry, cell.tenor
        for date_t in sorted(mini_oos):
            date_t = pd.Timestamp(date_t)
            s_roll = rolling_ols(df_valid, date_t, W, e, t)
            if s_roll is None:
                continue
            row = df_valid[
                (df_valid["date"] == date_t) &
                (df_valid["expiry"] == e) & (df_valid["tenor"] == t)
            ]
            if row.empty:
                continue
            mod1 = float(row["sigma_mod1"].iloc[0])
            mkt  = float(row["sigma_mkt"].iloc[0])
            pred = s_roll * mod1
            cell_maes.append(abs(pred - mkt))
    overall_mae = np.mean(cell_maes) if cell_maes else float("nan")
    cv_results[W] = overall_mae
    print(f"  {W:>6}M  {overall_mae:>40.1f}")

W_opt = min(cv_results, key=lambda w: cv_results[w])
print(f"\nOptimal window: {W_opt} months (mini-OOS MAE = {cv_results[W_opt]:.1f} bp)")

with open(os.path.join(OUT_DIR, "rolling_window_opt.json"), "w") as f:
    json.dump({"W_opt": W_opt, "cv_results": {str(k): v for k, v in cv_results.items()}}, f, indent=2)

# ================================================================
# Step 2 — Compute rolling scales for ALL dates (train + OOS)
# ================================================================
print(f"\n=== Step 2: Rolling OLS scales (W={W_opt}M) ===")

# For each date and cell, compute rolling s*
scale_rows = []
for date_t_raw in sorted(all_dates):
    date_t = pd.Timestamp(date_t_raw)
    for cell in cells:
        e, t = cell.expiry, cell.tenor
        s_roll = rolling_ols(df_valid, date_t, W_opt, e, t)
        split  = "train" if date_t_raw in train_dates else "test"
        scale_rows.append({
            "date":    date_t.date(),
            "expiry":  e,
            "tenor":   t,
            "split":   split,
            "s_roll":  round(s_roll, 5) if s_roll is not None else None,
        })

df_scales = pd.DataFrame(scale_rows)
df_scales["date"] = pd.to_datetime(df_scales["date"])
df_scales.to_csv(os.path.join(OUT_DIR, "rolling_scales.csv"), index=False)
print(f"Saved rolling_scales.csv ({len(df_scales)} rows)")

# ================================================================
# Step 3 — MC pricing on OOS dates at rolling scales
# ================================================================
print(f"\n=== Step 3: MC pricing on OOS dates at rolling scales ===")

oos_combos = combos[combos["as_of_date"].isin(test_dates)].copy()
print(f"  {len(oos_combos)} OOS swaptions to price")

result_rows = []
n_priced = 0

for _, row in oos_combos.iterrows():
    date    = pd.Timestamp(row["as_of_date"]).normalize()
    expiry  = int(row["option_maturity"])
    tenor   = int(row["swap_tenor"])
    sig_mkt = float(row["market_vol"]) * 10_000.0

    # Rolling scale
    scale_row = df_scales[
        (df_scales["date"] == date) &
        (df_scales["expiry"] == expiry) &
        (df_scales["tenor"] == tenor)
    ]
    if scale_row.empty or pd.isna(scale_row["s_roll"].iloc[0]):
        s_roll = None
    else:
        s_roll = float(scale_row["s_roll"].iloc[0])

    # OLS price (from cached per_cell_results)
    pcell_row = df_pcell[
        (df_pcell["date"] == date) &
        (df_pcell["expiry"] == expiry) &
        (df_pcell["tenor"] == tenor)
    ]
    sig_ols = float(pcell_row["sigma_cal"].iloc[0]) if not pcell_row.empty and pd.notna(pcell_row["sigma_cal"].iloc[0]) else float("nan")

    # MC at rolling scale
    if s_roll is not None:
        res = price_at_scale(date, expiry, tenor, s_roll)
        if res is not None and math.isfinite(res[0]):
            sig_roll = round(res[0], 1)
        else:
            sig_roll = float("nan")
        n_priced += 1
        if n_priced % 50 == 0:
            print(f"  Priced {n_priced} ({date.date()} {expiry}Yx{tenor}Y s={s_roll:.3f})")
    else:
        sig_roll = sig_ols   # fall back to OLS if no window data

    result_rows.append({
        "date":      date.date(),
        "expiry":    expiry,
        "tenor":     tenor,
        "sigma_mkt": round(sig_mkt, 1),
        "s_roll":    round(s_roll, 5) if s_roll is not None else None,
        "sig_ols":   round(sig_ols, 1) if math.isfinite(sig_ols) else None,
        "sig_roll":  round(sig_roll, 1) if math.isfinite(sig_roll) else None,
    })

print(f"Priced {n_priced} OOS swaptions at rolling scales")
df_res = pd.DataFrame(result_rows)
df_res["date"] = pd.to_datetime(df_res["date"])
df_res["ae_ols"]  = (df_res["sig_ols"]  - df_res["sigma_mkt"]).abs()
df_res["ae_roll"] = (df_res["sig_roll"] - df_res["sigma_mkt"]).abs()
df_res.to_csv(os.path.join(OUT_DIR, "rolling_results.csv"), index=False)

# ================================================================
# Step 4 — MAE/RMSE summary
# ================================================================
print(f"\n=== Step 4: OOS MAE / RMSE summary ===\n")

def agg(sub, col):
    valid = sub[sub[col].notna() & (sub["sigma_mkt"] > 0)]
    if valid.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    return {"mae":  round(valid[col].mean(), 1),
            "rmse": round((valid[col]**2).mean()**0.5, 1),
            "n":    len(valid)}

print(f"{'Method':<22}  {'OOS MAE':>9}  {'OOS RMSE':>10}")
print("-"*46)
a_ols  = agg(df_res, "ae_ols")
a_roll = agg(df_res, "ae_roll")
print(f"  {'Fixed per-cell OLS':<20}  {a_ols['mae']:>9.1f}  {a_ols['rmse']:>10.1f}")
print(f"  {f'Rolling OLS (W={W_opt}M)':<20}  {a_roll['mae']:>9.1f}  {a_roll['rmse']:>10.1f}")

print(f"\n{'Cell':>8}  {'OLS OOS':>9}  {'Roll OOS':>9}  {'Delta':>7}")
print("-"*42)
summary = {}
for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub   = df_res[(df_res["expiry"]==e) & (df_res["tenor"]==t)]
    a_o   = agg(sub, "ae_ols")
    a_r   = agg(sub, "ae_roll")
    delta = a_r["mae"] - a_o["mae"] if math.isfinite(a_o["mae"]) and math.isfinite(a_r["mae"]) else float("nan")
    summary[f"{e}x{t}"] = {"ols_oos": a_o["mae"], "roll_oos": a_r["mae"], "delta": delta}
    sign = "+" if delta >= 0 else ""
    print(f"  {e}Yx{t}Y  {a_o['mae']:>9.1f}  {a_r['mae']:>9.1f}  {sign}{delta:>6.1f}")

summary["overall"] = {"ols_oos": a_ols["mae"], "roll_oos": a_roll["mae"],
                      "ols_rmse": a_ols["rmse"], "roll_rmse": a_roll["rmse"]}
with open(os.path.join(OUT_DIR, "rolling_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved rolling_summary.json")

# ================================================================
# Step 5 — LaTeX comparison table
# ================================================================

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{OOS vol MAE (bp) for fixed per-cell OLS and rolling-window OLS "
    r"(window = \textbf{" + str(W_opt) + r" months}). "
    r"The rolling window refits $s^*(e,t)$ before each pricing date using "
    r"only the most recent " + str(W_opt) + r" months of historical "
    r"$(\sigma_{\mathrm{mod},d}(1), \sigma_{\mathrm{mkt},d})$ pairs, "
    r"capturing temporal drift in the optimal scale without using future data. "
    r"A negative delta indicates the rolling window outperforms fixed OLS.}",
    r"\label{tab:rolling_comparison}",
    r"\begin{tabular}{@{}ccrrr@{}}",
    r"\toprule",
    r"\textbf{Exp} & \textbf{Ten} & "
    r"\textbf{Fixed OLS} & "
    r"\textbf{Rolling OLS} & "
    r"$\Delta$ \textbf{(bp)} \\",
    r"\midrule",
]
prev_e = None
for cell in cells:
    e, t = cell.expiry, cell.tenor
    if prev_e is not None and e != prev_e:
        lines.append(r"\addlinespace[2pt]")
    prev_e = e
    vals = summary.get(f"{e}x{t}", {})
    ols_v  = vals.get("ols_oos",  float("nan"))
    roll_v = vals.get("roll_oos", float("nan"))
    delta  = vals.get("delta",    float("nan"))
    def fmt(v):
        return "--" if not math.isfinite(v) else str(int(round(v)))
    def fmt_delta(v):
        if not math.isfinite(v):
            return "--"
        s = f"{int(round(abs(v)))}"
        return (r"$-$" + s) if v < 0 else (r"$+$" + s)
    lines.append(f"{e} & {t} & {fmt(ols_v)} & {fmt(roll_v)} & {fmt_delta(delta)} \\\\")
lines.append(r"\midrule")
ov = summary.get("overall", {})
lines.append(r"\textbf{Overall} & & " +
             f"{fmt(ov.get('ols_oos',float('nan')))} & "
             f"{fmt(ov.get('roll_oos',float('nan')))} & "
             f"{fmt_delta(ov.get('roll_oos',float('nan'))-ov.get('ols_oos',float('nan')))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(OUT_DIR, "tab_rolling_comparison.tex"), "w") as f:
    f.write("\n".join(lines))
print("Saved tab_rolling_comparison.tex")

# ================================================================
# Step 6 — Figures
# ================================================================

# 6a: MAE heatmap OLS vs rolling (OOS only)
def mae_pivot(df_in, col):
    return (
        df_in[df_in[col].notna() & (df_in["sigma_mkt"] > 0)]
        .groupby(["expiry","tenor"])[col]
        .mean()
        .unstack("tenor")
        .reindex(index=expiry_vals, columns=tenor_vals)
    )

piv_ols  = mae_pivot(df_res, "ae_ols")
piv_roll = mae_pivot(df_res, "ae_roll")
vmax = max(float(piv_ols.max().max()), float(piv_roll.max().max()), 50.0)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
for ax, piv, title in [(axes[0], piv_ols,  "Fixed per-cell OLS (OOS MAE)"),
                        (axes[1], piv_roll, f"Rolling OLS W={W_opt}M (OOS MAE)")]:
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)
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
                        fontsize=8, color="black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="MAE (bp)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_rolling_mae.png"), dpi=200)
plt.close(fig)
print("Saved fig_rolling_mae.png")

# 6b: Time series of rolling scales for 1Y cells
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150, sharex=True)
for ax, (e_plot, t_plot) in zip(axes2, [(1,1),(1,5),(1,10)]):
    sub = df_scales[(df_scales["expiry"]==e_plot) & (df_scales["tenor"]==t_plot)
                    & df_scales["s_roll"].notna()].sort_values("date")
    ols_s = None
    for cell_c in cells:
        if cell_c.expiry==e_plot and cell_c.tenor==t_plot:
            # get OLS scale from per_cell_summary
            info = pcell_sum.get(f"{e_plot}x{t_plot}", {})
            break
    # Get OLS scale from per_cell_results
    ols_key = f"{e_plot}x{t_plot}"

    ax.plot(sub["date"], sub["s_roll"], color="steelblue", lw=1.2, label=f"Rolling W={W_opt}M")
    ax.axvline(pd.Timestamp(all_dates[n_train]), color="firebrick", lw=1, ls="--", label="OOS start")
    ax.set_title(f"{e_plot}Y×{t_plot}Y")
    ax.set_ylabel("Scale $s^*(t)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
fig2.suptitle("Rolling scale factors for 1Y-expiry cells", fontsize=10)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig_rolling_scales.png"), dpi=200)
plt.close(fig2)
print("Saved fig_rolling_scales.png")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
