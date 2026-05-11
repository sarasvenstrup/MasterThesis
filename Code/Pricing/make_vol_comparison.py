# ==================== Swaption Vol Comparison: All Three Models ====================
"""
Produces every pricing output needed for the thesis vol-comparison section.

Models compared
---------------
  1. baseline   dim4, recon-only  — SDE diverges; shows garbage prices
  2. stable     dim4, recon-only  — good recon, but paths NaN; vol not reliable
  3. stable_cal dim4, H-scale cal — stable + calibrated diffusion scale

Outputs (all to Figures/Pricing/vol_comparison/)
-------------------------------------------------
  vol_comparison.csv              raw per-(date,exp,ten) results for all models
  vol_summary_table.tex           LaTeX table: MAE/RMSE per (exp,ten), 3 columns
  fig_vol_heatmap_{model}.png     vol-error heatmaps (exp × ten)
  fig_vol_scatter_{model}.png     scatter mod vs mkt
  fig_path_survival.png           path-survival fractions by model
  calibration_summary.json        s_opt, MAE before/after

Usage
-----
  python Code/Pricing/make_vol_comparison.py

Requirements
------------
  Run calibrate_H_scale.py first to produce the calibrated checkpoint in
  Figures/TrainingResults/dim4_stable_hscale/.
"""

import copy
import json
import math
import os
import sys
import glob

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

OUT_DIR = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "vol_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

sqrt_2pi = math.sqrt(2.0 * math.pi)

# ================================================================
# Checkpoint definitions
# ================================================================

CKPT_BASELINE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_baseline", "checkpoint_dim4_ep5000.pt"
)
CKPT_STABLE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

# Read s_opt from calibration summary (produced by calibrate_H_scale.py).
# The calibrated model uses the ORIGINAL checkpoint + diffusion_scale=s_opt in
# simulation only. The checkpoint with scaled H weights changes the ODE
# coefficients (K and H appear in the reconstruction ODE), which corrupts the
# time-0 strike and expiry bond prices and gives a different — worse — result.
_hscale_dir = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
_cal_summary_path = os.path.join(_hscale_dir, "calibration_summary.json")
if os.path.isfile(_cal_summary_path):
    with open(_cal_summary_path) as _f:
        _cal_summary = json.load(_f)
    S_OPT = float(_cal_summary["s_ols"])
else:
    S_OPT = 1.0
    print("WARNING: calibration_summary.json not found; using s_opt=1.0")

# Baseline is excluded: the simulation chapter already documented its SDE
# divergence (||z_T - z_0|| ~ 10^13). Repeating it here as a vol number
# adds nothing to the story.
MODELS = [
    {
        "key":        "stable",
        "label":      "Stable (dim4)",
        "ckpt":       CKPT_STABLE,
        "is_stable":  True,
        "diff_scale": 1.0,
        "color":      "seagreen",
    },
    {
        "key":        "stable_cal",
        "label":      f"Stable + cal. H (s={S_OPT:.3f})",
        "ckpt":       CKPT_STABLE,           # original weights — ODE unaffected
        "is_stable":  True,
        "diff_scale": S_OPT,                 # scale applied in simulation only
        "color":      "firebrick",
    },
]

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

print(f"EUR: {len(meta_eur)} dates, {len(df_vol)} vol targets")
combos = df_vol[["as_of_date", "option_maturity", "swap_tenor", "market_vol"]].drop_duplicates()

# ================================================================
# Model loader
# ================================================================

def load_model(cfg: dict) -> FullModel:
    ckpt_path = cfg["ckpt"]
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(f"  WARNING: checkpoint not found for {cfg['label']}: {ckpt_path}")
        return None
    raw   = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw["model_state_dict"] if "model_state_dict" in raw else raw
    model = FullModel(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {cfg['label']}: {os.path.basename(ckpt_path)}")
    return model

# ================================================================
# Pricing function (no_grad)
# ================================================================

@torch.no_grad()
def price_swaption(model, date, expiry, tenor, diff_scale=1.0):
    """
    Returns (sigma_mod_bp, path_finite_frac) or None on failure.
    """
    if date not in date_to_idx:
        return None
    idx = date_to_idx[date]
    xb  = X_eur[idx:idx+1].to(device)
    z0  = model.encoder(xb)

    n_steps = max(12, int(round(expiry / DT)))
    dt_eff  = expiry / n_steps

    half = N_PATHS // 2
    z1, r1, _, _ = simulate_latent_paths(model, z0, n_paths=half,
                                          n_steps=n_steps, dt=dt_eff,
                                          device=device, diffusion_scale=diff_scale)
    z2, r2, _, _ = simulate_latent_paths(model, z0, n_paths=half,
                                          n_steps=n_steps, dt=dt_eff,
                                          device=device, diffusion_scale=diff_scale)

    z_T  = torch.cat([z1[:, -1, :], z2[:, -1, :]], dim=0)
    r_all = torch.cat([r1, r2], dim=0)
    D_T  = compute_discount_paths(r_all, dt_eff)[:, -1]

    finite = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    n_fin  = int(finite.sum())
    if n_fin < 32:
        return None

    z_keep = z_T[finite]
    D_keep = D_T[finite]

    _, aux_T = model.decode_from_z(z_keep, tau=None, return_aux=True)
    P_T = aux_T["P_full"]
    dec_ok = torch.isfinite(P_T).all(1)
    if int(dec_ok.sum()) < 32:
        return None

    F_T, A_T = swap_rate_torch(P_T[dec_ok], tenor=tenor)
    D_use     = D_keep[dec_ok]

    fa = torch.isfinite(F_T) & torch.isfinite(A_T)
    if int(fa.sum()) < 32:
        return None
    F_T, A_T, D_use = F_T[fa], A_T[fa], D_use[fa]

    _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
    F0_t, A0_t = swap_rate_torch(aux0["P_full"], tenor=tenor)
    F0, A0 = float(F0_t[0]), float(A0_t[0])
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 0):
        return None

    V_MC = (D_use * A_T * torch.relu(F_T - F0)).mean()
    if not torch.isfinite(V_MC):
        return None

    sigma_mod_bp = float(V_MC) * sqrt_2pi / (A0 * math.sqrt(expiry)) * 10_000.0
    path_frac    = n_fin / N_PATHS
    return sigma_mod_bp, path_frac

# ================================================================
# Run pricing for all models
# ================================================================

all_records = []

for cfg in MODELS:
    print(f"\n=== {cfg['label']} ===")
    model = load_model(cfg)
    if model is None:
        print(f"  Skipping {cfg['label']} (no checkpoint).")
        continue

    n_ok = 0
    for _, row in combos.iterrows():
        date    = pd.Timestamp(row["as_of_date"]).normalize()
        expiry  = int(row["option_maturity"])
        tenor   = int(row["swap_tenor"])
        sig_mkt = float(row["market_vol"]) * 10_000.0

        result = price_swaption(model, date, expiry, tenor, cfg["diff_scale"])
        if result is None:
            sig_mod, pfrac = float("nan"), 0.0
        else:
            sig_mod, pfrac = result
            if not math.isfinite(sig_mod) or sig_mod <= 0:
                sig_mod, pfrac = float("nan"), pfrac
            else:
                n_ok += 1

        all_records.append({
            "model":      cfg["key"],
            "label":      cfg["label"],
            "date":       date.date(),
            "expiry":     expiry,
            "tenor":      tenor,
            "sigma_mkt":  round(sig_mkt,  1),
            "sigma_mod":  round(sig_mod,  1) if math.isfinite(sig_mod) else None,
            "err_bp":     round(sig_mod - sig_mkt, 1) if math.isfinite(sig_mod) else None,
            "path_frac":  round(pfrac,   3),
        })

    pct_ok = 100 * n_ok / max(len(combos), 1)
    print(f"  Priced {n_ok}/{len(combos)} ({pct_ok:.0f}%)")

df_all = pd.DataFrame(all_records)
df_all.to_csv(os.path.join(OUT_DIR, "vol_comparison.csv"), index=False)
print(f"\nSaved raw results: {OUT_DIR}/vol_comparison.csv")

# ================================================================
# Summary statistics
# ================================================================

def model_stats(df, model_key):
    sub = df[(df["model"] == model_key) & df["err_bp"].notna()]
    if sub.empty:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "mean_path_frac": float("nan")}
    return {
        "n":              len(sub),
        "mae":            round(sub["err_bp"].abs().mean(), 1),
        "rmse":           round((sub["err_bp"]**2).mean()**0.5, 1),
        "mean_path_frac": round(sub["path_frac"].mean(), 3),
    }

print("\n=== Summary ===")
print(f"{'Model':<22}  {'N':>5}  {'MAE bp':>8}  {'RMSE bp':>9}  {'Path%':>7}")
print("-" * 58)
stats = {}
for cfg in MODELS:
    s = model_stats(df_all, cfg["key"])
    stats[cfg["key"]] = s
    print(f"{cfg['label']:<22}  {s['n']:>5}  {s['mae']:>8.1f}  "
          f"{s['rmse']:>9.1f}  {s['mean_path_frac']*100:>6.0f}%")

with open(os.path.join(OUT_DIR, "calibration_summary.json"), "w") as f:
    json.dump(stats, f, indent=2)

# ================================================================
# LaTeX summary table: MAE per (expiry, tenor), one column per model
# ================================================================

def pivot_mae(df, model_key):
    sub = df[(df["model"] == model_key) & df["err_bp"].notna()].copy()
    return sub.groupby(["expiry", "tenor"])["err_bp"].apply(
        lambda x: round(x.abs().mean(), 0)
    ).reset_index().rename(columns={"err_bp": f"mae_{model_key}"})

pivots = [pivot_mae(df_all, cfg["key"]) for cfg in MODELS]
from functools import reduce
df_tex = reduce(lambda a, b: pd.merge(a, b, on=["expiry", "tenor"], how="outer"), pivots)
df_tex = df_tex.sort_values(["expiry", "tenor"])

lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(r"\caption{ATM swaption vol MAE (bp) by model. Entries show the mean absolute "
             r"error $|\hat\sigma_{N,\mathrm{mod}} - \sigma_{\mathrm{mkt}}|$ averaged over "
             r"all available quote dates for each (expiry, tenor) pair.}")
lines.append(r"\label{tab:vol_comparison_mae}")
lines.append(r"\begin{tabular}{@{}rr" + "r" * len(MODELS) + r"@{}}")
lines.append(r"\toprule")
header_cols = " & ".join(
    r"\textbf{" + cfg["label"].replace("_", r"\_") + r"}"
    for cfg in MODELS
)
lines.append(r"\textbf{Exp} & \textbf{Ten} & " + header_cols + r" \\")
lines.append(r"\midrule")

prev_exp = None
for _, row in df_tex.iterrows():
    exp = int(row["expiry"]); ten = int(row["tenor"])
    if prev_exp is not None and exp != prev_exp:
        lines.append(r"\addlinespace[2pt]")
    prev_exp = exp
    vals = []
    for cfg in MODELS:
        v = row.get(f"mae_{cfg['key']}", float("nan"))
        vals.append("--" if (v is None or (isinstance(v, float) and math.isnan(v)))
                    else f"{int(v)}")
    lines.append(f"{exp} & {ten} & " + " & ".join(vals) + r" \\")

lines.append(r"\midrule")
summary_vals = []
for cfg in MODELS:
    s = stats.get(cfg["key"], {})
    mae = s.get("mae", float("nan"))
    summary_vals.append("--" if math.isnan(mae) else f"{mae:.1f}")
lines.append(r"\textbf{Overall MAE} & & " + " & ".join(summary_vals) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex_path = os.path.join(OUT_DIR, "vol_summary_table.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved LaTeX table: {tex_path}")

# ================================================================
# Figures
# ================================================================

expiries = sorted(df_all["expiry"].dropna().unique().astype(int))
tenors   = sorted(df_all["tenor"].dropna().unique().astype(int))

# --- vol-error heatmaps ---
for cfg in MODELS:
    sub = df_all[(df_all["model"] == cfg["key"]) & df_all["err_bp"].notna()]
    if sub.empty:
        continue
    pivot = sub.groupby(["expiry", "tenor"])["err_bp"].mean().unstack("tenor")
    fig, ax = plt.subplots(figsize=(max(4, len(tenors)), max(3, len(expiries))), dpi=150)
    vmax = max(50, float(pivot.abs().max().max()))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Tenor (Y)"); ax.set_ylabel("Expiry (Y)")
    ax.set_title(f"Vol error (mod − mkt, bp) — {cfg['label']}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, label="Error (bp)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig_vol_heatmap_{cfg['key']}.png"), dpi=200)
    plt.close(fig)

# --- scatter: mod vs mkt for calibrated model ---
for cfg in MODELS:
    sub = df_all[(df_all["model"] == cfg["key"]) & df_all["err_bp"].notna()]
    if sub.empty:
        continue
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.scatter(sub["sigma_mkt"], sub["sigma_mod"], alpha=0.4, s=12,
               color=cfg["color"], label=cfg["label"])
    lim = max(sub["sigma_mkt"].max(), sub["sigma_mod"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="perfect fit")
    ax.set_xlabel("Market vol (bp)"); ax.set_ylabel("Model vol (bp)")
    ax.set_title(f"Model vs market vol — {cfg['label']}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig_vol_scatter_{cfg['key']}.png"), dpi=200)
    plt.close(fig)

# --- path survival bar chart ---
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
for i, cfg in enumerate(MODELS):
    sub = df_all[df_all["model"] == cfg["key"]]
    mean_frac = sub["path_frac"].mean()
    ax.bar(i, mean_frac * 100, color=cfg["color"], label=cfg["label"], alpha=0.8)
    ax.text(i, mean_frac * 100 + 1, f"{mean_frac*100:.0f}%",
            ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([c["label"] for c in MODELS], rotation=15, ha="right")
ax.set_ylabel("Mean path-survival fraction (%)")
ax.set_title("Fraction of simulated paths producing finite decoded curves")
ax.set_ylim(0, 110); ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_path_survival.png"), dpi=200)
plt.close(fig)

# --- all-models scatter overlay ---
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
all_mkt = []
for cfg in MODELS:
    sub = df_all[(df_all["model"] == cfg["key"]) & df_all["err_bp"].notna()]
    if sub.empty:
        continue
    ax.scatter(sub["sigma_mkt"], sub["sigma_mod"], alpha=0.35, s=10,
               color=cfg["color"], label=cfg["label"])
    all_mkt.extend(sub["sigma_mkt"].tolist())
    all_mkt.extend(sub["sigma_mod"].dropna().tolist())
if all_mkt:
    lim = max(all_mkt) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("Market vol (bp)"); ax.set_ylabel("Model vol (bp)")
ax.set_title("All models: model vs market vol")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_vol_scatter_all.png"), dpi=200)
plt.close(fig)

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
