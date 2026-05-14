# =============================================================================
# ResultsGenerator_augmented.py
#
# Generates in-sample diagnostic figures for the augmented-input experiment.
# Loads the checkpoint produced by Training_augmented_input.py and produces:
#   1. Scatter of per-curve RMSE (bps) over time, coloured by regime
#   2. Combined regime table (N + Avg RMSE) saved as CSV for LaTeX
#
# Run from repo root:
#   python Code/Experiments/ResultsGenerator_augmented.py
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ── environment setup ─────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    REPO_ROOT = os.getcwd()  # PyCharm console: CWD is the project root

for _p in [REPO_ROOT, os.path.dirname(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Code.load_swapdata import my_data, custom_palette
from Code.model.full_model import FullModel

# ── settings ──────────────────────────────────────────────────────────────────
LATENT_DIM = 4
EPOCHS     = 5000
USE        = "bbg"

CKPT_DIR  = os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                         f"dim{LATENT_DIM}_augmented_input", f"ep{EPOCHS}")
CKPT_PATH = os.path.join(CKPT_DIR, f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt")

FIGURES_OUT = os.path.join(REPO_ROOT, "Figures", "thesis_results", "AutoencoderPerformanceAugmented")
os.makedirs(FIGURES_OUT, exist_ok=True)

CCY_ORDER = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ── augmentation (must match Training_augmented_input.py exactly) ─────────────
def augment(x: torch.Tensor) -> torch.Tensor:
    f1 = x[:, 4] - x[:, 0]
    f2 = x[:, 7] - x[:, 4]
    f3 = 2.0 * x[:, 4] - x[:, 0] - x[:, 7]
    return torch.cat([x, f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
meta, X_tensor, _, _, tenors, _, _, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

# ── load model ────────────────────────────────────────────────────────────────
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_PATH}\n"
        f"Run Training_augmented_input.py first."
    )

print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)

model_config = ckpt["model_config"]
latent_dim   = model_config["latent_dim"]
input_dim    = model_config["input_dim"]

model = FullModel(input_dim=input_dim, latent_dim=latent_dim).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded — latent_dim={latent_dim}, input_dim={input_dim}")

# ── inference ─────────────────────────────────────────────────────────────────
print("Running inference...")
BATCH_SIZE = 256
S_hat_list = []

with torch.no_grad():
    for i in range(0, X_tensor.shape[0], BATCH_SIZE):
        xb     = X_tensor[i:i + BATCH_SIZE].to(device)
        xb_aug = augment(xb)
        S_hat  = model(xb_aug)
        S_hat_list.append(S_hat.detach().cpu())

S_hat_all = torch.cat(S_hat_list, dim=0)   # (N, 8)

# ── per-curve RMSE (bps) ──────────────────────────────────────────────────────
X_np     = X_tensor.numpy()
S_np     = S_hat_all.numpy()
rmse_bps = np.sqrt(np.mean((X_np - S_np) ** 2, axis=1)) * 10_000

# ── regime flags ─────────────────────────────────────────────────────────────
# inverted: short rate (1Y, index 0) > long rate (30Y, index 7)
# negative: any tenor < 0
inverted = X_np[:, 0] > X_np[:, 7]
negative = (X_np < 0).any(axis=1)

df_regime = pd.DataFrame({
    "ccy":        meta["ccy"].values,
    "as_of_date": pd.to_datetime(meta["as_of_date"].values),
    "rmse_bps":   rmse_bps,
    "inverted":   inverted,
    "negative":   negative,
})

# ── helpers ───────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    p = os.path.join(FIGURES_OUT, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

def save_table(df, name):
    p = os.path.join(FIGURES_OUT, f"{name}.csv")
    df.to_csv(p)
    print(f"  Saved: {p}")

C_PALETTE = custom_palette

# ── figure: fitted vs actual (representative dates) ──────────────────────────
print("\nGenerating fitted vs actual figure...")

REPRESENTATIVE_DATES = {
    "Normal market (2016-08-31)": "2016-08-31",
    "Crisis (2020-03-31)":        "2020-03-31",
    "Low-rate (2019-06-30)":      "2019-06-30",
}
SHOW_CCYS = ["EUR", "USD"]
scale = 100.0 if SCALE_IS_PERCENT else 1.0

X_np_all = X_tensor.numpy()
S_np_all = S_hat_all.numpy()
dates_all = pd.to_datetime(meta["as_of_date"].values)
ccys_all  = meta["ccy"].values

fig, axes = plt.subplots(len(SHOW_CCYS), len(REPRESENTATIVE_DATES),
                         figsize=(5 * len(REPRESENTATIVE_DATES), 3.8 * len(SHOW_CCYS)),
                         sharey=False)

for col_i, (label, date_str) in enumerate(REPRESENTATIVE_DATES.items()):
    target_date = pd.Timestamp(date_str)
    for row_i, ccy in enumerate(SHOW_CCYS):
        ax = axes[row_i][col_i]
        mask_ccy = ccys_all == ccy
        if mask_ccy.sum() == 0:
            ax.set_visible(False)
            continue
        dates_ccy  = dates_all[mask_ccy]
        idx_local  = np.argmin(np.abs(dates_ccy - target_date))
        actual_date = dates_ccy[idx_local]
        global_idx  = np.where(mask_ccy)[0][idx_local]

        actual = X_np_all[global_idx] * scale
        fitted = S_np_all[global_idx] * scale
        fitted_color = custom_palette[CCY_ORDER.index(ccy) % len(custom_palette)]

        ax.plot(tenors, actual, "o-",  color="black",        linewidth=2.0, markersize=5)
        ax.plot(tenors, fitted, "s--", color=fitted_color,   linewidth=2.0, markersize=5)

        if row_i == 0:
            ax.set_title(label, fontsize=10, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel(f"{ccy}, {'Rate (%)' if SCALE_IS_PERCENT else 'Rate (dec.)'}",
                          fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.text(0.97, 0.05, pd.Timestamp(actual_date).strftime("%Y-%m-%d"),
                transform=ax.transAxes, fontsize=7, ha="right", color="0.4")

fig.tight_layout()
save_fig(fig, f"augmented_fitted_vs_actual_dim{LATENT_DIM}")

# ── figure: fitted vs actual — normal vs crisis ───────────────────────────────
print("\nGenerating normal vs crisis fitted vs actual figure...")

_rep_dates   = {
    "Normal (2014-08-29)": "2014-08-29",
    "Crisis (2020-03-31)": "2020-03-31",
}
_show_ccys   = ["EUR", "USD", "JPY", "CAD"]
_n_rows      = len(_rep_dates)
_n_cols      = len(_show_ccys)
_scale       = 100.0 if SCALE_IS_PERCENT else 1.0

fig, axes = plt.subplots(_n_rows, _n_cols,
                         figsize=(4 * _n_cols, 3.5 * _n_rows),
                         sharey=False)

for row_i, (label, date_str) in enumerate(_rep_dates.items()):
    target_date = pd.Timestamp(date_str)
    for col_i, ccy in enumerate(_show_ccys):
        ax = axes[row_i][col_i]
        mask_ccy = ccys_all == ccy
        if mask_ccy.sum() == 0:
            ax.set_visible(False)
            continue
        dates_ccy   = dates_all[mask_ccy]
        idx_local   = np.argmin(np.abs(dates_ccy - target_date))
        actual_date = dates_ccy[idx_local]
        global_idx  = np.where(mask_ccy)[0][idx_local]

        actual = X_np_all[global_idx] * _scale
        fitted = S_np_all[global_idx] * _scale
        fitted_color = custom_palette[CCY_ORDER.index(ccy) % len(custom_palette)]

        ax.plot(tenors, actual, "o-",  color="black",      linewidth=2.0,
                markersize=5, label="Actual", zorder=5)
        ax.plot(tenors, fitted, "s--", color=fitted_color, linewidth=1.8,
                label=f"Augmented ($\\ell={LATENT_DIM}$)")

        if row_i == 0:
            ax.set_title(ccy, fontsize=10, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel(f"{label}\n({'%' if SCALE_IS_PERCENT else 'dec.'})",
                          fontsize=9)
        if row_i == _n_rows - 1:
            ax.set_xlabel("Maturity", fontsize=9)
        ax.set_xticks(tenors)
        if row_i == _n_rows - 1:
            ax.set_xticklabels([str(int(t)) for t in tenors], fontsize=7)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=8)
        ax.text(0.97, 0.05, pd.Timestamp(actual_date).strftime("%Y-%m-%d"),
                transform=ax.transAxes, fontsize=7, ha="right", color="0.4")

fig.tight_layout()
save_fig(fig, f"augmented_fitted_vs_actual_normal_crisis_dim{LATENT_DIM}")

# ── figure: scatter RMSE over time by regime ─────────────────────────────────
print("\nGenerating scatter figure...")

scatter_groups = [
    (~df_regime["inverted"] & ~df_regime["negative"], "Normal Non-negative",   C_PALETTE[2]),
    ( df_regime["inverted"] & ~df_regime["negative"], "Inverted Non-negative", "black"),
    (~df_regime["inverted"] &  df_regime["negative"], "Normal Negative",       "indianred"),
    ( df_regime["inverted"] &  df_regime["negative"], "Inverted Negative",     C_PALETTE[8]),
]

fig, ax = plt.subplots(figsize=(11, 4))
for mask, lbl, col in scatter_groups:
    sub = df_regime[mask]
    if len(sub) == 0:
        continue
    ax.scatter(sub["as_of_date"], sub["rmse_bps"],
               s=4, alpha=0.4, color=col, marker="o", label=lbl, zorder=3)

ax.set_ylabel("RMSE (bps)")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
          ncol=1, fontsize=7, frameon=False, markerscale=3)
fig.autofmt_xdate()
fig.tight_layout()
save_fig(fig, f"augmented_is_scatter_regime_dim{LATENT_DIM}")

# ── table: combined regime (N + Avg RMSE per currency) ───────────────────────
print("\nGenerating regime table...")

groups = [
    ("Normal Non-negative",   ~df_regime["inverted"] & ~df_regime["negative"]),
    ("Inverted Non-negative",  df_regime["inverted"] & ~df_regime["negative"]),
    ("Normal Negative",       ~df_regime["inverted"] &  df_regime["negative"]),
    ("Inverted Negative",      df_regime["inverted"] &  df_regime["negative"]),
]

rows = {}
for lbl, mask in groups:
    for stat, fn in [("N",              lambda x: len(x)),
                     ("Avg RMSE (bps)", lambda x: round(x.mean(), 2)),
                     ("Std RMSE (bps)", lambda x: round(x.std(),  2))]:
        row = {}
        for ccy in CCY_ORDER:
            sub = df_regime.loc[mask & (df_regime["ccy"] == ccy), "rmse_bps"]
            row[ccy] = fn(sub) if len(sub) > 0 else np.nan
        sub_all = df_regime.loc[mask, "rmse_bps"]
        row["All"] = fn(sub_all) if len(sub_all) > 0 else np.nan
        rows[f"{lbl} — {stat}"] = row

tbl = pd.DataFrame(rows).T
save_table(tbl, f"augmented_is_rmse_combined_dim{LATENT_DIM}")
print(tbl.to_string())

# display version: N + Avg only, drop Std and empty Inverted Negative
display_rows = [r for r in tbl.index
                if "Std" not in r and "Inverted Negative" not in r]
disp = tbl.loc[display_rows].copy().astype(object)
for idx in disp.index:
    if idx.endswith("— N"):
        disp.loc[idx] = disp.loc[idx].apply(
            lambda v: str(int(v)) if pd.notna(v) else "---")
    else:
        disp.loc[idx] = disp.loc[idx].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "---")
save_table(disp, f"augmented_is_rmse_combined_display_dim{LATENT_DIM}")

# ── figure: fitted vs actual — all dims (ℓ=2,3,4) overlaid ──────────────────
print("\nGenerating fitted vs actual — all dims figure...")

_dims_aug    = [2, 3, 4]
_dim_colors  = {2: custom_palette[4], 3: custom_palette[0], 4: custom_palette[6]}
_dim_labels  = {d: r"$\ell$=" + str(d) for d in _dims_aug}

# load models and run inference for each dim
_aug_S_hat = {}
for _dim in _dims_aug:
    _ckpt_path = os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                              f"dim{_dim}_augmented_input", f"ep{EPOCHS}",
                              f"checkpoint_dim{_dim}_ep{EPOCHS}.pt")
    if not os.path.exists(_ckpt_path):
        print(f"  ⚠️  Checkpoint not found for dim={_dim}: {_ckpt_path} — skipping.")
        continue
    _ckpt = torch.load(_ckpt_path, map_location=device)
    _cfg  = _ckpt["model_config"]
    _m    = FullModel(input_dim=_cfg["input_dim"], latent_dim=_cfg["latent_dim"]).to(device)
    _m.load_state_dict(_ckpt["model_state_dict"])
    _m.eval()
    print(f"  Loaded augmented dim={_dim}")

    _s_list = []
    with torch.no_grad():
        for _i in range(0, X_tensor.shape[0], BATCH_SIZE):
            _xb = X_tensor[_i:_i + BATCH_SIZE].to(device)
            _s_list.append(_m(augment(_xb)).cpu())
    _aug_S_hat[_dim] = torch.cat(_s_list).numpy()

_rep_dates_ad = {
    "Normal (2014-08-29)": "2014-08-29",
    "Crisis (2020-03-31)": "2020-03-31",
}
_show_ccys_ad = ["EUR", "USD", "JPY", "CAD"]
_n_rows_ad    = len(_rep_dates_ad)
_n_cols_ad    = len(_show_ccys_ad)
_scale_ad     = 100.0 if SCALE_IS_PERCENT else 1.0

fig_ad, axes_ad = plt.subplots(
    _n_rows_ad, _n_cols_ad,
    figsize=(4 * _n_cols_ad, 3.5 * _n_rows_ad),
    sharey=False,
)

for _row_i, (_label, _date_str) in enumerate(_rep_dates_ad.items()):
    _target_date = pd.Timestamp(_date_str)
    for _col_i, _ccy in enumerate(_show_ccys_ad):
        ax = axes_ad[_row_i][_col_i]
        _mask_ccy   = ccys_all == _ccy
        if _mask_ccy.sum() == 0:
            ax.set_visible(False)
            continue
        _dates_ccy  = dates_all[_mask_ccy]
        _idx_local  = np.argmin(np.abs(_dates_ccy - _target_date))
        _actual_date = _dates_ccy[_idx_local]
        _global_idx  = np.where(_mask_ccy)[0][_idx_local]

        _actual = X_np_all[_global_idx] * _scale_ad
        ax.plot(tenors, _actual, "o-", color="black",
                linewidth=2.0, markersize=5, label="Actual", zorder=5)

        for _dim in _dims_aug:
            if _dim not in _aug_S_hat:
                continue
            _fitted = _aug_S_hat[_dim][_global_idx] * _scale_ad
            ax.plot(tenors, _fitted,
                    color=_dim_colors[_dim],
                    linewidth=1.8,
                    label=_dim_labels[_dim])

        if _row_i == 0:
            ax.set_title(_ccy, fontsize=12, fontweight="bold")
        if _col_i == 0:
            ax.set_ylabel(f"{_label}\n({'%' if SCALE_IS_PERCENT else 'dec.'})",
                          fontsize=11)
        if _row_i == _n_rows_ad - 1:
            ax.set_xlabel("Maturity", fontsize=11)
        ax.set_xticks(tenors)
        if _row_i == _n_rows_ad - 1:
            ax.set_xticklabels([str(int(t)) for t in tenors], fontsize=9)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=10)
        ax.text(0.97, 0.05, pd.Timestamp(_actual_date).strftime("%Y-%m-%d"),
                transform=ax.transAxes, fontsize=9, ha="right", color="0.4")

_h_ad, _l_ad = axes_ad[0][0].get_legend_handles_labels()
fig_ad.legend(_h_ad, _l_ad, loc="lower center",
              bbox_to_anchor=(0.5, -0.02),
              ncol=len(_dims_aug) + 1, frameon=False, fontsize=10)
fig_ad.tight_layout()
fig_ad.subplots_adjust(bottom=0.12)
save_fig(fig_ad, "augmented_fitted_vs_actual_all_dims")

print("\nResultsGenerator_augmented complete.")
