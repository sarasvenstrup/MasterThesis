# =============================================================================
# ResultsGenerator_augmented_stable.py
#
# Generates in-sample diagnostic figures for the augmented-input experiment.
# Loads the checkpoint produced by Training_augmented_stable.py and produces:
#   1. Scatter of per-curve RMSE (bps) over time, coloured by regime
#   2. Combined regime table (N + Avg RMSE) saved as CSV for LaTeX
#
# Run from repo root:
#   python Code/Experiments/ResultsGenerator_augmented_stable.py
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
from Code.model.full_model_stable import FullModel

# ── settings ──────────────────────────────────────────────────────────────────
LATENT_DIM = 4
EPOCHS     = 5000
USE        = "bbg"

CKPT_DIR  = os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                         f"dim{LATENT_DIM}_augmented_stable", f"ep{EPOCHS}")

CHECKPOINTS_DIR = os.path.join(os.path.dirname(REPO_ROOT), "checkpoints")

def _resolve_ckpt(variant_key, dim, epochs):
    """Return checkpoint path, checking Figures/TrainingResults first, then checkpoints/."""
    figures_path = os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                                f"dim{dim}_{variant_key}", f"ep{epochs}",
                                f"checkpoint_dim{dim}_ep{epochs}.pt")
    if os.path.exists(figures_path):
        return figures_path
    fallback = os.path.join(CHECKPOINTS_DIR,
                            f"fullmodel_{variant_key}_bbg_dim{dim}_ep{epochs}.pt")
    if os.path.exists(fallback):
        return fallback
    return figures_path  # return primary path so error message is informative

CKPT_PATH = _resolve_ckpt("augmented_stable", LATENT_DIM, EPOCHS)

FIGURES_OUT = os.path.join(REPO_ROOT, "Figures", "thesis_results", "AutoencoderPerformanceAugmentedStable")
os.makedirs(FIGURES_OUT, exist_ok=True)

CCY_ORDER = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ── augmentation (must match Training_augmented_stable.py exactly) ─────────────
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
        f"Run Training_augmented_stable.py first."
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
    "Calm market (2016-08-31)": "2016-08-31",
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
save_fig(fig, f"augmented_stable_fitted_vs_actual_dim{LATENT_DIM}")

# ── figure: fitted vs actual — normal vs crisis ───────────────────────────────
print("\nGenerating normal vs crisis fitted vs actual figure...")

_rep_dates   = {
    "Calm (2014-08-29)": "2014-08-29",
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
save_fig(fig, f"augmented_stable_fitted_vs_actual_normal_crisis_dim{LATENT_DIM}")

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
save_fig(fig, f"augmented_stable_is_scatter_regime_dim{LATENT_DIM}")

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
save_table(tbl, f"augmented_stable_is_rmse_combined_dim{LATENT_DIM}")
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
save_table(disp, f"augmented_stable_is_rmse_combined_display_dim{LATENT_DIM}")

# ── figure: fitted vs actual — all dims (ℓ=3,4) overlaid ────────────────────
print("\nGenerating fitted vs actual — all dims (augmented stable) figure...")

_dims_as     = [3, 4]
_dim_colors_as = {3: custom_palette[0], 4: custom_palette[6]}
_dim_labels_as = {d: r"$\ell$=" + str(d) for d in _dims_as}

dates_all = pd.to_datetime(meta["as_of_date"].values)
ccys_all  = meta["ccy"].values
X_np_all  = X_tensor.numpy()

_as_S_hat = {}
for _dim in _dims_as:
    _ckpt_path = _resolve_ckpt("augmented_stable", _dim, EPOCHS)
    if not os.path.exists(_ckpt_path):
        print(f"  ⚠️  Checkpoint not found for dim={_dim} — skipping.")
        continue
    _ckpt = torch.load(_ckpt_path, map_location=device)
    _cfg  = _ckpt["model_config"]
    _m    = FullModel(input_dim=_cfg["input_dim"], latent_dim=_cfg["latent_dim"]).to(device)
    _m.load_state_dict(_ckpt["model_state_dict"])
    _m.eval()
    print(f"  Loaded augmented stable dim={_dim}")

    _s_list = []
    with torch.no_grad():
        for _i in range(0, X_tensor.shape[0], BATCH_SIZE):
            _xb = X_tensor[_i:_i + BATCH_SIZE].to(device)
            _s_list.append(_m(augment(_xb)).cpu())
    _as_S_hat[_dim] = torch.cat(_s_list).numpy()

_rep_dates_as = {
    "Calm (2014-08-29)": "2014-08-29",
    "Crisis (2020-03-31)": "2020-03-31",
}
_show_ccys_as = ["EUR", "USD", "JPY", "CAD"]
_n_rows_as    = len(_rep_dates_as)
_n_cols_as    = len(_show_ccys_as)
_scale_as     = 100.0 if SCALE_IS_PERCENT else 1.0

fig_as, axes_as = plt.subplots(
    _n_rows_as, _n_cols_as,
    figsize=(4 * _n_cols_as, 3.5 * _n_rows_as),
    sharey=False,
)

for _row_i, (_label, _date_str) in enumerate(_rep_dates_as.items()):
    _target_date = pd.Timestamp(_date_str)
    for _col_i, _ccy in enumerate(_show_ccys_as):
        ax = axes_as[_row_i][_col_i]
        _mask_ccy    = ccys_all == _ccy
        if _mask_ccy.sum() == 0:
            ax.set_visible(False)
            continue
        _dates_ccy   = dates_all[_mask_ccy]
        _idx_local   = np.argmin(np.abs(_dates_ccy - _target_date))
        _actual_date = _dates_ccy[_idx_local]
        _global_idx  = np.where(_mask_ccy)[0][_idx_local]

        _actual = X_np_all[_global_idx] * _scale_as
        ax.plot(tenors, _actual, "o-", color="black",
                linewidth=2.0, markersize=5, label="Actual", zorder=5)

        for _dim in _dims_as:
            if _dim not in _as_S_hat:
                continue
            _fitted = _as_S_hat[_dim][_global_idx] * _scale_as
            ax.plot(tenors, _fitted,
                    color=_dim_colors_as[_dim],
                    linewidth=1.8,
                    label=_dim_labels_as[_dim])

        if _row_i == 0:
            ax.set_title(_ccy, fontsize=12, fontweight="bold")
        if _col_i == 0:
            ax.set_ylabel(f"{_label}\n({'%' if SCALE_IS_PERCENT else 'dec.'})",
                          fontsize=11)
        if _row_i == _n_rows_as - 1:
            ax.set_xlabel("Maturity", fontsize=11)
        ax.set_xticks(tenors)
        if _row_i == _n_rows_as - 1:
            ax.set_xticklabels([str(int(t)) for t in tenors], fontsize=9)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=10)
        ax.text(0.97, 0.05, pd.Timestamp(_actual_date).strftime("%Y-%m-%d"),
                transform=ax.transAxes, fontsize=9, ha="right", color="0.4")

_h_as, _l_as = axes_as[0][0].get_legend_handles_labels()
fig_as.legend(_h_as, _l_as, loc="lower center",
              bbox_to_anchor=(0.5, -0.02),
              ncol=len(_dims_as) + 1, frameon=False, fontsize=10)
fig_as.tight_layout()
fig_as.subplots_adjust(bottom=0.12)
save_fig(fig_as, "augmented_stable_fitted_vs_actual_all_dims")

# ── shared setup for all-models comparison figures ────────────────────────────
from Code.model.full_model import FullModel as _BaselineFullModel
import matplotlib.dates as _mdates
import matplotlib.lines as _mlines

_INPUT_DIM_BASE = X_tensor.shape[1]               # 8  (no augmentation)
_INPUT_DIM_AUG  = augment(X_tensor[:1]).shape[1]  # 11 (with augmentation)
_COMP_COLORS    = ["#2c4f8c", "#c0392b", "cornflowerblue", "palevioletred"]
_REP_DATES_COMP = {"Calm (2014-08-29)": "2014-08-29",
                   "Crisis (2020-03-31)":  "2020-03-31"}
_SHOW_CCYS_COMP = ["EUR", "USD", "JPY", "CAD"]
_SCALE_COMP     = 100.0 if SCALE_IS_PERCENT else 1.0

_ROLL_SUBDIR    = "train5Y_test6M_step6M"
_ROLL_EPOCHS    = 3500

# each tuple: (variant_key, label, ModelClass, use_augmentation, latent_dim)
_COMP_VARIANTS_MIXED = [
    ("baseline",         "Baseline ($\\ell=3$)",    _BaselineFullModel, False, 3),
    ("stable",           "Stable ($\\ell=4$)",      FullModel,          False, 4),
    ("augmented_input",  "Augmented ($\\ell=3$)",   _BaselineFullModel, True,  3),
    ("augmented_stable", "Aug. + Stable ($\\ell=3$)", FullModel,          True,  3),
]

_REGIME_GROUPS_OOS = [
    ("Normal, Non-negative",   False, False, custom_palette[2]),
    ("Inverted, Non-negative", True,  False, "black"),
    ("Normal, Negative",       False, True,  "indianred"),
    ("Inverted, Negative",     True,  True,  custom_palette[8]),
]

def _load_oos_preds(variant_key, dim):
    _path = os.path.join(
        REPO_ROOT, "Figures", "OOSResults", "Roll",
        f"OOS_roll_dim{dim}_{variant_key}",
        _ROLL_SUBDIR, f"ep{_ROLL_EPOCHS}",
        "predictions_test_all.csv",
    )
    if not os.path.exists(_path):
        print(f"  ⚠️  OOS predictions not found: {_path} — skipping.")
        return None
    _df = pd.read_csv(_path)
    _df["as_of_date"] = pd.to_datetime(_df["as_of_date"])
    _actual_cols = sorted([c for c in _df.columns if c.startswith("actual_")])
    _fitted_cols = sorted([c for c in _df.columns if c.startswith("fitted_")])
    _actual_v    = _df[_actual_cols].values
    _fitted_v    = _df[_fitted_cols].values
    _df["rmse_bps"] = np.sqrt(np.mean((_actual_v - _fitted_v) ** 2, axis=1)) * 1e4
    _df["inv_flag"] = _actual_v[:, 0] > _actual_v[:, -1]
    _df["neg_flag"] = (_actual_v < 0).any(axis=1)
    return _df

# ── figure: all-models fitted vs actual (mixed dims) ─────────────────────────
print("\nGenerating all-models comparison figure (mixed dims)...")

_comp_S_hat   = {}
_comp_r_tilde = {}
for (_vkey_c, _lbl_c, _ModelClass_c, _use_aug_c, _dim_c), _col_c in zip(
        _COMP_VARIANTS_MIXED, _COMP_COLORS):
    _ckpt_c = _resolve_ckpt(_vkey_c, _dim_c, EPOCHS)
    if not os.path.exists(_ckpt_c):
        print(f"  ⚠️  Checkpoint not found: {_ckpt_c} — skipping.")
        continue
    _raw_c = torch.load(_ckpt_c, map_location=device, weights_only=False)
    if isinstance(_raw_c, dict) and "model_config" in _raw_c:
        _in_dim_c  = _raw_c["model_config"]["input_dim"]
        _lat_dim_c = _raw_c["model_config"]["latent_dim"]
        _sd_c      = _raw_c["model_state_dict"]
    elif isinstance(_raw_c, dict) and "model_state_dict" in _raw_c:
        _in_dim_c  = _INPUT_DIM_AUG if _use_aug_c else _INPUT_DIM_BASE
        _lat_dim_c = _dim_c
        _sd_c      = _raw_c["model_state_dict"]
    else:
        _in_dim_c  = _INPUT_DIM_AUG if _use_aug_c else _INPUT_DIM_BASE
        _lat_dim_c = _dim_c
        _sd_c      = _raw_c
    _m_c = _ModelClass_c(input_dim=_in_dim_c, latent_dim=_lat_dim_c).to(device)
    _m_c.load_state_dict(_sd_c, strict=False)
    _m_c.eval()
    print(f"  Loaded {_lbl_c}")
    _s_list_c, _r_list_c = [], []
    with torch.no_grad():
        for _i_c in range(0, X_tensor.shape[0], BATCH_SIZE):
            _xb_c  = X_tensor[_i_c:_i_c + BATCH_SIZE].to(device)
            _inp_c = augment(_xb_c) if _use_aug_c else _xb_c
            _out_c, _aux_c = _m_c(_inp_c, return_aux=True)
            _s_list_c.append(_out_c.cpu())
            _r_list_c.append(_aux_c["r_tilde"].cpu())
    _comp_S_hat[_lbl_c]   = torch.cat(_s_list_c).numpy()
    _comp_r_tilde[_lbl_c] = torch.cat(_r_list_c).numpy()

_n_rows_comp = len(_REP_DATES_COMP)
_n_cols_comp = len(_SHOW_CCYS_COMP)

fig_comp, axes_comp = plt.subplots(
    _n_rows_comp, _n_cols_comp,
    figsize=(4 * _n_cols_comp, 3.5 * _n_rows_comp),
    sharey=False,
)

for _row_c, (_label_c, _date_str_c) in enumerate(_REP_DATES_COMP.items()):
    _target_c = pd.Timestamp(_date_str_c)
    for _col_c_i, _ccy_c in enumerate(_SHOW_CCYS_COMP):
        ax_c = axes_comp[_row_c][_col_c_i]
        _mask_c  = ccys_all == _ccy_c
        if _mask_c.sum() == 0:
            ax_c.set_visible(False)
            continue
        _dates_c    = dates_all[_mask_c]
        _idx_loc_c  = np.argmin(np.abs(_dates_c - _target_c))
        _act_date_c = _dates_c[_idx_loc_c]
        _gidx_c     = np.where(_mask_c)[0][_idx_loc_c]

        _actual_c = X_np_all[_gidx_c] * _SCALE_COMP
        ax_c.plot(tenors, _actual_c, "o-", color="black",
                  linewidth=2.0, markersize=5, label="Actual", zorder=6)

        for (_, _lbl_c2, _, _, _), _col_c2 in zip(_COMP_VARIANTS_MIXED, _COMP_COLORS):
            if _lbl_c2 not in _comp_S_hat:
                continue
            _fitted_c = _comp_S_hat[_lbl_c2][_gidx_c] * _SCALE_COMP
            ax_c.plot(tenors, _fitted_c, "-", color=_col_c2,
                      linewidth=1.6, label=_lbl_c2)

        if _row_c == 0:
            ax_c.set_title(_ccy_c, fontsize=12, fontweight="bold")
        if _col_c_i == 0:
            ax_c.set_ylabel(f"{_label_c}\n({'%' if SCALE_IS_PERCENT else 'dec.'})",
                            fontsize=11)
        if _row_c == _n_rows_comp - 1:
            ax_c.set_xlabel("Maturity", fontsize=11)
        ax_c.set_xticks(tenors)
        if _row_c == _n_rows_comp - 1:
            ax_c.set_xticklabels([str(int(t)) for t in tenors], fontsize=9)
        else:
            ax_c.set_xticklabels([])
        ax_c.tick_params(axis="y", labelsize=10)
        ax_c.text(0.97, 0.05, pd.Timestamp(_act_date_c).strftime("%Y-%m-%d"),
                  transform=ax_c.transAxes, fontsize=9, ha="right", color="0.4")

_h_comp, _l_comp = axes_comp[0][0].get_legend_handles_labels()
fig_comp.legend(_h_comp, _l_comp, loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=len(_COMP_VARIANTS_MIXED) + 1, frameon=False, fontsize=12)
fig_comp.tight_layout()
fig_comp.subplots_adjust(bottom=0.12)
save_fig(fig_comp, "all_models_fitted_vs_actual")

# ── figure: short rate over time — 2×2 grid, one panel per model, all 9 ccys ─
print("\nGenerating short rate time series figure (all models)...")

_SR_SCALE   = 100.0 if SCALE_IS_PERCENT else 1.0
_SR_YLABEL  = "Short rate (%)" if SCALE_IS_PERCENT else "Short rate"
_CCY_COLORS = plt.cm.tab10.colors

dates_all_sr = pd.to_datetime(meta["as_of_date"].values)
ccys_all_sr  = meta["ccy"].values

fig_sr, axes_sr = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
axes_sr_flat = axes_sr.flatten()

for _ax_i, ((_vkey_sr, _lbl_sr, _, _, _), _col_sr) in enumerate(
        zip(_COMP_VARIANTS_MIXED, _COMP_COLORS)):
    ax_sr = axes_sr_flat[_ax_i]

    if _lbl_sr not in _comp_r_tilde:
        print(f"  ⚠️  No r_tilde for {_lbl_sr} — skipping.")
        ax_sr.set_visible(False)
        continue

    _r_sr = _comp_r_tilde[_lbl_sr] * _SR_SCALE

    for _ci, _ccy_sr in enumerate(CCY_ORDER):
        _mask_sr  = ccys_all_sr == _ccy_sr
        if not _mask_sr.any():
            continue
        _dates_sr = dates_all_sr[_mask_sr]
        _r_ccy    = _r_sr[_mask_sr]
        _sort_idx = np.argsort(_dates_sr)
        ax_sr.plot(_dates_sr[_sort_idx], _r_ccy[_sort_idx],
                   linewidth=1.0, label=_ccy_sr, color=_CCY_COLORS[_ci])

    ax_sr.set_title(_lbl_sr, fontsize=11, fontweight="bold")
    if _ax_i % 2 == 0:
        ax_sr.set_ylabel(_SR_YLABEL, fontsize=10)
    ax_sr.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
    ax_sr.xaxis.set_major_locator(_mdates.YearLocator(2))
    ax_sr.grid(True, alpha=0.25)

fig_sr.autofmt_xdate()

_h_sr, _l_sr = axes_sr_flat[0].get_legend_handles_labels()
fig_sr.legend(_h_sr, _l_sr, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=len(CCY_ORDER), fontsize=9, frameon=False)
fig_sr.tight_layout()
fig_sr.subplots_adjust(bottom=0.12)
save_fig(fig_sr, "all_models_short_rate")

# ── figure: OOS regime scatter — all models (mixed dims) ─────────────────────
print("\nGenerating OOS regime scatter — all models (mixed dims) figure...")

fig_oos, axes_oos = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=False)
axes_oos_flat = axes_oos.flatten()

for _ax_i, (_vkey, _lbl_oos, _, _, _dim_oos) in enumerate(_COMP_VARIANTS_MIXED):
    _ax_oos = axes_oos_flat[_ax_i]
    _df_oos = _load_oos_preds(_vkey, _dim_oos)
    if _df_oos is None:
        _ax_oos.set_visible(False)
        continue
    print(f"  {_lbl_oos}: avg OOS RMSE = {_df_oos['rmse_bps'].mean():.2f} bps")

    for _lbl_r, _inv_r, _neg_r, _col_r in _REGIME_GROUPS_OOS:
        _mask_r = (_df_oos["inv_flag"] == _inv_r) & (_df_oos["neg_flag"] == _neg_r)
        if not _mask_r.any():
            continue
        _ax_oos.scatter(
            _df_oos.loc[_mask_r, "as_of_date"].values,
            _df_oos.loc[_mask_r, "rmse_bps"].values,
            s=3, alpha=0.35, color=_col_r, marker="o", label=_lbl_r, zorder=3,
        )

    if _ax_i % 2 == 0:
        _ax_oos.set_ylabel("RMSE (bps)", fontsize=9)
    _ax_oos.annotate(_lbl_oos, xy=(0.99, 0.97), xycoords="axes fraction",
                     ha="right", va="top", fontsize=10, fontweight="bold")
    _ax_oos.grid(True, alpha=0.3)

for _ax_bot in axes_oos[1]:
    _ax_bot.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
fig_oos.autofmt_xdate()

_leg_handles_oos = [
    _mlines.Line2D([], [], marker="o", color=_col_r, linestyle="None", markersize=5)
    for _lbl_r, _, _, _col_r in _REGIME_GROUPS_OOS if _lbl_r != "Inverted, Negative"
]
_leg_labels_oos = [
    _lbl_r for _lbl_r, _, _, _ in _REGIME_GROUPS_OOS if _lbl_r != "Inverted, Negative"
]
fig_oos.legend(_leg_handles_oos, _leg_labels_oos,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=4, fontsize=11, frameon=True, facecolor="white", edgecolor="none",
               markerscale=1.5)
fig_oos.tight_layout()
fig_oos.subplots_adjust(bottom=0.12)
save_fig(fig_oos, "oos_regime_scatter_all_models")

# ── table: IS vs OOS summary across all four models ──────────────────────────
print("\nGenerating IS vs OOS summary table...")

_tbl_rows = []
# clean model names for CSV (no LaTeX markup)
_clean_names = ["Baseline", "Stable", "Augmented", "Aug.+Stable"]
for (_vkey_t, _lbl_t, _, _, _dim_t), _clean_t in zip(_COMP_VARIANTS_MIXED, _clean_names):
    # IS RMSE — from the in-sample predictions already computed above
    if _lbl_t in _comp_S_hat:
        _is_rmse = np.sqrt(np.mean((X_np_all - _comp_S_hat[_lbl_t]) ** 2, axis=1)) * 1e4
        _is_avg    = f"{np.mean(_is_rmse):.2f}"
        _is_median = f"{np.median(_is_rmse):.2f}"
        _is_max    = f"{np.max(_is_rmse):.2f}"
    else:
        _is_avg = _is_median = _is_max = "---"

    # OOS RMSE — from rolling predictions CSV
    _df_t = _load_oos_preds(_vkey_t, _dim_t)
    if _df_t is not None:
        _oos_avg    = f"{_df_t['rmse_bps'].mean():.2f}"
        _oos_median = f"{_df_t['rmse_bps'].median():.2f}"
        _oos_max    = f"{_df_t['rmse_bps'].max():.2f}"
    else:
        _oos_avg = _oos_median = _oos_max = "---"

    _tbl_rows.append({
        "Model":      _clean_t,
        "ell":        str(_dim_t),
        "IS_Avg":     _is_avg,
        "IS_Median":  _is_median,
        "IS_Max":     _is_max,
        "OOS_Avg":    _oos_avg,
        "OOS_Median": _oos_median,
        "OOS_Max":    _oos_max,
    })

_tbl_df = pd.DataFrame(_tbl_rows).set_index("Model")

# save CSV
_tbl_csv = os.path.join(FIGURES_OUT, "all_models_is_oos_summary.csv")
_tbl_df.to_csv(_tbl_csv)
print(f"  Saved: {_tbl_csv}")
print(_tbl_df.to_string())

print("\nResultsGenerator_augmented complete.")
