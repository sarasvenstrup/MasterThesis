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
LATENT_DIM = 3
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
    f1 = x[:, 4] - x[:, 0]                          # 10Y − 1Y
    f2 = x[:, 7] - x[:, 4]                          # 30Y − 10Y
    f3 = 2.0 * x[:, 4] - x[:, 0] - x[:, 7]         # 2×10Y − 1Y − 30Y
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

# ── shared setup for all-models comparison figures ────────────────────────────
from Code.model.full_model import FullModel as _BaselineFullModel
import matplotlib.dates as _mdates
import matplotlib.lines as _mlines
import matplotlib.transforms

EVENTS = {
    "GFC\n(15 Sep 2008)":      "2008-09-15",
    "QE\n(22 Jan 2015)":       "2015-01-22",
    "COVID\n(1 Mar 2020)":     "2020-03-01",
    "Inflation\n(1 Mar 2022)": "2022-03-01",
}

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
    ("baseline",         "Baseline ($\\ell=2$)",    _BaselineFullModel, False, 2),
    ("stable",           "Stable ($\\ell=4$)",      FullModel,          False, 4),
    ("augmented_input",  "Augmented ($\\ell=2$)",   _BaselineFullModel, True,  2),
    ("augmented_stable", "Aug. + Stable ($\\ell=3$)", FullModel,          True,  3),
]

_REGIME_GROUPS_OOS = [
    ("Normal",          "normal_flag",    custom_palette[2]),
    ("Inverted",        "inv_flag",       "black"),
    ("Crossing",        "cross_flag",     "indianred"),
    ("Deeply Negative", "deep_flag",      "#00FF7F"),
    ("Other Negative",  "other_neg_flag", "#E67E22"),
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
    _actual_cols = sorted([c for c in _df.columns if c.startswith("actual_")],
                          key=lambda c: int(c.split("_")[-1]))
    _fitted_cols = sorted([c for c in _df.columns if c.startswith("fitted_")],
                          key=lambda c: int(c.split("_")[-1]))
    _actual_v    = _df[_actual_cols].values
    _fitted_v    = _df[_fitted_cols].values
    _df["rmse_bps"]       = np.sqrt(np.mean((_actual_v - _fitted_v) ** 2, axis=1)) * 1e4
    _deep                 = (_actual_v < 0).sum(axis=1) >= 7
    _cross                = ((_actual_v[:, :5] < 0).all(axis=1)) & (_actual_v[:, -1] > 0) & ~_deep
    _neg_any              = (_actual_v < 0).any(axis=1)
    _inv                  = (_actual_v[:, 0] > _actual_v[:, -1]) & ~_neg_any
    _df["deep_flag"]      = _deep
    _df["cross_flag"]     = _cross
    _df["inv_flag"]       = _inv
    _df["other_neg_flag"] = _neg_any & ~_deep & ~_cross
    _df["normal_flag"]    = ~_neg_any & ~_inv
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
        _in_dim_c  = _raw_c["model_config"].get("input_dim",
                         _INPUT_DIM_AUG if _use_aug_c else _INPUT_DIM_BASE)
        _lat_dim_c = _raw_c["model_config"].get("latent_dim", _dim_c)
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

# ── figure: failure modes — 2×2 grid, all four comparison models ──────────────
# Same four representative curves as failure_modes_all_models.png in the
# augmented chapter, but with all four model comparison variants overlaid.
print("\nGenerating failure modes comparison figure (all models)...")

# ── curve indices ──────────────────────────────────────────────────────────────
# Compute baseline RMSE for worst-curve selection
_fmc_b_rmse = np.sqrt(np.mean(
    (X_np_all - _comp_S_hat["Baseline ($\\ell=2$)"]) ** 2, axis=1
)) * 1e4

# Panel TL: EUR 2015-03-31
_fmc_eur15_idx = np.where(
    (meta["ccy"].values == "EUR") &
    (pd.to_datetime(meta["as_of_date"].values) == pd.Timestamp("2015-03-31"))
)[0]
_fmc_gi_eur15 = _fmc_eur15_idx[0] if len(_fmc_eur15_idx) > 0 else None

# Panel TR: worst deeply negative curve (≥7/8 tenors < 0)
_fmc_mask_deep = (X_np_all < 0).sum(axis=1) >= 7
_fmc_idx_deep  = np.where(_fmc_mask_deep)[0]
_fmc_gi_deep   = _fmc_idx_deep[np.argmax(_fmc_b_rmse[_fmc_idx_deep])]

# Panel BL: JPY 2016-09-30 — crossing curve
_fmc_jpy_idx = np.where(
    (meta["ccy"].values == "JPY") &
    (pd.to_datetime(meta["as_of_date"].values) == pd.Timestamp("2016-09-30"))
)[0]
if len(_fmc_jpy_idx) == 0:
    print("  ⚠️  JPY 2016-09-30 not found — falling back to worst crossing curve")
    _fmc_mask_cross = (
        (X_np_all[:, :5] < 0).all(axis=1) & (X_np_all[:, -1] > 0) & ~_fmc_mask_deep
    )
    _fmc_idx_cross = np.where(_fmc_mask_cross)[0]
    _fmc_gi_cross  = _fmc_idx_cross[np.argmax(_fmc_b_rmse[_fmc_idx_cross])]
else:
    _fmc_gi_cross = _fmc_jpy_idx[0]

# Panel BR: CAD 2023-01-31
_fmc_cad_idx = np.where(
    (meta["ccy"].values == "CAD") &
    (pd.to_datetime(meta["as_of_date"].values) == pd.Timestamp("2023-01-31"))
)[0]
_fmc_gi_cad = _fmc_cad_idx[0] if len(_fmc_cad_idx) > 0 else None

# Short RMSE labels matching _COMP_VARIANTS_MIXED order
_fmc_short_labels = ["B", "S", "A", "AS"]

# (global_index, show_ylabel, show_xlabel, rmse_y, rmse_va, rmse_x, rmse_ha)
_fmc_panels = [
    (_fmc_gi_eur15, True,  False, 0.03, "bottom", 0.03, "left"),
    (_fmc_gi_deep,  False, False, 0.97, "top",    0.97, "right"),
    (_fmc_gi_cross, True,  True,  0.03, "bottom", 0.03, "left"),
    (_fmc_gi_cad,   False, True,  0.97, "top",    0.97, "right"),
]

fig_fmc, axes_fmc = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
axes_fmc_flat = axes_fmc.flatten()

for _pan_i, (gi, show_ylabel, show_xlabel,
             rmse_y, rmse_va, rmse_x, rmse_ha) in enumerate(_fmc_panels):
    ax_fmc = axes_fmc_flat[_pan_i]
    if gi is None:
        ax_fmc.set_visible(False)
        continue

    _act_fmc  = X_np_all[gi] * 100.0
    _ccy_fmc  = meta["ccy"].values[gi]
    _date_fmc = pd.to_datetime(meta["as_of_date"].values[gi]).strftime("%Y-%m-%d")

    ax_fmc.plot(tenors, _act_fmc, "o-", color="black",
                linewidth=1.8, markersize=5, label="Actual", zorder=6)

    _rmse_parts = []
    for (_short, (_, _lbl_fmc, _, _, _), _col_fmc) in zip(
            _fmc_short_labels, _COMP_VARIANTS_MIXED, _COMP_COLORS):
        if _lbl_fmc not in _comp_S_hat:
            continue
        _fitted_fmc = _comp_S_hat[_lbl_fmc][gi] * 100.0
        _rmse_val   = float(np.sqrt(np.mean(
            (X_np_all[gi] - _comp_S_hat[_lbl_fmc][gi]) ** 2
        ))) * 1e4
        ax_fmc.plot(tenors, _fitted_fmc, "-", color=_col_fmc,
                    linewidth=1.8, label=_lbl_fmc)
        _rmse_parts.append(f"{_short}:{_rmse_val:.1f}")

    _rmse_txt = " / ".join(_rmse_parts) + " bps"

    # y-axis limits from all plotted values
    _all_fmc_vals = np.concatenate(
        [_act_fmc] + [
            _comp_S_hat[_lbl_fmc][gi] * 100.0
            for _, _lbl_fmc, _, _, _ in _COMP_VARIANTS_MIXED
            if _lbl_fmc in _comp_S_hat
        ]
    )
    _ymin_fmc = _all_fmc_vals.min()
    _ymax_fmc = _all_fmc_vals.max()
    _ypad_fmc = (_ymax_fmc - _ymin_fmc) * 0.15
    ax_fmc.set_ylim(_ymin_fmc - _ypad_fmc, _ymax_fmc + _ypad_fmc)

    ax_fmc.axhline(0, color="0.7", linewidth=0.8, linestyle=":")
    ax_fmc.set_title(f"{_ccy_fmc}  {_date_fmc}", fontsize=13)
    ax_fmc.text(rmse_x, rmse_y, _rmse_txt,
                transform=ax_fmc.transAxes, fontsize=9,
                ha=rmse_ha, va=rmse_va, color="0.4")
    if show_xlabel:
        ax_fmc.set_xlabel("Maturity", fontsize=12)
    if show_ylabel:
        ax_fmc.set_ylabel("Swap rate (%)", fontsize=12)
    ax_fmc.tick_params(labelsize=11)
    ax_fmc.spines["top"].set_visible(False)
    ax_fmc.spines["right"].set_visible(False)

_h_fmc, _l_fmc = axes_fmc_flat[0].get_legend_handles_labels()
fig_fmc.legend(_h_fmc, _l_fmc, loc="lower center",
               bbox_to_anchor=(0.5, -0.04),
               ncol=len(_COMP_VARIANTS_MIXED) + 1,
               fontsize=11, frameon=False)
fig_fmc.tight_layout()
fig_fmc.subplots_adjust(bottom=0.12)
save_fig(fig_fmc, "failure_modes_model_comparison")

# ── figure: worst reconstruction — 2×2 grid, one panel per model ─────────────
print("\nGenerating worst reconstruction figure (all models)...")

fig_wr, axes_wr = plt.subplots(2, 3, figsize=(17, 8), sharey=False)
axes_wr_flat = axes_wr.flatten()

for _ax_i, ((_vkey_wr, _lbl_wr, _, _, _), _col_wr) in enumerate(
        zip(_COMP_VARIANTS_MIXED, _COMP_COLORS)):
    ax_wr = axes_wr_flat[_ax_i]

    if _lbl_wr not in _comp_S_hat:
        ax_wr.set_visible(False)
        continue

    _fitted_wr  = _comp_S_hat[_lbl_wr]
    _rmse_wr    = np.sqrt(np.mean((X_np_all - _fitted_wr) ** 2, axis=1)) * 1e4
    _worst_idx  = int(np.argmax(_rmse_wr))
    _worst_rmse = _rmse_wr[_worst_idx]
    _worst_date = pd.Timestamp(dates_all[_worst_idx]).strftime("%Y-%m-%d")
    _worst_ccy  = ccys_all[_worst_idx]

    _actual_wr = X_np_all[_worst_idx] * _SCALE_COMP
    _fitted_wr_curve = _fitted_wr[_worst_idx] * _SCALE_COMP

    ax_wr.plot(tenors, _actual_wr, "o-", color="black",
               linewidth=2.0, markersize=5, label="Actual", zorder=5)
    ax_wr.plot(tenors, _fitted_wr_curve, "-", color=_col_wr,
               linewidth=1.8, label=_lbl_wr)

    ax_wr.set_title(_lbl_wr, fontsize=11, fontweight="bold")
    ax_wr.set_xticks(tenors)
    ax_wr.set_xticklabels([str(int(t)) for t in tenors], fontsize=9)
    ax_wr.set_xlabel("Maturity", fontsize=10)
    if _ax_i % 2 == 0:
        ax_wr.set_ylabel(f"Rate ({'%' if SCALE_IS_PERCENT else 'dec.'})", fontsize=10)
    ax_wr.tick_params(axis="y", labelsize=9)
    if _ax_i == 0:
        _ann_x, _ann_y, _ann_ha, _ann_va = 0.03, 0.97, "left", "top"
    else:
        _ann_x, _ann_y, _ann_ha, _ann_va = 0.97, 0.03, "right", "bottom"
    ax_wr.text(_ann_x, _ann_y,
               f"{_worst_ccy}  {_worst_date}\nRMSE = {_worst_rmse:.1f} bps",
               transform=ax_wr.transAxes, fontsize=9,
               ha=_ann_ha, va=_ann_va, color="0.3")

# build legend excluding the baseline label
_h_wr_all, _l_wr_all = [], []
for _ax_wr_leg in axes_wr_flat:
    for _h, _l in zip(*_ax_wr_leg.get_legend_handles_labels()):
        if _l not in _l_wr_all and _l != list(_COMP_VARIANTS_MIXED[0])[0] and "Baseline" not in _l:
            _h_wr_all.append(_h)
            _l_wr_all.append(_l)
fig_wr.legend(_h_wr_all, _l_wr_all, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=len(_l_wr_all), frameon=False, fontsize=11)
fig_wr.tight_layout()
fig_wr.subplots_adjust(bottom=0.10)
save_fig(fig_wr, "all_models_worst_reconstruction")

# ── figure: worst OOS reconstruction — 2×2 grid, one panel per model ─────────
print("\nGenerating worst OOS reconstruction figure (all models)...")

fig_wr_oos, axes_wr_oos = plt.subplots(2, 3, figsize=(17, 8), sharey=False)
axes_wr_oos_flat = axes_wr_oos.flatten()

for _ax_i, ((_vkey_wr, _lbl_wr, _, _, _dim_wr), _col_wr) in enumerate(
        zip(_COMP_VARIANTS_MIXED, _COMP_COLORS)):
    ax_wr = axes_wr_oos_flat[_ax_i]

    _df_wr = _load_oos_preds(_vkey_wr, _dim_wr)
    if _df_wr is None:
        ax_wr.set_visible(False)
        continue

    _actual_cols_wr = sorted([c for c in _df_wr.columns if c.startswith("actual_")],
                              key=lambda c: int(c.split("_")[-1]))
    _fitted_cols_wr = sorted([c for c in _df_wr.columns if c.startswith("fitted_")],
                              key=lambda c: int(c.split("_")[-1]))
    _worst_idx  = int(_df_wr["rmse_bps"].idxmax())
    _worst_rmse = _df_wr.loc[_worst_idx, "rmse_bps"]
    _worst_date = _df_wr.loc[_worst_idx, "as_of_date"].strftime("%Y-%m-%d")
    _worst_ccy  = _df_wr.loc[_worst_idx, "ccy"]

    _actual_wr_oos = _df_wr.loc[_worst_idx, _actual_cols_wr].values.astype(float) * _SCALE_COMP
    _fitted_wr_oos = _df_wr.loc[_worst_idx, _fitted_cols_wr].values.astype(float) * _SCALE_COMP

    ax_wr.plot(tenors, _actual_wr_oos, "o-", color="black",
               linewidth=2.0, markersize=5, label="Actual", zorder=5)
    ax_wr.plot(tenors, _fitted_wr_oos, "-", color=_col_wr,
               linewidth=1.8, label=_lbl_wr)

    ax_wr.set_title(_lbl_wr, fontsize=11, fontweight="bold")
    ax_wr.set_xticks(tenors)
    ax_wr.set_xticklabels([str(int(t)) for t in tenors], fontsize=9)
    ax_wr.set_xlabel("Maturity", fontsize=10)
    if _ax_i % 2 == 0:
        ax_wr.set_ylabel(f"Rate ({'%' if SCALE_IS_PERCENT else 'dec.'})", fontsize=10)
    ax_wr.tick_params(axis="y", labelsize=9)
    if _ax_i == 0:
        _ann_x, _ann_y, _ann_ha, _ann_va = 0.03, 0.97, "left", "top"
    else:
        _ann_x, _ann_y, _ann_ha, _ann_va = 0.97, 0.03, "right", "bottom"
    ax_wr.text(_ann_x, _ann_y,
               f"{_worst_ccy}  {_worst_date}\nRMSE = {_worst_rmse:.1f} bps",
               transform=ax_wr.transAxes, fontsize=9,
               ha=_ann_ha, va=_ann_va, color="0.3")

_h_wr_oos, _l_wr_oos = [], []
for _ax_leg in axes_wr_oos_flat:
    for _h, _l in zip(*_ax_leg.get_legend_handles_labels()):
        if _l not in _l_wr_oos and "Baseline" not in _l:
            _h_wr_oos.append(_h)
            _l_wr_oos.append(_l)
fig_wr_oos.legend(_h_wr_oos, _l_wr_oos, loc="lower center", bbox_to_anchor=(0.5, -0.02),
                  ncol=len(_l_wr_oos), frameon=False, fontsize=11)
fig_wr_oos.tight_layout()
fig_wr_oos.subplots_adjust(bottom=0.10)
save_fig(fig_wr_oos, "all_models_worst_oos_reconstruction")

# ── figure: worst OOS actual curve only — 2×2 grid, one panel per model ──────
print("\nGenerating worst OOS actual-only figure (all models)...")

fig_wr_act, axes_wr_act = plt.subplots(2, 3, figsize=(17, 8), sharey=False)
axes_wr_act_flat = axes_wr_act.flatten()

for _ax_i, ((_vkey_wr, _lbl_wr, _, _, _dim_wr), _col_wr) in enumerate(
        zip(_COMP_VARIANTS_MIXED, _COMP_COLORS)):
    ax_wr = axes_wr_act_flat[_ax_i]

    _df_wr = _load_oos_preds(_vkey_wr, _dim_wr)
    if _df_wr is None:
        ax_wr.set_visible(False)
        continue

    _actual_cols_wr = sorted([c for c in _df_wr.columns if c.startswith("actual_")],
                              key=lambda c: int(c.split("_")[-1]))
    _worst_idx  = int(_df_wr["rmse_bps"].idxmax())
    _worst_rmse = _df_wr.loc[_worst_idx, "rmse_bps"]
    _worst_date = _df_wr.loc[_worst_idx, "as_of_date"].strftime("%Y-%m-%d")
    _worst_ccy  = _df_wr.loc[_worst_idx, "ccy"]

    _actual_wr_act = _df_wr.loc[_worst_idx, _actual_cols_wr].values.astype(float) * _SCALE_COMP

    ax_wr.plot(tenors, _actual_wr_act, "o-", color="black",
               linewidth=2.0, markersize=5)

    ax_wr.set_title(_lbl_wr, fontsize=11, fontweight="bold")
    ax_wr.set_xticks(tenors)
    ax_wr.set_xticklabels([str(int(t)) for t in tenors], fontsize=9)
    ax_wr.set_xlabel("Maturity", fontsize=10)
    if _ax_i % 2 == 0:
        ax_wr.set_ylabel(f"Rate ({'%' if SCALE_IS_PERCENT else 'dec.'})", fontsize=10)
    ax_wr.tick_params(axis="y", labelsize=9)
    if _ax_i == 0:
        _ann_x, _ann_y, _ann_ha, _ann_va = 0.03, 0.97, "left", "top"
    else:
        _ann_x, _ann_y, _ann_ha, _ann_va = 0.97, 0.03, "right", "bottom"
    ax_wr.text(_ann_x, _ann_y,
               f"{_worst_ccy}  {_worst_date}\nRMSE = {_worst_rmse:.1f} bps",
               transform=ax_wr.transAxes, fontsize=9,
               ha=_ann_ha, va=_ann_va, color="0.3")

fig_wr_act.tight_layout()
save_fig(fig_wr_act, "all_models_worst_oos_actual_only")

# ── figure: actual curve one month before worst OOS — 2×2 grid ────────────────
print("\nGenerating actual curve one month before worst OOS figure (all models)...")

fig_wr_prev, axes_wr_prev = plt.subplots(2, 3, figsize=(17, 8), sharey=False)
axes_wr_prev_flat = axes_wr_prev.flatten()

for _ax_i, ((_vkey_wr, _lbl_wr, _, _, _dim_wr), _col_wr) in enumerate(
        zip(_COMP_VARIANTS_MIXED, _COMP_COLORS)):
    ax_wr = axes_wr_prev_flat[_ax_i]

    _df_wr = _load_oos_preds(_vkey_wr, _dim_wr)
    if _df_wr is None:
        ax_wr.set_visible(False)
        continue

    _actual_cols_wr = sorted([c for c in _df_wr.columns if c.startswith("actual_")],
                              key=lambda c: int(c.split("_")[-1]))
    _worst_idx  = int(_df_wr["rmse_bps"].idxmax())
    _worst_date = _df_wr.loc[_worst_idx, "as_of_date"]
    _worst_ccy  = _df_wr.loc[_worst_idx, "ccy"]
    _target_date = _worst_date - pd.DateOffset(months=1)

    # find closest observation for same currency ~1 month before
    _mask_ccy = _df_wr["ccy"] == _worst_ccy
    _dates_ccy = _df_wr.loc[_mask_ccy, "as_of_date"]
    _prev_candidates = _dates_ccy[_dates_ccy <= _target_date]
    if _prev_candidates.empty:
        ax_wr.set_visible(False)
        continue
    _prev_local_idx = (_prev_candidates - _target_date).abs().idxmin()
    _prev_date = _df_wr.loc[_prev_local_idx, "as_of_date"].strftime("%Y-%m-%d")
    _prev_actual = _df_wr.loc[_prev_local_idx, _actual_cols_wr].values.astype(float) * _SCALE_COMP

    ax_wr.plot(tenors, _prev_actual, "o-", color="black",
               linewidth=2.0, markersize=5)

    ax_wr.set_title(_lbl_wr, fontsize=11, fontweight="bold")
    ax_wr.set_xticks(tenors)
    ax_wr.set_xticklabels([str(int(t)) for t in tenors], fontsize=9)
    ax_wr.set_xlabel("Maturity", fontsize=10)
    if _ax_i % 2 == 0:
        ax_wr.set_ylabel(f"Rate ({'%' if SCALE_IS_PERCENT else 'dec.'})", fontsize=10)
    ax_wr.tick_params(axis="y", labelsize=9)
    if _ax_i == 0:
        _ann_x, _ann_y, _ann_ha, _ann_va = 0.03, 0.97, "left", "top"
    else:
        _ann_x, _ann_y, _ann_ha, _ann_va = 0.97, 0.03, "right", "bottom"
    ax_wr.text(_ann_x, _ann_y,
               f"{_worst_ccy}  {_prev_date}",
               transform=ax_wr.transAxes, fontsize=9,
               ha=_ann_ha, va=_ann_va, color="0.3")

fig_wr_prev.tight_layout()
save_fig(fig_wr_prev, "all_models_worst_oos_prev_month")

# ── figure: short rate over time — 2×2 grid, one panel per model, all 9 ccys ─
print("\nGenerating short rate time series figure (all models)...")

_SR_SCALE   = 100.0 if SCALE_IS_PERCENT else 1.0
_SR_YLABEL  = "Short rate (%)" if SCALE_IS_PERCENT else "Short rate"
_CCY_COLORS = plt.cm.tab10.colors

dates_all_sr = pd.to_datetime(meta["as_of_date"].values)
ccys_all_sr  = meta["ccy"].values

fig_sr, axes_sr = plt.subplots(2, 3, figsize=(19, 8), sharex=True, sharey=False)
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

_CLIP_BPS = 215

# ── pre-pass: find global max quarterly clipped count across all panels ───────
_global_max_clipped = 0
for _vkey_pre, _, _, _, _dim_pre in _COMP_VARIANTS_MIXED:
    _df_pre = _load_oos_preds(_vkey_pre, _dim_pre)
    if _df_pre is None:
        continue
    _cl_pre = _df_pre[_df_pre["rmse_bps"] > _CLIP_BPS].copy()
    if len(_cl_pre) == 0:
        continue
    _cl_pre["quarter"] = _cl_pre["as_of_date"].dt.to_period("Q")
    _qtot = _cl_pre.groupby("quarter").size()
    _global_max_clipped = max(_global_max_clipped, _qtot.max())

_bar_ylim_top = 10

fig_oos, axes_oos = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=False)
axes_oos_flat = axes_oos.flatten()

for _ax_i, (_vkey, _lbl_oos, _, _, _dim_oos) in enumerate(_COMP_VARIANTS_MIXED):
    _ax_oos = axes_oos_flat[_ax_i]
    _df_oos = _load_oos_preds(_vkey, _dim_oos)
    if _df_oos is None:
        _ax_oos.set_visible(False)
        continue
    print(f"  {_lbl_oos}: avg OOS RMSE = {_df_oos['rmse_bps'].mean():.2f} bps")

    # ── background: quarterly stacked bar chart on twin axis ─────────────────
    _ax_bar = _ax_oos.twinx()
    _ax_bar.patch.set_visible(False)   # transparent — ax_oos keeps theme background

    _clipped_df = _df_oos[_df_oos["rmse_bps"] > _CLIP_BPS].copy()
    if len(_clipped_df) > 0:
        _clipped_df["quarter"] = _clipped_df["as_of_date"].dt.to_period("Q")
        _all_quarters = sorted(_clipped_df["quarter"].unique())
        _qmid_num = _mdates.date2num([
            q.start_time + pd.Timedelta(days=45) for q in _all_quarters
        ])
        _bar_w_num = 70

        _bottom_arr = np.zeros(len(_all_quarters))
        for _lbl_r, _flag_r, _col_r in _REGIME_GROUPS_OOS:
            _counts = np.array([
                int(((_clipped_df["quarter"] == _q) & (_clipped_df[_flag_r])).sum())
                for _q in _all_quarters
            ])
            if _counts.sum() == 0:
                continue
            _ax_bar.bar(_qmid_num, _counts, bottom=_bottom_arr,
                        color=_col_r, alpha=0.25, width=_bar_w_num, zorder=1)
            _bottom_arr += _counts

        _ax_bar.set_ylim(0, _bar_ylim_top)

    _ax_bar.grid(False)
    _ax_bar.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _ax_bar.tick_params(axis="y", labelsize=9, colors="0.5")
    if _ax_i % 2 == 1:   # right column only
        _ax_bar.set_ylabel("N clipped (quarterly)", fontsize=9, color="0.5")
    else:
        _ax_bar.set_ylabel("")
    _ax_bar.spines[["top", "left"]].set_visible(False)
    _ax_bar.spines["right"].set_color("0.7")

    # ── scatter: only plot points within clip threshold ───────────────────────
    for _lbl_r, _flag_r, _col_r in _REGIME_GROUPS_OOS:
        _mask_r = _df_oos[_flag_r] & (_df_oos["rmse_bps"] <= _CLIP_BPS)
        if not _mask_r.any():
            continue
        _ax_oos.scatter(
            _df_oos.loc[_mask_r, "as_of_date"].values,
            _df_oos.loc[_mask_r, "rmse_bps"].values,
            s=3, alpha=0.35, color=_col_r, marker="o", label=_lbl_r, zorder=3,
        )

    # ── clip boundary ─────────────────────────────────────────────────────────
    _ax_oos.set_ylim(0, _CLIP_BPS)
    _ax_oos.axhline(_CLIP_BPS, color="0.5", linewidth=0.8, linestyle="--", zorder=2)
    if "Stable" not in _lbl_oos or "Aug" in _lbl_oos:
        _ax_oos.text(0.01, 0.985, f"clipped above {_CLIP_BPS} bps",
                     transform=_ax_oos.transAxes, fontsize=6, va="top", color="0.5")

    if _ax_i % 2 == 0:
        _ax_oos.set_ylabel("RMSE (bps)", fontsize=9)
    _ax_oos.set_title(_lbl_oos, fontsize=10, fontweight="bold")
    _ax_oos.grid(True, alpha=0.3)

for _ax_bot in axes_oos[1]:
    _ax_bot.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
fig_oos.autofmt_xdate()

_leg_handles_oos = [
    _mlines.Line2D([], [], marker="o", color=_col_r, linestyle="None", markersize=5)
    for _lbl_r, _, _col_r in _REGIME_GROUPS_OOS
]
_leg_labels_oos = [_lbl_r for _lbl_r, _, _ in _REGIME_GROUPS_OOS]
fig_oos.legend(_leg_handles_oos, _leg_labels_oos,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=5, fontsize=11, frameon=True, facecolor="white", edgecolor="none",
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
    # IS RMSE — from rolling window summary CSV (avg_in_rmse_bps per window),
    # consistent with how Table 2.2 reports IS RMSE
    _roll_summary_path = os.path.join(
        REPO_ROOT, "Figures", "OOSResults", "Roll",
        f"OOS_roll_dim{_dim_t}_{_vkey_t}",
        _ROLL_SUBDIR, f"ep{_ROLL_EPOCHS}",
        f"oos_rolling_bbg_dim{_dim_t}_{_ROLL_SUBDIR}.csv",
    )
    if os.path.exists(_roll_summary_path):
        _df_is = pd.read_csv(_roll_summary_path)
        _is_vals   = _df_is["avg_in_rmse_bps"].dropna().values
        _is_avg    = f"{np.mean(_is_vals):.2f}"
        _is_median = f"{np.median(_is_vals):.2f}"
        _is_max    = f"{np.max(_is_vals):.2f}"
    else:
        print(f"  ⚠️  Rolling summary CSV not found for IS: {_roll_summary_path}")
        _is_avg = _is_median = _is_max = "---"

    # OOS RMSE — avg/median from individual curve predictions; max from
    # per-window averages (rolling summary CSV) so it reflects the worst
    # window average rather than the worst single curve
    _df_t = _load_oos_preds(_vkey_t, _dim_t)
    if _df_t is not None:
        _oos_avg    = f"{_df_t['rmse_bps'].mean():.2f}"
        _oos_median = f"{_df_t['rmse_bps'].median():.2f}"
    else:
        _oos_avg = _oos_median = "---"

    if os.path.exists(_roll_summary_path):
        _oos_max = f"{_df_is['avg_rmse_bps'].dropna().max():.2f}"
    else:
        _oos_max = "---"

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

# ── table: IS vs OOS summary for augmented_stable across ℓ = 2, 3, 4 ─────────
print("\nGenerating augmented_stable dimension comparison table...")

_dim_rows = []
for _d in [2, 3, 4]:
    _roll_path_d = os.path.join(
        REPO_ROOT, "Figures", "OOSResults", "Roll",
        f"OOS_roll_dim{_d}_augmented_stable",
        _ROLL_SUBDIR, f"ep{_ROLL_EPOCHS}",
        f"oos_rolling_bbg_dim{_d}_{_ROLL_SUBDIR}.csv",
    )
    if os.path.exists(_roll_path_d):
        _df_d = pd.read_csv(_roll_path_d)
        _is_avg    = f"{_df_d['avg_in_rmse_bps'].dropna().mean():.2f}"
        _is_median = f"{_df_d['avg_in_rmse_bps'].dropna().median():.2f}"
        _is_max    = f"{_df_d['avg_in_rmse_bps'].dropna().max():.2f}"
        _oos_max   = f"{_df_d['avg_rmse_bps'].dropna().max():.2f}"
    else:
        print(f"  ⚠️  Rolling summary not found: {_roll_path_d}")
        _is_avg = _is_median = _is_max = _oos_max = "---"

    _df_oos_d = _load_oos_preds("augmented_stable", _d)
    if _df_oos_d is not None:
        _oos_avg    = f"{_df_oos_d['rmse_bps'].mean():.2f}"
        _oos_median = f"{_df_oos_d['rmse_bps'].median():.2f}"
    else:
        _oos_avg = _oos_median = "---"

    _dim_rows.append({
        "ell":        str(_d),
        "IS_Avg":     _is_avg,
        "IS_Median":  _is_median,
        "IS_Max":     _is_max,
        "OOS_Avg":    _oos_avg,
        "OOS_Median": _oos_median,
        "OOS_Max":    _oos_max,
    })

_dim_tbl = pd.DataFrame(_dim_rows).set_index("ell")
_dim_csv = os.path.join(FIGURES_OUT, "augmented_stable_dim_comparison.csv")
_dim_tbl.to_csv(_dim_csv)
print(f"  Saved: {_dim_csv}")
print(_dim_tbl.to_string())

# ── table: IS vs OOS summary for augmented_input across ℓ = 2, 3, 4 ──────────
print("\nGenerating augmented_input dimension comparison table...")

_dim_rows_aug = []
for _d in [2, 3, 4]:
    _roll_path_aug = os.path.join(
        REPO_ROOT, "Figures", "OOSResults", "Roll",
        f"OOS_roll_dim{_d}_augmented_input",
        _ROLL_SUBDIR, f"ep{_ROLL_EPOCHS}",
        f"oos_rolling_bbg_dim{_d}_{_ROLL_SUBDIR}.csv",
    )
    if os.path.exists(_roll_path_aug):
        _df_aug = pd.read_csv(_roll_path_aug)
        _is_avg_aug    = f"{_df_aug['avg_in_rmse_bps'].dropna().mean():.2f}"
        _is_median_aug = f"{_df_aug['avg_in_rmse_bps'].dropna().median():.2f}"
        _is_max_aug    = f"{_df_aug['avg_in_rmse_bps'].dropna().max():.2f}"
        _oos_max_aug   = f"{_df_aug['avg_rmse_bps'].dropna().max():.2f}"
    else:
        print(f"  ⚠️  Rolling summary not found: {_roll_path_aug}")
        _is_avg_aug = _is_median_aug = _is_max_aug = _oos_max_aug = "---"

    _df_oos_aug = _load_oos_preds("augmented_input", _d)
    if _df_oos_aug is not None:
        _oos_avg_aug    = f"{_df_oos_aug['rmse_bps'].mean():.2f}"
        _oos_median_aug = f"{_df_oos_aug['rmse_bps'].median():.2f}"
    else:
        _oos_avg_aug = _oos_median_aug = "---"

    _dim_rows_aug.append({
        "ell":        str(_d),
        "IS_Avg":     _is_avg_aug,
        "IS_Median":  _is_median_aug,
        "IS_Max":     _is_max_aug,
        "OOS_Avg":    _oos_avg_aug,
        "OOS_Median": _oos_median_aug,
        "OOS_Max":    _oos_max_aug,
    })

_dim_tbl_aug = pd.DataFrame(_dim_rows_aug).set_index("ell")
_dim_csv_aug = os.path.join(FIGURES_OUT, "augmented_input_dim_comparison.csv")
_dim_tbl_aug.to_csv(_dim_csv_aug)
print(f"  Saved: {_dim_csv_aug}")
print(_dim_tbl_aug.to_string())

# ── table: combined dimension comparison across all four model types ──────────
print("\nGenerating combined dimension comparison table (all models)...")

_ALL_MODEL_VARIANTS = [
    ("Baseline",    "baseline",         2),
    ("Stable",      "stable",           4),
    ("Augmented",   "augmented_input",  3),
    ("Aug.+Stable", "augmented_stable", 3),
]
_SELECTED_DIMS = {
    "baseline":         2,
    "stable":           4,
    "augmented_input":  3,
    "augmented_stable": 3,
}

_big_rows = []
for _model_name, _vkey_b, _ in _ALL_MODEL_VARIANTS:
    for _d in [2, 3, 4]:
        _roll_path_b = os.path.join(
            REPO_ROOT, "Figures", "OOSResults", "Roll",
            f"OOS_roll_dim{_d}_{_vkey_b}",
            _ROLL_SUBDIR, f"ep{_ROLL_EPOCHS}",
            f"oos_rolling_bbg_dim{_d}_{_ROLL_SUBDIR}.csv",
        )
        if os.path.exists(_roll_path_b):
            _df_b = pd.read_csv(_roll_path_b)
            _is_avg_b    = f"{_df_b['avg_in_rmse_bps'].dropna().mean():.2f}"
            _is_median_b = f"{_df_b['avg_in_rmse_bps'].dropna().median():.2f}"
            _is_max_b    = f"{_df_b['avg_in_rmse_bps'].dropna().max():.2f}"
            _oos_max_b   = f"{_df_b['avg_rmse_bps'].dropna().max():.2f}"
        else:
            print(f"  ⚠️  Not found: {_roll_path_b}")
            _is_avg_b = _is_median_b = _is_max_b = _oos_max_b = "---"

        _df_oos_b = _load_oos_preds(_vkey_b, _d)
        if _df_oos_b is not None:
            _oos_avg_b    = f"{_df_oos_b['rmse_bps'].mean():.2f}"
            _oos_median_b = f"{_df_oos_b['rmse_bps'].median():.2f}"
        else:
            _oos_avg_b = _oos_median_b = "---"

        _big_rows.append({
            "Model":      _model_name,
            "ell":        str(_d),
            "IS_Avg":     _is_avg_b,
            "IS_Median":  _is_median_b,
            "IS_Max":     _is_max_b,
            "OOS_Avg":    _oos_avg_b,
            "OOS_Median": _oos_median_b,
            "OOS_Max":    _oos_max_b,
            "selected":   _d == _SELECTED_DIMS[_vkey_b],
        })

_big_tbl = pd.DataFrame(_big_rows)

# wrap selected rows in \textbf{} and blank out repeated model names
_val_cols = ["IS_Avg", "IS_Median", "IS_Max", "OOS_Avg", "OOS_Median", "OOS_Max"]
for i, (_, row) in enumerate(_big_tbl.iterrows()):
    if row["selected"]:
        for _c in _val_cols + ["ell"]:
            _big_tbl.at[i, _c] = r"\textbf{" + str(_big_tbl.at[i, _c]) + r"}"
    _big_tbl.at[i, "Model"] = row["Model"] if i % 3 == 0 else ""

_big_csv = os.path.join(FIGURES_OUT, "all_models_dim_comparison.csv")
_big_tbl.to_csv(_big_csv, index=False)
print(f"  Saved: {_big_csv}")
print(_big_tbl.drop(columns="selected").to_string(index=False))

# ── Rolling regime dual-axis figure (all four models) ────────────────────────
# Bars:  % Neg / % Inv in train and test sets (same for all models)
# Lines: OOS RMSE per variant, clipped at 50 bps
# ─────────────────────────────────────────────────────────────────────────────
print("\n── All-models rolling regime dual-axis figure ──")

_AM_ACTUAL_COLS = [f"actual_tenor_{i}" for i in range(8)]

def _am_regime_counts(pred_df):
    """Return per-window (test_start) counts of negative and inverted curves."""
    if pred_df is None or pred_df.empty:
        return {}
    counts = {}
    for ts, grp in pred_df.groupby("test_start"):
        neg = int((grp[_AM_ACTUAL_COLS].min(axis=1) < 0).sum())
        inv = int((grp["actual_tenor_0"] > grp["actual_tenor_7"]).sum())
        counts[ts] = {"n_neg": neg, "n_inv": inv}
    return counts

# load rolling summary CSVs and regime counts for each variant
_am_roll_dfs    = {}   # (vkey, dim) -> rolling summary DataFrame
_am_train_counts = None
_am_test_counts  = None

for _vkey_am, _lbl_am, _, _, _dim_am in _COMP_VARIANTS_MIXED:
    _roll_csv = os.path.join(
        REPO_ROOT, "Figures", "OOSResults", "Roll",
        f"OOS_roll_dim{_dim_am}_{_vkey_am}",
        _ROLL_SUBDIR, f"ep{_ROLL_EPOCHS}",
        f"oos_rolling_bbg_dim{_dim_am}_train5Y_test6M_step6M.csv",
    )
    if os.path.exists(_roll_csv):
        _am_roll_dfs[(_vkey_am, _dim_am)] = pd.read_csv(
            _roll_csv, parse_dates=["test_start", "test_end", "train_start"]
        )
    else:
        print(f"  ⚠️  Rolling CSV not found: {_roll_csv}")

    # regime counts — computed once from the first available variant
    if _am_train_counts is None:
        _tr_path = os.path.join(
            REPO_ROOT, "Figures", "OOSResults", "Roll",
            f"OOS_roll_dim{_dim_am}_{_vkey_am}",
            _ROLL_SUBDIR, f"ep{_ROLL_EPOCHS}",
            "predictions_train_all.csv",
        )
        _te_path = os.path.join(
            REPO_ROOT, "Figures", "OOSResults", "Roll",
            f"OOS_roll_dim{_dim_am}_{_vkey_am}",
            _ROLL_SUBDIR, f"ep{_ROLL_EPOCHS}",
            "predictions_test_all.csv",
        )
        if os.path.exists(_tr_path) and os.path.exists(_te_path):
            _am_train_counts = _am_regime_counts(pd.read_csv(_tr_path))
            _am_test_counts  = _am_regime_counts(pd.read_csv(_te_path))

if not _am_roll_dfs:
    print("  ⚠️  No rolling CSVs found — skipping all-models regime figure.")
else:
    # use first available rolling CSV to define the window grid
    _am_ref_df = next(iter(_am_roll_dfs.values()))
    _am_rows = []
    for _, rw in _am_ref_df.iterrows():
        ts  = str(rw["test_start"])[:10]
        te  = str(rw["test_end"])[:10]
        window_label = f"{rw['train_start'].strftime('%Y-%m')} / {te}"

        tc = _am_train_counts.get(ts, {}) if _am_train_counts else {}
        ec = _am_test_counts.get(ts,  {}) if _am_test_counts  else {}

        _ref_rw = _am_ref_df[_am_ref_df["test_start"].dt.strftime("%Y-%m-%d") == ts]
        _n_train = float(_ref_rw["n_train"].values[0]) if len(_ref_rw) else np.nan
        _n_test  = float(_ref_rw["n_test"].values[0])  if len(_ref_rw) else np.nan

        row = {
            "Window": window_label,
            "Train % Neg": round(100 * tc.get("n_neg", np.nan) / _n_train, 1) if np.isfinite(_n_train) and _n_train > 0 else np.nan,
            "Train % Inv": round(100 * tc.get("n_inv", np.nan) / _n_train, 1) if np.isfinite(_n_train) and _n_train > 0 else np.nan,
            "Test % Neg":  round(100 * ec.get("n_neg", np.nan) / _n_test,  1) if np.isfinite(_n_test)  and _n_test  > 0 else np.nan,
            "Test % Inv":  round(100 * ec.get("n_inv", np.nan) / _n_test,  1) if np.isfinite(_n_test)  and _n_test  > 0 else np.nan,
        }
        for (_vk, _dm), _rdf in _am_roll_dfs.items():
            _drow = _rdf[_rdf["test_start"].dt.strftime("%Y-%m-%d") == ts]
            row[f"OOS_{_vk}_{_dm}"] = round(float(_drow["avg_rmse_bps"].values[0]), 2) if len(_drow) else np.nan

        _am_rows.append(row)

    _am_windows  = [r["Window"].split(" / ")[1][:7] for r in _am_rows]
    _am_x        = np.arange(len(_am_windows))
    _am_neg      = np.array([r["Test % Neg"]  for r in _am_rows], dtype=float)
    _am_inv      = np.array([r["Test % Inv"]  for r in _am_rows], dtype=float)
    _am_tr_neg   = np.array([r["Train % Neg"] for r in _am_rows], dtype=float)
    _am_tr_inv   = np.array([r["Train % Inv"] for r in _am_rows], dtype=float)

    _am_neg_col = custom_palette[2]
    _am_inv_col = custom_palette[5]

    fig_am, ax_am_a = plt.subplots(figsize=(13, 5))
    ax_am_b = ax_am_a.twinx()

    _am_w = 0.2
    ax_am_a.bar(_am_x - 1.5*_am_w, _am_tr_neg, width=_am_w, label="% Neg (Train)", color="slategrey", alpha=0.4)
    ax_am_a.bar(_am_x - 0.5*_am_w, _am_neg,    width=_am_w, label="% Neg (Test)",  color="slategrey", alpha=0.9)
    ax_am_a.bar(_am_x + 0.5*_am_w, _am_tr_inv, width=_am_w, label="% Inv (Train)", color=_am_inv_col, alpha=0.5)
    ax_am_a.bar(_am_x + 1.5*_am_w, _am_inv,    width=_am_w, label="% Inv (Test)",  color=_am_inv_col, alpha=1.0)
    ax_am_a.set_ylabel("% of curves in set", fontsize=11)
    ax_am_a.set_xticks(_am_x)
    ax_am_a.set_xticklabels(_am_windows, rotation=45, ha="right", fontsize=10)
    ax_am_a.tick_params(axis="y", labelsize=10)

    for _vi_model, ((_vk_p, _dm_p, _lbl_p), _col_p) in enumerate(zip(
        [(_vk, _dm, _lbl) for (_vk, _lbl, _, _, _dm) in _COMP_VARIANTS_MIXED],
        _COMP_COLORS
    )):
        _key_p = f"OOS_{_vk_p}_{_dm_p}"
        _oos_p = np.array([r.get(_key_p, np.nan) for r in _am_rows], dtype=float)
        _oos_clipped_p = np.clip(_oos_p, 0, 50)
        ax_am_b.plot(_am_x, _oos_clipped_p, marker="o", markersize=4, linewidth=2.2,
                     label=_lbl_p, color=_col_p)
        for _xi_p, (_raw_p, _clip_p) in enumerate(zip(_oos_p, _oos_clipped_p)):
            if _raw_p > 50:
                _on_left_p = _am_windows[_xi_p].startswith("2022-06")
                ax_am_b.annotate(
                    f"{_raw_p:.0f}",
                    xy=(_am_x[_xi_p], 50),
                    xytext=(-5 if _on_left_p else 5, 2 - _vi_model * 8),
                    textcoords="offset points",
                    ha="right" if _on_left_p else "left",
                    va="top", fontsize=10, color=_col_p,
                )

    ax_am_b.set_ylabel("OOS RMSE (bps, clipped at 50)", fontsize=11)
    ax_am_b.tick_params(axis="y", labelsize=10)

    # event markers — interpolate event dates onto the integer x-axis
    _am_win_ts = np.array([pd.Timestamp(w + "-01").value for w in _am_windows], dtype=float)
    for _ev_label, _ev_date_str in EVENTS.items():
        _ev_ts = pd.Timestamp(_ev_date_str).value
        if _am_win_ts[0] <= _ev_ts <= _am_win_ts[-1]:
            _ev_x = float(np.interp(_ev_ts, _am_win_ts, _am_x))
            ax_am_a.axvline(_ev_x, color="0.5", linewidth=1.0, linestyle="--", zorder=0)
            _ev_trans = matplotlib.transforms.blended_transform_factory(
                ax_am_b.transData, ax_am_b.transAxes)
            ax_am_b.text(_ev_x, 1.02, _ev_label, fontsize=9, ha="center", va="bottom",
                         color="0.4", transform=_ev_trans, clip_on=False)

    _lines_a, _labs_a = ax_am_a.get_legend_handles_labels()
    _lines_b, _labs_b = ax_am_b.get_legend_handles_labels()
    fig_am.legend(_lines_a + _lines_b, _labs_a + _labs_b,
                  loc="lower center", bbox_to_anchor=(0.5, -0.08),
                  ncol=4, fontsize=10, frameon=False)
    fig_am.tight_layout()
    fig_am.subplots_adjust(bottom=0.22)
    save_fig(fig_am, "all_models_rolling_regime_dual_axis")

print("\nResultsGenerator_augmented_stable complete.")
