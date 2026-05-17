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
LATENT_DIM = 3
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
        fitted_color = plt.cm.tab10.colors[CCY_ORDER.index(ccy) % 10]

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
        fitted_color = plt.cm.tab10.colors[CCY_ORDER.index(ccy) % 10]

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

# load baseline dim=2 as reference (no augmentation)
_baseline_ckpt_path = os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                                   "dim2_baseline", f"ep{EPOCHS}",
                                   f"checkpoint_dim2_ep{EPOCHS}.pt")
_baseline_S_hat = None
if os.path.exists(_baseline_ckpt_path):
    _b_ckpt = torch.load(_baseline_ckpt_path, map_location=device)
    _b_cfg  = _b_ckpt["model_config"]
    _b_m    = FullModel(input_dim=_b_cfg.get("input_dim", X_tensor.shape[1]), latent_dim=_b_cfg["latent_dim"]).to(device)
    _b_m.load_state_dict(_b_ckpt["model_state_dict"])
    _b_m.eval()
    _b_list = []
    with torch.no_grad():
        for _i in range(0, X_tensor.shape[0], BATCH_SIZE):
            _xb = X_tensor[_i:_i + BATCH_SIZE].to(device)
            _b_list.append(_b_m(_xb).cpu())
    _baseline_S_hat = torch.cat(_b_list).numpy()
    print("  Loaded baseline dim=2 as reference")
else:
    print(f"  ⚠️  Baseline dim=2 checkpoint not found: {_baseline_ckpt_path}")

_rep_dates_ad = {
    "Calm (2014-08-29)": "2014-08-29",
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

        if _baseline_S_hat is not None:
            _b_fitted = _baseline_S_hat[_global_idx] * _scale_ad
            ax.plot(tenors, _b_fitted, color="black", linewidth=1.5,
                    linestyle="--", label="Baseline ($\\ell=2$)")

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

# ── Q8a-style: rolling regime dual-axis — augmented_input dims 2, 3, 4 ────────
import matplotlib.transforms

EVENTS = {
    "GFC\n(15 Sep 2008)":      "2008-09-15",
    "QE\n(22 Jan 2015)":       "2015-01-22",
    "COVID\n(1 Mar 2020)":     "2020-03-01",
    "Inflation\n(1 Mar 2022)": "2022-03-01",
}

_ROLL_DIMS_AUG  = [2, 3, 4]
_ROLL_SUBDIR_A  = "train5Y_test6M_step6M"
_ROLL_EPOCHS_A  = 3500
_DIM_COLORS_AUG = {2: custom_palette[4], 3: custom_palette[0], 4: custom_palette[6]}

def _aug_regime_counts(pred_df):
    """Return {test_start_str: {n_neg, n_inv}} from a predictions CSV.
    Derives regimes from actual tenor values: negative if any rate < 0,
    inverted if shortest tenor > longest tenor."""
    _tenors_aug = [1, 2, 3, 5, 10, 15, 20, 30]
    _act_cols   = [f"actual_tenor_{t}" for t in _tenors_aug
                   if f"actual_tenor_{t}" in pred_df.columns]
    _short_col  = f"actual_tenor_{_tenors_aug[0]}"
    _long_col   = f"actual_tenor_{_tenors_aug[-1]}"
    counts = {}
    for ts, grp in pred_df.groupby("test_start"):
        ts_str = str(ts)[:10]
        neg = int((grp[_act_cols].min(axis=1) < 0).sum()) if _act_cols else 0
        inv = int((grp[_short_col] > grp[_long_col]).sum()) if (_short_col in grp.columns and _long_col in grp.columns) else 0
        counts[ts_str] = {"n_neg": neg, "n_inv": inv}
    return counts

print("\nGenerating augmented rolling regime dual-axis figure...")

_aug_roll_dfs   = {}
_aug_train_cts  = None
_aug_test_cts   = None

for _d_r in _ROLL_DIMS_AUG:
    _roll_csv_a = os.path.join(
        REPO_ROOT, "Figures", "OOSResults", "Roll",
        f"OOS_roll_dim{_d_r}_augmented_input",
        _ROLL_SUBDIR_A, f"ep{_ROLL_EPOCHS_A}",
        f"oos_rolling_bbg_dim{_d_r}_train5Y_test6M_step6M.csv",
    )
    if os.path.exists(_roll_csv_a):
        _aug_roll_dfs[_d_r] = pd.read_csv(_roll_csv_a)
        for _col_r in ["test_start", "test_end", "train_start"]:
            _aug_roll_dfs[_d_r][_col_r] = pd.to_datetime(_aug_roll_dfs[_d_r][_col_r])
        # regime counts from first available dim
        if _aug_train_cts is None:
            _tr_a = os.path.join(
                REPO_ROOT, "Figures", "OOSResults", "Roll",
                f"OOS_roll_dim{_d_r}_augmented_input",
                _ROLL_SUBDIR_A, f"ep{_ROLL_EPOCHS_A}", "predictions_train_all.csv",
            )
            _te_a = os.path.join(
                REPO_ROOT, "Figures", "OOSResults", "Roll",
                f"OOS_roll_dim{_d_r}_augmented_input",
                _ROLL_SUBDIR_A, f"ep{_ROLL_EPOCHS_A}", "predictions_test_all.csv",
            )
            if os.path.exists(_tr_a) and os.path.exists(_te_a):
                _aug_train_cts = _aug_regime_counts(pd.read_csv(_tr_a))
                _aug_test_cts  = _aug_regime_counts(pd.read_csv(_te_a))
    else:
        print(f"  ⚠️  Rolling CSV not found for dim={_d_r} — skipping.")

if not _aug_roll_dfs:
    print("  ⚠️  No rolling CSVs found — skipping augmented regime figure.")
else:
    _aug_ref_df = next(iter(_aug_roll_dfs.values()))
    _aug_rows = []
    for _, _rw in _aug_ref_df.iterrows():
        _ts  = str(_rw["test_start"])[:10]
        _n_train = float(_rw["n_train"]) if "n_train" in _rw and np.isfinite(_rw["n_train"]) else np.nan
        _n_test  = float(_rw["n_test"])  if "n_test"  in _rw and np.isfinite(_rw["n_test"])  else np.nan
        _tc = _aug_train_cts.get(_ts, {}) if _aug_train_cts else {}
        _ec = _aug_test_cts.get(_ts,  {}) if _aug_test_cts  else {}
        _row_a = {
            "Window": f"{str(_rw['train_start'])[:7]} / {str(_rw['test_end'])[:7]}",
            "Train % Neg": round(100 * _tc.get("n_neg", np.nan) / _n_train, 1) if np.isfinite(_n_train) and _n_train > 0 else np.nan,
            "Train % Inv": round(100 * _tc.get("n_inv", np.nan) / _n_train, 1) if np.isfinite(_n_train) and _n_train > 0 else np.nan,
            "Test % Neg":  round(100 * _ec.get("n_neg", np.nan) / _n_test,  1) if np.isfinite(_n_test)  and _n_test  > 0 else np.nan,
            "Test % Inv":  round(100 * _ec.get("n_inv", np.nan) / _n_test,  1) if np.isfinite(_n_test)  and _n_test  > 0 else np.nan,
        }
        for _d_r, _rdf_r in _aug_roll_dfs.items():
            _drow_r = _rdf_r[_rdf_r["test_start"].dt.strftime("%Y-%m-%d") == _ts]
            _row_a[f"OOS_dim{_d_r}"] = round(float(_drow_r["avg_rmse_bps"].values[0]), 2) if len(_drow_r) else np.nan
        _aug_rows.append(_row_a)

    _aug_windows = [r["Window"].split(" / ")[1][:7] for r in _aug_rows]
    _aug_x       = np.arange(len(_aug_windows))
    _aug_neg     = np.array([r["Test % Neg"]  for r in _aug_rows], dtype=float)
    _aug_inv     = np.array([r["Test % Inv"]  for r in _aug_rows], dtype=float)
    _aug_tr_neg  = np.array([r["Train % Neg"] for r in _aug_rows], dtype=float)
    _aug_tr_inv  = np.array([r["Train % Inv"] for r in _aug_rows], dtype=float)

    _inv_col_a = custom_palette[5]

    fig_ra, ax_ra_a = plt.subplots(figsize=(12, 5))
    ax_ra_b = ax_ra_a.twinx()

    _w_ra = 0.2
    ax_ra_a.bar(_aug_x - 1.5*_w_ra, _aug_tr_neg, width=_w_ra, label="% Neg (Train)", color="slategrey", alpha=0.4)
    ax_ra_a.bar(_aug_x - 0.5*_w_ra, _aug_neg,    width=_w_ra, label="% Neg (Test)",  color="slategrey", alpha=0.9)
    ax_ra_a.bar(_aug_x + 0.5*_w_ra, _aug_tr_inv, width=_w_ra, label="% Inv (Train)", color=_inv_col_a,  alpha=0.5)
    ax_ra_a.bar(_aug_x + 1.5*_w_ra, _aug_inv,    width=_w_ra, label="% Inv (Test)",  color=_inv_col_a,  alpha=1.0)
    ax_ra_a.set_ylabel("% of curves in set", fontsize=11)
    ax_ra_a.set_xticks(_aug_x)
    ax_ra_a.set_xticklabels(_aug_windows, rotation=45, ha="right", fontsize=10)
    ax_ra_a.tick_params(axis="y", labelsize=10)

    for _di_ra, _d_ra in enumerate(sorted(_aug_roll_dfs.keys())):
        _oos_ra       = np.array([r.get(f"OOS_dim{_d_ra}", np.nan) for r in _aug_rows], dtype=float)
        _oos_clipped_ra = np.clip(_oos_ra, 0, 50)
        ax_ra_b.plot(_aug_x, _oos_clipped_ra, marker="o", markersize=4, linewidth=2.2,
                     label=f"OOS RMSE $\\ell={_d_ra}$", color=_DIM_COLORS_AUG[_d_ra])
        for _xi_ra, (_raw_ra, _clip_ra) in enumerate(zip(_oos_ra, _oos_clipped_ra)):
            if _raw_ra > 50:
                _on_left_ra = _aug_windows[_xi_ra].startswith("2022-06")
                ax_ra_b.annotate(
                    f"{_raw_ra:.0f}",
                    xy=(_aug_x[_xi_ra], 50),
                    xytext=(-5 if _on_left_ra else 5, 2 - _di_ra * 8),
                    textcoords="offset points",
                    ha="right" if _on_left_ra else "left",
                    va="top", fontsize=10, color=_DIM_COLORS_AUG[_d_ra],
                )

    ax_ra_b.set_ylabel("OOS RMSE (bps, clipped at 50)", fontsize=11)
    ax_ra_b.tick_params(axis="y", labelsize=10)

    # event markers
    _aug_win_ts = np.array([pd.Timestamp(w + "-01").value for w in _aug_windows], dtype=float)
    for _ev_lbl, _ev_date in EVENTS.items():
        _ev_ts = pd.Timestamp(_ev_date).value
        if _aug_win_ts[0] <= _ev_ts <= _aug_win_ts[-1]:
            _ev_x_ra = float(np.interp(_ev_ts, _aug_win_ts, _aug_x))
            ax_ra_a.axvline(_ev_x_ra, color="0.5", linewidth=1.0, linestyle="--", zorder=0)
            _ev_tr = matplotlib.transforms.blended_transform_factory(
                ax_ra_b.transData, ax_ra_b.transAxes)
            ax_ra_b.text(_ev_x_ra, 1.02, _ev_lbl, fontsize=9, ha="center", va="bottom",
                         color="0.4", transform=_ev_tr, clip_on=False)

    _lines_ra_a, _labs_ra_a = ax_ra_a.get_legend_handles_labels()
    _lines_ra_b, _labs_ra_b = ax_ra_b.get_legend_handles_labels()
    fig_ra.legend(_lines_ra_a + _lines_ra_b, _labs_ra_a + _labs_ra_b,
                  loc="lower center", bbox_to_anchor=(0.5, -0.08),
                  ncol=4, fontsize=10, frameon=False)
    fig_ra.tight_layout()
    fig_ra.subplots_adjust(bottom=0.22)
    save_fig(fig_ra, "augmented_rolling_regime_dual_axis")

print("\nResultsGenerator_augmented complete.")
