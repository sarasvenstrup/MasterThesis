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
from Code.model.full_model_stable import FullModel as FullModelStable

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

# ── figure: augmented_latent_space_regime ─────────────────────────────────────
print("\nGenerating augmented latent space regime figure...")
if _baseline_S_hat is not None:
    # Encode all in-sample curves with baseline dim=2 linear encoder
    _enc_w  = _b_m.encoder.lin.weight.detach().numpy()   # (2, 8)
    _Z_all  = X_tensor.numpy() @ _enc_w.T                # (N, 2)

    # Regime masks (reuse already-computed boolean arrays)
    _ls_normal   = ~inverted & ~negative
    _ls_inverted =  inverted & ~negative
    _ls_negative =  negative                              # superset: also catches neg+inv

    _col_ls_normal   = custom_palette[2]
    _col_ls_inverted = "black"
    _col_ls_negative = "indianred"

    fig_ls, ax_ls = plt.subplots(figsize=(10, 5))

    # Background scatter coloured by regime — z_2 on x-axis, z_1 on y-axis
    for _ls_mask, _ls_col, _ls_lbl, _ls_zorder in [
        (_ls_normal,   _col_ls_normal,   "Normal",   1),
        (_ls_inverted, _col_ls_inverted, "Inverted", 2),
        (_ls_negative, _col_ls_negative, "Negative", 3),
    ]:
        ax_ls.scatter(
            _Z_all[_ls_mask, 1], _Z_all[_ls_mask, 0],
            color=_ls_col, alpha=0.25, s=8,
            label=_ls_lbl, zorder=_ls_zorder, linewidths=0,
        )

    # Four specific overlay points — coords stored as (z_1, z_2), plot as (z_2, z_1)
    # (coord, marker, color, filled, legend_label)
    _ls_pts = [
        ((-0.0088, -0.0217), "*", _col_ls_normal,   True,
         r"$\mathbf{z}$ (Normal, EUR 2014-08-29)"),
        (( 0.0046, -0.0375), "*", _col_ls_normal,   False,
         r"$\mathbf{z}_{\mathrm{flat}}$ (flat counterpart)"),
        ((-0.0057,  0.0089), "D", _col_ls_negative, True,
         r"$\mathbf{z}^*$ (Negative, EUR 2020-03-31)"),
        ((-0.0004,  0.0032), "D", _col_ls_negative, False,
         r"$\mathbf{z}^*_{\mathrm{flat}}$ (flat counterpart)"),
    ]
    _ls_marker_sizes = {"*": 120, "D": 50}
    for _ls_coord, _ls_mk, _ls_col, _ls_filled, _ls_lbl in _ls_pts:
        _ls_fc = _ls_col if _ls_filled else "none"
        ax_ls.scatter(
            [_ls_coord[1]], [_ls_coord[0]],   # x=z_2, y=z_1
            marker=_ls_mk,
            facecolors=_ls_fc, edgecolors=_ls_col,
            s=_ls_marker_sizes[_ls_mk], linewidths=1.8,
            zorder=7, label=_ls_lbl,
        )

    # Line segments with Euclidean distance annotations — coords as (z_1, z_2)
    _ls_z           = np.array([-0.0088, -0.0217])
    _ls_z_flat      = np.array([ 0.0046, -0.0375])
    _ls_z_star      = np.array([-0.0057,  0.0089])
    _ls_z_flat_star = np.array([-0.0004,  0.0032])

    ax_ls.plot(
        [_ls_z[1], _ls_z_flat[1]], [_ls_z[0], _ls_z_flat[0]],   # x=z_2, y=z_1
        color=_col_ls_normal, linewidth=1.5, linestyle="--", zorder=5,
    )
    _ls_mid1 = (_ls_z + _ls_z_flat) / 2
    ax_ls.annotate(
        r"$d = 0.0208$",
        xy=(_ls_mid1[1], _ls_mid1[0]), xytext=(-6, 10), textcoords="offset points",
        fontsize=9, color=_col_ls_normal, ha="right",
    )

    ax_ls.plot(
        [_ls_z_star[1], _ls_z_flat_star[1]], [_ls_z_star[0], _ls_z_flat_star[0]],
        color=_col_ls_negative, linewidth=1.5, linestyle="--", zorder=5,
    )
    _ls_mid2 = (_ls_z_star + _ls_z_flat_star) / 2
    ax_ls.annotate(
        r"$d = 0.0078$",
        xy=(_ls_mid2[1], _ls_mid2[0]), xytext=(8, -16), textcoords="offset points",
        fontsize=9, color=_col_ls_negative, ha="left",
    )

    ax_ls.set_xlabel(r"$z_2$", fontsize=12)
    ax_ls.set_ylabel(r"$z_1$", fontsize=12)
    ax_ls.tick_params(labelsize=10)
    ax_ls.spines["top"].set_visible(False)
    ax_ls.spines["right"].set_visible(False)
    _leg = ax_ls.legend(fontsize=9, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    # Make the first 3 handles (Normal, Inverted, Negative) fully opaque and larger
    for _lh in _leg.legend_handles[:3]:
        _lh.set_alpha(1.0)
        _lh.set_sizes([40])
    fig_ls.tight_layout()
    fig_ls.subplots_adjust(right=0.72)
    save_fig(fig_ls, "augmented_latent_space_regime")
else:
    print("  ⚠️  Skipping augmented_latent_space_regime (no baseline dim=2 checkpoint found)")

# ── figure: augmented_latent_space_regime_shift ──────────────────────────────
print("\nGenerating augmented latent space regime shift figure...")
if _baseline_S_hat is not None:
    _enc_w_ns = _b_m.encoder.lin.weight.detach().numpy()   # (2, 8)
    _Z_ns     = X_tensor.numpy() @ _enc_w_ns.T             # (N, 2)

    _ns_normal   = ~inverted & ~negative
    _ns_inverted =  inverted & ~negative
    _ns_negative =  negative

    _col_ns_normal   = custom_palette[2]
    _col_ns_inverted = "black"
    _col_ns_negative = "indianred"

    _ns_dates = pd.to_datetime(meta["as_of_date"].values)
    _ns_ccys  = meta["ccy"].values
    _ns_X_np  = X_tensor.numpy()

    _ns_idx = np.where((_ns_ccys == "EUR") & (_ns_dates == pd.Timestamp("2014-08-29")))[0]

    if len(_ns_idx) == 0:
        print("  ⚠️  EUR 2014-08-29 not found — skipping augmented_latent_space_regime_shift")
    else:
        # ── stylised example points (hardcoded to match LaTeX example) ──────
        # Coordinates are (z_1, z_2); plot convention: x=z_2, y=z_1
        _ns_z  = _enc_w_ns @ _ns_X_np[_ns_idx[0]]   # computed from actual encoder
        _ns_z1 = np.array([-0.0164,  0.0400])         # −150 bps (Ex. 1)
        _ns_z2 = np.array([-0.0119,  0.0030])         # −60 bps  (Ex. 2)

        print(f"    z   = ({_ns_z[0]:.4f}, {_ns_z[1]:.4f})")
        print(f"    z1  = ({_ns_z1[0]:.4f}, {_ns_z1[1]:.4f})  [−150 bps Ex.1]")
        print(f"    z2  = ({_ns_z2[0]:.4f}, {_ns_z2[1]:.4f})  [−60 bps  Ex.2]")

        # ── actual-data failure masks ─────────────────────────────────────────
        _ns_X = _ns_X_np   # (N, 8)
        # ≥7 of 8 tenors negative
        _ns_mask_deep = (_ns_X < 0).sum(axis=1) >= 7
        # first 4 tenors negative AND last 4 tenors positive
        _ns_mask_cross = (
            (_ns_X[:, :4] < 0).all(axis=1) & (_ns_X[:, 4:] > 0).all(axis=1)
        )
        print(f"    deep-negative curves (≥7/8 neg): {_ns_mask_deep.sum()}")
        print(f"    crossing curves (first4<0, last4>0): {_ns_mask_cross.sum()}")

        fig_ns, ax_ns = plt.subplots(figsize=(11, 5))

        # ── background scatter coloured by regime (z_2 on x-axis, z_1 on y) ─
        for _ns_mask, _ns_col, _ns_lbl, _ns_zo in [
            (_ns_normal,   _col_ns_normal,   "Normal",   1),
            (_ns_inverted, _col_ns_inverted, "Inverted", 2),
            (_ns_negative, _col_ns_negative, "Negative", 3),
        ]:
            ax_ns.scatter(
                _Z_ns[_ns_mask, 1], _Z_ns[_ns_mask, 0],
                color=_ns_col, alpha=0.25, s=8,
                label=_ns_lbl, zorder=_ns_zo, linewidths=0,
            )

        # ── actual-data failure overlays ──────────────────────────────────────
        ax_ns.scatter(
            _Z_ns[_ns_mask_deep, 1], _Z_ns[_ns_mask_deep, 0],
            marker="x", color=_col_ns_negative, s=40, linewidths=1.5,
            zorder=5, label=r"$\geq\!7/8$ negative tenors",
        )
        ax_ns.scatter(
            _Z_ns[_ns_mask_cross, 1], _Z_ns[_ns_mask_cross, 0],
            marker="^", facecolors="none", edgecolors=_col_ns_negative,
            s=40, linewidths=1.5,
            zorder=5, label="Crossing (first 4 neg, last 4 pos)",
        )

        # ── stylised example points ───────────────────────────────────────────
        # z — solid blue star (original curve)
        ax_ns.scatter(
            [_ns_z[1]], [_ns_z[0]],
            marker="*", facecolors=_col_ns_normal, edgecolors=_col_ns_normal,
            s=220, linewidths=1.5, zorder=8,
            label=r"$\mathbf{z}$ (EUR 2014-08-29, original)",
        )
        # z1 — solid red diamond (−150 bps, Ex. 1)
        ax_ns.scatter(
            [_ns_z1[1]], [_ns_z1[0]],
            marker="D", facecolors=_col_ns_negative, edgecolors=_col_ns_negative,
            s=70, linewidths=1.5, zorder=8,
            label=r"$\mathbf{z}_1$ (Ex.\,1: $-150$ bps)",
        )
        # z2 — solid red circle (−60 bps, Ex. 2)
        ax_ns.scatter(
            [_ns_z2[1]], [_ns_z2[0]],
            marker="o", facecolors=_col_ns_negative, edgecolors=_col_ns_negative,
            s=60, linewidths=1.5, zorder=8,
            label=r"$\mathbf{z}_2$ (Ex.\,2: $-60$ bps)",
        )

        # ── dashed arrows z → z1 and z → z2 ─────────────────────────────────
        _arrow_kw = dict(
            arrowstyle="->",
            color=_col_ns_negative,
            lw=1.4,
            linestyle="dashed",
            connectionstyle="arc3,rad=0.0",
        )
        ax_ns.annotate(
            "", xy=(_ns_z1[1], _ns_z1[0]), xytext=(_ns_z[1], _ns_z[0]),
            arrowprops=_arrow_kw, zorder=7,
        )
        ax_ns.annotate(
            "", xy=(_ns_z2[1], _ns_z2[0]), xytext=(_ns_z[1], _ns_z[0]),
            arrowprops=_arrow_kw, zorder=7,
        )

        # Arrow labels at midpoints, offset to avoid the line
        _ns_mid1 = (_ns_z + _ns_z1) / 2
        ax_ns.text(
            _ns_mid1[1] + 0.003, _ns_mid1[0] - 0.001,
            r"$-150\,$bps (Ex.\,1)",
            fontsize=8, color=_col_ns_negative, ha="left", va="top",
        )
        _ns_mid2 = (_ns_z + _ns_z2) / 2
        ax_ns.text(
            _ns_mid2[1] + 0.003, _ns_mid2[0] + 0.001,
            r"$-60\,$bps (Ex.\,2)",
            fontsize=8, color=_col_ns_negative, ha="left", va="bottom",
        )

        ax_ns.set_xlabel(r"$z_2$", fontsize=12)
        ax_ns.set_ylabel(r"$z_1$", fontsize=12)
        ax_ns.tick_params(labelsize=10)
        ax_ns.spines["top"].set_visible(False)
        ax_ns.spines["right"].set_visible(False)

        _leg_ns = ax_ns.legend(fontsize=8.5, frameon=False,
                               loc="center left", bbox_to_anchor=(1.02, 0.5))
        # Make background scatter handles fully opaque and a bit larger
        for _lh_ns in _leg_ns.legend_handles[:3]:
            _lh_ns.set_alpha(1.0)
            _lh_ns.set_sizes([40])

        fig_ns.tight_layout()
        fig_ns.subplots_adjust(right=0.68)
        save_fig(fig_ns, "augmented_latent_space_regime_shift")
else:
    print("  ⚠️  Skipping augmented_latent_space_regime_shift (no baseline dim=2 checkpoint found)")

# ── figure: latent_space_comparison (baseline ℓ=2 vs augmented ℓ=3) ──────────
print("\nGenerating latent space comparison figure (baseline vs augmented)...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping latent_space_comparison (no baseline dim=2 checkpoint found)")
else:
    # Baseline encoder (2, 8): z_2 on x-axis, z_1 on y-axis
    _cmp_enc_b = _b_m.encoder.lin.weight.detach().numpy()    # (2, 8)
    _cmp_Z_b   = X_tensor.numpy() @ _cmp_enc_b.T             # (N, 2)

    # Augmented encoder (3, 11): also plot z_2 on x-axis, z_1 on y-axis
    _cmp_enc_a = model.encoder.lin.weight.detach().numpy()   # (3, 11)
    _cmp_X_aug = augment(X_tensor).numpy()                   # (N, 11)
    _cmp_Z_a   = _cmp_X_aug @ _cmp_enc_a.T                  # (N, 3)

    # Regime colours (shared)
    _cmp_col_normal   = custom_palette[2]
    _cmp_col_inverted = "black"
    _cmp_col_negative = "indianred"
    _cmp_normal   = ~inverted & ~negative
    _cmp_inverted =  inverted & ~negative
    _cmp_negative =  negative

    # Failure masks (based on original 8-tenor curves, same for both panels)
    _cmp_X_np        = X_tensor.numpy()
    _cmp_mask_deep   = (_cmp_X_np < 0).sum(axis=1) >= 7
    _cmp_mask_cross  = (
        (_cmp_X_np[:, :4] < 0).all(axis=1) & (_cmp_X_np[:, 4:] > 0).all(axis=1)
    )

    # Stylised example points for the left panel
    _cmp_idx = np.where(
        (meta["ccy"].values == "EUR") &
        (pd.to_datetime(meta["as_of_date"].values) == pd.Timestamp("2014-08-29"))
    )[0]
    _cmp_z  = _cmp_enc_b @ _cmp_X_np[_cmp_idx[0]]   # original, baseline encoder
    _cmp_z1 = np.array([-0.0164,  0.0400])            # −150 bps (Ex. 1)
    _cmp_z2 = np.array([-0.0119,  0.0030])            # −60 bps  (Ex. 2)

    fig_cmp, (ax_bl, ax_au) = plt.subplots(1, 2, figsize=(16, 5))

    for _ax, _Z, _title, _show_ex in [
        (ax_bl, _cmp_Z_b, r"Baseline ($\ell=2$)",  True),
        (ax_au, _cmp_Z_a, r"Augmented ($\ell=3$)", False),
    ]:
        # Background scatter
        for _mask, _col, _lbl, _zo in [
            (_cmp_normal,   _cmp_col_normal,   "Normal",   1),
            (_cmp_inverted, _cmp_col_inverted, "Inverted", 2),
            (_cmp_negative, _cmp_col_negative, "Negative", 3),
        ]:
            _ax.scatter(
                _Z[_mask, 1], _Z[_mask, 0],
                color=_col, alpha=0.25, s=8,
                label=_lbl, zorder=_zo, linewidths=0,
            )

        # Failure overlays
        _ax.scatter(
            _Z[_cmp_mask_deep, 1], _Z[_cmp_mask_deep, 0],
            marker="x", color=_cmp_col_negative, s=40, linewidths=1.5,
            zorder=5, label=r"$\geq\!7/8$ negative tenors",
        )
        _ax.scatter(
            _Z[_cmp_mask_cross, 1], _Z[_cmp_mask_cross, 0],
            marker="^", facecolors="none", edgecolors=_cmp_col_negative,
            s=40, linewidths=1.5,
            zorder=5, label="Crossing (first 4 neg, last 4 pos)",
        )

        if _show_ex:
            # z — solid blue star (original EUR 2014-08-29)
            _ax.scatter(
                [_cmp_z[1]], [_cmp_z[0]],
                marker="*", facecolors=_cmp_col_normal, edgecolors=_cmp_col_normal,
                s=220, linewidths=1.5, zorder=8,
                label=r"$\mathbf{z}$ (EUR 2014-08-29)",
            )
            # z1 — solid red diamond (−150 bps, Ex. 1)
            _ax.scatter(
                [_cmp_z1[1]], [_cmp_z1[0]],
                marker="D", facecolors=_cmp_col_negative, edgecolors=_cmp_col_negative,
                s=70, linewidths=1.5, zorder=8,
                label=r"$\mathbf{z}_1$ (Ex.\,1: $-150$ bps)",
            )
            # z2 — solid red circle (−60 bps, Ex. 2)
            _ax.scatter(
                [_cmp_z2[1]], [_cmp_z2[0]],
                marker="o", facecolors=_cmp_col_negative, edgecolors=_cmp_col_negative,
                s=60, linewidths=1.5, zorder=8,
                label=r"$\mathbf{z}_2$ (Ex.\,2: $-60$ bps)",
            )
            # Dashed arrows z → z1 and z → z2
            _cmp_arrow_kw = dict(
                arrowstyle="->", color=_cmp_col_negative, lw=1.4,
                linestyle="dashed", connectionstyle="arc3,rad=0.0",
            )
            _ax.annotate(
                "", xy=(_cmp_z1[1], _cmp_z1[0]), xytext=(_cmp_z[1], _cmp_z[0]),
                arrowprops=_cmp_arrow_kw, zorder=7,
            )
            _ax.annotate(
                "", xy=(_cmp_z2[1], _cmp_z2[0]), xytext=(_cmp_z[1], _cmp_z[0]),
                arrowprops=_cmp_arrow_kw, zorder=7,
            )
            _cmp_mid1 = (_cmp_z + _cmp_z1) / 2
            _ax.text(
                _cmp_mid1[1] + 0.003, _cmp_mid1[0] - 0.001,
                r"$-150\,$bps (Ex.\,1)", fontsize=8,
                color=_cmp_col_negative, ha="left", va="top",
            )
            _cmp_mid2 = (_cmp_z + _cmp_z2) / 2
            _ax.text(
                _cmp_mid2[1] + 0.003, _cmp_mid2[0] + 0.001,
                r"$-60\,$bps (Ex.\,2)", fontsize=8,
                color=_cmp_col_negative, ha="left", va="bottom",
            )

        _ax.set_xlabel(r"$z_2$", fontsize=12)
        _ax.set_ylabel(r"$z_1$", fontsize=12)
        _ax.set_title(_title, fontsize=12)
        _ax.tick_params(labelsize=10)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    # Shared legend: collect all handles from left panel (has the most entries)
    _cmp_handles, _cmp_labels = ax_bl.get_legend_handles_labels()
    _cmp_leg = fig_cmp.legend(
        _cmp_handles, _cmp_labels,
        fontsize=8.5, frameon=False,
        loc="center left", bbox_to_anchor=(1.0, 0.5),
    )
    # Make background scatter handles fully opaque
    for _lh in _cmp_leg.legend_handles[:3]:
        _lh.set_alpha(1.0)
        _lh.set_sizes([40])

    fig_cmp.tight_layout()
    fig_cmp.subplots_adjust(right=0.80)
    save_fig(fig_cmp, "latent_space_comparison")

# ── figure: augmented_latent_space_shift ──────────────────────────────────────
print("\nGenerating augmented latent space shift figure...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping augmented_latent_space_shift (no baseline dim=2 checkpoint found)")
else:
    # Encoder weights and full latent cloud
    _enc_w_sh = _b_m.encoder.lin.weight.detach().numpy()   # (2, 8)
    _Z_sh     = X_tensor.numpy() @ _enc_w_sh.T             # (N, 2)

    _sh_normal   = ~inverted & ~negative
    _sh_inverted =  inverted & ~negative
    _sh_negative =  negative

    _col_sh_normal   = custom_palette[2]
    _col_sh_inverted = "black"
    _col_sh_negative = "indianred"

    # ── Compute overlay points by encoding shifted swap rate curves ───────────
    _sh_dates = pd.to_datetime(meta["as_of_date"].values)
    _sh_ccys  = meta["ccy"].values
    _sh_X_np  = X_tensor.numpy()

    _sh_idx_z    = np.where((_sh_ccys == "EUR") &
                            (_sh_dates == pd.Timestamp("2014-08-29")))[0]
    _sh_idx_star = np.where((_sh_ccys == "EUR") &
                            (_sh_dates == pd.Timestamp("2020-03-31")))[0]

    if len(_sh_idx_z) == 0 or len(_sh_idx_star) == 0:
        print("  ⚠️  Could not find required EUR dates — skipping shift figure")
    else:
        _sh_S_eur  = _sh_X_np[_sh_idx_z[0]]          # (8,) reference normal curve
        _sh_S_down = _sh_S_eur - 0.005                # all tenors shifted -0.005
        _sh_S_up   = _sh_S_eur + 0.005                # all tenors shifted +0.005
        _sh_S_star = _sh_X_np[_sh_idx_star[0]]        # actual negative EUR curve

        # Encode: z = enc_w @ S  →  (z_1, z_2)
        _sh_z      = tuple(_enc_w_sh @ _sh_S_eur)
        _sh_z_down = tuple(_enc_w_sh @ _sh_S_down)
        _sh_z_up   = tuple(_enc_w_sh @ _sh_S_up)
        _sh_z_star = tuple(_enc_w_sh @ _sh_S_star)

        print(f"    z      = ({_sh_z[0]:.4f}, {_sh_z[1]:.4f})")
        print(f"    z_down = ({_sh_z_down[0]:.4f}, {_sh_z_down[1]:.4f})")
        print(f"    z_up   = ({_sh_z_up[0]:.4f}, {_sh_z_up[1]:.4f})")
        print(f"    z_star = ({_sh_z_star[0]:.4f}, {_sh_z_star[1]:.4f})")

        fig_sh, ax_sh = plt.subplots(figsize=(10, 5))

        # Background scatter — z_2 on x-axis, z_1 on y-axis
        for _sh_mask, _sh_col, _sh_lbl, _sh_zo in [
            (_sh_normal,   _col_sh_normal,   "Normal",   1),
            (_sh_inverted, _col_sh_inverted, "Inverted", 2),
            (_sh_negative, _col_sh_negative, "Negative", 3),
        ]:
            ax_sh.scatter(
                _Z_sh[_sh_mask, 1], _Z_sh[_sh_mask, 0],
                color=_sh_col, alpha=0.25, s=8,
                label=_sh_lbl, zorder=_sh_zo, linewidths=0,
            )

        # ── Overlay points ────────────────────────────────────────────────────
        _sh_star_s = 200   # large enough for +/- text to sit inside

        # z — reference normal curve (solid star, no symbol)
        ax_sh.scatter(
            [_sh_z[1]], [_sh_z[0]], marker="*",
            facecolors=_col_sh_normal, edgecolors=_col_sh_normal,
            s=_sh_star_s, linewidths=1.5, zorder=7,
            label=r"$\mathbf{z}$ (Normal, EUR 2014-08-29)",
        )

        # z_down — star with white "−" inside
        ax_sh.scatter(
            [_sh_z_down[1]], [_sh_z_down[0]], marker="*",
            facecolors=_col_sh_normal, edgecolors=_col_sh_normal,
            s=_sh_star_s, linewidths=1.5, zorder=7,
            label="_nolegend_",
        )
        ax_sh.text(
            _sh_z_down[1], _sh_z_down[0], r"$-$",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="white", zorder=8,
        )

        # z_up — star with white "+" inside
        ax_sh.scatter(
            [_sh_z_up[1]], [_sh_z_up[0]], marker="*",
            facecolors=_col_sh_normal, edgecolors=_col_sh_normal,
            s=_sh_star_s, linewidths=1.5, zorder=7,
            label="_nolegend_",
        )
        ax_sh.text(
            _sh_z_up[1], _sh_z_up[0], r"$+$",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="white", zorder=8,
        )

        # z_star — actual negative EUR curve reference (solid diamond)
        ax_sh.scatter(
            [_sh_z_star[1]], [_sh_z_star[0]], marker="D",
            facecolors=_col_sh_negative, edgecolors=_col_sh_negative,
            s=50, linewidths=1.5, zorder=7,
            label=r"$\mathbf{z}^*$ (Negative, EUR 2020-03-31)",
        )

        # ── Dashed arrows z → z_down and z → z_up ────────────────────────────
        _sh_arrow_kw = dict(arrowstyle="-|>", color=_col_sh_normal,
                            lw=1.4, linestyle="dashed", mutation_scale=10)

        ax_sh.annotate(
            "", xy=(_sh_z_down[1], _sh_z_down[0]),
            xytext=(_sh_z[1], _sh_z[0]),
            arrowprops=_sh_arrow_kw, zorder=5,
        )
        ax_sh.annotate(
            r"$\mathbf{z}_{-}$",
            xy=(_sh_z_down[1], _sh_z_down[0]),
            xytext=(6, -12), textcoords="offset points",
            fontsize=10, color=_col_sh_normal, ha="left",
        )

        ax_sh.annotate(
            "", xy=(_sh_z_up[1], _sh_z_up[0]),
            xytext=(_sh_z[1], _sh_z[0]),
            arrowprops=_sh_arrow_kw, zorder=5,
        )
        ax_sh.annotate(
            r"$\mathbf{z}_{+}$",
            xy=(_sh_z_up[1], _sh_z_up[0]),
            xytext=(-6, 10), textcoords="offset points",
            fontsize=10, color=_col_sh_normal, ha="right",
        )

        ax_sh.set_xlabel(r"$z_2$", fontsize=12)
        ax_sh.set_ylabel(r"$z_1$", fontsize=12)
        ax_sh.tick_params(labelsize=10)
        ax_sh.spines["top"].set_visible(False)
        ax_sh.spines["right"].set_visible(False)
        _leg_sh = ax_sh.legend(fontsize=9, frameon=False,
                               loc="center left", bbox_to_anchor=(1.02, 0.5))
        for _lh_sh in _leg_sh.legend_handles[:3]:
            _lh_sh.set_alpha(1.0)
            _lh_sh.set_sizes([40])
        fig_sh.tight_layout()
        fig_sh.subplots_adjust(right=0.72)
        save_fig(fig_sh, "augmented_latent_space_shift")

# ── helper: 3×6 actual-vs-fit grid for a given index set ─────────────────────
def _plot_fit_grid(indices, regime_label, fname, seed=42):
    N_PLOTS = 18
    if len(indices) < N_PLOTS:
        print(f"  ⚠️  Only {len(indices)} curves for '{regime_label}' — skipping")
        return
    _rng = np.random.default_rng(seed=seed)
    _sel = _rng.choice(indices, size=N_PLOTS, replace=False)
    _sel = _sel[np.argsort(_sel)]

    fig_g, axes_g = plt.subplots(3, 6, figsize=(18, 9), sharey=False)

    for _pi, _gi in enumerate(_sel):
        _ax   = axes_g[_pi // 6, _pi % 6]
        _act  = X_np_all[_gi] * 10_000.0
        _fit  = _baseline_S_hat[_gi] * 10_000.0
        _ccy  = meta["ccy"].values[_gi]
        _date = pd.to_datetime(meta["as_of_date"].values[_gi]).strftime("%Y-%m-%d")
        _rmse = float(np.sqrt(np.mean((_act - _fit) ** 2)))

        _ax.plot(tenors, _act, "o-",  color="black",           linewidth=1.5,
                 markersize=3, label="Actual")
        _ax.plot(tenors, _fit, "s--", color=custom_palette[2], linewidth=1.5,
                 markersize=3, label=r"Baseline ($\ell=2$)")
        _ax.axhline(0, color="0.7", linewidth=0.8, linestyle=":")
        _ax.set_title(f"{_ccy}  {_date}", fontsize=8)
        _ax.text(0.97, 0.97, f"RMSE = {_rmse:.1f} bps",
                 transform=_ax.transAxes, fontsize=7,
                 ha="right", va="top", color="0.4")
        _ax.tick_params(labelsize=7)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        if _pi % 6 == 0:
            _ax.set_ylabel("Swap rate (bps)", fontsize=7)
        if _pi // 6 == 2:
            _ax.set_xlabel("Tenor (years)", fontsize=7)

    _handles, _labels = axes_g[0, 0].get_legend_handles_labels()
    fig_g.legend(_handles, _labels, loc="lower center",
                 bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9, frameon=False)
    fig_g.suptitle(
        f"Baseline model ($\\ell=2$): actual vs fit — {regime_label} curves",
        fontsize=11, y=1.01,
    )
    fig_g.tight_layout()
    save_fig(fig_g, fname)


# ── figure: baseline fit — negative curves (3×6) ─────────────────────────────
print("\nGenerating baseline fit on negative curves figure...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping — no baseline dim=2 checkpoint found")
else:
    _neg_indices = np.where(negative)[0]
    _plot_fit_grid(_neg_indices, "negative rate", "baseline_fit_negative_curves")

# ── figure: baseline fit — normal curves (3×6) ───────────────────────────────
print("\nGenerating baseline fit on normal curves figure...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping — no baseline dim=2 checkpoint found")
else:
    _norm_indices = np.where(~inverted & ~negative)[0]
    _plot_fit_grid(_norm_indices, "normal", "baseline_fit_normal_curves")

# ── figure: baseline fit — deeply negative curves (3×6) ──────────────────────
# "Deeply negative" = mean swap rate <= 0, i.e. the curve is centred at or
# below zero (approximately symmetric around zero or more negative than that).
print("\nGenerating baseline fit on deeply negative curves figure...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping — no baseline dim=2 checkpoint found")
else:
    _deep_neg_mask    = (X_np_all.mean(axis=1) <= 0) & negative
    _deep_neg_indices = np.where(_deep_neg_mask)[0]
    print(f"  Found {len(_deep_neg_indices)} deeply negative curves "
          f"(mean swap rate <= 0)")
    _plot_fit_grid(
        _deep_neg_indices,
        "deeply negative (mean swap rate $\\leq 0$)",
        "baseline_fit_deep_negative_curves",
    )

# ── figure: almost-all-negative curves — all three models (3×6) ─────────────
print("\nGenerating almost-all-negative curves figure...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping — no baseline dim=2 checkpoint found")
else:
    # Load stable dim=4 if not already loaded
    _aan_stable_path = os.path.join(
        REPO_ROOT, "Figures", "TrainingResults",
        "dim4_stable", f"ep{EPOCHS}",
        f"checkpoint_dim4_ep{EPOCHS}.pt",
    )
    _aan_stable_S_hat = None
    if os.path.exists(_aan_stable_path):
        _aan_ckpt  = torch.load(_aan_stable_path, map_location=device,
                                weights_only=False)
        _aan_state = (_aan_ckpt["model_state_dict"]
                      if "model_state_dict" in _aan_ckpt else _aan_ckpt)
        _aan_cfg   = (_aan_ckpt.get("model_config", {})
                      if isinstance(_aan_ckpt, dict) else {})
        _aan_ldim  = _aan_cfg.get("latent_dim", 4)
        _aan_m     = FullModelStable(latent_dim=_aan_ldim).to(device)
        _aan_m.load_state_dict(_aan_state, strict=False)
        _aan_m.eval()
        _aan_list  = []
        with torch.no_grad():
            for _i in range(0, X_tensor.shape[0], BATCH_SIZE):
                _xb = X_tensor[_i:_i + BATCH_SIZE].to(device)
                _aan_list.append(_aan_m(_xb).cpu())
        _aan_stable_S_hat = torch.cat(_aan_list).numpy()
        print("  Loaded stable dim=4 model for almost-all-negative figure")
    else:
        print(f"  ⚠️  Stable dim=4 checkpoint not found: {_aan_stable_path}")

    # Group 1: at least 7 of 8 tenors are negative
    _aan_neg_count  = (X_np_all < 0).sum(axis=1)
    _aan_mask_deep  = _aan_neg_count >= 7
    _aan_idx_deep   = np.where(_aan_mask_deep)[0]
    print(f"  Found {len(_aan_idx_deep)} curves with ≥7/8 negative tenors")

    # Group 2: crossing curves — first 4 tenors all negative, last tenor positive,
    #           not already in group 1
    _aan_mask_sym = (
        (X_np_all[:, :5] < 0).all(axis=1) & (X_np_all[:, -1] > 0) & ~_aan_mask_deep
    )
    _aan_idx_sym = np.where(_aan_mask_sym)[0]
    print(f"  Found {len(_aan_idx_sym)} crossing curves (short-neg, long-pos)")

    N_AAN  = 18
    n_deep = len(_aan_idx_deep)
    n_fill = N_AAN - n_deep
    _aan_rng = np.random.default_rng(seed=42)
    if len(_aan_idx_sym) >= n_fill:
        _aan_idx_sym_sel = _aan_rng.choice(_aan_idx_sym, size=n_fill, replace=False)
    else:
        _aan_idx_sym_sel = _aan_idx_sym
        print(f"  ⚠️  Only {len(_aan_idx_sym)} crossing curves — using all")

    _aan_sel = np.concatenate([_aan_idx_deep, _aan_idx_sym_sel])
    _aan_sel = _aan_sel[np.argsort(_aan_sel)]
    print(f"  Total selected: {len(_aan_sel)} ({n_deep} deeply negative + "
          f"{len(_aan_idx_sym_sel)} crossing)")

    fig_aan, axes_aan = plt.subplots(3, 6, figsize=(18, 9), sharey=False)

    for _pi, _gi in enumerate(_aan_sel):
        _ax    = axes_aan[_pi // 6, _pi % 6]
        _act   = X_np_all[_gi] * 10_000.0
        _fit_b = _baseline_S_hat[_gi] * 10_000.0
        _fit_a = S_np_all[_gi] * 10_000.0
        _ccy   = meta["ccy"].values[_gi]
        _date  = pd.to_datetime(meta["as_of_date"].values[_gi]).strftime("%Y-%m-%d")
        _rmse_b = float(np.sqrt(np.mean((_act - _fit_b) ** 2)))
        _rmse_a = float(np.sqrt(np.mean((_act - _fit_a) ** 2)))

        _ax.plot(tenors, _act,   "o-", color="black",         linewidth=1.5,
                 markersize=3, label="Actual")
        _ax.plot(tenors, _fit_b,       color="#2c4f8c",       linewidth=1.5,
                 label=r"Baseline ($\ell=2$)")
        _ax.plot(tenors, _fit_a,       color="palevioletred", linewidth=1.5,
                 label=r"Augmented ($\ell=3$)")

        if _aan_stable_S_hat is not None:
            _fit_s  = _aan_stable_S_hat[_gi] * 10_000.0
            _rmse_s = float(np.sqrt(np.mean((_act - _fit_s) ** 2)))
            _ax.plot(tenors, _fit_s,   color="#c0392b",       linewidth=1.5,
                     label=r"Stable ($\ell=4$)")
            _rmse_txt = f"B:{_rmse_b:.1f} / Aug:{_rmse_a:.1f} / S:{_rmse_s:.1f} bps"
        else:
            _rmse_txt = f"B:{_rmse_b:.1f} / Aug:{_rmse_a:.1f} bps"

        _ax.axhline(0, color="0.7", linewidth=0.8, linestyle=":")
        _ax.set_title(f"{_ccy}  {_date}", fontsize=8)
        _ax.text(0.97, 0.97, _rmse_txt,
                 transform=_ax.transAxes, fontsize=6.5,
                 ha="right", va="top", color="0.4")
        _ax.tick_params(labelsize=7)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        if _pi % 6 == 0:
            _ax.set_ylabel("Swap rate (bps)", fontsize=7)
        if _pi // 6 == 2:
            _ax.set_xlabel("Tenor (years)", fontsize=7)

    _handles_aan, _labels_aan = axes_aan[0, 0].get_legend_handles_labels()
    fig_aan.legend(_handles_aan, _labels_aan, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9, frameon=False)
    fig_aan.suptitle(
        r"Deeply negative and crossing curves: "
        r"baseline ($\ell=2$), augmented input ($\ell=3$), stable ($\ell=4$)"
        "\n"
        r"($\geq 7/8$ negative tenors, or first 5 tenors negative with positive long end)",
        fontsize=10, y=1.02,
    )
    fig_aan.tight_layout()
    save_fig(fig_aan, "baseline_fit_all_negative_curves")

# ── figure: two failure modes — baseline only (GridSpec 2×2) ─────────────────
print("\nGenerating two failure modes figure (baseline only)...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping — no baseline dim=2 checkpoint found")
else:
    _b_rmse_all = np.sqrt(np.mean((X_np_all - _baseline_S_hat) ** 2, axis=1)) * 10_000

    # ── curve indices ─────────────────────────────────────────────────────────
    # Panel 1 (top-left): worst deeply negative curve (≥7/8 tenors < 0)
    _fm_mask_deep = (X_np_all < 0).sum(axis=1) >= 7
    _fm_idx_deep  = np.where(_fm_mask_deep)[0]
    _fm_gi_deep   = _fm_idx_deep[np.argmax(_b_rmse_all[_fm_idx_deep])]

    # Panel 1 (left): EUR 2015-03-31 — same shape as deeply negative, all positive
    _fm_eur15_idx = np.where(
        (meta["ccy"].values == "EUR") &
        (pd.to_datetime(meta["as_of_date"].values) == pd.Timestamp("2015-03-31"))
    )[0]
    if len(_fm_eur15_idx) == 0:
        print("  ⚠️  EUR 2015-03-31 not found in dataset")
        _fm_gi_eur15 = None
    else:
        _fm_gi_eur15 = _fm_eur15_idx[0]

    # Panel 3 (right column): JPY 2016-09-30 — crossing failure
    _fm_mask_cross = (
        (X_np_all[:, :5] < 0).all(axis=1) & (X_np_all[:, -1] > 0) & ~_fm_mask_deep
    )
    _fm_jpy_idx = np.where(
        (meta["ccy"].values == "JPY") &
        (pd.to_datetime(meta["as_of_date"].values) == pd.Timestamp("2016-09-30"))
    )[0]
    if len(_fm_jpy_idx) == 0:
        print("  ⚠️  JPY 2016-09-30 not found — falling back to worst crossing curve")
        _fm_idx_cross = np.where(_fm_mask_cross)[0]
        _fm_gi_cross  = _fm_idx_cross[np.argmax(_b_rmse_all[_fm_idx_cross])]
    else:
        _fm_gi_cross = _fm_jpy_idx[0]

    # ── layout: 1 row × 3 columns ─────────────────────────────────────────────
    fig_fm, (ax_fm1, ax_fm2, ax_fm3) = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    def _draw_fm_panel(ax, gi, show_ylabel, rmse_y, rmse_va):
        _act      = X_np_all[gi] * 100.0
        _fit_b    = _baseline_S_hat[gi] * 100.0
        _rmse_bps = float(np.sqrt(np.mean(
            (X_np_all[gi] - _baseline_S_hat[gi]) ** 2
        ))) * 10_000
        _ccy  = meta["ccy"].values[gi]
        _date = pd.to_datetime(meta["as_of_date"].values[gi]).strftime("%Y-%m-%d")

        ax.plot(tenors, _act,   "o-", color="black",   linewidth=1.5,
                markersize=4, label="Actual")
        ax.plot(tenors, _fit_b,       color="#2c4f8c", linewidth=1.5,
                label=r"Baseline ($\ell=2$)")
        ax.axhline(0, color="0.7", linewidth=0.8, linestyle=":")
        ax.set_title(f"{_ccy}  {_date}", fontsize=9)
        ax.text(0.97, rmse_y, f"RMSE: {_rmse_bps:.1f} bps",
                transform=ax.transAxes, fontsize=8.5,
                ha="right", va=rmse_va, color="0.4")
        ax.set_xlabel("Maturity", fontsize=9)
        if show_ylabel:
            ax.set_ylabel("Swap rate (%)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if _fm_gi_eur15 is not None:
        _draw_fm_panel(ax_fm1, _fm_gi_eur15, show_ylabel=True,  rmse_y=0.97, rmse_va="top")
    _draw_fm_panel(ax_fm2, _fm_gi_deep,  show_ylabel=False, rmse_y=0.97, rmse_va="top")
    _draw_fm_panel(ax_fm3, _fm_gi_cross, show_ylabel=False, rmse_y=0.03, rmse_va="bottom")

    _handles_fm, _labels_fm = ax_fm1.get_legend_handles_labels()
    fig_fm.legend(_handles_fm, _labels_fm, loc="lower center",
                  bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False)
    fig_fm.tight_layout()
    save_fig(fig_fm, "baseline_fit_failure_modes")

# ── figure: worst curves — baseline dim=2 + stable dim=4 (3×6) ──────────────
print("\nGenerating worst curves figure (baseline dim=2 + stable dim=4)...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping — no baseline dim=2 checkpoint found")
else:
    # Load stable dim=4 model
    _stable_ckpt_path = os.path.join(
        REPO_ROOT, "Figures", "TrainingResults",
        "dim4_stable", f"ep{EPOCHS}",
        f"checkpoint_dim4_ep{EPOCHS}.pt",
    )
    _stable_S_hat = None
    if os.path.exists(_stable_ckpt_path):
        _st_ckpt = torch.load(_stable_ckpt_path, map_location=device,
                              weights_only=False)
        _st_state = (_st_ckpt["model_state_dict"]
                     if "model_state_dict" in _st_ckpt else _st_ckpt)
        _st_cfg   = (_st_ckpt.get("model_config", {})
                     if isinstance(_st_ckpt, dict) else {})
        _st_ldim  = _st_cfg.get("latent_dim", 4)
        _st_idim  = _st_cfg.get("input_dim",  X_tensor.shape[1])
        _st_m     = FullModelStable(latent_dim=_st_ldim).to(device)
        _st_m.load_state_dict(_st_state, strict=False)
        _st_m.eval()
        _st_list  = []
        with torch.no_grad():
            for _i in range(0, X_tensor.shape[0], BATCH_SIZE):
                _xb = X_tensor[_i:_i + BATCH_SIZE].to(device)
                _st_list.append(_st_m(_xb).cpu())
        _stable_S_hat = torch.cat(_st_list).numpy()
        print("  Loaded stable dim=4 model")
    else:
        print(f"  ⚠️  Stable dim=4 checkpoint not found: {_stable_ckpt_path}")

    # Select 18 worst by baseline RMSE
    _b_rmse    = np.sqrt(np.mean((X_np_all - _baseline_S_hat) ** 2, axis=1)) * 10_000
    _worst_idx = np.argsort(_b_rmse)[::-1][:18]
    _worst_idx = _worst_idx[np.argsort(_worst_idx)]   # sort by dataset index
    print(f"  Worst RMSE range: {_b_rmse[np.argsort(_b_rmse)[::-1][0]]:.1f} – "
          f"{_b_rmse[np.argsort(_b_rmse)[::-1][17]]:.1f} bps")

    fig_wc, axes_wc = plt.subplots(3, 6, figsize=(18, 9), sharey=False)

    for _pi, _gi in enumerate(_worst_idx):
        _ax   = axes_wc[_pi // 6, _pi % 6]
        _act  = X_np_all[_gi] * 10_000.0
        _fit_b = _baseline_S_hat[_gi] * 10_000.0
        _ccy  = meta["ccy"].values[_gi]
        _date = pd.to_datetime(meta["as_of_date"].values[_gi]).strftime("%Y-%m-%d")
        _rmse_b = float(np.sqrt(np.mean((_act - _fit_b) ** 2)))

        _fit_a  = S_np_all[_gi] * 10_000.0
        _rmse_a = float(np.sqrt(np.mean((_act - _fit_a) ** 2)))

        _ax.plot(tenors, _act,   "o-", color="black",         linewidth=1.5,
                 markersize=3, label="Actual")
        _ax.plot(tenors, _fit_b,       color="#2c4f8c",      linewidth=1.5,
                 label=r"Baseline ($\ell=2$)")
        _ax.plot(tenors, _fit_a,       color="palevioletred", linewidth=1.5,
                 label=r"Aug. + Stable ($\ell=3$)")

        if _stable_S_hat is not None:
            _fit_s  = _stable_S_hat[_gi] * 10_000.0
            _rmse_s = float(np.sqrt(np.mean((_act - _fit_s) ** 2)))
            _ax.plot(tenors, _fit_s,   color="#c0392b",      linewidth=1.5,
                     label=r"Stable ($\ell=4$)")
            _rmse_txt = (f"B:{_rmse_b:.1f} / A:{_rmse_a:.1f} / "
                         f"S:{_rmse_s:.1f} bps")
        else:
            _rmse_txt = f"B:{_rmse_b:.1f} / A:{_rmse_a:.1f} bps"

        _ax.axhline(0, color="0.7", linewidth=0.8, linestyle=":")
        _ax.set_title(f"{_ccy}  {_date}", fontsize=8)
        _ax.text(0.97, 0.97, _rmse_txt,
                 transform=_ax.transAxes, fontsize=6.5,
                 ha="right", va="top", color="0.4")
        _ax.tick_params(labelsize=7)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        if _pi % 6 == 0:
            _ax.set_ylabel("Swap rate (bps)", fontsize=7)
        if _pi // 6 == 2:
            _ax.set_xlabel("Tenor (years)", fontsize=7)

    _handles_wc, _labels_wc = axes_wc[0, 0].get_legend_handles_labels()
    fig_wc.legend(_handles_wc, _labels_wc, loc="lower center",
                  bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9, frameon=False)
    fig_wc.suptitle(
        r"Worst-fit curves: baseline ($\ell=2$), augmented ($\ell=3$), stable ($\ell=4$)"
        r" — ranked by baseline RMSE",
        fontsize=11, y=1.01,
    )
    fig_wc.tight_layout()
    save_fig(fig_wc, "baseline_fit_worst_curves")

# ── figure: failure modes — baseline + augmented dim 2/3/4 (1×3) ─────────────
print("\nGenerating failure modes (augmented dims) figure...")
if _baseline_S_hat is None:
    print("  ⚠️  Skipping — no baseline dim=2 checkpoint found")
else:
    # Load augmented dim=2 and dim=4 (dim=3 is already S_np_all)
    _fma_aug_hats = {3: S_np_all}   # dim → reconstructions (N, 8)
    for _fma_dim in [2, 4]:
        _fma_ckpt_path = os.path.join(
            REPO_ROOT, "Figures", "TrainingResults",
            f"dim{_fma_dim}_augmented_input", f"ep{EPOCHS}",
            f"checkpoint_dim{_fma_dim}_ep{EPOCHS}.pt",
        )
        if not os.path.exists(_fma_ckpt_path):
            print(f"  ⚠️  Augmented dim={_fma_dim} checkpoint not found: {_fma_ckpt_path}")
            continue
        _fma_ckpt  = torch.load(_fma_ckpt_path, map_location=device, weights_only=False)
        _fma_state = (_fma_ckpt["model_state_dict"]
                      if "model_state_dict" in _fma_ckpt else _fma_ckpt)
        _fma_cfg   = (_fma_ckpt.get("model_config", {})
                      if isinstance(_fma_ckpt, dict) else {})
        _fma_idim  = _fma_cfg.get("input_dim", augment(X_tensor).shape[1])
        _fma_m     = FullModel(input_dim=_fma_idim, latent_dim=_fma_dim).to(device)
        _fma_m.load_state_dict(_fma_state, strict=False)
        _fma_m.eval()
        _fma_list  = []
        with torch.no_grad():
            for _i in range(0, X_tensor.shape[0], BATCH_SIZE):
                _xb_aug = augment(X_tensor[_i:_i + BATCH_SIZE].to(device))
                _fma_list.append(_fma_m(_xb_aug).cpu())
        _fma_aug_hats[_fma_dim] = torch.cat(_fma_list).numpy()
        print(f"  Loaded augmented dim={_fma_dim}")

    # Colors matching DIM_COLORS in ResultsGeneratorBaseline.py
    # DIM_COLORS = {2: custom_palette[4], 3: custom_palette[0], 4: custom_palette[6]}
    _fma_colors = {2: custom_palette[4], 3: custom_palette[0], 4: custom_palette[6]}
    _fma_labels = {2: r"Augmented ($\ell=2$)",
                   3: r"Augmented ($\ell=3$)",
                   4: r"Augmented ($\ell=4$)"}

    fig_fma, (ax_fma1, ax_fma2, ax_fma3) = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    def _draw_fma_panel(ax, gi, show_ylabel, rmse_y, rmse_va):
        _act   = X_np_all[gi] * 100.0
        _fit_b = _baseline_S_hat[gi] * 100.0
        _rmse_b = float(np.sqrt(np.mean((X_np_all[gi] - _baseline_S_hat[gi]) ** 2))) * 10_000
        _ccy   = meta["ccy"].values[gi]
        _date  = pd.to_datetime(meta["as_of_date"].values[gi]).strftime("%Y-%m-%d")

        ax.plot(tenors, _act,   "o-", color="black", linewidth=1.5,
                markersize=4, label="Actual")
        ax.plot(tenors, _fit_b,       color="black",  linewidth=1.5,
                linestyle="--", label=r"Baseline ($\ell=2$)")

        _rmse_parts = [f"B:{_rmse_b:.1f}"]
        for _d in [2, 3, 4]:
            if _d not in _fma_aug_hats:
                continue
            _fit_d  = _fma_aug_hats[_d][gi] * 100.0
            _rmse_d = float(np.sqrt(np.mean((X_np_all[gi] - _fma_aug_hats[_d][gi]) ** 2))) * 10_000
            ax.plot(tenors, _fit_d, color=_fma_colors[_d], linewidth=1.5,
                    label=_fma_labels[_d])
            _rmse_parts.append(f"A{_d}:{_rmse_d:.1f}")

        _rmse_txt = " / ".join(_rmse_parts) + " bps"
        ax.axhline(0, color="0.7", linewidth=0.8, linestyle=":")
        ax.set_title(f"{_ccy}  {_date}", fontsize=9)
        ax.text(0.97, rmse_y, _rmse_txt,
                transform=ax.transAxes, fontsize=7.5,
                ha="right", va=rmse_va, color="0.4")
        ax.set_xlabel("Maturity", fontsize=9)
        if show_ylabel:
            ax.set_ylabel("Swap rate (%)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if _fm_gi_eur15 is not None:
        _draw_fma_panel(ax_fma1, _fm_gi_eur15, show_ylabel=True,  rmse_y=0.97, rmse_va="top")
    _draw_fma_panel(ax_fma2, _fm_gi_deep,  show_ylabel=False, rmse_y=0.97, rmse_va="top")
    _draw_fma_panel(ax_fma3, _fm_gi_cross, show_ylabel=False, rmse_y=0.03, rmse_va="bottom")

    _handles_fma, _labels_fma = ax_fma1.get_legend_handles_labels()
    fig_fma.legend(_handles_fma, _labels_fma, loc="lower center",
                   bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=9, frameon=False)
    fig_fma.tight_layout()
    save_fig(fig_fma, "failure_modes_all_models")

print("\nResultsGenerator_augmented complete.")
