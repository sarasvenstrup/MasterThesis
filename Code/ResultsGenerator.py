# ResultsGenerator.py
# Generates all thesis result figures and tables from existing checkpoints and CSVs.
# Run from repo root: python Code/ResultsGenerator.py
#
# Outputs:
#   Figures/thesis_results/   → all .png figures
#   Tables/                   → all .csv tables
#
# Dependencies:
#   - OOS_split_dim{1,2,3,4}/ep2500/rmse_summary.csv       (Q1, Q2, Q3)
#   - OOS_split_dim3/ep2500/checkpoint_seed*.pt             (Q1, Q5, Q6)
#   - OOS_split_dim3/ep2500/run_manifest.json               (Q3 seeds table)
#   - kalman_benchmark_oos/ekf_dns_{1,2,3,4}f/rmse_summary.csv (Q4)
#   - OOS_roll_dim{1,2,3,4}/train3Y_test3M_step6M/ep2500/   (Q2, Q3, Q4)
#     └─ rolling CSVs for dim1/dim2 may still be running — those sections
#        are skipped gracefully with a warning if not yet available.

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

# ── path setup ─────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(REPO_ROOT)   # go up from Code/ to repo root
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS, set_paper_theme
from Code.model.full_model import FullModel

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
device = torch.device("cpu")   # inference only — CPU is fine

# ── output directories ─────────────────────────────────────────────────────────
THESIS_RESULTS = os.path.join(REPO_ROOT, "Figures", "thesis_results")
FIGURES_OUT    = os.path.join(THESIS_RESULTS, "AutoencoderPerformance")
TABLES_OUT     = os.path.join(THESIS_RESULTS, "AutoencoderPerformance")
PARAMS_DIR     = os.path.join(THESIS_RESULTS, "parameters")
os.makedirs(FIGURES_OUT, exist_ok=True)
os.makedirs(PARAMS_DIR,  exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
CCY_ORDER   = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
DIM_COLORS  = {1: custom_palette[0], 2: custom_palette[1],
               3: custom_palette[2], 4: custom_palette[3]}
LATENT_DIM       = 3
SPLIT_EPOCHS     = 2500
TRAIN_LOG_EPOCHS = 5000
KALMAN_DIMS      = [1, 2, 3, 4]
ALL_DIMS_PARAM   = [1, 2, 3, 4]

# Key market event dates for annotation
EVENTS = {
    "GFC\n(15 Sep 2008)":      "2008-09-15",
    "ECB QE\n(22 Jan 2015)":   "2015-01-22",
    "COVID\n(1 Mar 2020)":     "2020-03-01",
    "Rate hikes\n(1 Mar 2022)": "2022-03-01",
}

# ── apply paper theme ──────────────────────────────────────────────────────────
set_paper_theme()
currency_color_map = {ccy: custom_palette[i % len(custom_palette)]
                      for i, ccy in enumerate(CCY_ORDER)}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP: load data and best checkpoint
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = \
    my_data(use="bbg")

# Use full dataset for IS analysis, split for OOS
X_full   = X_tensor_full.float()
meta_full_df = meta_full.copy()
meta_full_df["as_of_date"] = pd.to_datetime(meta_full_df["as_of_date"])

TRAIN_MASK = (meta_full_df["as_of_date"] >= "2004-01-01") & \
             (meta_full_df["as_of_date"] <= "2020-12-31")
TEST_MASK  = (meta_full_df["as_of_date"] >= "2021-01-01") & \
             (meta_full_df["as_of_date"] <= "2022-12-31")

X_train    = X_full[TRAIN_MASK.values]
X_test     = X_full[TEST_MASK.values]
meta_train = meta_full_df.loc[TRAIN_MASK.values].reset_index(drop=True)
meta_test  = meta_full_df.loc[TEST_MASK.values].reset_index(drop=True)

TENOR_COLS = list(TARGET_TENORS)   # [1, 2, 3, 5, 10, 15, 20, 30]

# ── model loading helpers ─────────────────────────────────────────────────────
CKPT_DIR      = os.path.join(REPO_ROOT, "Figures", f"OOS_split_dim{LATENT_DIM}",
                              f"ep{SPLIT_EPOCHS}")
MANIFEST_PATH = os.path.join(CKPT_DIR, "run_manifest.json")

def _load_state_dict_compat(model, ckpt_path):
    """Load state dict, remapping old KMu keys (K.lin.*) to new (K.V / K.N)."""
    state = torch.load(ckpt_path, map_location=device)
    if any(k.startswith("K.lin") for k in state):
        warnings.warn(f"Remapping old KMu keys (K.lin → K.V/N) in {ckpt_path}")
        new_state = {}
        for k, v in state.items():
            if k == "K.lin.weight":
                new_state["K.V"] = v        # same shape (d, d)
            elif k == "K.lin.bias":
                new_state["K.N"] = v        # same shape (d,)
            else:
                new_state[k] = v
        state = new_state
    model.load_state_dict(state, strict=True)

def load_ep5000_model(dim):
    """Load ep5000 training checkpoint from Figures/dim{N}/ep5000/.
    Falls back to OOSSplit best-seed checkpoint if ep5000 checkpoint not yet available."""
    ckpt_path = os.path.join(REPO_ROOT, "Figures", f"dim{dim}",
                             f"ep{TRAIN_LOG_EPOCHS}",
                             f"checkpoint_dim{dim}_ep{TRAIN_LOG_EPOCHS}.pt")
    if os.path.exists(ckpt_path):
        m = FullModel(latent_dim=dim).to(device)
        _load_state_dict_compat(m, ckpt_path)
        m.eval()
        print(f"  Loaded ep5000 checkpoint dim={dim}: {ckpt_path}")
        return m, "ep5000"

    # fallback: OOSSplit best-seed checkpoint
    warnings.warn(f"ep5000 checkpoint not found for dim={dim} — falling back to OOSSplit ep{SPLIT_EPOCHS}")
    ckpt_dir = os.path.join(REPO_ROOT, "Figures", f"OOS_split_dim{dim}", f"ep{SPLIT_EPOCHS}")
    manifest_p = os.path.join(ckpt_dir, "run_manifest.json")
    seed = 0
    if os.path.exists(manifest_p):
        with open(manifest_p) as f:
            mf = json.load(f)
        seed = mf.get("best_seed", 0)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        warnings.warn(f"Fallback checkpoint not found: {ckpt_path}")
        return None, None
    m = FullModel(latent_dim=dim).to(device)
    _load_state_dict_compat(m, ckpt_path)
    m.eval()
    print(f"  Loaded OOSSplit fallback dim={dim} seed={seed}: {ckpt_path}")
    return m, f"OOSSplit_seed{seed}"

print(f"Loading ℓ={LATENT_DIM} model (ep5000, fallback to OOSSplit)...")
best_model, best_model_source = load_ep5000_model(LATENT_DIM)
print(f"  Source: {best_model_source}")

# ── inference helpers ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, X, batch=256):
    """Returns S_hat, z, mu, sigma_L, r_tilde for the full dataset X."""
    S_list, z_list, mu_list, L_list, r_list = [], [], [], [], []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i+batch].to(device)
        S_hat, z, _, _, _, _, mu, sigma_L, r_tilde, _ = model(xb)
        S_list.append(S_hat.cpu());  z_list.append(z.cpu())
        mu_list.append(mu.cpu());    L_list.append(sigma_L.cpu())
        r_list.append(r_tilde.cpu())
    return (torch.cat(S_list), torch.cat(z_list),
            torch.cat(mu_list), torch.cat(L_list), torch.cat(r_list))

print("Running inference on train + test sets...")
S_hat_train, Z_train, _, _, _ = run_inference(best_model, X_train)
S_hat_test,  Z_test,  _, _, _ = run_inference(best_model, X_test)

# finite masks
def finite_mask(X, S):
    return torch.isfinite(X).all(1) & torch.isfinite(S).all(1)

mask_train = finite_mask(X_train, S_hat_train)
mask_test  = finite_mask(X_test,  S_hat_test)

def save_fig(fig, name):
    path = os.path.join(FIGURES_OUT, name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def save_params_fig(fig, name):
    path = os.path.join(PARAMS_DIR, name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def save_table(df, name):
    path = os.path.join(TABLES_OUT, name + ".csv")
    df.reset_index().to_csv(path, index=False)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Q1a — Table: IS RMSE per currency × latent dim (d = 1, 2, 3, 4)
#        Source: Figures/dim{N}/ep5000/train_rmse_log_bbg_dim{N}_ep5000.csv
#        Uses the final epoch row (epoch 4999) — single full-dataset training run
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q1a: IS RMSE table (all dims, ep5000 training logs) ──")

CCY_LOG_COLS = [f"rmse_bps_{c}" for c in CCY_ORDER]

def load_training_log_rmse(dim, epochs=TRAIN_LOG_EPOCHS):
    """Load final-epoch IS RMSE per currency from dim{N}/ep{E} training log CSV."""
    path = os.path.join(REPO_ROOT, "Figures", f"dim{dim}",
                        f"ep{epochs}", f"train_rmse_log_bbg_dim{dim}_ep{epochs}.csv")
    if not os.path.exists(path):
        warnings.warn(f"Missing training log: {path}")
        return None
    df = pd.read_csv(path)
    # Last row = final epoch; guard against incomplete runs (only epoch 0)
    last = df.iloc[-1]
    if int(last["epoch"]) < epochs - 2:
        warnings.warn(f"dim{dim} ep{epochs} log only has epoch {int(last['epoch'])} "
                      f"— run may be incomplete. Skipping.")
        return None
    result = pd.Series({ccy: float(last[f"rmse_bps_{ccy}"]) for ccy in CCY_ORDER})
    result["Average"] = float(last["avg_rmse_bps"])
    return result

rows_is = {}
for dim in [1, 2, 3, 4]:
    is_rmse = load_training_log_rmse(dim)
    if is_rmse is not None:
        rows_is[f"$\\ell={dim}$"] = is_rmse
        print(f"  ell={dim}: avg IS RMSE = {is_rmse['Average']:.2f} bps")

table_q1a = pd.DataFrame(rows_is).T          # rows=dims, cols=currencies
table_q1a = table_q1a[[c for c in CCY_ORDER + ["Average"] if c in table_q1a.columns]]
table_q1a = table_q1a.round(2)
save_table(table_q1a, "Q1a_IS_rmse_all_dims")
print(table_q1a.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load OOS RMSE from OOSSplit runs (used by Q2a, Q4)
# Source: OOS_split_dim{N}/ep{E}/run_manifest.json (preferred) or rmse_summary.csv
# ─────────────────────────────────────────────────────────────────────────────
def load_split_rmse(dim, epochs=SPLIT_EPOCHS):
    """Return (IS mean series, OOS mean series) from OOSSplit results."""
    manifest_p = os.path.join(REPO_ROOT, "Figures", f"OOS_split_dim{dim}",
                              f"ep{epochs}", "run_manifest.json")
    if os.path.exists(manifest_p):
        with open(manifest_p) as f:
            mf = json.load(f)
        results = mf.get("seed_results", {})
        if results:
            ccys = CCY_ORDER + ["Average"]
            is_vals, oos_vals = {c: [] for c in ccys}, {c: [] for c in ccys}
            DIVERGE_THRESHOLD = 100.0
            n_skipped = 0
            for s_info in results.values():
                oos_avg = s_info["oos_avg_bps"]
                if oos_avg is None or (isinstance(oos_avg, float) and
                                       (np.isnan(oos_avg) or oos_avg > DIVERGE_THRESHOLD)):
                    n_skipped += 1
                    continue
                for ccy in CCY_ORDER:
                    is_vals[ccy].append(s_info["is_per_ccy_bps"].get(ccy, np.nan))
                    oos_vals[ccy].append(s_info["oos_per_ccy_bps"].get(ccy, np.nan))
                is_vals["Average"].append(s_info["is_avg_bps"])
                oos_vals["Average"].append(s_info["oos_avg_bps"])
            if n_skipped:
                print(f"  [dim={dim}] Excluded {n_skipped}/{len(results)} diverged seeds")
            return (pd.Series({c: np.nanmean(v) for c, v in is_vals.items()}),
                    pd.Series({c: np.nanmean(v) for c, v in oos_vals.items()}))

    # fallback: rmse_summary.csv
    path = os.path.join(REPO_ROOT, "Figures", f"OOS_split_dim{dim}",
                        f"ep{epochs}", "rmse_summary.csv")
    if not os.path.exists(path):
        warnings.warn(f"Missing: {path}")
        return None, None
    df = pd.read_csv(path, index_col=0)
    is_col  = [c for c in df.columns if "IS mean"  in c][0]
    oos_col = [c for c in df.columns if "OOS mean" in c][0]
    return df[is_col], df[oos_col]


# ─────────────────────────────────────────────────────────────────────────────
# Q1b — Plot: Fitted vs actual for 3 representative dates (IS)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q1b: Fitted vs actual (representative dates) ──")

REPRESENTATIVE_DATES = {
    "Normal market (2016-08-31)":  "2016-08-31",
    "Crisis (2020-03-31)":         "2020-03-31",
    "Low-rate (2019-06-30)":       "2019-06-30",
}
SHOW_CCYS = ["EUR", "USD"]   # show 2 currencies per date for clarity

scale = 100.0 if SCALE_IS_PERCENT else 1.0

fig, axes = plt.subplots(len(SHOW_CCYS), len(REPRESENTATIVE_DATES),
                         figsize=(5 * len(REPRESENTATIVE_DATES), 3.8 * len(SHOW_CCYS)),
                         sharey=False)

for col_i, (label, date_str) in enumerate(REPRESENTATIVE_DATES.items()):
    target_date = pd.Timestamp(date_str)

    for row_i, ccy in enumerate(SHOW_CCYS):
        ax = axes[row_i][col_i]

        # pick closest available date for this currency (train set)
        mask_ccy = (meta_train["ccy"] == ccy).values & mask_train.numpy()
        if mask_ccy.sum() == 0:
            ax.set_visible(False)
            continue

        dates_ccy = pd.to_datetime(meta_train.loc[mask_ccy, "as_of_date"])
        idx_local  = (dates_ccy - target_date).abs().argmin()
        actual_date = dates_ccy.iloc[idx_local]
        global_idx  = np.where(mask_ccy)[0][idx_local]

        actual = X_train[global_idx].numpy() * scale
        fitted = S_hat_train[global_idx].numpy() * scale
        fitted_color = custom_palette[CCY_ORDER.index(ccy) % len(custom_palette)]

        ax.plot(tenors, actual, "o-",  color="black", linewidth=2.0, markersize=5)
        ax.plot(tenors, fitted, "s--", color=fitted_color, linewidth=2.0, markersize=5)

        if row_i == 0:
            ax.set_title(label, fontsize=10, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel(f"{ccy}, {'Rate (%)' if SCALE_IS_PERCENT else 'Rate (dec.)'}",
                          fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.text(0.97, 0.05, actual_date.strftime("%Y-%m-%d"),
                transform=ax.transAxes, fontsize=7, ha="right", color="0.4")

fig.tight_layout()
save_fig(fig, "Q1b_fitted_vs_actual")


# ─────────────────────────────────────────────────────────────────────────────
# Q1d — Plot: Fitted vs actual, all latent dims (ℓ=2, 3, 4) overlaid
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q1d: Fitted vs actual — all dims overlaid ──")

# Load ep5000 models for Q1d and Q6d (fallback to OOSSplit if not yet available)
# ℓ=3 is already loaded as best_model above — reuse it to avoid loading twice
dim_models = {}
dim_model_sources = {}
for _dim in [2, 3, 4]:
    if _dim == LATENT_DIM:
        dim_models[_dim] = best_model
        dim_model_sources[_dim] = best_model_source
    else:
        _m, _src = load_ep5000_model(_dim)
        if _m is not None:
            dim_models[_dim] = _m
            dim_model_sources[_dim] = _src

# Pre-compute S_hat and Z for each dim on X_train
dim_S_hat = {}
dim_Z_hat = {}
for _dim, _m in dim_models.items():
    _S, _Z, _, _, _ = run_inference(_m, X_train)
    dim_S_hat[_dim] = _S
    dim_Z_hat[_dim] = _Z

DIMS_PLOT   = sorted(dim_models.keys())
DIM_LABELS  = {d: r"$\ell$=" + str(d) for d in DIMS_PLOT}
DIM_STYLES  = {2: "-",  3: "--", 4: ":"}

# Load dim=1 model for histograms (dims 2,3,4 already in dim_S_hat)
_all_dim_S_hat = dict(dim_S_hat)
_m1, _src1 = load_ep5000_model(1)
if _m1 is not None:
    _S1, _, _, _, _ = run_inference(_m1, X_train)
    _all_dim_S_hat[1] = _S1

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes_flat = axes.flatten()

for ax_i, _dim in enumerate([1, 2, 3, 4]):
    ax = axes_flat[ax_i]
    if _dim not in _all_dim_S_hat:
        ax.set_visible(False)
        continue

    resid = (X_train[mask_train] - _all_dim_S_hat[_dim][mask_train]).numpy() * 10000
    resid_flat = resid.flatten()
    resid_flat = resid_flat[np.isfinite(resid_flat)]

    ax.hist(resid_flat, bins=120, color=DIM_COLORS[_dim], edgecolor="none", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(np.mean(resid_flat), color="#d7191c", linewidth=1.5, linestyle="--")
    ax.axvline(np.percentile(resid_flat,  5), color="0.4", linewidth=1.0, linestyle=":")
    ax.axvline(np.percentile(resid_flat, 95), color="0.4", linewidth=1.0, linestyle=":")
    ax.set_title(r"$\ell=" + str(_dim) + r"$", fontsize=11, fontweight="bold")
    ax.set_xlabel("Residual (bps)")
    ax.set_ylabel("Count")
    ax.text(0.97, 0.95,
            f"N={len(resid_flat):,}\nStd={np.std(resid_flat):.2f} bps\n"
            f"Kurt={float(pd.Series(resid_flat).kurt()):.2f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

fig.tight_layout()
save_fig(fig, "Q1d_residual_histograms_all_dims")


# ─────────────────────────────────────────────────────────────────────────────
# Q1c — Plot: Histogram of residuals (actual − fitted) in bps
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q1c: Residual histogram ──")

resid_train = (X_train[mask_train] - S_hat_train[mask_train]).numpy() * 10000  # → bps
resid_flat  = resid_train.flatten()
resid_flat  = resid_flat[np.isfinite(resid_flat)]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(resid_flat, bins=120, color=custom_palette[0], edgecolor="none", alpha=0.85)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(np.mean(resid_flat),  color="#d7191c", linewidth=1.5,
           linestyle="--", label=f"Mean = {np.mean(resid_flat):.2f} bps")
ax.axvline(np.percentile(resid_flat, 5),  color="0.4", linewidth=1.0,
           linestyle=":", label=f"5th/95th pct = {np.percentile(resid_flat,5):.1f} / "
                                f"{np.percentile(resid_flat,95):.1f} bps")
ax.axvline(np.percentile(resid_flat, 95), color="0.4", linewidth=1.0, linestyle=":")
ax.set_xlabel("Residual (bps)")
ax.set_ylabel("Count")
ax.legend(fontsize=9, frameon=False)
ax.text(0.97, 0.95,
        f"N={len(resid_flat):,}\nStd={np.std(resid_flat):.2f} bps\n"
        f"Kurt={float(pd.Series(resid_flat).kurt()):.2f}",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
fig.tight_layout()
save_fig(fig, "Q1c_residual_histogram")


# ─────────────────────────────────────────────────────────────────────────────
# Q2a — Table: IS vs OOS RMSE side-by-side for d = 1, 2, 3, 4
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q2a: IS vs  OOS RMSE table (all dims) ──")

rows_q2 = {}
for dim in [1, 2, 3, 4]:
    is_mean, oos_mean = load_split_rmse(dim)
    if is_mean is not None:
        rows_q2[("IS",  f"$\\ell={dim}$")] = is_mean
        rows_q2[("OOS", f"$\\ell={dim}$")] = oos_mean

table_q2a = pd.DataFrame(rows_q2).T
table_q2a.index = pd.MultiIndex.from_tuples(table_q2a.index, names=["Split", "Model"])
table_q2a = table_q2a[[c for c in CCY_ORDER + ["Average"] if c in table_q2a.columns]]
table_q2a = table_q2a.round(2)
save_table(table_q2a, "Q2a_IS_vs_OOS_all_dims")
print(table_q2a.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Q2b — Plot: OOS RMSE vs latent dimension (rolling average per dim)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q2b: Rolling OOS RMSE vs dim ──")

ROLL_SUBDIR = "train3Y_test3M_step6M"

ROLL_DIVERGE_THRESHOLD = 100.0  # bps — rolling windows above this are training failures

def load_rolling_avg(dim):
    """Return average OOS RMSE across valid rolling windows for a given dim.
    Windows where avg_rmse_bps > ROLL_DIVERGE_THRESHOLD are excluded (diverged training)."""
    path = os.path.join(REPO_ROOT, "Figures", f"OOS_roll_dim{dim}",
                        ROLL_SUBDIR, f"ep{SPLIT_EPOCHS}",
                        f"oos_rolling_bbg_dim{dim}_train3Y_test3M_step6M.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    n_bad = len(df) - len(valid)
    if n_bad:
        print(f"  [dim={dim}] Excluded {n_bad}/{len(df)} diverged rolling windows "
              f"(avg_rmse_bps > {ROLL_DIVERGE_THRESHOLD} bps)")
    return float(valid["avg_rmse_bps"].mean())

roll_avgs = {}
for dim in [1, 2, 3, 4]:
    avg = load_rolling_avg(dim)
    if avg is not None:
        roll_avgs[dim] = avg

if len(roll_avgs) >= 2:
    dims = sorted(roll_avgs.keys())
    avgs = [roll_avgs[d] for d in dims]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(d) for d in dims], avgs,
                  color=[DIM_COLORS[d] for d in dims],
                  width=0.55, edgecolor="none")
    best_dim = min(roll_avgs, key=roll_avgs.get)
    for bar, d, val in zip(bars, dims, avgs):
        label = f"{val:.1f}" + (" *" if d == best_dim else "")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Average Rolling OOS RMSE (bps)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    save_fig(fig, "Q2b_rolling_oos_vs_dim")
else:
    print(f"  SKIPPED — only {len(roll_avgs)}/4 rolling results available. "
          f"Re-run once OutOfSampleRoll.py finishes for missing dims.")


# ─────────────────────────────────────────────────────────────────────────────
# Q3a — Table: OOS RMSE per currency, all seeds + mean ± std (d=3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q3a: OOS seeds table (d=3) ──")

if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH) as f:
        mf = json.load(f)

    seed_results = mf.get("seed_results", {})
    seed_keys    = sorted(seed_results.keys(), key=int)

    oos_rows = {}
    for sk in seed_keys:
        per_ccy = seed_results[sk]["oos_per_ccy_bps"]
        oos_rows[f"Seed {sk}"] = {ccy: per_ccy.get(ccy, np.nan) for ccy in CCY_ORDER}
        oos_rows[f"Seed {sk}"]["Average"] = seed_results[sk]["oos_avg_bps"]

    table_q3a = pd.DataFrame(oos_rows).T
    table_q3a = table_q3a[[c for c in CCY_ORDER + ["Average"] if c in table_q3a.columns]]
    table_q3a = table_q3a.round(2)

    # valid mean: exclude diverged seeds (avg > 100 bps or NaN)
    _avg_num = pd.to_numeric(table_q3a["Average"], errors="coerce")
    _valid_mask = _avg_num.notna() & (_avg_num < 100)
    print(f"  Valid seeds: {list(table_q3a.index[_valid_mask])}")
    _mean_valid = pd.Series(np.nan, index=table_q3a.columns)
    _mean_valid["Average"] = round(_avg_num[_valid_mask].mean(), 2)
    print(f"  Mean (valid) Average: {_mean_valid['Average']}")
    table_q3a.loc["Mean (valid)"] = _mean_valid

    save_table(table_q3a, "Q3a_OOS_seeds_table_dim3")
    print(table_q3a.to_string())
else:
    print("  SKIPPED  — run_manifest.json not found.")


# ─────────────────────────────────────────────────────────────────────────────
# Q3b — Plot: Rolling OOS RMSE over time (d=3, per-currency + average)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q3b: Rolling RMSE over time (d=3) ── ")

roll_path_d3 = os.path.join(REPO_ROOT, "Figures", f"OOS_roll_dim{LATENT_DIM}",
                             ROLL_SUBDIR, f"ep{SPLIT_EPOCHS}",
                             f"oos_rolling_bbg_dim{LATENT_DIM}_train3Y_test3M_step6M.csv")

if os.path.exists(roll_path_d3):
    df_roll = pd.read_csv(roll_path_d3)
    df_roll["test_start"] = pd.to_datetime(df_roll["test_start"])

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for ccy in CCY_ORDER:
        col = f"rmse_bps_{ccy}"
        if col in df_roll.columns:
            valid = df_roll[col].notna()
            ax.plot(df_roll.loc[valid, "test_start"], df_roll.loc[valid, col],
                    linewidth=1.0, alpha=0.55, color=currency_color_map[ccy], label=ccy)

    # average line — thin dashed
    ax.plot(df_roll["test_start"], df_roll["avg_rmse_bps"],
            linewidth=1.2, color="black", linestyle="--", label="Average", zorder=5)

    # event markers
    for label, date_str in EVENTS.items():
        d = pd.Timestamp(date_str)
        if df_roll["test_start"].min() <= d <= df_roll["test_start"].max():
            ax.axvline(d, color="0.5", linewidth=1.0, linestyle="--")
            ax.text(d, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1,
                    label, fontsize=10, ha="center", va="bottom", color="0.4",
                    rotation=0)

    ax.set_ylabel("OOS RMSE (bps)")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, "Q3b_rolling_rmse_over_time")
else:
    print(f"  SKIPPED — rolling CSV not found at {roll_path_d3}")


# ─────────────────────────────────────────────────────────────────────────────
# Q4a — Table: Autoencoder d=3 vs EKF DNS 1f/2f/3f/4f, OOS RMSE per currency
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q4a: AE vs Kalman table ──")

def load_kalman_rmse(dim):
    path = os.path.join(REPO_ROOT, "Figures", "kalman_benchmark_oos",
                        f"ekf_dns_{dim}f", "rmse_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    oos_col = [c for c in df.columns if "OOS mean" in c][0]
    return df[oos_col]

rows_q4 = {}
# interleave AE and EKF DNS by dimension: (AE l=2, EKF 2f), (AE l=3, EKF 3f), (AE l=4, EKF 4f)
for dim in [2, 3, 4]:
    _, oos_ae_dim = load_split_rmse(dim)
    if oos_ae_dim is not None:
        rows_q4[rf"AE $\ell$={dim}"] = oos_ae_dim
    oos_k = load_kalman_rmse(dim)
    if oos_k is not None:
        rows_q4[rf"EKF DNS $\ell$={dim}"] = oos_k

table_q4a = pd.DataFrame(rows_q4).T
table_q4a = table_q4a[[c for c in CCY_ORDER + ["Average"] if c in table_q4a.columns]]
table_q4a = table_q4a.round(2)
save_table(table_q4a, "Q4a_AE_vs_Kalman_OOS")
print(table_q4a.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Q4b — Plot: Per-currency bar chart, AE d=3 vs best Kalman (EKF DNS 3f & 4f)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q4b: Per-currency bar chart ──")

_, oos_ae3  = load_split_rmse(3)
oos_k3 = load_kalman_rmse(3)
oos_k4 = load_kalman_rmse(4)

if oos_ae3 is not None and oos_k3 is not None:
    x      = np.arange(len(CCY_ORDER))
    width  = 0.26
    labels = CCY_ORDER

    fig, ax = plt.subplots(figsize=(11, 5))

    bars_ae = ax.bar(x - width / 2,
                     [oos_ae3.get(c, np.nan) for c in labels],
                     width, label=r"AE ($\ell$=3)",
                     color=custom_palette[2], edgecolor="none")

    bars_k3 = ax.bar(x + width / 2,
                     [oos_k3.get(c, np.nan) for c in labels],
                     width, label=r"EKF DNS ($\ell$=3)",
                     color=custom_palette[0], edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("OOS RMSE (bps)")
    ax.legend(frameon=False, fontsize=9)

    # Add horizontal average lines
    avg_ae = oos_ae3.drop("Average", errors="ignore").mean()
    ax.axhline(avg_ae, color=custom_palette[2], linewidth=1.0, linestyle="--", alpha=0.7)
    avg_k3 = oos_k3.drop("Average", errors="ignore").mean()
    ax.axhline(avg_k3, color=custom_palette[0], linewidth=1.0, linestyle="--", alpha=0.7)

    fig.tight_layout()
    save_fig(fig, "Q4b_per_currency_bar_chart")
else:
    print("  SKIPPED — missing AE or Kalman OOS data.")


# ─────────────────────────────────────────────────────────────────────────────
# Q4c — Plot: Per-currency bar chart, AE ℓ=2,3,4 vs EKF DNS 3f
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q4c: Per-currency bar chart (AE ℓ=2,3,4 vs EKF DNS 3f) ──")

_, oos_ae2 = load_split_rmse(2)
_, oos_ae3c = load_split_rmse(3)
_, oos_ae4 = load_split_rmse(4)
oos_k3c = load_kalman_rmse(3)

if oos_ae2 is not None and oos_ae3c is not None and oos_ae4 is not None and oos_k3c is not None:
    x      = np.arange(len(CCY_ORDER))
    width  = 0.20
    labels = CCY_ORDER

    fig, ax = plt.subplots(figsize=(13, 5))

    bars_ae2 = ax.bar(x - 1.5 * width,
                      [oos_ae2.get(c, np.nan) for c in labels],
                      width, label=r"AE $\ell$=2",
                      color=custom_palette[1], edgecolor="none")

    bars_ae3 = ax.bar(x - 0.5 * width,
                      [oos_ae3c.get(c, np.nan) for c in labels],
                      width, label=r"AE $\ell$=3",
                      color=custom_palette[2], edgecolor="none")

    bars_ae4 = ax.bar(x + 0.5 * width,
                      [oos_ae4.get(c, np.nan) for c in labels],
                      width, label=r"AE $\ell$=4",
                      color=custom_palette[3], edgecolor="none")

    bars_k3 = ax.bar(x + 1.5 * width,
                     [oos_k3c.get(c, np.nan) for c in labels],
                     width, label=r"EKF DNS ($\ell$=3)",
                     color=custom_palette[0], edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("OOS RMSE (bps)")
    ax.legend(frameon=False, fontsize=9)

    # average lines — distinct styles matching DIM_STYLES
    _avg_styles = [(oos_ae2, custom_palette[1], "-"),
                   (oos_ae3c, custom_palette[2], "--"),
                   (oos_ae4, custom_palette[3], ":"),
                   (oos_k3c, custom_palette[0], "-")]
    for oos, col, ls in _avg_styles:
        avg = oos.drop("Average", errors="ignore").mean()
        ax.axhline(avg, color=col, linewidth=1.2, linestyle=ls, alpha=0.85)

    fig.tight_layout()
    save_fig(fig, "Q4c_per_currency_bar_chart_all_dims")
else:
    print("  SKIPPED — missing AE or Kalman OOS data.")


# ─────────────────────────────────────────────────────────────────────────────
# Q5a — Plot: Latent factors over time (full IS sample, per currency)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q5a: Latent factors over time ──")

Z_np    = Z_train[mask_train].numpy()
dates_z = pd.to_datetime(meta_train.loc[mask_train.numpy(), "as_of_date"])
ccys_z  = meta_train.loc[mask_train.numpy(), "ccy"].values

fig, axes = plt.subplots(LATENT_DIM, 1,
                         figsize=(12, 3.2 * LATENT_DIM), sharex=True)
if LATENT_DIM == 1:
    axes = [axes]

for ccy in CCY_ORDER:
    idx = (ccys_z == ccy)
    if idx.sum() == 0:
        continue
    sort_i = np.argsort(dates_z.values[idx])
    for dim_i, ax in enumerate(axes):
        ax.plot(dates_z.values[idx][sort_i], Z_np[idx][:, dim_i][sort_i],
                linewidth=1.1, alpha=0.8, label=ccy,
                color=currency_color_map[ccy])

for dim_i, ax in enumerate(axes):
    ax.set_ylabel(f"$z_{dim_i+1}$", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    # event markers
    for ev_label, ev_date in EVENTS.items():
        ax.axvline(pd.Timestamp(ev_date), color="0.6", linewidth=0.8, linestyle=":")
    pass  # legend removed — currency colours described in caption

# add event text on top panel only
for ev_label, ev_date in EVENTS.items():
    axes[0].text(pd.Timestamp(ev_date), axes[0].get_ylim()[1],
                 ev_label, fontsize=10, ha="center", va="bottom", color="0.4")
fig.tight_layout()
save_fig(fig, "Q5a_latent_factors_over_time")


# ─────────────────────────────────────────────────────────────────────────────
# Q5b — Tables: Correlation of latent factors with level / slope / curvature
#        One table per latent dimension (ℓ=2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q5b: Factor correlation tables (all dims) ──")

_corr_matrices = {}  # dim → DataFrame, collected for heatmap below

# Rebuild wide swap data aligned to training observations
df_is = df_wide_all.copy()
df_is["as_of_date"] = pd.to_datetime(df_is["as_of_date"])
df_is["ccy"] = df_is["ccy"].str.upper()

scale_div = 100.0 if SCALE_IS_PERCENT else 1.0

# ── Global PCA on IS swap rates (all currencies stacked) ─────────────────────
# Used as model-neutral reference basis for Q5b
from sklearn.decomposition import PCA as _SKLearnPCA
_X_is_all  = X_train[mask_train].numpy()                     # (N_is, 8)
_finite_is = np.isfinite(_X_is_all).all(axis=1)
_X_is_pca  = _X_is_all[_finite_is] / scale_div               # scale-consistent
_global_pca = _SKLearnPCA(n_components=8)
_global_pca.fit(_X_is_pca)
_pc_vecs = _global_pca.components_                            # (8, 8) — rows are eigenvectors
print(f"  Global PCA explained variance ratios: "
      f"{np.round(_global_pca.explained_variance_ratio_ * 100, 2)}")

# ── Plot: first 3 eigenvectors across the 8 maturities ───────────────────────
_tenor_labels = [str(t) for t in TENOR_COLS]
_n_plot_pcs   = 5
_pc_plot_labels = ["PC1 (level)", "PC2 (slope)", "PC3 (curvature)", "PC4", "PC5"]

fig, ax = plt.subplots(figsize=(7, 3.5))
for j in range(_n_plot_pcs):
    _v = _pc_vecs[j]
    # sign-normalise: make largest-absolute-value entry positive
    if _v[np.argmax(np.abs(_v))] < 0:
        _v = -_v
    ax.plot(range(8), _v, marker="o", linewidth=2,
            color=custom_palette[j], label=_pc_plot_labels[j])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(range(8))
ax.set_xticklabels(_tenor_labels, fontsize=10)
ax.set_xlabel("Maturity", fontsize=10)
ax.set_ylabel("Eigenvector loading", fontsize=10)
ax.legend(fontsize=10)
fig.tight_layout()
save_fig(fig, "Q5b_pca_eigenvectors")

for _dim in DIMS_PLOT:
    _Z    = dim_Z_hat[_dim]
    _mask = finite_mask(X_train, dim_S_hat[_dim]) & mask_train  # IS only
    _Z_np = _Z[_mask].numpy()
    _X_np = X_train[_mask].numpy() / scale_div                  # (N, 8) scaled

    meta_z = meta_train.loc[_mask.numpy()].copy().reset_index(drop=True)
    meta_z["as_of_date"] = pd.to_datetime(meta_z["as_of_date"])
    for k in range(_dim):
        meta_z[f"z{k+1}"] = _Z_np[:, k]

    # PC scores: project IS swap rates onto all 8 global PCA eigenvectors
    for j in range(8):
        meta_z[f"PC{j+1}"] = _X_np @ _pc_vecs[j]

    z_cols  = [f"z{k+1}"  for k in range(_dim)]
    pc_cols = [f"PC{j+1}" for j in range(8)]

    corr_rows = {}
    for zc in z_cols:
        row = {}
        for pc in pc_cols:
            valid = meta_z[[zc, pc]].dropna()
            row[pc] = round(float(valid[zc].corr(valid[pc])), 3)
        corr_rows[zc] = row

    table_q5b = pd.DataFrame(corr_rows).T
    table_q5b.index   = [f"$z_{k+1}$" for k in range(_dim)]
    table_q5b.columns = pc_cols
    _corr_matrices[_dim] = table_q5b.copy()
    save_table(table_q5b, f"Q5b_factor_correlations_dim{_dim}")
    print(f"\n  ℓ={_dim} (AE factor–PC correlations):")
    print(table_q5b.to_string())

    # ── Weight projection: squared cosine similarity × 100 (Rolf Poulsen method)
    # w_{ij} = (W_i · V_j)^2 / ||W_i||^2 × 100  →  sums to 100% across all 8 PCs
    _W     = dim_models[_dim].encoder.lin.weight.detach().numpy()  # (d, 8)
    _W_hat = _W / np.linalg.norm(_W, axis=1, keepdims=True)        # unit-norm rows
    _cos   = _W_hat @ _pc_vecs.T                                    # (d, 8) cosine similarities
    _proj  = np.round(_cos ** 2 * 100, 2)                          # (d, 8) percentages
    table_wp = pd.DataFrame(
        _proj,
        index   = [f"$z_{k+1}$" for k in range(_dim)],
        columns = [f"PC{j+1}" for j in range(8)],
    )
    save_table(table_wp, f"Q5b_weight_projection_dim{_dim}")
    print(f"\n  ℓ={_dim} weight projection (squared cosine × 100, sums to 100%):")
    print(table_wp.to_string())

# ── Combined weight projection CSV (all dims stacked, PC1–PC4 columns) ───────
_pc_all_cols = [f"PC{j+1}" for j in range(8)]

_wp_rows = []
_wp_data  = {}   # dim → np array (d, 8), kept for bar plot
for _dim in DIMS_PLOT:
    _wp_path = os.path.join(TABLES_OUT, f"Q5b_weight_projection_dim{_dim}.csv")
    if not os.path.exists(_wp_path):
        continue
    _wp = pd.read_csv(_wp_path, index_col=0)   # (d, 8)
    _wp_data[_dim] = _wp.values.astype(float)
    for k in range(_dim):
        row = {"model": f"$\\ell={_dim}$" if k == 0 else "", "factor": f"$z_{k+1}$"}
        for j in range(8):
            row[f"PC{j+1}"] = f"{_wp.iloc[k, j]:.2f}"
        _wp_rows.append(row)
_wp_combined = pd.DataFrame(_wp_rows, columns=["model", "factor"] + _pc_all_cols)
_wp_combined.to_csv(os.path.join(TABLES_OUT, "Q5b_weight_projection_combined.csv"), index=False)
print("  Saved: Q5b_weight_projection_combined.csv")

# ── Bar plot: ρ_j scree plot (Rolf Figure 4 style) ───────────────────────────
_rho       = _global_pca.explained_variance_ratio_   # (8,) — Rolf's ρ_j
_pc_labels = [str(j+1) for j in range(8)]
_x         = np.arange(8)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(_x, _rho, width=0.6, color="gray", alpha=0.8, zorder=2)
ax.set_xticks(_x)
ax.set_xticklabels(_pc_labels, fontsize=10)
ax.set_xlabel("Principal component", fontsize=10)
ax.set_ylabel(r"Relative weight of eigenvalue $\rho_j$", fontsize=10)
ax.set_ylim(0, 1)
ax.tick_params(axis="x", length=0)
fig.tight_layout()
save_fig(fig, "Q5b_weight_projection_barplot")

# ── Q5b heatmap: all dims side by side ───────────────────────────────────────
if _corr_matrices:
    from matplotlib.colors import LinearSegmentedColormap

    # Red (−1) → white (0) → blue (+1), using custom_palette colours
    _cmap_q5b = LinearSegmentedColormap.from_list(
        "q5b_div",
        [custom_palette[4], "white", custom_palette[0]],
        N=256
    )

    _hm_dims    = sorted(_corr_matrices.keys())
    _n_panels   = len(_hm_dims)
    _row_counts = [len(_corr_matrices[d]) for d in _hm_dims]   # [2, 3, 4]
    _total_rows = sum(_row_counts)

    # height ratios proportional to number of factors per model
    fig, axes = plt.subplots(
        _n_panels, 1,
        figsize=(9, 0.8 * _total_rows + 0.5 * _n_panels),
        gridspec_kw={"height_ratios": _row_counts, "hspace": 0.4},
    )
    if _n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(right=0.88)   # leave room for colorbar

    for ax, _dim in zip(axes, _hm_dims):
        _mat           = _corr_matrices[_dim].values.astype(float)
        _nrows, _ncols = _mat.shape
        _row_labels    = [f"$z_{k+1}$" for k in range(_nrows)]
        _col_labels    = [f"PC{j+1}"   for j in range(_ncols)]

        _X = np.arange(_ncols + 1)
        _Y = np.arange(_nrows + 1)
        im = ax.pcolormesh(_X, _Y, _mat, cmap=_cmap_q5b,
                           vmin=-1, vmax=1, edgecolors="face")
        ax.invert_yaxis()
        ax.set_xticks([c + 0.5 for c in range(_ncols)])
        ax.set_xticklabels(_col_labels, fontsize=9)
        ax.set_yticks([i + 0.5 for i in range(_nrows)])
        ax.set_yticklabels(_row_labels, fontsize=10)
        ax.tick_params(length=0)
        ax.set_title(r"$\ell=" + str(_dim) + r"$", fontsize=11,
                     fontweight="bold", loc="left", pad=4)

        for r in range(_nrows):
            for c in range(_ncols):
                val = _mat[r, c]
                txt_color = "white" if abs(val) > 0.6 else "black"
                ax.text(c + 0.5, r + 0.5, f"{val:.3f}",
                        ha="center", va="center", fontsize=9, color=txt_color)

    # shared colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Pearson $r$")
    save_fig(fig, "Q5b_factor_correlation_heatmap")


# Q5c removed — EKF DNS factor correlation heatmap dropped from analysis


# ─────────────────────────────────────────────────────────────────────────────
# Q6a — Plot: RMSE broken down by tenor (1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q6a: RMSE by tenor (full dataset, ep5000 training run) ──")

X_eval = X_train[mask_train].numpy()
S_eval = S_hat_train[mask_train].numpy()

# per-tenor RMSE across all currencies and dates
rmse_by_tenor = np.sqrt(np.mean((X_eval - S_eval)**2, axis=0)) * 10000  # bps

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# left: overall
ax = axes[0]
ax.bar([str(t) for t in TENOR_COLS], rmse_by_tenor,
       color=custom_palette[2], edgecolor="none")
ax.set_ylabel("RMSE (bps)")
ax.set_title("In-Sample RMSE by Tenor")
for i, v in enumerate(rmse_by_tenor):
    ax.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

# right: per-currency lines
ax = axes[1]
for ccy in CCY_ORDER:
    idx = (meta_train.loc[mask_train.numpy(), "ccy"].values == ccy)
    if idx.sum() == 0:
        continue
    rmse_ccy = np.sqrt(np.mean((X_eval[idx] - S_eval[idx])**2, axis=0)) * 10000
    ax.plot(TENOR_COLS, rmse_ccy, marker="o", linewidth=1.4,
            markersize=4, color=currency_color_map[ccy])

ax.set_ylabel("RMSE (bps)")
ax.set_title("In-Sample RMSE by Tenor Per Currency")
ax.set_xticks(TENOR_COLS)

fig.tight_layout()
save_fig(fig, "Q6a_rmse_by_tenor")


# ─────────────────────────────────────────────────────────────────────────────
# Q6d — Plot: IS RMSE by tenor — all latent dims overlaid (ℓ=2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q6d: RMSE by tenor — all dims (bar chart) ──")

# Compute per-tenor OOS RMSE for each dim
_oos_mask = ~mask_train
rmse_by_dim = {}
for _dim in DIMS_PLOT:
    _S_hat  = dim_S_hat[_dim]
    _X_oos  = X_train[_oos_mask].numpy()
    _S_oos  = _S_hat[_oos_mask].numpy()
    _finite = np.isfinite(_X_oos).all(1) & np.isfinite(_S_oos).all(1)
    rmse_by_dim[_dim] = np.sqrt(
        np.mean((_X_oos[_finite] - _S_oos[_finite])**2, axis=0)) * 10000  # bps

# Load EKF DNS 3f average OOS RMSE for reference line
_oos_k3_q6d = load_kalman_rmse(3)

n_tenors = len(TENOR_COLS)
n_dims   = len(DIMS_PLOT)
width    = 0.22
x        = np.arange(n_tenors)

_dim_styles_q6d = {2: "-", 3: "--", 4: ":"}

fig, ax = plt.subplots(figsize=(11, 4.5))

for i, _dim in enumerate(DIMS_PLOT):
    offset = (i - (n_dims - 1) / 2) * width
    ax.bar(x + offset, rmse_by_dim[_dim], width,
           label=r"AE ($\ell$=" + str(_dim) + ")",
           color=DIM_COLORS[_dim], edgecolor="none")

# average lines per AE dim — distinct styles
for _dim in DIMS_PLOT:
    avg = float(np.mean(rmse_by_dim[_dim]))
    ax.axhline(avg, color=DIM_COLORS[_dim], linewidth=1.2,
               linestyle=_dim_styles_q6d.get(_dim, "--"), alpha=0.85)

# EKF DNS 3f overall average as horizontal reference
if _oos_k3_q6d is not None:
    _avg_k3 = float(_oos_k3_q6d.drop("Average", errors="ignore").mean())
    ax.axhline(_avg_k3, color=custom_palette[0], linewidth=1.4, linestyle="-.",
               label=r"EKF DNS ($\ell$=3) avg", zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in TENOR_COLS])
ax.set_ylabel("OOS RMSE (bps)")
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
save_fig(fig, "Q6d_rmse_by_tenor_all_dims")


# ─────────────────────────────────────────────────────────────────────────────
# Q6b — Plot: RMSE over time (monthly average IS)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q6b: RMSE over time ──")

meta_eval_z = meta_train.loc[mask_train.numpy()].copy().reset_index(drop=True)
meta_eval_z["as_of_date"] = pd.to_datetime(meta_eval_z["as_of_date"])
meta_eval_z["rmse_bps"] = np.sqrt(
    np.mean((X_eval - S_eval)**2, axis=1)) * 10000

meta_eval_z["ym"] = meta_eval_z["as_of_date"].dt.to_period("M")

# monthly average across all currencies
monthly_avg = meta_eval_z.groupby("ym")["rmse_bps"].mean()
monthly_dates = monthly_avg.index.to_timestamp()

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# top: average across currencies
ax = axes[0]
ax.plot(monthly_dates, monthly_avg.values, linewidth=1.2, color="black",
        linestyle="--", label="All-ccy avg")
ax.set_ylabel("Avg RMSE (bps)")

# bottom: per-currency
ax = axes[1]
for ccy in CCY_ORDER:
    idx_c = meta_eval_z["ccy"] == ccy
    if idx_c.sum() == 0:
        continue
    m_ccy = meta_eval_z.loc[idx_c].groupby("ym")["rmse_bps"].mean()
    ax.plot(m_ccy.index.to_timestamp(), m_ccy.values,
            linewidth=1.1, alpha=0.75, color=currency_color_map[ccy])

ax.set_ylabel("RMSE (bps)")

# event shading on both panels
for ev_label, ev_date in EVENTS.items():
    for axi in axes:
        axi.axvline(pd.Timestamp(ev_date), color="0.55",
                    linewidth=1.0, linestyle="--")
    axes[0].text(pd.Timestamp(ev_date), axes[0].get_ylim()[1],
                 ev_label, fontsize=10, ha="center", va="bottom", color="0.4")

fig.autofmt_xdate()
fig.tight_layout()
save_fig(fig, "Q6b_rmse_over_time")


# ─────────────────────────────────────────────────────────────────────────────
# Q6c — Table: Top 10 worst-fitting (currency, date) pairs
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q6c: Top worst-fitting pairs ──")

meta_eval_z2 = meta_train.loc[mask_train.numpy()].copy().reset_index(drop=True)
meta_eval_z2["as_of_date"] = pd.to_datetime(meta_eval_z2["as_of_date"])
meta_eval_z2["rmse_bps"] = np.sqrt(
    np.mean((X_eval - S_eval)**2, axis=1)) * 10000

# also store per-tenor residuals for context
for i, t in enumerate(TENOR_COLS):
    meta_eval_z2[f"resid_{t}Y_bps"] = (S_eval[:, i] - X_eval[:, i]) * 10000

worst = (meta_eval_z2.nlargest(10, "rmse_bps")
         [["as_of_date", "ccy", "rmse_bps"] +
          [f"resid_{t}Y_bps" for t in TENOR_COLS]]
         .reset_index(drop=True))
worst["as_of_date"] = worst["as_of_date"].dt.strftime("%Y-%m-%d")
worst = worst.round(2)
worst.index = worst.index + 1   # rank from 1

save_table(worst, "Q6c_top10_worst_pairs")
print(worst[["as_of_date", "ccy", "rmse_bps"]].to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Q7 — IS Sharpe ratio by tenor, one plot per latent dimension
#      Source: ep5000 checkpoint (fallback: OOSSplit best seed)
#      Data:   X_train (2004-2020)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q7: IS Sharpe ratio by tenor (all dims) ──")

TAU_GRID = np.arange(1, 31)   # tenors 1..30

@torch.no_grad()
def extract_sharpe(model, X, batch=256):
    """Run forward pass on X and return SR_tau (N, 30)."""
    sr_list = []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i+batch].to(device)
        _, _, _, _, _, _, _, _, _, arb = model(xb)
        sr_list.append(arb["SR_tau"].cpu())
    return torch.cat(sr_list)   # (N, 30)

for _dim in ALL_DIMS_PARAM:
    print(f"  ℓ={_dim} ...", end=" ")
    _m, _src = load_ep5000_model(_dim)
    if _m is None:
        print("skipped (no model)")
        continue

    SR_all  = extract_sharpe(_m, X_train)          # (N, 30)

    # apply finite mask on X_train rows
    x_finite = torch.isfinite(X_train).all(1)
    SR_all   = SR_all[x_finite]                    # (N_valid, 30)

    sr_mean = SR_all.mean(dim=0).numpy()           # (30,)
    sr_std  = SR_all.std(dim=0).numpy()            # (30,)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(TAU_GRID, sr_mean,
            color=DIM_COLORS[_dim], linewidth=1.6,
            label="Mean")
    ax.fill_between(TAU_GRID,
                    sr_mean - sr_std,
                    sr_mean + sr_std,
                    color=DIM_COLORS[_dim], alpha=0.2,
                    label=r"$\pm$1 std")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Approx. Sharpe ratio")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, f"Q7_sharpe_ratio_IS_dim{_dim}")
    print("done")


# ─────────────────────────────────────────────────────────────────────────────
# P — Parameter plots over time (one figure per latent dimension)
#     Source: ep5000 checkpoint (fallback: OOSSplit best seed)
#     Data:   X_train (2004-2020)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Parameter plots ──")

# CCY colours (cycle through custom_palette)
_ccy_colors = {ccy: custom_palette[i % len(custom_palette)]
               for i, ccy in enumerate(CCY_ORDER)}

def extract_parameters(model, X_data, meta_df, mask):
    """
    Run encoder on X_data[mask] and extract μ, σ, ρ, r̃ per observation.
    Returns a DataFrame with columns: as_of_date, ccy, mu_1..d,
    sigma_1..d, rho_12.., r_tilde.
    """
    model.eval()
    with torch.no_grad():
        X_m   = X_data[mask]
        z     = model.encoder(X_m)                    # (N, d)
        mu    = model.K(z)                            # (N, d)
        sigmas, rhos = model.H(z)                     # (N,d), (N,n_corr)
        r_til = model.R(z).squeeze(-1)                # (N,)

    d      = model.latent_dim
    n_corr = d * (d - 1) // 2

    rec = meta_df.loc[mask.numpy()].copy().reset_index(drop=True)
    rec["as_of_date"] = pd.to_datetime(rec["as_of_date"])

    for k in range(d):
        rec[f"mu_{k+1}"]    = mu[:, k].cpu().numpy()
        rec[f"sigma_{k+1}"] = sigmas[:, k].cpu().numpy()

    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            rec[f"rho_{i+1}{j+1}"] = rhos[:, idx].cpu().numpy()
            idx += 1

    rec["r_tilde"] = r_til.cpu().numpy()
    return rec


def _param_label(name):
    """Convert column name to LaTeX-style label."""
    if name.startswith("mu_"):
        k = name.split("_")[1]
        return r"$\mu_{" + k + r"}$"
    if name.startswith("sigma_"):
        k = name.split("_")[1]
        return r"$\sigma_{" + k + r"}$"
    if name.startswith("rho_"):
        ij = name.split("_")[1]
        return r"$\rho_{" + ",".join(ij) + r"}$"
    if name == "r_tilde":
        return r"$\tilde{r}$"
    return name


for _dim in ALL_DIMS_PARAM:
    print(f"\n── Parameters: ℓ={_dim} ──")
    _m, _src = load_ep5000_model(_dim)
    if _m is None:
        print(f"  No model for ℓ={_dim}, skipping.")
        continue

    # create dim subfolder inside parameters/
    _dim_dir = os.path.join(PARAMS_DIR, f"dim{_dim}")
    os.makedirs(_dim_dir, exist_ok=True)

    if _dim in dim_S_hat:
        _mask = finite_mask(X_train, dim_S_hat[_dim])
    else:
        with torch.no_grad():
            _S_tmp, _, _, _, _, _, _, _, _, _ = _m(X_train)
        _mask = finite_mask(X_train, _S_tmp)

    df_p = extract_parameters(_m, X_train, meta_train, _mask)

    # Build list of parameter columns in display order
    d = _dim
    mu_cols    = [f"mu_{k+1}"    for k in range(d)]
    sig_cols   = [f"sigma_{k+1}" for k in range(d)]
    rho_cols   = [f"rho_{i+1}{j+1}"
                  for i in range(d) for j in range(i + 1, d)]
    param_cols = mu_cols + sig_cols + rho_cols + ["r_tilde"]

    for col in param_cols:
        fig, ax = plt.subplots(figsize=(5, 3.5))

        for ccy in CCY_ORDER:
            sub = df_p[df_p["ccy"] == ccy].sort_values("as_of_date")
            if sub.empty:
                continue
            ax.plot(sub["as_of_date"], sub[col],
                    color=_ccy_colors[ccy], linewidth=0.8,
                    alpha=0.75)

        ax.set_title(_param_label(col), fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        # save to parameters/dim{N}/{col}.png
        out_path = os.path.join(_dim_dir, col + ".png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Q outputs saved to:  {FIGURES_OUT}")
print(f"Param plots saved to: {PARAMS_DIR}")
print(f"{'='*60}")
