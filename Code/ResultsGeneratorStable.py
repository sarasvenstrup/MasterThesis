# ResultsGeneratorStable.py
# Generates thesis result figures and tables from STABLE variant rolling OOS CSVs.
# Run from repo root: python Code/ResultsGeneratorStable.py
#
# Outputs:
#   Figures/thesis_results/AutoencoderPerformanceStable/   → all .png figures + .csv tables
#
# Figures generated:
#   Q2b_stable  — Average rolling OOS RMSE bar chart (stable AE vs EKF DNS)
#   Q3b_stable  — Rolling OOS RMSE over time (stable dims 2, 3, 4)
#   Q4a_stable  — AE stable vs EKF DNS OOS RMSE table

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

# ── path setup ─────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(REPO_ROOT)
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import custom_palette, set_paper_theme, my_data, TARGET_TENORS
from Code.model.full_model import FullModel
from Code import config
config.VARIANT = "stable"

# ── output directory ────────────────────────────────────────────────────────────
FIGURES_OUT = os.path.join(REPO_ROOT, "Figures", "thesis_results", "AutoencoderPerformanceStable")
os.makedirs(FIGURES_OUT, exist_ok=True)

# ── constants ───────────────────────────────────────────────────────────────────
CCY_ORDER  = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
DIM_COLORS = {1: custom_palette[8], 2: custom_palette[4],
              3: custom_palette[0], 4: custom_palette[6]}

# dim=2 stable ep2500 checkpoint
STABLE_DIM       = 2
STABLE_EP        = 2500
TRAIN_START      = "2010-01-01"
TRAIN_END        = "2020-12-31"

PARAMS_DIR = os.path.join(FIGURES_OUT, "parameters_dim2")
os.makedirs(PARAMS_DIR, exist_ok=True)

ROLL_SUBDIR            = "train5Y_test6M_step6M"
ROLL_DIVERGE_THRESHOLD = 100.0
_ROLL_FALLBACK_SUBDIR  = "train3Y_test3M_step6M"
ROLL_EP                = 2500

EVENTS = {
    "GFC\n(15 Sep 2008)":       "2008-09-15",
    "ECB QE\n(22 Jan 2015)":    "2015-01-22",
    "COVID\n(1 Mar 2020)":      "2020-03-01",
    "Rate hikes\n(1 Mar 2022)": "2022-03-01",
}

set_paper_theme()

# ── save helpers ────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = os.path.join(FIGURES_OUT, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")

def save_table(df, name):
    path = os.path.join(FIGURES_OUT, f"{name}.csv")
    df.to_csv(path, index_label="index")
    print(f"  Saved → {path}")

# ── rolling helpers (stable variant) ───────────────────────────────────────────
def load_rolling_df(dim):
    """Return full rolling CSV DataFrame for stable dim."""
    path = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Roll",
                        f"OOS_roll_dim{dim}_stable",
                        ROLL_SUBDIR, f"ep{ROLL_EP}",
                        f"oos_rolling_bbg_dim{dim}_{ROLL_SUBDIR}.csv")
    if not os.path.exists(path):
        path = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Roll",
                            f"OOS_roll_dim{dim}_stable",
                            _ROLL_FALLBACK_SUBDIR, f"ep{ROLL_EP}",
                            f"oos_rolling_bbg_dim{dim}_train3Y_test3M_step6M.csv")
        if not os.path.exists(path):
            return None
        print(f"  [dim={dim}] Using fallback rolling CSV: {_ROLL_FALLBACK_SUBDIR}")
    df = pd.read_csv(path)
    df["test_start"] = pd.to_datetime(df["test_start"])
    return df

def load_rolling_avg(dim):
    """Average OOS RMSE across valid stable rolling windows."""
    df = load_rolling_df(dim)
    if df is None:
        return None
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    if len(valid) == 0:
        return None
    return float(valid["avg_rmse_bps"].mean())

def load_rolling_oos_per_ccy(dim):
    """Average per-currency OOS RMSE across valid stable rolling windows."""
    df = load_rolling_df(dim)
    if df is None:
        return None
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    if len(valid) == 0:
        return None
    result = pd.Series({
        ccy: float(valid[f"rmse_bps_{ccy}"].mean())
        for ccy in CCY_ORDER if f"rmse_bps_{ccy}" in valid.columns
    })
    result["Average"] = float(valid["avg_rmse_bps"].mean())
    return result

def load_rolling_train_time_min(dim):
    """Average training time (minutes) per stable rolling window."""
    df = load_rolling_df(dim)
    if df is None or "time_train_sec" not in df.columns:
        return None
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    if len(valid) == 0:
        return None
    return float(valid["time_train_sec"].mean()) / 60.0

def load_baseline_rolling_df(dim):
    """Return full rolling CSV DataFrame for baseline dim."""
    path = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Roll",
                        f"OOS_roll_dim{dim}_baseline",
                        ROLL_SUBDIR, f"ep{ROLL_EP}",
                        f"oos_rolling_bbg_dim{dim}_{ROLL_SUBDIR}.csv")
    if not os.path.exists(path):
        path = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Roll",
                            f"OOS_roll_dim{dim}_baseline",
                            _ROLL_FALLBACK_SUBDIR, f"ep{ROLL_EP}",
                            f"oos_rolling_bbg_dim{dim}_train3Y_test3M_step6M.csv")
        if not os.path.exists(path):
            return None
    df = pd.read_csv(path)
    df["test_start"] = pd.to_datetime(df["test_start"])
    return df

def load_baseline_rolling_oos_per_ccy(dim):
    """Average per-currency OOS RMSE across valid baseline rolling windows."""
    df = load_baseline_rolling_df(dim)
    if df is None:
        return None
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    if len(valid) == 0:
        return None
    result = pd.Series({
        ccy: float(valid[f"rmse_bps_{ccy}"].mean())
        for ccy in CCY_ORDER if f"rmse_bps_{ccy}" in valid.columns
    })
    result["Average"] = float(valid["avg_rmse_bps"].mean())
    return result

def load_baseline_rolling_train_time_min(dim):
    """Average training time (minutes) per baseline rolling window."""
    df = load_baseline_rolling_df(dim)
    if df is None or "time_train_sec" not in df.columns:
        return None
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    if len(valid) == 0:
        return None
    return float(valid["time_train_sec"].mean()) / 60.0

# ── EKF rolling helpers (shared — same as baseline) ────────────────────────────
def load_ekf_rolling_df(n_factors):
    path = os.path.join(REPO_ROOT, "Figures", "KalmanBenchmarkResults",
                        "ekf_dns_rolling",
                        f"oos_rolling_ekf_{n_factors}f_train5Y_test6M_step6M.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["test_start"] = pd.to_datetime(df["test_start"])
    return df

def load_ekf_rolling_avg(n_factors):
    df = load_ekf_rolling_df(n_factors)
    if df is None:
        return None
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    if len(valid) == 0:
        return None
    return float(valid["avg_rmse_bps"].mean())

def load_ekf_rolling_oos_per_ccy(dim):
    df = load_ekf_rolling_df(dim)
    if df is None:
        return None
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    if len(valid) == 0:
        return None
    result = pd.Series({
        ccy: float(valid[ccy].mean())
        for ccy in CCY_ORDER if ccy in valid.columns
    })
    result["Average"] = float(valid["avg_rmse_bps"].mean())
    return result

def load_ekf_rolling_train_time_min(dim):
    df = load_ekf_rolling_df(dim)
    if df is None or "time_train_sec" not in df.columns:
        return None
    valid = df[df["avg_rmse_bps"] <= ROLL_DIVERGE_THRESHOLD]
    if len(valid) == 0:
        return None
    return float(valid["time_train_sec"].mean()) / 60.0

# ─────────────────────────────────────────────────────────────────────────────
# Q2b_stable — Bar chart: average rolling OOS RMSE by dim (stable AE vs EKF)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q2b_stable: Average rolling OOS RMSE bar chart ──")

roll_avgs     = {d: load_rolling_avg(d)    for d in [2, 3, 4]}
ekf_roll_avgs = {d: load_ekf_rolling_avg(d) for d in [2, 3, 4]}
roll_avgs     = {d: v for d, v in roll_avgs.items()     if v is not None}
ekf_roll_avgs = {d: v for d, v in ekf_roll_avgs.items() if v is not None}

if len(roll_avgs) >= 2:
    _dims = [d for d in [2, 3, 4] if d in roll_avgs]
    _x    = np.arange(len(_dims))
    _w    = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    _ae_bars = ax.bar(_x - _w/2, [roll_avgs[d] for d in _dims],
                      width=_w, color=[DIM_COLORS[d] for d in _dims],
                      edgecolor="none", label="Autoencoder (stable)")
    _ekf_vals = [ekf_roll_avgs.get(d, np.nan) for d in _dims]
    _ekf_bars = ax.bar(_x + _w/2, _ekf_vals,
                       width=_w, color="lightgray",
                       edgecolor="none", label="EKF DNS")

    for bar, val in zip(_ae_bars, [roll_avgs[d] for d in _dims]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(_ekf_bars, _ekf_vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(_x)
    ax.set_xticklabels([f"$\\ell={d}$" for d in _dims], fontsize=10)
    ax.set_ylabel("Average Rolling OOS RMSE (bps)", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    save_fig(fig, "Q2b_rolling_oos_vs_dim_stable")
else:
    print(f"  SKIPPED — only {len(roll_avgs)}/3 rolling results available.")

# ─────────────────────────────────────────────────────────────────────────────
# Q3b_stable — Rolling OOS RMSE over time (dims 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q3b_stable: Rolling RMSE over time (ℓ=2,3,4) ──")

CLIP         = 100
BAR_WIDTH    = pd.Timedelta(days=50)
_Q3b_COLORS  = {d: DIM_COLORS[d] for d in [2, 3, 4]}
DIM_OFFSETS  = {2: pd.Timedelta(days=-35), 3: pd.Timedelta(days=0), 4: pd.Timedelta(days=35)}

_any_plotted = False
fig, ax = plt.subplots(figsize=(11, 4.5))

for _dim, _col in _Q3b_COLORS.items():
    _df = load_rolling_df(_dim)
    if _df is None:
        print(f"  SKIPPED dim={_dim} — no rolling CSV found")
        continue
    _any_plotted = True

    avg_clipped = _df["avg_rmse_bps"].where(_df["avg_rmse_bps"] <= CLIP)
    valid_avg   = avg_clipped.notna()
    ax.plot(_df.loc[valid_avg, "test_start"], avg_clipped[valid_avg],
            linewidth=1.8, color=_col, label=f"$\\ell={_dim}$", zorder=5)

    ccy_cols     = [f"rmse_bps_{c}" for c in CCY_ORDER if f"rmse_bps_{c}" in _df.columns]
    explode_max  = _df[ccy_cols].max(axis=1)
    explode_mask = explode_max > CLIP
    _bar_added   = False
    for _, row in _df[explode_mask].iterrows():
        max_val  = explode_max[row.name]
        bar_date = row["test_start"] + DIM_OFFSETS[_dim]
        ax.bar(bar_date, CLIP, width=BAR_WIDTH,
               color=_col, alpha=0.30, zorder=3,
               label=f"$\\ell={_dim}$ exploded" if not _bar_added else "_nolegend_")
        ax.text(bar_date, CLIP * 0.97,
                f"{max_val:.0f}", fontsize=7, ha="center", va="top",
                rotation=90, color=_col, fontweight="bold", zorder=4)
        _bar_added = True

if not _any_plotted:
    print("  Q3b_stable SKIPPED — no rolling CSVs found for any dim")
    plt.close(fig)
else:
    ax.set_ylim(0, CLIP)
    _loaded = [load_rolling_df(d) for d in _Q3b_COLORS if load_rolling_df(d) is not None]
    _x_min  = min(d["test_start"].min() for d in _loaded)
    _x_max  = max(d["test_start"].max() for d in _loaded)
    for label, date_str in EVENTS.items():
        d = pd.Timestamp(date_str)
        if _x_min <= d <= _x_max:
            ax.axvline(d, color="0.5", linewidth=1.0, linestyle="--")
            ax.text(d, CLIP, label, fontsize=10, ha="center", va="bottom", color="0.4")
    ax.set_ylabel("OOS RMSE (bps)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, "Q3b_rolling_rmse_over_time_stable")

# ─────────────────────────────────────────────────────────────────────────────
# Q4a_stable — Table: stable AE vs EKF DNS, rolling OOS RMSE per currency
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q4a_stable: Stable AE vs EKF DNS OOS table ──")

rows_q4 = {}
for dim in [2, 3, 4]:
    oos_base = load_baseline_rolling_oos_per_ccy(dim)
    if oos_base is not None:
        oos_base["Time (min)"] = load_baseline_rolling_train_time_min(dim)
        rows_q4[rf"AE baseline $\ell$={dim}"] = oos_base
    oos_stable = load_rolling_oos_per_ccy(dim)
    if oos_stable is not None:
        oos_stable["Time (min)"] = load_rolling_train_time_min(dim)
        rows_q4[rf"AE stable $\ell$={dim}"] = oos_stable
    oos_k = load_ekf_rolling_oos_per_ccy(dim)
    if oos_k is not None:
        oos_k["Time (min)"] = load_ekf_rolling_train_time_min(dim)
        rows_q4[rf"EKF DNS $\ell$={dim}"] = oos_k

if rows_q4:
    table_q4a = pd.DataFrame(rows_q4).T
    table_q4a = table_q4a[[c for c in CCY_ORDER + ["Average", "Time (min)"]
                            if c in table_q4a.columns]]
    table_q4a = table_q4a.round(2)
    save_table(table_q4a, "Q4a_AE_vs_Kalman_OOS_stable")
    print(table_q4a.to_string())
else:
    print("  SKIPPED — no rolling OOS data found for stable variant.")

# ─────────────────────────────────────────────────────────────────────────────
# Load stable dim=2 ep2500 model + training data
# ─────────────────────────────────────────────────────────────────────────────
def load_stable_dim2_model():
    ckpt = os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                        f"dim{STABLE_DIM}_stable", f"ep{STABLE_EP}",
                        f"checkpoint_dim{STABLE_DIM}_ep{STABLE_EP}.pt")
    if not os.path.exists(ckpt):
        print(f"  ⚠️  Checkpoint not found: {ckpt}")
        return None
    state = torch.load(ckpt, map_location="cpu")
    model = FullModel(latent_dim=STABLE_DIM)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    result = model.load_state_dict(sd, strict=False)
    if result.unexpected_keys:
        print(f"  [load] dropped old params: {result.unexpected_keys}")
    model.eval()
    print(f"  Loaded stable dim={STABLE_DIM} ep={STABLE_EP} checkpoint.")
    return model

def _param_label(name):
    if name.startswith("mu_"):
        k = name.split("_")[1]; return r"$\mu_{" + k + r"}$"
    if name.startswith("sigma_"):
        k = name.split("_")[1]; return r"$\sigma_{" + k + r"}$"
    if name.startswith("rho_"):
        ij = name.split("_")[1]; return r"$\rho_{" + ",".join(ij) + r"}$"
    if name == "r_tilde":
        return r"$\tilde{r}$"
    return name

def finite_mask(X, S):
    return torch.isfinite(X).all(1) & torch.isfinite(S).all(1)

def extract_parameters(model, X_data, meta_df, mask):
    model.eval()
    with torch.no_grad():
        X_m   = X_data[mask]
        z     = model.encoder(X_m)
        mu    = model.K(z)
        sigmas, rhos = model.H(z)
        r_til = model.R(z).squeeze(-1)
    d      = model.latent_dim
    rec    = meta_df.loc[mask.numpy()].copy().reset_index(drop=True)
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

@torch.no_grad()
def extract_sharpe(model, X, batch=256):
    sr_list = []
    for i in range(0, len(X), batch):
        xb = X[i:i+batch]
        _, aux = model(xb, return_aux=True, do_arb_checks=True)
        sr_list.append(aux["arb"]["SR_tau"].cpu())
    return torch.cat(sr_list, dim=0)

_ccy_colors = {ccy: custom_palette[i]
               for i, ccy in enumerate(CCY_ORDER)}

# ─────────────────────────────────────────────────────────────────────────────
# Load data + model
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data for stable dim=2 parameter / Sharpe plots...")
meta_train, X_train, *_ = my_data()

_stable_model = load_stable_dim2_model()

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — stable dim=2
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Parameters: stable ℓ=2 ──")
if _stable_model is None:
    print("  ⚠️  Skipped — no checkpoint.")
else:
    with torch.no_grad():
        _S_tmp = _stable_model(X_train)
    _mask = finite_mask(X_train, _S_tmp)
    df_p  = extract_parameters(_stable_model, X_train, meta_train, _mask)

    d = STABLE_DIM
    mu_cols  = [f"mu_{k+1}"    for k in range(d)]
    sig_cols = [f"sigma_{k+1}" for k in range(d)]
    rho_cols = [f"rho_{i+1}{j+1}"
                for i in range(d) for j in range(i + 1, d)]
    param_cols = mu_cols + sig_cols + rho_cols + ["r_tilde"]

    for col in param_cols:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for ccy in CCY_ORDER:
            sub = df_p[df_p["ccy"] == ccy].sort_values("as_of_date")
            if sub.empty:
                continue
            ax.plot(sub["as_of_date"], sub[col],
                    color=_ccy_colors[ccy], linewidth=0.8, alpha=0.75)
        ax.set_title(_param_label(col), fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out_path = os.path.join(PARAMS_DIR, f"param_{col}_stable_dim2.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {len(param_cols)} parameter plots → {PARAMS_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Q7_sharpe_stable — IS Sharpe ratio by tenor, stable dim=2
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q7_sharpe_stable: IS Sharpe ratio (stable ℓ=2) ──")
TAU_GRID = np.arange(1, 31)

if _stable_model is None:
    print("  ⚠️  Skipped — no checkpoint.")
else:
    SR_all = extract_sharpe(_stable_model, X_train)   # (N, 30)
    x_fin  = torch.isfinite(X_train).all(1)
    SR_np  = SR_all[x_fin].numpy()
    meta_fin = meta_train[x_fin.numpy()].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for ccy in CCY_ORDER:
        ccy_mask = (meta_fin["ccy"] == ccy).values
        if ccy_mask.sum() == 0:
            continue
        sr_ccy = np.nanmedian(SR_np[ccy_mask], axis=0)   # (30,) median over dates
        ax.plot(TAU_GRID, sr_ccy, color=_ccy_colors[ccy],
                linewidth=1.0, alpha=0.85, label=ccy)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(rf"IS Sharpe ratio by tenor and currency — stable $\ell={STABLE_DIM}$", fontsize=11)
    ax.set_xticks(TAU_GRID[::2])
    ax.legend(fontsize=7, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "Q7_sharpe_ratio_IS_stable_dim2")
    print("  done")

print("\nResultsGeneratorStable complete.")
