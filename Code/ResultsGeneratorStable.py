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

# ── path setup ─────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(REPO_ROOT)
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import custom_palette, set_paper_theme
from Code import config
config.VARIANT = "stable"

# ── output directory ────────────────────────────────────────────────────────────
FIGURES_OUT = os.path.join(REPO_ROOT, "Figures", "thesis_results", "AutoencoderPerformanceStable")
os.makedirs(FIGURES_OUT, exist_ok=True)

# ── constants ───────────────────────────────────────────────────────────────────
CCY_ORDER  = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
DIM_COLORS = {1: custom_palette[8], 2: custom_palette[4],
              3: custom_palette[0], 4: custom_palette[6]}

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
    oos_ae = load_rolling_oos_per_ccy(dim)
    if oos_ae is not None:
        oos_ae["Time (min)"] = load_rolling_train_time_min(dim)
        rows_q4[rf"AE stable $\ell$={dim}"] = oos_ae
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

print("\nResultsGeneratorStable complete.")
