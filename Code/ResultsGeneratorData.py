# ResultsGeneratorData.py
# Generates data visualisation figures and tables for the thesis.
# Run from repo root: python Code/ResultsGeneratorData.py
#
# Outputs:
#   Figures/thesis_results/DataVisualizations/   → all .png figures and .csv tables

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── path setup ─────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(REPO_ROOT)   # go up from Code/ to repo root
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS, set_paper_theme

# ── output directory ───────────────────────────────────────────────────────────
FIGURES_OUT = os.path.join(REPO_ROOT, "Figures", "thesis_results", "DataVisualizations")
os.makedirs(FIGURES_OUT, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
CCY_ORDER = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

EVENTS = {
    "GFC\n(15 Sep 2008)":       "2008-09-15",
    "QE\n(22 Jan 2015)":    "2015-01-22",
    "COVID\n(1 Mar 2020)":      "2020-03-01",
    "Inflation\n(1 Mar 2022)": "2022-03-01",
}

# ── apply paper theme ──────────────────────────────────────────────────────────
set_paper_theme()
currency_color_map = {ccy: plt.cm.tab10.colors[i % 10]
                      for i, ccy in enumerate(CCY_ORDER)}

# ── save helpers ───────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = os.path.join(FIGURES_OUT, name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def save_table(df, name):
    path = os.path.join(FIGURES_OUT, name + ".csv")
    df.to_csv(path)
    print(f"  Saved: {path}")

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
_extended_tenors = sorted(set(list(TARGET_TENORS) + [7]))
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data("bbg", _extended_tenors)
df_wide_all["as_of_date"] = pd.to_datetime(df_wide_all["as_of_date"])
tenor_cols = _extended_tenors
print(f"  Loaded {len(df_wide_all)} observations (full history)")

# ─────────────────────────────────────────────────────────────────────────────
# Table: Missing data summary per currency
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Missing data table ──")

rows = []
for ccy in CCY_ORDER:
    ccy_df = df_wide_all[df_wide_all["ccy"] == ccy].copy()
    if ccy_df.empty:
        rows.append({
            "Currency": ccy,
            "First date": "N/A",
            "Last date":  "N/A",
            "Total obs":  0,
            "Missing curves": "N/A",
            "Incomplete curves": 0,
        })
        continue

    first_date = ccy_df["as_of_date"].min().strftime("%Y-%m-%d")
    last_date  = ccy_df["as_of_date"].max().strftime("%Y-%m-%d")
    total_obs  = len(ccy_df)

    # Build full monthly calendar between first and last date
    monthly_range = pd.date_range(start=first_date, end=last_date, freq="MS")
    present_months = set(ccy_df["as_of_date"].dt.to_period("M"))
    missing_curves = len([d for d in monthly_range
                          if pd.Timestamp(d).to_period("M") not in present_months])

    # Rows where curve exists but has at least one NaN tenor
    tenor_data = ccy_df[tenor_cols]
    incomplete_curves = int(tenor_data.isnull().any(axis=1).sum())

    rows.append({
        "Currency":          ccy,
        "First date":        first_date,
        "Last date":         last_date,
        "Total obs":         total_obs,
        "Missing curves":    missing_curves,
        "Incomplete curves": incomplete_curves,
    })

table_missing = pd.DataFrame(rows).set_index("Currency")
save_table(table_missing, "D1_missing_data_summary")
print(table_missing.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Figure: Swap curves on 2016-08-31 (left) + 10Y time series (right)
# ─────────────────────────────────────────────────────────────────────────────
_target_date = pd.Timestamp("2016-08-31")
_scale = 0.01 if SCALE_IS_PERCENT else 1.0  # convert to decimal

# ── D2a: Swap curves on 2016-08-31 ───────────────────────────────────────────
print("\n── D2a: Swap curves ──")

fig, ax_curves = plt.subplots(figsize=(7, 4))

for ccy in CCY_ORDER:
    ccy_df = df_wide_all[df_wide_all["ccy"] == ccy].copy()
    if ccy_df.empty:
        continue
    idx = (ccy_df["as_of_date"] - _target_date).abs().argmin()
    row = ccy_df.iloc[idx]
    rates = [float(row[t]) * _scale for t in tenor_cols]
    ax_curves.plot(tenor_cols, rates,
                   color=currency_color_map[ccy],
                   linewidth=1.5, label=ccy)

ax_curves.set_xlabel("Maturity", fontsize=10)
ax_curves.set_ylabel("Swap rate", fontsize=10)
ax_curves.set_xticks(tenor_cols)
ax_curves.set_xticklabels([str(int(t)) for t in tenor_cols], fontsize=8)

fig.tight_layout()
fig.legend(
    *ax_curves.get_legend_handles_labels(),
    loc="lower center", bbox_to_anchor=(0.5, -0.02),
    ncol=9, frameon=False, fontsize=8,
)
fig.subplots_adjust(bottom=0.14)
save_fig(fig, "D2a_swap_curves")

# ── D2b: 10Y swap rate time series ───────────────────────────────────────────
print("\n── D2b: 10Y time series ──")

_tenor_10y = 10
fig, ax_10y = plt.subplots(figsize=(9, 4))

for ccy in CCY_ORDER:
    ccy_df = df_wide_all[df_wide_all["ccy"] == ccy].copy()
    ccy_df["as_of_date"] = pd.to_datetime(ccy_df["as_of_date"])
    if ccy_df.empty or _tenor_10y not in ccy_df.columns:
        continue
    ccy_df = ccy_df.sort_values("as_of_date")
    ax_10y.plot(ccy_df["as_of_date"],
                ccy_df[_tenor_10y] * _scale,
                color=currency_color_map[ccy],
                linewidth=1.0, label=ccy)
    # circle markers at the first and last observation
    for row in [ccy_df.iloc[0], ccy_df.iloc[-1]]:
        ax_10y.plot(row["as_of_date"], row[_tenor_10y] * _scale,
                    marker="o", markersize=5, color=currency_color_map[ccy],
                    markeredgecolor="white", markeredgewidth=0.6,
                    linestyle="none", zorder=5)

# event lines
for label, date_str in EVENTS.items():
    d = pd.Timestamp(date_str)
    ax_10y.axvline(d, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_10y.text(d, ax_10y.get_ylim()[1] if ax_10y.get_ylim()[1] != 1.0 else 0.08,
                label, fontsize=7, ha="center", va="bottom", color="dimgray")

ax_10y.set_ylabel("10Y swap rate", fontsize=10)

fig.tight_layout()
fig.legend(
    *ax_10y.get_legend_handles_labels(),
    loc="lower center", bbox_to_anchor=(0.5, -0.02),
    ncol=9, frameon=False, fontsize=8,
)
fig.subplots_adjust(bottom=0.12)
save_fig(fig, "D2b_10y_timeseries")

print("\nAll data figures saved.")
