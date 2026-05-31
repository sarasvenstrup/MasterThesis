"""
Compare model-implied swaption volatilities with market data.

Loads pre-computed model prices (from price_swaptions.py) and market
normal vols (EUR SwapNVol OIS), merges them, computes error metrics,
and saves time-series comparison plots.

Workflow:
    1. python price_swaptions.py          # slow — runs once, saves model_prices.csv
    2. python compare_with_market.py      # fast — loads CSV, plots, metrics

Usage:
    python compare_with_market.py
    python compare_with_market.py --model_prices path/to/model_prices.csv
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Add parent directories to path
_HERE = Path(__file__).resolve().parent
_CODE_ROOT = _HERE.parent
_UTILS_DIR = _CODE_ROOT / "utils"

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from helpers import PlotConfig, save_figure

import sys as _sys
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from Code.load_swapdata import custom_palette

# =============================================================================
# CONFIGURATION  — edit these to change behaviour on right-click Run
# =============================================================================

DEFAULT_CCY     = 'EUR'   # 'EUR' or 'USD'
DEFAULT_N_PATHS = 10_000

# NVol sheet name per currency in SwapVol.xlsx
NVOL_SHEETS: dict = {
    'EUR': 'EUR SwapNVol OIS',
    'USD': 'USD SwapNVol OIS',
}


# =============================================================================
# MARKET DATA LOADING
# =============================================================================

def load_market_vols(excel_path: Path, sheet_name: str = 'EUR SwapNVol OIS') -> pd.DataFrame:
    """
    Load market swaption normal volatilities (bps) from SwapVol.xlsx.
    Uses the NVol sheet because the model outputs normal (Bachelier) vols in bps.

    The file layout (read with header=None):
      Row 0 : 'Period', 'M', 11, NaT, 15, NaT, 110, NaT, 51, NaT, 55, NaT,
               510, NaT, 101, NaT, 105, NaT, 1010
      Rows 1-4 : metadata / ticker rows (skip)
      Row 5+  : date in col 1, vol in the same column as the integer key

    Returns DataFrame with columns:
        - date: datetime
        - expiry: int (years)
        - tenor: int (years)
        - market_vol_bps: float
    """
    print(f"\nLoading market data from: {excel_path}")
    print(f"Sheet: {sheet_name}")

    # Read WITHOUT auto-header so we can inspect row 0 ourselves
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # Map integer key → (expiry, tenor)
    structure_map = {
        11:   (1, 1),
        15:   (1, 5),
        110:  (1, 10),
        51:   (5, 1),
        55:   (5, 5),
        510:  (5, 10),
        101:  (10, 1),
        105:  (10, 5),
        1010: (10, 10),
    }

    # Identify column indices from row 0
    # Date column is the one headed 'M' (column 1)
    date_col_idx = 1
    vol_col_map: dict = {}  # col_idx → (expiry, tenor)

    for col_idx, header_val in enumerate(df_raw.iloc[0]):
        # Try to coerce to int (handles both int and float like 11.0)
        try:
            key = int(header_val)
        except (ValueError, TypeError):
            continue
        if key in structure_map:
            vol_col_map[col_idx] = structure_map[key]

    if not vol_col_map:
        raise ValueError(
            f"No recognised structure columns found in row 0 of sheet '{sheet_name}'. "
            f"Row 0 values: {list(df_raw.iloc[0])}"
        )

    # Data rows start at index 5 (rows 0-4 are headers / metadata)
    results = []
    for row_idx in range(5, len(df_raw)):
        date_val = df_raw.iloc[row_idx, date_col_idx]
        try:
            if pd.isna(date_val):
                continue
            date = pd.to_datetime(date_val).normalize()
        except Exception:
            continue

        for col_idx, (expiry, tenor) in vol_col_map.items():
            vol_val = df_raw.iloc[row_idx, col_idx]
            try:
                if pd.isna(vol_val):
                    continue
                vol_bps = float(vol_val)
            except Exception:
                continue

            results.append({
                'date': date,
                'expiry': expiry,
                'tenor': tenor,
                'market_vol_bps': vol_bps,
            })

    df_market = pd.DataFrame(results)

    if df_market.empty:
        raise ValueError("No market data found in Excel file!")

    df_market = df_market.sort_values(['date', 'expiry', 'tenor']).drop_duplicates()

    print(f"Loaded {len(df_market)} market vol quotes")
    print(f"  Date range: {df_market['date'].min().date()} to {df_market['date'].max().date()}")
    print(f"  Structures: {df_market.groupby(['expiry', 'tenor']).size().to_dict()}")

    return df_market


# =============================================================================
# COMPARISON AND METRICS
# =============================================================================

def compute_comparison_metrics(df_comparison: pd.DataFrame) -> Dict:
    """Compute error metrics between model and market vols."""
    
    model_vols = df_comparison['model_vol_bps'].values
    market_vols = df_comparison['market_vol_bps'].values
    
    errors = model_vols - market_vols
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / market_vols * 100
    
    metrics = {
        'n_observations': len(df_comparison),
        'mean_absolute_error': np.mean(abs_errors),
        'median_absolute_error': np.median(abs_errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mean_relative_error_pct': np.mean(rel_errors),
        'correlation': pearsonr(model_vols, market_vols)[0] if len(model_vols) > 1 else np.nan,
        'r_squared': pearsonr(model_vols, market_vols)[0]**2 if len(model_vols) > 1 else np.nan,
    }
    
    return metrics


def per_cell_summary(df_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-(expiry, tenor) average model vol, market vol, absolute and
    relative error.  Used both for terminal reporting and for the LaTeX
    table written by `write_latex_table`.
    """
    rows = []
    for (exp, ten), grp in df_comparison.groupby(['expiry', 'tenor']):
        model = grp['model_vol_bps'].mean()
        market = grp['market_vol_bps'].mean()
        abs_err = (grp['model_vol_bps'] - grp['market_vol_bps']).abs().mean()
        # Average of pointwise relative error (in %), signed
        rel_err = ((grp['model_vol_bps'] - grp['market_vol_bps']) /
                   grp['market_vol_bps'] * 100).mean()
        rows.append({
            'expiry': int(exp),
            'tenor': int(ten),
            'model_bps': model,
            'market_bps': market,
            'abs_err_bps': abs_err,
            'rel_err_pct': rel_err,
            'n_obs': len(grp),
        })
    return pd.DataFrame(rows).sort_values(['expiry', 'tenor']).reset_index(drop=True)


def write_latex_table(df_cells: pd.DataFrame,
                      metrics: Dict,
                      tex_path: Path,
                      ccy: str,
                      n_paths: int,
                      n_dates: int) -> None:
    """
    Write a stand-alone LaTeX `table` environment that can be `\\input{}`'d
    from the thesis.  The table reports per-cell averages of model vol,
    market vol, absolute error in bps and signed relative error in per
    cent, with a caption summarising the run.
    """
    lines: List[str] = []
    lines.append("% Auto-generated by Code/Pricing/compare_with_market.py — do not edit by hand.")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    caption = (
        f"Average normal implied volatility produced by the Monte Carlo "
        f"straddle pricer (column~3) versus the Bloomberg "
        f"\\texttt{{{ccy}SV}} quote (column~4), per swaption cell. "
        f"Absolute errors are in basis points. "
        f"dim-$4$ stable checkpoint, $N_{{\\mathrm{{MC}}}}={n_paths:,}$, "
        f"{n_dates} as-of dates."
    ).replace(",", "{,}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(r"  \label{tab:market_comparison}")
    lines.append(r"  \begin{tabular}{rrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Expiry & Tenor & Model & Market & $\lvert\Delta\rvert$ \\")
    lines.append(r"    (yr) & (yr) & (bps) & (bps) & (bps) \\")
    lines.append(r"    \midrule")
    for _, r in df_cells.iterrows():
        lines.append(
            f"    {int(r['expiry']):>2} & {int(r['tenor']):>2} & "
            f"${r['model_bps']:.1f}$ & ${r['market_bps']:.1f}$ & "
            f"${r['abs_err_bps']:.1f}$ \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    # Footnote-style summary below table
    summary = (
        f"\\\\[0.5em]\n  \\footnotesize Across all "
        f"{metrics['n_observations']:,} observations: MAE "
        f"${metrics['mean_absolute_error']:.0f}$ bps, median "
        f"${metrics['median_absolute_error']:.0f}$ bps, RMSE "
        f"${metrics['rmse']:.0f}$ bps, correlation "
        f"${metrics['correlation']:.2f}$, $R^2={metrics['r_squared']:.2f}$."
    ).replace(",", "{,}")
    lines.append(summary)
    lines.append(r"\end{table}")
    lines.append("")

    tex_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_model_vs_market(df_comparison: pd.DataFrame, save_dir: Path):
    """Create comparison plots in the visualize_swapvol time-series style."""

    palette = sns.color_palette("husl", 9)
    structures = sorted(df_comparison.groupby(['expiry', 'tenor']).groups.keys())
    plot_cfg = PlotConfig(figures_dir=str(save_dir), dpi=300)

    # colours from shared custom palette
    _col_market = custom_palette[4]
    _col_model  = custom_palette[0]

    # ------------------------------------------------------------------
    # Plot 1: Time series — market vs model, one subplot per structure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    # compute shared y-axis range across all structures
    _y_min = df_comparison[['market_vol_bps', 'model_vol_bps']].min().min()
    _y_max = df_comparison[['market_vol_bps', 'model_vol_bps']].max().max()
    _y_pad = (_y_max - _y_min) * 0.05
    _leg_h, _leg_l = None, None

    for idx, (expiry, tenor) in enumerate(structures[:9]):
        ax = axes[idx]
        subset = df_comparison[
            (df_comparison['expiry'] == expiry) & (df_comparison['tenor'] == tenor)
        ].sort_values('date')

        if not subset.empty:
            l1, = ax.plot(subset['date'], subset['market_vol_bps'],
                          linewidth=1.5, alpha=0.9,
                          label='Market', color=_col_market)
            l2, = ax.plot(subset['date'], subset['model_vol_bps'],
                          linewidth=1.5, alpha=0.9,
                          linestyle='-', label='Model', color=_col_model)
            if _leg_h is None:
                _leg_h = [l1, l2]
                _leg_l = ['Market', 'Model']

        ax.set_title(f'{expiry}Y × {tenor}Y', fontsize=16, fontweight='bold')
        ax.set_ylim(_y_min - _y_pad, _y_max + _y_pad)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, alpha=0.3)

        # y-axis label only for left column
        if idx % 3 == 0:
            ax.set_ylabel('Vol (bps)', fontsize=15)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

        # x-axis label only for bottom row
        if idx >= 6:
            ax.set_xlabel('Date', fontsize=15)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=13)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)

    # single shared legend below all plots
    if _leg_h:
        fig.legend(_leg_h, _leg_l, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                   ncol=2, frameon=False, fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    save_figure(fig, plot_cfg, 'model_vs_market_timeseries')
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Scatter — model vs market coloured by structure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    for (expiry, tenor), color in zip(structures, palette):
        subset = df_comparison[
            (df_comparison['expiry'] == expiry) & (df_comparison['tenor'] == tenor)
        ]
        ax.scatter(subset['market_vol_bps'], subset['model_vol_bps'],
                   label=f'{expiry}Y × {tenor}Y', alpha=0.7, s=60, color=color)

    lo = min(df_comparison['market_vol_bps'].min(), df_comparison['model_vol_bps'].min()) * 0.95
    hi = max(df_comparison['market_vol_bps'].max(), df_comparison['model_vol_bps'].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, alpha=0.5, label='Perfect fit')

    ax.set_xlabel('Market Vol (bps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Vol (bps)', fontsize=12, fontweight='bold')
    ax.set_title('Model vs Market — Scatter', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, plot_cfg, 'model_vs_market_scatter')
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Pricing error time series, one subplot per structure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, ((expiry, tenor), color) in enumerate(zip(structures[:9], palette)):
        ax = axes[idx]
        subset = df_comparison[
            (df_comparison['expiry'] == expiry) & (df_comparison['tenor'] == tenor)
        ].sort_values('date')

        if not subset.empty:
            err = subset['model_vol_bps'] - subset['market_vol_bps']
            ax.plot(subset['date'], err,
                    linewidth=1.5, alpha=0.8, color=color)
            ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
            ax.axhline(err.mean(), color='tomato', linewidth=1.0, linestyle=':',
                       alpha=0.8, label=f'Mean {err.mean():+.1f} bps')

        ax.set_title(f'{expiry}Y × {tenor}Y', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Error (bps)', fontsize=10)
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle('Model − Market Pricing Error Over Time',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, plot_cfg, 'model_vs_market_errors')
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare pre-computed model prices with market NVol data'
    )
    parser.add_argument('--ccy', type=str, default=DEFAULT_CCY,
                        help=f'Currency (default: {DEFAULT_CCY})')
    parser.add_argument('--n_paths', type=int, default=DEFAULT_N_PATHS,
                        help=f'MC paths used when pricing (default: {DEFAULT_N_PATHS}). '
                             'Only used to build the default model_prices path.')
    parser.add_argument('--model_prices', type=str, default=None,
                        help='Path to model_prices.csv from price_swaptions.py '
                             '(default: Figures/PricingResults/{CCY}_{n_paths}/model_prices.csv)')
    parser.add_argument('--market_data', type=str, default=None,
                        help='Path to SwapVol.xlsx (default: SwapData/SwapVol.xlsx)')
    parser.add_argument('--latent_dim', type=int, default=4,
                        help='Latent dimension used for pricing (default: 4). '
                             'Only used to label the output directory.')
    parser.add_argument('--model_tag', type=str, default='stable',
                        help="Model variant tag for output directory "
                             "(default: 'stable').")

    args = parser.parse_args()

    repo_root = _HERE.parent.parent
    ccy = args.ccy.upper()

    if ccy not in NVOL_SHEETS:
        print(f"✗ Unsupported currency '{ccy}'. Available: {list(NVOL_SHEETS.keys())}")
        return 1

    run_tag = f"{ccy}_{args.n_paths}"
    model_prices_path = Path(args.model_prices) if args.model_prices else (
        repo_root / "Figures" / "PricingResults" / run_tag / "model_prices.csv"
    )
    market_data_path = Path(args.market_data) if args.market_data else (
        repo_root / "SwapData" / "SwapVol.xlsx"
    )

    # Output directory tagged by CCY_DIM_MODEL (no timestamp), so the
    # thesis can \input{} a stable path. Existing contents are overwritten
    # in place on subsequent runs.
    out_tag = f"{ccy}_dim{args.latent_dim}_{args.model_tag}"
    run_dir = repo_root / "Figures" / "MarketComparison" / out_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"MODEL VS MARKET COMPARISON — {ccy}")
    print("=" * 70)
    print(f"  Model prices : {model_prices_path}")
    print(f"  Market data  : {market_data_path}")
    print(f"  Output dir   : {run_dir}")

    # ------------------------------------------------------------------ #
    # Load model prices
    # ------------------------------------------------------------------ #
    if not model_prices_path.exists():
        print(f"\n✗ Model prices file not found: {model_prices_path}")
        print("  Run price_swaptions.py first to generate it.")
        return 1

    df_model = pd.read_csv(model_prices_path, parse_dates=['date'])
    df_model['date'] = pd.to_datetime(df_model['date']).dt.normalize()
    # Keep only successful rows and rename column to match market data
    df_model = df_model[df_model['success'] == True][
        ['date', 'expiry', 'tenor', 'implied_vol_bps']
    ].rename(columns={'implied_vol_bps': 'model_vol_bps'})

    print(f"\n✓ Loaded {len(df_model)} model price rows "
          f"({df_model['date'].nunique()} dates, "
          f"{df_model.groupby(['expiry','tenor']).ngroups} structures)")

    # ------------------------------------------------------------------ #
    # Load market NVol data
    # ------------------------------------------------------------------ #
    df_market = load_market_vols(market_data_path, sheet_name=NVOL_SHEETS[ccy])

    # ------------------------------------------------------------------ #
    # Merge
    # ------------------------------------------------------------------ #
    df_comparison = pd.merge(
        df_model, df_market,
        on=['date', 'expiry', 'tenor'],
        how='inner'
    )

    if df_comparison.empty:
        print("\n✗ No overlapping dates between model prices and market data.")
        print(f"  Model dates : {df_model['date'].min().date()} – {df_model['date'].max().date()}")
        print(f"  Market dates: {df_market['date'].min().date()} – {df_market['date'].max().date()}")
        return 1

    print(f"✓ Merged: {len(df_comparison)} rows "
          f"across {df_comparison['date'].nunique()} dates")

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    metrics = compute_comparison_metrics(df_comparison)

    print("\n" + "=" * 70)
    print("ERROR METRICS")
    print("=" * 70)
    print(f"  Observations         : {metrics['n_observations']}")
    print(f"  Mean Absolute Error  : {metrics['mean_absolute_error']:.2f} bps")
    print(f"  Median Absolute Error: {metrics['median_absolute_error']:.2f} bps")
    print(f"  RMSE                 : {metrics['rmse']:.2f} bps")
    print(f"  Mean Relative Error  : {metrics['mean_relative_error_pct']:.1f}%")
    print(f"  Correlation          : {metrics['correlation']:.4f}")
    print(f"  R²                   : {metrics['r_squared']:.4f}")

    # Per-structure metrics
    df_cells = per_cell_summary(df_comparison)
    print("\nPer-structure summary (averages):")
    print(df_cells.to_string(index=False,
        formatters={
            'model_bps':   '{:8.2f}'.format,
            'market_bps':  '{:8.2f}'.format,
            'abs_err_bps': '{:8.2f}'.format,
            'rel_err_pct': '{:+8.1f}'.format,
        }))

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    df_comparison.to_csv(run_dir / "comparison_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
    df_cells.to_csv(run_dir / "per_cell_summary.csv", index=False)

    tex_path = run_dir / "market_comparison_table.tex"
    write_latex_table(
        df_cells=df_cells,
        metrics=metrics,
        tex_path=tex_path,
        ccy=ccy,
        n_paths=args.n_paths,
        n_dates=df_comparison['date'].nunique(),
    )

    print(f"\n✓ Saved results  → {run_dir / 'comparison_results.csv'}")
    print(f"✓ Saved metrics  → {run_dir / 'metrics.csv'}")
    print(f"✓ Saved LaTeX    → {tex_path}")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("\nGenerating plots...")
    plot_model_vs_market(df_comparison, run_dir)
    print(f"✓ Plots saved   → {run_dir}")

    print("\n" + "=" * 70)
    print("✓ COMPARISON COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

