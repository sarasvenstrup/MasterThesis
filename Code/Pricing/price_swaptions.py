"""
Swaption pricing using trained autoencoder model.

This script:
1. Loads a trained model checkpoint
2. Simulates forward interest rate curves
3. Prices ATM swaptions via Monte Carlo
4. Computes model-implied normal volatilities
5. Compares with market data (if available)
6. Saves results to CSV and generates plots

Usage:
    python price_swaptions.py
    python price_swaptions.py --checkpoint path/to/checkpoint.pt
    python price_swaptions.py --n_paths 5000 --ccy EUR
"""

import sys
from pathlib import Path
import argparse
from typing import Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directories to path
_HERE = Path(__file__).resolve().parent
_CODE_ROOT = _HERE.parent
_UTILS_DIR = _CODE_ROOT / "utils"

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from helpers import PlotConfig, save_figure
from Simulation.simulate_model import run_simulation
from load_swapdata import my_data as load_dataset
import pricing

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION  — edit these to change behaviour on right-click Run
# =============================================================================

DEFAULT_CCY      = 'EUR'   # 'EUR' or 'USD'
DEFAULT_N_PATHS  = 10_000

STANDARD_STRUCTURES = [
    (1, 1), (1, 5), (1, 10),
    (5, 1), (5, 5), (5, 10),
    (10, 1), (10, 5), (10, 10),
]

# NVol sheet name per currency in SwapVol.xlsx
NVOL_SHEETS: dict = {
    'EUR': 'EUR SwapNVol OIS',
    'USD': 'USD SwapNVol OIS',
}


# =============================================================================
# PRICING FUNCTIONS
# =============================================================================

def price_swaption_structure(
    ctx: dict,
    expiry: float,
    tenor: int,
    accrual: float = 1.0,
    verbose: bool = False
) -> dict:
    """
    Price a single ATM straddle (payer + receiver, averaged) and report
    the straddle implied normal volatility and the forward centring bias
    diagnostic.

    Returns
    -------
    dict
        Pricing results: straddle price, payer and receiver legs, forward
        centring bias, implied vol, and pathwise averages. On failure,
        all numeric fields are NaN and 'success' is False.
    """
    try:
        result = pricing.atm_swaption_straddle_mc_price_from_simulation(
            ctx=ctx,
            expiry=expiry,
            tenor=tenor,
            accrual=accrual,
            notional=1.0,
            verbose=verbose,
        )

        quote = result['quote']
        payer_res = result['payer']
        valid_mask = result['valid_mask']

        iv = result['implied_normal_vol']

        return {
            'expiry': expiry,
            'tenor': tenor,
            # Time-0 quantities (from initial curve)
            'forward_swap_0': quote['forward_swap'],
            'annuity_0': quote['annuity'],
            'strike': quote['strike'],
            # Straddle MC pricing
            'payer_price': result['payer_price'],
            'receiver_price': result['receiver_price'],
            'mc_price': result['straddle_price'],          # straddle price
            'mc_stderr': result['straddle_stderr'],
            'mc_std': float('nan'),                        # not meaningful for straddle
            # Forward centring bias diagnostic
            'forward_bias': result['forward_bias'],
            # Implied volatility (from straddle)
            'implied_vol': iv,
            'implied_vol_bps': iv * 10000 if iv is not None and np.isfinite(iv) else np.nan,
            # Path statistics (intersection of payer & receiver validity)
            'valid_paths': int(valid_mask.sum()),
            'total_paths': int(len(valid_mask)),
            # At-expiry pathwise averages (taken from the payer leg — same paths)
            'mean_annuity_expiry': np.mean(payer_res['annuity_paths'][valid_mask])
                if valid_mask.sum() > 0 else np.nan,
            'mean_swap_rate_expiry': np.mean(payer_res['swap_rate_paths'][valid_mask])
                if valid_mask.sum() > 0 else np.nan,
            'mean_discount_to_expiry': np.mean(payer_res['discount_to_expiry_paths'][valid_mask])
                if valid_mask.sum() > 0 else np.nan,
            'success': iv is not None and np.isfinite(iv),
        }
    except Exception as e:
        print(f"  ✗ Error pricing {expiry}Y x {tenor}Y: {e}")
        return {
            'expiry': expiry,
            'tenor': tenor,
            'forward_swap_0': np.nan,
            'annuity_0': np.nan,
            'strike': np.nan,
            'payer_price': np.nan,
            'receiver_price': np.nan,
            'mc_price': np.nan,
            'mc_stderr': np.nan,
            'mc_std': np.nan,
            'forward_bias': np.nan,
            'implied_vol': np.nan,
            'implied_vol_bps': np.nan,
            'valid_paths': 0,
            'total_paths': 0,
            'mean_annuity_expiry': np.nan,
            'mean_swap_rate_expiry': np.nan,
            'mean_discount_to_expiry': np.nan,
            'success': False,
            'error': str(e),
        }


def price_all_structures(
    ctx: dict,
    structures: List[Tuple[int, int]],
    accrual: float = 1.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Price multiple swaption structures using the SAME simulated paths.

    Parameters
    ----------
    ctx : dict
        Simulation context from run_simulation()
    structures : list of (expiry, tenor) tuples
        Swaption structures to price
    accrual : float
        Payment frequency
    verbose : bool
        Print detailed output for each structure
    
    Returns
    -------
    pd.DataFrame
        Pricing results for all structures with diagnostic data:
        - Time-0 quantities: forward_swap_0, annuity_0, strike
        - MC pricing: mc_price, mc_stderr, mc_std
        - Implied vol: implied_vol, implied_vol_bps
        - Expiry quantities: mean_annuity_expiry, mean_swap_rate_expiry, mean_discount_to_expiry
        - Path stats: valid_paths, total_paths
    """
    results = []
    
    print(f"\nPricing {len(structures)} swaption structures using the SAME paths...")
    print("=" * 70)
    
    for i, (expiry, tenor) in enumerate(structures, 1):
        print(f"\n[{i}/{len(structures)}] Pricing {expiry}Y x {tenor}Y swaption...")
        
        result = price_swaption_structure(
            ctx=ctx,
            expiry=expiry,
            tenor=tenor,
            accrual=accrual,
            verbose=verbose,
        )
        
        if result['success']:
            print(f"  ✓ MC Price: {result['mc_price']:.6f} ± {result['mc_stderr']:.6f}")
            print(f"  ✓ Implied Vol: {result['implied_vol_bps']:.2f} bps")
            print(f"  ✓ Forward Rate (t=0): {result['forward_swap_0']*10000:.2f} bps")
            print(f"  ✓ Annuity (t=0): {result['annuity_0']:.6f}")
            print(f"  ✓ Valid Paths: {result['valid_paths']}/{result['total_paths']}")
        else:
            print(f"  ✗ Pricing failed")
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Add structure label
    df['structure'] = df['expiry'].astype(str) + 'x' + df['tenor'].astype(str)
    
    return df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_implied_vol_surface(
    df: pd.DataFrame,
    plot_cfg: PlotConfig,
    title: str = "Model-Implied Swaption Volatility Surface"
):
    """Plot volatility surface as heatmap."""
    # Filter successful pricing
    df_valid = df[df['success']].copy()
    
    if df_valid.empty:
        print("⚠ No valid pricing results to plot")
        return
    
    # Pivot to create surface
    pivot = df_valid.pivot_table(
        values='implied_vol_bps',
        index='expiry',
        columns='tenor',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Implied Vol (bps)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Swap Tenor (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Option Expiry (years)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, plot_cfg, "implied_vol_surface")
    plt.close()


def plot_vol_by_expiry(
    df: pd.DataFrame,
    plot_cfg: PlotConfig,
    market_data: Optional[pd.DataFrame] = None
):
    """Plot implied volatility by expiry for different tenors."""
    df_valid = df[df['success']].copy()
    
    if df_valid.empty:
        print("⚠ No valid pricing results to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot model vols
    for tenor in sorted(df_valid['tenor'].unique()):
        subset = df_valid[df_valid['tenor'] == tenor].sort_values('expiry')
        ax.plot(
            subset['expiry'],
            subset['implied_vol_bps'],
            marker='o',
            markersize=8,
            label=f"{tenor}Y Swap (Model)",
            linewidth=2,
            alpha=0.8
        )
    
    # Overlay market data if available
    if market_data is not None:
        for tenor in sorted(market_data['tenor'].unique()):
            subset = market_data[market_data['tenor'] == tenor].sort_values('expiry')
            ax.plot(
                subset['expiry'],
                subset['vol_bps'],
                marker='s',
                markersize=6,
                linestyle='--',
                label=f"{tenor}Y Swap (Market)",
                linewidth=1.5,
                alpha=0.6
            )
    
    ax.set_xlabel('Option Expiry (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Implied Normal Vol (bps)', fontsize=12, fontweight='bold')
    ax.set_title('Model-Implied vs Market Swaption Volatilities', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, plot_cfg, "implied_vol_by_expiry")
    plt.close()


def plot_forward_rates(
    df: pd.DataFrame,
    plot_cfg: PlotConfig
):
    """Plot forward swap rates by expiry and tenor."""
    df_valid = df[df['success']].copy()
    
    if df_valid.empty:
        print("⚠ No valid pricing results to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for tenor in sorted(df_valid['tenor'].unique()):
        subset = df_valid[df_valid['tenor'] == tenor].sort_values('expiry')
        ax.plot(
            subset['expiry'],
            subset['forward_swap_0'] * 10000,
            marker='o',
            markersize=8,
            label=f"{tenor}Y Swap",
            linewidth=2,
            alpha=0.8
        )
    
    ax.set_xlabel('Option Expiry (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Forward Swap Rate (bps)', fontsize=12, fontweight='bold')
    ax.set_title('Forward Swap Rates from Model', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, plot_cfg, "forward_swap_rates")
    plt.close()


def plot_mc_convergence(
    df: pd.DataFrame,
    plot_cfg: PlotConfig
):
    """Plot MC standard errors as percentage of price."""
    df_valid = df[df['success']].copy()
    
    if df_valid.empty:
        print("⚠ No valid pricing results to plot")
        return
    
    df_valid['stderr_pct'] = (df_valid['mc_stderr'] / df_valid['mc_price']) * 100
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_labels = df_valid['structure']
    y_values = df_valid['stderr_pct']
    
    bars = ax.bar(range(len(x_labels)), y_values, alpha=0.7, color='steelblue')
    
    # Color code by quality
    for i, val in enumerate(y_values):
        if val < 1.0:
            bars[i].set_color('green')
        elif val < 5.0:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('red')
    
    ax.set_xlabel('Swaption Structure', fontsize=12, fontweight='bold')
    ax.set_ylabel('MC Std Error (% of price)', fontsize=12, fontweight='bold')
    ax.set_title('Monte Carlo Convergence Quality', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.axhline(y=1.0, color='green', linestyle='--', label='< 1% (Good)', alpha=0.5)
    ax.axhline(y=5.0, color='red', linestyle='--', label='< 5% (Acceptable)', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, plot_cfg, "mc_convergence")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def _get_nvol_dates(excel_path: Path, sheet_name: str) -> List[pd.Timestamp]:
    """Return sorted unique dates from the NVol sheet."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    dates = []
    for row_idx in range(4, len(df)):
        val = df.iloc[row_idx, 1]   # 'M' column
        try:
            if pd.notna(val):
                dates.append(pd.to_datetime(val).normalize())
        except Exception:
            pass
    return sorted(set(dates))


def main():
    parser = argparse.ArgumentParser(
        description='Price swaptions for all market dates and save results to CSV'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--market_data', type=str, default=None,
                        help='Path to SwapVol.xlsx (default: SwapData/SwapVol.xlsx)')
    parser.add_argument('--ccy', type=str, default=DEFAULT_CCY,
                        help=f'Currency (default: {DEFAULT_CCY})')
    parser.add_argument('--n_paths', type=int, default=DEFAULT_N_PATHS,
                        help=f'MC paths per date (default: {DEFAULT_N_PATHS})')
    parser.add_argument('--n_dates', type=int, default=None,
                        help='Cap number of dates (default: all overlapping dates)')
    parser.add_argument('--latent_dim', type=int, default=4,
                        help='Latent dimension (default: 4)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: Figures/PricingResults/model_prices.csv)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh — ignore any existing results file')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose pricing output')
    
    args = parser.parse_args()

    repo_root = _HERE.parent.parent

    if args.checkpoint is None:
        args.checkpoint = str(
            repo_root / "Figures" / "TrainingResults" / "dim4_stable" /
            "ep5000" / "checkpoint_dim4_ep5000.pt"
        )
    if args.market_data is None:
        args.market_data = str(repo_root / "SwapData" / "SwapVol.xlsx")

    ccy = args.ccy.upper()
    if ccy not in NVOL_SHEETS:
        print(f"✗ Unsupported currency '{ccy}'. Available: {list(NVOL_SHEETS.keys())}")
        return 1

    run_tag = f"{ccy}_{args.n_paths}"
    csv_path = Path(args.output) if args.output else (
        repo_root / "Figures" / "PricingResults" / run_tag / "model_prices.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build date pairs: (vol_date, pricing_date)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print(f"SWAPTION PRICING — {ccy}  ({args.n_paths:,} paths)")
    print("=" * 70)

    # Dataset dates (swap-rate data — defines what can be priced)
    dataset = load_dataset(ccy_filter=ccy)
    df_ds = dataset[0]
    ds_dates = sorted(pd.to_datetime(df_ds['as_of_date']).dt.normalize().unique())
    ds_min, ds_max = ds_dates[0], ds_dates[-1]

    def nearest_ds_date(target):
        idx = np.searchsorted(ds_dates, target)
        candidates = []
        if idx < len(ds_dates): candidates.append(ds_dates[idx])
        if idx > 0:             candidates.append(ds_dates[idx - 1])
        return min(candidates, key=lambda d: abs(d - target))

    # NVol dates (market vol dates — defines what we compare against)
    nvol_dates = _get_nvol_dates(Path(args.market_data), sheet_name=NVOL_SHEETS[ccy])
    vol_dates_in_range = sorted(
        [d for d in nvol_dates if ds_min <= d <= ds_max], reverse=True
    )

    if not vol_dates_in_range:
        print(f"✗ No NVol dates in dataset range {ds_min.date()} – {ds_max.date()}")
        return 1

    date_pairs = [(d, nearest_ds_date(d)) for d in vol_dates_in_range]
    if args.n_dates is not None:
        date_pairs = date_pairs[:args.n_dates]

    print(f"\nDataset range : {ds_min.date()} – {ds_max.date()}")
    print(f"NVol dates in range : {len(vol_dates_in_range)}")
    print(f"Dates to price      : {len(date_pairs)}")
    print(f"MC paths per date   : {args.n_paths:,}")
    print(f"Output CSV          : {csv_path}")

    # ------------------------------------------------------------------ #
    # Resume: skip dates already in the output CSV
    # ------------------------------------------------------------------ #
    done_dates: set = set()
    if csv_path.exists() and not args.fresh:
        df_existing = pd.read_csv(csv_path, parse_dates=['date'])
        done_dates = set(df_existing['date'].dt.normalize().unique())
        print(f"\nResuming — {len(done_dates)} date(s) already complete, "
              f"{sum(1 for v,_ in date_pairs if v not in done_dates)} remaining")
    elif args.fresh and csv_path.exists():
        csv_path.unlink()
        print("\n--fresh: removed existing results file")

    # Decode steps: only the expiry time-steps actually needed
    _dt = 1 / 12
    _n_steps = 120
    unique_expiries = sorted({e for e, t in STANDARD_STRUCTURES})
    decode_steps = [0] + [round(e / _dt) for e in unique_expiries]

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    n_done = 0
    n_failed = 0

    for i, (vol_date, pricing_date) in enumerate(date_pairs, 1):
        if vol_date in done_dates:
            print(f"[{i}/{len(date_pairs)}] {vol_date.date()} — already done, skipping")
            continue

        tag = "" if vol_date == pricing_date else f" (curve {pricing_date.date()})"
        print(f"\n[{i}/{len(date_pairs)}] Pricing {vol_date.date()}{tag} ...")

        # Run simulation
        try:
            ctx = run_simulation(
                checkpoint_path=args.checkpoint,
                ccy_filter=ccy,
                latent_dim=args.latent_dim,
                n_paths=args.n_paths,
                n_steps=_n_steps,
                dt=_dt,
                diffusion_scale=1.0,
                use_antithetic=True,
                as_of_date=pricing_date,
                show_plot=False,
                decode_steps=decode_steps,
            )
        except Exception as e:
            print(f"  ✗ Simulation failed: {e}")
            n_failed += 1
            continue

        # Price all structures
        date_df = price_all_structures(
            ctx=ctx,
            structures=STANDARD_STRUCTURES,
            accrual=1.0,
            verbose=args.verbose,
        )

        # Attach dates
        date_df['date']         = vol_date
        date_df['pricing_date'] = pricing_date

        # Append to CSV (write header only on first write)
        write_header = not csv_path.exists()
        date_df.to_csv(csv_path, mode='a', header=write_header, index=False)

        n_done += 1
        n_success = date_df['success'].sum()
        print(f"  ✓ Saved {n_success}/{len(date_df)} structures → {csv_path.name}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PRICING COMPLETE")
    print("=" * 70)
    print(f"  Dates priced this run : {n_done}")
    print(f"  Dates failed          : {n_failed}")
    print(f"  Results CSV           : {csv_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

