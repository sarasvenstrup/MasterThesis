"""
Visualization tools for swaption volatility market data.

This module provides various plots to analyze and understand the structure
of the swaption volatility surface over time, including:
    - Time series of volatilities for different swaption structures
    - Volatility surface heatmaps and 3D surfaces
    - Term structure and expiry structure analysis
    - Distribution analysis
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add utils to path for importing helpers
_HERE = Path(__file__).resolve().parent
_UTILS_DIR = _HERE.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from helpers import PlotConfig, save_figure
from load_swapvol_ois import load_swaption_vol_data


# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_vol_timeseries(
    df,
    selected_structures=None,
    title="Swaption Volatility Time Series",
    figsize=(14, 8),
    plot_cfg=None,
    save_name=None
):
    """
    Plot volatility time series for selected swaption structures.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output from load_swaption_vol_data with columns:
        [currency, as_of_date, option_maturity, swap_tenor, vol]
    selected_structures : list of tuples or None
        List of (option_maturity, swap_tenor) pairs to plot.
        If None, selects a few representative structures.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    plot_cfg : PlotConfig or None
        PlotConfig instance for consistent saving. If None, displays plot.
    save_name : str or None
        Name for saved figure (without extension). Only used if plot_cfg is provided.

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    """
    if selected_structures is None:
        # Select common ATM structures
        selected_structures = [
            (1, 1), (1, 5), (1, 10),
            (5, 5), (5, 10),
            (10, 10)
        ]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for opt_mat, swap_ten in selected_structures:
        mask = (df['option_maturity'] == opt_mat) & (df['swap_tenor'] == swap_ten)
        subset = df[mask].sort_values('as_of_date')
        
        if len(subset) > 0:
            label = f"{opt_mat}Y x {swap_ten}Y"
            ax.plot(subset['as_of_date'], subset['vol'], 
                   marker='o', markersize=3, label=label, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volatility (bps)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if plot_cfg and save_name:
        save_figure(fig, plot_cfg, save_name)
    elif plot_cfg:
        save_figure(fig, plot_cfg, "vol_timeseries")
    else:
        plt.show()
    
    return fig, ax


def plot_vol_surface_heatmap(
    df,
    date=None,
    title_prefix="Swaption Volatility Surface",
    figsize=(10, 8),
    plot_cfg=None,
    save_name=None,
    cmap='RdYlGn_r'
):
    """
    Plot volatility surface as a heatmap for a specific date.
    
    Parameters
    ----------
    df : pd.DataFrame
        Swaption vol data
    date : str, datetime, or None
        Date for which to plot the surface. If None, uses the most recent date.
    title_prefix : str
        Prefix for plot title
    figsize : tuple
        Figure size
    plot_cfg : PlotConfig or None
        PlotConfig instance for consistent saving. If None, displays plot.
    save_name : str or None
        Name for saved figure (without extension). Only used if plot_cfg is provided.
    cmap : str
        Colormap name

    Returns
    -------
    fig : matplotlib.Figure or None
    ax : matplotlib.Axes or None
    """
    if date is None:
        date = df['as_of_date'].max()
    else:
        date = pd.to_datetime(date)

    subset = df[df['as_of_date'] == date].copy()

    if subset.empty:
        print(f"No data available for date {date}")
        return None, None

    # Pivot to create surface matrix
    pivot = subset.pivot_table(
        values='vol',
        index='option_maturity',
        columns='swap_tenor',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        cbar_kws={'label': 'Volatility (bps)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Swap Tenor (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Option Maturity (years)', fontsize=12, fontweight='bold')
    ax.set_title(f"{title_prefix}\n{date.strftime('%Y-%m-%d')}", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if plot_cfg and save_name:
        save_figure(fig, plot_cfg, save_name)
    elif plot_cfg:
        save_figure(fig, plot_cfg, "vol_surface_heatmap")
    else:
        plt.show()
    
    return fig, ax


def plot_vol_surface_3d(
    df,
    date=None,
    title_prefix="Swaption Volatility Surface (3D)",
    figsize=(12, 9),
    plot_cfg=None,
    save_name=None,
    cmap='viridis'
):
    """
    Plot volatility surface as a 3D surface for a specific date.
    
    Parameters
    ----------
    df : pd.DataFrame
        Swaption vol data
    date : str, datetime, or None
        Date for which to plot the surface. If None, uses the most recent date.
    title_prefix : str
        Prefix for plot title
    figsize : tuple
        Figure size
    plot_cfg : PlotConfig or None
        PlotConfig instance for consistent saving. If None, displays plot.
    save_name : str or None
        Name for saved figure (without extension). Only used if plot_cfg is provided.
    cmap : str
        Colormap name

    Returns
    -------
    fig : matplotlib.Figure or None
    ax : matplotlib.Axes or None
    """
    if date is None:
        date = df['as_of_date'].max()
    else:
        date = pd.to_datetime(date)

    subset = df[df['as_of_date'] == date].copy()

    if subset.empty:
        print(f"No data available for date {date}")
        return None, None

    # Pivot to create surface matrix
    pivot = subset.pivot_table(
        values='vol',
        index='option_maturity',
        columns='swap_tenor',
        aggfunc='mean'
    )

    # Create meshgrid for 3D plotting
    X = pivot.columns.values  # swap_tenor
    Y = pivot.index.values    # option_maturity
    X, Y = np.meshgrid(X, Y)
    Z = pivot.values
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.2,
        antialiased=True
    )
    
    ax.set_xlabel('Swap Tenor (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Option Maturity (years)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Volatility (bps)', fontsize=11, fontweight='bold')
    ax.set_title(f"{title_prefix}\n{date.strftime('%Y-%m-%d')}", 
                fontsize=13, fontweight='bold')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Volatility (bps)')
    
    # Adjust viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    if plot_cfg and save_name:
        save_figure(fig, plot_cfg, save_name)
    elif plot_cfg:
        save_figure(fig, plot_cfg, "vol_surface_3d")
    else:
        plt.show()
    
    return fig, ax


def plot_term_structure(
    df,
    option_maturities=None,
    date=None,
    title="Volatility Term Structure",
    figsize=(12, 7),
    plot_cfg=None,
    save_name=None
):
    """
    Plot volatility term structure: vol vs swap tenor for fixed option maturities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Swaption vol data
    option_maturities : list or None
        List of option maturities to plot. If None, plots common values.
    date : str, datetime, or None
        Date for which to plot. If None, uses the most recent date.
    title : str
        Plot title
    figsize : tuple
        Figure size
    plot_cfg : PlotConfig or None
        PlotConfig instance for consistent saving. If None, displays plot.
    save_name : str or None
        Name for saved figure (without extension). Only used if plot_cfg is provided.

    Returns
    -------
    fig : matplotlib.Figure or None
    ax : matplotlib.Axes or None
    """
    if date is None:
        date = df['as_of_date'].max()
    else:
        date = pd.to_datetime(date)

    subset = df[df['as_of_date'] == date].copy()

    if subset.empty:
        print(f"No data available for date {date}")
        return None, None

    if option_maturities is None:
        option_maturities = sorted(subset['option_maturity'].unique())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for opt_mat in option_maturities:
        data = subset[subset['option_maturity'] == opt_mat].sort_values('swap_tenor')
        if len(data) > 0:
            ax.plot(data['swap_tenor'], data['vol'], 
                   marker='o', markersize=8, label=f"{opt_mat}Y option",
                   linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Swap Tenor (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volatility (bps)', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\n{date.strftime('%Y-%m-%d')}", 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if plot_cfg and save_name:
        save_figure(fig, plot_cfg, save_name)
    elif plot_cfg:
        save_figure(fig, plot_cfg, "vol_term_structure")
    else:
        plt.show()
    
    return fig, ax


def plot_expiry_structure(
    df,
    swap_tenors=None,
    date=None,
    title="Volatility Expiry Structure",
    figsize=(12, 7),
    plot_cfg=None,
    save_name=None
):
    """
    Plot volatility expiry structure: vol vs option maturity for fixed swap tenors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Swaption vol data
    swap_tenors : list or None
        List of swap tenors to plot. If None, plots common values.
    date : str, datetime, or None
        Date for which to plot. If None, uses the most recent date.
    title : str
        Plot title
    figsize : tuple
        Figure size
    plot_cfg : PlotConfig or None
        PlotConfig instance for consistent saving. If None, displays plot.
    save_name : str or None
        Name for saved figure (without extension). Only used if plot_cfg is provided.

    Returns
    -------
    fig : matplotlib.Figure or None
    ax : matplotlib.Axes or None
    """
    if date is None:
        date = df['as_of_date'].max()
    else:
        date = pd.to_datetime(date)

    subset = df[df['as_of_date'] == date].copy()

    if subset.empty:
        print(f"No data available for date {date}")
        return None, None

    if swap_tenors is None:
        swap_tenors = sorted(subset['swap_tenor'].unique())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for swap_ten in swap_tenors:
        data = subset[subset['swap_tenor'] == swap_ten].sort_values('option_maturity')
        if len(data) > 0:
            ax.plot(data['option_maturity'], data['vol'], 
                   marker='s', markersize=8, label=f"{swap_ten}Y swap",
                   linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Option Maturity (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volatility (bps)', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\n{date.strftime('%Y-%m-%d')}", 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if plot_cfg and save_name:
        save_figure(fig, plot_cfg, save_name)
    elif plot_cfg:
        save_figure(fig, plot_cfg, "vol_expiry_structure")
    else:
        plt.show()
    
    return fig, ax


def plot_vol_distributions(
    df,
    by_structure=True,
    title="Volatility Distributions",
    figsize=(14, 8),
    plot_cfg=None,
    save_name=None
):
    """
    Plot distributions of volatility values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Swaption vol data
    by_structure : bool
        If True, plots separate distributions for each (option_maturity, swap_tenor).
        If False, plots overall distribution and by option_maturity and swap_tenor separately.
    title : str
        Plot title
    figsize : tuple
        Figure size
    plot_cfg : PlotConfig or None
        PlotConfig instance for consistent saving. If None, displays plot.
    save_name : str or None
        Name for saved figure (without extension). Only used if plot_cfg is provided.

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes or np.ndarray of Axes
    """
    if by_structure:
        # Box plots for each structure
        df_copy = df.copy()
        df_copy['structure'] = (
            df_copy['option_maturity'].astype(str) + 'x' + 
            df_copy['swap_tenor'].astype(str)
        )
        
        # Sort by structure for better visualization
        structure_order = df_copy.groupby('structure')['vol'].median().sort_values().index
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.boxplot(
            data=df_copy,
            x='structure',
            y='vol',
            order=structure_order,
            ax=ax,
            hue='structure',
            palette='Set2',
            legend=False
        )
        
        ax.set_xlabel('Swaption Structure (Option x Swap)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (bps)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
    else:
        # Multiple distribution views
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Overall distribution
        axes[0, 0].hist(df['vol'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Volatility (bps)', fontsize=10)
        axes[0, 0].set_ylabel('Frequency', fontsize=10)
        axes[0, 0].set_title('Overall Distribution', fontsize=11, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # By option maturity
        for opt_mat in sorted(df['option_maturity'].unique()):
            subset = df[df['option_maturity'] == opt_mat]
            axes[0, 1].hist(subset['vol'], bins=30, alpha=0.5, 
                          label=f"{opt_mat}Y", edgecolor='black')
        axes[0, 1].set_xlabel('Volatility (bps)', fontsize=10)
        axes[0, 1].set_ylabel('Frequency', fontsize=10)
        axes[0, 1].set_title('By Option Maturity', fontsize=11, fontweight='bold')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # By swap tenor
        for swap_ten in sorted(df['swap_tenor'].unique()):
            subset = df[df['swap_tenor'] == swap_ten]
            axes[1, 0].hist(subset['vol'], bins=30, alpha=0.5, 
                          label=f"{swap_ten}Y", edgecolor='black')
        axes[1, 0].set_xlabel('Volatility (bps)', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        axes[1, 0].set_title('By Swap Tenor', fontsize=11, fontweight='bold')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""
        Summary Statistics:
        Mean: {df['vol'].mean():.2f} bps
        Median: {df['vol'].median():.2f} bps
        Std Dev: {df['vol'].std():.2f} bps
        Min: {df['vol'].min():.2f} bps
        Max: {df['vol'].max():.2f} bps
        
        Date Range:
        {df['as_of_date'].min().strftime('%Y-%m-%d')}
        to
        {df['as_of_date'].max().strftime('%Y-%m-%d')}
        
        Total Observations: {len(df):,}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if plot_cfg and save_name:
        save_figure(fig, plot_cfg, save_name)
    elif plot_cfg:
        save_figure(fig, plot_cfg, "vol_distributions")
    else:
        plt.show()
    
    return fig, axes if not by_structure else (fig, ax)


def plot_vol_evolution_multiple_dates(
    df,
    dates=None,
    n_dates=4,
    title_prefix="Volatility Surface Evolution",
    figsize=(16, 10),
    plot_cfg=None,
    save_name=None,
    cmap='RdYlGn_r'
):
    """
    Plot volatility surfaces at multiple dates to show evolution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Swaption vol data
    dates : list or None
        Specific dates to plot. If None, selects n_dates evenly spaced.
    n_dates : int
        Number of dates to plot if dates is None.
    title_prefix : str
        Prefix for plot title
    figsize : tuple
        Figure size
    plot_cfg : PlotConfig or None
        PlotConfig instance for consistent saving. If None, displays plot.
    save_name : str or None
        Name for saved figure (without extension). Only used if plot_cfg is provided.
    cmap : str
        Colormap name

    Returns
    -------
    fig : matplotlib.Figure
    axes : np.ndarray of Axes
    """
    if dates is None:
        # Select evenly spaced dates
        all_dates = sorted(df['as_of_date'].unique())
        if len(all_dates) < n_dates:
            selected_dates = all_dates
        else:
            indices = np.linspace(0, len(all_dates) - 1, n_dates, dtype=int)
            selected_dates = [all_dates[i] for i in indices]
    else:
        selected_dates = [pd.to_datetime(d) for d in dates]
    
    n_plots = len(selected_dates)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Find global min/max for consistent color scale
    vmin = df['vol'].min()
    vmax = df['vol'].max()
    
    for idx, date in enumerate(selected_dates):
        ax = axes[idx]
        subset = df[df['as_of_date'] == date].copy()
        
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for\n{date.strftime('%Y-%m-%d')}", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        pivot = subset.pivot_table(
            values='vol',
            index='option_maturity',
            columns='swap_tenor',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Vol (bps)'}
        )
        
        ax.set_xlabel('Swap Tenor (Y)', fontsize=10)
        ax.set_ylabel('Option Mat. (Y)', fontsize=10)
        ax.set_title(date.strftime('%Y-%m-%d'), fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(selected_dates), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if plot_cfg and save_name:
        save_figure(fig, plot_cfg, save_name)
    elif plot_cfg:
        save_figure(fig, plot_cfg, "vol_evolution")
    else:
        plt.show()
    
    return fig, axes


def generate_full_report(
    df,
    output_dir=None,
    currency='EUR',
    date=None,
    use_tag=""
):
    """
    Generate a full visualization report with all standard plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Swaption vol data
    output_dir : str or None
        Directory to save figures. If None, uses ../../Figures/SwapVolAnalysis
    currency : str
        Currency code for file naming
    date : str, datetime, or None
        Reference date for surface plots. If None, uses most recent.
    use_tag : str
        Optional tag to append to filenames (e.g., "updated", "prelim")
    """
    if output_dir is None:
        here = Path(__file__).resolve().parent
        output_dir = here.parent.parent / 'Figures' / 'SwapVolAnalysis'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating full visualization report in: {output_dir}\n")
    
    currency = currency.upper()
    
    # Create PlotConfig for consistent saving
    plot_cfg = PlotConfig(
        figures_dir=str(output_dir),
        use_tag=use_tag,
        dpi=300
    )
    
    # 1. Time series
    print("1. Generating time series plot...")
    plot_vol_timeseries(
        df,
        title=f"{currency} Swaption Volatility Time Series",
        plot_cfg=plot_cfg,
        save_name=f'{currency}_vol_timeseries'
    )
    plt.close()
    
    # 2. Volatility surface heatmap
    print("2. Generating volatility surface heatmap...")
    plot_vol_surface_heatmap(
        df,
        date=date,
        title_prefix=f"{currency} Swaption Volatility Surface",
        plot_cfg=plot_cfg,
        save_name=f'{currency}_vol_surface_heatmap'
    )
    plt.close()
    
    # 3. 3D surface
    print("3. Generating 3D volatility surface...")
    plot_vol_surface_3d(
        df,
        date=date,
        title_prefix=f"{currency} Swaption Volatility Surface (3D)",
        plot_cfg=plot_cfg,
        save_name=f'{currency}_vol_surface_3d'
    )
    plt.close()
    
    # 4. Term structure
    print("4. Generating term structure plot...")
    plot_term_structure(
        df,
        date=date,
        title=f"{currency} Volatility Term Structure",
        plot_cfg=plot_cfg,
        save_name=f'{currency}_vol_term_structure'
    )
    plt.close()
    
    # 5. Expiry structure
    print("5. Generating expiry structure plot...")
    plot_expiry_structure(
        df,
        date=date,
        title=f"{currency} Volatility Expiry Structure",
        plot_cfg=plot_cfg,
        save_name=f'{currency}_vol_expiry_structure'
    )
    plt.close()
    
    # 6. Distributions
    print("6. Generating distribution plots...")
    plot_vol_distributions(
        df,
        by_structure=True,
        title=f"{currency} Volatility Distributions by Structure",
        plot_cfg=plot_cfg,
        save_name=f'{currency}_vol_distributions'
    )
    plt.close()
    
    # 7. Evolution over time
    print("7. Generating surface evolution plot...")
    plot_vol_evolution_multiple_dates(
        df,
        n_dates=6,
        title_prefix=f"{currency} Volatility Surface Evolution",
        plot_cfg=plot_cfg,
        save_name=f'{currency}_vol_evolution'
    )
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_dir}\n")


if __name__ == "__main__":
    # Example usage
    print("Loading swaption volatility data...")
    df_vol = load_swaption_vol_data(currency="EUR", verbose=False)

    print(f"\nLoaded {len(df_vol):,} observations")
    print(f"Date range: {df_vol['as_of_date'].min()} to {df_vol['as_of_date'].max()}")
    print(f"Unique structures: {df_vol[['option_maturity', 'swap_tenor']].drop_duplicates().shape[0]}")

    # Generate full report
    generate_full_report(df_vol, currency='EUR')

    print("\nDone! Check the Figures/SwapVolAnalysis directory for outputs.")

