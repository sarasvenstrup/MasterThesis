"""
Consolidated results generator for the Pricing chapter.

Generates all tables and figures referenced in the final thesis chapter:
  - Table 3.2: Combined pricing results table
  - Table 3.3: Term structure trained table
  - Figure 3.2: Timeseries comparison
  - Figure 3.3: Term structure vol by tenor (base model)
  - Figure 3.4: Scaling sensitivity
  - Figure 3.5: Forward bias timeseries
  - Figure 3.6: F_T distribution per expiry

Run from repo root:
    python Code/Pricing/generate_pricing_chapter_results.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Combined Pricing Results (Tab 3.2)
# ══════════════════════════════════════════════════════════════════════════════

def generate_combined_table():
    """Generate Table 3.2: market, stable, pricing layer, MAE (all/train/test), fwd bias"""
    print("\n[1/7] Generating combined pricing table (Table 3.2)...")
    
    BASE_CSV = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "eval_base", "per_cell_final.csv")
    CMPR_CSV = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "eval_constant_mpr", "per_cell_final.csv")
    OUT_DIR  = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "comparison")
    os.makedirs(OUT_DIR, exist_ok=True)

    df_base = pd.read_csv(BASE_CSV)
    df_cmpr = pd.read_csv(CMPR_CSV)

    EXPIRIES = [1, 5, 10]
    TENORS   = [1, 5, 10]

    rows = []
    for exp in EXPIRIES:
        for ten in TENORS:
            b_all  = df_base[(df_base["expiry"]==exp) & (df_base["tenor"]==ten)]
            c_all  = df_cmpr[(df_cmpr["expiry"]==exp) & (df_cmpr["tenor"]==ten)]
            c_train = c_all[c_all["split"]=="train"]
            c_test  = c_all[c_all["split"]=="test"]
            
            if len(c_all) == 0:
                continue
            
            mkt    = c_all["mkt_bp"].mean()
            stable = b_all["sigma_str_bp"].mean() if len(b_all) else np.nan
            layer  = c_all["sigma_str_bp"].mean()
            
            stable_mae = b_all["vol_error_bp"].abs().mean() if len(b_all) else np.nan
            mae_all   = c_all["vol_error_bp"].abs().mean()
            mae_train = c_train["vol_error_bp"].abs().mean() if len(c_train) else np.nan
            mae_test  = c_test["vol_error_bp"].abs().mean() if len(c_test) else np.nan
            fwd_bias  = c_all["forward_bias_bp"].mean()
            
            rows.append({
                "exp": exp, "ten": ten, "mkt": mkt, "stable": stable, "layer": layer,
                "stable_mae": stable_mae, "mae_all": mae_all, "mae_train": mae_train, 
                "mae_test": mae_test, "fwd_bias": fwd_bias,
            })

    # Overall row
    mkt_overall    = df_cmpr["mkt_bp"].mean()
    stable_overall = df_base["sigma_str_bp"].mean()
    layer_overall  = df_cmpr["sigma_str_bp"].mean()
    stable_mae_overall = df_base["vol_error_bp"].abs().mean()
    mae_all_overall   = df_cmpr["vol_error_bp"].abs().mean()
    mae_train_overall = df_cmpr[df_cmpr["split"]=="train"]["vol_error_bp"].abs().mean()
    mae_test_overall  = df_cmpr[df_cmpr["split"]=="test"]["vol_error_bp"].abs().mean()
    fwd_bias_overall  = df_cmpr["forward_bias_bp"].mean()

    # Write LaTeX table
    out_path = os.path.join(OUT_DIR, "tab_pricing_combined.tex")
    with open(out_path, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Per-cell ATM straddle normal volatility: market, stable model, ")
        f.write("and calibrated pricing layer (all dates, EUR). MAE is reported for both stable model and pricing layer. ")
        f.write("Values in basis points.}\n")
        f.write("\\label{tab:pricing_combined}\n")
        f.write("\\small\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{@{}ccrrrrrrrr@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Exp} & \\textbf{Ten}\n")
        f.write("  & \\textbf{Market}\n  & \\textbf{Stable}\n  & \\textbf{Pricing layer}\n")
        f.write("  & \\textbf{Stable MAE}\n  & \\textbf{MAE (all)}\n  & \\textbf{MAE (train)}\n  & \\textbf{MAE (test)}\n")
        f.write("  & \\textbf{Fwd bias} \\\\\n")
        f.write("\\midrule\n")
        
        for i, r in enumerate(rows):
            if i > 0 and rows[i-1]["exp"] != r["exp"]:
                f.write("\\midrule\n")
            
            bias_str = f"{r['fwd_bias']:+.0f}"
            if r["exp"] == 10:
                bias_str += "\\(^\\dagger\\)"
            
            f.write(f"  {r['exp']}Y & {r['ten']}Y")
            f.write(f"  & {r['mkt']:.0f}  & {r['stable']:.0f}  & {r['layer']:.0f}")
            f.write(f"  & {r['stable_mae']:.1f}  & {r['mae_all']:.1f}  & {r['mae_train']:.1f}  & {r['mae_test']:.1f}")
            f.write(f"  & {bias_str} \\\\\n")
        
        f.write("\\midrule\n")
        f.write("  \\multicolumn{2}{c}{\\textbf{All cells}}")
        f.write(f" & {mkt_overall:.0f}  & {stable_overall:.0f}  & {layer_overall:.0f}")
        f.write(f" & \\textbf{{{stable_mae_overall:.1f}}}  & \\textbf{{{mae_all_overall:.1f}}}  & {mae_train_overall:.1f}  & {mae_test_overall:.1f}")
        f.write(f" & {fwd_bias_overall:+.0f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n")
        f.write("\\smallskip\n\n")
        f.write("\\noindent\\footnotesize{\\(^\\dagger\\) Large positive forward bias at 10Y expiry masks under-volatility; ")
        f.write("see Section~\\ref{subsec:cmpr_term_structure}.}\n")
        f.write("\\end{table}\n")

    print(f"  ✓ Saved: {out_path}")
    print(f"    Stable MAE={stable_mae_overall:.1f}, Pricing Layer MAEs: All={mae_all_overall:.1f}, Train={mae_train_overall:.1f}, Test={mae_test_overall:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: Term Structure Trained (Tab 3.3)
# ══════════════════════════════════════════════════════════════════════════════

def generate_term_structure_table():
    """Generate Table 3.3: model vs market vol after calibration with ratios"""
    print("\n[2/7] Generating term structure trained table (Table 3.3)...")
    
    IN_CSV = os.path.join(
        PROJECT_ROOT, "Figures", "TrainingResults", "dim4_constant_mpr",
        "term_structure_diag_trained", "term_structure_table_trained.csv"
    )
    OUT_DIR = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "comparison")
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_TEX = os.path.join(OUT_DIR, "tab_term_structure_trained.tex")

    df = pd.read_csv(IN_CSV)
    expiries = sorted(df["Expiry"].unique())
    tenors = sorted(df["Tenor"].unique())

    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model-implied versus market normal volatility after pricing-layer calibration ")
        f.write("(mean over a 40-date sample). The ratio mod/mkt above 1 indicates over-volatility; ")
        f.write("below 1 indicates under-volatility. Values in basis points.}\n")
        f.write("\\label{tab:term_structure_trained}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{@{}ccrrc@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Expiry} & \\textbf{Tenor} & \\textbf{Model \\(\\sigma_F\\)} & ")
        f.write("\\textbf{Market \\(\\sigma_N\\)} & \\textbf{Ratio} \\\\\n")
        f.write("\\midrule\n")
        
        for i, exp in enumerate(expiries):
            if i > 0:
                f.write("\\midrule\n")
            for ten in tenors:
                row = df[(df["Expiry"] == exp) & (df["Tenor"] == ten)]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                model_vol = row["Model σ_F (bp)"]
                market_vol = row["Market σ_N (bp)"]
                ratio = row["Ratio (mod/mkt)"]
                f.write(f"{exp}Y  & {ten}Y  & {model_vol:.0f} & {market_vol:.0f} & {ratio:.2f}$\\times$ \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  ✓ Saved: {OUT_TEX}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Timeseries Comparison (Fig 3.2)
# ══════════════════════════════════════════════════════════════════════════════

def generate_timeseries_figure():
    """Generate Figure 3.2: Vol error timeseries"""
    print("\n[3/7] Generating timeseries comparison (Figure 3.2)...")
    
    CMPR_CSV = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "eval_constant_mpr", "per_cell_final.csv")
    OUT_DIR  = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "comparison")
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CMPR_CSV):
        print(f"  ✗ Evaluation data not found: {CMPR_CSV}")
        return
    
    df = pd.read_csv(CMPR_CSV)
    df["date"] = pd.to_datetime(df["date"])

    EXPIRIES = [1, 5, 10]
    TENORS = [1, 5, 10]
    row_colors = ["#2563eb", "#16a34a", "#dc2626"]  # blue, green, red by expiry

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    
    for i, e in enumerate(EXPIRIES):
        for j, t in enumerate(TENORS):
            ax = axes[i][j]
            cell = df[(df["expiry"]==e) & (df["tenor"]==t)].sort_values("date")
            if len(cell) == 0:
                continue
            
            ax.plot(cell["date"], cell["vol_error_bp"], color=row_colors[i], linewidth=1.2)
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            
            # Shade test period
            test_start = pd.Timestamp("2022-01-01")
            ax.axvspan(test_start, cell["date"].max(), alpha=0.1, color="grey")
            
            ax.set_title(f"{e}Y×{t}Y", fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 2:
                ax.set_xlabel("Date", fontsize=9)
            if j == 0:
                ax.set_ylabel("Error (bp)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_pricing_comparison_timeseries.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3.3: Term Structure Vol by Tenor (from base model diagnostics)
# ══════════════════════════════════════════════════════════════════════════════

def generate_term_structure_vol_figure():
    """Generate Figure 3.3: σ_F vs expiry per tenor (base model)"""
    print("\n[4/7] Generating term structure vol by tenor (Figure 3.3)...")
    
    IN_CSV = os.path.join(
        PROJECT_ROOT, "Figures", "TrainingResults", "dim4_constant_mpr",
        "term_structure_diag", "term_structure_table.csv"
    )
    OUT_DIR = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "comparison")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(IN_CSV):
        print(f"  ✗ Base model diagnostics not found: {IN_CSV}")
        print("    Run _term_structure_vol_diagnostics.py first to generate base model data.")
        return
    
    df = pd.read_csv(IN_CSV)
    EXPIRIES = sorted(df["Expiry"].unique())
    TENORS = sorted(df["Tenor"].unique())
    
    fig, axes = plt.subplots(1, len(TENORS), figsize=(5 * len(TENORS), 4.5), dpi=150, sharey=True)
    colors_mod = ["#d62728", "#e5771a", "#9467bd"]
    colors_mkt = ["#1f77b4", "#2ca02c", "#8c564b"]
    
    for j, ten in enumerate(TENORS):
        ax = axes[j] if len(TENORS) > 1 else axes
        mod_means = []
        mkt_means = []
        
        for e in EXPIRIES:
            row = df[(df["Expiry"] == e) & (df["Tenor"] == ten)]
            if len(row) == 0:
                mod_means.append(np.nan)
                mkt_means.append(np.nan)
            else:
                mod_means.append(row.iloc[0]["Model σ_F (bp)"])
                mkt_means.append(row.iloc[0]["Market σ_N (bp)"])
        
        ax.plot(EXPIRIES, mod_means, "o-", color=colors_mod[j], lw=2.0, ms=7,
                label="Model $\\sigma_F$ (base, $s=1$)")
        ax.plot(EXPIRIES, mkt_means, "s--", color=colors_mkt[j], lw=2.0, ms=7,
                label="Market $\\sigma_N$")
        
        ax.set_title(f"Tenor = {ten}Y", fontsize=12)
        ax.set_xlabel("Expiry (years)", fontsize=11)
        ax.set_xticks(EXPIRIES)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    if len(TENORS) > 1:
        axes[0].set_ylabel("ATM normal vol (bp)", fontsize=11)
    else:
        axes.set_ylabel("ATM normal vol (bp)", fontsize=11)
    
    fig.suptitle(
        "Term structure of $\\sigma_F$: base model vs market\n"
        "(Decreasing model vol reveals structural mean-reversion mismatch)",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "term_structure_vol_by_tenor.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3.4: Scaling Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def generate_scaling_sensitivity_figure():
    """Generate Figure 3.4: Scaling sensitivity — what σ_vec does to each expiry"""
    print("\n[5/7] Generating scaling sensitivity (Figure 3.4)...")
    
    IN_CSV = os.path.join(
        PROJECT_ROOT, "Figures", "TrainingResults", "dim4_constant_mpr",
        "term_structure_diag", "term_structure_table.csv"
    )
    OUT_DIR = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "comparison")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(IN_CSV):
        print(f"  ✗ Base model diagnostics not found: {IN_CSV}")
        return
    
    df = pd.read_csv(IN_CSV)
    EXPIRIES = sorted(df["Expiry"].unique())
    TENORS = sorted(df["Tenor"].unique())
    
    scale_factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    tenor_ref = TENORS[0]  # show for first tenor only
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    colors_exp = ["#d62728", "#ff7f0e", "#2ca02c"]
    
    for i, exp in enumerate(EXPIRIES):
        row = df[(df["Expiry"] == exp) & (df["Tenor"] == tenor_ref)]
        if len(row) == 0:
            continue
        
        base_sig = row.iloc[0]["Model σ_F (bp)"]
        mkt_sig = row.iloc[0]["Market σ_N (bp)"]
        
        # σ_F scales linearly with σ_vec
        scaled_sigs = base_sig * scale_factors
        ax.plot(scale_factors, scaled_sigs, "o-", color=colors_exp[i], lw=2, ms=7,
                label=f"Expiry {exp}Y (model)")
        ax.axhline(float(mkt_sig), color=colors_exp[i], lw=1.5, ls="--",
                   alpha=0.7, label=f"Market {exp}Y = {mkt_sig:.0f} bp")
    
    ax.set_xlabel("σ_vec scale factor  $s$", fontsize=11)
    ax.set_ylabel("σ_F (bp)", fontsize=11)
    ax.set_title(
        f"No single $s$ reconciles all expiries  (tenor = {tenor_ref}Y)\n"
        "Dashed lines = market target per expiry",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "scaling_sensitivity.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3.5: Forward Bias Timeseries
# ══════════════════════════════════════════════════════════════════════════════

def generate_forward_bias_figure():
    """Generate Figure 3.5: Forward bias over time"""
    print("\n[6/7] Generating forward bias timeseries (Figure 3.5)...")
    
    CMPR_CSV = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "eval_constant_mpr", "per_cell_final.csv")
    OUT_DIR = os.path.join(PROJECT_ROOT, "Figures", "Pricing", "comparison")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(CMPR_CSV):
        print(f"  ✗ Evaluation data not found: {CMPR_CSV}")
        return
    
    df = pd.read_csv(CMPR_CSV)
    df["date"] = pd.to_datetime(df["date"])
    
    EXPIRIES = [1, 5, 10]
    TENORS = [1, 5, 10]
    
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for i, e in enumerate(EXPIRIES):
        for j, t in enumerate(TENORS):
            ax = axes[i][j]
            sub = df[(df["expiry"] == e) & (df["tenor"] == t)].sort_values("date")
            if len(sub) == 0:
                ax.set_visible(False)
                continue
            
            # Shade test region
            test_sub = sub[sub["split"] == "test"]["date"]
            if len(test_sub):
                ax.axvspan(test_sub.min(), test_sub.max(), alpha=0.07, color="#f59e0b")
            
            ax.plot(sub["date"], sub["forward_bias_bp"], color="#7c3aed", lw=1.1)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.fill_between(sub["date"], sub["forward_bias_bp"], 0,
                           where=sub["forward_bias_bp"] > 0, alpha=0.12, color="#dc2626")
            ax.fill_between(sub["date"], sub["forward_bias_bp"], 0,
                           where=sub["forward_bias_bp"] < 0, alpha=0.12, color="#2563eb")
            
            mean_fwd = sub["forward_bias_bp"].mean()
            ax.text(0.03, 0.97, f"mean = {mean_fwd:+.0f} bp",
                   transform=ax.transAxes, fontsize=7, va="top",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
            
            ax.set_title(f"{e}Yx{t}Y", fontsize=9, fontweight="bold")
            if i == 2:
                ax.set_xlabel("Date", fontsize=7)
            if j == 0:
                ax.set_ylabel("Fwd bias (bp)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
    
    fig.suptitle("Forward Bias $(V_\\mathrm{pay} - V_\\mathrm{rec})/A_0$"
                 " (target: 0 bp, amber = test)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUT_DIR, "fig_forward_bias_timeseries.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3.6: F_T Distribution per Expiry
# ══════════════════════════════════════════════════════════════════════════════

def generate_ft_distribution_figure():
    """Generate Figure 3.6: Distribution of F_T - F_0 per expiry"""
    print("\n[7/7] Generating F_T distribution per expiry (Figure 3.6)...")
    
    # This figure requires Monte Carlo simulation data, which would typically be
    # generated by _term_structure_vol_diagnostics.py. For now, we note that this
    # data needs to be available or generated separately.
    
    print("  ⚠ Figure 3.6 requires Monte Carlo simulation data from base model diagnostics.")
    print("    Run _term_structure_vol_diagnostics.py first to generate this data.")
    print("    (This figure shows the F_T distribution variance saturation effect)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*80)
    print("PRICING CHAPTER RESULTS GENERATOR")
    print("="*80)
    print("\nGenerating all tables and figures for the Pricing chapter...")
    
    # Tables
    generate_combined_table()
    generate_term_structure_table()
    
    # Figures
    generate_timeseries_figure()
    generate_term_structure_vol_figure()
    generate_scaling_sensitivity_figure()
    generate_forward_bias_figure()
    generate_ft_distribution_figure()
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print("\nAll pricing chapter results generated!")
    print("\nTables:")
    print("  - Figures/Pricing/comparison/tab_pricing_combined.tex (Table 3.2)")
    print("  - Figures/Pricing/comparison/tab_term_structure_trained.tex (Table 3.3)")
    print("\nFigures:")
    print("  - Figures/Pricing/comparison/fig_pricing_comparison_timeseries.png (Figure 3.2)")
    print("  - Figures/Pricing/comparison/term_structure_vol_by_tenor.png (Figure 3.3)")
    print("  - Figures/Pricing/comparison/scaling_sensitivity.png (Figure 3.4)")
    print("  - Figures/Pricing/comparison/fig_forward_bias_timeseries.png (Figure 3.5)")
    print("\nNote: Figure 3.6 (F_T distribution) requires Monte Carlo simulation data.")
    print("      Run _term_structure_vol_diagnostics.py first if this data is needed.")
    print("="*80)


if __name__ == "__main__":
    main()

