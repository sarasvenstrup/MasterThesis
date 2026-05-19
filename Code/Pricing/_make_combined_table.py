"""
Generate combined pricing table for thesis: market, base, pricing layer, MAE (all/train/test), fwd bias.

Saves to: Figures/Pricing/comparison/tab_pricing_combined.tex
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load data
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
        base   = b_all["sigma_str_bp"].mean() if len(b_all) else np.nan
        layer  = c_all["sigma_str_bp"].mean()
        
        mae_all   = c_all["vol_error_bp"].abs().mean()
        mae_train = c_train["vol_error_bp"].abs().mean() if len(c_train) else np.nan
        mae_test  = c_test["vol_error_bp"].abs().mean() if len(c_test) else np.nan
        fwd_bias  = c_all["forward_bias_bp"].mean()
        
        rows.append({
            "exp": exp,
            "ten": ten,
            "mkt": mkt,
            "base": base,
            "layer": layer,
            "mae_all": mae_all,
            "mae_train": mae_train,
            "mae_test": mae_test,
            "fwd_bias": fwd_bias,
        })

# Overall row
mkt_overall    = df_cmpr["mkt_bp"].mean()
base_overall   = df_base["sigma_str_bp"].mean()
layer_overall  = df_cmpr["sigma_str_bp"].mean()
mae_all_overall   = df_cmpr["vol_error_bp"].abs().mean()
mae_train_overall = df_cmpr[df_cmpr["split"]=="train"]["vol_error_bp"].abs().mean()
mae_test_overall  = df_cmpr[df_cmpr["split"]=="test"]["vol_error_bp"].abs().mean()
fwd_bias_overall  = df_cmpr["forward_bias_bp"].mean()

# Write LaTeX table
out_path = os.path.join(OUT_DIR, "tab_pricing_combined.tex")

with open(out_path, "w") as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{Per-cell ATM straddle normal volatility: market, base model (no pricing adjustment), "
            "and calibrated pricing layer (all dates, EUR). MAE and forward bias are reported for the pricing layer. "
            "Values in basis points.}\n")
    f.write("\\label{tab:pricing_combined}\n")
    f.write("\\small\n")
    f.write("\\resizebox{\\textwidth}{!}{%\n")
    f.write("\\begin{tabular}{@{}ccrrrrrrr@{}}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Exp} & \\textbf{Ten}\n")
    f.write("  & \\textbf{Market}\n")
    f.write("  & \\textbf{Base model}\n")
    f.write("  & \\textbf{Pricing layer}\n")
    f.write("  & \\textbf{MAE (all)}\n")
    f.write("  & \\textbf{MAE (train)}\n")
    f.write("  & \\textbf{MAE (test)}\n")
    f.write("  & \\textbf{Fwd bias} \\\\\n")
    f.write("\\midrule\n")
    
    for i, r in enumerate(rows):
        exp_label = f"{r['exp']}Y"
        ten_label = f"{r['ten']}Y"
        
        # Add midrule between expiry groups
        if i > 0 and rows[i-1]["exp"] != r["exp"]:
            f.write("\\midrule\n")
        
        # Mark 10Y cells with dagger for masking effect
        bias_str = f"{r['fwd_bias']:+.0f}"
        if r["exp"] == 10:
            bias_str += "\\(^\\dagger\\)"
        
        f.write(f"  {exp_label} & {ten_label}")
        f.write(f"  & {r['mkt']:.0f}")
        f.write(f"  & {r['base']:.0f}")
        f.write(f"  & {r['layer']:.0f}")
        f.write(f"  & {r['mae_all']:.1f}")
        f.write(f"  & {r['mae_train']:.1f}")
        f.write(f"  & {r['mae_test']:.1f}")
        f.write(f"  & {bias_str} \\\\\n")
    
    # Overall row
    f.write("\\midrule\n")
    f.write("  \\multicolumn{2}{c}{\\textbf{All cells}}")
    f.write(f" & {mkt_overall:.0f}")
    f.write(f" & {base_overall:.0f}")
    f.write(f" & {layer_overall:.0f}")
    f.write(f" & \\textbf{{{mae_all_overall:.1f}}}")
    f.write(f" & {mae_train_overall:.1f}")
    f.write(f" & {mae_test_overall:.1f}")
    f.write(f" & {fwd_bias_overall:+.0f} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}}\n")
    f.write("\\smallskip\n\n")
    f.write("\\noindent\\footnotesize{\\(^\\dagger\\) Large positive forward bias at 10Y expiry masks under-volatility; ")
    f.write("see Section~\\ref{subsec:cmpr_term_structure}.}\n")
    f.write("\\end{table}\n")

print(f"Saved: {out_path}")
print(f"\nOverall MAEs: All={mae_all_overall:.1f}, Train={mae_train_overall:.1f}, Test={mae_test_overall:.1f}")

