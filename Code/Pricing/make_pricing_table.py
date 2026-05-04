"""
make_pricing_table.py
=====================
Generate the LaTeX summary table for the pricing chapter from the two
Excel files produced by compare_swapvol.py.

The table reports, for each of {Stage-1, Stage-2, Flat-vol baseline}
× {train, test (OOS)} × overall:
    • MAE (bp)
    • RMSE (bp)
    • fraction of cells within 5 bp of market
    • median MC SE (vol space, bp)   [Stage-1/2 only]

Output
------
    swapvol_results/tab_vol_comparison.tex    (LaTeX fragment, \input{} ready)
    swapvol_results/tab_vol_comparison.xlsx   (wide-format for checking)

Usage
-----
    python make_pricing_table.py

Or call run(pre_xlsx, post_xlsx, split_date, out_dir) from another script.
"""

import os
import sys

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in [SCRIPT_DIR, PROJECT_ROOT, THESIS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

# =============================================================================
# DEFAULTS
# =============================================================================
PRE_XLSX   = os.path.join(SCRIPT_DIR, "swapvol_results",
                          "swaption_vol_comparison_stage1.xlsx")
POST_XLSX  = os.path.join(SCRIPT_DIR, "swapvol_results",
                          "swaption_vol_comparison_stage2.xlsx")
SPLIT_DATE = "2018-12-31"
OUT_DIR    = os.path.join(SCRIPT_DIR, "swapvol_results")

# Threshold for "within X bp" statistic
WITHIN_BP  = 5.0


# =============================================================================
# HELPER — load one Excel
# =============================================================================

def _load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df["market_as_of_date"] = pd.to_datetime(df["market_as_of_date"]).dt.normalize()
    df["option_maturity"]   = df["option_maturity"].astype(int)
    df["swap_tenor"]        = df["swap_tenor"].astype(int)
    for col in ["market_vol_bp", "model_vol_bp", "vol_error_bp",
                "abs_vol_error_bp", "mc_se_vol_bp", "flat_vol_bp",
                "flat_vol_error_bp", "abs_flat_error_bp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "split" not in df.columns:
        df["split"] = "all"
    return df


# =============================================================================
# COMPUTE METRICS for one (dataframe, error_col) slice
# =============================================================================

def _metrics(df, error_bp_col, abs_error_col, mc_se_col=None, within_bp=WITHIN_BP):
    """Return dict of scalar metrics for one subset."""
    sub = df.dropna(subset=[abs_error_col])
    n   = len(sub)
    if n == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, within_pct=np.nan, median_se=np.nan)

    mae      = float(sub[abs_error_col].mean())
    rmse     = float((sub[error_bp_col] ** 2).mean() ** 0.5)
    within   = float((sub[abs_error_col] <= within_bp).mean() * 100.0)
    median_se = (float(sub[mc_se_col].median())
                 if (mc_se_col is not None and mc_se_col in sub.columns)
                 else np.nan)
    return dict(n=n, mae=mae, rmse=rmse, within_pct=within, median_se=median_se)


# =============================================================================
# BUILD SUMMARY DATA-FRAME
# =============================================================================

def build_summary(df_pre, df_post, split_date=None, within_bp=WITHIN_BP):
    """
    Return a long-format DataFrame with one row per
    (model, split) combination.
    """
    # Flat-vol baseline: estimated from training rows of df_pre
    if split_date is not None and "split" in df_pre.columns:
        train_rows = df_pre[df_pre["split"] == "train"]
    else:
        train_rows = df_pre

    sigma_flat_bp = float(train_rows["market_vol_bp"].mean())
    for df in (df_pre, df_post):
        df["flat_vol_bp"]       = sigma_flat_bp
        df["flat_vol_error_bp"] = df["flat_vol_bp"] - df["market_vol_bp"]
        df["abs_flat_error_bp"] = df["flat_vol_error_bp"].abs()

    splits = ["all"]
    if split_date is not None:
        splits = ["train", "test", "all"]

    rows = []
    for split in splits:
        for model_label, df, err_col, abs_col, se_col in [
            ("Stage-1 (pre)",  df_pre,  "vol_error_bp",      "abs_vol_error_bp",  "mc_se_vol_bp"),
            ("Stage-2 (post)", df_post, "vol_error_bp",      "abs_vol_error_bp",  "mc_se_vol_bp"),
            ("Flat-vol",       df_pre,  "flat_vol_error_bp", "abs_flat_error_bp", None),
        ]:
            if split == "all":
                sub = df
            else:
                sub = df[df["split"] == split] if "split" in df.columns else df

            m = _metrics(sub, err_col, abs_col, mc_se_col=se_col, within_bp=within_bp)
            rows.append({
                "model"      : model_label,
                "split"      : split,
                "N"          : m["n"],
                "MAE (bp)"   : round(m["mae"],  1) if np.isfinite(m["mae"]) else np.nan,
                "RMSE (bp)"  : round(m["rmse"], 1) if np.isfinite(m["rmse"]) else np.nan,
                f"Within {within_bp:.0f} bp (%)": round(m["within_pct"], 1)
                               if np.isfinite(m["within_pct"]) else np.nan,
                "Median MC SE (bp)": round(m["median_se"], 2)
                               if (m["median_se"] is not None and np.isfinite(m["median_se"]))
                               else "—",
            })

    return pd.DataFrame(rows), sigma_flat_bp


# =============================================================================
# RENDER LATEX
# =============================================================================

def _fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, str):
        return x
    return str(x)


def render_latex(df_summary, sigma_flat_bp, within_bp=WITHIN_BP, split_date=None):
    """
    Return a LaTeX booktabs table string.
    """
    within_col = f"Within {within_bp:.0f} bp (%)"
    cols = ["MAE (bp)", "RMSE (bp)", within_col, "Median MC SE (bp)"]

    splits_in_table = df_summary["split"].unique().tolist()
    show_split = len(splits_in_table) > 1 and "all" not in splits_in_table

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    split_tag = (f", train/test split at {split_date}" if split_date else "")
    lines.append(r"\caption{ATM swaption vol comparison: model vs.\ market normal vol"
                 f" (EUR{split_tag}).  "
                 r"MAE and RMSE in basis points.  "
                 r"``Within 5 bp'' is the fraction of cells within 5 bp of market vol.  "
                 r"MC SE is the median vol-space Monte Carlo standard error.}"
                 )
    lines.append(r"\label{tab:vol_comparison}")

    if show_split:
        n_cols = 1 + 1 + len(cols)   # model + split + metrics
        lines.append(r"\begin{tabular}{ll" + "r" * len(cols) + r"}")
        lines.append(r"\toprule")
        header = "Model & Split & " + " & ".join(cols) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        for _, row in df_summary.iterrows():
            model_str = row["model"]
            sp_str    = row["split"].capitalize() if row["split"] != "all" else "All"
            vals      = " & ".join(_fmt(row[c]) for c in cols)
            if row["split"] == "train":
                lines.append(r"\addlinespace[2pt]")
            lines.append(f"{model_str} & {sp_str} & {vals} " + r"\\")
    else:
        lines.append(r"\begin{tabular}{l" + "r" * len(cols) + r"}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(cols) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        for _, row in df_summary.iterrows():
            vals = " & ".join(_fmt(row[c]) for c in cols)
            lines.append(f"{row['model']} & {vals} " + r"\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(f"\\item Flat-vol baseline uses $\\hat{{\\sigma}}_{{\\rm flat}} = "
                 f"{sigma_flat_bp:.1f}$ bp, estimated as the mean ATM market vol over "
                 f"{'training dates' if split_date else 'all dates'}.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================

def run(
    pre_xlsx   = PRE_XLSX,
    post_xlsx  = POST_XLSX,
    split_date = SPLIT_DATE,
    out_dir    = OUT_DIR,
    within_bp  = WITHIN_BP,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading pre  : {pre_xlsx}")
    df_pre  = _load(pre_xlsx)
    print(f"Loading post : {post_xlsx}")
    df_post = _load(post_xlsx)

    df_summary, sigma_flat_bp = build_summary(
        df_pre, df_post, split_date=split_date, within_bp=within_bp
    )

    print("\nSummary table:")
    print(df_summary.to_string(index=False))

    # ── Save Excel ───────────────────────────────────────────────────
    xlsx_path = os.path.join(out_dir, "tab_vol_comparison.xlsx")
    df_summary.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"\nSaved Excel → {xlsx_path}")

    # ── Save LaTeX ───────────────────────────────────────────────────
    latex_str = render_latex(df_summary, sigma_flat_bp,
                             within_bp=within_bp, split_date=split_date)
    tex_path = os.path.join(out_dir, "tab_vol_comparison.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"Saved LaTeX → {tex_path}")

    return df_summary, latex_str


if __name__ == "__main__":
    run()


