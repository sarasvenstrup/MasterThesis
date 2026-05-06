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

import argparse
import glob
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
# HELPER — locate Excel produced by compare_swapvol.py
# =============================================================================

def _find_swapvol_file(checkpoint_name: str, search_dir: str) -> str:
    """
    Return the path of the Excel file for *checkpoint_name* in *search_dir*.
    Raises FileNotFoundError if not found.
    """
    candidate = os.path.join(search_dir, f"swaption_vol_comparison_{checkpoint_name}.xlsx")
    if os.path.isfile(candidate):
        return candidate
    matches = glob.glob(os.path.join(search_dir, f"*{checkpoint_name}*.xlsx"))
    vol_matches = [m for m in matches if "vol_comparison" in os.path.basename(m)]
    if vol_matches:
        return vol_matches[0]
    raise FileNotFoundError(
        f"No swaption_vol_comparison Excel found for '{checkpoint_name}' in {search_dir!r}"
    )


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

def build_summary(dataframes, labels, split_date=None, within_bp=WITHIN_BP):
    """
    Return a long-format DataFrame with one row per (model, split) combination,
    plus a flat-vol baseline row estimated from the first dataframe's training rows.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        Each DataFrame is the output of compare_swapvol.comparison_table().
    labels : list of str
        Human-readable label for each DataFrame.
    split_date : str or None
    within_bp : float
    """
    if not dataframes:
        return pd.DataFrame(), np.nan

    # Flat-vol baseline estimated from the first dataframe's training (or all) rows
    df_ref = dataframes[0]
    if split_date is not None and "split" in df_ref.columns:
        train_rows = df_ref[df_ref["split"] == "train"]
    else:
        train_rows = df_ref
    sigma_flat_bp = float(train_rows["market_vol_bp"].mean())

    for df in dataframes:
        df["flat_vol_bp"]       = sigma_flat_bp
        df["flat_vol_error_bp"] = df["flat_vol_bp"] - df["market_vol_bp"]
        df["abs_flat_error_bp"] = df["flat_vol_error_bp"].abs()

    splits = ["all"]
    if split_date is not None:
        splits = ["train", "test", "all"]

    rows = []
    for split in splits:
        # Model rows
        for label, df in zip(labels, dataframes):
            sub = df if split == "all" else (
                df[df["split"] == split] if "split" in df.columns else df
            )
            m = _metrics(sub, "vol_error_bp", "abs_vol_error_bp",
                         mc_se_col="mc_se_vol_bp", within_bp=within_bp)
            rows.append({
                "model"      : label,
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

        # Flat-vol baseline (uses first df only)
        sub = df_ref if split == "all" else (
            df_ref[df_ref["split"] == split] if "split" in df_ref.columns else df_ref
        )
        m = _metrics(sub, "flat_vol_error_bp", "abs_flat_error_bp",
                     mc_se_col=None, within_bp=within_bp)
        rows.append({
            "model"      : "Flat-vol baseline",
            "split"      : split,
            "N"          : m["n"],
            "MAE (bp)"   : round(m["mae"],  1) if np.isfinite(m["mae"]) else np.nan,
            "RMSE (bp)"  : round(m["rmse"], 1) if np.isfinite(m["rmse"]) else np.nan,
            f"Within {within_bp:.0f} bp (%)": round(m["within_pct"], 1)
                           if np.isfinite(m["within_pct"]) else np.nan,
            "Median MC SE (bp)": "—",
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
    checkpoint_names,
    labels=None,
    split_date=None,
    out_dir=OUT_DIR,
    within_bp=WITHIN_BP,
):
    """
    Generate comparison table for multiple checkpoints.
    
    Args:
        checkpoint_names: list of checkpoint names (without extension)
        labels: list of human-readable labels for each model
        split_date: train/test split date (ISO string or None)
        out_dir: output directory
        within_bp: threshold for "within X bp" metric
    """
    os.makedirs(out_dir, exist_ok=True)

    if labels is None:
        labels = checkpoint_names

    if len(checkpoint_names) != len(labels):
        raise ValueError(f"Number of checkpoint_names ({len(checkpoint_names)}) must match number of labels ({len(labels)})")

    # Load swapvol data
    dataframes = []
    for name in checkpoint_names:
        try:
            xlsx_path = _find_swapvol_file(name, out_dir)
            print(f"Loading {name}: {xlsx_path}")
            df = _load(xlsx_path)
            dataframes.append(df)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    if not dataframes:
        print("ERROR: No swapvol files found!")
        return None, None
    
    # Update labels if some files were skipped
    if len(dataframes) < len(checkpoint_names):
        loaded = []
        for i, name in enumerate(checkpoint_names):
            try:
                _find_swapvol_file(name, out_dir)
                loaded.append(labels[i])
            except FileNotFoundError:
                pass
        labels = loaded

    df_summary, sigma_flat_bp = build_summary(
        dataframes, labels, split_date=split_date, within_bp=within_bp
    )

    print("\nSummary table:")
    print(df_summary.to_string(index=False))


    # Create tag from checkpoint names
    tag = "_vs_".join(checkpoint_names[:2]) if len(checkpoint_names) <= 2 else f"_{len(checkpoint_names)}models"

    # ── Save Excel ───────────────────────────────────────────────────
    xlsx_path = os.path.join(out_dir, f"tab_vol_comparison_{tag}.xlsx")
    df_summary.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"\nSaved Excel → {xlsx_path}")

    # ── Save LaTeX ───────────────────────────────────────────────────
    latex_str = render_latex(df_summary, sigma_flat_bp,
                             within_bp=within_bp, split_date=split_date)
    tex_path = os.path.join(out_dir, f"tab_vol_comparison_{tag}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"Saved LaTeX → {tex_path}")

    return df_summary, latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pricing comparison table")
    parser.add_argument("--files", nargs="+", required=True,
                       help="Checkpoint names (e.g., checkpoint_dim4_ep3500)")
    parser.add_argument("--labels", nargs="+", default=None,
                       help="Labels for each model (optional)")
    parser.add_argument("--split_date", type=str, default=None,
                       help="Train/test split date (YYYY-MM-DD)")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                       help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--within_bp", type=float, default=WITHIN_BP,
                       help=f"Threshold for 'within X bp' metric (default: {WITHIN_BP})")
    
    args = parser.parse_args()
    
    run(
        checkpoint_names=args.files,
        labels=args.labels,
        split_date=args.split_date,
        out_dir=args.out_dir,
        within_bp=args.within_bp,
    )


