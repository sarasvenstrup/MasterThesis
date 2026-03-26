import os
import re
import pandas as pd
from typing import Tuple, Dict, List
import numpy as np
from utils import helpers as H

import torch
import matplotlib as mpl
import seaborn as sns
from cycler import cycler

# ---------------------------------------------------------
# 0) Repo-rooted path (no Desktop). Adjust if needed.
# ---------------------------------------------------------
# Assumes you run from repo root.
#REPO_ROOT = os.getcwd()
#ROOT = os.path.join(REPO_ROOT, "SwapData")  # <-- your folder in the repo

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # <- parent of Code
ROOT = os.path.join(REPO_ROOT, "SwapData")
print("Repo root:", REPO_ROOT)
#print("Data root:", DATA_ROOT)


# If you want both datasets on the same maturity grid (as in your model/paper)
TARGET_TENORS: List[int] = [1, 2, 3, 5, 10, 15, 20, 30]

# ============================= Set Figure/Plot Theme ===============================

def set_paper_theme():
    # 1) Use seaborn only to define a nice clean theme (works for matplotlib plots too)
    sns.set_theme(context="paper", style="darkgrid", font_scale=1.05)

    # Customize tab20b palette
    full_palette = sns.color_palette("tab20b", 20)
    selected_indices = [0, 1, 2, 3, 12, 13, 14, 15, 8]
    palette = [full_palette[i] for i in selected_indices]


    # 3) Global matplotlib defaults (applies to ALL figures you create afterwards)
    mpl.rcParams.update({
        # Figure / saving
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Light grey full frame
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "0.8",  # light grey frame
        "axes.linewidth": 1.0,

        # Grid styling
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 1.0,

        # Text
        "font.size": 11,
        "axes.labelcolor": "0.2",
        "xtick.color": "0.2",
        "ytick.color": "0.2",

        # Legend
        "legend.frameon": False,

        # Lines default
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,


    })

    # 4) Make the palette the default color cycle for matplotlib
    mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)

    return palette
def style_axis(ax, title=None, xlabel=None, ylabel=None, legend=True, legend_kwargs=None):
    """Optional helper you can call per-figure for consistent finishing touches."""
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Ensure consistent grid/spines (in case some plots override)
    ax.grid(True, which="major", axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if legend:
        kw = dict(frameon=False)
        if legend_kwargs:
            kw.update(legend_kwargs)
        ax.legend(**kw)

# We now call the above function to ensure the wanted theme of the outputs from this script.
custom_palette = set_paper_theme()

# -----------------------------
# Currency rename + colors
# -----------------------------
currency_rename_map = {
    "ad": "AUD", "AD": "AUD",
    "cd": "CAD", "CD": "CAD",
    "dk": "DKK", "DK": "DKK",
    "eu": "EUR", "EU": "EUR",
    "jy": "JPY", "JY": "JPY",
    "nk": "NOK", "NK": "NOK",
    "sw": "SEK", "SK": "SEK",
    "uk": "GBP", "UK": "GBP",
    "us": "USD", "US": "USD",
}

# ---------------------------------------------------------
# 1) Helpers
# ---------------------------------------------------------
def extract_maturity_years(filename: str) -> int:
    m = re.search(r"(\d+)\s*year", filename.lower())
    if not m:
        raise ValueError(f"Cannot infer maturity from filename: {filename}")
    return int(m.group(1))


def read_bloomberg_style_excel(path: str) -> pd.DataFrame:
    """
    Handles files with metadata rows + a data table with headers: Date | PX_LAST
    Returns columns: as_of_date (datetime64[ns]), swap_rate (float)
    """
    raw = pd.read_excel(path, header=None, engine="openpyxl")

    header_row = None
    for i in range(min(len(raw), 200)):
        row = raw.iloc[i].astype(str).str.strip().str.lower().values
        if ("date" in row) and ("px_last" in row):
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Header row with Date/PX_LAST not found in {path}")

    df = pd.read_excel(path, header=header_row, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    df = df[["Date", "PX_LAST"]].dropna(subset=["Date", "PX_LAST"])
    df["as_of_date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Handle comma decimals like 4,66571 and spaces
    s = (
        df["PX_LAST"]
        .astype(str)
        .str.replace(" ", "")
        .str.replace(",", ".", regex=False)
    )
    df["swap_rate"] = pd.to_numeric(s, errors="coerce")

    df = df.dropna(subset=["as_of_date", "swap_rate"])
    return df[["as_of_date", "swap_rate"]]


def load_dataset(dataset_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads all excel files under dataset_root, infers:
      - ccy = folder right above file
      - maturity_years from filename "<something>10year<something>.xlsx"
    Returns:
      df_long: [as_of_date, ccy, maturity_years, swap_rate, source_file]
      df_errors: [source_file, error]
    """
    frames = []
    errors = []

    for dirpath, _, filenames in os.walk(dataset_root):
        for fn in filenames:
            if not fn.lower().endswith((".xlsx", ".xls")):
                continue

            path = os.path.join(dirpath, fn)
            ccy = os.path.basename(os.path.dirname(path)).lower()

            try:
                maturity = extract_maturity_years(fn)
                tmp = read_bloomberg_style_excel(path)
                tmp["ccy"] = ccy
                tmp["maturity_years"] = maturity
                tmp["source_file"] = path
                frames.append(tmp)
            except Exception as e:
                errors.append({"source_file": path, "error": str(e)})

    df_errors = pd.DataFrame(errors)

    if not frames:
        df_long = pd.DataFrame(
            columns=["as_of_date", "ccy", "maturity_years", "swap_rate", "source_file"]
        )
        return df_long, df_errors

    df_long = pd.concat(frames, ignore_index=True)
    df_long = df_long[["as_of_date", "ccy", "maturity_years", "swap_rate", "source_file"]]

    # Sort
    df_long = df_long.sort_values(["ccy", "maturity_years", "as_of_date"]).reset_index(drop=True)

    # If duplicates (same date/ccy/maturity across multiple files), keep last
    df_long = (
        df_long.sort_values(["ccy", "maturity_years", "as_of_date", "source_file"])
        .drop_duplicates(subset=["as_of_date", "ccy", "maturity_years"], keep="last")
        .reset_index(drop=True)
    )

    return df_long, df_errors


def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long swap quotes into curve matrix:
      index: (as_of_date, ccy)
      columns: maturity_years
      values: swap_rate
    """
    if df_long.empty:
        return pd.DataFrame()

    df_wide = (
        df_long.pivot_table(
            index=["as_of_date", "ccy"],
            columns="maturity_years",
            values="swap_rate",
            aggfunc="mean",
        )
        .sort_index()
    )

    # Ensure sorted maturities
    df_wide = df_wide.reindex(sorted(df_wide.columns), axis=1)

    # Keep maturity columns as ints (not np.int64) and reset index
    df_wide.columns = [int(c) for c in df_wide.columns]
    df_wide = df_wide.reset_index()

    return df_wide


def show_failures(df_errors: pd.DataFrame, title: str, n: int = 20) -> None:
    """Print failing files + errors (first n)."""
    if df_errors is None or df_errors.empty:
        return
    print(f"\n--- {title} failures (first {min(n, len(df_errors))}) ---")
    print(df_errors.head(n).to_string(index=False))


def align_to_target_tenors(df_wide: pd.DataFrame, tenors: List[int]) -> pd.DataFrame:
    """
    Keep only the tenors in 'tenors' (plus id columns).
    If a tenor doesn't exist, it will be created with NaN.
    """
    if df_wide.empty:
        return df_wide

    out = df_wide.copy()

    # Ensure missing tenor columns exist
    for t in tenors:
        if t not in out.columns:
            out[t] = pd.NA

    keep_cols = ["as_of_date", "ccy"] + tenors
    return out[keep_cols]


def keep_complete_curves(df_wide: pd.DataFrame, tenors: List[int]) -> pd.DataFrame:
    """Drop curves (rows) that have any missing swap_rate among the tenors."""
    if df_wide.empty:
        return df_wide
    out = df_wide.dropna(subset=tenors).copy()
    return out.sort_values(["ccy", "as_of_date"]).reset_index(drop=True)


def build_all_dataframes(root: str = ROOT, target_tenors: List[int] = TARGET_TENORS) -> Dict[str, object]:
    """
    Loads both TestData and Bloombergdata from repo-rooted SwapDAta folder,
    returns long+wide frames + errors.
    Also:
      - prints failing files (if any)
      - aligns both wide frames to the same tenor grid (target_tenors)
      - produces *_full versions containing only complete curves on that grid
    """
    test_root = os.path.join(root, "TestData")
    bbg_root = os.path.join(root, "Bloombergdata")

    df_long_test, err_test = load_dataset(test_root)
    df_long_bbg, err_bbg = load_dataset(bbg_root)

    df_wide_test = long_to_wide(df_long_test)
    df_wide_bbg = long_to_wide(df_long_bbg)

    # Tenors as plain python ints (cosmetic + convenient)
    tenors_test = sorted(map(int, df_long_test["maturity_years"].unique())) if not df_long_test.empty else []
    tenors_bbg  = sorted(map(int, df_long_bbg["maturity_years"].unique()))  if not df_long_bbg.empty  else []

    # Show failing files (if any)
    show_failures(err_test, "TestData")
    show_failures(err_bbg, "Bloombergdata")

    # Align to common grid
    df_wide_test_aligned = align_to_target_tenors(df_wide_test, target_tenors)
    df_wide_bbg_aligned  = align_to_target_tenors(df_wide_bbg,  target_tenors)

    # Keep only complete curves for training
    df_wide_test_full = keep_complete_curves(df_wide_test_aligned, target_tenors)
    df_wide_bbg_full  = keep_complete_curves(df_wide_bbg_aligned,  target_tenors)

    return {
        # long
        "df_long_test": df_long_test,
        "df_long_bbg": df_long_bbg,

        # wide (raw pivot)
        "df_wide_test": df_wide_test,
        "df_wide_bbg": df_wide_bbg,

        # wide (aligned tenor grid)
        "df_wide_test_aligned": df_wide_test_aligned,
        "df_wide_bbg_aligned": df_wide_bbg_aligned,

        # wide (aligned + complete only)
        "df_wide_test_full": df_wide_test_full,
        "df_wide_bbg_full": df_wide_bbg_full,

        # tenors + errors
        "tenors_test": tenors_test,
        "tenors_bbg": tenors_bbg,
        "target_tenors": list(target_tenors),
        "errors_test": err_test,
        "errors_bbg": err_bbg,
        "root_used": root,
    }


def my_data(use: str = "bbg", target_tenors: List[int] = TARGET_TENORS):
    assert use in {"test", "bbg"}, f"Unknown use='{use}'"
    assert len(target_tenors) == 8, f"Expected 8 target tenors, got {len(target_tenors)}"

    data = build_all_dataframes(root=ROOT, target_tenors=target_tenors)

    if use == "test":
        df_wide_complete = data["df_wide_test_full"].copy()
    else:
        df_wide_complete = data["df_wide_bbg_full"].copy()

    if df_wide_complete.empty:
        raise ValueError(f"No complete curves found for use='{use}' and tenors={target_tenors}")

    tenors = np.array([float(x) for x in target_tenors], dtype=float)

    df_wide_complete["as_of_date"] = pd.to_datetime(df_wide_complete["as_of_date"])

    # full sample before date cut
    df_wide_all = df_wide_complete[["as_of_date", "ccy"] + list(target_tenors)].copy()
    meta_full = df_wide_all[["as_of_date", "ccy"]].reset_index(drop=True)

    X_full = df_wide_all[list(target_tenors)].to_numpy(dtype=np.float32)

    median_abs = float(np.nanmedian(np.abs(X_full)))
    SCALE_IS_PERCENT = median_abs > 0.5

    if SCALE_IS_PERCENT:
        X_full = X_full / 100.0

    X_tensor_full = torch.from_numpy(X_full)

    # training sample after date cut
    df_wide = df_wide_all[df_wide_all["as_of_date"] >= "2010-01-01"].copy()
    if df_wide.empty:
        raise ValueError("No training rows remain after date filter >= 2010-01-01")

    meta = df_wide[["as_of_date", "ccy"]].reset_index(drop=True)

    X = df_wide[list(target_tenors)].to_numpy(dtype=np.float32)
    if SCALE_IS_PERCENT:
        X = X / 100.0

    X_tensor = torch.from_numpy(X)

    # rename currencies consistently
    meta["ccy"] = meta["ccy"].map(lambda x: currency_rename_map.get(x, x))
    meta_full["ccy"] = meta_full["ccy"].map(lambda x: currency_rename_map.get(x, x))
    df_wide["ccy"] = df_wide["ccy"].map(lambda x: currency_rename_map.get(x, x))
    df_wide_all["ccy"] = df_wide_all["ccy"].map(lambda x: currency_rename_map.get(x, x))

    return meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT


if __name__ == "__main__":

    # =============================
    # PAPER PLOTS A + B (Observed only)
    #   A) swap curves on one date
    #   B) 10Y time series
    # =============================

    # Use your theme palette for consistent currency colors
    ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
    currency_color_map = {ccy: custom_palette[i % len(custom_palette)] for i, ccy in enumerate(ccy_order)}

    USE = "bbg"  # "test" first, then "bbg"

    # Where we save our figures, according to the dataset used to train.
    FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", USE)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use = USE)

    plot_cfg = H.PlotConfig(
        figures_dir=FIGURES_DIR,
        use_tag=USE,
        currency_colors=currency_color_map,
        dpi=300,
    )

    data_cfg = H.DataConfig(
        target_tenors=list(TARGET_TENORS),
        tenor_years=tenors,
        scale_is_percent=SCALE_IS_PERCENT,
    )

    # Build decimals version of observed df (so plots match model scale)
    df_wide_dec = df_wide_all.copy()
    if SCALE_IS_PERCENT:
        for col in TARGET_TENORS:
            df_wide_dec[col] = df_wide_dec[col].astype(float) / 100.0

    print(df_wide_dec["as_of_date"])

    # A) Choose paper date if it exists, otherwise first available
    paper_date = pd.to_datetime("2016-08-30")

    print("Date range:", df_wide_dec["as_of_date"].min(), "to", df_wide_dec["as_of_date"].max())
    print("Paper date exists:", (df_wide_dec["as_of_date"] == paper_date).any())

    print(df_wide_dec[df_wide_dec["as_of_date"] == paper_date])

    closest_idx = (df_wide_dec["as_of_date"] - paper_date).abs().argsort().iloc[0]
    closest_date = df_wide_dec["as_of_date"].iloc[closest_idx]
    print("Closest available date:", closest_date)

    #date_pick_A = paper_date if (df_wide_dec["as_of_date"] == paper_date).any() else df_wide_dec["as_of_date"].iloc[0]

    H.plot_swap_curves_on_date_observed(
        df_wide_obs=df_wide_dec,
        target_tenors=TARGET_TENORS,
        tenors_years=tenors,
        currency_colors=currency_color_map,
        date_pick=closest_date,
        plot_cfg=plot_cfg,
    )

    # B) 10Y time series (or closest tenor to 10)
    TENOR_10Y = 10
    if TENOR_10Y not in TARGET_TENORS:
        TENOR_10Y = min(TARGET_TENORS, key=lambda t: abs(float(t) - 10.0))

    H.plot_swap_timeseries_one_tenor_observed(
        df_wide_obs=df_wide_dec,
        tenor_col=TENOR_10Y,
        currency_colors=currency_color_map,
        plot_cfg=plot_cfg,
        title=f"Observed {TENOR_10Y}Y swap rate over time (all currencies)",
    )
