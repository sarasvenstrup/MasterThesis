"""Swap data loading, preprocessing, and dataset-building utilities."""

import os
import re
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union, Sequence
import numpy as np
try:
    from Code.utils import helpers as H
except ModuleNotFoundError:
    from utils import helpers as H

import torch
import matplotlib as mpl
import seaborn as sns
from cycler import cycler

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # <- parent of Code
ROOT = os.path.join(REPO_ROOT, "SwapData")
print("Repo root:", REPO_ROOT)


# If you want both datasets on the same maturity grid (as in your model/paper)
TARGET_TENORS: List[int] = [1, 2, 3, 5, 10, 15, 20, 30]

# ============================= Set Figure/Plot Theme ===============================

def set_paper_theme():
    """Apply the paper plot theme and return the custom colour palette."""
    # Configure seaborn theme
    sns.set_theme(context="paper", style="darkgrid", font_scale=1.05)

    # Customize tab20b palette
    full_palette = sns.color_palette("tab20b", 20)
    selected_indices = [0, 1, 2, 3, 12, 13, 14, 15, 8]
    palette = [full_palette[i] for i in selected_indices]

    # Global matplotlib defaults
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

    # Set default colour cycle
    mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)

    return palette
def style_axis(ax, title=None, xlabel=None, ylabel=None, legend=True, legend_kwargs=None):
    """Apply consistent axis formatting (title, labels, grid, legend)."""
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
    """Parse maturity in years from a filename containing a pattern like '10year'."""
    m = re.search(r"(\d+)\s*year", filename.lower())
    if not m:
        raise ValueError(f"Cannot infer maturity from filename: {filename}")
    return int(m.group(1))


def read_bloomberg_style_excel(path: str) -> pd.DataFrame:
    """
    Read a Bloomberg-style Excel file and return a tidy swap-rate time series.

    Parameters
    ----------
    path : str
        Path to the Excel file. Metadata rows before the header are skipped
        automatically by searching for a row containing 'Date' and 'PX_LAST'.

    Returns
    -------
    pd.DataFrame
        Columns: as_of_date (datetime64[ns]), swap_rate (float).
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
    Recursively load all Excel swap-rate files under dataset_root.

    Currency is inferred from the immediate parent folder name; maturity in
    years is parsed from the filename (e.g. '10year').

    Parameters
    ----------
    dataset_root : str
        Root directory to walk for .xlsx/.xls files.

    Returns
    -------
    df_long : pd.DataFrame
        Columns: as_of_date, ccy, maturity_years, swap_rate, source_file.
    df_errors : pd.DataFrame
        Columns: source_file, error — files that could not be parsed.
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
    Pivot a long swap-rate DataFrame to a wide curve matrix.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format data with columns as_of_date, ccy, maturity_years, swap_rate.

    Returns
    -------
    pd.DataFrame
        Wide-format with columns as_of_date, ccy, and one column per maturity year.
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
    Restrict a wide DataFrame to the specified tenors, adding NaN columns for any missing.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format curve matrix with maturity year columns.
    tenors : list of int
        Target tenor columns to keep.

    Returns
    -------
    pd.DataFrame
        Columns: as_of_date, ccy, and one column per tenor in tenors.
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
    Load both TestData and Bloombergdata, align to the target tenor grid, and
    return a bundle of long, wide, and complete-curves DataFrames.

    Parameters
    ----------
    root : str
        Root directory containing the TestData and Bloombergdata subfolders.
    target_tenors : list of int
        Tenor grid to align both datasets to.

    Returns
    -------
    dict
        Keys: df_long_test, df_long_bbg, df_wide_test, df_wide_bbg,
        df_wide_test_aligned, df_wide_bbg_aligned, df_wide_test_full,
        df_wide_bbg_full, tenors_test, tenors_bbg, target_tenors,
        errors_test, errors_bbg, root_used.
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

def filter_dataset_by_currency(meta, X_tensor, ccy_filter: Optional[Union[str, Sequence[str]]]):
    """Filter meta and X_tensor to rows matching the given currency or currencies."""
    if ccy_filter is None:
        return meta.reset_index(drop=True), X_tensor

    # Allow both "EUR" and ["EUR", "USD"]
    if isinstance(ccy_filter, str):
        ccy_list = [ccy_filter]
    else:
        ccy_list = list(ccy_filter)

    ccy_list = [str(c).strip().upper() for c in ccy_list if str(c).strip() != ""]

    if len(ccy_list) == 0:
        return meta.reset_index(drop=True), X_tensor

    mask = meta["ccy"].astype(str).str.upper().isin(ccy_list)
    n_keep = int(mask.sum())

    if n_keep == 0:
        available = sorted(meta["ccy"].astype(str).str.upper().unique().tolist())
        raise ValueError(
            f"No rows found for ccy_filter={ccy_list}. "
            f"Available currencies: {available}"
        )

    meta_f = meta.loc[mask].reset_index(drop=True)
    X_tensor_f = X_tensor[mask.to_numpy()]

    print(f"Filtered dataset to currencies {ccy_list}: kept {n_keep} rows")
    return meta_f, X_tensor_f

def my_data(
    use: str = "bbg",
    target_tenors: List[int] = TARGET_TENORS,
    ccy_filter: Optional[Union[str, Sequence[str]]] = None,
):
    """
    Load and preprocess swap-rate data for model training.

    Parameters
    ----------
    use : str, default "bbg"
        Data source: "bbg" for Bloomberg data, "test" for test data.
    target_tenors : list of int
        Swap tenors in years to include.
    ccy_filter : str, list of str, or None
        Currency or currencies to retain. If None, all currencies are included.

    Returns
    -------
    meta : pd.DataFrame
        Training-sample metadata (as_of_date, ccy), from 2010-01-01 onward.
    X_tensor : torch.Tensor, shape (N, d)
        Training swap rates in decimal form.
    meta_full : pd.DataFrame
        Full-sample metadata (all dates).
    X_tensor_full : torch.Tensor, shape (N_full, d)
        Full-sample swap rates in decimal form.
    tenors : np.ndarray
        Tenor grid as float array.
    df_wide : pd.DataFrame
        Training-sample wide DataFrame (from 2010-01-01).
    df_wide_all : pd.DataFrame
        Full-sample wide DataFrame.
    SCALE_IS_PERCENT : bool
        True if the raw data was in percent units and has been divided by 100.
    """
    assert use in {"test", "bbg"}, f"Unknown use='{use}'"
    assert len(target_tenors) >= 1, f"Expected at least 1 target tenor, got {len(target_tenors)}"

    data = build_all_dataframes(root=ROOT, target_tenors=target_tenors)

    if use == "test":
        df_wide_complete = data["df_wide_test_full"].copy()
    else:
        df_wide_complete = data["df_wide_bbg_full"].copy()

    if df_wide_complete.empty:
        raise ValueError(f"No complete curves found for use='{use}' and tenors={target_tenors}")

    tenors = np.array([float(x) for x in target_tenors], dtype=float)

    df_wide_complete["as_of_date"] = pd.to_datetime(df_wide_complete["as_of_date"])

    # rename currencies consistently BEFORE filtering
    df_wide_complete["ccy"] = df_wide_complete["ccy"].map(lambda x: currency_rename_map.get(x, x))

    # full sample before date cut
    df_wide_all = df_wide_complete[["as_of_date", "ccy"] + list(target_tenors)].copy()

    # -------- apply optional currency filter to full sample --------
    if ccy_filter is not None:
        if isinstance(ccy_filter, str):
            ccy_list = [ccy_filter]
        else:
            ccy_list = list(ccy_filter)

        ccy_list = [str(c).strip().upper() for c in ccy_list if str(c).strip() != ""]

        if len(ccy_list) > 0:
            mask_full = df_wide_all["ccy"].astype(str).str.upper().isin(ccy_list)

            if not mask_full.any():
                available = sorted(df_wide_all["ccy"].astype(str).str.upper().unique().tolist())
                raise ValueError(
                    f"No rows found for ccy_filter={ccy_list}. "
                    f"Available currencies: {available}"
                )

            df_wide_all = df_wide_all.loc[mask_full].reset_index(drop=True)
            print(f"Filtered full dataset to currencies {ccy_list}: kept {len(df_wide_all)} rows")
    # ---------------------------------------------------------------

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
