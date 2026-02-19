import os
import re
import pandas as pd
from typing import Tuple, Dict, List




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


# ---------------------------------------------------------
# 2) Example usage (no file writing)
# ---------------------------------------------------------
data = build_all_dataframes()

df_long_test = data["df_long_test"]
df_long_bbg  = data["df_long_bbg"]

df_wide_test = data["df_wide_test"]
df_wide_bbg  = data["df_wide_bbg"]

df_wide_test_aligned = data["df_wide_test_aligned"]
df_wide_bbg_aligned  = data["df_wide_bbg_aligned"]

df_wide_test_full = data["df_wide_test_full"]
df_wide_bbg_full  = data["df_wide_bbg_full"]

print("\nRoot used:", data["root_used"])
print("Target tenors:", data["target_tenors"])

print("\ndf_long_test:", df_long_test.shape, "df_wide_test:", df_wide_test.shape,
      "df_wide_test_aligned:", df_wide_test_aligned.shape, "df_wide_test_full:", df_wide_test_full.shape)

print("df_long_bbg :", df_long_bbg.shape,  "df_wide_bbg :", df_wide_bbg.shape,
      "df_wide_bbg_aligned :", df_wide_bbg_aligned.shape, "df_wide_bbg_full:", df_wide_bbg_full.shape)

print("\nTest tenors:", data["tenors_test"])
print("BBG  tenors:", data["tenors_bbg"])
print("Errors test:", len(data["errors_test"]), "Errors bbg:", len(data["errors_bbg"]))
