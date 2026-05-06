import os
import pandas as pd

# Default path resolves relative to this file so the script works on any machine.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EXCEL = os.path.abspath(
    os.path.join(_HERE, "..", "..", "SwapData", "SwapVol.xlsx")
)


def parse_swaption_code(code):
    """
    Parse Bloomberg-style code into:
        (option_maturity, swap_tenor)

    Examples
    --------
    "15"   -> (1, 5)
    "110"  -> (1, 10)
    "510"  -> (5, 10)
    "101"  -> (10, 1)
    "1010" -> (10, 10)
    """
    code_str = str(code).strip() if code is not None else ""

    if not code_str.isdigit():
        return None, None

    n = len(code_str)

    if n == 2:
        option_maturity = int(code_str[0])
        swap_tenor = int(code_str[1])
    elif n == 3:
        if code_str[:2] == "10":
            option_maturity = 10
            swap_tenor = int(code_str[2])
        else:
            option_maturity = int(code_str[0])
            swap_tenor = int(code_str[1:])
    elif n == 4:
        option_maturity = int(code_str[:2])
        swap_tenor = int(code_str[2:])
    else:
        option_maturity = None
        swap_tenor = None

    return option_maturity, swap_tenor


def load_swaption_vol_data(
    excel_path=None,
    currency="EUR",
    sheet_name=0,
    code_row_idx=0,
    data_start_row_idx=5,
    first_date_col_pos=1,
    save_csv=False,
    output_csv_path=None,
    verbose=False,
):
    """
    Load swaption vol data from Bloomberg-style Excel export.

    Parameters
    ----------
    excel_path : str or None
        Path to the Bloomberg-style Excel file.  When *None* (default),
        resolves to ``<repo_root>/SwapData/SwapVol.xlsx`` relative to this
        file so the script runs unchanged on any machine.

    Expected sheet structure
    ------------------------
    Row code_row_idx contains swaption codes in every SECOND column:
        e.g. col 2 -> 11, col 4 -> 15, col 6 -> 110, ...

    Starting from data_start_row_idx, the sheet contains repeating pairs:
        [date_col, vol_col, date_col, vol_col, ...]

    Example:
        col 1 = dates for code in col 2
        col 2 = vols  for code in col 2
        col 3 = dates for code in col 4
        col 4 = vols  for code in col 4
        etc.

    Returns columns:
        currency, as_of_date, option_maturity, swap_tenor, vol
    """
    if excel_path is None:
        excel_path = _DEFAULT_EXCEL

    raw = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )

    if verbose:
        print("First 8 rows of raw Excel:")
        print(raw.head(8))
        print("\nCode row:")
        print(list(raw.iloc[code_row_idx]))

    n_cols = raw.shape[1]
    out_frames = []

    # Loop over date/vol pairs:
    # first pair is (1,2), next (3,4), ..., so vol cols are 2,4,6,...
    for vol_col in range(first_date_col_pos + 1, n_cols, 2):
        date_col = vol_col - 1

        code = raw.iat[code_row_idx, vol_col] if vol_col < n_cols else None
        option_maturity, swap_tenor = parse_swaption_code(code)

        if option_maturity is None or swap_tenor is None:
            if verbose:
                print(f"Skipping vol_col={vol_col}: invalid code={code}")
            continue

        tmp = raw.iloc[data_start_row_idx:, [date_col, vol_col]].copy()
        tmp.columns = ["as_of_date", "vol"]

        tmp["as_of_date"] = pd.to_datetime(tmp["as_of_date"], errors="coerce")
        tmp["vol"] = pd.to_numeric(tmp["vol"], errors="coerce")

        tmp = tmp.dropna(subset=["as_of_date", "vol"]).copy()
        if tmp.empty:
            if verbose:
                print(
                    f"No valid rows for code={code} "
                    f"(expiry={option_maturity}, tenor={swap_tenor})"
                )
            continue

        tmp["currency"] = str(currency).upper()
        tmp["option_maturity"] = int(option_maturity)
        tmp["swap_tenor"] = int(swap_tenor)

        out_frames.append(
            tmp[["currency", "as_of_date", "option_maturity", "swap_tenor", "vol"]]
        )

        if verbose:
            print(
                f"Loaded code={code}: "
                f"expiry={option_maturity}, tenor={swap_tenor}, rows={len(tmp)}"
            )

    if not out_frames:
        result = pd.DataFrame(
            columns=["currency", "as_of_date", "option_maturity", "swap_tenor", "vol"]
        )
    else:
        result = pd.concat(out_frames, ignore_index=True)
        result = result.sort_values(
            ["currency", "as_of_date", "option_maturity", "swap_tenor"]
        ).reset_index(drop=True)

    if save_csv:
        if output_csv_path is None:
            try:
                base_dir = os.path.dirname(__file__)
            except NameError:
                base_dir = os.getcwd()

            output_csv_path = os.path.abspath(
                os.path.join(base_dir, "..", "SwapVol_OIS_formatted.csv")
            )

        result.to_csv(output_csv_path, index=False)
        if verbose:
            print(f"\nSaved formatted data to {output_csv_path}")

    return result


if __name__ == "__main__":
    df_swaption_vol = load_swaption_vol_data(save_csv=True, verbose=True)
    print("\nParsed dataframe:")
    print(df_swaption_vol.head(20))
    print("\nShape:", df_swaption_vol.shape)