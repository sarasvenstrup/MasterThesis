import os
import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
try:
    CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CODE_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
import load_swapvol_ois
import pricing
import load_swapdata


def prepare_market_table(
    ccy="EUR",
    max_rows=None,
    vol_in_bp=True,
):
    # -----------------------------------------------------------------
    # 1) load market vol data
    # -----------------------------------------------------------------
    df_market = load_swapvol_ois.load_swaption_vol_data()

    if df_market.empty:
        raise ValueError("No market vol data found.")

    df_market = df_market.copy()
    df_market["currency"] = df_market["currency"].astype(str).str.upper().str.strip()
    df_market = df_market[df_market["currency"] == str(ccy).upper()].copy()

    df_market["as_of_date"] = pd.to_datetime(df_market["as_of_date"], errors="coerce").dt.normalize()
    df_market["option_maturity"] = pd.to_numeric(df_market["option_maturity"], errors="coerce")
    df_market["swap_tenor"] = pd.to_numeric(df_market["swap_tenor"], errors="coerce")
    df_market["vol"] = pd.to_numeric(df_market["vol"], errors="coerce")

    df_market = df_market.dropna(
        subset=["as_of_date", "option_maturity", "swap_tenor", "vol"]
    ).copy()

    df_market["option_maturity"] = df_market["option_maturity"].astype(int)
    df_market["swap_tenor"] = df_market["swap_tenor"].astype(int)

    # average duplicates if any
    df_market = (
        df_market.groupby(
            ["currency", "as_of_date", "option_maturity", "swap_tenor"],
            as_index=False
        )["vol"]
        .mean()
    )

    if vol_in_bp:
        df_market["market_vol_bp"] = df_market["vol"]
        df_market["market_vol"] = df_market["market_vol_bp"] / 10000.0
    else:
        df_market["market_vol"] = df_market["vol"]
        df_market["market_vol_bp"] = 10000.0 * df_market["market_vol"]

    # -----------------------------------------------------------------
    # 2) load swap data
    # -----------------------------------------------------------------
    df_swap, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = load_swapdata.my_data()

    if df_swap.empty:
        raise ValueError("No swap data found.")

    df_swap = df_swap.copy()

    if "as_of_date" in df_swap.columns:
        swap_date_col = "as_of_date"
    elif "Date" in df_swap.columns:
        swap_date_col = "Date"
    elif "date" in df_swap.columns:
        swap_date_col = "date"
    else:
        raise KeyError("Could not find a date column in swap data.")

    df_swap[swap_date_col] = pd.to_datetime(df_swap[swap_date_col], errors="coerce").dt.normalize()
    df_swap = df_swap.dropna(subset=[swap_date_col]).copy()

    # -----------------------------------------------------------------
    # 3) exact common dates only
    # -----------------------------------------------------------------
    common_dates = sorted(set(df_market["as_of_date"]).intersection(set(df_swap[swap_date_col])))

    if len(common_dates) == 0:
        raise ValueError("No exact common dates found between market vol data and swap data.")

    df_compare = df_market[df_market["as_of_date"].isin(common_dates)].copy()
    df_compare["market_as_of_date"] = df_compare["as_of_date"]
    df_compare["model_as_of_date"] = df_compare["as_of_date"]
    df_compare["date_diff_days"] = 0

    df_compare = df_compare.sort_values(
        ["market_as_of_date", "option_maturity", "swap_tenor"]
    ).reset_index(drop=True)

    if max_rows is not None:
        df_compare = df_compare.iloc[:max_rows].copy()

    print("\nAll exact common dates:")
    for d in common_dates:
        print(" ", pd.Timestamp(d).date())

    print("\nRows after exact-date matching:")
    print(
        df_compare[
            [
                "market_as_of_date",
                "model_as_of_date",
                "date_diff_days",
                "option_maturity",
                "swap_tenor",
                "market_vol_bp",
            ]
        ].head(20)
    )

    return df_compare

def comparison_table(
    checkpoint_path,
    ccy="EUR",
    n_paths=1000,
    n_steps=120,
    dt=1 / 12,
    payer=True,
    accrual=1.0,
    notional=1.0,
    vol_in_bp=True,
    max_rows=None,
):
    df_compare = prepare_market_table(
        ccy=ccy,
        vol_in_bp=vol_in_bp,
        max_rows=max_rows
    )

    # add empty model columns
    df_compare["model_vol"] = np.nan
    df_compare["model_vol_bp"] = np.nan
    df_compare["model_price"] = np.nan

    sim_cache = {}

    horizon = n_steps * dt

    for idx, row in df_compare.iterrows():
        market_date = pd.Timestamp(row["market_as_of_date"]).normalize()
        model_date = pd.Timestamp(row["model_as_of_date"]).normalize()
        expiry = int(row["option_maturity"])
        tenor = int(row["swap_tenor"])

        # skip expiries beyond simulation horizon
        if expiry > horizon:
            warnings.warn(
                f"Skipping row {idx}: expiry={expiry} exceeds simulation horizon={horizon:.2f}",
                RuntimeWarning,
            )
            continue

        if model_date not in sim_cache:
            print(f"\nRunning simulation for model date {model_date.date()} "
                  f"(market date {market_date.date()})")

            sim_cache[model_date] = pricing.run_simulation(
                checkpoint_path=checkpoint_path,
                ccy_filter=ccy,
                as_of_date=str(model_date.date()),
                n_paths=n_paths,
                n_steps=n_steps,
                dt=dt,
                show_plot=False,
            )

        ctx = sim_cache[model_date]

        # Sanity check: print F0 for each expiry/tenor before pricing
        from pricing import quote_swaption_time0
        for exp in [1, 5, 10]:
            for ten in [1, 5, 10]:
                q = quote_swaption_time0(ctx, expiry=exp, tenor=ten,
                                         strike_atm=True, payer=True, accrual=1.0)
                print(f"  F0({exp}Y×{ten}Y) = {q['forward_swap'] * 10000:.1f} bp  "
                      f"A0 = {q['annuity']:.4f}")

        try:
            res = pricing.atm_swaption_mc_price_from_simulation(
                ctx=ctx,
                expiry=expiry,
                tenor=tenor,
                payer=payer,
                accrual=accrual,
                notional=notional,
            )

            model_vol = float(res["implied_normal_vol"])
            model_price = float(res["mc_price"])

            df_compare.at[idx, "model_vol"] = model_vol
            df_compare.at[idx, "model_vol_bp"] = 10000.0 * model_vol
            df_compare.at[idx, "model_price"] = model_price

        except Exception as e:
            warnings.warn(
                f"Failed at row {idx} for market_date={market_date.date()}, "
                f"model_date={model_date.date()}, expiry={expiry}, tenor={tenor}: {e}",
                RuntimeWarning,
            )

    df_compare["vol_error"] = df_compare["model_vol"] - df_compare["market_vol"]
    df_compare["vol_error_bp"] = df_compare["model_vol_bp"] - df_compare["market_vol_bp"]
    df_compare["abs_vol_error_bp"] = df_compare["vol_error_bp"].abs()

    return df_compare


def main():
    checkpoint_path = r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults\dim2_stable\pricing_simple_v2\final_checkpoint.pt"

    df_compare = comparison_table(
        checkpoint_path=checkpoint_path,
        ccy="EUR",
        n_paths=2000,
        n_steps=120,
        dt=1 / 12,
        payer=True,
        accrual=1.0,
        notional=1.0,
        vol_in_bp=True,
    )

    print("\nComparison table:")
    print(
        df_compare[
            [
                "market_as_of_date",
                "model_as_of_date",
                "date_diff_days",
                "option_maturity",
                "swap_tenor",
                "market_vol_bp",
                "model_vol_bp",
                "vol_error_bp",
            ]
        ].head(20)
    )

    out_xlsx = os.path.join(CODE_ROOT, "swaption_vol_comparison.xlsx")

    df_save = df_compare.copy()
    df_save["market_as_of_date"] = pd.to_datetime(df_save["market_as_of_date"]).dt.date
    df_save["model_as_of_date"] = pd.to_datetime(df_save["model_as_of_date"]).dt.date

    df_save.to_excel(out_xlsx, index=False, engine="openpyxl")
    print(f"\nSaved comparison to {out_xlsx}")


if __name__ == "__main__":
    main()