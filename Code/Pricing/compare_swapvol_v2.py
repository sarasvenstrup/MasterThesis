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


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def safe_std(x, ddof=1):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= ddof:
        return np.nan
    return float(np.std(x, ddof=ddof))


def safe_min(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.min(x))


def safe_max(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.max(x))


def safe_quantile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


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

    # -----------------------------------------------------------------
    # Add output columns
    # -----------------------------------------------------------------
    extra_cols = [
        "model_vol",
        "model_vol_bp",
        "model_price",

        # time-0 quote ingredients
        "forward_swap",
        "forward_swap_bp",
        "annuity",
        "strike",
        "strike_bp",
        "intrinsic_lower_bound",

        # market/model prices using same time-0 quote
        "market_price_from_market_vol",
        "model_price_from_model_vol",
        "model_reprice_error",

        # MC diagnostics
        "mc_std",
        "mc_stderr",
        "n_valid_paths",
        "n_total_paths",

        # expiry-path diagnostics
        "mean_swap_rate_at_expiry",
        "std_swap_rate_at_expiry",
        "min_swap_rate_at_expiry",
        "max_swap_rate_at_expiry",
        "q05_swap_rate_at_expiry",
        "q50_swap_rate_at_expiry",
        "q95_swap_rate_at_expiry",

        "mean_annuity_at_expiry",
        "std_annuity_at_expiry",
        "min_annuity_at_expiry",
        "max_annuity_at_expiry",

        "mean_payoff_at_expiry",
        "std_payoff_at_expiry",
        "q95_payoff_at_expiry",

        "mean_discount_to_expiry",
        "std_discount_to_expiry",
        "min_discount_to_expiry",
        "max_discount_to_expiry",

        "mean_pv",
        "std_pv",
        "q95_pv",

        # some extra helpful diagnostics
        "mean_swap_minus_strike_at_expiry",
        "mean_positive_part_swap_minus_strike",
        "price_error",
    ]

    for c in extra_cols:
        df_compare[c] = np.nan

    sim_cache = {}
    horizon = n_steps * dt

    for idx, row in df_compare.iterrows():
        market_date = pd.Timestamp(row["market_as_of_date"]).normalize()
        model_date = pd.Timestamp(row["model_as_of_date"]).normalize()
        expiry = int(row["option_maturity"])
        tenor = int(row["swap_tenor"])
        market_vol = float(row["market_vol"])

        # skip expiries beyond simulation horizon
        if expiry > horizon:
            warnings.warn(
                f"Skipping row {idx}: expiry={expiry} exceeds simulation horizon={horizon:.2f}",
                RuntimeWarning,
            )
            continue

        if model_date not in sim_cache:
            print(
                f"\nRunning simulation for model date {model_date.date()} "
                f"(market date {market_date.date()})"
            )

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

        try:
            res = pricing.atm_swaption_mc_price_from_simulation(
                ctx=ctx,
                expiry=expiry,
                tenor=tenor,
                payer=payer,
                accrual=accrual,
                notional=notional,
            )

            quote = res["quote"]

            model_vol = float(res["implied_normal_vol"])
            model_price = float(res["mc_price"])

            forward_swap = float(quote["forward_swap"])
            annuity = float(quote["annuity"])
            strike = float(quote["strike"])
            intrinsic_lb = float(quote["intrinsic_lower_bound"])

            # Prices from Bachelier formula using the SAME time-0 ingredients
            market_price_from_market_vol = pricing.bachelier_price(
                forward=forward_swap,
                strike=strike,
                normal_vol=market_vol,
                expiry=expiry,
                annuity=annuity,
                notional=notional,
                payer=payer,
            )

            model_price_from_model_vol = pricing.bachelier_price(
                forward=forward_swap,
                strike=strike,
                normal_vol=model_vol,
                expiry=expiry,
                annuity=annuity,
                notional=notional,
                payer=payer,
            )

            # Pathwise arrays
            swap_rate_paths = np.asarray(res["swap_rate_paths"], dtype=float)
            annuity_paths = np.asarray(res["annuity_paths"], dtype=float)
            payoff_paths = np.asarray(res["payoff_paths"], dtype=float)
            discount_to_expiry_paths = np.asarray(res["discount_to_expiry_paths"], dtype=float)
            pv_paths = np.asarray(res["pv_paths"], dtype=float)
            valid_mask = np.asarray(res["valid_mask"], dtype=bool)

            swap_rate_valid = swap_rate_paths[valid_mask]
            annuity_valid = annuity_paths[valid_mask]
            payoff_valid = payoff_paths[valid_mask]
            discount_valid = discount_to_expiry_paths[valid_mask]
            pv_valid = pv_paths[valid_mask]

            swap_minus_strike = swap_rate_valid - strike
            positive_part = np.maximum(swap_minus_strike, 0.0) if payer else np.maximum(-swap_minus_strike, 0.0)

            # save scalar diagnostics
            df_compare.at[idx, "model_vol"] = model_vol
            df_compare.at[idx, "model_vol_bp"] = 10000.0 * model_vol
            df_compare.at[idx, "model_price"] = model_price

            df_compare.at[idx, "forward_swap"] = forward_swap
            df_compare.at[idx, "forward_swap_bp"] = 10000.0 * forward_swap
            df_compare.at[idx, "annuity"] = annuity
            df_compare.at[idx, "strike"] = strike
            df_compare.at[idx, "strike_bp"] = 10000.0 * strike
            df_compare.at[idx, "intrinsic_lower_bound"] = intrinsic_lb

            df_compare.at[idx, "market_price_from_market_vol"] = market_price_from_market_vol
            df_compare.at[idx, "model_price_from_model_vol"] = model_price_from_model_vol
            df_compare.at[idx, "model_reprice_error"] = model_price_from_model_vol - model_price

            df_compare.at[idx, "mc_std"] = float(res["mc_std"])
            df_compare.at[idx, "mc_stderr"] = float(res["mc_stderr"])
            df_compare.at[idx, "n_valid_paths"] = int(valid_mask.sum())
            df_compare.at[idx, "n_total_paths"] = int(len(valid_mask))

            df_compare.at[idx, "mean_swap_rate_at_expiry"] = safe_mean(swap_rate_valid)
            df_compare.at[idx, "std_swap_rate_at_expiry"] = safe_std(swap_rate_valid)
            df_compare.at[idx, "min_swap_rate_at_expiry"] = safe_min(swap_rate_valid)
            df_compare.at[idx, "max_swap_rate_at_expiry"] = safe_max(swap_rate_valid)
            df_compare.at[idx, "q05_swap_rate_at_expiry"] = safe_quantile(swap_rate_valid, 0.05)
            df_compare.at[idx, "q50_swap_rate_at_expiry"] = safe_quantile(swap_rate_valid, 0.50)
            df_compare.at[idx, "q95_swap_rate_at_expiry"] = safe_quantile(swap_rate_valid, 0.95)

            df_compare.at[idx, "mean_annuity_at_expiry"] = safe_mean(annuity_valid)
            df_compare.at[idx, "std_annuity_at_expiry"] = safe_std(annuity_valid)
            df_compare.at[idx, "min_annuity_at_expiry"] = safe_min(annuity_valid)
            df_compare.at[idx, "max_annuity_at_expiry"] = safe_max(annuity_valid)

            df_compare.at[idx, "mean_payoff_at_expiry"] = safe_mean(payoff_valid)
            df_compare.at[idx, "std_payoff_at_expiry"] = safe_std(payoff_valid)
            df_compare.at[idx, "q95_payoff_at_expiry"] = safe_quantile(payoff_valid, 0.95)

            df_compare.at[idx, "mean_discount_to_expiry"] = safe_mean(discount_valid)
            df_compare.at[idx, "std_discount_to_expiry"] = safe_std(discount_valid)
            df_compare.at[idx, "min_discount_to_expiry"] = safe_min(discount_valid)
            df_compare.at[idx, "max_discount_to_expiry"] = safe_max(discount_valid)

            df_compare.at[idx, "mean_pv"] = safe_mean(pv_valid)
            df_compare.at[idx, "std_pv"] = safe_std(pv_valid)
            df_compare.at[idx, "q95_pv"] = safe_quantile(pv_valid, 0.95)

            df_compare.at[idx, "mean_swap_minus_strike_at_expiry"] = safe_mean(swap_minus_strike)
            df_compare.at[idx, "mean_positive_part_swap_minus_strike"] = safe_mean(positive_part)

            df_compare.at[idx, "price_error"] = model_price - market_price_from_market_vol

        except Exception as e:
            warnings.warn(
                f"Failed at row {idx} for market_date={market_date.date()}, "
                f"model_date={model_date.date()}, expiry={expiry}, tenor={tenor}: {e}",
                RuntimeWarning,
            )

    # -----------------------------------------------------------------
    # final errors
    # -----------------------------------------------------------------
    df_compare["vol_error"] = df_compare["model_vol"] - df_compare["market_vol"]
    df_compare["vol_error_bp"] = df_compare["model_vol_bp"] - df_compare["market_vol_bp"]
    df_compare["abs_vol_error_bp"] = df_compare["vol_error_bp"].abs()
    df_compare["abs_price_error"] = df_compare["price_error"].abs()

    return df_compare


def main():
    checkpoint_path = (
        r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults"
        r"\dim2_stable\ep3500\checkpoint_dim2_ep3500.pt"
    )

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
        max_rows=100,
    )

    print("\nComparison table:")
    print(
        df_compare[
            [
                "market_as_of_date",
                "option_maturity",
                "swap_tenor",
                "market_vol_bp",
                "model_vol_bp",
                "vol_error_bp",
                "forward_swap_bp",
                "strike_bp",
                "annuity",
                "market_price_from_market_vol",
                "model_price",
                "price_error",
            ]
        ].head(20)
    )

    out_xlsx = os.path.join(CODE_ROOT, "swaption_vol_comparison_diagnostics.xlsx")

    df_save = df_compare.copy()
    df_save["market_as_of_date"] = pd.to_datetime(df_save["market_as_of_date"]).dt.date
    df_save["model_as_of_date"] = pd.to_datetime(df_save["model_as_of_date"]).dt.date

    df_save.to_excel(out_xlsx, index=False, engine="openpyxl")
    print(f"\nSaved comparison to {out_xlsx}")


if __name__ == "__main__":
    main()