import os
import sys
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm as _scipy_norm

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
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import (
    run_simulation, atm_swaption_mc_price_from_simulation, quote_swaption_time0,
)
from Code.load_swapdata import my_data as _load_swapdata


def prepare_market_table(
    ccy="EUR",
    max_rows=None,
    vol_in_bp=True,
    split_date=None,
):
    """
    Load and merge market vol data with swap data on exact common dates.

    Parameters
    ----------
    split_date : str or pd.Timestamp, optional
        If provided, rows are labelled 'train' (date <= split_date) or
        'test' (date > split_date) in a new column ``split``.
    """
    # -----------------------------------------------------------------
    # 1) load market vol data
    # -----------------------------------------------------------------
    df_market = load_swaption_vol_data()

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
    df_swap, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = _load_swapdata()

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

    # -----------------------------------------------------------------
    # 4) train / test label
    # -----------------------------------------------------------------
    if split_date is not None:
        split_ts = pd.Timestamp(split_date).normalize()
        df_compare["split"] = np.where(
            df_compare["as_of_date"] <= split_ts, "train", "test"
        )
    else:
        df_compare["split"] = "all"

    df_compare = df_compare.sort_values(
        ["market_as_of_date", "option_maturity", "swap_tenor"]
    ).reset_index(drop=True)

    if max_rows is not None:
        df_compare = df_compare.iloc[:max_rows].copy()

    print("\nAll exact common dates:")
    for d in common_dates:
        print(" ", pd.Timestamp(d).date())

    if split_date is not None:
        n_train = (df_compare["split"] == "train").sum()
        n_test  = (df_compare["split"] == "test").sum()
        print(f"\nTrain rows (≤ {split_date}): {n_train}   Test rows (> {split_date}): {n_test}")

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


def _vega_from_bachelier(forward, strike, expiry, annuity, normal_vol, notional=1.0):
    """Bachelier vega: ∂V/∂σ = N·A·√T·φ(d)."""
    T_sqrt = math.sqrt(max(float(expiry), 1e-12))
    vol_term = max(float(normal_vol) * T_sqrt, 1e-12)
    d = (float(forward) - float(strike)) / vol_term
    return float(notional) * float(annuity) * T_sqrt * _scipy_norm.pdf(d)


def comparison_table(
    checkpoint_path,
    label="run",
    out_dir=None,
    ccy="EUR",
    n_paths=1000,
    n_steps=120,
    dt=1 / 12,
    payer=True,
    accrual=1.0,
    notional=1.0,
    vol_in_bp=True,
    max_rows=None,
    split_date=None,
    verbose=False,
):
    """
    Run ATM swaption pricing comparison: model MC vol vs market Bachelier vol.

    Parameters
    ----------
    label      : str  — appended to the output filename, e.g. "pre_stage2", "post_stage2"
    out_dir    : str  — directory for the output Excel; defaults to CODE_ROOT
    split_date : str  — ISO date string; rows > split_date are marked 'test', others 'train'.
                        The model is always run on all rows, but the summary distinguishes them.
    verbose    : bool — if True, print per-call pricing summaries and the F0 sanity grid
                        (generates a lot of output on long runs; leave False for full dataset).
    """
    if out_dir is None:
        out_dir = CODE_ROOT
    os.makedirs(out_dir, exist_ok=True)

    df_compare = prepare_market_table(
        ccy=ccy,
        vol_in_bp=vol_in_bp,
        max_rows=max_rows,
        split_date=split_date,
    )

    # add empty model columns
    df_compare["model_vol"]         = np.nan
    df_compare["model_vol_bp"]      = np.nan
    df_compare["model_price"]       = np.nan
    df_compare["mc_stderr"]         = np.nan   # MC standard error of the price
    df_compare["mc_se_vol_bp"]      = np.nan   # MC SE translated to vol space (bp)
    df_compare["vol_inversion_fail"]= ""       # failure reason if NaN vol

    sim_cache = {}

    horizon = n_steps * dt

    # Pre-compute the maximum expiry needed per model date so we only simulate
    # as far as necessary (avoids running 10-year paths just for a 1Y option).
    max_expiry_per_date: dict = {}
    for _, row in df_compare.iterrows():
        d = pd.Timestamp(row["model_as_of_date"]).normalize()
        exp = int(row["option_maturity"])
        if exp <= horizon:
            max_expiry_per_date[d] = max(max_expiry_per_date.get(d, 0), exp)

    for idx, row in df_compare.iterrows():
        market_date = pd.Timestamp(row["market_as_of_date"]).normalize()
        model_date  = pd.Timestamp(row["model_as_of_date"]).normalize()
        expiry      = int(row["option_maturity"])
        tenor       = int(row["swap_tenor"])

        # skip expiries beyond simulation horizon
        if expiry > horizon:
            warnings.warn(
                f"Skipping row {idx}: expiry={expiry} exceeds simulation horizon={horizon:.2f}",
                RuntimeWarning,
            )
            continue

        if model_date not in sim_cache:
            # Only simulate as far as the longest expiry needed for this date.
            max_exp = max_expiry_per_date.get(model_date, n_steps * dt)
            steps_needed = max(1, math.ceil(max_exp / dt))
            print(f"\n[{label}] Simulating {model_date.date()} (max_expiry={max_exp}Y, n_steps={steps_needed}) ...")

            sim_cache[model_date] = run_simulation(
                checkpoint_path=checkpoint_path,
                ccy_filter=ccy,
                as_of_date=str(model_date.date()),
                n_paths=n_paths,
                n_steps=steps_needed,
                dt=dt,
                show_plot=False,
            )

            if verbose:
                ctx = sim_cache[model_date]
                print(f"  F0 grid for {model_date.date()}:")
                for exp in [1, 5, 10]:
                    for ten in [1, 5, 10]:
                        try:
                            q = quote_swaption_time0(
                                ctx, expiry=exp, tenor=ten,
                                strike_atm=True, payer=True, accrual=1.0
                            )
                            print(f"    F0({exp}Yx{ten}Y) = {q['forward_swap']*10000:.1f}bp  "
                                  f"A0={q['annuity']:.4f}")
                        except Exception:
                            pass

        ctx = sim_cache[model_date]

        try:
            res = atm_swaption_mc_price_from_simulation(
                ctx=ctx,
                expiry=expiry,
                tenor=tenor,
                payer=payer,
                accrual=accrual,
                notional=notional,
                verbose=verbose,
            )

            model_vol    = float(res["implied_normal_vol"]) if res["implied_normal_vol"] is not None else np.nan
            model_price  = float(res["mc_price"])
            mc_stderr    = float(res["mc_stderr"])
            fail_reason  = res.get("implied_normal_vol_failure") or ""

            df_compare.at[idx, "model_vol"]         = model_vol
            df_compare.at[idx, "model_vol_bp"]      = 10000.0 * model_vol if np.isfinite(model_vol) else np.nan
            df_compare.at[idx, "model_price"]       = model_price
            df_compare.at[idx, "mc_stderr"]         = mc_stderr
            df_compare.at[idx, "vol_inversion_fail"]= str(fail_reason)

            # Translate price-space MC SE into vol-space SE via vega
            if np.isfinite(model_vol) and model_vol > 0:
                try:
                    quote = res["quote"]
                    vega  = _vega_from_bachelier(
                        forward   = quote["forward_swap"],
                        strike    = quote["strike"],
                        expiry    = expiry,
                        annuity   = quote["annuity"],
                        normal_vol= model_vol,
                        notional  = notional,
                    )
                    se_vol_bp = (mc_stderr / max(vega, 1e-16)) * 10_000
                    df_compare.at[idx, "mc_se_vol_bp"] = se_vol_bp
                except Exception:
                    pass

        except Exception as e:
            warnings.warn(
                f"Failed at row {idx} for {market_date.date()}, "
                f"expiry={expiry}, tenor={tenor}: {e}",
                RuntimeWarning,
            )
            df_compare.at[idx, "vol_inversion_fail"] = str(e)

    df_compare["vol_error"]         = df_compare["model_vol"]    - df_compare["market_vol"]
    df_compare["vol_error_bp"]      = df_compare["model_vol_bp"] - df_compare["market_vol_bp"]
    df_compare["abs_vol_error_bp"]  = df_compare["vol_error_bp"].abs()

    # ── Save labelled Excel ──────────────────────────────────────────
    out_xlsx = os.path.join(out_dir, f"swaption_vol_comparison_{label}.xlsx")
    df_save = df_compare.copy()
    df_save["market_as_of_date"] = pd.to_datetime(df_save["market_as_of_date"]).dt.date
    df_save["model_as_of_date"]  = pd.to_datetime(df_save["model_as_of_date"]).dt.date
    df_save.to_excel(out_xlsx, index=False, engine="openpyxl")
    print(f"\n[{label}] Saved → {out_xlsx}")

    # ── Console summary ──────────────────────────────────────────────
    _print_summary(df_compare, label)

    return df_compare


def _print_summary(df, label):
    """Print MAE/RMSE overall and by (expiry × tenor) heatmap."""
    valid = df.dropna(subset=["model_vol_bp", "vol_error_bp"])
    if valid.empty:
        print(f"[{label}] No valid model vols computed.")
        return

    splits = valid["split"].unique().tolist() if "split" in valid.columns else ["all"]

    for sp in splits:
        sub = valid[valid["split"] == sp] if sp != "all" else valid
        if sub.empty:
            continue
        mae  = sub["abs_vol_error_bp"].mean()
        rmse = (sub["vol_error_bp"] ** 2).mean() ** 0.5
        median_se = sub["mc_se_vol_bp"].median() if "mc_se_vol_bp" in sub else np.nan
        tag = f"[{label}|{sp}]" if sp != "all" else f"[{label}]"
        print(f"{tag:30s}  MAE={mae:6.1f} bp   RMSE={rmse:6.1f} bp   "
              f"median MC SE={median_se:.2f} bp   N={len(sub)}")

    # Heatmap of MAE by (expiry × tenor) across all rows
    print(f"\n[{label}] MAE heatmap (bp) — rows=expiry, cols=tenor:")
    pivot_mae = (
        valid.groupby(["option_maturity", "swap_tenor"])["abs_vol_error_bp"]
        .mean()
        .unstack("swap_tenor")
    )
    print(pivot_mae.round(1).to_string())

    # NaN / failure inventory
    failed = df[df["model_vol_bp"].isna() & (df["vol_inversion_fail"].str.len() > 0)]
    if not failed.empty:
        print(f"\n[{label}] Inversion failures ({len(failed)} rows):")
        for _, row in failed.iterrows():
            print(f"  {row.get('market_as_of_date', '?').date() if hasattr(row.get('market_as_of_date',''), 'date') else row.get('market_as_of_date','?')}  "
                  f"exp={row['option_maturity']}Y x ten={row['swap_tenor']}Y  "
                  f"reason: {row['vol_inversion_fail']}")


# =============================================================================
# Flat-vol baseline (single σ estimated from training-period market average)
# =============================================================================

def flat_vol_baseline(df_compare, split_date=None, label="flat_vol"):
    """
    Compute a flat-vol benchmark that ignores the term-structure model.

    If split_date is given, σ_flat is estimated as the mean market vol
    over training rows only, then applied to all rows.  Otherwise the
    grand mean is used.

    Returns df_compare with additional columns: flat_vol_bp, flat_vol_error_bp.
    """
    df = df_compare.copy()

    if split_date is not None and "split" in df.columns:
        train_rows = df[df["split"] == "train"]
    else:
        train_rows = df

    sigma_flat_bp = float(train_rows["market_vol_bp"].mean())
    print(f"\n[{label}] Flat vol estimated from {len(train_rows)} training rows: "
          f"σ_flat = {sigma_flat_bp:.2f} bp")

    df["flat_vol_bp"]       = sigma_flat_bp
    df["flat_vol_error_bp"] = df["flat_vol_bp"] - df["market_vol_bp"]
    df["abs_flat_error_bp"] = df["flat_vol_error_bp"].abs()

    valid = df.dropna(subset=["market_vol_bp"])
    splits = valid["split"].unique().tolist() if "split" in valid.columns else ["all"]
    for sp in splits:
        sub = valid[valid["split"] == sp] if sp != "all" else valid
        if sub.empty:
            continue
        mae  = sub["abs_flat_error_bp"].mean()
        rmse = (sub["flat_vol_error_bp"] ** 2).mean() ** 0.5
        tag  = f"[{label}|{sp}]" if sp != "all" else f"[{label}]"
        print(f"{tag:30s}  MAE={mae:6.1f} bp   RMSE={rmse:6.1f} bp   N={len(sub)}")

    return df


# =============================================================================
# MC convergence diagnostic
# =============================================================================

def mc_convergence_check(
    checkpoint_path,
    as_of_date,
    expiry,
    tenor,
    n_paths_list=(500, 2000, 8000),
    n_steps=120,
    dt=1 / 12,
    ccy="EUR",
    payer=True,
    accrual=1.0,
    notional=1.0,
):
    """
    Price one swaption with several n_paths values and tabulate
    mc_price, mc_stderr, implied_normal_vol_bp.

    Use this to confirm the MC noise floor is small relative to
    model-market errors.
    """
    rows = []
    for n in n_paths_list:
        print(f"\n[mc_convergence] n_paths={n} ...")
        ctx = run_simulation(
            checkpoint_path=checkpoint_path,
            ccy_filter=ccy,
            as_of_date=str(as_of_date),
            n_paths=n,
            n_steps=n_steps,
            dt=dt,
            show_plot=False,
        )
        res = atm_swaption_mc_price_from_simulation(
            ctx=ctx,
            expiry=expiry,
            tenor=tenor,
            payer=payer,
            accrual=accrual,
            notional=notional,
        )
        rows.append({
            "n_paths"       : n,
            "mc_price"      : res["mc_price"],
            "mc_stderr"     : res["mc_stderr"],
            "vol_bp"        : res["implied_normal_vol"] * 10_000
                               if res["implied_normal_vol"] is not None and
                                  np.isfinite(res["implied_normal_vol"])
                               else np.nan,
        })

    df_conv = pd.DataFrame(rows)
    print(f"\nMC convergence for {expiry}Yx{tenor}Y on {as_of_date}:")
    print(df_conv.to_string(index=False))
    return df_conv


if __name__ == "__main__":
    import argparse

    # =========================================================================
    # Command-line arguments
    # =========================================================================
    parser = argparse.ArgumentParser(description="Compare model swaption vols vs market")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--ccy", type=str, default="EUR", help="Currency (default: EUR)")
    parser.add_argument("--n_paths", type=int, default=2000, help="MC paths (default: 2000)")
    parser.add_argument("--n_steps", type=int, default=120, help="Time steps (default: 120)")
    parser.add_argument("--split_date", type=str, default=None, help="Train/test split date (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    CCY = args.ccy
    N_PATHS = args.n_paths
    N_STEPS = args.n_steps
    DT = 1 / 12
    SPLIT_DATE = args.split_date
    VERBOSE = args.verbose

    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    # Extract checkpoint name for output filename
    ckpt_name = os.path.splitext(os.path.basename(CHECKPOINT))[0]
    
    # Output directory
    OUT_DIR = os.path.join(THESIS_ROOT, "Figures", "Pricing")
    os.makedirs(OUT_DIR, exist_ok=True)

    # =========================================================================

    print("=" * 70)
    print("SWAPTION VOL COMPARISON")
    print("=" * 70)
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Out dir   : {OUT_DIR}")
    print(f"  CCY={CCY}  n_paths={N_PATHS}  n_steps={N_STEPS}  split={SPLIT_DATE}")
    print("=" * 70)

    # ── Model pricing ─────────────────────────────────────────────────────────
    print(f"\n--- Pricing with {ckpt_name} ---")
    df_model = comparison_table(
        checkpoint_path = CHECKPOINT,
        label           = ckpt_name,
        out_dir         = OUT_DIR,
        ccy             = CCY,
        n_paths         = N_PATHS,
        n_steps         = N_STEPS,
        dt              = DT,
        payer           = True,
        accrual         = 1.0,
        notional        = 1.0,
        split_date      = SPLIT_DATE,
        verbose         = VERBOSE,
    )

    # ── Flat-vol baseline ─────────────────────────────────────────────────────
    print("\n--- Flat-vol baseline ---")
    df_flat = flat_vol_baseline(df_model, split_date=SPLIT_DATE, label="flat_vol")
    
    # Save flat-vol results
    flat_xlsx = os.path.join(OUT_DIR, f"swaption_vol_flat_{ckpt_name}.xlsx")
    df_flat_save = df_flat.copy()
    df_flat_save["market_as_of_date"] = pd.to_datetime(df_flat_save["market_as_of_date"]).dt.date
    df_flat_save["model_as_of_date"]  = pd.to_datetime(df_flat_save["model_as_of_date"]).dt.date
    df_flat_save.to_excel(flat_xlsx, index=False, engine="openpyxl")
    print(f"[flat_vol] Saved → {flat_xlsx}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"SUMMARY  —  {ckpt_name}")
    print("=" * 70)

    splits = (["train", "test"] if SPLIT_DATE else ["all"])
    print(f"  {'Label':<30}  {'MAE (bp)':>8}  {'RMSE (bp)':>9}  {'N':>6}")
    print("  " + "-" * 58)

    for sp in splits:
        # Model results
        sub_model = (df_model[df_model["split"] == sp] if "split" in df_model.columns and sp != "all"
                     else df_model).dropna(subset=["model_vol_bp", "vol_error_bp"])
        if not sub_model.empty:
            mae  = sub_model["abs_vol_error_bp"].mean()
            rmse = (sub_model["vol_error_bp"] ** 2).mean() ** 0.5
            tag  = f"[{ckpt_name}|{sp}]" if sp != "all" else f"[{ckpt_name}]"
            print(f"  {tag:<30}  {mae:8.1f}  {rmse:9.1f}  {len(sub_model):6d}")
        
        # Flat-vol results
        sub_flat = (df_flat[df_flat["split"] == sp] if "split" in df_flat.columns and sp != "all"
                    else df_flat).dropna(subset=["market_vol_bp"])
        if not sub_flat.empty:
            mae_flat  = sub_flat["abs_flat_error_bp"].mean()
            rmse_flat = (sub_flat["flat_vol_error_bp"] ** 2).mean() ** 0.5
            tag  = f"[flat_vol|{sp}]" if sp != "all" else f"[flat_vol]"
            print(f"  {tag:<30}  {mae_flat:8.1f}  {rmse_flat:9.1f}  {len(sub_flat):6d}")

    print(f"\n  Results saved to: {OUT_DIR}/")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

