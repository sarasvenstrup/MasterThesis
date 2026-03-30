"""
compare_swapvol.py
==================
Compare model-implied ATM normal swaption vols vs Bloomberg EUR market quotes.

Methodology
-----------
For each monthly date in the Bloomberg vol file that also has a EUR swap-rate
curve in the training data:
  1. Encode the EUR curve â†’ z0.
  2. Decode z0 â†’ discount factors â†’ compute the forward-starting swap rate
     for each (expiry, tenor) pair.  The ATM forward rate is used as the strike.
  3. Simulate N_PATHS latent paths (one shared simulation per date, reused
     across all swaption structures on that date).
  4. Price each ATM payer swaption by MC, invert to Bachelier normal vol.
  5. Compare with the Bloomberg quoted vol.

Bloomberg code convention (EUVE codes)
---------------------------------------
  len-2  e.g. "15"   â†’ om=1,  st=5   (1Y option, 5Y swap)
  len-3  starts "10" â†’ om=10, st=code[2]  e.g. "101" â†’ (10, 1)
  len-3  other       â†’ om=code[0], st=code[1:]  e.g. "110" â†’ (1, 10)
  len-4               â†’ om=code[:2], st=code[2:]  e.g. "1010" â†’ (10, 10)

Vol units
---------
  Bloomberg quotes normal vol in basis-points (e.g. 20 bp).
  Divide by 10 000 to convert to decimal for the Bachelier formula.

Output: Figures/Pricing/vol_comparison/
    vol_comparison.csv
    vol_scatter.png
    vol_error_by_type.png
    vol_timeseries.png
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# â”€â”€ Path setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CODE_ROOT    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT  = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))

for _p in [CODE_ROOT, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# â”€â”€ Config + imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from Code import config
config.VARIANT = "stable"

from Code.load_swapdata import my_data

try:
    from simulate_model import (
        simulate_latent_paths,
        decode_from_latent_script,
        resolve_checkpoint_path,
    )
    from price_derivatives import (
        load_model,
        price_swaption,
        implied_normal_vol,
        forward_start_swap_and_annuity_from_discount,
        spot_swap_and_annuity_from_discount,
        discount_factors_from_short_rate_paths,
    )

except ImportError:
    from Code.Pricing.simulate_model import (
        simulate_latent_paths,
        decode_from_latent_script,
        resolve_checkpoint_path,
    )
    from Code.Pricing.price_derivatives import (
        load_model,
        price_swaption,
        implied_normal_vol,
        forward_start_swap_and_annuity_from_discount,
        spot_swap_and_annuity_from_discount,
        discount_factors_from_short_rate_paths,
    )

# â”€â”€ Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LATENT_DIM = 2
EPOCHS     = 2500
USE        = "bbg"
CCY        = "EUR"      # currency to compare against vol data

N_PATHS    = 1000       # MC paths per date  (â†‘ â†’ less noise, â†‘ runtime)
N_STEPS    = 120        # 10-year horizon at monthly steps
DT         = 1 / 12
MAX_DATES  = 30         # cap number of dates for manageable runtime

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCEL_PATH = os.path.join(THESIS_ROOT, "SwapData", "SwapVol.xlsx")
OUT_DIR    = os.path.join(THESIS_ROOT, "Figures", "Pricing", "vol_comparison")


# â”€â”€ Bloomberg code parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _parse_bbg_code(cs: str):
    """
    Parse a Bloomberg swaption vol code into (option_maturity, swap_tenor) years.

    Convention (EUVE):
      "11"   â†’ (1, 1)     len-2: single-digit option, single-digit tenor
      "15"   â†’ (1, 5)
      "110"  â†’ (1, 10)    len-3, NOT starting "10"
      "510"  â†’ (5, 10)
      "101"  â†’ (10, 1)    len-3, starts "10"
      "105"  â†’ (10, 5)
      "1010" â†’ (10, 10)   len-4
    """
    if not cs.isdigit():
        return None, None
    n = len(cs)
    if n == 2:
        return int(cs[0]), int(cs[1])
    elif n == 3:
        if cs[:2] == "10":          # 10Y option
            return 10, int(cs[2])
        else:                        # 1Y or 5Y option, 10Y tenor
            return int(cs[0]), int(cs[1:])
    elif n == 4:                     # e.g. 1010
        return int(cs[:2]), int(cs[2:])
    return None, None


# â”€â”€ Market vol loader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_market_vols(excel_path: str) -> pd.DataFrame:
    """
    Load Bloomberg EUR swaption vol surface from Excel into tidy long-form data.

    Returned columns:
        as_of_date      â€“ datetime (end-of-month)
        option_maturity â€“ int  (years)
        swap_tenor      â€“ int  (years)
        vol             â€“ float (decimal normal vol, e.g. 0.002 for 20 bp)
    """
    raw      = pd.read_excel(excel_path, sheet_name=0, header=None)
    code_row = raw.iloc[0]
    data     = pd.read_excel(excel_path, sheet_name=0, header=4)

    # Column-1 in the raw layout is the date column
    date_col = data.columns[1]   # 'Unnamed: 1'

    # Build column â†’ (option_maturity, swap_tenor) mapping using code row
    col_map = {}
    for idx, col in enumerate(data.columns):
        raw_code = code_row.iloc[idx] if idx < len(code_row) else None
        cs = str(raw_code).strip() if pd.notna(raw_code) else ""
        col_map[col] = _parse_bbg_code(cs)

    # Drop fully-empty columns
    keep = [c for c in data.columns if not data[c].isnull().all()]
    data = data[keep].copy()

    # Melt to long form; the odd-indexed "Unnamed" cols are duplicate date cols
    # â€” they get (None, None) in col_map and are filtered out below
    melted = data.melt(id_vars=[date_col], var_name="swap_col", value_name="vol")

    melted["option_maturity"] = melted["swap_col"].map(
        lambda c: col_map.get(c, (None, None))[0]
    )
    melted["swap_tenor"] = melted["swap_col"].map(
        lambda c: col_map.get(c, (None, None))[1]
    )

    melted = melted.dropna(subset=["option_maturity", "swap_tenor"])
    melted = melted.rename(columns={date_col: "as_of_date"})
    melted["as_of_date"] = pd.to_datetime(melted["as_of_date"], errors="coerce")
    melted = melted.dropna(subset=["as_of_date", "vol"])

    melted["option_maturity"] = melted["option_maturity"].astype(int)
    melted["swap_tenor"]      = melted["swap_tenor"].astype(int)
    melted["vol"]             = pd.to_numeric(melted["vol"], errors="coerce")
    melted = melted.dropna(subset=["vol"])

    # Detect and normalise vol units
    med = melted["vol"].median()
    if med > 1.0:
        print(f"  [units] median={med:.2f} -> basis-points -> dividing by 10,000")
        melted["vol"] /= 10_000.0
    elif med > 0.01:
        print(f"  [units] median={med:.4f} -> percent -> dividing by 100")
        melted["vol"] /= 100.0
    else:
        print(f"  [units] median={med:.6f} -> already decimal")

    return melted[["as_of_date", "option_maturity", "swap_tenor", "vol"]].reset_index(drop=True)


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def nearest_idx(dates: pd.Series, target: pd.Timestamp) -> int:
    """Index of the row in `dates` closest to `target`."""
    return int((dates - target).abs().argmin())


@torch.no_grad()
def atm_forward_annuity(
    model,
    z0: torch.Tensor,
    expiry: int,
    tenor: int,
) -> tuple:
    """
    Decode z0 â†’ P_mkt, then compute ATM (forward swap rate, annuity) for the
    forward-starting swaption with `expiry`-year option on a `tenor`-year swap.
    Returns Python floats.
    """
    P_mkt, *_ = decode_from_latent_script(model, z0)
    fwd, ann  = forward_start_swap_and_annuity_from_discount(P_mkt, expiry, tenor)
    return fwd.mean().item(), ann.mean().item()


def price_date(
    model,
    z0: torch.Tensor,
    pairs: list,
    device: torch.device,
) -> dict:
    """
    Simulate N_PATHS z-paths and price each (expiry, tenor) ATM payer swaption
    using standard Monte Carlo.

    For each path:
      1. Decode z(T) -> P_mkt(T)
      2. Compute spot swap rate S(T) and annuity A(T)
      3. Payoff = max(S(T) - K, 0) * A(T)  where K = F_market = time-0 forward
      4. Discount: PV = D(0,T) * payoff
    5. Average across all paths to get the swaption price

    Returns: {(expiry, tenor): {F_market, mc_price, model_vol}}
    """
    with torch.no_grad():
        z_paths, r_paths, _, _ = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            dt=DT,
            device=device,
            discretization="euler",
        )
        D = discount_factors_from_short_rate_paths(r_paths, DT)  # (n_paths, N_STEPS+1)

    n_steps = z_paths.shape[1] - 1

    # Group tenors by expiry so we decode each expiry only once
    from collections import defaultdict
    expiry_to_tenors = defaultdict(list)
    for expiry, tenor in pairs:
        expiry_to_tenors[expiry].append(tenor)

    results = {}

    for expiry, tenors in sorted(expiry_to_tenors.items()):
        expiry_idx = min(int(round(expiry / DT)), n_steps)
        z_at_exp   = z_paths[:, expiry_idx, :]          # (n_paths, d)
        D_exp      = D[:, expiry_idx]                    # (n_paths,)

        # Decode all n_paths at this expiry in one batched call
        try:
            with torch.no_grad():
                P_mkt_T, *_ = decode_from_latent_script(model, z_at_exp)
        except RuntimeError as exc:
            warnings.warn(f"  decode failed at expiry={expiry}: {exc}", RuntimeWarning)
            continue

        for tenor in tenors:
            # ── Time-0 market ATM strike ──────────────────────────────────────────
            try:
                F_market, ann_0 = atm_forward_annuity(model, z0, expiry, tenor)
            except (ValueError, RuntimeError) as exc:
                warnings.warn(f"  [{expiry}Yx{tenor}Y] time-0 ATM failed: {exc}", RuntimeWarning)
                continue

            # ── MC: compute discounted payer payoff for each path ──────────────────
            try:
                with torch.no_grad():
                    # Spot-start swap rates at time T for each path
                    swap_rate_T, ann_T = spot_swap_and_annuity_from_discount(P_mkt_T, tenor)
                    
                    # Validity check: reject if too many paths have degenerate rates
                    # (negative rates or annuities < 0.5 indicate model extrapolation failure)
                    bad_paths = (swap_rate_T < 0) | (ann_T < 0.5) | ~torch.isfinite(swap_rate_T) | ~torch.isfinite(ann_T)
                    frac_bad = bad_paths.float().mean().item()
                    
                    if frac_bad > 0.1:
                        warnings.warn(
                            f"  [{expiry}Yx{tenor}Y] {frac_bad*100:.1f}% of paths have degenerate rates "
                            f"(negative or invalid) — model extrapolation failing at this maturity — skipping",
                            RuntimeWarning,
                        )
                        continue
                    
                    # Payer payoff: max(S(T) - K, 0) * A(T)
                    payoff = torch.clamp(swap_rate_T - F_market, min=0.0) * ann_T
                    
                    # Discount each path and average
                    pv = D_exp * payoff                              # (n_paths,)
                    pv_valid = pv[torch.isfinite(pv)]
                    
                    if pv_valid.numel() == 0:
                        warnings.warn(
                            f"  [{expiry}Yx{tenor}Y] all payoffs are non-finite — skipping",
                            RuntimeWarning,
                        )
                        continue
                    
                    mc_price = pv_valid.mean().item()
                    
            except (ValueError, RuntimeError) as exc:
                warnings.warn(f"  [{expiry}Yx{tenor}Y] MC pricing failed: {exc}", RuntimeWarning)
                continue

            # ── Invert to Bachelier normal vol using time-0 annuity A(0) ─────────
            # Price formula: V = A(0) * Bachelier(F_market, F_market, sigma, T)
            model_vol = implied_normal_vol(
                market_price=mc_price,
                forward=F_market,
                strike=F_market,
                expiry=float(expiry),
                annuity=ann_0,
                notional=1.0,
                is_call=True,
            )

            results[(expiry, tenor)] = {
                "F_market":  F_market,
                "mc_price":  mc_price,
                "model_vol": model_vol,
            }

    return results


# â”€â”€ Main comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_comparison():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Market data
    print("=" * 60)
    print("Loading EUR market vol data â€¦")
    mkt = load_market_vols(EXCEL_PATH)
    structures = sorted({(int(e), int(t)) for e, t in zip(mkt["option_maturity"], mkt["swap_tenor"])})
    print(f"  {len(mkt)} quotes | {mkt['as_of_date'].nunique()} dates")
    print(f"  Structures: {structures}")
    print(f"  Date range: {mkt['as_of_date'].min().date()} â†’ {mkt['as_of_date'].max().date()}")

    # 2. Model + EUR swap curves
    print("\nLoading model and EUR swap-curve data â€¦")
    ckpt  = resolve_checkpoint_path(CODE_ROOT, USE, LATENT_DIM, EPOCHS)
    model = load_model(ckpt, DEVICE, latent_dim=LATENT_DIM)

    (_, _, meta_full, X_full_raw, _, _, _, _) = my_data(use=USE)

    eur_mask  = meta_full["ccy"] == CCY
    meta_eur  = meta_full[eur_mask].reset_index(drop=True)
    X_eur     = X_full_raw.double()[eur_mask.values]
    dates_eur = pd.to_datetime(meta_eur["as_of_date"]).reset_index(drop=True)

    print(f"  {CCY} curves: {len(dates_eur)} rows, "
          f"{dates_eur.min().date()} â†’ {dates_eur.max().date()}")

    # 3. Overlapping dates between vol file and EUR curve data
    vol_dates = pd.to_datetime(sorted(mkt["as_of_date"].unique()))
    lo, hi    = dates_eur.min(), dates_eur.max()
    vol_dates = vol_dates[(vol_dates >= lo) & (vol_dates <= hi)]

    if len(vol_dates) == 0:
        print("\n[ERROR] No date overlap between vol data and EUR curves.")
        print(f"  Market vol : {mkt['as_of_date'].min().date()} â†’ {mkt['as_of_date'].max().date()}")
        print(f"  EUR curves : {lo.date()} â†’ {hi.date()}")
        return None

    if len(vol_dates) > MAX_DATES:
        step      = max(1, len(vol_dates) // MAX_DATES)
        vol_dates = vol_dates[::step][:MAX_DATES]
        print(f"  Subsampled to {len(vol_dates)} dates (step={step})")
    else:
        print(f"  Processing {len(vol_dates)} dates")

    # 4. Swaption structures that fit within simulation horizon and tau_max
    max_h  = int(N_STEPS * DT)   # 10 years
    tau_max = model.tau_max       # 30 years
    pairs  = sorted({
        (int(e), int(t))
        for e, t in structures
        if int(e) <= max_h and int(e) + int(t) <= tau_max
    })
    if not pairs:
        print("[ERROR] No (expiry, tenor) pairs fit within the model horizon.")
        return None
    print(f"  Structures in scope: {pairs}")

    # 5. Main loop â€“ one simulation per date, all structures priced from it
    records   = []
    n_total   = len(vol_dates)

    for i, date in enumerate(vol_dates):
        print(f"\n[{i + 1}/{n_total}] {date.date()}")

        idx        = nearest_idx(dates_eur, date)
        curve_date = dates_eur.iloc[idx]
        gap_days   = abs((curve_date - date).days)

        if gap_days > 30:
            print(f"  SKIP â€“ nearest {CCY} curve is {gap_days}d away ({curve_date.date()})")
            continue

        print(f"  {CCY} curve: {curve_date.date()}  (gap={gap_days}d, idx={idx})")

        S0 = X_eur[idx: idx + 1].to(DEVICE)
        with torch.no_grad():
            z0 = model.encoder(S0)

        pricing  = price_date(model, z0, pairs, DEVICE)
        day_mkt  = mkt[mkt["as_of_date"] == date]
        n_matched = 0

        for (exp, ten), res in pricing.items():
            mkt_row = day_mkt[
                (day_mkt["option_maturity"] == exp) &
                (day_mkt["swap_tenor"]      == ten)
            ]
            mkt_vol   = float(mkt_row["vol"].values[0]) if len(mkt_row) > 0 else np.nan
            model_vol = res["model_vol"]
            err_bp    = (
                (model_vol - mkt_vol) * 10_000
                if np.isfinite(model_vol) and np.isfinite(mkt_vol)
                else np.nan
            )
            if np.isfinite(mkt_vol):
                n_matched += 1

            records.append({
                "as_of_date":      date,
                "curve_date":      curve_date,
                "option_maturity": exp,
                "swap_tenor":      ten,
                "label":           f"{exp}Yx{ten}Y",
                "F_market_pct":    res["F_market"] * 100,
                "mc_price":        res["mc_price"],
                "model_vol_bp":    model_vol * 10_000 if np.isfinite(model_vol) else np.nan,
                "market_vol_bp":   mkt_vol   * 10_000 if np.isfinite(mkt_vol)   else np.nan,
                "vol_error_bp":    err_bp,
            })

        print(f"  Priced {len(pricing)} structures | market matched: {n_matched}")

    if not records:
        print("\n[ERROR] No records produced.")
        return None

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "vol_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison CSV -> {csv_path}")

    # Export to Excel with formatting
    xlsx_path = os.path.join(OUT_DIR, "vol_comparison.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Comparison")
            
            # Basic formatting
            worksheet = writer.sheets["Comparison"]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Freeze header row
            worksheet.freeze_panes = "A2"
        
        print(f"Saved comparison XLSX -> {xlsx_path}")
    except Exception as e:
        print(f"[WARNING] Could not save Excel file: {e}")
        print(f"  (openpyxl may not be installed. CSV file saved successfully.)")

    # â”€â”€ Summary table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df_v = df.dropna(subset=["model_vol_bp", "market_vol_bp"])
    if df_v.empty:
        print("[WARNING] No rows with both model and market vol â€” skipping plots.")
        return df

    print("\n" + "=" * 60)
    print(f"SUMMARY  ({CCY} ATM swaptions,  model - market,  basis points)")
    print("=" * 60)
    summ = (
        df_v.groupby("label")
        .agg(
            mean_err_bp = ("vol_error_bp", "mean"),
            mae_bp      = ("vol_error_bp", lambda x: np.abs(x).mean()),
            rmse_bp     = ("vol_error_bp", lambda x: np.sqrt((x ** 2).mean())),
            n           = ("vol_error_bp", "count"),
        )
        .reset_index()
        .sort_values("mean_err_bp")
    )
    print(summ.to_string(index=False))

    # â”€â”€ Plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _plot_scatter(df_v, OUT_DIR)
    _plot_error_bars(summ, OUT_DIR)
    _plot_timeseries(df_v, OUT_DIR)

    return df


def _plot_scatter(df_v: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df_v["market_vol_bp"], df_v["model_vol_bp"],
        c=df_v["option_maturity"], cmap="viridis",
        alpha=0.65, s=35, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Option maturity (years)")
    lo = min(df_v["market_vol_bp"].min(), df_v["model_vol_bp"].min()) - 2
    hi = max(df_v["market_vol_bp"].max(), df_v["model_vol_bp"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="45Â° line")
    ax.set_xlabel("Market normal vol (bp)")
    ax.set_ylabel("Model implied normal vol (bp)")
    ax.set_title(f"{CCY} ATM Swaption Vol: Model vs Market")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "vol_scatter.png")
    plt.savefig(p, dpi=200)
    print(f"Saved â†’ {p}")
    plt.show()
    plt.close()


def _plot_error_bars(summ: pd.DataFrame, out_dir: str):
    colors = ["#d73027" if v > 0 else "#4575b4" for v in summ["mean_err_bp"]]
    fig, ax = plt.subplots(figsize=(max(6, len(summ) * 0.9), 4))
    ax.bar(summ["label"], summ["mean_err_bp"], color=colors, zorder=2)
    ax.errorbar(
        summ["label"], summ["mean_err_bp"],
        yerr=summ["rmse_bp"], fmt="none",
        color="black", capsize=3, lw=1, zorder=3,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Swaption (option expiry Ã— swap tenor)")
    ax.set_ylabel("Mean vol error  model âˆ’ market  (bp)")
    ax.set_title(f"{CCY} ATM Swaption: Average Vol Error by Structure\n(error bars = RMSE)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3, zorder=1)
    plt.tight_layout()
    p = os.path.join(out_dir, "vol_error_by_type.png")
    plt.savefig(p, dpi=200)
    print(f"Saved â†’ {p}")
    plt.show()
    plt.close()


def _plot_timeseries(df_v: pd.DataFrame, out_dir: str):
    available = sorted(df_v["label"].unique())
    priority  = ["1Yx5Y", "1Yx10Y", "5Yx5Y", "5Yx10Y", "10Yx5Y", "10Yx10Y"]
    labels    = [l for l in priority if l in available] or available[:4]

    fig, axes = plt.subplots(
        len(labels), 1,
        figsize=(11, 3 * len(labels)),
        sharex=True,
    )
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        sub = df_v[df_v["label"] == label].sort_values("as_of_date")
        ax.plot(sub["as_of_date"], sub["market_vol_bp"],
                color="black",   lw=1.5, label="Market")
        ax.plot(sub["as_of_date"], sub["model_vol_bp"],
                color="#d73027", lw=1.5, ls="--", label="Model")
        ax.set_ylabel("Normal vol (bp)", fontsize=9)
        ax.set_title(f"{label}  ATM  {CCY}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Model vs Market ATM {CCY} Swaption Volatility", fontsize=12, y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, "vol_timeseries.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Saved’ {p}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    run_comparison()

