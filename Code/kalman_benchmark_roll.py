# Code/kalman_benchmark_roll.py
#
# Rolling window OOS evaluation for EKF DNS benchmark.
# Mirrors OutOfSampleRoll.py setup exactly:
#   TRAIN_YEARS = 5, TEST_MONTHS = 6, STEP_MONTHS = 6
#
# For each rolling window and each n_factors in {2, 3, 4}:
#   1. Estimate EKF DNS parameters on training window  (timed)
#   2. Run forward-only EKF on test window             (timed)
#   3. Record RMSE per currency + timing
#
# Output:
#   Figures/kalman_benchmark_oos/ekf_dns_rolling/
#     oos_rolling_ekf_{n}f_train5Y_test6M_step6M.csv   (one per n_factors)
#     manifest.json

from __future__ import annotations

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REPO_ROOT = os.path.dirname(REPO_ROOT)
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data, TARGET_TENORS
from Code.config import VARIANT
from Code.kalman_benchmark import (
    ns_loadings, ekf_filter_smoother, ekf_forward_only,
    estimate_A_Q, estimate_R, rmse_bps, swap_curve_from_ns,
    CCY_FREQ, LAM_GRID, P0_scale, A_shrink,
)

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_YEARS    = 5
TEST_MONTHS    = 6
STEP_MONTHS    = 6
N_FACTORS_LIST = [2, 3, 4]
CCY_ORDER      = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

OUT_ROOT = os.path.join(REPO_ROOT, "Figures", "KalmanBenchmarkResults", "ekf_dns_rolling")
os.makedirs(OUT_ROOT, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = \
    my_data(use="bbg")

df = df_wide.copy()
if SCALE_IS_PERCENT:
    for col in TARGET_TENORS:
        df[col] = df[col].astype(float) / 100.0

df["as_of_date"] = pd.to_datetime(df["as_of_date"])
df["ccy"]        = df["ccy"].astype(str)
tenors_arr       = np.array(TARGET_TENORS, dtype=float)

# ── rolling window schedule ───────────────────────────────────────────────────
date_min = max(df["as_of_date"].min(), pd.Timestamp("2010-01-01"))
date_max = df["as_of_date"].max()

roll_starts = []
d = date_min + pd.DateOffset(years=TRAIN_YEARS)
while d + pd.DateOffset(months=TEST_MONTHS) <= date_max + pd.DateOffset(days=1):
    roll_starts.append(d)
    d = d + pd.DateOffset(months=STEP_MONTHS)

n_windows = len(roll_starts)
print(f"Rolling windows: {n_windows}  "
      f"({roll_starts[0].date()} → {roll_starts[-1].date()})")


# ══════════════════════════════════════════════════════════════════════════════
# Core function: run EKF DNS on one window
# ══════════════════════════════════════════════════════════════════════════════

def run_one_window(df_tr, df_te, n_factors):
    """
    Estimate EKF DNS on df_tr, evaluate on df_te.
    Returns:
        rmse_dict      : {ccy: rmse_bps}
        time_train_sec : seconds for parameter estimation
        time_test_sec  : seconds for forward EKF evaluation
    """
    rmse_dict = {}

    t_train_start = time.time()

    # ── per-currency parameter estimation ────────────────────────────────────
    best_per_ccy = {}
    for ccy in CCY_ORDER:
        dfi_tr = df_tr[df_tr["ccy"] == ccy].sort_values("as_of_date")
        if len(dfi_tr) < 10:
            continue

        Y_tr = dfi_tr[TARGET_TENORS].astype(float).to_numpy()
        freq = CCY_FREQ.get(ccy, 1)
        best = None

        for lam in LAM_GRID:
            H0   = ns_loadings(tenors_arr, lam, n_factors)
            x0, *_ = np.linalg.lstsq(H0, Y_tr[0], rcond=None)
            P0   = np.eye(n_factors) * P0_scale
            A    = 0.90 * np.eye(n_factors)
            Q    = 1e-6 * np.eye(n_factors)
            R    = 1e-6 * np.eye(len(tenors_arr))

            Xs1, _, Yfit1 = ekf_filter_smoother(
                Y_tr, tenors_arr, lam, A, Q, R, x0, P0, freq=freq)
            A2, Q2 = estimate_A_Q(Xs1, A_shrink=A_shrink)
            R2     = estimate_R(Y_tr - Yfit1)
            Xs2, _, Yfit2 = ekf_filter_smoother(
                Y_tr, tenors_arr, lam, A2, Q2, R2, x0, P0, freq=freq)

            bps = rmse_bps(Y_tr, Yfit2)
            if best is None or bps < best["rmse_is"]:
                best = dict(lam=lam, A=A2, Q=Q2, R=R2,
                            x0=x0, P0=P0,
                            Xs_tr=Xs2, rmse_is=bps)

        best_per_ccy[ccy] = best

    time_train_sec = time.time() - t_train_start

    # ── per-currency forward EKF on test window ───────────────────────────────
    t_test_start = time.time()

    for ccy in CCY_ORDER:
        if ccy not in best_per_ccy:
            rmse_dict[ccy] = np.nan
            continue

        dfi_te = df_te[df_te["ccy"] == ccy].sort_values("as_of_date")
        if len(dfi_te) == 0:
            rmse_dict[ccy] = np.nan
            continue

        Y_te   = dfi_te[TARGET_TENORS].astype(float).to_numpy()
        freq   = CCY_FREQ.get(ccy, 1)
        best   = best_per_ccy[ccy]
        x_init = best["Xs_tr"][-1]
        P_init = np.eye(n_factors) * P0_scale

        _, Yfit_te = ekf_forward_only(
            Y_te, tenors_arr, best["lam"],
            best["A"], best["Q"], best["R"],
            x_init, P_init, freq=freq
        )
        rmse_dict[ccy] = rmse_bps(Y_te, Yfit_te)

    time_test_sec = time.time() - t_test_start

    return rmse_dict, time_train_sec, time_test_sec


# ══════════════════════════════════════════════════════════════════════════════
# Main rolling loop
# ══════════════════════════════════════════════════════════════════════════════

manifest = {
    "train_years":  TRAIN_YEARS,
    "test_months":  TEST_MONTHS,
    "step_months":  STEP_MONTHS,
    "n_windows":    n_windows,
    "n_factors_list": N_FACTORS_LIST,
    "window_results": {},
}

for n_factors in N_FACTORS_LIST:
    print(f"\n{'='*60}")
    print(f"EKF DNS  {n_factors}-factor  —  {n_windows} rolling windows")
    print(f"{'='*60}")

    rows = []

    for k, test_start in enumerate(roll_starts):
        train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
        train_end   = test_start - pd.DateOffset(days=1)
        test_end    = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)

        df_tr = df[(df["as_of_date"] >= train_start) & (df["as_of_date"] <= train_end)]
        df_te = df[(df["as_of_date"] >= test_start)  & (df["as_of_date"] <= test_end)]

        n_train = len(df_tr)
        n_test  = len(df_te)

        print(f"\n[{k+1:02d}/{n_windows}]  "
              f"train {train_start.date()}..{train_end.date()} (n={n_train}) | "
              f"test  {test_start.date()}..{test_end.date()}  (n={n_test})")

        if n_train < 50 or n_test == 0:
            warnings.warn(f"  Skipping window {k+1}: insufficient data")
            continue

        rmse_dict, t_train, t_test = run_one_window(df_tr, df_te, n_factors)

        valid_rmse = [v for v in rmse_dict.values() if np.isfinite(v)]
        avg_rmse   = float(np.mean(valid_rmse)) if valid_rmse else np.nan

        print(f"  avg RMSE = {avg_rmse:.2f} bps | "
              f"train {t_train:.1f}s | test {t_test:.1f}s")

        row = {
            "roll_start":      test_start.date().isoformat(),
            "train_start":     train_start.date().isoformat(),
            "train_end":       train_end.date().isoformat(),
            "test_start":      test_start.date().isoformat(),
            "test_end":        test_end.date().isoformat(),
            "n_train":         n_train,
            "n_test":          n_test,
            "time_train_sec":  round(t_train, 2),
            "time_test_sec":   round(t_test,  2),
        }
        for ccy in CCY_ORDER:
            row[ccy] = round(rmse_dict.get(ccy, np.nan), 4)
        row["avg_rmse_bps"] = round(avg_rmse, 4)
        rows.append(row)

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_name = f"oos_rolling_ekf_{n_factors}f_train{TRAIN_YEARS}Y_test{TEST_MONTHS}M_step{STEP_MONTHS}M.csv"
    csv_path = os.path.join(OUT_ROOT, csv_name)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    manifest["window_results"][f"{n_factors}f"] = {
        "n_windows_run": len(rows),
        "csv":           csv_name,
    }

# ── save manifest ─────────────────────────────────────────────────────────────
manifest_path = os.path.join(OUT_ROOT, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest saved: {manifest_path}")
print("\nDone.")
