# Code/kalman_dns_ekf_oos.py
#
# Correct OOS evaluation:
#   1. Estimate lambda, A, Q, R on TRAIN data only (2004-2020)
#   2. Freeze parameters and run FORWARD-ONLY EKF on TEST data (2021-2022)
#
# Supports n_factors in {1, 2, 3, 4}
#
# Produces per factor model:
#   - rmse_summary.csv
#   - is_fitted_vs_actual.png
#   - oos_fitted_vs_actual.png
#   - latent_factors_train.png
#   - latent_factors_oos.png
#
# Produces comparison plots (one subplot per currency, 4 lines: Actual, 2F, 3F, 4F):
#   - comparison_oos_fitted_vs_actual.png
#   - comparison_is_fitted_vs_actual.png
#
# Output folder: Figures/kalman_benchmark_oos/

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Code.load_swapdata import my_data, TARGET_TENORS

# ── config ─────────────────────────────────────────────────────────────────────
TRAIN_START = "2004-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2022-12-31"

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

CCY_FREQ = {
    "USD": 2, "GBP": 2, "JPY": 2, "CAD": 2, "AUD": 2,
    "EUR": 1, "DKK": 1, "SEK": 1, "NOK": 1,
}

LAM_GRID = np.linspace(0.05, 2.00, 30)
P0_scale = 1.0
A_shrink = 0.00

FACTOR_COLS = {
    1: ["Level"],
    2: ["Level", "Slope"],
    3: ["Level", "Slope", "Curvature"],
    4: ["Level", "Slope", "Curvature", "LongCurv"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: find index closest to mid-period (numpy-based)
# ══════════════════════════════════════════════════════════════════════════════

def mid_period_idx(dates):
    ts  = np.array(dates, dtype="datetime64[ns]").astype(np.int64)
    mid = (ts.min() + ts.max()) // 2
    return int(np.abs(ts - mid).argmin())


# ══════════════════════════════════════════════════════════════════════════════
# Nelson-Siegel helpers
# ══════════════════════════════════════════════════════════════════════════════

def ns_loadings(taus, lam, n_factors):
    taus = np.asarray(taus, dtype=float)
    a = (1.0 - np.exp(-lam * taus)) / (lam * taus)
    b = a - np.exp(-lam * taus)
    c = taus * np.exp(-lam * taus)
    if n_factors == 1:
        return np.column_stack([np.ones_like(taus)])
    elif n_factors == 2:
        return np.column_stack([np.ones_like(taus), a])
    elif n_factors == 3:
        return np.column_stack([np.ones_like(taus), a, b])
    elif n_factors == 4:
        return np.column_stack([np.ones_like(taus), a, b, c])
    raise ValueError("n_factors must be 1, 2, 3, or 4")


def swap_curve_from_ns(tenors_years, x, lam, freq=1):
    tenors_years = np.asarray(tenors_years, dtype=float)
    out = np.empty_like(tenors_years, dtype=x.dtype if np.iscomplexobj(x) else float)
    for i, T in enumerate(tenors_years):
        N = max(int(round(T * freq)), 1)
        taus = np.arange(1, N + 1, dtype=float) / float(freq)
        y = ns_loadings(taus, lam, n_factors=x.size) @ x
        P = np.exp(-y * taus)
        delta = 1.0 / float(freq)
        denom = (delta * P).sum()
        denom = denom if abs(denom) > 1e-14 else 1e-14
        out[i] = (1.0 - P[-1]) / denom
    return out


def jacobian_cs(func, x, h=1e-20):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    m, d = f0.size, x.size
    J = np.zeros((m, d))
    for j in range(d):
        xc = x.astype(complex)
        xc[j] += 1j * h
        J[:, j] = np.imag(func(xc)) / h
    return J


# ══════════════════════════════════════════════════════════════════════════════
# EKF + RTS smoother  (used on TRAIN only)
# ══════════════════════════════════════════════════════════════════════════════

def ekf_filter_smoother(Y, tenors, lam, A, Q, R, x0, P0, freq=1):
    T, m = Y.shape
    n = A.shape[0]
    I = np.eye(n)

    def h(x):
        return swap_curve_from_ns(tenors, x, lam, freq=freq)

    x_pred = np.zeros((T, n)); P_pred = np.zeros((T, n, n))
    x_filt = np.zeros((T, n)); P_filt = np.zeros((T, n, n))
    x_prev, P_prev = x0.copy(), P0.copy()

    for t in range(T):
        xp = A @ x_prev
        Pp = A @ P_prev @ A.T + Q
        yp = np.asarray(h(xp), dtype=float)
        H  = jacobian_cs(h, xp)
        v  = Y[t] - yp
        S  = H @ Pp @ H.T + R
        K  = Pp @ H.T @ np.linalg.inv(S)
        xf = xp + K @ v
        KH = K @ H
        Pf = (I - KH) @ Pp @ (I - KH).T + K @ R @ K.T
        x_pred[t], P_pred[t] = xp, Pp
        x_filt[t], P_filt[t] = xf, Pf
        x_prev, P_prev = xf, Pf

    x_smooth = x_filt.copy(); P_smooth = P_filt.copy()
    for t in range(T - 2, -1, -1):
        G = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

    Y_fit = np.vstack([np.asarray(h(x_smooth[t]), dtype=float) for t in range(T)])
    return x_smooth, P_smooth, Y_fit


# ══════════════════════════════════════════════════════════════════════════════
# Forward-only EKF  (used on TEST — no future info)
# ══════════════════════════════════════════════════════════════════════════════

def ekf_forward_only(Y, tenors, lam, A, Q, R, x0, P0, freq=1):
    T, m = Y.shape
    n = A.shape[0]
    I = np.eye(n)

    def h(x):
        return swap_curve_from_ns(tenors, x, lam, freq=freq)

    x_filt = np.zeros((T, n))
    x_prev, P_prev = x0.copy(), P0.copy()

    for t in range(T):
        xp = A @ x_prev
        Pp = A @ P_prev @ A.T + Q
        yp = np.asarray(h(xp), dtype=float)
        H  = jacobian_cs(h, xp)
        v  = Y[t] - yp
        S  = H @ Pp @ H.T + R
        K  = Pp @ H.T @ np.linalg.inv(S)
        xf = xp + K @ v
        KH = K @ H
        Pf = (I - KH) @ Pp @ (I - KH).T + K @ R @ K.T
        x_filt[t] = xf
        x_prev, P_prev = xf, Pf

    Y_fit = np.vstack([np.asarray(h(x_filt[t]), dtype=float) for t in range(T)])
    return x_filt, Y_fit


# ══════════════════════════════════════════════════════════════════════════════
# Quasi-EM parameter estimation (train only)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_A_Q(X, jitter=1e-10, A_shrink=0.0):
    X0, X1 = X[:-1], X[1:]
    n = X.shape[1]
    XtX = X0.T @ X0 + jitter * np.eye(n)
    A = (X1.T @ X0) @ np.linalg.inv(XtX)
    if A_shrink > 0:
        A = (1 - A_shrink) * A + A_shrink * np.diag(np.diag(A))
    eps = X1 - X0 @ A.T
    Q = np.diag(np.var(eps, axis=0)) + jitter * np.eye(n)
    return A, Q


def estimate_R(resid, jitter=1e-10):
    return np.diag(np.var(resid, axis=0)) + jitter * np.eye(resid.shape[1])


def rmse_bps(y_true, y_fit):
    return float(np.sqrt(np.nanmean((y_fit - y_true) ** 2)) * 1e4)


# ══════════════════════════════════════════════════════════════════════════════
# Main model runner
# ══════════════════════════════════════════════════════════════════════════════

def run_model_oos(df_train, df_test, tenors, n_factors, CCY_FREQ, LAM_GRID,
                  out_dir, P0_scale=1.0, A_shrink=0.0):
    os.makedirs(out_dir, exist_ok=True)

    is_rmse_rows  = []
    oos_rmse_rows = []
    fit_store     = {}

    for ccy in ccy_order:
        dfi_tr = df_train[df_train["ccy"] == ccy].sort_values("as_of_date")
        dfi_te = df_test[df_test["ccy"] == ccy].sort_values("as_of_date")

        if len(dfi_tr) == 0:
            continue

        Y_tr     = dfi_tr[TARGET_TENORS].astype(float).to_numpy()
        dates_tr = dfi_tr["as_of_date"].to_numpy()
        freq     = CCY_FREQ.get(ccy, 1)

        Y_te     = dfi_te[TARGET_TENORS].astype(float).to_numpy() if len(dfi_te) > 0 else None
        dates_te = dfi_te["as_of_date"].to_numpy() if len(dfi_te) > 0 else None

        best = None

        for lam in LAM_GRID:
            H0 = ns_loadings(tenors, lam, n_factors)
            x0, *_ = np.linalg.lstsq(H0, Y_tr[0], rcond=None)
            P0 = np.eye(n_factors) * P0_scale
            A  = 0.90 * np.eye(n_factors)
            Q  = 1e-6 * np.eye(n_factors)
            R  = 1e-6 * np.eye(len(tenors))

            Xs1, _, Yfit1 = ekf_filter_smoother(Y_tr, tenors, lam, A, Q, R, x0, P0, freq=freq)
            A2, Q2 = estimate_A_Q(Xs1, A_shrink=A_shrink)
            R2     = estimate_R(Y_tr - Yfit1)
            Xs2, _, Yfit2 = ekf_filter_smoother(Y_tr, tenors, lam, A2, Q2, R2, x0, P0, freq=freq)

            bps = rmse_bps(Y_tr, Yfit2)
            if best is None or bps < best["rmse_is"]:
                best = dict(lam=lam, A=A2, Q=Q2, R=R2,
                            x0=x0, P0=P0,
                            Xs_tr=Xs2, Yfit_tr=Yfit2,
                            rmse_is=bps)

        is_rmse_rows.append({"Currency": ccy, "RMSE_bps": best["rmse_is"]})
        print(f"  {ccy} ({n_factors}F)  IS RMSE = {best['rmse_is']:.2f} bps | lambda* = {best['lam']:.3f}")

        oos_bps = np.nan
        Xs_te, Yfit_te = None, None

        if Y_te is not None and len(Y_te) > 0:
            x_init = best["Xs_tr"][-1]
            P_init = np.eye(n_factors) * P0_scale
            Xs_te, Yfit_te = ekf_forward_only(
                Y_te, tenors, best["lam"],
                best["A"], best["Q"], best["R"],
                x_init, P_init, freq=freq
            )
            oos_bps = rmse_bps(Y_te, Yfit_te)
            print(f"  {ccy} ({n_factors}F) OOS RMSE = {oos_bps:.2f} bps")

        oos_rmse_rows.append({"Currency": ccy, "RMSE_bps": oos_bps})

        fit_store[ccy] = dict(
            dates_tr=dates_tr, Y_tr=Y_tr,  Yfit_tr=best["Yfit_tr"], Xs_tr=best["Xs_tr"],
            dates_te=dates_te, Y_te=Y_te,  Yfit_te=Yfit_te,         Xs_te=Xs_te,
        )

    return is_rmse_rows, oos_rmse_rows, fit_store


# ══════════════════════════════════════════════════════════════════════════════
# Per-model plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_fitted_vs_actual(fit_store, tenors, split, out_path, n_factors):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for ax, ccy in zip(axes, ccy_order):
        if ccy not in fit_store:
            ax.set_visible(False); continue

        d = fit_store[ccy]
        if split == "train":
            dates = d["dates_tr"]; Y = d["Y_tr"]; Y_fit = d["Yfit_tr"]
        else:
            if d["Y_te"] is None:
                ax.set_visible(False); continue
            dates = d["dates_te"]; Y = d["Y_te"]; Y_fit = d["Yfit_te"]

        idx      = mid_period_idx(dates)
        date_str = pd.Timestamp(dates[idx]).strftime("%Y-%m-%d")

        ax.plot(tenors, Y[idx] * 100,     "o-",  label="Actual", linewidth=1.8)
        ax.plot(tenors, Y_fit[idx] * 100, "s--", label="Fitted", linewidth=1.8)
        ax.set_title(f"{ccy}  ({date_str})")
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("Rate (%)")
        ax.legend(fontsize=7)

    split_label = "OOS" if split == "oos" else "In-Sample"
    fig.suptitle(f"{split_label}: Fitted vs Actual Swap Curves — EKF DNS {n_factors}F",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_latent_factors(fit_store, split, out_path, n_factors):
    factor_names = FACTOR_COLS[n_factors]
    fig, axes = plt.subplots(n_factors, 1, figsize=(12, 3 * n_factors), sharex=True)
    if n_factors == 1:
        axes = [axes]

    for ccy in ccy_order:
        if ccy not in fit_store:
            continue
        d      = fit_store[ccy]
        dates  = d["dates_tr"] if split == "train" else d["dates_te"]
        Xs     = d["Xs_tr"]    if split == "train" else d["Xs_te"]
        if dates is None or Xs is None:
            continue
        sort_idx = np.argsort(dates)
        for i, ax in enumerate(axes):
            ax.plot(dates[sort_idx], Xs[sort_idx, i],
                    linewidth=1.2, label=ccy, alpha=0.85)

    for i, ax in enumerate(axes):
        ax.set_ylabel(factor_names[i])
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if i == 0:
            ax.legend(ncol=3, fontsize=7)

    axes[-1].set_xlabel("Date")
    split_label = "OOS" if split == "oos" else "train"
    fig.suptitle(f"Latent factor paths — {split_label} — EKF DNS {n_factors}F",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_rmse_summary(is_rows, oos_rows, out_path):
    is_df  = pd.DataFrame(is_rows).set_index("Currency")
    oos_df = pd.DataFrame(oos_rows).set_index("Currency")
    summary = pd.DataFrame({
        "IS mean (bps)":  is_df["RMSE_bps"],
        "IS std (bps)":   np.nan,
        "OOS mean (bps)": oos_df["RMSE_bps"],
        "OOS std (bps)":  np.nan,
    })
    summary.loc["Average"] = summary.mean()
    summary.to_csv(out_path)
    print(f"\nRMSE summary:\n{summary.to_string()}")
    print(f"Saved: {out_path}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Comparison plot: 3x3 grid, one subplot per currency
# Each subplot: Actual + 2F fitted + 3F fitted + 4F fitted
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(fit_stores, tenors, split, out_path, factor_list=(2, 3, 4)):
    """
    fit_stores: {n_factors: fit_store}
    3x3 grid, one subplot per currency.
    Each subplot shows: Actual swap curve + one fitted line per factor model.
    """
    styles = {
        2: ("s--", "2-factor"),
        3: ("^-.", "3-factor"),
        4: ("D:",  "4-factor"),
    }

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for ax, ccy in zip(axes, ccy_order):
        # Get dates and actual from any factor model (they share the same actual data)
        ref_store = fit_stores[factor_list[0]]
        if ccy not in ref_store:
            ax.set_visible(False); continue

        d_ref = ref_store[ccy]
        if split == "train":
            dates  = d_ref["dates_tr"]
            Y_actual = d_ref["Y_tr"]
        else:
            if d_ref["Y_te"] is None:
                ax.set_visible(False); continue
            dates    = d_ref["dates_te"]
            Y_actual = d_ref["Y_te"]

        idx      = mid_period_idx(dates)
        date_str = pd.Timestamp(dates[idx]).strftime("%Y-%m-%d")

        # Plot actual
        ax.plot(tenors, Y_actual[idx] * 100, "o-", color="black",
                label="Actual", linewidth=1.8)

        # Plot one fitted line per factor model
        for n_factors in factor_list:
            if n_factors not in fit_stores:
                continue
            d = fit_stores[n_factors][ccy]
            Y_fit = d["Yfit_tr"] if split == "train" else d["Yfit_te"]
            if Y_fit is None:
                continue
            marker_style, label = styles[n_factors]
            ax.plot(tenors, Y_fit[idx] * 100, marker_style,
                    label=label, linewidth=1.5)

        ax.set_title(f"{ccy}  ({date_str})")
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("Rate (%)")
        ax.legend(fontsize=7)

    split_label = "OOS" if split == "oos" else "In-Sample"
    fig.suptitle(f"{split_label}: Fitted vs Actual Swap Curves — EKF DNS 2F / 3F / 4F",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    USE = "bbg"

    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)

    df = df_wide.copy()
    if SCALE_IS_PERCENT:
        for col in TARGET_TENORS:
            df[col] = df[col].astype(float) / 100.0

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["ccy"]        = df["ccy"].astype(str)

    df_train = df[(df["as_of_date"] >= TRAIN_START) & (df["as_of_date"] <= TRAIN_END)]
    df_test  = df[(df["as_of_date"] >= TEST_START)  & (df["as_of_date"] <= TEST_END)]

    print(f"Train: {TRAIN_START} – {TRAIN_END}  n={len(df_train)}")
    print(f"Test:  {TEST_START}  – {TEST_END}   n={len(df_test)}")

    tenors = np.array(TARGET_TENORS, dtype=float)

    from Code.config import VARIANT
    root_dir   = os.path.join(os.getcwd(), "Figures", "KalmanBenchmarkResults")
    fit_stores = {}  # {n_factors: fit_store}

    for n_factors in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Running EKF DNS  {n_factors}-factor model")
        print(f"{'='*60}")

        out_dir = os.path.join(root_dir, f"ekf_dns_{n_factors}f")
        os.makedirs(out_dir, exist_ok=True)

        is_rows, oos_rows, fit_store = run_model_oos(
            df_train, df_test, tenors, n_factors,
            CCY_FREQ, LAM_GRID, out_dir,
            P0_scale=P0_scale, A_shrink=A_shrink,
        )
        fit_stores[n_factors] = fit_store

        save_rmse_summary(is_rows, oos_rows,
                          os.path.join(out_dir, "rmse_summary.csv"))

        # save IS latent factors to CSV for correlation analysis
        _lf_rows = []
        for ccy, res in fit_store.items():
            for date, xs in zip(res["dates_tr"], res["Xs_tr"]):
                row = {"as_of_date": date, "ccy": ccy}
                for k in range(n_factors):
                    row[f"z{k+1}"] = xs[k]
                _lf_rows.append(row)
        pd.DataFrame(_lf_rows).to_csv(
            os.path.join(out_dir, "latent_factors_train.csv"), index=False
        )

        plot_fitted_vs_actual(fit_store, tenors, split="train",
                              out_path=os.path.join(out_dir, "is_fitted_vs_actual.png"),
                              n_factors=n_factors)

        plot_fitted_vs_actual(fit_store, tenors, split="oos",
                              out_path=os.path.join(out_dir, "oos_fitted_vs_actual.png"),
                              n_factors=n_factors)

        plot_latent_factors(fit_store, split="train",
                            out_path=os.path.join(out_dir, "latent_factors_train.png"),
                            n_factors=n_factors)

        plot_latent_factors(fit_store, split="oos",
                            out_path=os.path.join(out_dir, "latent_factors_oos.png"),
                            n_factors=n_factors)

        print(f"All outputs saved to: {out_dir}")

    # ── comparison plots (saved in root folder) ───────────────────────────────
    print(f"\n{'='*60}")
    print("Generating comparison plots (2F / 3F / 4F)")
    print(f"{'='*60}")

    plot_comparison(fit_stores, tenors, split="oos",
                    out_path=os.path.join(root_dir, "comparison_oos_fitted_vs_actual.png"))

    plot_comparison(fit_stores, tenors, split="train",
                    out_path=os.path.join(root_dir, "comparison_is_fitted_vs_actual.png"))

    print(f"\nComparison plots saved to: {root_dir}")