# Code/kalman_dns_ekf_benchmark_compare.py
# EKF DNS on par swaps (nonlinear measurement), with:
# - runs BOTH 2F and 3F
# - currency-specific fixed-leg frequency for swap pricing
# - complex-step Jacobian (stable/accurate)
# - Joseph-form covariance update (PSD-stable)
# - per-currency grid search for lambda
# - two-pass quasi-EM:
#     * estimates FULL A (not diagonal)
#     * estimates DIAGONAL Q
#     * estimates DIAGONAL R
# - saves:
#     * RMSE table for 2F
#     * RMSE table for 3F
#     * comparison table (2F vs 3F)
#     * per-currency factors, fitted swaps, residual plots
#
# Output folders:
#   Figures/bbg/kalman_benchmark/ekf_dns_2f/
#   Figures/bbg/kalman_benchmark/ekf_dns_3f/

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Code.load_swapdata import my_data, TARGET_TENORS


# -----------------------------
# Helper functions for plotting residuals
# -----------------------------
def plot_residuals_time_series(
    dates: np.ndarray,
    resid_bps: np.ndarray,          # (T,m)
    tenors: np.ndarray,             # (m,)
    title: str,
    out_path: str,
    max_legend_cols: int = 4,
):
    """Plots residuals over time for each tenor."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, T in enumerate(tenors):
        ax.plot(dates, resid_bps[:, j], label=f"{int(T)}Y")

    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (bps)")
    ax.grid(True)

    n = len(tenors)
    ncol = min(max_legend_cols, max(1, n // 2))
    ax.legend(ncol=ncol, fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_rmse_over_time(
    dates: np.ndarray,
    resid_bps: np.ndarray,          # (T,m)
    title: str,
    out_path: str,
):
    """Plots cross-tenor RMSE at each date."""
    rmse_t = np.sqrt(np.mean(resid_bps ** 2, axis=1))  # (T,)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, rmse_t)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("RMSE (bps)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Nelson–Siegel loadings / yields (2 or 3 factors)
# -----------------------------
def ns_loadings(taus: np.ndarray, lam: float, n_factors: int) -> np.ndarray:
    taus = np.asarray(taus, dtype=float)
    a = (1.0 - np.exp(-lam * taus)) / (lam * taus)
    if n_factors == 2:
        return np.column_stack([np.ones_like(taus), a])  # (m,2)
    elif n_factors == 3:
        b = a - np.exp(-lam * taus)
        return np.column_stack([np.ones_like(taus), a, b])  # (m,3)
    else:
        raise ValueError("n_factors must be 2 or 3")


def ns_yield(taus: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:
    taus = np.asarray(taus, dtype=float)
    x = np.asarray(x)  # may be complex under complex-step Jacobian
    H = ns_loadings(taus, lam, n_factors=x.size)
    return H @ x


def discounts_from_ns(taus: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:
    y = ns_yield(taus, x, lam)
    return np.exp(-y * taus)


# -----------------------------
# Swap pricing from discounts
# -----------------------------
def par_swap_from_discounts(P: np.ndarray, freq: int = 1, jitter: float = 1e-14) -> float:
    delta = 1.0 / float(freq)
    denom = (delta * P).sum()
    denom = denom if abs(denom) > jitter else (jitter + 0j if np.iscomplexobj(denom) else jitter)
    PT = P[-1]
    return (1.0 - PT) / denom


def swap_curve_from_ns(tenors_years: np.ndarray, x: np.ndarray, lam: float, freq: int = 1) -> np.ndarray:
    tenors_years = np.asarray(tenors_years, dtype=float)
    out = np.empty_like(tenors_years, dtype=x.dtype if np.iscomplexobj(x) else float)

    for i, T in enumerate(tenors_years):
        N = int(round(T * freq))
        N = max(N, 1)
        taus = (np.arange(1, N + 1, dtype=float) / float(freq))
        P = discounts_from_ns(taus, x, lam)
        out[i] = par_swap_from_discounts(P, freq=freq)
    return out


# -----------------------------
# Complex-step Jacobian (stable & very accurate)
# -----------------------------
def jacobian_cs(func, x: np.ndarray, h: float = 1e-20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    m, d = f0.size, x.size
    J = np.zeros((m, d), dtype=float)
    for j in range(d):
        xc = x.astype(complex)
        xc[j] += 1j * h
        fc = func(xc)
        J[:, j] = np.imag(fc) / h
    return J


# -----------------------------
# EKF + RTS smoother
# -----------------------------
def ekf_filter_smoother(
    Y: np.ndarray,              # (T, m) observed swaps
    tenors: np.ndarray,         # (m,)
    lam: float,
    A: np.ndarray,              # (n,n)
    Q: np.ndarray,              # (n,n)
    R: np.ndarray,              # (m,m)
    x0: np.ndarray,             # (n,)
    P0: np.ndarray,             # (n,n)
    freq: int = 1,
):
    T, m = Y.shape
    n = A.shape[0]
    assert A.shape == (n, n) and Q.shape == (n, n) and P0.shape == (n, n)
    assert x0.shape == (n,) and R.shape == (m, m)

    def h(x):
        return swap_curve_from_ns(tenors, x, lam, freq=freq)

    I = np.eye(n)

    x_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))

    x_prev = x0.copy()
    P_prev = P0.copy()

    for t in range(T):
        # ---- Transition (prediction) ----
        xp = A @ x_prev
        Pp = A @ P_prev @ A.T + Q

        # ---- Measurement (update) ----
        yp = np.asarray(h(xp), dtype=float)
        H = jacobian_cs(h, xp)  # (m,n)

        v = Y[t] - yp
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)

        xf = xp + K @ v

        # Joseph form covariance update (PSD-stable)
        KH = K @ H
        Pf = (I - KH) @ Pp @ (I - KH).T + K @ R @ K.T

        x_pred[t], P_pred[t] = xp, Pp
        x_filt[t], P_filt[t] = xf, Pf

        x_prev, P_prev = xf, Pf

    # RTS smoother
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(T - 2, -1, -1):
        Pp_next = P_pred[t + 1]
        G = P_filt[t] @ A.T @ np.linalg.inv(Pp_next)
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - Pp_next) @ G.T

    Y_fit = np.vstack([np.asarray(h(x_smooth[t]), dtype=float) for t in range(T)])
    return x_smooth, P_smooth, Y_fit


def rmse_bps(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    err = y_fit - y_true
    return float(np.sqrt(np.mean(err ** 2)) * 1e4)


# -----------------------------
# Quasi-EM style updates: FULL A, DIAGONAL Q, DIAGONAL R
# -----------------------------
def estimate_A_full_Q_diag_from_smoothed(
    X: np.ndarray,
    jitter: float = 1e-10,
    A_shrink: float = 0.00,     # shrink A toward diagonal if needed (0 = none)
):
    """
    Estimate FULL A by least squares:
        A = (X1^T X0) (X0^T X0)^{-1}
    then innovations:
        eps = X1 - A X0
    but force diagonal Q:
        Q = diag(var(eps))
    """
    X0 = X[:-1]  # (T-1, n)
    X1 = X[1:]   # (T-1, n)
    n = X.shape[1]

    XtX = X0.T @ X0 + jitter * np.eye(n)
    A = (X1.T @ X0) @ np.linalg.inv(XtX)

    if A_shrink > 0:
        A = (1.0 - A_shrink) * A + A_shrink * np.diag(np.diag(A))

    eps = X1 - (X0 @ A.T)  # (T-1, n)

    q = np.var(eps, axis=0)  # (n,)
    Q = np.diag(q) + jitter * np.eye(n)
    return A, Q


def estimate_R_diag_from_residuals(
    resid: np.ndarray,
    jitter: float = 1e-10,
):
    """
    Diagonal measurement covariance:
        R = diag(var(resid))
    """
    m = resid.shape[1]
    r = np.var(resid, axis=0)  # (m,)
    R = np.diag(r) + jitter * np.eye(m)
    return R


# -----------------------------
# Run one model (2F or 3F) and return its RMSE table
# -----------------------------
def run_model(
    df: pd.DataFrame,
    tenors: np.ndarray,
    n_factors: int,
    CCY_FREQ: dict,
    LAM_GRID: np.ndarray,
    out_root: str,
    P0_scale: float = 1.0,
    # kept for API compatibility (not used for diagonal Q/R)
    R_shrink: float = 0.05,
    Q_shrink: float = 0.05,
    A_shrink: float = 0.00,
):
    m = len(tenors)

    out_dir = os.path.join(out_root, f"ekf_dns_{n_factors}f")
    os.makedirs(out_dir, exist_ok=True)

    resid_dir = os.path.join(out_dir, "residuals")
    os.makedirs(resid_dir, exist_ok=True)

    factor_cols = ["Level", "Slope"] if n_factors == 2 else ["Level", "Slope", "Curvature"]

    rows = []
    for ccy in sorted(df["ccy"].unique()):
        dfi = df[df["ccy"] == ccy].sort_values("as_of_date").copy()
        Y = dfi[TARGET_TENORS].astype(float).to_numpy()  # (T,m)
        dates = dfi["as_of_date"].to_numpy()
        freq = CCY_FREQ.get(ccy, 1)

        best = None

        for lam in LAM_GRID:
            # init from OLS on day 0 (standard starting point)
            H0 = ns_loadings(tenors, lam, n_factors=n_factors)
            x0, *_ = np.linalg.lstsq(H0, Y[0], rcond=None)

            P0 = np.eye(n_factors) * P0_scale

            # conservative initial dynamics
            A = 0.90 * np.eye(n_factors)
            Q = 1e-6 * np.eye(n_factors)  # diagonal
            R = 1e-6 * np.eye(m)          # diagonal

            # Pass 1
            Xs1, Ps1, Yfit1 = ekf_filter_smoother(Y, tenors, lam, A, Q, R, x0, P0, freq=freq)

            # quasi-EM update: FULL A, DIAG Q, DIAG R
            A2, Q2 = estimate_A_full_Q_diag_from_smoothed(Xs1, A_shrink=A_shrink)
            R2 = estimate_R_diag_from_residuals(Y - Yfit1)

            # Pass 2
            Xs2, Ps2, Yfit2 = ekf_filter_smoother(Y, tenors, lam, A2, Q2, R2, x0, P0, freq=freq)
            bps = rmse_bps(Y, Yfit2)

            if (best is None) or (bps < best["rmse"]):
                best = {
                    "lam": float(lam),
                    "rmse": float(bps),
                    "A": A2,
                    "Q": Q2,
                    "R": R2,
                    "Xs": Xs2,
                    "Yfit": Yfit2,
                }

        rows.append({"ccy": ccy, "rmse_bps": best["rmse"], "lambda": best["lam"]})
        print(f"{ccy} ({n_factors}F): RMSE = {best['rmse']:.2f} bps | lambda* = {best['lam']:.3f}")

        # save factors
        fac_df = pd.DataFrame(best["Xs"], columns=factor_cols)
        fac_df.insert(0, "as_of_date", dates)
        fac_df.to_csv(os.path.join(out_dir, f"ekf_dns_factors_{ccy}.csv"), index=False)

        # save fitted vs true swaps
        fit_df = pd.DataFrame(best["Yfit"], columns=[f"fit_{t}" for t in TARGET_TENORS])
        true_df = pd.DataFrame(Y, columns=[f"true_{t}" for t in TARGET_TENORS])
        out_df = pd.concat([pd.DataFrame({"as_of_date": dates}), true_df, fit_df], axis=1)
        out_df.to_csv(os.path.join(out_dir, f"ekf_dns_swaps_fit_{ccy}.csv"), index=False)

        # residual plots
        resid_bps = (Y - best["Yfit"]) * 1e4  # (T,m)
        plot_residuals_time_series(
            dates=dates,
            resid_bps=resid_bps,
            tenors=tenors,
            title=f"{ccy} ({n_factors}F) residuals by tenor (bps)",
            out_path=os.path.join(resid_dir, f"residuals_by_tenor_{ccy}.png"),
        )
        plot_rmse_over_time(
            dates=dates,
            resid_bps=resid_bps,
            title=f"{ccy} ({n_factors}F) cross-tenor RMSE over time (bps)",
            out_path=os.path.join(resid_dir, f"rmse_over_time_{ccy}.png"),
        )

        # save params (still saved as matrices; now Q and R are diagonal matrices)
        pd.DataFrame(best["A"]).to_csv(os.path.join(out_dir, f"ekf_dns_params_A_{ccy}.csv"), index=False)
        pd.DataFrame(best["Q"]).to_csv(os.path.join(out_dir, f"ekf_dns_params_Q_{ccy}.csv"), index=False)
        pd.DataFrame(best["R"], index=TARGET_TENORS, columns=TARGET_TENORS).to_csv(
            os.path.join(out_dir, f"ekf_dns_params_R_{ccy}.csv")
        )

    rmse_df = pd.DataFrame(rows).sort_values("rmse_bps", na_position="last")
    avg_rmse = rmse_df["rmse_bps"].mean()
    rmse_df.loc[len(rmse_df)] = {"ccy": "Average", "rmse_bps": avg_rmse, "lambda": np.nan}

    rmse_path = os.path.join(out_dir, "ekf_dns_rmse_bps.csv")
    rmse_df.to_csv(rmse_path, index=False)

    print(f"\nEKF DNS RMSE (par swaps, bps) — {n_factors} factors:")
    print(rmse_df)
    print(f"\nAverage EKF DNS RMSE across currencies ({n_factors}F): {avg_rmse:.2f} bps")
    print(f"\nSaved: {rmse_path}\n")

    return rmse_df, out_dir


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    USE = "bbg"

    CCY_FREQ = {
        "USD": 2, "GBP": 2, "JPY": 2, "CAD": 2, "AUD": 2,
        "EUR": 1, "DKK": 1, "SEK": 1, "NOK": 1
    }

    # Lambda grid
    LAM_GRID = np.linspace(0.05, 2.00, 30)

    # Stability knobs
    # (R_shrink/Q_shrink are not used anymore because Q/R are forced diagonal)
    P0_scale = 1.0
    R_shrink = 0.05
    Q_shrink = 0.05
    A_shrink = 0.00

    meta, X_tensor, tenors_meta, df_wide, SCALE_IS_PERCENT = my_data(use=USE)

    df = df_wide.copy()
    if SCALE_IS_PERCENT:
        for col in TARGET_TENORS:
            df[col] = df[col].astype(float) / 100.0

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["ccy"] = df["ccy"].astype(str)

    tenors = np.array(TARGET_TENORS, dtype=float)

    out_root = os.path.join(os.getcwd(), "Figures", "bbg", "kalman_benchmark")
    os.makedirs(out_root, exist_ok=True)

    # Run 2F and 3F
    rmse_2f, dir_2f = run_model(
        df, tenors, 2, CCY_FREQ, LAM_GRID, out_root,
        P0_scale=P0_scale, R_shrink=R_shrink, Q_shrink=Q_shrink, A_shrink=A_shrink
    )
    rmse_3f, dir_3f = run_model(
        df, tenors, 3, CCY_FREQ, LAM_GRID, out_root,
        P0_scale=P0_scale, R_shrink=R_shrink, Q_shrink=Q_shrink, A_shrink=A_shrink
    )

    # -----------------------------
    # Comparison table: RMSE 2F vs RMSE 3F
    # -----------------------------
    df2 = rmse_2f[rmse_2f["ccy"] != "Average"].copy()
    df3 = rmse_3f[rmse_3f["ccy"] != "Average"].copy()

    comp = df2.merge(df3, on="ccy", how="inner", suffixes=("_2f", "_3f"))
    comp["abs_improvement_bps"] = comp["rmse_bps_2f"] - comp["rmse_bps_3f"]
    comp["rel_improvement_pct"] = 100.0 * comp["abs_improvement_bps"] / comp["rmse_bps_2f"]

    avg_row = {
        "ccy": "Average",
        "rmse_bps_2f": df2["rmse_bps"].mean(),
        "lambda_2f": np.nan,
        "rmse_bps_3f": df3["rmse_bps"].mean(),
        "lambda_3f": np.nan,
    }
    avg_row["abs_improvement_bps"] = avg_row["rmse_bps_2f"] - avg_row["rmse_bps_3f"]
    avg_row["rel_improvement_pct"] = 100.0 * avg_row["abs_improvement_bps"] / avg_row["rmse_bps_2f"]

    comp = comp.sort_values("rmse_bps_3f").reset_index(drop=True)
    comp = pd.concat([comp, pd.DataFrame([avg_row])], ignore_index=True)

    print("\nRMSE comparison table: 2F vs 3F")
    print(comp[[
        "ccy",
        "rmse_bps_2f", "lambda_2f",
        "rmse_bps_3f", "lambda_3f",
        "abs_improvement_bps", "rel_improvement_pct"
    ]])

    comp_path = os.path.join(out_root, "rmse_comparison_2f_vs_3f.csv")
    comp.to_csv(comp_path, index=False)
    print(f"\nSaved RMSE comparison table: {comp_path}")