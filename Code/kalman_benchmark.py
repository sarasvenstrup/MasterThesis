# Code/kalman_dns_ekf_benchmark_compare.py
# EKF DNS on par swaps (nonlinear measurement), with:
# - runs BOTH 2F and 3F
# - currency-specific fixed-leg frequency for swap pricing
# - complex-step Jacobian (stable/accurate)
# - Joseph-form covariance update (PSD-stable)
# - diagonal tenor-dependent measurement covariance R estimated from residuals (two-pass "quasi-EM")
# - per-currency grid search for lambda
# - two-pass update of A (phi) and Q from smoothed states (quasi-EM)
# - prints and saves:
#     * RMSE table for 2F
#     * RMSE table for 3F
#     * Comparison table: RMSE(2F) vs RMSE(3F) + improvements
# - NEW:
#     * residual plots per currency (by tenor, and RMSE over time), saved under each model folder:
#         Figures/bbg/kalman_benchmark/ekf_dns_2f/residuals/
#         Figures/bbg/kalman_benchmark/ekf_dns_3f/residuals/

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
    """
    Plots residuals over time for each tenor.
    resid_bps = (y_true - y_fit) in basis points.
    """
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
    """
    Plots cross-tenor RMSE at each date:
      rmse_t = sqrt(mean_j resid(t,j)^2)
    """
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
# EKF + RTS smoother (dimension-agnostic)
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
        xp = A @ x_prev
        Pp = A @ P_prev @ A.T + Q

        yp = np.asarray(h(xp), dtype=float)
        H = jacobian_cs(h, xp)  # (m,n)

        v = Y[t] - yp
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)

        xf = xp + K @ v

        # Joseph form update (PSD-stable)
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
# Quasi-EM updates for A, Q, R (diagonal)
# -----------------------------
def estimate_AQ_from_smoothed(X: np.ndarray, jitter: float = 1e-12):
    X0 = X[:-1]
    X1 = X[1:]
    denom = (X0 ** 2).sum(axis=0) + jitter
    phi = (X1 * X0).sum(axis=0) / denom
    phi = np.clip(phi, -0.995, 0.995)
    A = np.diag(phi)

    innov = X1 - (X0 * phi)
    q = innov.var(axis=0) + jitter
    Q = np.diag(q)
    return A, Q


def estimate_R_from_residuals(resid: np.ndarray, jitter: float = 1e-12):
    r = resid.var(axis=0) + jitter
    return np.diag(r)


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
):
    m = len(tenors)

    out_dir = os.path.join(out_root, f"ekf_dns_{n_factors}f")
    os.makedirs(out_dir, exist_ok=True)

    # residual plot directory
    resid_dir = os.path.join(out_dir, "residuals")
    os.makedirs(resid_dir, exist_ok=True)

    factor_cols = ["Level", "Slope"] if n_factors == 2 else ["Level", "Slope", "Curvature"]

    rows = []
    for ccy in sorted(df["ccy"].unique()):
        dfi = df[df["ccy"] == ccy].sort_values("as_of_date").copy()
        Y = dfi[TARGET_TENORS].astype(float).to_numpy()
        dates = dfi["as_of_date"].to_numpy()
        freq = CCY_FREQ.get(ccy, 1)

        best = None

        for lam in LAM_GRID:
            # init from OLS on day 0
            H0 = ns_loadings(tenors, lam, n_factors=n_factors)
            x0, *_ = np.linalg.lstsq(H0, Y[0], rcond=None)
            P0 = np.eye(n_factors) * P0_scale

            # conservative init (updated after pass 1)
            if n_factors == 2:
                A = np.diag([0.95, 0.90])
                Q = np.diag([1e-6, 1e-6])
            else:
                A = np.diag([0.95, 0.90, 0.85])
                Q = np.diag([1e-6, 1e-6, 1e-6])

            R = np.eye(m) * 1e-6

            # Pass 1
            Xs1, Ps1, Yfit1 = ekf_filter_smoother(Y, tenors, lam, A, Q, R, x0, P0, freq=freq)

            # quasi-EM update
            A2, Q2 = estimate_AQ_from_smoothed(Xs1)
            R2 = estimate_R_from_residuals(Y - Yfit1)

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

        # record
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

        # -----------------------------
        # NEW: Residual plots (bps)
        # -----------------------------
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

        # save diagonal params
        diagA = np.diag(best["A"]).copy()
        diagQ = np.diag(best["Q"]).copy()
        diagR = np.diag(best["R"]).copy()
        pd.DataFrame({"phi": diagA, "q": diagQ}).to_csv(os.path.join(out_dir, f"ekf_dns_params_AQ_{ccy}.csv"), index=False)
        pd.DataFrame({"tenor": TARGET_TENORS, "r": diagR}).to_csv(os.path.join(out_dir, f"ekf_dns_params_R_{ccy}.csv"), index=False)

    rmse_df = pd.DataFrame(rows).sort_values("rmse_bps", na_position="last")
    avg_rmse = rmse_df["rmse_bps"].mean()
    rmse_df.loc[len(rmse_df)] = {"ccy": "Average", "rmse_bps": avg_rmse, "lambda": np.nan}

    rmse_df.to_csv(os.path.join(out_dir, "ekf_dns_rmse_bps.csv"), index=False)

    print(f"\nEKF DNS RMSE (par swaps, bps) — {n_factors} factors:")
    print(rmse_df)
    print(f"\nAverage EKF DNS RMSE across currencies ({n_factors}F): {avg_rmse:.2f} bps")
    print(f"\nSaved: {os.path.join(out_dir, 'ekf_dns_rmse_bps.csv')}\n")

    return rmse_df, out_dir


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    USE = "bbg"

    CCY_FREQ = {"USD": 2, "GBP": 2, "JPY": 2, "CAD": 2, "AUD": 2, "EUR": 1, "DKK": 1, "SEK": 1, "NOK": 1}
    LAM_GRID = np.linspace(0.05, 2.00, 30)
    P0_scale = 1.0

    meta, X_tensor, tenors_meta, df_wide, SCALE_IS_PERCENT = my_data(use=USE)

    df = df_wide.copy()
    if SCALE_IS_PERCENT:
        for col in TARGET_TENORS:
            df[col] = df[col].astype(float) / 100.0

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["ccy"] = df["ccy"].astype(str)

    tenors = np.array(TARGET_TENORS, dtype=float)

    # Root output folder (BBG Kalman benchmark)
    out_root = os.path.join(os.getcwd(), "Figures", "bbg", "kalman_benchmark")
    os.makedirs(out_root, exist_ok=True)

    # Run 2F and 3F
    rmse_2f, dir_2f = run_model(df, tenors, 2, CCY_FREQ, LAM_GRID, out_root, P0_scale=P0_scale)
    rmse_3f, dir_3f = run_model(df, tenors, 3, CCY_FREQ, LAM_GRID, out_root, P0_scale=P0_scale)

    # -----------------------------
    # Comparison table: RMSE 2F vs RMSE 3F
    # -----------------------------
    df2 = rmse_2f[rmse_2f["ccy"] != "Average"].copy()
    df3 = rmse_3f[rmse_3f["ccy"] != "Average"].copy()

    comp = df2.merge(df3, on="ccy", how="inner", suffixes=("_2f", "_3f"))
    comp["abs_improvement_bps"] = comp["rmse_bps_2f"] - comp["rmse_bps_3f"]
    comp["rel_improvement_pct"] = 100.0 * comp["abs_improvement_bps"] / comp["rmse_bps_2f"]

    # Add average row
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