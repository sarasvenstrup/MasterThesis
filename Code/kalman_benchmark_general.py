# Code/kalman_benchmark_general.py
# Generic EKF benchmark on PAR SWAP RATES with a pluggable curve model (NOT Nelson–Siegel).
#
# This version:
# - Runs BOTH 2F and 3F
# - Uses a GENERAL exponential-basis yield curve:
#       2F: y(τ)=x1 + x2 exp(-k τ)
#       3F: y(τ)=x1 + x2 exp(-k τ) + x3 exp(-2k τ)
#   then P(τ)=exp(-τ y(τ)) and swaps are priced from discounts.
# - Performs a per-currency GRID SEARCH over k (like lambda in your DNS EKF)
# - Saves:
#     * RMSE table for 2F (best k per ccy)
#     * RMSE table for 3F (best k per ccy)
#     * Comparison table: RMSE(2F) vs RMSE(3F) + improvements
#     * per-currency fitted swaps and smoothed factors (for best k)
#     * residual plots per currency (by tenor + RMSE over time)
#
# Output folders:
#   Figures/bbg/kalman_benchmark_general/general_2f/
#   Figures/bbg/kalman_benchmark_general/general_3f/
#   Figures/bbg/kalman_benchmark_general/rmse_comparison_2f_vs_3f.csv

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Code.load_swapdata import my_data, TARGET_TENORS


# -----------------------------
# Plot helpers (residuals)
# -----------------------------
def plot_residuals_time_series(
    dates: np.ndarray,
    resid_bps: np.ndarray,          # (T,m)
    tenors: np.ndarray,             # (m,)
    title: str,
    out_path: str,
    max_legend_cols: int = 4,
):
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, T_ in enumerate(tenors):
        ax.plot(dates, resid_bps[:, j], label=f"{int(T_)}Y")
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
# Swap pricing from discounts
# -----------------------------
def par_swap_from_discounts(P: np.ndarray, freq: int = 1, jitter: float = 1e-14) -> float:
    delta = 1.0 / float(freq)
    denom = (delta * P).sum()
    if abs(denom) < jitter:
        denom = jitter + (0j if np.iscomplexobj(denom) else 0.0)
    PT = P[-1]
    return (1.0 - PT) / denom


def swap_curve_from_discounts_fn(
    tenors_years: np.ndarray,
    x: np.ndarray,
    discounts_fn,
    freq: int = 1,
) -> np.ndarray:
    tenors_years = np.asarray(tenors_years, dtype=float)
    out = np.empty_like(tenors_years, dtype=x.dtype if np.iscomplexobj(x) else float)

    for i, T in enumerate(tenors_years):
        N = max(int(round(T * freq)), 1)
        taus = (np.arange(1, N + 1, dtype=float) / float(freq))
        P = discounts_fn(taus, x)  # (N,)
        out[i] = par_swap_from_discounts(P, freq=freq)
    return out


# -----------------------------
# Complex-step Jacobian
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
# Generic EKF + RTS smoother
# -----------------------------
def ekf_filter_smoother_generic(
    Y: np.ndarray,              # (T,m) observed swaps
    tenors: np.ndarray,         # (m,)
    A: np.ndarray,              # (n,n)
    Q: np.ndarray,              # (n,n)
    R: np.ndarray,              # (m,m)
    x0: np.ndarray,             # (n,)
    P0: np.ndarray,             # (n,n)
    discounts_fn,               # discounts_fn(taus, x) -> P(taus)
    freq: int = 1,
):
    T, m = Y.shape
    n = A.shape[0]
    assert A.shape == (n, n) and Q.shape == (n, n) and P0.shape == (n, n)
    assert x0.shape == (n,) and R.shape == (m, m)

    I = np.eye(n)

    def h_meas(x):
        return swap_curve_from_discounts_fn(tenors, x, discounts_fn, freq=freq)

    x_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))

    x_prev = x0.copy()
    P_prev = P0.copy()

    for t in range(T):
        xp = A @ x_prev
        Pp = A @ P_prev @ A.T + Q

        yp = np.asarray(h_meas(xp), dtype=float)
        H = jacobian_cs(h_meas, xp)  # (m,n)

        v = Y[t] - yp
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)

        xf = xp + K @ v

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

    Y_fit = np.vstack([np.asarray(h_meas(x_smooth[t]), dtype=float) for t in range(T)])
    return x_smooth, P_smooth, Y_fit


def rmse_bps(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    err = y_fit - y_true
    return float(np.sqrt(np.mean(err ** 2)) * 1e4)


# ============================================================
# General exponential-basis curve model (2F or 3F), hyperparam k
# ============================================================
def make_general_exponential_discounts(n_factors: int, k: float):
    """
    2F:
        y(τ)= x1 + x2 exp(-k τ)
    3F:
        y(τ)= x1 + x2 exp(-k τ) + x3 exp(-2k τ)

    P(τ) = exp( -τ y(τ) )
    """
    if n_factors not in (2, 3):
        raise ValueError("n_factors must be 2 or 3")

    def discounts_fn(taus: np.ndarray, x: np.ndarray) -> np.ndarray:
        taus = np.asarray(taus, dtype=float)
        taus_ = taus.astype(complex) if np.iscomplexobj(x) else taus

        e1 = np.exp(-k * taus_)
        y = x[0] + x[1] * e1
        if n_factors == 3:
            e2 = np.exp(-2.0 * k * taus_)
            y = y + x[2] * e2
        return np.exp(-y * taus_)

    return discounts_fn


# -----------------------------
# Run one model (2F or 3F) with per-ccy grid search over k
# -----------------------------
def run_model(
    df: pd.DataFrame,
    tenors: np.ndarray,
    n_factors: int,
    CCY_FREQ: dict,
    K_GRID: np.ndarray,
    out_root: str,
    P0_scale: float = 1.0,
):
    m = len(tenors)

    out_dir = os.path.join(out_root, f"general_{n_factors}f")
    os.makedirs(out_dir, exist_ok=True)

    resid_dir = os.path.join(out_dir, "residuals")
    os.makedirs(resid_dir, exist_ok=True)

    # Simple diagonal dynamics (same style as your DNS benchmark init)
    if n_factors == 2:
        A = np.diag([0.95, 0.90])
        Q = np.diag([1e-6, 1e-6])
        factor_cols = ["Factor1", "Factor2"]
    else:
        A = np.diag([0.95, 0.90, 0.85])
        Q = np.diag([1e-6, 1e-6, 1e-6])
        factor_cols = ["Factor1", "Factor2", "Factor3"]

    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors) * P0_scale

    # Measurement noise (diagonal baseline)
    R0 = np.eye(m) * 1e-6

    rows = []

    for ccy in sorted(df["ccy"].unique()):
        dfi = df[df["ccy"] == ccy].sort_values("as_of_date").copy()
        Y = dfi[TARGET_TENORS].astype(float).to_numpy()
        dates = dfi["as_of_date"].to_numpy()
        freq = CCY_FREQ.get(ccy, 1)

        best = None

        for k in K_GRID:
            discounts_fn = make_general_exponential_discounts(n_factors=n_factors, k=float(k))

            Xs, Ps, Yfit = ekf_filter_smoother_generic(
                Y=Y,
                tenors=tenors,
                A=A,
                Q=Q,
                R=R0,
                x0=x0,
                P0=P0,
                discounts_fn=discounts_fn,
                freq=freq,
            )

            bps = rmse_bps(Y, Yfit)

            if (best is None) or (bps < best["rmse"]):
                best = {
                    "k": float(k),
                    "rmse": float(bps),
                    "Xs": Xs,
                    "Yfit": Yfit,
                }

        # record best
        rows.append({"ccy": ccy, "rmse_bps": best["rmse"], "k": best["k"]})
        print(f"{ccy} ({n_factors}F): RMSE = {best['rmse']:.2f} bps | k* = {best['k']:.3f}")

        # Save factors (best k)
        fac_df = pd.DataFrame(best["Xs"], columns=factor_cols)
        fac_df.insert(0, "as_of_date", dates)
        fac_df.to_csv(os.path.join(out_dir, f"general_factors_{ccy}.csv"), index=False)

        # Save fitted vs true swaps (best k)
        fit_df = pd.DataFrame(best["Yfit"], columns=[f"fit_{t}" for t in TARGET_TENORS])
        true_df = pd.DataFrame(Y, columns=[f"true_{t}" for t in TARGET_TENORS])
        out_df = pd.concat([pd.DataFrame({"as_of_date": dates}), true_df, fit_df], axis=1)
        out_df.to_csv(os.path.join(out_dir, f"general_swaps_fit_{ccy}.csv"), index=False)

        # Residual plots (best k)
        resid_bps = (Y - best["Yfit"]) * 1e4

        plot_residuals_time_series(
            dates=dates,
            resid_bps=resid_bps,
            tenors=tenors,
            title=f"{ccy} ({n_factors}F) residuals by tenor (bps) — general model (k*={best['k']:.3f})",
            out_path=os.path.join(resid_dir, f"residuals_by_tenor_{ccy}.png"),
        )

        plot_rmse_over_time(
            dates=dates,
            resid_bps=resid_bps,
            title=f"{ccy} ({n_factors}F) cross-tenor RMSE over time (bps) — general model (k*={best['k']:.3f})",
            out_path=os.path.join(resid_dir, f"rmse_over_time_{ccy}.png"),
        )

    rmse_df = pd.DataFrame(rows).sort_values("rmse_bps", na_position="last")
    avg_rmse = rmse_df["rmse_bps"].mean()
    rmse_df.loc[len(rmse_df)] = {"ccy": "Average", "rmse_bps": avg_rmse, "k": np.nan}

    rmse_path = os.path.join(out_dir, "general_rmse_bps.csv")
    rmse_df.to_csv(rmse_path, index=False)

    print(f"\nGeneral EKF RMSE (par swaps, bps) — {n_factors} factors:")
    print(rmse_df)
    print(f"\nAverage RMSE across currencies ({n_factors}F): {avg_rmse:.2f} bps")
    print(f"\nSaved: {rmse_path}\n")

    return rmse_df, out_dir


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    USE = "bbg"

    CCY_FREQ = {"USD": 2, "GBP": 2, "JPY": 2, "CAD": 2, "AUD": 2, "EUR": 1, "DKK": 1, "SEK": 1, "NOK": 1}
    P0_scale = 1.0

    # Grid over k (analogous to lambda grid in DNS)
    K_GRID = np.linspace(0.01, 2.00, 30)

    meta, X_tensor, tenors_meta, df_wide, SCALE_IS_PERCENT = my_data(use=USE)

    df = df_wide.copy()
    if SCALE_IS_PERCENT:
        for col in TARGET_TENORS:
            df[col] = df[col].astype(float) / 100.0

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["ccy"] = df["ccy"].astype(str)

    tenors = np.array(TARGET_TENORS, dtype=float)

    # Root output folder
    out_root = os.path.join(os.getcwd(), "Figures", "bbg", "kalman_benchmark_general")
    os.makedirs(out_root, exist_ok=True)

    # Run 2F and 3F
    rmse_2f, dir_2f = run_model(df, tenors, 2, CCY_FREQ, K_GRID, out_root, P0_scale=P0_scale)
    rmse_3f, dir_3f = run_model(df, tenors, 3, CCY_FREQ, K_GRID, out_root, P0_scale=P0_scale)

    # Comparison table: RMSE 2F vs RMSE 3F
    df2 = rmse_2f[rmse_2f["ccy"] != "Average"].copy()
    df3 = rmse_3f[rmse_3f["ccy"] != "Average"].copy()

    comp = df2.merge(df3, on="ccy", how="inner", suffixes=("_2f", "_3f"))
    comp["abs_improvement_bps"] = comp["rmse_bps_2f"] - comp["rmse_bps_3f"]
    comp["rel_improvement_pct"] = 100.0 * comp["abs_improvement_bps"] / comp["rmse_bps_2f"]

    # Add average row
    avg_row = {
        "ccy": "Average",
        "rmse_bps_2f": df2["rmse_bps"].mean(),
        "k_2f": np.nan,
        "rmse_bps_3f": df3["rmse_bps"].mean(),
        "k_3f": np.nan,
    }
    avg_row["abs_improvement_bps"] = avg_row["rmse_bps_2f"] - avg_row["rmse_bps_3f"]
    avg_row["rel_improvement_pct"] = 100.0 * avg_row["abs_improvement_bps"] / avg_row["rmse_bps_2f"]

    comp = comp.sort_values("rmse_bps_3f").reset_index(drop=True)
    comp = pd.concat([comp, pd.DataFrame([avg_row])], ignore_index=True)

    print("\nRMSE comparison table: 2F vs 3F (general model)")
    print(comp[[
        "ccy",
        "rmse_bps_2f", "k_2f",
        "rmse_bps_3f", "k_3f",
        "abs_improvement_bps", "rel_improvement_pct"
    ]])

    comp_path = os.path.join(out_root, "rmse_comparison_2f_vs_3f.csv")
    comp.to_csv(comp_path, index=False)
    print(f"\nSaved RMSE comparison table: {comp_path}")