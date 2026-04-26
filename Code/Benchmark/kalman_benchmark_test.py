# kalman_benchmark_test.py
#
# Theoretically correct EKF-DNS benchmark — in-sample only.
#
# Uses the same data period as Training_baseline.py: 2010-01-01 onwards,
# no upper cutoff.
#
# State space model:
#   Transition:  x_{t+1} = A x_t + b + w_t,   w_t ~ N(0, Q)
#   Measurement: y_t     = h(x_t) + eps_t,   eps_t ~ N(0, R)
#
# where:
#   - A is diagonal (d x d), entries in (0,1) — stationarity
#   - b is a free drift vector (d,)
#   - Q is diagonal (d x d), positive
#   - R is diagonal (m x m), positive
#   - h(x) = par swap rates from Nelson-Siegel factors (nonlinear)
#   - lambda (NS decay) estimated jointly with A, b, Q, R by MLE
#
# Estimation:
#   All parameters estimated jointly by maximum likelihood via the
#   prediction error decomposition of the EKF:
#     log L = sum_t [ -m/2 log(2pi) - 1/2 log|S_t| - 1/2 v_t' S_t^{-1} v_t ]
#
# Output: Figures/KalmanBenchmarkTest/ekf_dns_{n}f/

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data, TARGET_TENORS

# ── config ─────────────────────────────────────────────────────────────────────
# Same start date as Training_baseline.py — no upper cutoff
DATA_START = "2010-01-01"

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

CCY_FREQ = {
    "USD": 2, "GBP": 2, "JPY": 2, "CAD": 2, "AUD": 2,
    "EUR": 1, "DKK": 1, "SEK": 1, "NOK": 1,
}

P0_SCALE = 1.0

FACTOR_COLS = {
    1: ["Level"],
    2: ["Level", "Slope"],
    3: ["Level", "Slope", "Curvature"],
    4: ["Level", "Slope", "Curvature", "LongCurv"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Nelson-Siegel helpers
# ══════════════════════════════════════════════════════════════════════════════

def ns_loadings(taus, lam, n_factors):
    taus = np.asarray(taus, dtype=float)
    a    = (1.0 - np.exp(-lam * taus)) / (lam * taus)
    b    = a - np.exp(-lam * taus)
    c    = taus * np.exp(-lam * taus)
    if n_factors == 1:
        return np.column_stack([np.ones_like(taus)])
    elif n_factors == 2:
        return np.column_stack([np.ones_like(taus), a])
    elif n_factors == 3:
        return np.column_stack([np.ones_like(taus), a, b])
    elif n_factors == 4:
        return np.column_stack([np.ones_like(taus), a, b, c])
    raise ValueError("n_factors must be 1, 2, 3 or 4")


def swap_curve_from_ns(tenors_years, x, lam, freq=1):
    tenors_years = np.asarray(tenors_years, dtype=float)
    out = np.empty_like(tenors_years, dtype=float)
    for i, T in enumerate(tenors_years):
        N     = max(int(round(T * freq)), 1)
        taus  = np.arange(1, N + 1, dtype=float) / float(freq)
        y     = ns_loadings(taus, lam, n_factors=x.size) @ x.real
        P     = np.exp(-y * taus)
        delta = 1.0 / float(freq)
        denom = (delta * P).sum()
        denom = denom if abs(denom) > 1e-14 else 1e-14
        out[i] = (1.0 - P[-1]) / denom
    return out


def jacobian_cs(func, x, h=1e-20):
    """Complex-step Jacobian."""
    x  = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    m, d = f0.size, x.size
    J = np.zeros((m, d))
    for j in range(d):
        xc     = x.astype(complex)
        xc[j] += 1j * h
        J[:, j] = np.imag(func(xc)) / h
    return J


def mid_period_idx(dates):
    ts  = np.array(dates, dtype="datetime64[ns]").astype(np.int64)
    mid = (ts.min() + ts.max()) // 2
    return int(np.abs(ts - mid).argmin())


# ══════════════════════════════════════════════════════════════════════════════
# Parameter packing / unpacking
#
# Free vector theta = [ logit(A_diag),  b,  log(Q_diag),  log(R_diag),  log(lam) ]
# ══════════════════════════════════════════════════════════════════════════════

def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1.0 - x))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def pack_theta(A_diag, b, Q_diag, R_diag, lam):
    return np.concatenate([
        _logit(A_diag),
        b,
        np.log(Q_diag + 1e-15),
        np.log(R_diag + 1e-15),
        [np.log(float(lam))],
    ])


def unpack_theta(theta, d, m):
    A_diag = _sigmoid(theta[:d])
    b_vec  = theta[d:2*d]
    Q_diag = np.exp(theta[2*d:3*d])
    R_diag = np.exp(theta[3*d:3*d+m])
    lam    = float(np.exp(theta[3*d+m]))
    return np.diag(A_diag), b_vec, np.diag(Q_diag), np.diag(R_diag), lam


# ══════════════════════════════════════════════════════════════════════════════
# Negative log-likelihood (forward EKF only — used inside MLE)
# ══════════════════════════════════════════════════════════════════════════════

def neg_loglik(theta, Y, tenors, n_factors, freq):
    d, m_obs = n_factors, Y.shape[1]

    try:
        A, b_vec, Q, R_mat, lam = unpack_theta(theta, d, m_obs)
    except Exception:
        return 1e10

    if lam <= 1e-4 or lam > 10.0:
        return 1e10

    def h(x):
        return swap_curve_from_ns(tenors, x, lam, freq=freq)

    try:
        H0     = ns_loadings(tenors, lam, n_factors)
        x0, *_ = np.linalg.lstsq(H0, Y[0], rcond=None)
        P0     = np.eye(d) * P0_SCALE
    except Exception:
        return 1e10

    x_prev, P_prev = x0.copy(), P0.copy()
    I_d = np.eye(d)
    ll  = 0.0

    for t in range(len(Y)):
        # Prediction
        xp = A @ x_prev + b_vec
        Pp = A @ P_prev @ A.T + Q

        # Measurement
        try:
            yp = np.asarray(h(xp), dtype=float)
            H  = jacobian_cs(h, xp)
        except Exception:
            return 1e10

        v = Y[t] - yp
        S = H @ Pp @ H.T + R_mat
        S = 0.5 * (S + S.T)

        # Log-likelihood contribution
        try:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                return 1e10
            ll += -0.5 * (m_obs * np.log(2.0 * np.pi) + logdet
                          + v @ np.linalg.solve(S, v))
        except Exception:
            return 1e10

        # Update (Joseph form)
        K  = Pp @ H.T @ np.linalg.inv(S)
        KH = K @ H
        xf = xp + K @ v
        Pf = (I_d - KH) @ Pp @ (I_d - KH).T + K @ R_mat @ K.T
        Pf = 0.5 * (Pf + Pf.T)

        x_prev, P_prev = xf, Pf

    return -ll


# ══════════════════════════════════════════════════════════════════════════════
# MLE fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_mle(Y, tenors, n_factors, freq):
    """
    Estimate A (diag), b, Q (diag), R (diag), lambda jointly by MLE.
    Returns: A, b, Q, R_mat, lam, optimisation result
    """
    d, m_obs = n_factors, len(tenors)

    theta0 = pack_theta(
        A_diag  = np.full(d, 0.95),
        b       = np.zeros(d),
        Q_diag  = np.full(d, 1e-4),
        R_diag  = np.full(m_obs, 1e-4),
        lam     = 0.5,
    )

    print(f"    MLE: optimising {len(theta0)} parameters ...")

    result = scipy.optimize.minimize(
        neg_loglik,
        theta0,
        args=(Y, tenors, n_factors, freq),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-8},
    )

    if not result.success:
        print(f"    [WARN] MLE did not converge: {result.message}")

    A, b_vec, Q, R_mat, lam = unpack_theta(result.x, d, m_obs)
    print(f"    MLE: loglik={-result.fun:.4f}  lambda*={lam:.4f}")
    print(f"    A_diag = {np.diag(A).round(4)}")
    print(f"    b      = {b_vec.round(4)}")

    return A, b_vec, Q, R_mat, lam, result


# ══════════════════════════════════════════════════════════════════════════════
# EKF + RTS smoother
# ══════════════════════════════════════════════════════════════════════════════

def ekf_filter_smoother(Y, tenors, lam, A, b_vec, Q, R_mat, x0, P0, freq=1):
    T, m_obs = Y.shape
    d = A.shape[0]
    I_d = np.eye(d)

    def h(x):
        return swap_curve_from_ns(tenors, x, lam, freq=freq)

    x_pred = np.zeros((T, d));  P_pred = np.zeros((T, d, d))
    x_filt = np.zeros((T, d));  P_filt = np.zeros((T, d, d))
    x_prev, P_prev = x0.copy(), P0.copy()

    for t in range(T):
        xp = A @ x_prev + b_vec
        Pp = A @ P_prev @ A.T + Q

        yp = np.asarray(h(xp), dtype=float)
        H  = jacobian_cs(h, xp)
        v  = Y[t] - yp
        S  = H @ Pp @ H.T + R_mat
        S  = 0.5 * (S + S.T)

        K  = Pp @ H.T @ np.linalg.inv(S)
        KH = K @ H
        xf = xp + K @ v
        Pf = (I_d - KH) @ Pp @ (I_d - KH).T + K @ R_mat @ K.T
        Pf = 0.5 * (Pf + Pf.T)

        x_pred[t], P_pred[t] = xp, Pp
        x_filt[t], P_filt[t] = xf, Pf
        x_prev, P_prev = xf, Pf

    # RTS backward smoother
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(T - 2, -1, -1):
        G = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

    Y_fit = np.vstack([np.asarray(h(x_smooth[t]), dtype=float) for t in range(T)])
    return x_smooth, P_smooth, Y_fit


# ══════════════════════════════════════════════════════════════════════════════
# RMSE helper
# ══════════════════════════════════════════════════════════════════════════════

def rmse_bps(y_true, y_fit):
    return float(np.sqrt(np.nanmean((y_fit - y_true) ** 2)) * 1e4)


# ══════════════════════════════════════════════════════════════════════════════
# Main model runner
# ══════════════════════════════════════════════════════════════════════════════

def run_model(df, tenors, n_factors, CCY_FREQ, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    rmse_rows = []
    fit_store = {}

    for ccy in ccy_order:
        dfi = df[df["ccy"] == ccy].sort_values("as_of_date")

        if len(dfi) == 0:
            print(f"  {ccy}: no data, skipping.")
            continue

        Y      = dfi[TARGET_TENORS].astype(float).to_numpy()
        dates  = dfi["as_of_date"].to_numpy()
        freq   = CCY_FREQ.get(ccy, 1)

        print(f"\n  {ccy} ({n_factors}F) — fitting MLE  (n={len(Y)}) ...")

        A, b_vec, Q, R_mat, lam, opt_result = fit_mle(Y, tenors, n_factors, freq)

        H0     = ns_loadings(tenors, lam, n_factors)
        x0, *_ = np.linalg.lstsq(H0, Y[0], rcond=None)
        P0     = np.eye(n_factors) * P0_SCALE

        Xs, _, Y_fit = ekf_filter_smoother(
            Y, tenors, lam, A, b_vec, Q, R_mat, x0, P0, freq=freq
        )

        bps = rmse_bps(Y, Y_fit)
        print(f"  {ccy} ({n_factors}F)  IS RMSE = {bps:.2f} bps | lambda* = {lam:.4f}")
        rmse_rows.append({"Currency": ccy, "RMSE_bps": bps})

        fit_store[ccy] = dict(
            dates=dates, Y=Y, Y_fit=Y_fit, Xs=Xs,
            lam=lam, A=A, b=b_vec, Q=Q, R=R_mat,
            opt_success=opt_result.success,
        )

    return rmse_rows, fit_store


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_fitted_vs_actual(fit_store, tenors, out_path, n_factors):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for ax, ccy in zip(axes, ccy_order):
        if ccy not in fit_store:
            ax.set_visible(False); continue
        d = fit_store[ccy]

        idx      = mid_period_idx(d["dates"])
        date_str = pd.Timestamp(d["dates"][idx]).strftime("%Y-%m-%d")

        ax.plot(tenors, d["Y"][idx] * 100,     "o-",  label="Actual", linewidth=1.8)
        ax.plot(tenors, d["Y_fit"][idx] * 100, "s--", label="Fitted", linewidth=1.8)
        ax.set_title(f"{ccy}  ({date_str})")
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("Rate (%)")
        ax.legend(fontsize=7)

    fig.suptitle(f"Fitted vs Actual — EKF DNS {n_factors}F (MLE)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_latent_factors(fit_store, out_path, n_factors):
    factor_names = FACTOR_COLS[n_factors]
    fig, axes = plt.subplots(n_factors, 1, figsize=(12, 3 * n_factors), sharex=True)
    if n_factors == 1:
        axes = [axes]

    for ccy in ccy_order:
        if ccy not in fit_store:
            continue
        d        = fit_store[ccy]
        sort_idx = np.argsort(d["dates"])
        for i, ax in enumerate(axes):
            ax.plot(d["dates"][sort_idx], d["Xs"][sort_idx, i],
                    linewidth=1.2, label=ccy, alpha=0.85)

    for i, ax in enumerate(axes):
        ax.set_ylabel(factor_names[i])
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if i == 0:
            ax.legend(ncol=3, fontsize=7)

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Latent factor paths — EKF DNS {n_factors}F (MLE)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_rmse_summary(rmse_rows, out_path):
    df = pd.DataFrame(rmse_rows).set_index("Currency")
    df.columns = ["IS RMSE (bps)"]
    df.loc["Average"] = df.mean()
    df.to_csv(out_path)
    print(f"\nRMSE summary:\n{df.to_string()}")
    print(f"Saved: {out_path}")
    return df


def save_mle_params(fit_store, out_path, n_factors):
    rows = []
    for ccy, res in fit_store.items():
        row = {"Currency": ccy, "lambda": res["lam"], "mle_converged": res["opt_success"]}
        for k in range(n_factors):
            row[f"A_{k+1}"] = res["A"][k, k]
            row[f"b_{k+1}"] = res["b"][k]
            row[f"Q_{k+1}"] = res["Q"][k, k]
        for j in range(len(TARGET_TENORS)):
            row[f"R_{j+1}"] = res["R"][j, j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved MLE parameters: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    USE = "bbg"

    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)

    # df_wide already filters to >= 2010-01-01 (same as Training_baseline.py)
    df = df_wide.copy()
    if SCALE_IS_PERCENT:
        for col in TARGET_TENORS:
            df[col] = df[col].astype(float) / 100.0

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["ccy"]        = df["ccy"].astype(str)

    print(f"Data: {DATA_START} onwards  n={len(df)}")
    print(f"Date range: {df['as_of_date'].min().date()} – {df['as_of_date'].max().date()}")

    tenors = np.array(TARGET_TENORS, dtype=float)

    root_dir = os.path.join(REPO_ROOT, "Figures", "KalmanBenchmarkTest")

    for n_factors in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Running EKF DNS  {n_factors}-factor model  (MLE)")
        print(f"{'='*60}")

        out_dir = os.path.join(root_dir, f"ekf_dns_{n_factors}f")
        os.makedirs(out_dir, exist_ok=True)

        rmse_rows, fit_store = run_model(df, tenors, n_factors, CCY_FREQ, out_dir)

        save_rmse_summary(rmse_rows,
                          os.path.join(out_dir, "rmse_summary.csv"))

        save_mle_params(fit_store,
                        os.path.join(out_dir, "mle_params.csv"),
                        n_factors)

        # Save latent factors to CSV
        _lf_rows = []
        for ccy, res in fit_store.items():
            for date, xs in zip(res["dates"], res["Xs"]):
                row = {"as_of_date": date, "ccy": ccy}
                for k in range(n_factors):
                    row[f"z{k+1}"] = xs[k]
                _lf_rows.append(row)
        pd.DataFrame(_lf_rows).to_csv(
            os.path.join(out_dir, "latent_factors.csv"), index=False
        )

        plot_fitted_vs_actual(fit_store, tenors,
                              out_path=os.path.join(out_dir, "fitted_vs_actual.png"),
                              n_factors=n_factors)

        plot_latent_factors(fit_store,
                            out_path=os.path.join(out_dir, "latent_factors.png"),
                            n_factors=n_factors)

        print(f"All outputs saved to: {out_dir}")

    print(f"\nAll factor models complete. Results in: {root_dir}")
