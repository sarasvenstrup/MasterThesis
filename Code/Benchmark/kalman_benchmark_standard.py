# Code/kalman_dns_kf_oos.py
#
# Correct OOS evaluation — standard LINEAR Kalman filter DNS on yields:
#   1. Estimate lambda, A, Q, R on TRAIN data only (2004-2020)
#      - lambda grid search + 2-pass quasi-EM + RTS smoother (all in-sample, fine)
#   2. Freeze parameters and run FORWARD-ONLY KF on TEST data (2021-2022)
#      - No RTS smoother on test (that would use future info)
#      - No re-estimation on test
#
# Supports n_factors in {1, 2, 3, 4}
#
# Produces (per factor model):
#   - rmse_summary.csv  (IS mean, IS std, OOS mean, OOS std)  <- matches OutOfSampleSplit.py
#   - oos_fitted_vs_actual.png                                <- same layout as autoencoder plot
#   - is_fitted_vs_actual.png
#   - latent_factors_train.png / latent_factors_oos.png
#
# Output folder:
#   Figures/kalman_benchmark_oos/kf_dns_{n_factors}f/

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Code.load_swapdata import my_data, TARGET_TENORS

# ── config ─────────────────────────────────────────────────────────────────────
TRAIN_START = "2010-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2022-12-31"

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

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
# Nelson-Siegel loadings
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


# ══════════════════════════════════════════════════════════════════════════════
# Linear KF + RTS smoother  (used on TRAIN only)
# ══════════════════════════════════════════════════════════════════════════════

def kf_filter_smoother(Y, H, A, Q, R, x0, P0):
    T, m = Y.shape
    n = A.shape[0]
    I = np.eye(n)

    x_pred = np.zeros((T, n)); P_pred = np.zeros((T, n, n))
    x_filt = np.zeros((T, n)); P_filt = np.zeros((T, n, n))
    x_prev, P_prev = x0.copy(), P0.copy()

    for t in range(T):
        xp = A @ x_prev
        Pp = A @ P_prev @ A.T + Q
        yp = H @ xp
        v  = Y[t] - yp
        S  = H @ Pp @ H.T + R
        K  = Pp @ H.T @ np.linalg.inv(S)
        xf = xp + K @ v
        KH = K @ H
        Pf = (I - KH) @ Pp @ (I - KH).T + K @ R @ K.T
        x_pred[t], P_pred[t] = xp, Pp
        x_filt[t], P_filt[t] = xf, Pf
        x_prev, P_prev = xf, Pf

    # RTS smoother (backward pass — fine on train only)
    x_smooth = x_filt.copy(); P_smooth = P_filt.copy()
    for t in range(T - 2, -1, -1):
        G = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

    Y_fit = x_smooth @ H.T  # (T, m)
    return x_smooth, P_smooth, Y_fit


# ══════════════════════════════════════════════════════════════════════════════
# Forward-only linear KF  (used on TEST — no future info)
# ══════════════════════════════════════════════════════════════════════════════

def kf_forward_only(Y, H, A, Q, R, x0, P0):
    """Sequential forward KF with frozen parameters. No smoother."""
    T, m = Y.shape
    n = A.shape[0]
    I = np.eye(n)

    x_filt = np.zeros((T, n))
    x_prev, P_prev = x0.copy(), P0.copy()

    for t in range(T):
        xp = A @ x_prev
        Pp = A @ P_prev @ A.T + Q
        yp = H @ xp
        v  = Y[t] - yp
        S  = H @ Pp @ H.T + R
        K  = Pp @ H.T @ np.linalg.inv(S)
        xf = xp + K @ v
        KH = K @ H
        Pf = (I - KH) @ Pp @ (I - KH).T + K @ R @ K.T
        x_filt[t] = xf
        x_prev, P_prev = xf, Pf

    Y_fit = x_filt @ H.T  # (T, m)
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

def run_model_oos(df_train, df_test, tenors, n_factors, LAM_GRID,
                  out_dir, P0_scale=1.0, A_shrink=0.0):
    """
    For each currency:
      1. Grid-search lambda + quasi-EM on train  -> best params
      2. Forward-only KF on test with frozen params
    Returns IS and OOS per-currency RMSE.
    """
    os.makedirs(out_dir, exist_ok=True)
    factor_cols = FACTOR_COLS[n_factors]

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

        Y_te     = dfi_te[TARGET_TENORS].astype(float).to_numpy() if len(dfi_te) > 0 else None
        dates_te = dfi_te["as_of_date"].to_numpy() if len(dfi_te) > 0 else None

        best = None

        # ── Step 1: grid search + quasi-EM on train ───────────────────────────
        for lam in LAM_GRID:
            H = ns_loadings(tenors, lam, n_factors)   # (m, n) — fixed for linear KF
            x0, *_ = np.linalg.lstsq(H, Y_tr[0], rcond=None)
            P0 = np.eye(n_factors) * P0_scale
            A  = 0.90 * np.eye(n_factors)
            Q  = 1e-6 * np.eye(n_factors)
            R  = 1e-6 * np.eye(len(tenors))

            # Pass 1
            Xs1, _, Yfit1 = kf_filter_smoother(Y_tr, H, A, Q, R, x0, P0)
            # Quasi-EM update
            A2, Q2 = estimate_A_Q(Xs1, A_shrink=A_shrink)
            R2     = estimate_R(Y_tr - Yfit1)
            # Pass 2
            Xs2, _, Yfit2 = kf_filter_smoother(Y_tr, H, A2, Q2, R2, x0, P0)

            bps = rmse_bps(Y_tr, Yfit2)
            if best is None or bps < best["rmse_is"]:
                best = dict(lam=lam, H=H, A=A2, Q=Q2, R=R2,
                            x0=x0, P0=P0,
                            Xs_tr=Xs2, Yfit_tr=Yfit2,
                            rmse_is=bps)

        is_rmse_rows.append({"Currency": ccy, "RMSE_bps": best["rmse_is"]})
        print(f"  {ccy} ({n_factors}F)  IS RMSE = {best['rmse_is']:.2f} bps | lambda* = {best['lam']:.3f}")

        # ── Step 2: forward-only KF on test with frozen params ────────────────
        oos_bps = np.nan
        Xs_te, Yfit_te = None, None

        if Y_te is not None and len(Y_te) > 0:
            x_init = best["Xs_tr"][-1]
            P_init = np.eye(n_factors) * P0_scale

            Xs_te, Yfit_te = kf_forward_only(
                Y_te, best["H"],
                best["A"], best["Q"], best["R"],
                x_init, P_init
            )
            oos_bps = rmse_bps(Y_te, Yfit_te)
            print(f"  {ccy} ({n_factors}F) OOS RMSE = {oos_bps:.2f} bps")

        oos_rmse_rows.append({"Currency": ccy, "RMSE_bps": oos_bps})

        fit_store[ccy] = dict(
            dates_tr=dates_tr, Y_tr=Y_tr,  Yfit_tr=best["Yfit_tr"], Xs_tr=best["Xs_tr"],
            dates_te=dates_te, Y_te=Y_te,  Yfit_te=Yfit_te,         Xs_te=Xs_te,
        )

        # Save per-currency CSVs
        pd.DataFrame(best["Xs_tr"], columns=factor_cols).assign(as_of_date=dates_tr)\
          .to_csv(os.path.join(out_dir, f"factors_train_{ccy}.csv"), index=False)
        if Xs_te is not None:
            pd.DataFrame(Xs_te, columns=factor_cols).assign(as_of_date=dates_te)\
              .to_csv(os.path.join(out_dir, f"factors_oos_{ccy}.csv"), index=False)

    return is_rmse_rows, oos_rmse_rows, fit_store


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers  (same layout as OutOfSampleSplit.py)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fitted_vs_actual(fit_store, tenors, split, out_path, n_factors):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for ax, ccy in zip(axes, ccy_order):
        if ccy not in fit_store:
            ax.set_visible(False)
            continue

        d = fit_store[ccy]
        if split == "train":
            dates = pd.to_datetime(d["dates_tr"])
            Y     = d["Y_tr"]
            Y_fit = d["Yfit_tr"]
        else:
            if d["Y_te"] is None:
                ax.set_visible(False)
                continue
            dates = pd.to_datetime(d["dates_te"])
            Y     = d["Y_te"]
            Y_fit = d["Yfit_te"]

        mid_date = dates.min() + (dates.max() - dates.min()) / 2
        deltas = pd.to_numeric((dates - mid_date).abs())
        idx = int(deltas.argmin())

        ax.plot(tenors, Y[idx] * 100,     "o-",  label="Actual", linewidth=1.8)
        ax.plot(tenors, Y_fit[idx] * 100, "s--", label="Fitted", linewidth=1.8)
        ax.set_title(f"{ccy}  ({dates[idx].strftime('%Y-%m-%d')})")
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("Rate (%)")
        ax.legend(fontsize=7)

    split_label = "OOS" if split == "oos" else "In-Sample"
    fig.suptitle(f"{split_label}: Fitted vs Actual Swap Curves — KF DNS {n_factors}F",
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
    fig.suptitle(f"Latent factor paths — {split_label} — KF DNS {n_factors}F",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_rmse_summary(is_rows, oos_rows, out_path):
    """
    Saves rmse_summary.csv in the same format as OutOfSampleSplit.py:
      Currency | IS mean (bps) | IS std (bps) | OOS mean (bps) | OOS std (bps)
    Std = NaN for Kalman (single run, no seeds).
    """
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

    for n_factors in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Running KF DNS  {n_factors}-factor model")
        print(f"{'='*60}")

        out_dir = os.path.join(
            os.getcwd(), "Figures", "kalman_benchmark_oos", f"kf_dns_{n_factors}f"
        )
        os.makedirs(out_dir, exist_ok=True)

        is_rows, oos_rows, fit_store = run_model_oos(
            df_train, df_test, tenors, n_factors,
            LAM_GRID, out_dir,
            P0_scale=P0_scale, A_shrink=A_shrink,
        )

        # RMSE summary table
        save_rmse_summary(
            is_rows, oos_rows,
            os.path.join(out_dir, "rmse_summary.csv")
        )

        # Fitted vs actual plots
        plot_fitted_vs_actual(
            fit_store, tenors, split="train",
            out_path=os.path.join(out_dir, "is_fitted_vs_actual.png"),
            n_factors=n_factors,
        )
        plot_fitted_vs_actual(
            fit_store, tenors, split="oos",
            out_path=os.path.join(out_dir, "oos_fitted_vs_actual.png"),
            n_factors=n_factors,
        )

        # Latent factor path plots
        plot_latent_factors(
            fit_store, split="train",
            out_path=os.path.join(out_dir, "latent_factors_train.png"),
            n_factors=n_factors,
        )
        plot_latent_factors(
            fit_store, split="oos",
            out_path=os.path.join(out_dir, "latent_factors_oos.png"),
            n_factors=n_factors,
        )

        print(f"\nAll outputs saved to: {out_dir}")