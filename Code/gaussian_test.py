import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import build_all_dataframes, TARGET_TENORS
from Code.utils.rates import par_swap_from_discount
from Code.utils.gaussian_fit import fit_optionA_2f
from Code.model.gaussian import Gaussian2F


# -----------------------------
# Config
# -----------------------------
DEVICE = "cpu"
TENORS = list(TARGET_TENORS)

SAMPLE_PER_CCY = 200
OUTER_STEPS = 10
INNER_STEPS = 80
LR_PARAMS = 5e-2
LR_Z = 1e-1

SAVE_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)


def to_decimals(Y: np.ndarray) -> np.ndarray:
    """Convert percent to decimals if needed."""
    if np.nanmean(Y) > 1.0:
        return Y / 100.0
    return Y


def compute_rmse_bps(S_pred: np.ndarray, S_obs: np.ndarray, tenors: list[int]):
    """Return total RMSE bps and per-tenor RMSE bps."""
    err = (S_pred - S_obs)  # decimals
    rmse_total_bps = 1e4 * float(np.sqrt(np.mean(err ** 2)))
    rmse_by_tenor_bps = 1e4 * np.sqrt(np.mean(err ** 2, axis=0))
    return rmse_total_bps, dict(zip(tenors, rmse_by_tenor_bps))


if __name__ == "__main__":
    # -----------------------------
    # Load data
    # -----------------------------
    data = build_all_dataframes()
    df = data["df_wide_bbg_full"].copy()  # or df_wide_test_full
    df = df.dropna(subset=TENORS).copy()

    # -----------------------------
    # Pick one currency to demo
    # -----------------------------
    ccy = df["ccy"].value_counts().index[0]
    df_ccy_all = df[df["ccy"] == ccy].sort_values("as_of_date").reset_index(drop=True)
    print("Using currency:", ccy, " (#curves:", len(df_ccy_all), ")")

    # sample curves for speed
    if len(df_ccy_all) < SAMPLE_PER_CCY:
        raise ValueError(f"Not enough curves for {ccy}: {len(df_ccy_all)} < {SAMPLE_PER_CCY}")

    df_ccy = (
        df_ccy_all.sample(SAMPLE_PER_CCY, random_state=0)
        .sort_values("as_of_date")
        .reset_index(drop=True)
    )

    Y = to_decimals(df_ccy[TENORS].to_numpy(dtype=float))  # (N,K)

    # -----------------------------
    # Fit 2F Gaussian (Option A)
    # -----------------------------
    g2_model, z = fit_optionA_2f(
        Y, TENORS,
        outer_steps=OUTER_STEPS,
        inner_steps=INNER_STEPS,
        lr_params=LR_PARAMS,
        lr_z=LR_Z,
        device=DEVICE
    )

    print("\nFitted 2F Gaussian params:")
    print("  kappa1:", float(g2_model.kappa1.item()))
    print("  theta1:", float(g2_model.theta1.item()))
    print("  sigma1:", float(g2_model.sigma1.item()))
    print("  kappa2:", float(g2_model.kappa2.item()))
    print("  theta2:", float(g2_model.theta2.item()))
    print("  sigma2:", float(g2_model.sigma2.item()))

    # -----------------------------
    # Predict and compute RMSE
    # -----------------------------
    with torch.no_grad():
        P_all = g2_model.discount_curve_annual(z, T_max=max(TENORS))          # (N,T)
        S_all = par_swap_from_discount(P_all, TENORS).cpu().numpy()          # (N,K)

    rmse_total_bps, rmse_by_tenor = compute_rmse_bps(S_all, Y, TENORS)
    print("\n2F Gaussian RMSE:")
    print(f"  Total RMSE (bps): {rmse_total_bps:.3f}")
    print("  RMSE by tenor (bps):")
    for t in TENORS:
        print(f"    {t:>2}Y: {rmse_by_tenor[t]:.3f}")

    # -----------------------------
    # Plot one representative curve
    # -----------------------------
    pick_idx = len(df_ccy) // 2
    row = df_ccy.iloc[pick_idx]
    S_obs = Y[pick_idx]
    S_pred = S_all[pick_idx]

    plt.figure()
    plt.plot(TENORS, S_obs, marker="o", label="Observed")
    plt.plot(TENORS, S_pred, marker="o", label="2F Gaussian fitted (Option A)")
    plt.title(f"2F Gaussian Option A fit (ccy={ccy}, date={pd.to_datetime(row['as_of_date']).date()})")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Par swap rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # OPTIONAL: run across currencies and save a CSV table
    # ============================================================
    print("\n=== Running 2F Gaussian Option A across currencies (sample "
          f"{SAMPLE_PER_CCY} curves each) ===")

    results = []
    for ccy_i in sorted(df["ccy"].unique()):
        df_i = df[df["ccy"] == ccy_i].sort_values("as_of_date").reset_index(drop=True)
        if len(df_i) < SAMPLE_PER_CCY:
            continue

        df_sub = (
            df_i.sample(SAMPLE_PER_CCY, random_state=0)
            .sort_values("as_of_date")
            .reset_index(drop=True)
        )
        Y_i = to_decimals(df_sub[TENORS].to_numpy(dtype=float))

        g2_i, z_i = fit_optionA_2f(
            Y_i, TENORS,
            outer_steps=OUTER_STEPS,
            inner_steps=INNER_STEPS,
            lr_params=LR_PARAMS,
            lr_z=LR_Z,
            device=DEVICE
        )

        with torch.no_grad():
            P_i = g2_i.discount_curve_annual(z_i, T_max=max(TENORS))
            S_i = par_swap_from_discount(P_i, TENORS).cpu().numpy()

        rmse_total_bps_i, rmse_by_tenor_i = compute_rmse_bps(S_i, Y_i, TENORS)

        results.append({
            "ccy": ccy_i,
            "n_curves": len(df_sub),
            "rmse_bps_total": rmse_total_bps_i,
            "kappa1": float(g2_i.kappa1.item()),
            "theta1": float(g2_i.theta1.item()),
            "sigma1": float(g2_i.sigma1.item()),
            "kappa2": float(g2_i.kappa2.item()),
            "theta2": float(g2_i.theta2.item()),
            "sigma2": float(g2_i.sigma2.item()),
            **{f"rmse_{t}y_bps": float(rmse_by_tenor_i[t]) for t in TENORS}
        })

    res_df = pd.DataFrame(results).sort_values("rmse_bps_total")
    print("\n=== 2F Gaussian summary (sorted by total RMSE) ===")
    if len(res_df) == 0:
        print("No currencies had enough curves.")
    else:
        print(
            res_df[
                ["ccy", "n_curves", "rmse_bps_total",
                 "kappa1", "theta1", "sigma1",
                 "kappa2", "theta2", "sigma2"]
            ].to_string(index=False)
        )

        out_path = os.path.join(SAVE_DIR, "gaussian2f_optionA_rmse_table.csv")
        res_df.to_csv(out_path, index=False)
        print("\nSaved:", out_path)
