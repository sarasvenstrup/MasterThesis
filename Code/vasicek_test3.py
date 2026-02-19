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
from Code.utils.vasicek_fit import fit_r0_single_curve, fit_optionA
from Code.utils.rates import par_swap_from_discount

# -----------------------------
# Config
# -----------------------------
DEVICE = "cpu"
TENORS = list(TARGET_TENORS)

DO_R0_ONLY_SANITY = True          # set False once you trust plumbing
SAMPLE_PER_CCY = 200              # curves per currency for calibration/table
OUTER_STEPS = 10
INNER_STEPS = 80
LR_PARAMS = 5e-2
LR_R0 = 1e-1

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


def optionA_fit_and_predict(Y: np.ndarray, tenors: list[int]):
    """Fit Option A and return fitted model, r0s, and predicted swap matrix."""
    vas_model, r0s = fit_optionA(
        Y, tenors,
        outer_steps=OUTER_STEPS,
        inner_steps=INNER_STEPS,
        lr_params=LR_PARAMS,
        lr_r0=LR_R0,
        device=DEVICE
    )
    with torch.no_grad():
        P_all = vas_model.discount_curve_annual(r0s, T_max=max(tenors))
        S_all = par_swap_from_discount(P_all, tenors).cpu().numpy()
    return vas_model, r0s, S_all


if __name__ == "__main__":
    # -----------------------------
    # Load data once
    # -----------------------------
    data = build_all_dataframes()
    df = data["df_wide_bbg_full"].copy()  # or df_wide_test_full

    # -----------------------------
    # Pick a representative currency (most curves)
    # -----------------------------
    ccy = df["ccy"].value_counts().index[0]
    df_ccy_all = df[df["ccy"] == ccy].dropna(subset=TENORS).sort_values("as_of_date").reset_index(drop=True)

    print("Using currency for single-ccy demo:", ccy, " (#curves:", len(df_ccy_all), ")")

    # -----------------------------
    # Optional: r0-only sanity (one curve)
    # -----------------------------
    if DO_R0_ONLY_SANITY:
        pick_idx = len(df_ccy_all) // 2
        row = df_ccy_all.iloc[pick_idx]
        y = np.array([row[t] for t in TENORS], dtype=float)
        y = to_decimals(y)

        S_obs = torch.tensor(y, device=DEVICE, dtype=torch.float64)

        print("\nPicked curve (sanity):")
        print("  ccy:", ccy)
        print("  date:", pd.to_datetime(row["as_of_date"]).date())
        print("  observed (decimals):", S_obs.cpu().numpy())

        # fixed params just for sanity
        kappa0, theta0, sigma0 = 0.5, 0.02, 0.01
        r0_hat, model, S_pred, loss = fit_r0_single_curve(
            S_obs, TENORS, kappa=kappa0, theta=theta0, sigma=sigma0,
            n_steps=800, lr=5e-2
        )

        print("\nSanity fit (r0 only):")
        print("  r0_hat:", float(r0_hat.item()))
        print("  loss (MSE):", float(loss.item()))
        print("  params used:", "kappa", float(model.kappa.item()),
              "theta", float(model.theta.item()),
              "sigma", float(model.sigma.item()))

        plt.figure()
        plt.plot(TENORS, S_obs.cpu().numpy(), marker="o", label="Observed")
        plt.plot(TENORS, S_pred.cpu().numpy(), marker="o", label="Vasicek fitted (r0 only)")
        plt.xlabel("Maturity (years)")
        plt.ylabel("Par swap rate")
        plt.title(f"One-curve Vasicek sanity (ccy={ccy}, date={pd.to_datetime(row['as_of_date']).date()})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Option A: one currency demo + plot + RMSE
    # -----------------------------
    df_ccy = df_ccy_all.sample(SAMPLE_PER_CCY, random_state=0).sort_values("as_of_date").reset_index(drop=True)
    Y = to_decimals(df_ccy[TENORS].to_numpy(dtype=float))

    vas_model, r0s, S_all = optionA_fit_and_predict(Y, TENORS)

    print("\nOption A fitted params (single ccy sample):")
    print("  kappa:", float(vas_model.kappa.item()))
    print("  theta:", float(vas_model.theta.item()))
    print("  sigma:", float(vas_model.sigma.item()))

    rmse_total_bps, rmse_by_tenor = compute_rmse_bps(S_all, Y, TENORS)
    print("\nOption A RMSE (single ccy sample):")
    print(f"  Total RMSE (bps): {rmse_total_bps:.3f}")
    print("  RMSE by tenor (bps):")
    for t in TENORS:
        print(f"    {t:>2}Y: {rmse_by_tenor[t]:.3f}")

    # plot one date in the sample
    pick_idx = len(df_ccy) // 2
    row = df_ccy.iloc[pick_idx]
    S_obs = Y[pick_idx]
    S_pred = S_all[pick_idx]

    plt.figure()
    plt.plot(TENORS, S_obs, marker="o", label="Observed")
    plt.plot(TENORS, S_pred, marker="o", label="Vasicek fitted (Option A)")
    plt.title(f"Vasicek Option A fit (ccy={ccy}, date={pd.to_datetime(row['as_of_date']).date()})")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Par swap rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Option A: across currencies table + save CSV
    # -----------------------------
    print("\n=== Running Vasicek Option A across currencies (sample "
          f"{SAMPLE_PER_CCY} curves each) ===")

    results = []
    for ccy_i in sorted(df["ccy"].unique()):
        df_i = df[df["ccy"] == ccy_i].dropna(subset=TENORS).sort_values("as_of_date").reset_index(drop=True)
        if len(df_i) < SAMPLE_PER_CCY:
            continue

        df_sub = df_i.sample(SAMPLE_PER_CCY, random_state=0).sort_values("as_of_date").reset_index(drop=True)
        Y_i = to_decimals(df_sub[TENORS].to_numpy(dtype=float))

        vas_i, r0s_i, S_i = optionA_fit_and_predict(Y_i, TENORS)
        rmse_total_bps_i, rmse_by_tenor_i = compute_rmse_bps(S_i, Y_i, TENORS)

        results.append({
            "ccy": ccy_i,
            "n_curves": len(df_sub),
            "rmse_bps_total": rmse_total_bps_i,
            "kappa": float(vas_i.kappa.item()),
            "theta": float(vas_i.theta.item()),
            "sigma": float(vas_i.sigma.item()),
            **{f"rmse_{t}y_bps": float(rmse_by_tenor_i[t]) for t in TENORS}
        })

    res_df = pd.DataFrame(results).sort_values("rmse_bps_total")
    print("\n=== Vasicek Option A summary (sorted by total RMSE) ===")
    if len(res_df) == 0:
        print("No currencies had enough curves.")
    else:
        print(res_df[["ccy", "n_curves", "rmse_bps_total", "kappa", "theta", "sigma"]].to_string(index=False))

        out_path = os.path.join(SAVE_DIR, "vasicek_optionA_rmse_table.csv")
        res_df.to_csv(out_path, index=False)
        print("\nSaved:", out_path)

    print("\nNext: 2-factor Gaussian (2-factor Vasicek / G2++) baseline.")
