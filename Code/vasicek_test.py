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
from Code.model.vasicek import Vasicek
from Code.utils.vasicek_fit import fit_r0_single_curve
from Code.utils.vasicek_fit import fit_optionA
from Code.utils.rates import par_swap_from_discount

if __name__ == "__main__":
    device = "cpu"
    tenors = list(TARGET_TENORS)  # [1,2,3,5,10,15,20,30]

    data = build_all_dataframes()
    df = data["df_wide_bbg_full"].copy()  # or df_wide_test_full

    # --- pick one currency with many curves ---
    ccy_counts = df["ccy"].value_counts()
    ccy = ccy_counts.index[0]
    df_ccy = df[df["ccy"] == ccy].sort_values("as_of_date").reset_index(drop=True)

    # --- pick one date (middle-ish so it's not extreme) ---
    pick_idx = len(df_ccy) // 2
    row = df_ccy.iloc[pick_idx]

    # observed swaps (likely in percent -> convert to decimals if needed)
    y = np.array([row[t] for t in tenors], dtype=float)

    # Heuristic: if average > 1, it's probably percent (e.g. 4.5), convert to decimal.
    if np.nanmean(y) > 1.0:
        y = y / 100.0

    S_obs = torch.tensor(y, device=device, dtype=torch.float64)

    print("Picked curve:")
    print("  ccy:", ccy)
    print("  date:", pd.to_datetime(row["as_of_date"]).date())
    print("  observed (decimals):", S_obs.cpu().numpy())

    # Fixed Vasicek params for the test
    kappa0, theta0, sigma0 = 0.5, 0.02, 0.01

    r0_hat, model, S_pred, loss = fit_r0_single_curve(
        S_obs, tenors, kappa=kappa0, theta=theta0, sigma=sigma0,
        n_steps=800, lr=5e-2
    )

    print("\nFit results:")
    print("  r0_hat:", float(r0_hat.item()))
    print("  loss (MSE):", float(loss.item()))
    print("  params used:", "kappa", float(model.kappa.item()),
          "theta", float(model.theta.item()),
          "sigma", float(model.sigma.item()))

    # Plot obs vs fitted
    plt.figure()
    plt.plot(tenors, S_obs.cpu().numpy(), marker="o", label="Observed")
    plt.plot(tenors, S_pred.cpu().numpy(), marker="o", label="Vasicek fitted (r0 only)")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Par swap rate")
    plt.title(f"One-curve Vasicek fit (ccy={ccy}, date={pd.to_datetime(row['as_of_date']).date()})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # pick a single currency
    df = data["df_wide_bbg_full"].copy()
    ccy = df["ccy"].value_counts().index[0]
    df_ccy = df[df["ccy"] == ccy].sort_values("as_of_date").reset_index(drop=True)

    df_ccy = df_ccy.sample(200, random_state=0).sort_values("as_of_date").reset_index(drop=True)

    tenors = list(TARGET_TENORS)
    Y = df_ccy[tenors].to_numpy(dtype=float)
    if np.nanmean(Y) > 1.0:
        Y = Y / 100.0

    # run option A calibration (start with smaller settings to test)
    vas_model, r0s = fit_optionA(
        Y, tenors,
        outer_steps=10,
        inner_steps=80,
        lr_params=5e-2,
        lr_r0=1e-1,
        device="cpu"
    )

    print("Fitted params:")
    print("  kappa:", float(vas_model.kappa.item()))
    print("  theta:", float(vas_model.theta.item()))
    print("  sigma:", float(vas_model.sigma.item()))

    # plot the same middle date again
    pick_idx = len(df_ccy) // 2
    row = df_ccy.iloc[pick_idx]
    S_obs = Y[pick_idx]  # (8,)

    with torch.no_grad():
        T_max = max(tenors)
        P = vas_model.discount_curve_annual(r0s[pick_idx], T_max=T_max)  # (1,Tmax)
        S_pred = par_swap_from_discount(P, tenors).squeeze(0).cpu().numpy()

    plt.figure()
    plt.plot(tenors, S_obs, marker="o", label="Observed")
    plt.plot(tenors, S_pred, marker="o", label="Vasicek fitted (Option A)")
    plt.title(f"Vasicek Option A fit (ccy={ccy}, date={pd.to_datetime(row['as_of_date']).date()})")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Par swap rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


