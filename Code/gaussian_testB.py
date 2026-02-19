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
from Code.utils.gaussian_fit import fit_optionB_2f

DEVICE = "cpu"
TENORS = list(TARGET_TENORS)

def to_decimals(Y):
    return Y/100.0 if np.nanmean(Y) > 1.0 else Y

if __name__ == "__main__":
    data = build_all_dataframes()
    df = data["df_wide_bbg_full"].copy().dropna(subset=TENORS)

    ccy = df["ccy"].value_counts().index[0]
    df_ccy = df[df["ccy"] == ccy].sort_values("as_of_date").reset_index(drop=True)

    df_sub = df_ccy.sample(200, random_state=0).sort_values("as_of_date").reset_index(drop=True)
    Y = to_decimals(df_sub[TENORS].to_numpy(dtype=float))

    g2_model, encoder, z, loss = fit_optionB_2f(
        Y, TENORS,
        n_steps=4000,
        lr=1e-3,
        device=DEVICE
    )

    with torch.no_grad():
        P_all = g2_model.discount_curve_annual(z, T_max=max(TENORS))
        S_all = par_swap_from_discount(P_all, TENORS).cpu().numpy()

    err = S_all - Y
    rmse_bps = 1e4 * np.sqrt(np.mean(err**2))
    print("\nOption B 2F Gaussian RMSE (bps):", rmse_bps)

    print("\nFitted params:")
    print("  kappa1:", float(g2_model.kappa1.item()))
    print("  theta1:", float(g2_model.theta1.item()))
    print("  sigma1:", float(g2_model.sigma1.item()))
    print("  kappa2:", float(g2_model.kappa2.item()))
    print("  theta2:", float(g2_model.theta2.item()))
    print("  sigma2:", float(g2_model.sigma2.item()))

    # plot one curve
    pick_idx = len(df_sub) // 2
    row = df_sub.iloc[pick_idx]
    plt.figure()
    plt.plot(TENORS, Y[pick_idx], marker="o", label="Observed")
    plt.plot(TENORS, S_all[pick_idx], marker="o", label="2F Gaussian fitted (Option B)")
    plt.title(f"2F Gaussian Option B fit (ccy={ccy}, date={pd.to_datetime(row['as_of_date']).date()})")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Par swap rate")
    plt.legend()
    plt.tight_layout()
    plt.show()
