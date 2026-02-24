import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import build_all_dataframes, TARGET_TENORS
from Code.Vasicek.vasicek_fit import fit_r0_single_curve, fit_global_params_and_r0s
from Code.utils.rates import par_swap_from_discount


def to_decimals(Y: np.ndarray) -> np.ndarray:
    return Y / 100.0 if np.nanmean(Y) > 1.0 else Y


def rmse_bps(S_pred: np.ndarray, S_obs: np.ndarray):
    err = (S_pred - S_obs)  # decimals
    total = 1e4 * float(np.sqrt(np.mean(err ** 2)))
    by_tenor = 1e4 * np.sqrt(np.mean(err ** 2, axis=0))
    return total, by_tenor


if __name__ == "__main__":
    DEVICE = "cpu"
    TENORS = list(TARGET_TENORS)
    SAMPLE_PER_CCY = 200

    SAVE_DIR = os.path.join(REPO_ROOT, "figures")
    os.makedirs(SAVE_DIR, exist_ok=True)

    data = build_all_dataframes()
    df = data["df_wide_bbg_full"].copy()

    # -----------------------------
    # 1) r0-only sanity on one curve
    # -----------------------------
    ccy = df["ccy"].value_counts().index[0]
    df_ccy_all = df[df["ccy"] == ccy].dropna(subset=TENORS).sort_values("as_of_date").reset_index(drop=True)

    pick_idx = len(df_ccy_all) // 2
    row = df_ccy_all.iloc[pick_idx]
    y = np.array([row[t] for t in TENORS], dtype=float)
    y = to_decimals(y)
    S_obs = torch.tensor(y, device=DEVICE, dtype=torch.float64)

    r0_hat, model_fixed, S_pred, loss = fit_r0_single_curve(
        S_obs, TENORS, kappa=0.5, theta=0.02, sigma=0.01,
        n_steps=800, lr=5e-2
    )
    print("\nSanity r0-only:")
    print("  ccy:", ccy, "date:", pd.to_datetime(row["as_of_date"]).date())
    print("  r0_hat:", float(r0_hat.item()), "MSE:", float(loss.item()))

    plt.figure()
    plt.plot(TENORS, S_obs.cpu().numpy(), marker="o", label="Observed")
    plt.plot(TENORS, S_pred.cpu().numpy(), marker="o", label="Vasicek (fit r0)")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Par swap rate")
    plt.title(f"Vasicek sanity (ccy={ccy}, date={pd.to_datetime(row['as_of_date']).date()})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 2) Option A on one currency
    # -----------------------------
    df_ccy = df_ccy_all.sample(SAMPLE_PER_CCY, random_state=0).sort_values("as_of_date").reset_index(drop=True)
    Y = to_decimals(df_ccy[TENORS].to_numpy(dtype=float))

    vas_model, r0s, _ = fit_global_params_and_r0s(
        Y, TENORS,
        outer_steps=10,
        inner_steps=80,
        lr_params=5e-2,
        lr_r0=1e-1,
        device=DEVICE,
        print_every=1
    )

    with torch.no_grad():
        P_all = vas_model.discount_curve_annual(r0s, T_max=max(TENORS))
        S_all = par_swap_from_discount(P_all, TENORS).cpu().numpy()

    total, by_tenor = rmse_bps(S_all, Y)
    print("\nOption A (single ccy) params:")
    print("  kappa:", float(vas_model.kappa.item()))
    print("  theta:", float(vas_model.theta.item()))
    print("  sigma:", float(vas_model.sigma.item()))
    print(f"  RMSE total (bps): {total:.3f}")
    for t, v in zip(TENORS, by_tenor):
        print(f"    {t:>2}Y: {v:.3f}")

    pick_idx = len(df_ccy) // 2
    plt.figure()
    plt.plot(TENORS, Y[pick_idx], marker="o", label="Observed")
    plt.plot(TENORS, S_all[pick_idx], marker="o", label="Vasicek (Option A)")
    plt.title(f"Vasicek Option A fit (ccy={ccy}, date={pd.to_datetime(df_ccy.iloc[pick_idx]['as_of_date']).date()})")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Par swap rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 3) Option A across currencies table
    # -----------------------------
    results = []
    for ccy_i in sorted(df["ccy"].unique()):
        df_i = df[df["ccy"] == ccy_i].dropna(subset=TENORS).sort_values("as_of_date").reset_index(drop=True)
        if len(df_i) < SAMPLE_PER_CCY:
            continue

        df_sub = df_i.sample(SAMPLE_PER_CCY, random_state=0).sort_values("as_of_date").reset_index(drop=True)
        Y_i = to_decimals(df_sub[TENORS].to_numpy(dtype=float))

        vas_i, r0s_i, _ = fit_global_params_and_r0s(
            Y_i, TENORS, outer_steps=10, inner_steps=80,
            lr_params=5e-2, lr_r0=1e-1, device=DEVICE, print_every=0
        )
        with torch.no_grad():
            P_i = vas_i.discount_curve_annual(r0s_i, T_max=max(TENORS))
            S_i = par_swap_from_discount(P_i, TENORS).cpu().numpy()

        total_i, by_tenor_i = rmse_bps(S_i, Y_i)
        results.append({
            "ccy": ccy_i,
            "n_curves": len(df_sub),
            "rmse_bps_total": total_i,
            "kappa": float(vas_i.kappa.item()),
            "theta": float(vas_i.theta.item()),
            "sigma": float(vas_i.sigma.item()),
            **{f"rmse_{t}y_bps": float(v) for t, v in zip(TENORS, by_tenor_i)}
        })

    res_df = pd.DataFrame(results).sort_values("rmse_bps_total")
    print("\n=== Vasicek Option A summary ===")
    if len(res_df) == 0:
        print("No currencies had enough curves.")
    else:
        print(res_df[["ccy","n_curves","rmse_bps_total","kappa","theta","sigma"]].to_string(index=False))
        out_path = os.path.join(SAVE_DIR, "vasicek_optionA_rmse_table.csv")
        res_df.to_csv(out_path, index=False)
        print("\nSaved:", out_path)
