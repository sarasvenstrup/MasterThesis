import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import build_all_dataframes, TARGET_TENORS
from Code.utils.rates import par_swap_from_discount
from Code.utils.gaussian_fit import (
    to_decimals,
    compute_rmse_bps,
    fit_optionA_2f,
    fit_optionB_2f,
)

# -----------------------------
# Config
# -----------------------------
DEVICE = "cpu"
TENORS = list(TARGET_TENORS)
SAMPLE_PER_CCY = 200
SAVE_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

RUN_OPTION = "A"   # "A" (alternating) or "B" (encoder end-to-end)

# Option A hyperparams
OUTER_STEPS = 10
INNER_STEPS = 80
LR_PARAMS = 5e-2
LR_Z = 1e-1

# Option B hyperparams
N_STEPS = 4000
LR = 1e-3


if __name__ == "__main__":
    # -----------------------------
    # Load data
    # -----------------------------
    data = build_all_dataframes()
    df = data["df_wide_bbg_full"].copy().dropna(subset=TENORS)

    # -----------------------------
    # Pick one currency (most curves)
    # -----------------------------
    ccy = df["ccy"].value_counts().index[0]
    df_ccy_all = df[df["ccy"] == ccy].sort_values("as_of_date").reset_index(drop=True)
    print("Using currency:", ccy, " (#curves:", len(df_ccy_all), ")")

    if len(df_ccy_all) < SAMPLE_PER_CCY:
        raise ValueError(f"Not enough curves for {ccy}: {len(df_ccy_all)} < {SAMPLE_PER_CCY}")

    df_ccy = (
        df_ccy_all.sample(SAMPLE_PER_CCY, random_state=0)
        .sort_values("as_of_date")
        .reset_index(drop=True)
    )

    Y = to_decimals(df_ccy[TENORS].to_numpy(dtype=float))  # (N,K)

    # -----------------------------
    # Fit
    # -----------------------------
    if RUN_OPTION.upper() == "A":
        model, z, loss = fit_optionA_2f(
            Y, TENORS,
            outer_steps=OUTER_STEPS,
            inner_steps=INNER_STEPS,
            lr_params=LR_PARAMS,
            lr_z=LR_Z,
            device=DEVICE,
            print_every=1
        )
    elif RUN_OPTION.upper() == "B":
        model, encoder, z, loss = fit_optionB_2f(
            Y, TENORS,
            n_steps=N_STEPS,
            lr=LR,
            device=DEVICE,
            print_every=500
        )
    else:
        raise ValueError("RUN_OPTION must be 'A' or 'B'")

    print("\nFitted 2F params:")
    print("  kappa1:", float(model.kappa1.item()))
    print("  theta1:", float(model.theta1.item()))
    print("  sigma1:", float(model.sigma1.item()))
    print("  kappa2:", float(model.kappa2.item()))
    print("  theta2:", float(model.theta2.item()))
    print("  sigma2:", float(model.sigma2.item()))
    print("  final loss:", float(loss.item()))

    # -----------------------------
    # Predict + RMSE
    # -----------------------------
    with torch.no_grad():
        P_all = model.discount_curve_annual(z, T_max=max(TENORS))          # (N,T)
        S_all = par_swap_from_discount(P_all, TENORS).cpu().numpy()        # (N,K)

    rmse_total_bps, rmse_by_tenor = compute_rmse_bps(S_all, Y)
    print("\n2F Gaussian RMSE:")
    print(f"  Total RMSE (bps): {rmse_total_bps:.3f}")
    for t, v in zip(TENORS, rmse_by_tenor):
        print(f"    {t:>2}Y: {v:.3f}")

    # -----------------------------
    # Plot one representative curve
    # -----------------------------
    pick_idx = len(df_ccy) // 2
    row = df_ccy.iloc[pick_idx]
    S_obs = Y[pick_idx]
    S_pred = S_all[pick_idx]

    plt.figure()
    plt.plot(TENORS, S_obs, marker="o", label="Observed")
    plt.plot(TENORS, S_pred, marker="o", label=f"2F Gaussian (Option {RUN_OPTION.upper()})")
    plt.title(f"2F Gaussian fit (ccy={ccy}, date={pd.to_datetime(row['as_of_date']).date()})")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Par swap rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Optional: across currencies table + save
    # -----------------------------
    print("\n=== Running 2F Gaussian across currencies (sample "
          f"{SAMPLE_PER_CCY} curves each; Option {RUN_OPTION.upper()}) ===")

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

        if RUN_OPTION.upper() == "A":
            m_i, z_i, _ = fit_optionA_2f(
                Y_i, TENORS,
                outer_steps=OUTER_STEPS,
                inner_steps=INNER_STEPS,
                lr_params=LR_PARAMS,
                lr_z=LR_Z,
                device=DEVICE,
                print_every=0
            )
        else:
            m_i, enc_i, z_i, _ = fit_optionB_2f(
                Y_i, TENORS,
                n_steps=N_STEPS,
                lr=LR,
                device=DEVICE,
                print_every=0
            )

        with torch.no_grad():
            P_i = m_i.discount_curve_annual(z_i, T_max=max(TENORS))
            S_i = par_swap_from_discount(P_i, TENORS).cpu().numpy()

        rmse_total_i, rmse_by_tenor_i = compute_rmse_bps(S_i, Y_i)

        results.append({
            "ccy": ccy_i,
            "n_curves": len(df_sub),
            "rmse_bps_total": rmse_total_i,
            "kappa1": float(m_i.kappa1.item()),
            "theta1": float(m_i.theta1.item()),
            "sigma1": float(m_i.sigma1.item()),
            "kappa2": float(m_i.kappa2.item()),
            "theta2": float(m_i.theta2.item()),
            "sigma2": float(m_i.sigma2.item()),
            **{f"rmse_{t}y_bps": float(v) for t, v in zip(TENORS, rmse_by_tenor_i)}
        })

    res_df = pd.DataFrame(results).sort_values("rmse_bps_total")
    print("\n=== 2F Gaussian summary (sorted by total RMSE) ===")
    if len(res_df) == 0:
        print("No currencies had enough curves.")
    else:
        print(res_df[["ccy", "n_curves", "rmse_bps_total"]].to_string(index=False))

        out_path = os.path.join(SAVE_DIR, f"gaussian2f_option{RUN_OPTION.lower()}_rmse_table.csv")
        res_df.to_csv(out_path, index=False)
        print("\nSaved:", out_path)
