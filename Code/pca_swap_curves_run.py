# Code/pca_swap_curves_run.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# Repo-root detection
# ---------------------------------------------------------
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data, TARGET_TENORS


# =========================
# Helpers
# =========================

def rmse_bps(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) * 1e4)


def fit_pca_reconstruct(X: np.ndarray, k: int):
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    pca = PCA(n_components=k)
    Z = pca.fit_transform(Xc)
    Xc_hat = pca.inverse_transform(Z)
    X_hat = Xc_hat + X_mean
    return X_hat


# =========================
# Main comparison routine
# =========================

def main(USE="bbg"):

    meta, X_tensor, tenors, df_wide, SCALE_IS_PERCENT = my_data(use=USE)

    # X already decimals
    X = X_tensor.detach().cpu().numpy()

    # Remove non-finite rows
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    meta = meta.loc[mask].reset_index(drop=True)

    print(f"USE={USE}, total curves={X.shape[0]}")

    ccy_list = sorted(meta["ccy"].unique())

    rows = []

    for ccy in ccy_list:

        idx = meta.index[meta["ccy"] == ccy].to_numpy()
        Xc = X[idx]

        # ---- PCA k=2 ----
        X_hat_2 = fit_pca_reconstruct(Xc, k=2)
        rmse2 = rmse_bps(Xc, X_hat_2)

        # ---- PCA k=3 ----
        X_hat_3 = fit_pca_reconstruct(Xc, k=3)
        rmse3 = rmse_bps(Xc, X_hat_3)

        rows.append({
            "ccy": ccy,
            "RMSE_PCA_k2_bps": rmse2,
            "RMSE_PCA_k3_bps": rmse3,
            "n_obs": Xc.shape[0],
        })

    result = pd.DataFrame(rows).sort_values("ccy").reset_index(drop=True)

    print("\n=== PCA RMSE comparison (per currency) ===")
    print(result)

    OUT_DIR = os.path.join(REPO_ROOT, "Figures", USE, "PCA")
    os.makedirs(OUT_DIR, exist_ok=True)

    out_path = os.path.join(OUT_DIR, "pca_rmse_comparison.csv")
    result.to_csv(out_path, index=False)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main(USE="bbg")