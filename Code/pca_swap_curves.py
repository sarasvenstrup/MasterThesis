# Code/pca_swap_curves.py
import os
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =========================
# Small helpers
# =========================

def rmse_bps(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE in basis points assuming inputs are DECIMAL rates."""
    e = a - b
    return float(np.sqrt(np.mean(e * e)) * 1e4)


def fit_pca_reconstruct(X: np.ndarray, k: int) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Fit PCA(k) on centered X and reconstruct.

    Returns
    -------
    pca : fitted sklearn PCA
    Z   : PCA scores (N,k)
    X_hat : reconstruction of original X (N,8)
    """
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    pca = PCA(n_components=k)
    Z = pca.fit_transform(Xc)
    Xc_hat = pca.inverse_transform(Z)
    X_hat = Xc_hat + X_mean
    return pca, Z, X_hat


def finite_row_mask(X: np.ndarray) -> np.ndarray:
    return np.isfinite(X).all(axis=1)


def currency_order_from_meta(meta: pd.DataFrame, preferred=None) -> list:
    if preferred is None:
        preferred = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
    present = sorted(meta["ccy"].unique())
    # keep preferred order first, then any extras
    in_pref = [c for c in preferred if c in present]
    extras = [c for c in present if c not in set(preferred)]
    return in_pref + extras


# =========================
# Core API
# =========================

def pooled_pca(
    meta: pd.DataFrame,
    X: np.ndarray,
    tenor_cols: Iterable[int],
    ks: Iterable[int] = (2, 3),
) -> pd.DataFrame:
    """
    One PCA fit on ALL curves (all currencies pooled).
    Returns a small summary table with one row per k.
    """
    tenor_cols = list(tenor_cols)

    rows = []
    for k in ks:
        pca, Z, X_hat = fit_pca_reconstruct(X, k)
        evr = pca.explained_variance_ratio_
        cum = float(evr.sum())
        overall = rmse_bps(X, X_hat)

        rows.append({
            "scope": "pooled",
            "ccy": "ALL",
            "n": int(X.shape[0]),
            "k": int(k),
            "pc1": float(evr[0]) if len(evr) > 0 else np.nan,
            "pc2": float(evr[1]) if len(evr) > 1 else np.nan,
            "pc3": float(evr[2]) if len(evr) > 2 else np.nan,
            "cumEV": cum,
            "rmse_bps": overall,
        })

    return pd.DataFrame(rows)


def per_currency_pca(
    meta: pd.DataFrame,
    X: np.ndarray,
    tenor_cols: Iterable[int],
    ks: Iterable[int] = (2, 3),
    preferred_ccy_order=None,
    min_rows: int = 25,
) -> pd.DataFrame:
    """
    Fit PCA separately for each currency.
    Returns one row per (ccy, k).
    """
    tenor_cols = list(tenor_cols)
    ccy_list = currency_order_from_meta(meta, preferred=preferred_ccy_order)

    rows = []
    for ccy in ccy_list:
        idx = meta.index[meta["ccy"] == ccy].to_numpy()
        Xc = X[idx]

        if Xc.shape[0] < max(min_rows, max(ks) + 5):
            # too few rows for stable PCA
            continue

        for k in ks:
            pca, Z, X_hat = fit_pca_reconstruct(Xc, k)
            evr = pca.explained_variance_ratio_
            cum = float(evr.sum())
            overall = rmse_bps(Xc, X_hat)

            rows.append({
                "scope": "per_currency",
                "ccy": ccy,
                "n": int(Xc.shape[0]),
                "k": int(k),
                "pc1": float(evr[0]) if len(evr) > 0 else np.nan,
                "pc2": float(evr[1]) if len(evr) > 1 else np.nan,
                "pc3": float(evr[2]) if len(evr) > 2 else np.nan,
                "cumEV": cum,
                "rmse_bps": overall,
            })

    return pd.DataFrame(rows).sort_values(["ccy", "k"]).reset_index(drop=True)


# =========================
# Plotting helpers (saved to disk)
# =========================

def save_pooled_plots(
    X: np.ndarray,
    tenor_cols: Iterable[int],
    out_dir: str,
    k: int,
):
    """
    Saves:
      explained_variance_k{k}.png
      loadings_k{k}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    tenor_cols = list(tenor_cols)

    pca, Z, X_hat = fit_pca_reconstruct(X, k)
    evr = pca.explained_variance_ratio_

    # explained variance
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=160)
    ax.plot(np.arange(1, k + 1), np.cumsum(evr), marker="o")
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"Pooled PCA explained variance — k={k}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"explained_variance_k{k}.png"))
    plt.close(fig)

    # loadings
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=160)
    for i in range(k):
        ax.plot(tenor_cols, pca.components_[i], marker="o", label=f"PC{i+1}")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Tenor (year)")
    ax.set_ylabel("Loading")
    ax.set_title(f"Pooled PCA loadings — k={k}")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"loadings_k{k}.png"))
    plt.close(fig)


def save_per_currency_loadings_plots(
    meta: pd.DataFrame,
    X: np.ndarray,
    tenor_cols: Iterable[int],
    out_dir: str,
    ks: Iterable[int] = (2, 3),
    preferred_ccy_order=None,
    min_rows: int = 25,
):
    """
    Saves:
      loadings_{CCY}_k2.png, loadings_{CCY}_k3.png, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    tenor_cols = list(tenor_cols)
    ccy_list = currency_order_from_meta(meta, preferred=preferred_ccy_order)

    for ccy in ccy_list:
        idx = meta.index[meta["ccy"] == ccy].to_numpy()
        Xc = X[idx]
        if Xc.shape[0] < max(min_rows, max(ks) + 5):
            continue

        for k in ks:
            pca, Z, X_hat = fit_pca_reconstruct(Xc, k)

            fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=160)
            for i in range(k):
                ax.plot(tenor_cols, pca.components_[i], marker="o", label=f"PC{i+1}")
            ax.axhline(0.0, linewidth=0.8)
            ax.set_xlabel("Tenor (year)")
            ax.set_ylabel("Loading")
            ax.set_title(f"{ccy} PCA loadings — k={k}")
            ax.legend(ncol=3, fontsize=8, frameon=False)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"loadings_{ccy}_k{k}.png"))
            plt.close(fig)