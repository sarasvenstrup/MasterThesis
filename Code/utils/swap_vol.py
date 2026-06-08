"""EWMA realized volatility features computed from swap-rate panels."""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional

def _ewma_vol_bps_one_group(
    S_dec: np.ndarray,          # (T, d) decimals
    half_life_months: float,
    init_window: int = 12,
    eps: float = 1e-12,
    annualize: bool = False,
) -> np.ndarray:
    """
    Compute EWMA realized volatility per tenor for a single time-ordered currency group.

    Monthly changes in bps: x_t = 10000 * (S_t - S_{t-1})
    EWMA variance update:   v_t = lam * v_{t-1} + (1 - lam) * x_t^2

    Parameters
    ----------
    S_dec : np.ndarray, shape (T, d)
        Swap rates in decimal form, time-ordered.
    half_life_months : float
        EWMA half-life in months.
    init_window : int, default 12
        Number of initial changes used to seed the variance estimate.
    eps : float, default 1e-12
        Minimum variance clamp for numerical stability.
    annualize : bool, default False
        If True, multiplies volatility by sqrt(12) to annualize.

    Returns
    -------
    np.ndarray, shape (T, d)
        Realized volatility in bps per month (or annualized if annualize=True).
    """
    S_dec = np.asarray(S_dec, dtype=float)
    if S_dec.ndim != 2:
        raise ValueError(f"S_dec must be (T,d), got {S_dec.shape}")
    T, d = S_dec.shape
    if T == 0:
        return np.zeros((0, d), dtype=float)

    lam = 2.0 ** (-1.0 / float(half_life_months))

    # monthly changes in bps
    dS = np.full((T, d), np.nan, dtype=float)
    if T >= 2:
        dS[1:] = 10000.0 * (S_dec[1:] - S_dec[:-1])  # bps

    # init variance from first init_window changes (t=1..m)
    m = int(max(1, init_window))
    end = min(T, m + 1)
    init_slice = dS[1:end]  # (<=m, d)

    init_var = np.nanvar(init_slice, axis=0, ddof=1)
    init_var = np.where(np.isfinite(init_var), init_var, 0.0)
    init_var = np.maximum(init_var, eps)

    var = np.full((T, d), np.nan, dtype=float)
    var[0] = init_var

    for t in range(1, T):
        x = dS[t]          # (d,)
        prev = var[t - 1]  # (d,)
        new = lam * prev + (1.0 - lam) * np.where(np.isfinite(x), x * x, 0.0)
        new = np.where(np.isfinite(x), new, prev)  # carry forward where x is NaN
        var[t] = np.maximum(new, eps)

    V = np.sqrt(var)
    if annualize:
        V = V * np.sqrt(12.0)
    return V


def ewma_vol_panel_from_meta(
    meta: pd.DataFrame,
    X_tensor: torch.Tensor,          # (N, d) decimals
    half_life_months: float = 12.0,
    init_window: int = 12,
    annualize: bool = False,
) -> torch.Tensor:
    """
    Compute EWMA realized volatility for each row in a stacked multi-currency panel.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata with columns 'as_of_date' and 'ccy', aligned with X_tensor rows.
    X_tensor : torch.Tensor, shape (N, d)
        Swap rates in decimal form.
    half_life_months : float, default 12.0
        EWMA half-life in months.
    init_window : int, default 12
        Number of initial changes used to seed the variance estimate per currency.
    annualize : bool, default False
        If True, multiplies volatility by sqrt(12) to annualize.

    Returns
    -------
    torch.Tensor, shape (N, d)
        Realized volatility in bps per month (or annualized if annualize=True).
    """
    if not {"as_of_date", "ccy"}.issubset(meta.columns):
        raise ValueError("meta must contain columns ['as_of_date','ccy'].")

    X_np = X_tensor.detach().cpu().numpy().astype(float)  # decimals
    N, d = X_np.shape

    # sort by (ccy, date) to compute within each currency
    order = meta.sort_values(["ccy", "as_of_date"]).index.to_numpy()
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    meta_s = meta.loc[order].reset_index(drop=True)
    X_s = X_np[order]

    V_s = np.zeros((N, d), dtype=float)

    # compute per currency group
    for ccy, idx in meta_s.groupby("ccy").indices.items():
        idx = np.asarray(idx, dtype=int)
        V_s[idx] = _ewma_vol_bps_one_group(
            S_dec=X_s[idx],
            half_life_months=half_life_months,
            init_window=init_window,
            annualize=annualize,
        )

    # unsort back to original row order
    V = V_s[inv]
    return torch.from_numpy(V).to(dtype=X_tensor.dtype, device=X_tensor.device)


def stack_S_and_V(
    X_tensor: torch.Tensor,   # (N,d)
    V_tensor: torch.Tensor,   # (N,d)
) -> torch.Tensor:
    """Return [S|V] with shape (N, 2d)."""
    if X_tensor.shape != V_tensor.shape:
        raise ValueError(f"Shape mismatch: X {tuple(X_tensor.shape)} vs V {tuple(V_tensor.shape)}")
    return torch.cat([X_tensor, V_tensor], dim=1)