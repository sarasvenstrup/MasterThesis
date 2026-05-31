"""
Swaption pricing and analysis module.

This module provides both differentiable (PyTorch) and non-differentiable (NumPy)
implementations for swaption pricing and related calculations.

Shared core
    - _swap_from_payment_dfs: forward swap rate + annuity from payment-date
      discount factors. Accepts np.ndarray or torch.Tensor. All other
      forward-swap helpers are thin wrappers around this.

Torch-based functions (differentiable, for calibration)
    - bachelier_price_torch: Bachelier formula with autograd support
    - swap_rate_torch: Swap rate from discount curve (batched)

NumPy-based functions (for Monte Carlo and analysis)
    - bachelier_price: Standard Bachelier pricing
    - bachelier_greeks: Analytic Greeks (delta, vega, gamma, etc.)
    - implied_bachelier_vol: Implied volatility inversion
    - time0_forward_swap_and_annuity: Forward F0, A0 from a time-0 curve
    - swap_from_discount_curve_at_expiry: F(Te), A(Te) from one path slice
    - swaption_payoff_from_simulation: Vectorised pathwise payoff at expiry
    - swaption_mc_price_from_simulation: Vectorised MC pricer (single leg)
    - atm_swaption_mc_price_from_simulation: ATM convenience wrapper
    - atm_swaption_straddle_mc_price_from_simulation: Single-pass dual-leg
      ATM straddle pricer + forward-centring-bias diagnostic.

The separation allows gradient-based optimization (Torch) while maintaining
efficient, well-tested implementations for simulation and risk analysis (NumPy).
"""
import os
import sys

import math
import warnings
from typing import Union, Optional
from scipy.stats import norm
from scipy.optimize import brentq

import numpy as np
import torch

try:
    CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CODE_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT, "..", ".."))

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from Code.Simulation.simulate_model import run_simulation


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_grid_index_for_value(
    grid: Union[np.ndarray, list, torch.Tensor],
    value: float,
    tol: float = 1e-10
) -> int:
    """
    Find the index of a value in a grid (tau or time grid).
    
    Parameters
    ----------
    grid : array-like
        1-D array of grid points (e.g., tau values or time points)
    value : float
        Value to locate in the grid
    tol : float, default 1e-10
        Maximum allowed distance between value and grid point
    
    Returns
    -------
    int
        Index of the grid point closest to value
    
    Raises
    ------
    ValueError
        If no grid point is within tolerance of value
    
    Examples
    --------
    >>> grid = np.array([0, 1, 2, 5, 10])
    >>> get_grid_index_for_value(grid, 2.0)
    2
    >>> get_grid_index_for_value(grid, 2.001)  # Within default tol
    2
    >>> get_grid_index_for_value(grid, 3.0)  # Raises ValueError
    """
    grid = np.asarray(grid, dtype=float)
    diffs = np.abs(grid - float(value))
    idx = int(np.argmin(diffs))
    if diffs[idx] > tol:
        raise ValueError(
            f"Value {value} not found in grid within tol={tol}. "
            f"Closest value is {grid[idx]}."
        )
    return idx


# =============================================================================
# SHARED CORE: swap rate + annuity from a set of payment-date discount factors
# =============================================================================

# Loose physical upper bound on a single-period discount factor. In any
# realistic EUR/USD term structure P(tau) is at most slightly above 1
# (negative-rate environments). Anything beyond this is a decoder
# extrapolation pathology that would otherwise dominate the MC mean: one
# surviving path with P ~ 1e200 makes the average meaningless.
# Applied at the PAYMENT-DATE slice only, so a path that's weird at
# tau=20 can still price a 1x5 swaption.
P_MAX_PHYSICAL = 10.0


def _physically_valid(payment_dfs) -> np.ndarray:
    """
    Per-path validity mask for an (n_paths, tenor) array of payment-date
    discount factors. A path is valid iff every payment-date discount
    factor is finite, strictly positive, and not larger than
    :data:`P_MAX_PHYSICAL`.
    """
    arr = np.asarray(payment_dfs)
    return (
        np.isfinite(arr).all(axis=-1)
        & (arr > 0.0).all(axis=-1)
        & (arr <= P_MAX_PHYSICAL).all(axis=-1)
    )


def _swap_from_payment_dfs(payment_dfs, accrual: float = 1.0):
    """
    Core 3-line swap-rate computation, shared by every torch/numpy/batched
    wrapper in this module.

    Given the discount factors at the fixed-leg payment dates, returns:

        A = accrual * sum(payment_dfs[..., :])
        F = (1 - payment_dfs[..., -1]) / A

    Accepts either ``np.ndarray`` or ``torch.Tensor`` on the last axis.
    Preserves dtype/device and (for torch) autograd connectivity of the
    input ``payment_dfs``.

    Parameters
    ----------
    payment_dfs : np.ndarray or torch.Tensor
        Shape (..., tenor) — discount factors at payment dates.
    accrual : float
        Day-count / payment frequency.

    Returns
    -------
    F, A : same type as ``payment_dfs``
        Forward (or spot) swap rate and annuity.
    """
    if isinstance(payment_dfs, torch.Tensor):
        A = accrual * payment_dfs.sum(dim=-1)
        F = (1.0 - payment_dfs[..., -1]) / A
        return F, A
    arr = np.asarray(payment_dfs, dtype=float)
    A = float(accrual) * arr.sum(axis=-1)
    F = (1.0 - arr[..., -1]) / A
    return F, A


# =============================================================================
# DIFFERENTIABLE BACHELIER PRICE  (torch-native, grad-compatible)
# =============================================================================

def bachelier_price_torch(
    F: Union[float, torch.Tensor],
    K: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
    expiry: float,
    A: Union[float, torch.Tensor],
    notional: float = 1.0,
    payer: bool = True
) -> Union[float, torch.Tensor]:
    """
    Analytic Bachelier (normal-model) swaption price with autograd support.

    Payer swaption:
        V = notional * A * [ (F-K)*Φ(d) + σ√T * φ(d) ]
    
    Receiver swaption:
        V = notional * A * [ (K-F)*Φ(-d) + σ√T * φ(d) ]
    
    where:
        d = (F - K) / (σ√T)

    Works for scalar Python floats OR 0-d/scalar torch tensors.
    When any of F, K, sigma, A is a torch.Tensor the result is a tensor
    and gradients flow through it normally.

    For ATM (K = F):
        V = notional * A * σ * √T / √(2π)

    Parameters
    ----------
    F : float or torch.Tensor
        Forward swap rate
    K : float or torch.Tensor
        Strike rate
    sigma : float or torch.Tensor
        Normal (Bachelier) volatility
    expiry : float
        Time to expiry in years
    A : float or torch.Tensor
        Annuity factor
    notional : float, default 1.0
        Notional amount
    payer : bool, default True
        If True, prices payer swaption; if False, prices receiver

    Returns
    -------
    float or torch.Tensor
        Swaption price. Type matches input (Tensor if any input is Tensor).

    Notes
    -----
    The Python-float path uses scipy for CDF/PDF, suitable for computing
    fixed market price targets outside any autograd context.
    The Torch path is fully differentiable via torch.erf.
    """
    T_sqrt = math.sqrt(max(float(expiry), 1e-12))

    if any(isinstance(x, torch.Tensor) for x in (F, K, sigma, A)):
        # Torch path — fully differentiable via torch.erf
        vol_term = sigma * T_sqrt
        d   = (F - K) / vol_term
        
        if payer:
            Phi = 0.5 * (1.0 + torch.erf(d / math.sqrt(2.0)))
            phi = torch.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
            return notional * A * ((F - K) * Phi + vol_term * phi)
        else:
            Phi_neg = 0.5 * (1.0 + torch.erf(-d / math.sqrt(2.0)))
            phi = torch.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
            return notional * A * ((K - F) * Phi_neg + vol_term * phi)
    else:
        # Pure-Python path — no grad, uses scipy
        vol_term = max(float(sigma) * T_sqrt, 1e-12)
        d   = (float(F) - float(K)) / vol_term
        phi = norm.pdf(d)
        
        if payer:
            Phi = norm.cdf(d)
            return notional * float(A) * ((float(F) - float(K)) * Phi + vol_term * phi)
        else:
            Phi_neg = norm.cdf(-d)
            return notional * float(A) * ((float(K) - float(F)) * Phi_neg + vol_term * phi)


# =============================================================================
# DIFFERENTIABLE SWAP RATE FROM BOND PRICE CURVE  (torch-native)
# =============================================================================

def swap_rate_torch(
    P_full : torch.Tensor,    # (n_paths, tau_max+1)
    tenor  : int,
    accrual: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable ATM spot-starting swap rate F and annuity A from a full
    discount curve ``P_full`` whose first index is the as-of time slice.

    Thin wrapper around :func:`_swap_from_payment_dfs`.

    Returns
    -------
    F : (n_paths,)  forward swap rates — WITH grad w.r.t. upstream params
    A : (n_paths,)  annuities
    """
    pay_idx     = [int(round(accrual * j)) for j in range(1, tenor + 1)]
    payment_dfs = P_full[:, pay_idx]
    return _swap_from_payment_dfs(payment_dfs, accrual=accrual)


def extract_discount(
    P_full_0: Union[np.ndarray, torch.Tensor],
    tau_grid: Union[np.ndarray, torch.Tensor],
    tau: float
) -> float:
    """
    Extract discount factor P(0, tau) from a discount curve.
    
    Parameters
    ----------
    P_full_0 : array-like, shape (tau_max+1,) or (1, tau_max+1)
        Time-0 discount curve
    tau_grid : array-like, shape (tau_max+1,)
        Maturity grid corresponding to P_full_0
    tau : float
        Maturity for which to extract discount factor
    
    Returns
    -------
    float
        Discount factor P(0, tau)
    
    Raises
    ------
    ValueError
        If tau is not found in tau_grid
    
    Notes
    -----
    If P_full_0 is 2-D (e.g., from batched simulation), uses first row.
    """
    P = np.asarray(P_full_0, dtype=float)
    if P.ndim == 2:
        P = P[0]
    idx = get_grid_index_for_value(tau_grid, tau)
    return float(P[idx])


def time0_forward_swap_and_annuity(
    P_full_0: Union[np.ndarray, torch.Tensor],
    tau_grid: Union[np.ndarray, torch.Tensor],
    expiry: float,
    tenor: int,
    accrual: float = 1.0
) -> dict:
    """
    Compute forward swap rate and annuity from time-0 discount curve.
    
    This is the NumPy-based version for analysis and validation.
    For differentiable computation, use :func:`swap_rate_torch` (slice the
    time-0 curve at the appropriate offset) or call
    :func:`_swap_from_payment_dfs` directly with a torch tensor.
    
    Parameters
    ----------
    P_full_0 : array-like
        Time-0 discount curve
    tau_grid : array-like
        Maturity grid corresponding to P_full_0
    expiry : float
        Swaption expiry in years
    tenor : int
        Swap tenor in years
    accrual : float, default 1.0
        Payment frequency in years
    
    Returns
    -------
    dict
        Dictionary containing:
        - forward_swap : float - Forward swap rate
        - annuity : float - Forward annuity
        - payment_times : list - Payment dates
        - payment_dfs : np.ndarray - Discount factors at payment dates
        - P_start : float - Discount to expiry
        - P_end : float - Discount to final payment
    
    Raises
    ------
    ValueError
        If expiry < 0, tenor <= 0, or accrual <= 0
    """
    if expiry < 0:
        raise ValueError("expiry must be non-negative")
    if tenor <= 0:
        raise ValueError("tenor must be positive")
    if accrual <= 0:
        raise ValueError("accrual must be positive")

    payment_times = [expiry + accrual * j for j in range(1, tenor + 1)]
    P_payments = np.array(
        [extract_discount(P_full_0, tau_grid, t) for t in payment_times],
        dtype=float,
    )

    F0, A0 = _swap_from_payment_dfs(P_payments, accrual=accrual)
    F0, A0 = float(F0), float(A0)

    if expiry > 0:
        P_start = extract_discount(P_full_0, tau_grid, expiry)
    else:
        P_start = 1.0

    P_end = float(P_payments[-1])
    # Forward swap rate uses P_start (not 1) as numerator
    F0 = (P_start - P_end) / A0

    return {
        "forward_swap": F0,
        "annuity": A0,
        "payment_times": payment_times,
        "payment_dfs": P_payments,
        "P_start": P_start,
        "P_end": P_end,
    }


def quote_swaption_time0(
    ctx: dict,
    expiry: float,
    tenor: int,
    strike: Optional[float] = None,
    strike_atm: bool = False,
    payer: bool = True,
    accrual: float = 1.0,
    verbose: bool = False
) -> dict:
    """
    Generate time-0 swaption quote from simulation context.
    
    Parameters
    ----------
    ctx : dict
        Simulation context containing P_full_0 and tau_grid
    expiry : float
        Swaption expiry in years
    tenor : int
        Swap tenor in years
    strike : float, optional
        Strike rate. If None, must set strike_atm=True
    strike_atm : bool, default False
        If True, uses forward swap rate as strike
    payer : bool, default True
        If True, payer swaption; if False, receiver
    accrual : float, default 1.0
        Payment frequency in years
    verbose : bool, default False
        If True, prints detailed quote information
    
    Returns
    -------
    dict
        Dictionary containing:
        - expiry : float
        - tenor : int
        - payer : bool
        - forward_swap : float
        - annuity : float
        - strike : float
        - intrinsic_lower_bound : float
    
    Raises
    ------
    ValueError
        If neither strike nor strike_atm is provided
    """
    P_full_0 = ctx["P_full_0"].detach().cpu().numpy()
    tau_grid = ctx["tau_grid"].detach().cpu().numpy()

    q = time0_forward_swap_and_annuity(
        P_full_0=P_full_0,
        tau_grid=tau_grid,
        expiry=expiry,
        tenor=tenor,
        accrual=accrual,
    )

    if strike_atm:
        strike = q["forward_swap"]
    elif strike is None:
        raise ValueError("Either set strike or strike_atm=True.")

    intrinsic_lb = q["annuity"] * (
        max(q["forward_swap"] - strike, 0.0) if payer else max(strike - q["forward_swap"], 0.0)
    )

    out = {
        "expiry": expiry,
        "tenor": tenor,
        "payer": payer,
        "forward_swap": q["forward_swap"],
        "annuity": q["annuity"],
        "strike": strike,
        "intrinsic_lower_bound": intrinsic_lb,
    }

    if verbose:
        print("\nTime-0 swaption quote")
        print(f"  Expiry            : {expiry}Y")
        print(f"  Tenor             : {tenor}Y")
        print(f"  Forward swap F0   : {out['forward_swap']:.6f}  ({out['forward_swap']*10000:.1f} bp)")
        print(f"  Annuity A0        : {out['annuity']:.10f}")
        print(f"  Strike K          : {out['strike']:.10f}")
        print(f"  Intrinsic lb      : {out['intrinsic_lower_bound']:.10f}")

    return out


def swap_from_discount_curve_at_expiry(
    P_curve: Union[np.ndarray, list],
    tau_grid: Union[np.ndarray, list],
    tenor: int,
    accrual: float = 1.0
) -> dict:
    """
    Compute annuity and spot-starting swap rate from discount curve at expiry.
    
    Given a discount curve P(Te, Te+tau) at expiry time Te, computes the
    annuity and swap rate for a swap starting at Te with given tenor.
    
    Parameters
    ----------
    P_curve : array-like
        Discount curve at expiry: P(Te, Te+tau) for tau in tau_grid
    tau_grid : array-like
        Grid of maturities corresponding to P_curve
    tenor : int
        Swap tenor in years
    accrual : float, default 1.0
        Payment frequency in years
    
    Returns
    -------
    dict
        Dictionary containing:
        - annuity : float
        - swap_rate : float
        - payment_taus : list
        - payment_dfs : np.ndarray
        - P_end : float
    
    Raises
    ------
    ValueError
        If tenor <= 0, accrual <= 0, or annuity is non-positive
    """
    P_curve = np.asarray(P_curve, dtype=float).reshape(-1)
    tau_grid = np.asarray(tau_grid, dtype=float).reshape(-1)

    if tenor <= 0:
        raise ValueError("tenor must be positive")
    if accrual <= 0:
        raise ValueError("accrual must be positive")

    payment_taus = [accrual * j for j in range(1, tenor + 1)]
    tau_indices = [get_grid_index_for_value(tau_grid, tau) for tau in payment_taus]

    payment_dfs = np.array([P_curve[idx] for idx in tau_indices], dtype=float)

    # Reject paths whose payment discount factors are non-finite (decoder
    # overflow → inf/nan). No upper bound: P > 1 is physical in negative-rate
    # environments and arbitrage is not enforced here. Survival rate is
    # reported by callers via `valid_mask`.
    if not np.all(np.isfinite(payment_dfs)):
        raise ValueError("Non-finite discount factor in payment_dfs")

    swap_rate, annuity = _swap_from_payment_dfs(payment_dfs, accrual=accrual)
    swap_rate, annuity = float(swap_rate), float(annuity)
    if annuity <= 0.0:
        raise ValueError(f"Non-positive annuity encountered: {annuity}")

    return {
        "annuity": annuity,
        "swap_rate": swap_rate,
        "payment_taus": payment_taus,
        "payment_dfs": payment_dfs,
        "P_end": float(payment_dfs[-1]),
    }


def swaption_payoff_from_simulation(
    ctx: dict,
    expiry: float,
    tenor: int,
    strike: float,
    payer: bool = True,
    accrual: float = 1.0,
    notional: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    Compute pathwise swaption payoff at expiry, without discounting back to time 0.
    
    Parameters
    ----------
    ctx : dict
        Simulation context from run_simulation()
    expiry : float
        Swaption expiry in years
    tenor : int
        Swap tenor in years
    strike : float
        Strike rate
    payer : bool, default True
        If True, payer swaption; if False, receiver
    accrual : float, default 1.0
        Payment frequency in years
    notional : float, default 1.0
        Notional amount
    verbose : bool, default False
        If True, prints diagnostic information
    
    Returns
    -------
    dict
        Dictionary containing:
        - expiry : float
        - tenor : int
        - expiry_index : int
        - strike : float
        - payer : bool
        - annuity_paths : np.ndarray
        - swap_rate_paths : np.ndarray
        - payoff_paths : np.ndarray
        - valid_mask : np.ndarray (bool)
        - mean_payoff : float
    
    Raises
    ------
    ValueError
        If tenor <= 0, accrual <= 0, or notional <= 0
    RuntimeError
        If no valid paths are found
    """
    if tenor <= 0:
        raise ValueError("tenor must be positive")
    if accrual <= 0:
        raise ValueError("accrual must be positive")
    if notional <= 0:
        raise ValueError("notional must be positive")

    times = np.asarray(ctx["times"], dtype=float)
    tau_grid = ctx["tau_grid"].detach().cpu().numpy()
    P_full_paths = ctx["P_full_paths"].detach().cpu().numpy()

    exp_idx = get_grid_index_for_value(times, expiry)

    n_paths = P_full_paths.shape[0]

    # --- Vectorised pathwise annuity / swap-rate / payoff ----------------
    # Replaces the per-path Python loop: lookup payment-date column indices
    # once, slice in one shot, then broadcast.
    payment_taus = [accrual * j for j in range(1, tenor + 1)]
    tau_indices = np.array(
        [get_grid_index_for_value(tau_grid, t) for t in payment_taus],
        dtype=int,
    )

    payment_dfs = P_full_paths[:, exp_idx, tau_indices]              # (n_paths, tenor)
    # Conservative physical-validity gate: reject any path whose decoded
    # curve at expiry has ANY unphysical discount factor (not just at the
    # payment dates). A path with P(tau=10)=1e100 indicates the latent
    # state took an unphysical excursion that contaminates P(tau=1) too,
    # even if that single tau happens to look finite.
    curve_at_expiry = P_full_paths[:, exp_idx, :]                    # (n_paths, tau_max+1)
    finite_pay  = _physically_valid(curve_at_expiry)

    swap_rate_paths, annuity_paths = _swap_from_payment_dfs(payment_dfs, accrual=accrual)
    # Reject non-finite / non-positive annuities so downstream stats stay clean
    pos_annuity = np.isfinite(annuity_paths) & (annuity_paths > 0.0)
    valid_swap  = finite_pay & pos_annuity & np.isfinite(swap_rate_paths)

    swap_rate_paths = np.where(valid_swap, swap_rate_paths, np.nan)
    annuity_paths   = np.where(valid_swap, annuity_paths,   np.nan)

    if payer:
        intrinsic = np.maximum(swap_rate_paths - strike, 0.0)
    else:
        intrinsic = np.maximum(strike - swap_rate_paths, 0.0)
    payoff_paths = notional * annuity_paths * intrinsic

    valid_mask = (
        valid_swap
        & np.isfinite(payoff_paths)
    )
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise RuntimeError("No valid paths for swaption payoff computation.")

    mean_payoff = float(np.mean(payoff_paths[valid_mask]))

    if verbose:
        print("\nSwaption payoff from simulation")
        print(f"  Expiry            : {expiry}Y")
        print(f"  Tenor             : {tenor}Y")
        print(f"  Strike            : {strike:.10f}")
        print(f"  Payer             : {payer}")
        print(f"  Valid paths       : {n_valid}/{n_paths}")
        print(f"  Mean payoff @ Te  : {mean_payoff:.10f}")

    return {
        "expiry": expiry,
        "tenor": tenor,
        "expiry_index": exp_idx,
        "strike": strike,
        "payer": payer,
        "annuity_paths": annuity_paths,
        "swap_rate_paths": swap_rate_paths,
        "payoff_paths": payoff_paths,
        "valid_mask": valid_mask,
        "mean_payoff": mean_payoff,
    }


def swaption_mc_price_from_simulation(
    ctx: dict,
    expiry: float,
    tenor: int,
    strike: float,
    payer: bool = True,
    accrual: float = 1.0,
    notional: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    Full Monte Carlo time-0 swaption price:
      1) compute payoff at expiry pathwise
      2) discount each payoff back to time 0
      3) average across valid paths
    
    Parameters
    ----------
    ctx : dict
        Simulation context from run_simulation()
    expiry : float
        Swaption expiry in years
    tenor : int
        Swap tenor in years
    strike : float
        Strike rate
    payer : bool, default True
        If True, payer swaption; if False, receiver
    accrual : float, default 1.0
        Payment frequency in years
    notional : float, default 1.0
        Notional amount
    verbose : bool, default False
        If True, prints diagnostic information
    
    Returns
    -------
    dict
        Dictionary containing all fields from swaption_payoff_from_simulation plus:
        - discount_to_expiry_paths : np.ndarray
        - pv_paths : np.ndarray
        - mc_price : float
        - mc_std : float
        - mc_stderr : float
        - valid_mask : np.ndarray (bool) - updated to include discount validity
    
    Raises
    ------
    RuntimeError
        If no valid paths are found for pricing
    """
    payoff_res = swaption_payoff_from_simulation(
        ctx=ctx,
        expiry=expiry,
        tenor=tenor,
        strike=strike,
        payer=payer,
        accrual=accrual,
        notional=notional,
        verbose=verbose,
    )

    exp_idx = payoff_res["expiry_index"]
    payoff_paths = payoff_res["payoff_paths"]
    valid_mask = payoff_res["valid_mask"]

    # NOTE: ctx["discount_paths"] is on the FULL simulation grid
    # (n_paths, n_steps+1), while exp_idx is an index into the (possibly
    # downsampled) ctx["times"] array used by decode_steps. We must map the
    # expiry to its position on the FULL time grid before slicing the
    # discount factor, otherwise (e.g.) a 5Y expiry would be discounted
    # with D(0, 2 months) when decode_steps = [0, 12, 60, 120].
    discount_paths = ctx["discount_paths"].detach().cpu().numpy()
    n_full_steps = discount_paths.shape[1] - 1
    t_max = float(np.asarray(ctx["times"], dtype=float).max())
    dt_full = t_max / max(n_full_steps, 1)
    full_exp_idx = int(round(float(expiry) / dt_full))
    if not (0 <= full_exp_idx < discount_paths.shape[1]):
        raise ValueError(
            f"Expiry {expiry}Y maps to full-grid index {full_exp_idx} "
            f"outside [0, {discount_paths.shape[1]-1}]"
        )
    discount_to_expiry = np.asarray(discount_paths[:, full_exp_idx], dtype=float)

    pv_paths = discount_to_expiry * payoff_paths

    # Drop only non-finite paths (overflow). No upper bound on D(0,Te) —
    # negative integrated short rates are physical in EUR 2014--2022.
    valid_mask = (
        valid_mask
        & np.isfinite(discount_to_expiry)
        & np.isfinite(pv_paths)
    )
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise RuntimeError("No valid paths for Monte Carlo pricing.")

    mc_price = float(np.mean(pv_paths[valid_mask]))
    mc_std = float(np.std(pv_paths[valid_mask], ddof=1)) if n_valid > 1 else 0.0
    mc_se = mc_std / np.sqrt(n_valid)

    if verbose:
        print("\nMonte Carlo swaption price")
        print(f"  Expiry            : {expiry}Y")
        print(f"  Tenor             : {tenor}Y")
        print(f"  Strike            : {strike:.10f}")
        print(f"  Valid paths       : {n_valid}/{len(payoff_paths)}")
        print(f"  MC price          : {mc_price:.10f}")
        print(f"  MC std            : {mc_std:.10f}")
        print(f"  MC stderr         : {mc_se:.10f}")

    return {
        **payoff_res,
        "discount_to_expiry_paths": discount_to_expiry,
        "pv_paths": pv_paths,
        "mc_price": mc_price,
        "mc_std": mc_std,
        "mc_stderr": mc_se,
        "valid_mask": valid_mask,
    }


def bachelier_price(
    forward: float,
    strike: float,
    normal_vol: float,
    expiry: float,
    annuity: float,
    notional: float = 1.0,
    payer: bool = True
) -> float:
    """
    NumPy-based Bachelier (normal-model) swaption price.
    
    This is the non-differentiable version for Monte Carlo and analysis.
    For gradient-based calibration, use bachelier_price_torch instead.
    
    Parameters
    ----------
    forward : float
        Forward swap rate
    strike : float
        Strike rate
    normal_vol : float
        Normal (Bachelier) volatility
    expiry : float
        Time to expiry in years
    annuity : float
        Annuity factor
    notional : float, default 1.0
        Notional amount
    payer : bool, default True
        If True, payer swaption; if False, receiver
    
    Returns
    -------
    float
        Swaption price
    
    Notes
    -----
    Returns intrinsic value if expiry <= 0 or normal_vol <= 0.
    """
    intrinsic = max(forward - strike, 0.0) if payer else max(strike - forward, 0.0)

    if expiry <= 0.0 or normal_vol <= 0.0:
        return notional * annuity * intrinsic

    vol_term = normal_vol * math.sqrt(expiry)
    if vol_term < 1e-16:
        return notional * annuity * intrinsic

    d = (forward - strike) / vol_term

    if payer:
        return notional * annuity * ((forward - strike) * norm.cdf(d) + vol_term * norm.pdf(d))
    else:
        return notional * annuity * ((strike - forward) * norm.cdf(-d) + vol_term * norm.pdf(d))


def bachelier_greeks(
    forward,
    strike,
    normal_vol,
    expiry,
    annuity,
    notional: float = 1.0,
    payer: bool = True,
) -> dict:
    """
    Analytic Bachelier (normal-model) Greeks for a European payer or receiver
    swaption.

    The pricing formula is (payer):
        V = N · A · [(F − K) · Φ(d) + σ√T · φ(d)]
        d = (F − K) / (σ√T)

    Returns
    -------
    dict with keys:
        delta  : ∂V/∂F   — rate sensitivity
        vega   : ∂V/∂σ   — normal-vol sensitivity  (per unit of σ, i.e. absolute)
        gamma  : ∂²V/∂F² — convexity
        theta  : ∂V/∂T   — time value per unit of time-to-expiry (positive)
                           Note: conventional "time decay" = −∂V/∂t = +∂V/∂T.
        dv01   : ∂V/∂F · (1 bp) — dollar value of 1 basis point shift in F
        vanna  : ∂²V/(∂F ∂σ) — delta sensitivity to vol
        volga  : ∂²V/∂σ²     — vega convexity (vomma)
        d      : standardized moneyness
        phi_d  : φ(d) — standard normal density at d
        Phi_d  : Φ(d) or Φ(−d) depending on payer flag — CDF used in delta

    Edge cases
    ----------
    * expiry ≤ 0 or normal_vol ≤ 0 → returns intrinsic greeks
      (delta = Φ(d) in {0,1}, vega=gamma=theta=vanna=volga=0)
    """
    F  = float(forward)
    K  = float(strike)
    sv = float(normal_vol)
    T  = float(expiry)
    A  = float(annuity)
    N  = float(notional)

    intrinsic = N * A * (max(F - K, 0.0) if payer else max(K - F, 0.0))

    # --- Degenerate cases ---------------------------------------------------
    if T <= 0.0 or sv <= 0.0:
        # At or past expiry: delta = 1 if ITM, 0 if OTM; all other greeks = 0
        delta = N * A * (1.0 if (F > K if payer else K > F) else 0.0)
        return dict(delta=delta, vega=0.0, gamma=0.0, theta=0.0,
                    dv01=delta * 1e-4, vanna=0.0, volga=0.0,
                    d=float("inf") if F != K else 0.0,
                    phi_d=0.0, Phi_d=float(delta > 0))

    T_sqrt   = math.sqrt(T)
    vol_term = sv * T_sqrt                       # σ√T, always > 0

    d        = (F - K) / vol_term
    phi_d    = norm.pdf(d)                       # φ(d)  = φ(−d)
    Phi_d    = norm.cdf(d if payer else -d)      # Φ(d) payer, Φ(−d) receiver

    # --- First-order greeks -------------------------------------------------
    # Payer:   delta = N·A·Φ(d)
    # Receiver: delta = N·A·(Φ(d) − 1) = −N·A·Φ(−d)
    delta  = N * A * Phi_d if payer else -N * A * Phi_d

    # Vega is the same for payer and receiver (put-call symmetry in Bachelier)
    vega   = N * A * T_sqrt * phi_d

    # --- Second-order greeks ------------------------------------------------
    # Gamma = ∂²V/∂F² = N·A·φ(d) / (σ√T)
    gamma  = N * A * phi_d / vol_term

    # Theta = ∂V/∂T = N·A·σ·φ(d) / (2√T)   (positive: option gains value with T)
    # Equivalently derived from the ATM formula V_ATM = N·A·σ·√(T/2π):
    #   ∂V_ATM/∂T = N·A·σ / (2√(2πT))  = N·A·σ·φ(0) / (2√T)  [same formula]
    theta  = N * A * sv * phi_d / (2.0 * T_sqrt)

    # DV01 = value change for 1 bp shift in F
    dv01   = delta * 1e-4

    # Vanna = ∂²V / (∂F ∂σ) = d(delta)/dσ = −N·A·φ(d)·d / σ
    # Derivation: d(Φ(d))/dσ = φ(d)·∂d/∂σ = φ(d)·(−d/σ)
    vanna  = -N * A * phi_d * d / sv

    # Volga (Vomma) = ∂²V / ∂σ² = d(vega)/dσ
    # = N·A·√T · φ'(d) · ∂d/∂σ  =  N·A·√T · (−d·φ(d)) · (−d/σ)
    # = N·A·√T · d² · φ(d) / σ
    volga  = N * A * T_sqrt * (d ** 2) * phi_d / sv

    return dict(
        delta=delta,
        vega=vega,
        gamma=gamma,
        theta=theta,
        dv01=dv01,
        vanna=vanna,
        volga=volga,
        d=d,
        phi_d=phi_d,
        Phi_d=Phi_d,
    )


def implied_bachelier_vol(
    market_price: float,
    forward: float,
    strike: float,
    expiry: float,
    annuity: float,
    notional: float = 1.0,
    payer: bool = True,
    tol: float = 1e-12,
    _return_failure_reason: bool = False,
) -> Union[float, tuple[float, Optional[str]]]:
    """
    Invert market_price → normal (Bachelier) volatility via Brent's method.

    Parameters
    ----------
    market_price : float
        Observed market price to match
    forward : float
        Forward swap rate
    strike : float
        Strike rate
    expiry : float
        Time to expiry in years
    annuity : float
        Annuity factor
    notional : float, default 1.0
        Notional amount
    payer : bool, default True
        If True, payer swaption; if False, receiver
    tol : float, default 1e-12
        Tolerance for convergence and validity checks
    _return_failure_reason : bool, default False
        If True, return (vol_or_nan, reason_str) instead of just the vol.
        reason_str is None on success, a short string on failure.
    
    Returns
    -------
    float or tuple
        If _return_failure_reason=False:
            Implied volatility (float), or NaN if inversion fails
        If _return_failure_reason=True:
            (implied_vol, reason_str) where reason_str is None on success
    """
    intrinsic = notional * annuity * (
        max(forward - strike, 0.0) if payer else max(strike - forward, 0.0)
    )

    def _ret(val, reason=None):
        return (val, reason) if _return_failure_reason else val

    if expiry <= 0.0 or annuity <= 0.0 or notional <= 0.0:
        reason = f"degenerate inputs: expiry={expiry}, annuity={annuity}, notional={notional}"
        warnings.warn(f"implied_bachelier_vol: {reason}", RuntimeWarning)
        return _ret(np.nan, reason)

    if market_price < intrinsic - tol:
        reason = (
            f"price {market_price:.6g} below intrinsic {intrinsic:.6g} "
            f"(F={forward:.6g}, K={strike:.6g})"
        )
        warnings.warn(f"implied_bachelier_vol: {reason}", RuntimeWarning)
        return _ret(np.nan, reason)

    if abs(market_price - intrinsic) <= tol:
        return _ret(0.0, None)

    def objective(sigma):
        return bachelier_price(
            forward=forward,
            strike=strike,
            normal_vol=sigma,
            expiry=expiry,
            annuity=annuity,
            notional=notional,
            payer=payer,
        ) - market_price

    lower = 1e-12
    upper = 1e-4

    while objective(upper) < 0.0 and upper < 100.0:
        upper *= 2.0

    if objective(upper) < 0.0:
        reason = (
            f"bracket exhausted at upper={upper:.3g}: "
            f"price {market_price:.6g} may be too large "
            f"(F={forward:.6g}, K={strike:.6g}, T={expiry})"
        )
        warnings.warn(f"implied_bachelier_vol: {reason}", RuntimeWarning)
        return _ret(np.nan, reason)

    vol = brentq(objective, lower, upper, xtol=1e-12, rtol=1e-10, maxiter=200)
    return _ret(vol, None)

def atm_swaption_mc_price_from_simulation(
    ctx: dict,
    expiry: float,
    tenor: int,
    payer: bool = True,
    accrual: float = 1.0,
    notional: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    ATM swaption MC price with strike = time-0 forward swap rate.
    
    Convenience wrapper around swaption_mc_price_from_simulation that
    automatically uses the forward swap rate as the strike.
    
    Parameters
    ----------
    ctx : dict
        Simulation context from run_simulation()
    expiry : float
        Swaption expiry in years
    tenor : int
        Swap tenor in years
    payer : bool, default True
        If True, payer swaption; if False, receiver
    accrual : float, default 1.0
        Payment frequency in years
    notional : float, default 1.0
        Notional amount
    verbose : bool, default False
        If True, prints diagnostic information
    
    Returns
    -------
    dict
        Same as swaption_mc_price_from_simulation, plus:
        - forward_swap_rate : float - The ATM strike used
    
    See Also
    --------
    swaption_mc_price_from_simulation : Full MC pricing with custom strike
    """
    quote = quote_swaption_time0(
        ctx=ctx,
        expiry=expiry,
        tenor=tenor,
        strike_atm=True,
        payer=payer,
        accrual=accrual,
        verbose=verbose,
    )

    res = swaption_mc_price_from_simulation(
        ctx=ctx,
        expiry=expiry,
        tenor=tenor,
        strike=quote["strike"],
        payer=payer,
        accrual=accrual,
        notional=notional,
        verbose=verbose,
    )

    iv, iv_fail = implied_bachelier_vol(
        market_price=res["mc_price"],
        forward=quote["forward_swap"],
        strike=quote["strike"],
        expiry=expiry,
        annuity=quote["annuity"],
        notional=notional,
        payer=payer,
        _return_failure_reason=True,
    )

    res["quote"] = quote
    res["implied_normal_vol"] = iv
    res["implied_normal_vol_failure"] = iv_fail

    if verbose:
        print("\nModel-implied normal vol")
        iv_display = iv if (iv is not None and np.isfinite(iv)) else float("nan")
        print(f"  Vol (abs)         : {iv_display:.10f}")
        print(f"  Vol (bp)          : {iv_display * 10000:.2f}")

    return res


def atm_swaption_straddle_mc_price_from_simulation(
    ctx: dict,
    expiry: float,
    tenor: int,
    accrual: float = 1.0,
    notional: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    ATM straddle (payer + receiver averaged) MC price using the SAME
    simulated paths for both legs, computed in a single vectorised pass.

    For an at-the-money strike K = F_0 (the time-0 forward swap rate) the
    Bachelier payer and receiver prices are identical, so any pathwise
    difference between the two MC estimators is a pure directional bias
    coming from the simulated forward swap rate not centring on K.
    Averaging the two estimators,

        V_avg = 0.5 * (V_pay + V_rec),

    cancels this directional bias to first order, while

        b = (V_pay - V_rec) / A_0

    is an MC estimator of  E^A[S(T_mu) - K]  under the annuity measure.

    Implementation notes
    --------------------
    Earlier this function called ``atm_swaption_mc_price_from_simulation``
    twice (payer then receiver). Each call recomputed the time-0 quote, the
    per-path payment-date slice, and the discount-factor array — all of
    which are independent of the payer/receiver flag. The function now does
    ONE vectorised pass producing both leg payoffs from the same annuity /
    swap-rate / discount arrays, and inverts Bachelier only once (on the
    average).

    Returns
    -------
    dict with the same contract as before: ``payer_price``,
    ``receiver_price``, ``straddle_price``, ``straddle_stderr``,
    ``forward_bias``, ``implied_normal_vol``, ``quote``, ``payer``
    (single-leg dict with ``annuity_paths``, ``swap_rate_paths``,
    ``discount_to_expiry_paths``, ``valid_mask``, ...), ``receiver``
    (analogous), ``valid_mask`` (intersection).
    """
    # --- Time-0 ATM quote -------------------------------------------------
    quote = quote_swaption_time0(
        ctx=ctx, expiry=expiry, tenor=tenor,
        strike_atm=True, payer=True,
        accrual=accrual, verbose=False,
    )
    F0 = float(quote["forward_swap"])
    A0 = float(quote["annuity"])
    K  = float(quote["strike"])

    # --- One vectorised pass over the simulated paths --------------------
    times        = np.asarray(ctx["times"], dtype=float)
    tau_grid     = ctx["tau_grid"].detach().cpu().numpy()
    P_full_paths = ctx["P_full_paths"].detach().cpu().numpy()
    exp_idx      = get_grid_index_for_value(times, expiry)
    # NOTE: ctx["discount_paths"] lives on the FULL simulation grid, while
    # exp_idx is an index into the downsampled ctx["times"] used by
    # decode_steps. We must map the expiry to its position on the full
    # grid before slicing — see comment in
    # swaption_mc_price_from_simulation.
    _disc_arr      = ctx["discount_paths"].detach().cpu().numpy()
    _n_full_steps  = _disc_arr.shape[1] - 1
    _dt_full       = float(times.max()) / max(_n_full_steps, 1)
    _full_exp_idx  = int(round(float(expiry) / _dt_full))
    if not (0 <= _full_exp_idx < _disc_arr.shape[1]):
        raise ValueError(
            f"Expiry {expiry}Y maps to full-grid index {_full_exp_idx} "
            f"outside [0, {_disc_arr.shape[1]-1}]"
        )
    discount_to_expiry = np.asarray(_disc_arr[:, _full_exp_idx], dtype=float)

    payment_taus = [accrual * j for j in range(1, tenor + 1)]
    tau_indices  = np.array(
        [get_grid_index_for_value(tau_grid, t) for t in payment_taus],
        dtype=int,
    )

    payment_dfs = P_full_paths[:, exp_idx, tau_indices]          # (n_paths, tenor)
    # Conservative physical-validity gate: reject any path whose decoded
    # curve at expiry has ANY unphysical discount factor (see comment in
    # `swaption_payoff_from_simulation`).
    curve_at_expiry = P_full_paths[:, exp_idx, :]
    finite_pay  = _physically_valid(curve_at_expiry)

    swap_rate_paths, annuity_paths = _swap_from_payment_dfs(payment_dfs, accrual=accrual)
    valid_swap = (
        finite_pay
        & np.isfinite(annuity_paths) & (annuity_paths > 0.0)
        & np.isfinite(swap_rate_paths)
    )
    swap_rate_paths = np.where(valid_swap, swap_rate_paths, np.nan)
    annuity_paths   = np.where(valid_swap, annuity_paths,   np.nan)

    # Both legs share annuity / swap-rate / discount; only the payoff sign differs
    payer_payoff    = notional * annuity_paths * np.maximum(swap_rate_paths - K, 0.0)
    receiver_payoff = notional * annuity_paths * np.maximum(K - swap_rate_paths, 0.0)

    pv_pay = discount_to_expiry * payer_payoff
    pv_rec = discount_to_expiry * receiver_payoff

    valid_mask = (
        valid_swap
        & np.isfinite(discount_to_expiry)
        & np.isfinite(pv_pay)
        & np.isfinite(pv_rec)
    )
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise RuntimeError("No valid paths for Monte Carlo straddle pricing.")

    pv_pay_v = pv_pay[valid_mask]
    pv_rec_v = pv_rec[valid_mask]

    pay_price = float(np.mean(pv_pay_v))
    rec_price = float(np.mean(pv_rec_v))
    straddle_price = 0.5 * (pay_price + rec_price)

    # Standard error of the average uses the SAME paths and is therefore the
    # std of 0.5*(pv_pay + pv_rec) divided by sqrt(N) — no independence
    # assumption needed (this is tighter than the previous conservative
    # 0.5*sqrt(se_p^2 + se_r^2) upper bound).
    pv_avg_v = 0.5 * (pv_pay_v + pv_rec_v)
    straddle_se = float(np.std(pv_avg_v, ddof=1) / math.sqrt(n_valid)) if n_valid > 1 else 0.0
    se_pay = float(np.std(pv_pay_v, ddof=1) / math.sqrt(n_valid)) if n_valid > 1 else 0.0
    se_rec = float(np.std(pv_rec_v, ddof=1) / math.sqrt(n_valid)) if n_valid > 1 else 0.0

    forward_bias = (pay_price - rec_price) / A0 if A0 > 0 else float("nan")

    # ATM (F = K): Bachelier payer = receiver, so V_straddle = V_pay = V_rec
    # and σ_N^imp = V_straddle * sqrt(2π) / (A_0 * sqrt(T_μ)).
    iv_str, iv_fail = implied_bachelier_vol(
        market_price=straddle_price,
        forward=F0, strike=K, expiry=expiry,
        annuity=A0, notional=notional, payer=True,
        _return_failure_reason=True,
    )

    # --- Per-leg sub-dicts (same shape downstream code expects) ----------
    def _leg_dict(pv, payoff, price, se, payer_flag):
        return {
            "expiry": expiry,
            "tenor": tenor,
            "expiry_index": exp_idx,
            "strike": K,
            "payer": payer_flag,
            "annuity_paths": annuity_paths,
            "swap_rate_paths": swap_rate_paths,
            "payoff_paths": payoff,
            "discount_to_expiry_paths": discount_to_expiry,
            "pv_paths": pv,
            "mc_price": price,
            "mc_std": float(np.std(pv[valid_mask], ddof=1)) if n_valid > 1 else 0.0,
            "mc_stderr": se,
            "mean_payoff": float(np.mean(payoff[valid_mask])),
            "valid_mask": valid_mask,
        }

    res_pay = _leg_dict(pv_pay, payer_payoff,    pay_price, se_pay, True)
    res_rec = _leg_dict(pv_rec, receiver_payoff, rec_price, se_rec, False)

    if verbose:
        print("\nATM straddle Monte Carlo pricing (single-pass)")
        print(f"  Valid paths       : {n_valid}/{len(valid_mask)}")
        print(f"  Payer price       : {pay_price:.10f} ± {se_pay:.2e}")
        print(f"  Receiver price    : {rec_price:.10f} ± {se_rec:.2e}")
        print(f"  Straddle price    : {straddle_price:.10f} ± {straddle_se:.2e}")
        print(f"  Forward bias      : {forward_bias:+.6e}  ({forward_bias*1e4:+.3f} bp)")
        iv_disp = iv_str if (iv_str is not None and np.isfinite(iv_str)) else float("nan")
        print(f"  Implied vol (bp)  : {iv_disp * 10000:.2f}")

    return {
        "expiry": expiry,
        "tenor": tenor,
        "quote": quote,
        "payer": res_pay,
        "receiver": res_rec,
        "payer_price": pay_price,
        "receiver_price": rec_price,
        "straddle_price": straddle_price,
        "straddle_stderr": straddle_se,
        "forward_bias": forward_bias,
        "implied_normal_vol": iv_str,
        "implied_normal_vol_failure": iv_fail,
        "valid_mask": valid_mask,
    }


def main():
    checkpoint_path = r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults\dim4_stable\ep5000\checkpoint_dim4_ep5000.pt"

    # For 10-year horizon with monthly steps:
    # dt = 1/12 (monthly), n_steps = 120 gives 10 years
    # For 2Y expiry, need at least n_steps = 24 (2 years)
    
    ctx = run_simulation(
        checkpoint_path=checkpoint_path,
        ccy_filter="EUR",
        latent_dim=4,
        n_paths=2000,
        n_steps=24,  # 2 years with monthly steps for 2Y expiry
        dt=1 / 12,  # Monthly timesteps (not weekly!)
        diffusion_scale=0.2,  # Balances weak drift (||μ||/||σ|| = 0.055)
        show_plot=False,  # Faster execution
    )

    # First, check simulation health
    print("\n" + "="*60)
    print("PRE-FLIGHT SIMULATION DIAGNOSTICS")
    print("="*60)
    
    z_paths = ctx["z_paths"].detach().cpu().numpy()
    z_train_mean = ctx["z_train_mean"].cpu().numpy()
    z_train_std = ctx["z_train_std"].cpu().numpy()
    
    print("\nLatent state health check:")
    for d in range(z_paths.shape[2]):
        z_d = z_paths[:, :, d]
        train_min = z_train_mean[d] - 3*z_train_std[d]
        train_max = z_train_mean[d] + 3*z_train_std[d]
        sim_min = z_d.min()
        sim_max = z_d.max()
        pct_in_range = np.mean((z_d >= train_min) & (z_d <= train_max)) * 100
        
        print(f"  z[{d}]:")
        print(f"    Training μ±3σ  : [{train_min:.4f}, {train_max:.4f}]")
        print(f"    Simulation range: [{sim_min:.4f}, {sim_max:.4f}]")
        print(f"    % within 3σ    : {pct_in_range:.1f}%")
    
    # Check discount curves
    P_full = ctx["P_full_paths"].detach().cpu().numpy()
    print(f"\nDiscount curve health:")
    print(f"  Shape             : {P_full.shape}")
    print(f"  Min               : {np.nanmin(P_full):.6f}")
    print(f"  Max               : {np.nanmax(P_full):.6f}")
    print(f"  Paths with inf    : {np.isinf(P_full).any(axis=(1,2)).sum()}/{P_full.shape[0]}")
    print(f"  Paths with nan    : {np.isnan(P_full).any(axis=(1,2)).sum()}/{P_full.shape[0]}")
    print(f"  Paths with P>1    : {(P_full > 1.0).any(axis=(1,2)).sum()}/{P_full.shape[0]}")
    print(f"  Paths with P<0    : {(P_full < 0.0).any(axis=(1,2)).sum()}/{P_full.shape[0]}")
    
    if np.isinf(P_full).any():
        print(f"\n⚠️  WARNING: Infinity detected in discount curves!")
        print(f"   This indicates decoder numerical overflow.")
        print(f"   Try: diffusion_scale=0.5 or lower")
    
    res = atm_swaption_mc_price_from_simulation(
        ctx=ctx,
        expiry=2,
        tenor=5,
        payer=True,
        accrual=1.0,
        notional=1.0,
        verbose=True,  # Enable verbose output
    )

    print("\n" + "="*60)
    print("SIMULATION DIAGNOSTICS")
    print("="*60)
    
    # Check for path validity
    n_total = len(res["swap_rate_paths"])
    n_valid = res["valid_mask"].sum()
    n_invalid = n_total - n_valid
    
    print(f"\nPath validity:")
    print(f"  Total paths       : {n_total}")
    print(f"  Valid paths       : {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  Invalid paths     : {n_invalid} ({100*n_invalid/n_total:.1f}%)")
    
    # Check discount curve ranges
    P_full = ctx["P_full_paths"].detach().cpu().numpy()
    print(f"\nDiscount curve statistics:")
    print(f"  Min discount      : {np.nanmin(P_full):.6f}")
    print(f"  Max discount      : {np.nanmax(P_full):.6f}")
    print(f"  Paths with inf    : {np.isinf(P_full).any(axis=(1,2)).sum()}")
    print(f"  Paths with nan    : {np.isnan(P_full).any(axis=(1,2)).sum()}")

    # Valid swap rates statistics
    valid_swap_rates = res["swap_rate_paths"][res["valid_mask"]]
    if len(valid_swap_rates) > 0:
        print(f"\nValid swap rate statistics at expiry:")
        print(f"  Mean              : {np.mean(valid_swap_rates):.6f} ({np.mean(valid_swap_rates)*10000:.1f} bp)")
        print(f"  Std               : {np.std(valid_swap_rates):.6f} ({np.std(valid_swap_rates)*10000:.1f} bp)")
        print(f"  Min               : {np.min(valid_swap_rates):.6f} ({np.min(valid_swap_rates)*10000:.1f} bp)")
        print(f"  Max               : {np.max(valid_swap_rates):.6f} ({np.max(valid_swap_rates)*10000:.1f} bp)")

    print("\n" + "="*60)
    print("PRICING RESULTS")
    print("="*60)

    print("\nFirst 5 valid swap rates at expiry:")
    valid_indices = np.where(res["valid_mask"])[0][:5]
    for i, idx in enumerate(valid_indices):
        print(f"  Path {idx}: {res['swap_rate_paths'][idx]:.6f} ({res['swap_rate_paths'][idx]*10000:.1f} bp)")

    print("\nFirst 5 valid payoffs at expiry:")
    for i, idx in enumerate(valid_indices):
        print(f"  Path {idx}: {res['payoff_paths'][idx]:.10f}")

    print("\nFirst 5 valid discount factors to expiry:")
    for i, idx in enumerate(valid_indices):
        print(f"  Path {idx}: {res['discount_to_expiry_paths'][idx]:.10f}")

    print("\nFirst 5 valid discounted PVs:")
    for i, idx in enumerate(valid_indices):
        print(f"  Path {idx}: {res['pv_paths'][idx]:.10f}")

    print(f"\n{'='*60}")
    print(f"Monte Carlo price : {res['mc_price']:.10f}")
    print(f"MC std error      : {res['mc_stderr']:.10f}")
    print(f"Model normal vol  : {res['implied_normal_vol'] * 10000:.2f} bp")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()