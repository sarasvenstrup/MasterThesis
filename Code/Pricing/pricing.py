import os
import sys

import math
import warnings
from scipy.stats import norm
from scipy.optimize import brentq

import numpy as np

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

from Code.Pricing.simulate_model import run_simulation


def get_grid_index_for_value(grid, value, tol=1e-10):
    grid = np.asarray(grid, dtype=float)
    diffs = np.abs(grid - float(value))
    idx = int(np.argmin(diffs))
    if diffs[idx] > tol:
        raise ValueError(
            f"Value {value} not found in tau grid within tol={tol}. "
            f"Closest value is {grid[idx]}."
        )
    return idx


def extract_discount(P_full_0, tau_grid, tau):
    P = np.asarray(P_full_0, dtype=float)
    if P.ndim == 2:
        P = P[0]
    idx = get_grid_index_for_value(tau_grid, tau)
    return float(P[idx])


def time0_forward_swap_and_annuity(P_full_0, tau_grid, expiry, tenor, accrual=1.0):
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

    A0 = accrual * P_payments.sum()

    if expiry > 0:
        P_start = extract_discount(P_full_0, tau_grid, expiry)
    else:
        P_start = 1.0

    P_end = P_payments[-1]
    F0 = (P_start - P_end) / A0

    return {
        "forward_swap": F0,
        "annuity": A0,
        "payment_times": payment_times,
        "payment_dfs": P_payments,
        "P_start": P_start,
        "P_end": P_end,
    }


def quote_swaption_time0(ctx, expiry, tenor, strike=None, strike_atm=False, payer=True, accrual=1.0):
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

    print("\nTime-0 swaption quote")
    print(f"  Expiry            : {expiry}Y")
    print(f"  Tenor             : {tenor}Y")
    print(f"  Forward swap F0   : {out['forward_swap']:.6f}  ({out['forward_swap']*10000:.1f} bp)")
    print(f"  Annuity A0        : {out['annuity']:.10f}")
    print(f"  Strike K          : {out['strike']:.10f}")
    print(f"  Intrinsic lb      : {out['intrinsic_lower_bound']:.10f}")

    return out


def get_time_index_for_value(times, value, tol=1e-10):
    times = np.asarray(times, dtype=float)
    diffs = np.abs(times - float(value))
    idx = int(np.argmin(diffs))
    if diffs[idx] > tol:
        raise ValueError(
            f"Time {value} not found in simulation grid within tol={tol}. "
            f"Closest value is {times[idx]}."
        )
    return idx


def swap_from_discount_curve_at_expiry(P_curve, tau_grid, tenor, accrual=1.0):
    """
    One expiry curve P(Te, Te+tau) -> annuity and spot-starting swap rate.
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

    annuity = accrual * payment_dfs.sum()
    if annuity <= 0.0:
        raise ValueError(f"Non-positive annuity encountered: {annuity}")

    P_end = payment_dfs[-1]
    swap_rate = (1.0 - P_end) / annuity

    return {
        "annuity": annuity,
        "swap_rate": swap_rate,
        "payment_taus": payment_taus,
        "payment_dfs": payment_dfs,
        "P_end": P_end,
    }


def swaption_payoff_from_simulation(
    ctx,
    expiry,
    tenor,
    strike,
    payer=True,
    accrual=1.0,
    notional=1.0,
):
    """
    Compute pathwise swaption payoff at expiry, without discounting back to time 0.
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

    exp_idx = get_time_index_for_value(times, expiry)

    n_paths = P_full_paths.shape[0]
    annuity_paths = np.full(n_paths, np.nan, dtype=float)
    swap_rate_paths = np.full(n_paths, np.nan, dtype=float)
    payoff_paths = np.full(n_paths, np.nan, dtype=float)

    for m in range(n_paths):
        P_curve = P_full_paths[m, exp_idx, :]

        try:
            q = swap_from_discount_curve_at_expiry(
                P_curve=P_curve,
                tau_grid=tau_grid,
                tenor=tenor,
                accrual=accrual,
            )

            annuity_paths[m] = q["annuity"]
            swap_rate_paths[m] = q["swap_rate"]

            if payer:
                payoff_paths[m] = notional * q["annuity"] * max(q["swap_rate"] - strike, 0.0)
            else:
                payoff_paths[m] = notional * q["annuity"] * max(strike - q["swap_rate"], 0.0)

        except Exception:
            pass

    valid_mask = np.isfinite(annuity_paths) & np.isfinite(swap_rate_paths) & np.isfinite(payoff_paths)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise RuntimeError("No valid paths for swaption payoff computation.")

    mean_payoff = float(np.mean(payoff_paths[valid_mask]))

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
    ctx,
    expiry,
    tenor,
    strike,
    payer=True,
    accrual=1.0,
    notional=1.0,
):
    """
    Full Monte Carlo time-0 swaption price:
      1) compute payoff at expiry pathwise
      2) discount each payoff back to time 0
      3) average across valid paths
    """
    payoff_res = swaption_payoff_from_simulation(
        ctx=ctx,
        expiry=expiry,
        tenor=tenor,
        strike=strike,
        payer=payer,
        accrual=accrual,
        notional=notional,
    )

    exp_idx = payoff_res["expiry_index"]
    payoff_paths = payoff_res["payoff_paths"]
    valid_mask = payoff_res["valid_mask"]

    discount_paths = ctx["discount_paths"].detach().cpu().numpy()
    discount_to_expiry = np.asarray(discount_paths[:, exp_idx], dtype=float)

    pv_paths = discount_to_expiry * payoff_paths

    valid_mask = valid_mask & np.isfinite(discount_to_expiry) & np.isfinite(pv_paths)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise RuntimeError("No valid paths for Monte Carlo pricing.")

    mc_price = float(np.mean(pv_paths[valid_mask]))
    mc_std = float(np.std(pv_paths[valid_mask], ddof=1)) if n_valid > 1 else 0.0
    mc_se = mc_std / np.sqrt(n_valid)

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


def bachelier_price(forward, strike, normal_vol, expiry, annuity, notional=1.0, payer=True):
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


def implied_bachelier_vol(
    market_price,
    forward,
    strike,
    expiry,
    annuity,
    notional=1.0,
    payer=True,
    tol=1e-12,
):
    intrinsic = notional * annuity * (
        max(forward - strike, 0.0) if payer else max(strike - forward, 0.0)
    )

    if expiry <= 0.0 or annuity <= 0.0 or notional <= 0.0:
        return np.nan

    if market_price < intrinsic - tol:
        warnings.warn(
            f"Price {market_price:.12f} below intrinsic {intrinsic:.12f}; cannot infer normal vol.",
            RuntimeWarning,
        )
        return np.nan

    if abs(market_price - intrinsic) <= tol:
        return 0.0

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
        return np.nan

    return brentq(objective, lower, upper, xtol=1e-12, rtol=1e-10, maxiter=200)

def atm_swaption_mc_price_from_simulation(
    ctx,
    expiry,
    tenor,
    payer=True,
    accrual=1.0,
    notional=1.0,
):
    quote = quote_swaption_time0(
        ctx=ctx,
        expiry=expiry,
        tenor=tenor,
        strike_atm=True,
        payer=payer,
        accrual=accrual,
    )

    res = swaption_mc_price_from_simulation(
        ctx=ctx,
        expiry=expiry,
        tenor=tenor,
        strike=quote["strike"],
        payer=payer,
        accrual=accrual,
        notional=notional,
    )

    iv = implied_bachelier_vol(
        market_price=res["mc_price"],
        forward=quote["forward_swap"],
        strike=quote["strike"],
        expiry=expiry,
        annuity=quote["annuity"],
        notional=notional,
        payer=payer,
    )

    res["quote"] = quote
    res["implied_normal_vol"] = iv

    print("\nModel-implied normal vol")
    print(f"  Vol (abs)         : {iv:.10f}")
    print(f"  Vol (bp)          : {iv * 10000:.2f}")

    return res

def main():
    checkpoint_path = r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults\dim2_stable\ep200\checkpoint_dim2_ep200.pt"

    ctx = run_simulation(
        checkpoint_path=checkpoint_path,
        ccy_filter="EUR",
        n_paths=500,
        n_steps=24,
        dt=1 / 12,
        show_plot=False,
    )

    res = atm_swaption_mc_price_from_simulation(
        ctx=ctx,
        expiry=2,
        tenor=5,
        payer=True,
        accrual=1.0,
        notional=1.0,
    )

    print("\nFirst 5 swap rates at expiry:")
    print(res["swap_rate_paths"][:5])

    print("\nFirst 5 payoffs at expiry:")
    print(res["payoff_paths"][:5])

    print("\nFirst 5 discount factors to expiry:")
    print(res["discount_to_expiry_paths"][:5])

    print("\nFirst 5 discounted PVs:")
    print(res["pv_paths"][:5])

    print(f"\nMonte Carlo price: {res['mc_price']:.10f}")
    print(f"Model normal vol: {res['implied_normal_vol'] * 10000:.2f} bp")


if __name__ == "__main__":
    main()