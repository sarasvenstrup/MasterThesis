# =============================================================================
# diag_checkpoint_v4.py
# =============================================================================
# v3 showed the model produces 86 bp of *spot* swap rate std at 5Y.
# But the MC swaption gives only ~3 bp implied vol at 5Y×1Y.
#
# The missing piece: at expiry T, the swaption payoff is
#   max( F_T(T, T+n) - K, 0 )
# where F_T(T, T+n) is the spot-starting swap rate AT TIME T,
# and K = F_0(T, T+n) is the FORWARD swap rate seen from t=0.
#
# If K is set correctly (ATM at t=0), and F_T has std=86 bp, then
# implied vol should be roughly 86/sqrt(5) ≈ 38 bp — not 3 bp.
#
# So either:
#   (a) K is wrong (not matching F_0 properly), OR
#   (b) The distribution of F_T - K is highly skewed / peaked near zero
#       because paths mean-revert so strongly that F_T ≈ K for almost all paths
#       (i.e. the spot swap rate mean-reverts toward K, not away from it)
#
# This script checks both by:
#   1. Printing the distribution of F_T at each expiry for the 1Y tenor
#   2. Comparing F_T mean vs K (ATM strike)
#   3. Computing E[max(F_T - K, 0)] directly and converting to vol
#   4. Checking whether F_T std matches the spot swap std from v3
# =============================================================================

import argparse
import math
import os
import sys

import numpy as np
import torch

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

CODE_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT,  ".."))
for p in [THESIS_ROOT, CODE_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_CHECKPOINT = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults"
    r"\dim2_stable\ep300\checkpoint_dim2_ep300.pt"
)
DEFAULT_CCY  = "EUR"
DEFAULT_DATE = "2010-10-29"
N_PATHS      = 2000
N_STEPS      = 120
DT           = 1 / 12

from Code import config
from Code.model.full_model import FullModel
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.load_swapdata import my_data
from Code.Pricing.simulate_model import (
    resolve_curve_index, simulate_latent_paths, compute_discount_paths
)
from Code.Pricing.pricing import (
    time0_forward_swap_and_annuity,
    swap_from_discount_curve_at_expiry,
)
from scipy.optimize import brentq


def load_model(checkpoint_path, device):
    raw   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = raw["model_state_dict"] if (isinstance(raw, dict) and "model_state_dict" in raw) else raw
    model = FullModel(latent_dim=2).to(device).double()
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def bachelier_price(F, K, sigma, T, A):
    """Normal (Bachelier) swaption price."""
    if sigma <= 0 or T <= 0:
        return A * max(F - K, 0.0)
    d  = (F - K) / (sigma * math.sqrt(T))
    from scipy.stats import norm
    return A * (sigma * math.sqrt(T) * (d * norm.cdf(d) + norm.pdf(d)))


def implied_bachelier(price, F, K, T, A, payer=True):
    """Invert Bachelier price to get normal vol."""
    intrinsic = A * max(F - K, 0.0) if payer else A * max(K - F, 0.0)
    if price <= intrinsic + 1e-14:
        return 0.0
    def obj(sigma):
        return bachelier_price(F, K, sigma, T, A) - price
    try:
        return brentq(obj, 1e-8, 10.0, xtol=1e-12, maxiter=200)
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ccy",        default=DEFAULT_CCY)
    parser.add_argument("--date",       default=DEFAULT_DATE)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(1234)
    np.random.seed(1234)

    print("="*70)
    print("DIAGNOSTIC v4 — Forward swap distribution at expiry")
    print("="*70)

    model = load_model(args.checkpoint, device)

    meta, X_tensor, *_ = my_data(ccy_filter=args.ccy)
    dtype = next(model.parameters()).dtype
    X_tensor = X_tensor.to(dtype=dtype)

    idx = resolve_curve_index(meta, as_of_date=args.date)
    S0  = X_tensor[idx:idx+1].to(device=device, dtype=dtype)
    with torch.no_grad():
        z0 = model.encoder(S0)

    # -------------------------------------------------------------------------
    # Decode initial curve to get F0 and A0 for each (expiry, tenor)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        _, aux0 = model.decode_from_z(z0, tau=None, do_arb_checks=False, return_aux=True)
    P_full_0 = aux0["P_full"].detach().cpu().numpy()    # (1, tau_max+1)
    tau_grid = aux0["tau_grid"].detach().cpu().numpy()  # (tau_max+1,)

    print(f"\n  z0 = {z0.detach().cpu().numpy().flatten()}")
    print(f"  tau_grid range: [{tau_grid.min():.2f}, {tau_grid.max():.2f}]")

    # -------------------------------------------------------------------------
    # Simulate paths
    # -------------------------------------------------------------------------
    print(f"\n  Simulating {N_PATHS} paths x {N_STEPS} steps...")
    z_paths, r_paths, _, _ = simulate_latent_paths(
        model=model, z0=z0, n_paths=N_PATHS, n_steps=N_STEPS, dt=DT, device=device
    )
    discount_paths = compute_discount_paths(r_paths, dt=DT)

    # Decode P_full at each annual step (only need annual steps)
    annual_steps = {1: 12, 5: 60, 10: 120}   # expiry_yr -> step_index

    print(f"\n{'='*70}")
    print("FORWARD SWAP RATE DISTRIBUTION AT EXPIRY")
    print(f"{'='*70}")
    print(f"  K = F_0(expiry, tenor) = ATM strike set from t=0 initial curve")
    print()

    for tenor in [1, 5, 10]:
        print(f"  Tenor = {tenor}Y")
        print(f"  {'Expiry':>8}  {'K (bp)':>8}  {'F_T mean (bp)':>14}  "
              f"{'F_T std (bp)':>13}  {'E[payoff]':>11}  {'impl vol (bp)':>14}  "
              f"{'F_T≈K? (%)':>12}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*13}  {'-'*11}  {'-'*14}  {'-'*12}")

        for expiry_yr, exp_step in sorted(annual_steps.items()):

            # ATM strike K from t=0 curve
            q0 = time0_forward_swap_and_annuity(
                P_full_0, tau_grid, expiry=expiry_yr, tenor=tenor, accrual=1.0
            )
            K  = q0["forward_swap"]
            A0 = q0["annuity"]

            # Extract z at expiry step for all paths
            z_at_exp = z_paths[:, exp_step, :]   # (N_PATHS, d)

            # Decode P_full for all paths at expiry
            with torch.no_grad():
                _, aux_exp = model.decode_from_z(
                    z_at_exp, tau=None, do_arb_checks=False, return_aux=True
                )
            P_full_exp = aux_exp["P_full"].detach().cpu().numpy()   # (N_PATHS, tau_max+1)
            tau_grid_exp = aux_exp["tau_grid"].detach().cpu().numpy()

            # Compute spot-starting swap rate at expiry for each path
            swap_rates = []
            annuities  = []
            for pi in range(N_PATHS):
                try:
                    res = swap_from_discount_curve_at_expiry(
                        P_full_exp[pi], tau_grid_exp, tenor=tenor, accrual=1.0
                    )
                    swap_rates.append(res["swap_rate"])
                    annuities.append(res["annuity"])
                except Exception:
                    swap_rates.append(float("nan"))
                    annuities.append(float("nan"))

            F_T = np.array(swap_rates)
            A_T = np.array(annuities)
            valid = np.isfinite(F_T) & np.isfinite(A_T)
            F_T = F_T[valid]
            A_T = A_T[valid]

            # Discount factors to expiry for valid paths
            D_T = discount_paths[:, exp_step].detach().cpu().numpy()[valid]

            # Payer payoff: max(F_T - K, 0) * A_T, discounted to t=0
            payoff  = np.maximum(F_T - K, 0.0) * A_T
            pv      = D_T * payoff
            mc_price = pv.mean()

            # Implied vol (use A0 as annuity for Bachelier — standard convention)
            F0_val = float(K)   # ATM: F0 = K by construction
            iv = implied_bachelier(mc_price, F0_val, K, expiry_yr, A0)
            iv_bp = iv * 10000

            # Diagnostics
            F_T_mean_bp = F_T.mean() * 10000
            F_T_std_bp  = F_T.std()  * 10000
            K_bp        = K * 10000
            pct_near_K  = np.mean(np.abs(F_T - K) < 0.0005) * 100  # within 5 bp of K

            print(f"  {expiry_yr:>6}Y  {K_bp:>8.1f}  {F_T_mean_bp:>14.1f}  "
                  f"{F_T_std_bp:>13.1f}  {mc_price:>11.6f}  {iv_bp:>14.2f}  "
                  f"{pct_near_K:>11.1f}%")

        print()

    # -------------------------------------------------------------------------
    # Key check: does F_T mean ≈ K (martingale property)?
    # -------------------------------------------------------------------------
    print("="*70)
    print("MARTINGALE CHECK — F_T mean should ≈ K for ATM swaptions")
    print("="*70)
    print("  (Under the annuity measure, F_T is a martingale, so E[F_T] = F_0 = K)")
    print("  Large drift in F_T mean -> model is not martingale -> pricing error")
    print()

    for expiry_yr, exp_step in sorted(annual_steps.items()):
        z_at_exp = z_paths[:, exp_step, :]
        with torch.no_grad():
            _, aux_exp = model.decode_from_z(
                z_at_exp, tau=None, do_arb_checks=False, return_aux=True
            )
        P_full_exp = aux_exp["P_full"].detach().cpu().numpy()
        tau_grid_exp = aux_exp["tau_grid"].detach().cpu().numpy()

        for tenor in [1, 5]:
            q0 = time0_forward_swap_and_annuity(P_full_0, tau_grid, expiry_yr, tenor)
            K  = q0["forward_swap"]

            rates = []
            for pi in range(N_PATHS):
                try:
                    res = swap_from_discount_curve_at_expiry(
                        P_full_exp[pi], tau_grid_exp, tenor=tenor
                    )
                    rates.append(res["swap_rate"])
                except Exception:
                    pass

            F_T_arr = np.array(rates)
            drift_bp = (F_T_arr.mean() - K) * 10000
            print(f"  {expiry_yr}Y×{tenor}Y:  K={K*10000:.1f} bp,  "
                  f"E[F_T]={F_T_arr.mean()*10000:.1f} bp,  "
                  f"drift={drift_bp:+.1f} bp,  std={F_T_arr.std()*10000:.1f} bp")

    print()
    print("="*70)
    print("VERDICT")
    print("="*70)
    print()
    print("  If F_T std is large (e.g. 80+ bp) but implied vol is small:")
    print("    -> The payoff E[max(F_T - K, 0)] is small because F_T mean")
    print("       has drifted far above K (paths mostly in-the-money -> low optionality)")
    print("       OR far below K (paths mostly out-of-the-money).")
    print("    -> Check the 'drift' column in the martingale check above.")
    print()
    print("  If drift >> 0: F_T mean > K, paths mostly in-the-money,")
    print("    payoff ≈ (F_T_mean - K) * A — almost deterministic, low vol.")
    print("  If drift << 0: F_T mean < K, paths mostly out-of-the-money,")
    print("    payoff ≈ 0 for most paths, implied vol collapses.")


if __name__ == "__main__":
    main()