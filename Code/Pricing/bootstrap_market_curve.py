# ==================== Market discount-curve bootstrap ====================
"""
Bootstrap a zero-coupon discount curve P(0, k) from observed par swap rates
at non-uniform tenors.

Used by the pricing scripts that take F_0 / A_0 directly from the market
instead of from the decoder at t=0.  This decouples the strike from the
encoder/decoder roundtrip under the adjusted dynamics — the standard
practitioner approach (calibration to today's curve).

Par swap rate definition (annual fixed leg, single curve):

    S(T) = (1 - P(0, T)) / sum_{k=1}^{T} P(0, k)

Recursion:

    P(0, 1) = 1 / (1 + S(1))
    P(0, T) = (1 - S(T) * sum_{k=1}^{T-1} P(0, k)) / (1 + S(T)),   T >= 2

Inputs may be at irregular tenors (e.g. [1, 2, 3, 5, 10, 15, 20, 30]).
We linearly interpolate the par swap rate curve to all integer tenors
in [1, max_tenor] before bootstrapping year-by-year.
"""

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(float).flatten()
    return np.asarray(x, dtype=float).flatten()


def bootstrap_discount_curve(
    swap_rates,
    swap_tenors,
    max_tenor: int = 20,
):
    """
    Bootstrap P(0, k) for k = 0, 1, ..., max_tenor from market par swap rates.

    Parameters
    ----------
    swap_rates : 1-D tensor / array of par swap rates at tenors `swap_tenors`,
                 in decimal (e.g. 0.025 for 2.5%).
    swap_tenors : 1-D iterable of tenors (years) at which rates are observed,
                  e.g. [1, 2, 3, 5, 10, 15, 20, 30].
    max_tenor : highest tenor required in the output curve (years).

    Returns
    -------
    torch.Tensor of length (max_tenor + 1), with P[0] = 1.0 and
    P[k] = bootstrapped zero-coupon discount factor for k = 1, ..., max_tenor.

    Notes
    -----
    - Linear interpolation in par-swap-rate space between observed tenors.
    - For T below the smallest observed tenor (rare in practice), the rate at
      the smallest tenor is used.  For T above the largest, the rate at the
      largest tenor is used.
    """
    rates = _to_numpy(swap_rates)
    tenors = _to_numpy(swap_tenors)

    order = np.argsort(tenors)
    rates  = rates[order]
    tenors = tenors[order]

    int_tenors = np.arange(1, max_tenor + 1, dtype=float)

    # np.interp handles linear interpolation + endpoint constancy automatically.
    interp_rates = np.interp(int_tenors, tenors, rates)

    P = np.zeros(max_tenor + 1, dtype=float)
    P[0] = 1.0
    cum_prev = 0.0   # sum of P(1) ... P(T-1)
    for T in range(1, max_tenor + 1):
        S_T = interp_rates[T - 1]
        denom = 1.0 + S_T
        if denom <= 0.0:
            raise ValueError(
                f"Non-positive (1 + S(T)) at T={T}: S(T)={S_T}. "
                f"Bootstrap is undefined for S(T) <= -1."
            )
        P_T = (1.0 - S_T * cum_prev) / denom
        if P_T <= 0.0 or not np.isfinite(P_T):
            raise ValueError(
                f"Bootstrap produced non-positive or non-finite P(0,{T}) = "
                f"{P_T}. Input rates may be inconsistent."
            )
        P[T] = P_T
        cum_prev += P_T

    return torch.from_numpy(P).to(dtype=torch.float32)


if __name__ == "__main__":
    # Quick smoke test on the first EUR observation.
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from Code.load_swapdata import my_data

    meta, X_tensor, _, _, tenors, *_ = my_data(use="bbg")
    meta_eur = meta[meta["ccy"] == "EUR"].reset_index(drop=True)
    X_eur = X_tensor[meta["ccy"] == "EUR"]

    rates0 = X_eur[0].numpy()
    P0 = bootstrap_discount_curve(rates0, tenors, max_tenor=20)

    print(f"Date:      {meta_eur.iloc[0]['as_of_date']}")
    print(f"Tenors:    {tenors}")
    print(f"Rates (%): {(rates0*100).round(3)}")
    print()
    print(f"Bootstrapped P(0, k) for k = 0..20:")
    for k in range(21):
        print(f"  P(0, {k:>2d}) = {P0[k].item():.6f}")
    print()

    # Recovery check: at observed tenors, the par swap rate should round-trip
    from Code.Pricing.pricing import forward_swap_rate_torch
    print("Recovery check (forward swap rate at observed tenors, in bp):")
    for T in [1, 2, 3, 5, 10]:
        if T > 20: continue
        # spot swap rate = (1 - P(0,T)) / sum_{k=1}^T P(0,k)
        annuity = P0[1:T+1].sum().item()
        S_T = (1.0 - P0[T].item()) / annuity
        S_mkt = float(np.interp(T, tenors, rates0))
        print(f"  T={T:>2d}Y:  bootstrapped {S_T*1e4:>7.1f} bp,  market {S_mkt*1e4:>7.1f} bp")
