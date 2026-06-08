"""Swap-rate utilities: compute par swap rates from discount factor curves."""

import torch

def par_swap_from_discount(P: torch.Tensor, tenors: list[int]) -> torch.Tensor:
    """
    Compute par swap rates from a discount factor curve.

    Parameters
    ----------
    P : torch.Tensor, shape (B, T)
        Discount factors, where P[:, j] is the discount factor for maturity j+1.
    tenors : list of int
        Swap tenors in years, e.g. [1, 2, 3, 5, 10, 15, 20, 30].

    Returns
    -------
    torch.Tensor, shape (B, len(tenors))
        Par swap rates for each batch element and tenor.
    """
    B, T = P.shape
    device = P.device

    idx = torch.tensor([t - 1 for t in tenors], device=device, dtype=torch.long)  # (K,)
    P_tau = P.index_select(1, idx)                                               # (B,K)

    csum = torch.cumsum(P, dim=1)                                                # (B,T)
    denom = csum.index_select(1, idx)                                            # (B,K)

    S = (1.0 - P_tau) / denom                                                    # (B,K)
    return S
