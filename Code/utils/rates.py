#%pip install torch
#dbutils.library.restartPython()

import torch

def par_swap_from_discount_old(P, tenors):
    """
    Annual-pay fixed-for-float par swap rate from discount factors.
    P: (B,T) or (T,) discount factors for tau=1..T (annual grid)
    tenors: iterable of ints (e.g. [1,2,3,5,10,15,20,30])

    Returns: (B,len(tenors)) or (len(tenors),)
    """
    if P.dim() == 1:
        P = P.unsqueeze(0)  # (1,T)

    tenors = [int(t) for t in tenors]
    out = []
    for T in tenors:
        idx = T - 1
        PT = P[:, idx]                    # (B,)
        denom = P[:, :idx+1].sum(dim=1)   # (B,)
        S = (1.0 - PT) / denom            # (B,)
        out.append(S)

    return torch.stack(out, dim=1)        # (B, n_tenors)

def par_swap_from_discount(P: torch.Tensor, tenors: list[int]) -> torch.Tensor:
    """
    P: (B,T) with P[:, j] = discount factor for maturity (j+1)
    tenors: list like [1,2,3,5,10,15,20,30]
    returns: (B, len(tenors))
    """
    B, T = P.shape
    device = P.device

    idx = torch.tensor([t - 1 for t in tenors], device=device, dtype=torch.long)  # (K,)
    P_tau = P.index_select(1, idx)                                               # (B,K)

    csum = torch.cumsum(P, dim=1)                                                # (B,T)
    denom = csum.index_select(1, idx)                                            # (B,K)

    S = (1.0 - P_tau) / denom                                                    # (B,K)
    return S
