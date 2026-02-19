import torch

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
