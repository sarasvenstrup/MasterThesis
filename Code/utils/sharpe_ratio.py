# Code/analysis/sharpe_ratio.py
# ============================================================
# Zero-coupon bond "Sharpe ratio" diagnostic (paper Figure 3 style)
#
# Computes instantaneous SR for rolling-maturity ZCB price P(z,tau):
#   SR(z,tau) = (mu_P - r*P) / ||(∇_z P)^T L||
#
# where (under Q) the rolling-maturity drift is:
#   mu_P = -∂_tau P + (∇P)^T mu + 1/2 Tr( L^T H(P) L )
#
# NOTES:
# - This is an arbitrage diagnostic: if the no-arbitrage PDE holds,
#   SR should be ~0 (up to numerical error).
# - Requires your model to expose:
#     model.bond_price_from_z(z, tau) -> P  (autograd-enabled)
#     model.params_from_z(z) -> (mu, sigma_or_L, r_tilde)
#   where sigma_or_L is either:
#     (B,d)   diagonal vols  OR
#     (B,d,d) diffusion matrix L
# ============================================================

from __future__ import annotations
from typing import Dict, Union
import torch


def _hessian_scalar_wrt_z(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Hessian of scalar y_i w.r.t. z_i for each batch element i.

    y: (B,)  scalar per batch item
    z: (B,d) requires_grad=True
    returns: (B,d,d)
    """
    if y.ndim != 1:
        raise ValueError(f"y must be (B,), got {tuple(y.shape)}")
    if z.ndim != 2:
        raise ValueError(f"z must be (B,d), got {tuple(z.shape)}")

    grads = torch.autograd.grad(y.sum(), z, create_graph=True)[0]  # (B,d)

    rows = []
    d = z.shape[1]
    for j in range(d):
        gj = grads[:, j]  # (B,)
        Hj = torch.autograd.grad(gj.sum(), z, create_graph=True)[0]  # (B,d)
        rows.append(Hj)

    return torch.stack(rows, dim=1)  # (B,d,d)


def _row_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: (B,d)
    returns: (B,)
    """
    return torch.sqrt(torch.clamp((x * x).sum(dim=1), min=eps))


def _as_batch_tau(tau_in: Union[torch.Tensor, float, int], B: int, ref: torch.Tensor) -> torch.Tensor:
    """
    Convert tau_in to a (B,) tensor on same device/dtype as ref, with *independent storage*.
    (Avoids .expand() views that can cause confusing gradients.)
    """
    if not torch.is_tensor(tau_in):
        tau_in = torch.tensor(tau_in, dtype=ref.dtype, device=ref.device)

    if tau_in.ndim == 0:
        # allocate real storage, not expand-view
        tau = tau_in.detach().clone().repeat(B).requires_grad_(True)  # (B,)
        return tau

    if tau_in.ndim == 1:
        if tau_in.numel() == 1:
            tau = tau_in.detach().clone().repeat(B).requires_grad_(True)  # (B,)
            return tau
        if tau_in.numel() == B:
            tau = tau_in.detach().clone().requires_grad_(True)  # (B,)
            return tau

        raise ValueError(f"tau_in shape {tuple(tau_in.shape)} incompatible with batch B={B}")

    raise ValueError(f"tau_in must be scalar or (B,) or (1,), got {tuple(tau_in.shape)}")


def sharpe_ratio_zcb(
    model,
    z_in: torch.Tensor,
    tau_in: Union[torch.Tensor, float, int],
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute SR(z,tau) for each batch element.

    Inputs
    ------
    model:
        Must provide:
          - bond_price_from_z(z, tau) -> P (B,) or (B,1), autograd-enabled
          - params_from_z(z) -> (mu, sigma_or_L, r_tilde)
              mu: (B,d)
              sigma_or_L: (B,d) or (B,d,d)
              r_tilde: (B,) or (B,1)
    z_in: (B,d) tensor (or (d,))
    tau_in: scalar tensor/float, or (B,) tensor

    Returns
    -------
    SR: (B,) tensor (detached)
    """
    model.eval()

    # --- z with grad ---
    if z_in.ndim == 1:
        z_in = z_in.unsqueeze(0)
    if z_in.ndim != 2:
        raise ValueError(f"z_in must be (B,d) or (d,), got {tuple(z_in.shape)}")

    z = z_in.detach().clone().requires_grad_(True)  # (B,d)
    B = z.shape[0]

    # --- tau with grad (B,) ---
    tau = _as_batch_tau(tau_in, B=B, ref=z)  # (B,)

    # --- model params under Q ---
    mu, sigma_or_L, r = model.params_from_z(z)

    if r.ndim == 2 and r.shape[1] == 1:
        r = r.squeeze(1)
    if r.ndim != 1:
        raise ValueError(f"r_tilde must be (B,) or (B,1), got {tuple(r.shape)}")

    # interpret sigma_or_L
    if sigma_or_L.ndim == 2:
        L = torch.diag_embed(sigma_or_L)  # (B,d,d)
    elif sigma_or_L.ndim == 3:
        L = sigma_or_L  # (B,d,d)
    else:
        raise ValueError(f"sigma_or_L must be (B,d) or (B,d,d), got {tuple(sigma_or_L.shape)}")

    # --- bond price P(z,tau) ---
    P = model.bond_price_from_z(z, tau)
    if P.ndim == 2 and P.shape[1] == 1:
        P = P.squeeze(1)
    if P.ndim != 1:
        raise ValueError(f"P must be (B,) or (B,1), got {tuple(P.shape)}")

    # --- derivatives ---
    dP_dtau = torch.autograd.grad(P.sum(), tau, create_graph=True)[0]  # (B,)
    gradP = torch.autograd.grad(P.sum(), z, create_graph=True)[0]      # (B,d)
    H = _hessian_scalar_wrt_z(P, z)                                    # (B,d,d)

    # --- drift of P for rolling maturity: -∂τP + (∇P)^T mu + 1/2 Tr(L^T H L) ---
    term_tau = -dP_dtau
    term_mu = (gradP * mu).sum(dim=1)

    tmp = torch.matmul(H, L)                      # (B,d,d)
    quad = torch.matmul(L.transpose(1, 2), tmp)   # (B,d,d)
    term_trace = 0.5 * torch.diagonal(quad, dim1=1, dim2=2).sum(dim=1)

    mu_P = term_tau + term_mu + term_trace  # (B,)

    # --- vol vector of P: (∇P)^T L ---
    vol_vec = torch.matmul(gradP.unsqueeze(1), L).squeeze(1)  # (B,d)
    vol_price = _row_norm(vol_vec, eps=eps)                   # (B,)

    # --- Sharpe ratio (PDE residual scaled by price-vol norm) ---
    SR = (mu_P - r * P) / torch.clamp(vol_price, min=eps)
    return SR.detach()


def sharpe_ratio_zcb_debug(
    model,
    z_in: torch.Tensor,
    tau_in: Union[torch.Tensor, float, int],
    eps: float = 1e-12
) -> Dict[str, torch.Tensor]:
    """
    Returns all intermediate terms for debugging.
    """
    model.eval()

    if z_in.ndim == 1:
        z_in = z_in.unsqueeze(0)
    if z_in.ndim != 2:
        raise ValueError(f"z_in must be (B,d) or (d,), got {tuple(z_in.shape)}")

    z = z_in.detach().clone().requires_grad_(True)  # (B,d)
    B = z.shape[0]

    tau = _as_batch_tau(tau_in, B=B, ref=z)  # (B,)

    mu, sigma_or_L, r = model.params_from_z(z)
    if r.ndim == 2 and r.shape[1] == 1:
        r = r.squeeze(1)

    if sigma_or_L.ndim == 2:
        L = torch.diag_embed(sigma_or_L)
    elif sigma_or_L.ndim == 3:
        L = sigma_or_L
    else:
        raise ValueError(f"sigma_or_L must be (B,d) or (B,d,d), got {tuple(sigma_or_L.shape)}")

    P = model.bond_price_from_z(z, tau)
    if P.ndim == 2 and P.shape[1] == 1:
        P = P.squeeze(1)

    dP_dtau = torch.autograd.grad(P.sum(), tau, create_graph=True)[0]  # (B,)
    gradP = torch.autograd.grad(P.sum(), z, create_graph=True)[0]      # (B,d)
    H = _hessian_scalar_wrt_z(P, z)                                    # (B,d,d)

    term_tau = -dP_dtau
    term_mu = (gradP * mu).sum(dim=1)

    tmp = torch.matmul(H, L)
    quad = torch.matmul(L.transpose(1, 2), tmp)
    term_trace = 0.5 * torch.diagonal(quad, dim1=1, dim2=2).sum(dim=1)

    mu_P = term_tau + term_mu + term_trace

    vol_vec = torch.matmul(gradP.unsqueeze(1), L).squeeze(1)  # (B,d)
    vol_price = _row_norm(vol_vec, eps=eps)

    resid = mu_P - r * P
    SR = resid / torch.clamp(vol_price, min=eps)

    return {
        "P": P.detach(),
        "r": r.detach(),
        "rP": (r * P).detach(),
        "term_tau": term_tau.detach(),
        "term_mu": term_mu.detach(),
        "term_trace": term_trace.detach(),
        "mu_P": mu_P.detach(),
        "resid": resid.detach(),
        "vol_price": vol_price.detach(),
        "SR": SR.detach(),
    }
