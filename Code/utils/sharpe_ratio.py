# Code/analysis/sharpe_ratio.py
# ============================================================
# Zero-coupon bond "Sharpe ratio" diagnostic (paper Figure 3 style)
#
# Computes instantaneous SR for rolling-maturity ZCB price P(z,tau):
#   SR(z,tau) = [ (mu_P - r*P)/P ] / ||sigma_P||
#
# where (under Q):
#   dP = mu_P dt + (∇_z P)^T L dW
#   mu_P = ∂_tau P + (∇P)^T mu + 1/2 Tr( L^T H(P) L )
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
import torch


def _hessian_scalar_wrt_z(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Compute Hessian of scalar y_i wrt z_i for each batch element i.

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


def sharpe_ratio_zcb(
    model,
    z_in: torch.Tensor,
    tau_in: torch.Tensor,
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
    z_in: (B,d) tensor
    tau_in: scalar tensor (), or (B,) tensor (in years)

    Returns
    -------
    SR: (B,) tensor (detached)
    """
    model.eval()

    # --- z with grad ---
    if z_in.ndim == 1:
        z_in = z_in.unsqueeze(0)
    z = z_in.detach().requires_grad_(True)  # (B,d)

    # --- tau with grad ---
    if not torch.is_tensor(tau_in):
        tau_in = torch.tensor(tau_in, dtype=z.dtype, device=z.device)

    if tau_in.ndim == 0:
        tau = tau_in.expand(z.shape[0]).detach().requires_grad_(True)  # (B,)
    elif tau_in.ndim == 1:
        if tau_in.shape[0] == 1:
            tau = tau_in.expand(z.shape[0]).detach().requires_grad_(True)
        elif tau_in.shape[0] == z.shape[0]:
            tau = tau_in.detach().clone().requires_grad_(True)
        else:
            raise ValueError(f"tau_in shape {tuple(tau_in.shape)} incompatible with batch {z.shape[0]}")
    else:
        raise ValueError(f"tau_in must be scalar or (B,), got {tuple(tau_in.shape)}")

    # --- model params under Q ---
    mu, sigma_or_L, r = model.params_from_z(z)  # required wrapper
    mu = torch.zeros_like(mu)

    if r.ndim == 2 and r.shape[1] == 1:
        r = r.squeeze(1)
    if r.ndim != 1:
        raise ValueError(f"r_tilde must be (B,) or (B,1), got {tuple(r.shape)}")

    # interpret sigma_or_L
    if sigma_or_L.ndim == 2:
        # diagonal vols -> diffusion matrix L = diag(sigmas)
        L = torch.diag_embed(sigma_or_L)  # (B,d,d)
    elif sigma_or_L.ndim == 3:
        L = sigma_or_L  # (B,d,d)
    else:
        raise ValueError(f"sigma_or_L must be (B,d) or (B,d,d), got {tuple(sigma_or_L.shape)}")

    # --- bond price P(z,tau) ---
    P = model.bond_price_from_z(z, tau)  # required wrapper
    if P.ndim == 2 and P.shape[1] == 1:
        P = P.squeeze(1)
    if P.ndim != 1:
        raise ValueError(f"P must be (B,) or (B,1), got {tuple(P.shape)}")

    # --- derivatives ---
    dP_dtau = torch.autograd.grad(P.sum(), tau, create_graph=True)[0]  # (B,)
    gradP = torch.autograd.grad(P.sum(), z, create_graph=True)[0]  # (B,d)
    H = _hessian_scalar_wrt_z(P, z)  # (B,d,d)

    # --- drift of P for rolling maturity (IMPORTANT: -∂τP) ---
    term1 = -dP_dtau
    term2 = (gradP * mu).sum(dim=1)

    tmp = torch.matmul(H, L)  # (B,d,d)
    quad = torch.matmul(L.transpose(1, 2), tmp)  # (B,d,d)
    term3 = 0.5 * torch.diagonal(quad, dim1=1, dim2=2).sum(dim=1)

    mu_P = term1 + term2 + term3  # (B,)

    # --- vol vector of P: (∇P)^T L ---
    vol_vec = torch.matmul(gradP.unsqueeze(1), L).squeeze(1)  # (B,d)
    vol_price = _row_norm(vol_vec, eps=eps)  # (B,)

    # --- Sharpe ratio ---
    # Use cancellation-stable form:
    # SR = (mu_P - r*P) / ||(∇P)^T L||
    SR = (mu_P - r * P) / torch.clamp(vol_price, min=eps)

    return SR.detach()

