"""
Autograd helpers for Poulsen ODE coefficients and an RK4 solver for the (A, B)
term-structure system.
"""

import torch
import torch.nn as nn
from torch.func import vmap, jvp, jacfwd

# -------------------------
# alpha/beta/gamma helpers (Poulsen eq. 23–24)
# -------------------------
def paper_alpha_beta_gamma_trace(
    G: torch.Tensor,               # (B,N)
    dG_dtau: torch.Tensor,         # (B,N)
    grad_z_G: torch.Tensor,        # (B,N,d)
    trace_cov_hess: torch.Tensor,  # (B,N)
    mu: torch.Tensor,              # (B,d)
    sigma: torch.Tensor,           # (B,d,d)
    r_tilde: torch.Tensor,         # (B,) or (B,1)
    eps: float = 1e-4
):
    """
    Compute ODE coefficients alpha, beta, gamma from the Poulsen equations.

        alpha = (-dG/dtau + (∇G)^T mu + 1/2 Tr[σ^T H(G) σ]) / G
        beta  = r_tilde / G
        gamma = 1/2 || σ^T ∇G ||^2

    Parameters
    ----------
    G : torch.Tensor, shape (B, N)
        Numeraire values at each maturity.
    dG_dtau : torch.Tensor, shape (B, N)
        Maturity derivative of G.
    grad_z_G : torch.Tensor, shape (B, N, d)
        Gradient of G with respect to the latent state z.
    trace_cov_hess : torch.Tensor, shape (B, N)
        Trace term Tr[sigma^T H(G) sigma] for each maturity.
    mu : torch.Tensor, shape (B, d)
        Drift vector.
    sigma : torch.Tensor, shape (B, d, d)
        Diffusion matrix.
    r_tilde : torch.Tensor, shape (B,) or (B, 1)
        Short-rate values.
    eps : float, default 1e-4
        Stabiliser threshold for division by G.

    Returns
    -------
    alpha : torch.Tensor, shape (B, N)
    beta : torch.Tensor, shape (B, N)
    gamma : torch.Tensor, shape (B, N)
    """
    if r_tilde.ndim == 1:
        r_tilde = r_tilde.unsqueeze(1)  # (B,1)

    B, d = mu.shape

    # stabilise division by G
    sgn = torch.sign(G)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    G_safe = torch.where(G.abs() >= eps, G, eps * sgn)

    gTmu = (grad_z_G * mu.unsqueeze(1)).sum(dim=2)  # (B,N)

    alpha = (-dG_dtau + gTmu + 0.5 * trace_cov_hess) / G_safe   # (B,N)
    beta  = r_tilde / G_safe                                    # (B,N) via broadcast

    # v = σ^T ∇G  (B,N,d)
    v = torch.einsum("bij,bnj->bni", sigma.transpose(1, 2), grad_z_G)
    gamma = 0.5 * (v ** 2).sum(dim=2)                           # (B,N)

    return alpha, beta, gamma


# -------------------------
# d/dtau via JVP (nodewise)
# -------------------------
def d_tau_autograd_nodewise(G_module: nn.Module, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Return dG/dtau for each (z_i, tau_j) pair using forward-mode autodiff.

    Parameters
    ----------
    G_module : nn.Module
        Module mapping (z_batch, tau_batch) to shape (B, 1) or (B,).
    z : torch.Tensor, shape (B, d)
        Batch of latent states.
    tau : torch.Tensor, shape (N,)
        Maturity grid.

    Returns
    -------
    torch.Tensor, shape (B, N)
        Maturity derivative dG/dtau evaluated pointwise at each (z_i, tau_j).
    """
    tau = tau.to(device=z.device, dtype=z.dtype)

    def G_scalar(z_single: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        # ensure consistent shapes
        out = G_module(z_single.unsqueeze(0), t_scalar.view(1))  # (1,1) or (1,)
        return out.squeeze()

    def dG_one(z_single: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        (_, dG) = jvp(lambda tt: G_scalar(z_single, tt),
                      (t_scalar,),
                      (torch.ones_like(t_scalar),))
        return dG

    def dG_for_one_z(z_single: torch.Tensor) -> torch.Tensor:
        return vmap(lambda t: dG_one(z_single, t))(tau)  # (N,)

    return vmap(dG_for_one_z)(z)  # (B,N)


# -------------------------
# grad + trace(Cov Hess) of G wrt z
# -------------------------
def grad_and_trace_cov_hess_G(G_fn, z: torch.Tensor, sigma: torch.Tensor):
    """
    Compute the gradient and covariance-weighted Hessian trace of G with respect to z.

    Parameters
    ----------
    G_fn : callable
        Maps z_single of shape (d,) to shape (N,) for all maturities at once.
    z : torch.Tensor, shape (B, d)
        Batch of latent states.
    sigma : torch.Tensor, shape (B, d, d)
        Diffusion matrix.

    Returns
    -------
    grad_z_G : torch.Tensor, shape (B, N, d)
        Jacobian of G with respect to z.
    trace_cov_hess : torch.Tensor, shape (B, N)
        Trace Tr[sigma^T H(G) sigma] for each maturity.
    """

    # jac_single: (d,) -> (N,d)
    jac_single = jacfwd(G_fn)
    grad_z_G = vmap(jac_single)(z)  # (B,N,d)

    # directional second derivative helper:
    # returns (N,) for a given direction v (d,)
    def directional_second_derivative(z_single, v_single):
        def phi(z_):
            J = jac_single(z_)  # (N,d)
            return (J * v_single.unsqueeze(0)).sum(dim=1)  # (N,)
        _, jvp_val = jvp(phi, (z_single,), (v_single,))
        return jvp_val  # (N,)

    def trace_for_one(z_single, sigma_single):
        # Use the columns of σ^T as directions v_k
        V = sigma_single.T  # (d,d)
        sec = vmap(lambda v: directional_second_derivative(z_single, v))(V)  # (d,N)
        return sec.sum(dim=0)  # (N,)

    trace_cov_hess = vmap(trace_for_one)(z, sigma)  # (B,N)
    return grad_z_G, trace_cov_hess


# -------------------------
# Poulsen solver: RK4 (3/8 rule)
# -------------------------
def solve_AB(tau: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
    """
    Integrate the Poulsen ODE system for (A, B) using the RK4 3/8 rule.

    Parameters
    ----------
    tau : torch.Tensor, shape (N,)
        Maturity grid.
    alpha : torch.Tensor, shape (B, N)
        ODE coefficient alpha at each maturity.
    beta : torch.Tensor, shape (B, N)
        ODE coefficient beta at each maturity.
    gamma : torch.Tensor, shape (B, N)
        ODE coefficient gamma at each maturity.

    Returns
    -------
    A : torch.Tensor, shape (B, N)
        Integrated A trajectory.
    Bv : torch.Tensor, shape (B, N)
        Integrated B trajectory.
    """
    device = alpha.device
    dtype = alpha.dtype
    tau = tau.to(device=device, dtype=dtype)

    Bsz, N = alpha.shape
    assert beta.shape == (Bsz, N)
    assert gamma.shape == (Bsz, N)
    assert tau.shape[0] == N

    # Build trajectories without in-place writes to a big (B,N) tensor
    A_list = []
    B_list = []

    Acur = torch.zeros(Bsz, device=device, dtype=dtype)
    Bcur = torch.zeros(Bsz, device=device, dtype=dtype)

    A_list.append(Acur)
    B_list.append(Bcur)

    def interp_coeff(coeff, i, tfrac):
        return coeff[:, i] + tfrac * (coeff[:, i + 1] - coeff[:, i])

    for i in range(N - 1):
        h = tau[i + 1] - tau[i]

        # stage 1 (at i)
        a1 = alpha[:, i]
        b1 = beta[:, i]
        g1 = gamma[:, i]
        k1B = a1 * Bcur + b1
        k1A = g1 * (Bcur ** 2)

        # stage 2 (i + 1/3)
        A2 = Acur + h * (1.0 / 3.0) * k1A
        B2 = Bcur + h * (1.0 / 3.0) * k1B
        a2 = interp_coeff(alpha, i, 1.0 / 3.0)
        b2 = interp_coeff(beta,  i, 1.0 / 3.0)
        g2 = interp_coeff(gamma, i, 1.0 / 3.0)
        k2B = a2 * B2 + b2
        k2A = g2 * (B2 ** 2)

        # stage 3 (i + 2/3): (-1/3)*k1 + 1*k2
        A3 = Acur + h * ((-1.0 / 3.0) * k1A + 1.0 * k2A)
        B3 = Bcur + h * ((-1.0 / 3.0) * k1B + 1.0 * k2B)
        a3 = interp_coeff(alpha, i, 2.0 / 3.0)
        b3 = interp_coeff(beta,  i, 2.0 / 3.0)
        g3 = interp_coeff(gamma, i, 2.0 / 3.0)
        k3B = a3 * B3 + b3
        k3A = g3 * (B3 ** 2)

        # stage 4 (i + 1)
        A4 = Acur + h * (1.0 * k1A - 1.0 * k2A + 1.0 * k3A)
        B4 = Bcur + h * (1.0 * k1B - 1.0 * k2B + 1.0 * k3B)
        a4 = alpha[:, i + 1]
        b4 = beta[:, i + 1]
        g4 = gamma[:, i + 1]
        k4B = a4 * B4 + b4
        k4A = g4 * (B4 ** 2)

        # Update current state (no in-place into a big A/B tensor)
        Acur = Acur + h * (1.0/8.0 * k1A + 3.0/8.0 * k2A + 3.0/8.0 * k3A + 1.0/8.0 * k4A)
        Bcur = Bcur + h * (1.0/8.0 * k1B + 3.0/8.0 * k2B + 3.0/8.0 * k3B + 1.0/8.0 * k4B)

        A_list.append(Acur)
        B_list.append(Bcur)

    # (N,B) -> (B,N)
    A = torch.stack(A_list, dim=0).transpose(0, 1).contiguous()
    Bv = torch.stack(B_list, dim=0).transpose(0, 1).contiguous()
    return A, Bv