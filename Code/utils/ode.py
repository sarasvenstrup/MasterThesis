# utils/ode.py

import torch
from torch.func import grad, vmap, jvp, jacfwd

def grad_and_trace_cov_hess_G(G_fn, z: torch.Tensor, sigma: torch.Tensor):
    """
    Computes:
      grad_z_G: (B,N,d)
      trace_cov_hess: (B,N) where trace_cov_hess[b,n] = Tr( (sigma sigma^T) * Hess_z G_n )

    using HVP/JVP trick (no explicit Hessian).

    Inputs:
      G_fn: (d,) -> (N,)  for a single sample
      z: (B,d)
      sigma: (B,d,d)

    Returns:
      grad_z_G: (B,N,d)
      trace_cov_hess: (B,N)
    """
    B, d = z.shape

    # jacobian (vector output) -> (N,d)
    jac_single = jacfwd(G_fn)  # (d,) -> (N,d)

    # gradient for batch
    grad_z_G = vmap(jac_single)(z)  # (B,N,d)

    # For trace, use: Tr( (sigma sigma^T) H ) = sum_j (sigma[:, :, j]^T H sigma[:, :, j])
    # Each term is directional second derivative along v = sigma_col_j
    def directional_second_derivative(z_single, v_single):
        # returns (N,) of second directional derivatives for each output component
        # compute jvp of (jacobian dotted with v) in direction v:
        # First define phi(z) = jac(z) @ v  -> (N,)
        def phi(z_):
            J = jac_single(z_)      # (N,d)
            return (J * v_single.unsqueeze(0)).sum(dim=1)  # (N,)

        # directional derivative of phi along v is second directional derivative:
        _, jvp_val = jvp(phi, (z_single,), (v_single,))
        return jvp_val  # (N,)

    # vmap over batch and over the two columns j
    # sigma columns: (B,d,2) if d=2 (general d also works)
    V = sigma  # (B,d,d), use its columns as directions

    # compute sum over j of second directional derivs
    def trace_for_one(z_single, sigma_single):
        cols = [sigma_single[:, j] for j in range(d)]  # list of (d,)
        sec = [directional_second_derivative(z_single, v) for v in cols]  # list of (N,)
        return torch.stack(sec, dim=0).sum(dim=0)  # (N,)

    trace_cov_hess = vmap(trace_for_one)(z, sigma)  # (B,N)

    return grad_z_G, trace_cov_hess


# =========================
# Finite differences d/dtau
# =========================
def d_tau_fd_nodewise(G: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Differentiable nodewise approximation of ∂τG on a (possibly non-uniform) grid.
    - central diff on interior nodes
    - one-sided on boundaries
    G:   (batch, N)
    tau: (N,)
    returns: (batch, N)
    """
    if tau.ndim != 1:
        raise ValueError(f"tau must be 1D (N,), got {tau.shape}")
    if G.ndim != 2 or G.shape[1] != tau.numel():
        raise ValueError(f"G must be (batch, N) with N=len(tau). Got G={G.shape}, tau={tau.shape}")

    tau = tau.to(device=G.device, dtype=G.dtype)
    dt = tau[1:] - tau[:-1]
    if torch.any(dt <= 0):
        raise ValueError("tau must be strictly increasing (dt > 0 everywhere).")

    dG = torch.empty_like(G)

    # left boundary: forward
    dG[:, 0] = (G[:, 1] - G[:, 0]) / dt[0]
    # right boundary: backward
    dG[:, -1] = (G[:, -1] - G[:, -2]) / dt[-1]
    # interior: central on non-uniform grid
    dG[:, 1:-1] = (G[:, 2:] - G[:, :-2]) / (tau[2:] - tau[:-2]).unsqueeze(0)
    return dG

def d_tau_autograd_nodewise(G_module, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Exact nodewise ∂G/∂tau via autograd (JVP), no finite differences.

    G_module: DecoderG instance, callable as G_module(z, tau_vec)->(B,N)
    z:   (B,d)
    tau: (N,)   (the same tau you pass into DecoderG, e.g. tau_in in [0,1])

    returns: (B,N) with dG/dtau at each tau node
    """
    tau = tau.to(device=z.device, dtype=z.dtype)

    def G_scalar(z_single: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        # DecoderG expects tau as a 1D vector; use length-1 vector here
        return G_module(z_single.unsqueeze(0), t_scalar.view(1)).squeeze()

    def dG_one(z_single: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        (_, dG) = jvp(
            lambda tt: G_scalar(z_single, tt),
            (t_scalar,),
            (torch.ones_like(t_scalar),)   # direction 1 => d/dtau
        )
        return dG

    # one z -> (N,)
    def dG_for_one_z(z_single: torch.Tensor) -> torch.Tensor:
        return vmap(lambda t: dG_one(z_single, t))(tau)

    # batch -> (B,N)
    return vmap(dG_for_one_z)(z)

def paper_alpha_beta_gamma_trace(
    G, dG_dtau, grad_z_G, trace_cov_hess, mu, sigma, r_tilde, eps=1e-6
):
    if r_tilde.ndim == 1:
        r_tilde = r_tilde.unsqueeze(1)

    G_safe = G + eps * torch.sign(G)
    G_safe = torch.where(G_safe == 0, torch.full_like(G_safe, eps), G_safe)

    gTmu = (grad_z_G * mu.unsqueeze(1)).sum(dim=2)  # (B,N)

    alpha = (-dG_dtau + gTmu + 0.5 * trace_cov_hess) / G_safe
    beta = r_tilde / G_safe

    v = torch.einsum("bij,bnj->bni", sigma.transpose(1, 2), grad_z_G)  # (B,N,d)
    gamma = 0.5 * (v ** 2).sum(dim=2)

    return alpha, beta, gamma


def solve_AB_rk38(
    tau: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
):
    """
    Solve for A(tau), B(tau) on an arbitrary increasing tau grid.

    B' = alpha(t)*B + beta(t),   B(tau[0])=0
    A' = gamma(t)*B^2,          A(tau[0])=0

    Uses RK4 3/8 rule on each interval [tau_i, tau_{i+1}],
    with *linear interpolation* of (alpha,beta,gamma) inside the interval.

    tau:   (N,)
    alpha,beta,gamma: (B, N)
    returns: A,B each (B, N)
    """
    if tau.ndim != 1:
        raise ValueError("tau must be 1D (N,)")
    if alpha.shape != beta.shape or alpha.shape != gamma.shape:
        raise ValueError("alpha,beta,gamma must have same shape (B,N)")
    if alpha.shape[1] != tau.numel():
        raise ValueError("alpha.shape[1] must equal len(tau)")

    # align tau with coeff device/dtype
    tau = tau.to(device=alpha.device, dtype=alpha.dtype)

    Bsz, N = alpha.shape
    dt = tau[1:] - tau[:-1]
    if torch.any(dt <= 0):
        raise ValueError("tau must be strictly increasing")

    # constants on correct device/dtype
    th0  = tau.new_tensor(0.0)
    th13 = tau.new_tensor(1.0 / 3.0)
    th23 = tau.new_tensor(2.0 / 3.0)
    th1  = tau.new_tensor(1.0)

    def coeff(i: int, th: torch.Tensor):
        # th is scalar tensor on correct device/dtype
        a0 = alpha[:, i:i+1]; a1 = alpha[:, i+1:i+2]
        b0 = beta[:,  i:i+1]; b1 = beta[:,  i+1:i+2]
        g0 = gamma[:, i:i+1]; g1 = gamma[:, i+1:i+2]
        a = a0 + th * (a1 - a0)
        b = b0 + th * (b1 - b0)
        g = g0 + th * (g1 - g0)
        return a, b, g

    def rhs(i: int, th: torch.Tensor, Bcur: torch.Tensor):
        a, b, g = coeff(i, th)
        dB = a * Bcur + b
        dA = g * (Bcur ** 2)
        return dA, dB

    Acur = torch.zeros(Bsz, 1, device=alpha.device, dtype=alpha.dtype)
    Bcur = torch.zeros(Bsz, 1, device=alpha.device, dtype=alpha.dtype)

    A_list = [Acur]
    B_list = [Bcur]

    for i in range(N - 1):
        h = dt[i]  # scalar tensor

        k1A, k1B = rhs(i, th0,  Bcur)
        k2A, k2B = rhs(i, th13, Bcur + (h/3.0) * k1B)
        k3A, k3B = rhs(i, th23, Bcur + (2*h/3.0) * k2B)
        k4A, k4B = rhs(i, th1,  Bcur + h * (-k1B + k2B + k3B))

        Acur = Acur + (h/8.0) * (k1A + 3.0*k2A + 3.0*k3A + k4A)
        Bcur = Bcur + (h/8.0) * (k1B + 3.0*k2B + 3.0*k3B + k4B)

        A_list.append(Acur)
        B_list.append(Bcur)

    A = torch.cat(A_list, dim=1)
    B = torch.cat(B_list, dim=1)
    return A, B