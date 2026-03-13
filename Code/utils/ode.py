import torch
import torch.nn as nn
from torch.func import vmap, jvp, jacfwd

from torchdiffeq import odeint

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
    alpha = (-dG/dtau + (∇G)^T mu + 1/2 Tr[σ^T H(G) σ]) / G
    beta  = r_tilde / G
    gamma = 1/2 Tr[σ^T ∇G ∇G^T σ] = 1/2 || σ^T ∇G ||^2
    """
    if r_tilde.ndim == 1:
        r_tilde = r_tilde.unsqueeze(1)  # (B,1)

    B, d = mu.shape

    # stabiliser division med G
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
    Return dG/dtau evaluated for each z in batch and each tau in tau-grid.

    Assumptions:
      - G_module(z_batch, tau_batch) returns shape (B, 1) or (B,) for tau_batch shape (B,1) or (B,)
      - Here we evaluate pointwise: G(z_i, tau_j)
    Output:
      dG_dtau: (B,N)
    """
    tau = tau.to(device=z.device, dtype=z.dtype)

    def G_scalar(z_single: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        # ensure shapes consistent for your G
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
    G_fn: callable mapping (z_single: (d,)) -> (N,)  OR (N,1) squeezed to (N,)
          (i.e. returns all maturities in one go)

    Returns:
      grad_z_G: (B,N,d)
      trace_cov_hess: (B,N) = Tr[σ^T H(G) σ]  (per maturity)
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


def solve_AB_torchdiffeq(
    tau: torch.Tensor,          # (N,)
    alpha: torch.Tensor,        # (B,N)
    beta: torch.Tensor,         # (B,N)
    gamma: torch.Tensor,        # (B,N)
    adaptive_method: str = "dopri5",
    fallback_method: str = "rk38",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    fallback_step_size: float = 0.25,
):
    """
    Solve the Poulsen A/B system:
        dB/dtau = alpha(tau) * B + beta(tau)
        dA/dtau = gamma(tau) * B^2

    Returns:
        A_vals, B_vals  both of shape (B, N)
    """

    device = alpha.device
    dtype = alpha.dtype
    tau = tau.to(device=device, dtype=dtype)

    Bsz, N = alpha.shape
    assert beta.shape == (Bsz, N)
    assert gamma.shape == (Bsz, N)
    assert tau.shape == (N,)

    # y = [A, B], shape (B, 2)
    y0 = torch.zeros(Bsz, 2, device=device, dtype=dtype)

    def interp_coeff(coeff: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Piecewise-linear interpolation in tau.
        coeff: (B,N)
        t: scalar tensor
        returns: (B,)
        """
        # Handle boundaries explicitly
        if bool(t <= tau[0]):
            return coeff[:, 0]
        if bool(t >= tau[-1]):
            return coeff[:, -1]

        idx = torch.searchsorted(tau, t)
        idx = torch.clamp(idx, 1, N - 1)

        t0 = tau[idx - 1]
        t1 = tau[idx]
        w = (t - t0) / (t1 - t0 + 1e-12)

        return coeff[:, idx - 1] + w * (coeff[:, idx] - coeff[:, idx - 1])

    def rhs(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        t: scalar tensor
        y: (B,2)
        returns dy/dt: (B,2)
        """
        A = y[:, 0]
        Bv = y[:, 1]

        a = interp_coeff(alpha, t)   # (B,)
        b = interp_coeff(beta, t)    # (B,)
        g = interp_coeff(gamma, t)   # (B,)

        dA = g * (Bv ** 2)
        dB = a * Bv + b

        return torch.stack([dA, dB], dim=1)

    # First try adaptive solver
    try:
        sol = odeint(
            rhs,
            y0,
            tau,
            method=adaptive_method,
            rtol=rtol,
            atol=atol,
        )  # (N,B,2)
        used_fallback = False

    except Exception:
        # Fall back to fixed-step rk38
        sol = odeint(
            rhs,
            y0,
            tau,
            method=fallback_method,
            options={"step_size": fallback_step_size},
        )  # (N,B,2)
        used_fallback = True

    sol = sol.permute(1, 0, 2).contiguous()  # (B,N,2)
    A_vals = sol[:, :, 0]
    B_vals = sol[:, :, 1]

    return A_vals, B_vals, used_fallback

import torch
from torch.func import vmap, jvp, jacfwd
from torchdiffeq import odeint


def alpha_beta_gamma_at_t(
    G_module,
    z: torch.Tensor,          # (B,d)
    t: torch.Tensor,          # scalar tensor
    mu: torch.Tensor,         # (B,d)
    sigma: torch.Tensor,      # (B,d,d)
    r_tilde: torch.Tensor,    # (B,) or (B,1)
    eps: float = 1e-4,
):
    """
    Recompute alpha, beta, gamma at a single solver evaluation point t.

    Returns:
        alpha_t, beta_t, gamma_t, G_t
        all shape (B,)
    """
    device = z.device
    dtype = z.dtype
    t = t.to(device=device, dtype=dtype)

    if r_tilde.ndim == 2:
        r_tilde = r_tilde.squeeze(1)   # (B,)

    Bsz, d = z.shape

    # -------------------------
    # G(z,t) pointwise
    # -------------------------
    def G_scalar(z_single: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        out = G_module(z_single.unsqueeze(0), t_scalar.view(1))
        return out.squeeze()

    G_t = vmap(lambda z_single: G_scalar(z_single, t))(z)   # (B,)

    # -------------------------
    # dG/dtau at t
    # -------------------------
    def dG_dtau_single(z_single: torch.Tensor) -> torch.Tensor:
        (_, dG) = jvp(
            lambda tt: G_scalar(z_single, tt),
            (t,),
            (torch.ones_like(t),)
        )
        return dG

    dG_dtau_t = vmap(dG_dtau_single)(z)   # (B,)

    # -------------------------
    # grad_z G at t
    # -------------------------
    def G_of_z(z_single: torch.Tensor) -> torch.Tensor:
        return G_scalar(z_single, t)

    jac_single = jacfwd(G_of_z)           # (d,) -> (d,)
    grad_z_G_t = vmap(jac_single)(z)      # (B,d)

    # -------------------------
    # trace_cov_hess at t
    # -------------------------
    def directional_second_derivative(z_single, v_single):
        def phi(z_):
            J = jac_single(z_)                        # (d,)
            return (J * v_single).sum()              # scalar
        _, jvp_val = jvp(phi, (z_single,), (v_single,))
        return jvp_val                               # scalar

    def trace_for_one(z_single, sigma_single):
        V = sigma_single.T                           # (d,d)
        sec = vmap(lambda v: directional_second_derivative(z_single, v))(V)  # (d,)
        return sec.sum()                             # scalar

    trace_cov_hess_t = vmap(trace_for_one)(z, sigma)   # (B,)

    # -------------------------
    # alpha, beta, gamma
    # -------------------------
    sgn = torch.sign(G_t)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    G_safe = torch.where(G_t.abs() >= eps, G_t, eps * sgn)

    gTmu = (grad_z_G_t * mu).sum(dim=1)              # (B,)
    alpha_t = (-dG_dtau_t + gTmu + 0.5 * trace_cov_hess_t) / G_safe
    beta_t  = r_tilde / G_safe

    v = torch.einsum("bij,bj->bi", sigma.transpose(1, 2), grad_z_G_t)   # (B,d)
    gamma_t = 0.5 * (v ** 2).sum(dim=1)                                 # (B,)

    return alpha_t, beta_t, gamma_t, G_t

def solve_AB_torchdiffeq_recompute(
    tau: torch.Tensor,          # (N,)
    z: torch.Tensor,            # (B,d)
    G_module,
    mu: torch.Tensor,           # (B,d)
    sigma: torch.Tensor,        # (B,d,d)
    r_tilde: torch.Tensor,      # (B,) or (B,1)
    adaptive_method: str = "dopri5",
    fallback_method: str = "rk38",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    fallback_step_size: float = 0.25,
):
    """
    Solve:
        dB/dtau = alpha(t) * B + beta(t)
        dA/dtau = gamma(t) * B^2
    where alpha/beta/gamma are recomputed at every solver evaluation point.
    """
    device = z.device
    dtype = z.dtype
    tau = tau.to(device=device, dtype=dtype)

    Bsz = z.shape[0]
    y0 = torch.zeros(Bsz, 2, device=device, dtype=dtype)   # columns [A, B]

    def rhs(t, y):
        A = y[:, 0]
        Bv = y[:, 1]

        alpha_t, beta_t, gamma_t, _ = alpha_beta_gamma_at_t(
            G_module=G_module,
            z=z,
            t=t,
            mu=mu,
            sigma=sigma,
            r_tilde=r_tilde,
        )

        dA = gamma_t * (Bv ** 2)
        dB = alpha_t * Bv + beta_t

        return torch.stack([dA, dB], dim=1)

    try:
        sol = odeint(
            rhs,
            y0,
            tau,
            method=adaptive_method,
            rtol=rtol,
            atol=atol,
        )
        used_fallback = False

    except Exception:
        sol = odeint(
            rhs,
            y0,
            tau,
            method=fallback_method,
            options={"step_size": fallback_step_size},
        )
        used_fallback = True

    sol = sol.permute(1, 0, 2).contiguous()   # (B,N,2)
    A_vals = sol[:, :, 0]
    B_vals = sol[:, :, 1]
    return A_vals, B_vals, used_fallback