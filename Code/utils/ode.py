"""
ODE script (Poulsen) – endelig version

- Beregner alpha, beta, gamma som i Poulsen-paper (eq. 23–25).
- Løser (A,B)-ODE med:
    (a) RK4 3/8-regel (forward solve som i paperet)
    (b) "Chen" = adjoint backprop via torchdiffeq, men stadig RK4 forward

Brug:
    A, B = solve_AB(tau_grid, alpha, beta, gamma, solver="rk38")
eller
    A, B = solve_AB(tau_grid, alpha, beta, gamma, solver="chen")

Krav:
    - torch>=2.0 (for torch.func)
    - optional: pip install torchdiffeq  (kun hvis solver="chen")
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
    eps: float = 1e-6
):
    """
    alpha = (-dG/dtau + (∇G)^T mu + 1/2 Tr[σ^T H(G) σ]) / G
    beta  = r_tilde / G
    gamma = 1/2 Tr[σ^T ∇G ∇G^T σ] = 1/2 || σ^T ∇G ||^2
    """
    if r_tilde.ndim == 1:
        r_tilde = r_tilde.unsqueeze(1)  # (B,1)

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
    B, d = z.shape

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
def solve_AB_rk38(tau: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
    """
    Solve the coupled ODEs (Poulsen eq. 23–24) on a fixed grid tau (N,):

        dB/dtau = alpha(tau) * B + beta(tau)
        dA/dtau = gamma(tau) * B^2
        with A(0)=0, B(0)=0

    Inputs:
      tau   : (N,)
      alpha : (B,N)
      beta  : (B,N)
      gamma : (B,N)

    Outputs:
      A, B  : (B,N)
    """
    device = alpha.device
    dtype = alpha.dtype
    tau = tau.to(device=device, dtype=dtype)

    Bsz, N = alpha.shape
    assert beta.shape == (Bsz, N)
    assert gamma.shape == (Bsz, N)
    assert tau.shape[0] == N

    A = torch.zeros(Bsz, N, device=device, dtype=dtype)
    Bv = torch.zeros(Bsz, N, device=device, dtype=dtype)

    # 3/8-rule RK4 tableau:
    # c = [0, 1/3, 2/3, 1]
    # a21 = 1/3
    # a31 = -1/3, a32 = 1
    # a41 = 1, a42 = -1, a43 = 1
    # b  = [1/8, 3/8, 3/8, 1/8]

    def f(i, Acur, Bcur):
        # derivative at grid index i (using coeff at i)
        a = alpha[:, i]
        b = beta[:, i]
        g = gamma[:, i]
        dB = a * Bcur + b
        dA = g * (Bcur ** 2)
        return dA, dB

    for i in range(N - 1):
        h = tau[i + 1] - tau[i]
        # If your tau-grid is uniform annual, h=1; if not, this handles nonuniform.

        Acur = A[:, i]
        Bcur = Bv[:, i]

        # stage 1 at i
        k1A, k1B = f(i, Acur, Bcur)

        # stage 2 at i + 1/3
        A2 = Acur + h * (1.0 / 3.0) * k1A
        B2 = Bcur + h * (1.0 / 3.0) * k1B
        # coeffs: we evaluate by linear interpolation between i and i+1 for better accuracy on nonuniform
        # but simplest is to pick nearest; here we interpolate:
        def interp_coeff(coeff, tfrac):
            return coeff[:, i] + tfrac * (coeff[:, i + 1] - coeff[:, i])

        a2 = interp_coeff(alpha, 1.0 / 3.0)
        b2 = interp_coeff(beta,  1.0 / 3.0)
        g2 = interp_coeff(gamma, 1.0 / 3.0)
        k2B = a2 * B2 + b2
        k2A = g2 * (B2 ** 2)

        # stage 3 at i + 2/3: (-1/3)*k1 + 1*k2
        A3 = Acur + h * ((-1.0 / 3.0) * k1A + 1.0 * k2A)
        B3 = Bcur + h * ((-1.0 / 3.0) * k1B + 1.0 * k2B)
        a3 = interp_coeff(alpha, 2.0 / 3.0)
        b3 = interp_coeff(beta,  2.0 / 3.0)
        g3 = interp_coeff(gamma, 2.0 / 3.0)
        k3B = a3 * B3 + b3
        k3A = g3 * (B3 ** 2)

        # stage 4 at i + 1: 1*k1 -1*k2 +1*k3
        A4 = Acur + h * (1.0 * k1A - 1.0 * k2A + 1.0 * k3A)
        B4 = Bcur + h * (1.0 * k1B - 1.0 * k2B + 1.0 * k3B)
        a4 = alpha[:, i + 1]
        b4 = beta[:, i + 1]
        g4 = gamma[:, i + 1]
        k4B = a4 * B4 + b4
        k4A = g4 * (B4 ** 2)

        # combine
        A[:, i + 1] = Acur + h * (1.0 / 8.0 * k1A + 3.0 / 8.0 * k2A + 3.0 / 8.0 * k3A + 1.0 / 8.0 * k4A)
        Bv[:, i + 1] = Bcur + h * (1.0 / 8.0 * k1B + 3.0 / 8.0 * k2B + 3.0 / 8.0 * k3B + 1.0 / 8.0 * k4B)

    return A, Bv


# -------------------------
# Chen solver: torchdiffeq adjoint (optional)
# -------------------------
try:
    from torchdiffeq import odeint_adjoint as odeint  # Chen-style adjoint backprop
    _HAS_TORCHDIFFEQ = True
except Exception:
    _HAS_TORCHDIFFEQ = False
    odeint = None


class ABOdeFuncInterp(nn.Module):
    """
    Continuous-time wrapper around alpha/beta/gamma defined on tau_grid.
    Uses piecewise-linear interpolation to evaluate coefficients at intermediate t.

    Forward ODE:
        dB = alpha(t)*B + beta(t)
        dA = gamma(t)*B^2
    """

    def __init__(self, tau_grid: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
        super().__init__()
        self.register_buffer("tau", tau_grid)
        # alpha/beta/gamma are per-batch tensors. They must be in graph for gradients.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _interp(self, coeff: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        coeff: (B,N)
        t: scalar tensor
        return: (B,1)
        """
        tau = self.tau
        t = torch.clamp(t, tau[0], tau[-1])

        # i in [0, N-2]
        i = torch.searchsorted(tau, t, right=True) - 1
        i = torch.clamp(i, 0, tau.numel() - 2)

        t0 = tau[i]
        t1 = tau[i + 1]
        w = (t - t0) / (t1 - t0 + 1e-12)

        c0 = coeff[:, i:i + 1]
        c1 = coeff[:, i + 1:i + 2]
        return c0 + w * (c1 - c0)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: (B,2) with [A,B]
        Bcur = y[:, 1:2]  # (B,1)

        a = self._interp(self.alpha, t)
        b = self._interp(self.beta,  t)
        g = self._interp(self.gamma, t)

        dB = a * Bcur + b
        dA = g * (Bcur ** 2)
        return torch.cat([dA, dB], dim=1)


def solve_AB_torchdiffeq_adjoint(
    tau: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    method: str = "rk4",
):
    """
    Uses odeint_adjoint (Chen) for backprop through the solver, while forward uses `method` (rk4).
    """
    if not _HAS_TORCHDIFFEQ:
        raise ImportError("torchdiffeq is not installed. Run: pip install torchdiffeq")

    tau = tau.to(device=alpha.device, dtype=alpha.dtype)
    Bsz = alpha.shape[0]
    y0 = torch.zeros(Bsz, 2, device=alpha.device, dtype=alpha.dtype)

    func = ABOdeFuncInterp(tau, alpha, beta, gamma)

    # odeint returns (N,B,2)
    y = odeint(func, y0, tau, rtol=rtol, atol=atol, method=method)

    # reshape to (B,N)
    A = y[:, :, 0].transpose(0, 1).contiguous()
    B = y[:, :, 1].transpose(0, 1).contiguous()
    return A, B


def solve_AB(tau: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, solver: str = "rk38"):
    """
    solver:
      - "rk38" : Poulsen RK4(3/8) forward, plain autograd through steps
      - "chen" : torchdiffeq adjoint backprop, forward method defaults to rk4
    """
    if solver == "rk38":
        return solve_AB_rk38(tau, alpha, beta, gamma)
    elif solver == "chen":
        return solve_AB_torchdiffeq_adjoint(tau, alpha, beta, gamma, method="rk4")
    else:
        raise ValueError("solver must be 'rk38' or 'chen'")


# -------------------------
# Optional: convenience builder for G(z,tau_grid)
# -------------------------
def eval_G_on_grid(G_module: nn.Module, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Evaluate G(z,tau) on full grid.
    Returns G: (B,N)
    """
    tau = tau.to(device=z.device, dtype=z.dtype)

    def G_for_one_z(z_single: torch.Tensor) -> torch.Tensor:
        # vectorize over tau for one z
        def one_tau(t_scalar):
            out = G_module(z_single.unsqueeze(0), t_scalar.view(1))
            return out.squeeze()
        return vmap(one_tau)(tau)  # (N,)

    return vmap(G_for_one_z)(z)  # (B,N)