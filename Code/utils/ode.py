import torch
import torch.nn as nn
from torch.func import vmap, jvp, jacfwd

# -------------------------
# alpha/beta/gamma helpers
# -------------------------
def paper_alpha_beta_gamma_trace(
    G, dG_dtau, grad_z_G, trace_cov_hess, mu, sigma, r_tilde, eps=1e-6
):
    if r_tilde.ndim == 1:
        r_tilde = r_tilde.unsqueeze(1)

    sgn = torch.sign(G)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    G_safe = torch.where(G.abs() >= eps, G, eps * sgn)

    gTmu = (grad_z_G * mu.unsqueeze(1)).sum(dim=2)  # (B,N)
    alpha = (-dG_dtau + gTmu + 0.5 * trace_cov_hess) / G_safe
    beta  = r_tilde / G_safe

    v = torch.einsum("bij,bnj->bni", sigma.transpose(1, 2), grad_z_G)  # (B,N,d)
    gamma = 0.5 * (v ** 2).sum(dim=2)

    return alpha, beta, gamma


# -------------------------
# d/dtau via JVP
# -------------------------
def d_tau_autograd_nodewise(G_module, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    tau = tau.to(device=z.device, dtype=z.dtype)

    def G_scalar(z_single: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        return G_module(z_single.unsqueeze(0), t_scalar.view(1)).squeeze()

    def dG_one(z_single: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        (_, dG) = jvp(lambda tt: G_scalar(z_single, tt),
                      (t_scalar,),
                      (torch.ones_like(t_scalar),))
        return dG

    def dG_for_one_z(z_single: torch.Tensor) -> torch.Tensor:
        return vmap(lambda t: dG_one(z_single, t))(tau)

    return vmap(dG_for_one_z)(z)


# -------------------------
# grad + trace(Cov Hess)
# -------------------------
def grad_and_trace_cov_hess_G(G_fn, z: torch.Tensor, sigma: torch.Tensor):
    B, d = z.shape
    jac_single = jacfwd(G_fn)                 # (d,) -> (N,d)
    grad_z_G = vmap(jac_single)(z)            # (B,N,d)

    def directional_second_derivative(z_single, v_single):
        def phi(z_):
            J = jac_single(z_)  # (N,d)
            return (J * v_single.unsqueeze(0)).sum(dim=1)
        _, jvp_val = jvp(phi, (z_single,), (v_single,))
        return jvp_val  # (N,)

    def trace_for_one(z_single, sigma_single):
        V = sigma_single.T  # (d,d) directions
        sec = vmap(lambda v: directional_second_derivative(z_single, v))(V)  # (d,N)
        return sec.sum(dim=0)  # (N,)

    trace_cov_hess = vmap(trace_for_one)(z, sigma)  # (B,N)
    return grad_z_G, trace_cov_hess


# -------------------------
# Poulsen solver: RK4 (3/8)
# -------------------------
def solve_AB_rk38(tau, alpha, beta, gamma):
    # (your existing solve_AB_rk38 unchanged)
    ...


# -------------------------
# Chen solver: torchdiffeq adjoint
# -------------------------
from torchdiffeq import odeint_adjoint as odeint

class ABOdeFuncInterp(nn.Module):
    def __init__(self, tau_grid, alpha, beta, gamma):
        super().__init__()
        self.tau = tau_grid
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _interp(self, coeff, t):
        tau = self.tau
        t = torch.clamp(t, tau[0], tau[-1])
        i = torch.searchsorted(tau, t, right=True) - 1
        i = torch.clamp(i, 0, tau.numel() - 2)
        t0 = tau[i]
        t1 = tau[i + 1]
        w = (t - t0) / (t1 - t0 + 1e-12)
        c0 = coeff[:, i:i+1]
        c1 = coeff[:, i+1:i+2]
        return c0 + w * (c1 - c0)

    def forward(self, t, y):
        Bcur = y[:, 1:2]
        a = self._interp(self.alpha, t)
        b = self._interp(self.beta,  t)
        g = self._interp(self.gamma, t)
        dB = a * Bcur + b
        dA = g * (Bcur ** 2)
        return torch.cat([dA, dB], dim=1)

def solve_AB_torchdiffeq_adjoint(tau, alpha, beta, gamma, rtol=1e-5, atol=1e-7, method="rk4"):
    tau = tau.to(device=alpha.device, dtype=alpha.dtype)
    Bsz = alpha.shape[0]
    y0 = torch.zeros(Bsz, 2, device=alpha.device, dtype=alpha.dtype)
    func = ABOdeFuncInterp(tau, alpha, beta, gamma)
    y = odeint(func, y0, tau, rtol=rtol, atol=atol, method=method)   # (N,B,2)
    A = y[:, :, 0].transpose(0, 1).contiguous()
    B = y[:, :, 1].transpose(0, 1).contiguous()
    return A, B

def solve_AB(tau, alpha, beta, gamma, solver="rk38"):
    if solver == "rk38":
        return solve_AB_rk38(tau, alpha, beta, gamma)
    elif solver == "chen":
        return solve_AB_torchdiffeq_adjoint(tau, alpha, beta, gamma)
    else:
        raise ValueError("solver must be 'rk38' or 'chen'")