import torch
from Code.utils.ode import (
    d_tau_autograd_nodewise,
    grad_and_trace_cov_hess_G,
    paper_alpha_beta_gamma_trace,
    solve_AB_rk38,
)

from Code.utils.ode import d_tau_fd_nodewise

def sharpe_ratio_zcb_curve(
    model,
    z_in: torch.Tensor,
    tau_grid: torch.Tensor,
    eps: float = 1e-12,
    vol_floor: float = 1e-6,
    debug: bool = False,
) -> torch.Tensor:
    model.eval()

    if z_in.ndim == 1:
        z_in = z_in.unsqueeze(0)
    z = z_in.detach().clone()  # diagnostic; remove detach if you want grads
    B, d = z.shape

    tau = tau_grid.detach().clone().to(device=z.device, dtype=z.dtype)
    if tau.ndim != 1:
        raise ValueError("tau_grid must be 1D (N,)")
    if torch.any(tau[1:] <= tau[:-1]):
        raise ValueError("tau_grid must be strictly increasing")

    # === same normalization as your pricer ===
    u = tau / float(model.tau_max)        # (N,)

    # 1) G on grid
    G_vals = model.G(z, u)                # (B,N)

    # 2) risk-neutral params
    mu, L, r = model.params_from_z(z)     # mu (B,d), L (B,d,d), r (B,)

    # 3) dG/dtau via JVP (already in your codebase)
    dG_du = d_tau_autograd_nodewise(model.G, z, u)         # (B,N)
    dG_dtau = dG_du / float(model.tau_max)                 # chain rule

    # 4) grad_z G and Tr(Σ Hess G)
    def G_single(z_single):
        return model.G(z_single.unsqueeze(0), u).squeeze(0)  # (N,)
    grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, L)  # (B,N,d), (B,N)

    # 5) alpha,beta,gamma + solve A,B
    alpha, beta, gamma = paper_alpha_beta_gamma_trace(
        G=G_vals,
        dG_dtau=dG_dtau,
        grad_z_G=grad_z_G,
        trace_cov_hess=trace_cov_hess,
        mu=mu,
        sigma=L,
        r_tilde=r,
    )
    A_vals, B_vals = solve_AB_rk38(tau, alpha, beta, gamma)   # (B,N), (B,N)

    # 6) logP and P
    logP = A_vals - B_vals * G_vals
    P = torch.exp(logP)

    # === Ito pieces in logP-space ===
    # Approximate derivatives wrt z, using that A,B are treated as tau-only in the PDE construction
    grad_u = -B_vals.unsqueeze(2) * grad_z_G                  # (B,N,d)
    trace_u = -B_vals * trace_cov_hess                        # (B,N)

    # Need du/dtau. Since logP is on a grid, use finite diff nodewise (stable)
    # (you already have d_tau_fd_nodewise; use it)
    du_dtau = d_tau_fd_nodewise(logP, tau)                    # (B,N)

    gTmu = (grad_u * mu.unsqueeze(1)).sum(dim=2)              # (B,N)

    # sigma_u = grad_u^T L
    sigma_u = torch.matmul(grad_u, L)                         # (B,N,d)
    vol_u = torch.sqrt(torch.clamp((sigma_u * sigma_u).sum(dim=2), min=eps))  # (B,N)

    mu_u = -du_dtau + gTmu + 0.5 * trace_u                    # (B,N)
    muP_over_P = mu_u + 0.5 * (vol_u ** 2)                    # (B,N)

    # Sharpe (instantaneous)
    SR = (muP_over_P - r.unsqueeze(1)) / torch.clamp(vol_u, min=eps)
    SR = torch.where(vol_u > vol_floor, SR, torch.nan)

    if debug:
        resid = (muP_over_P - r.unsqueeze(1))
        print("max |muP/P - r|:", float(resid.abs().max()))
        print("min/max vol_u:", float(vol_u.min()), float(vol_u.max()))
        print("SR first 5 taus:", SR[0, :5].detach().cpu().numpy())

    return SR.detach()