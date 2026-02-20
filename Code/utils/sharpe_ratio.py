import torch
from Code.utils.ode import (
    d_tau_autograd_nodewise,
    grad_and_trace_cov_hess_G,
    paper_alpha_beta_gamma_trace,
    solve_AB_rk38,
)

def sharpe_ratio_zcb_curve_poulsen(
    model,
    z_in: torch.Tensor,
    tau_grid: torch.Tensor,
    eps: float = 1e-12,
    vol_floor: float = 1e-10,
    debug: bool = False,
) -> torch.Tensor:
    """
    Poulsen-style 'approximate Sharpe ratio' = (mu_P - r P) / ||sigma_P||,
    where mu_P and sigma_P come from Ito on P(z,tau) under Q, and the
    derivatives of P are computed using the paper's identities (eq 17-19)
    and A', B' from the ODE (eq 23-24).

    Returns: SR (B,N) on the given tau_grid.
    """
    model.eval()

    if z_in.ndim == 1:
        z_in = z_in.unsqueeze(0)
    z = z_in  # keep grads if you want; no detach needed for this diagnostic
    Bsz, d = z.shape

    tau = tau_grid.to(device=z.device, dtype=z.dtype)
    if tau.ndim != 1:
        raise ValueError("tau_grid must be 1D (N,)")
    if torch.any(tau[1:] <= tau[:-1]):
        raise ValueError("tau_grid must be strictly increasing")

    # normalize maturity input to decoder
    u = tau / float(model.tau_max)  # (N,)

    # risk-neutral params
    mu, L, r = model.params_from_z(z)  # mu (B,d), L (B,d,d), r (B,)

    # G and its tau-derivative
    G = model.G(z, u)  # (B,N)
    dG_du = d_tau_autograd_nodewise(model.G, z, u)      # (B,N)
    dG_dtau = dG_du / float(model.tau_max)             # (B,N)

    # grad_z G and Tr( Sigma * Hess_G )
    def G_single(z_single):
        return model.G(z_single.unsqueeze(0), u).squeeze(0)  # (N,)
    gradG, tr_S_HG = grad_and_trace_cov_hess_G(G_single, z, L)  # (B,N,d), (B,N)

    # alpha, beta, gamma and solve ODE for A,B
    alpha, beta, gamma = paper_alpha_beta_gamma_trace(
        G=G,
        dG_dtau=dG_dtau,
        grad_z_G=gradG,
        trace_cov_hess=tr_S_HG,
        mu=mu,
        sigma=L,
        r_tilde=r,
    )
    A, Bfun = solve_AB_rk38(tau, alpha, beta, gamma)  # (B,N), (B,N)

    # P
    logP = A - Bfun * G
    P = torch.exp(logP)  # (B,N)

    # ---- Paper identities ----
    # ODE derivatives (eq 23-24): B' = alpha*B + beta ; A' = gamma*B^2
    dB_dtau = alpha * Bfun + beta            # (B,N)
    dA_dtau = gamma * (Bfun ** 2)            # (B,N)

    # (eq 17) tau derivative of P
    dP_dtau = (dA_dtau - G * dB_dtau - Bfun * dG_dtau) * P  # (B,N)

    # (eq 18) gradient in z: ∇P = -B ∇G P
    gradP = -(Bfun.unsqueeze(2) * gradG) * P.unsqueeze(2)    # (B,N,d)

    # diffusion sigma_P = ∇P^T L  (B,N,d) ; its norm
    sigmaP = torch.matmul(gradP, L)                           # (B,N,d)
    volP = torch.sqrt(torch.clamp((sigmaP * sigmaP).sum(dim=2), min=eps))  # (B,N)

    # (eq 19) Hessian structure only needed through Tr( Sigma^T H_P Sigma )
    # Tr(S^T H_P S) = [ B^2 * || S^T ∇G ||^2  - B * Tr(S^T H_G S) ] * P
    SgradG = torch.matmul(gradG, L)                           # (B,N,d)  == (∇G)^T L
    norm_SgradG_sq = (SgradG * SgradG).sum(dim=2)             # (B,N)

    tr_ST_Hp_S = ((Bfun**2) * norm_SgradG_sq - Bfun * tr_S_HG) * P  # (B,N)

    # Ito drift of P under Q:
    # mu_P = dP/dtau + ∇P·mu + 1/2 Tr(S^T H_P S)
    gradP_dot_mu = (gradP * mu.unsqueeze(1)).sum(dim=2)       # (B,N)
    muP = dP_dtau + gradP_dot_mu + 0.5 * tr_ST_Hp_S           # (B,N)

    # Sharpe ratio diagnostic
    resid = muP - r.unsqueeze(1) * P
    SR = resid / torch.clamp(volP, min=eps)
    SR = torch.where(volP > vol_floor, SR, torch.nan)

    if debug:
        print("max |PDE residual|:", float(resid.abs().max()))
        print("min/max volP:", float(volP.min()), float(volP.max()))
        print("max |SR|:", float(torch.nanmax(SR.abs())))

    return SR.detach()