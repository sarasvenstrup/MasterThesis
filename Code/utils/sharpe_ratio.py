# Code/utils/sharpe_ratio.py
# FINAL (τ in YEARS everywhere; no [0,1] normalization)

import torch


# ============================================================
# Core / reference: Andreasen SR diagnostic (trusted)
# ============================================================
def SR_andreasen_reference(model, S_in, tau_max=30, sigma_bar=0.006, verbose=False):
    """
    Andreasen-style approximate Sharpe ratio for your model with FULL consistency:
      N(τ) = P(τ)   (your decoder ZCB price)
      LN(τ) = -∂τ N - r N + μ·∇_z N + 0.5 Tr( (ΣΣᵀ) Hess_z N )
      SR(τ) = LN(τ) / (τ * N(τ) * sigma_bar)

    Assumes the model uses τ in YEARS (0..tau_max) everywhere.
    """
    device = S_in.device
    dtype  = S_in.dtype
    Bsz    = S_in.shape[0]

    # τ grid including 0 for stable FD on ∂τ
    tau_full = torch.arange(0, tau_max + 1, device=device, dtype=dtype)  # (tau_max+1,)

    # forward WITH graph
    S_hat, z, P_full, A_vals, B_vals, G_vals, mu, sigma, r_tilde = model(S_in.requires_grad_(True))
    # P_full: (B, tau_max+1)

    if verbose:
        print("[gradcheck] z.requires_grad:", z.requires_grad)
        print("[gradcheck] G_vals.requires_grad:", G_vals.requires_grad)
        print("[gradcheck] A_vals.requires_grad:", A_vals.requires_grad)
        print("[gradcheck] B_vals.requires_grad:", B_vals.requires_grad)
        print("[gradcheck] P_full.requires_grad:", P_full.requires_grad)

        Az = torch.autograd.grad(A_vals.sum(), z, retain_graph=True, allow_unused=True)[0]
        Bz = torch.autograd.grad(B_vals.sum(), z, retain_graph=True, allow_unused=True)[0]
        Gz = torch.autograd.grad(G_vals.sum(), z, retain_graph=True, allow_unused=True)[0]
        print("[gradcheck] ||dA/dz||:", None if Az is None else float(Az.norm().detach().cpu()))
        print("[gradcheck] ||dB/dz||:", None if Bz is None else float(Bz.norm().detach().cpu()))
        print("[gradcheck] ||dG/dz||:", None if Gz is None else float(Gz.norm().detach().cpu()))

    # ∂τ N via FD on τ=0..tau_max (dt=1 year between nodes)
    dP_dtau_full = torch.zeros_like(P_full)
    dP_dtau_full[:, 0]  = (P_full[:, 1] - P_full[:, 0])          # forward at 0
    dP_dtau_full[:, -1] = (P_full[:, -1] - P_full[:, -2])        # backward at tau_max
    if tau_max >= 2:
        dP_dtau_full[:, 1:-1] = 0.5 * (P_full[:, 2:] - P_full[:, :-2])

    # slice to τ=1..tau_max
    N_tau   = P_full[:, 1:]          # (B, tau_max)
    dN_dtau = dP_dtau_full[:, 1:]    # (B, tau_max)

    r = r_tilde.view(-1, 1)          # (B,1)
    d = z.shape[1]

    # HVP trace term: 0.5 * Σ_j v_j^T Hess(N) v_j   where v_j are columns of sigma (Cholesky)
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    drift_term = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)
    trace_term = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)

    for m in range(tau_max):
        Nm = N_tau[:, m]  # (B,)
        g = torch.autograd.grad(Nm.sum(), z, create_graph=True)[0]  # (B,d)

        drift_term[:, m] = (g * mu).sum(dim=1)

        hvp_sum = torch.zeros(Bsz, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()  # scalar
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]  # (B,d)
            hvp_sum += (Hg_v * v).sum(dim=1)

        trace_term[:, m] = 0.5 * hvp_sum

    LN = -dN_dtau - (r * N_tau) + drift_term + trace_term  # (B, tau_max)

    tau = torch.arange(1, tau_max + 1, device=device, dtype=dtype).view(1, -1)
    SR = LN / (tau * N_tau * sigma_bar)

    return N_tau, LN, SR, tau.squeeze(0)


# ============================================================
# Fast diagnostic: detach A,B and keep graph only through z->G
# ============================================================
def LP_and_SR_approx_from_model_fast(
    model,
    S_in,                    # (B,8)
    tau_grid,                # (M,) integer maturities in YEARS (e.g. 1..30)
    sigma_bar=0.006,
):
    """
    Faster SR diagnostic:
      - Detaches ODE part (A_vals, B_vals) from autograd graph
      - Keeps grad only through encoder + G(z,τ)
      - LP formula with FD in τ and exact z-derivatives (explicit Hessian, small d)

    Assumes model.G expects τ in YEARS (not normalized).
    """
    device = S_in.device
    dtype  = S_in.dtype
    tau_grid = tau_grid.to(device=device, dtype=dtype)
    idx = tau_grid.long()

    # 1) Get A,B and params cheaply (no graph)
    with torch.no_grad():
        out = model(S_in)
        A_vals  = out[3]                      # (B, tau_max+1)
        B_vals  = out[4]                      # (B, tau_max+1)
        mu      = out[6]                      # (B,d)
        sigma   = out[7]                      # (B,d,d)
        r_tilde = out[8].view(-1, 1)          # (B,1)

        A_tau = A_vals[:, idx]                # (B,M)
        B_tau = B_vals[:, idx]                # (B,M)

    # 2) Build small graph: z -> G -> P (A_tau,B_tau treated as constants)
    z = model.encoder(S_in).requires_grad_(True)  # (B,d)

    # IMPORTANT: τ in YEARS directly
    G_tau = model.G(z, tau_grid)                  # (B,M)
    P_tau = torch.exp(A_tau - B_tau * G_tau)      # (B,M)

    Bsz, M = P_tau.shape
    d = z.shape[1]

    # 3) ∂τ P via FD on annual grid
    dP_dtau = torch.zeros_like(P_tau)
    dP_dtau[:, 0]  = (P_tau[:, 1] - P_tau[:, 0])
    dP_dtau[:, -1] = (P_tau[:, -1] - P_tau[:, -2])
    if M > 2:
        dP_dtau[:, 1:-1] = 0.5 * (P_tau[:, 2:] - P_tau[:, :-2])

    # 4) trace term 0.5 Tr(Cov Hess P) (explicit Hessian; small d)
    cov = sigma @ sigma.transpose(1, 2)  # (B,d,d)

    gradP = torch.zeros(Bsz, M, d, device=device, dtype=dtype)
    trace_term = torch.zeros(Bsz, M, device=device, dtype=dtype)

    for m in range(M):
        Pm = P_tau[:, m].sum()
        g = torch.autograd.grad(Pm, z, create_graph=True)[0]  # (B,d)
        gradP[:, m, :] = g

        H = torch.zeros(Bsz, d, d, device=device, dtype=dtype)
        for i in range(d):
            gi = g[:, i].sum()
            Hi = torch.autograd.grad(gi, z, create_graph=True)[0]  # (B,d)
            H[:, i, :] = Hi

        trace_term[:, m] = 0.5 * (cov * H).sum(dim=(1, 2))

    # 5) LP residual and SR
    drift_term = (gradP * mu.unsqueeze(1)).sum(dim=-1)   # (B,M)
    LP = -dP_dtau - (r_tilde * P_tau) + drift_term + trace_term

    tau_safe = tau_grid.clamp_min(1e-6).view(1, -1)
    SR = LP / (tau_safe * P_tau * sigma_bar)

    return P_tau, LP, SR


# ============================================================
# Fast + HVP trace (no explicit Hessian)
# ============================================================
def LP_and_SR_approx_from_model_hvp(
    model,
    S_in,                    # (B,8)
    tau_grid,                # (M,) integer maturities in YEARS (1..30)
    sigma_bar=0.006,
):
    """
    Fast SR diagnostic using Hessian-vector products (HVP), no explicit Hessian.

    - Detaches ODE part: A(τ), B(τ) treated as constants
    - Keeps grad only through encoder + G(z,τ) -> P(z,τ)
    - Computes 0.5 * Tr(Cov * Hess P) via sum_j v_j^T Hess(P) v_j
      where v_j are columns of sigma (Cholesky L)

    Assumes model.G expects τ in YEARS.
    """
    device = S_in.device
    dtype  = S_in.dtype
    tau_grid = tau_grid.to(device=device, dtype=dtype)
    idx = tau_grid.long()

    # ---- 1) Get A,B and params cheaply (no graph)
    with torch.no_grad():
        out = model(S_in)
        A_vals  = out[3]                      # (B, tau_max+1)
        B_vals  = out[4]                      # (B, tau_max+1)
        mu      = out[6]                      # (B,d)
        sigma   = out[7]                      # (B,d,d)
        r_tilde = out[8].view(-1, 1)          # (B,1)

        A_tau = A_vals[:, idx]                # (B,M)
        B_tau = B_vals[:, idx]                # (B,M)

    # ---- 2) Build small graph: z -> G -> P
    z = model.encoder(S_in).requires_grad_(True)  # (B,d)

    G_tau = model.G(z, tau_grid)                  # (B,M)
    P_tau = torch.exp(A_tau - B_tau * G_tau)      # (B,M)

    Bsz, M = P_tau.shape
    d = z.shape[1]
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    # ---- 3) ∂τ P via FD
    dP_dtau = torch.zeros_like(P_tau)
    dP_dtau[:, 0]  = (P_tau[:, 1] - P_tau[:, 0])
    dP_dtau[:, -1] = (P_tau[:, -1] - P_tau[:, -2])
    if M > 2:
        dP_dtau[:, 1:-1] = 0.5 * (P_tau[:, 2:] - P_tau[:, :-2])

    # ---- 4) drift + trace via HVP per maturity
    drift_term = torch.zeros(Bsz, M, device=device, dtype=dtype)
    trace_term = torch.zeros(Bsz, M, device=device, dtype=dtype)

    for m in range(M):
        Pm = P_tau[:, m]  # (B,)
        g = torch.autograd.grad(Pm.sum(), z, create_graph=True)[0]  # (B,d)

        drift_term[:, m] = (g * mu).sum(dim=1)

        hvp_sum = torch.zeros(Bsz, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]
            hvp_sum += (Hg_v * v).sum(dim=1)

        trace_term[:, m] = 0.5 * hvp_sum

    LP = -dP_dtau - (r_tilde * P_tau) + drift_term + trace_term

    tau_safe = tau_grid.clamp_min(1e-6).view(1, -1)
    SR = LP / (tau_safe * P_tau * sigma_bar)

    return P_tau, LP, SR


# ============================================================
# Convenience wrapper: SR on τ=1..tau_max using τ=0..tau_max FD
# ============================================================
def LP_and_SR_approx_from_model(
    model,
    S_in,                    # (B,8)
    tau_max=30,
    sigma_bar=0.006,
):
    """
    Computes SR on maturities 1..tau_max, but computes ∂τP using a grid 0..tau_max
    to avoid boundary artifacts at τ=1 and τ=tau_max.
    Uses HVP for trace term (fast).

    Assumes model.G expects τ in YEARS.
    """
    device = S_in.device
    dtype  = S_in.dtype
    Bsz    = S_in.shape[0]

    tau_full = torch.arange(0, tau_max + 1, device=device, dtype=dtype)

    # 1) A,B and params (no grad)
    with torch.no_grad():
        out = model(S_in)
        A_full  = out[3]                      # (B, tau_max+1)
        B_full  = out[4]                      # (B, tau_max+1)
        mu      = out[6]                      # (B,d)
        sigma   = out[7]                      # (B,d,d)
        r_tilde = out[8].view(-1, 1)          # (B,1)

    # 2) small graph: z -> G -> P on τ=0..tau_max
    z = model.encoder(S_in).requires_grad_(True)
    G_full = model.G(z, tau_full)                                 # (B, tau_max+1)
    P_full = torch.exp(A_full - B_full * G_full)                  # (B, tau_max+1)

    # 3) ∂τP on full grid
    dP_dtau_full = torch.zeros_like(P_full)
    dP_dtau_full[:, 0]  = (P_full[:, 1] - P_full[:, 0])
    dP_dtau_full[:, -1] = (P_full[:, -1] - P_full[:, -2])
    if tau_max >= 2:
        dP_dtau_full[:, 1:-1] = 0.5 * (P_full[:, 2:] - P_full[:, :-2])

    # slice τ=1..tau_max
    P_tau   = P_full[:, 1:]           # (B, tau_max)
    dP_dtau = dP_dtau_full[:, 1:]     # (B, tau_max)

    d = z.shape[1]
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    drift_term = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)
    trace_term = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)

    for m in range(tau_max):
        Pm = P_tau[:, m]
        g = torch.autograd.grad(Pm.sum(), z, create_graph=True)[0]

        drift_term[:, m] = (g * mu).sum(dim=1)

        hvp_sum = torch.zeros(Bsz, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]
            hvp_sum += (Hg_v * v).sum(dim=1)

        trace_term[:, m] = 0.5 * hvp_sum

    LP = -dP_dtau - (r_tilde * P_tau) + drift_term + trace_term

    tau_plot = torch.arange(1, tau_max + 1, device=device, dtype=dtype).view(1, -1)
    SR = LP / (tau_plot * P_tau * sigma_bar)

    return P_tau, LP, SR, tau_plot.squeeze(0)


# ============================================================
# ODE residual diagnostic aligned to paper system (uses YEARS)
# ============================================================
@torch.no_grad()
def SR_from_AB_ode_residuals(model, S_in, tau_max=30, sigma_bar=0.006):
    """
    Diagnostics aligned with the paper ODE system:
      B' = alpha * B + beta
      A' = gamma * B^2

    Returns:
      P_tau:    (B,tau_max)
      LN_ode:   (B,tau_max)  (residual scaled to LN-like magnitude)
      SR:       (B,tau_max)
      tau:      (tau_max,)
    Assumes model uses τ in YEARS in FullModel.forward() and DecoderG.
    """
    device, dtype = S_in.device, S_in.dtype
    tau = torch.arange(0, tau_max + 1, device=device, dtype=dtype)  # 0..T (years)

    # forward (no grad)
    S_hat, z, P_full, A_vals, B_vals, G_vals, mu, sigma, r_tilde = model(S_in)

    from Code.utils.ode import (
        d_tau_autograd_nodewise,
        grad_and_trace_cov_hess_G,
        paper_alpha_beta_gamma_trace,
    )

    def G_single(z_single):
        return model.G(z_single.unsqueeze(0), tau).squeeze(0)

    dG_dtau = d_tau_autograd_nodewise(model.G, z, tau)  # dG/dτ (years)

    grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, sigma)
    alpha, beta, gamma = paper_alpha_beta_gamma_trace(
        G=G_vals,
        dG_dtau=dG_dtau,
        grad_z_G=grad_z_G,
        trace_cov_hess=trace_cov_hess,
        mu=mu,
        sigma=sigma,
        r_tilde=r_tilde,
    )

    # FD A' and B' on 0..T grid
    dA = torch.zeros_like(A_vals)
    dB = torch.zeros_like(B_vals)
    dA[:, 0]  = (A_vals[:, 1] - A_vals[:, 0])
    dA[:, -1] = (A_vals[:, -1] - A_vals[:, -2])
    dB[:, 0]  = (B_vals[:, 1] - B_vals[:, 0])
    dB[:, -1] = (B_vals[:, -1] - B_vals[:, -2])
    if tau_max >= 2:
        dA[:, 1:-1] = 0.5 * (A_vals[:, 2:] - A_vals[:, :-2])
        dB[:, 1:-1] = 0.5 * (B_vals[:, 2:] - B_vals[:, :-2])

    resB = dB - (alpha * B_vals + beta)          # (B,tau_max+1)
    resA = dA - (gamma * (B_vals ** 2))          # (B,tau_max+1)

    LN_ode_full = P_full * (resA.abs() + (resB.abs() * G_vals.abs()))

    P_tau  = P_full[:, 1:]
    LN_ode = LN_ode_full[:, 1:]
    tau_plot = torch.arange(1, tau_max + 1, device=device, dtype=dtype).view(1, -1)

    SR = LN_ode / (tau_plot * P_tau * sigma_bar)
    return P_tau, LN_ode, SR, tau_plot.squeeze(0)

import torch

def SR_andreasen_reference_noFD(model, S_in, tau_max=30, sigma_bar=0.006, verbose=False):
    """
    Same SR diagnostic but WITHOUT finite differences.

    Uses exact:
      dP/dtau = P * (A' - B'G - B*G')
    with:
      B' = alpha*B + beta
      A' = gamma*B^2
    and G' from JVP (d_tau_autograd_nodewise).

    Keeps annual grid tau = 0..tau_max (years).
    """
    device = S_in.device
    dtype  = S_in.dtype
    Bsz    = S_in.shape[0]

    # annual grid
    tau_full = torch.arange(0, tau_max + 1, device=device, dtype=dtype)  # (T,)

    # forward WITH graph
    S_hat, z, P_full, A_vals, B_vals, G_vals, mu, sigma, r_tilde = model(S_in.requires_grad_(True))

    # --- recompute alpha,beta,gamma on the same grid (WITH graph) ---
    from Code.utils.ode import (
        d_tau_autograd_nodewise,
        grad_and_trace_cov_hess_G,
        paper_alpha_beta_gamma_trace,
    )

    # G' (years) from JVP, nodewise on annual grid
    dG_dtau = d_tau_autograd_nodewise(model.G, z, tau_full)  # (B,T)

    # For grad_z_G and trace_cov_hess you already have utilities.
    # Need a single-curve function that returns G(tau_full) for one z.
    def G_single(z_single):
        return model.G(z_single.unsqueeze(0), tau_full).squeeze(0)  # (T,)

    grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, sigma)  # (B,T,d), (B,T)

    alpha, beta, gamma = paper_alpha_beta_gamma_trace(
        G=G_vals,
        dG_dtau=dG_dtau,
        grad_z_G=grad_z_G,
        trace_cov_hess=trace_cov_hess,
        mu=mu,
        sigma=sigma,
        r_tilde=r_tilde,
    )  # each (B,T)

    # ODE-consistent derivatives A', B'
    B_prime = alpha * B_vals + beta          # (B,T)
    A_prime = gamma * (B_vals ** 2)          # (B,T)

    # EXACT dP/dtau from chain rule (no FD)
    dP_dtau_full = P_full * (A_prime - B_prime * G_vals - B_vals * dG_dtau)  # (B,T)

    # slice to τ=1..tau_max (like before)
    N_tau   = P_full[:, 1:]          # (B,tau_max)
    dN_dtau = dP_dtau_full[:, 1:]    # (B,tau_max)

    r = r_tilde.view(-1, 1)          # (B,1)
    d = z.shape[1]
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    # drift + trace terms via HVP on N (keeps FULL dependence through A,B,G)
    drift_term = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)
    trace_term = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)

    for m in range(tau_max):
        Nm = N_tau[:, m]  # (B,)
        g = torch.autograd.grad(Nm.sum(), z, create_graph=True)[0]  # (B,d)
        drift_term[:, m] = (g * mu).sum(dim=1)

        hvp_sum = torch.zeros(Bsz, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]
            hvp_sum += (Hg_v * v).sum(dim=1)

        trace_term[:, m] = 0.5 * hvp_sum

    LN = -dN_dtau - (r * N_tau) + drift_term + trace_term  # (B,tau_max)
    tau = torch.arange(1, tau_max + 1, device=device, dtype=dtype).view(1, -1)
    SR = LN / (tau * N_tau * sigma_bar)

    if verbose:
        print("finite N:", torch.isfinite(N_tau).all().item(),
              "finite dN:", torch.isfinite(dN_dtau).all().item(),
              "finite LN:", torch.isfinite(LN).all().item(),
              "finite SR:", torch.isfinite(SR).all().item())
        print("LN min/max:", float(LN.min().detach().cpu()), float(LN.max().detach().cpu()))
        print("SR min/max:", float(SR.min().detach().cpu()), float(SR.max().detach().cpu()))

    return N_tau, LN, SR, tau.squeeze(0)