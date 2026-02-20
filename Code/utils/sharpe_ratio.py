import torch

def LP_and_SR_approx_from_model_old(
    model,
    S_in,                    # (B,8) swap inputs (decimals)
    tau_grid,                # (M,) maturities in years, e.g. torch.arange(1,31)
    sigma_bar=0.006,         # Andreasen uses 0.60% as proxy
):
    """
    Computes:
      - LP(z, tau): no-arbitrage residual (should be ~0 if arbitrage-free)
      - SR_approx: Andreasen-style approximate Sharpe ratio = LP / (tau * P * sigma_bar)

    Returns:
      P_tau:    (B,M)
      LP:       (B,M)
      SR_approx:(B,M)
    """
    device = S_in.device
    dtype = S_in.dtype

    # --- 1) Encode and get parameters
    # IMPORTANT: we need z to require grad for ∇P and Hess(P)
    S_in = S_in.requires_grad_(False)
    _, z, _, _, _, _, mu, sigma, r_tilde = model(S_in)  # from your forward()
    z = z.detach().requires_grad_(True)                 # (B,d)
    mu = mu.detach()                                    # (B,d)
    sigma = sigma.detach()                              # (B,d,d)
    r = r_tilde.detach().view(-1, 1)                    # (B,1)

    # --- 2) Build P(z,tau) on arbitrary tau_grid using your internal decoder pieces
    # Your forward gives P only on integer grid 0..tau_max.
    # For SR like the papers, use annual grid 1..30; easiest is to re-run forward on that same grid.
    #
    # Here we assume tau_grid is integer years within [0, tau_max] so we can index your P.
    # If you want non-integer tau later, add a dedicated bond_price_from_z(tau) function.

    with torch.enable_grad():
        # Recompute full forward once, so P exists and is connected to z for autograd.
        # We need P that depends on z; so call the internal pricing path again.
        # Easiest: call model.forward but it recomputes z from S_in.
        # Instead, we rebuild P the same way your forward does is non-trivial here.
        #
        # Practical shortcut (works for annual grid checks):
        # call forward on S_in but then treat z as the encoded state and compute derivatives wrt z
        # using P expressed as P = exp(A - B*G(z,tau)). For that, you need A,B,G as functions of z.
        #
        # Since your forward already outputs A_vals, B_vals, G_vals on tau=0..tau_max,
        # we can use those and re-attach z-grad through G only (A,B were computed using G grads too).
        # If you want the *cleanest* math, implement bond_price_from_z_grid(z, tau_grid) in FullModel.

        # We'll call forward again, but keep graph:
        S_hat, z_fwd, P_full, A_vals, B_vals, G_vals, mu_fwd, sigma_fwd, r_tilde_fwd = model(S_in.requires_grad_(True))

        # Use the z from the graph:
        z = z_fwd  # (B,d) with grad
        mu = mu_fwd
        sigma = sigma_fwd
        r = r_tilde_fwd.view(-1, 1)

        # Select maturities:
        tau_grid = tau_grid.to(device=device, dtype=dtype)
        idx = tau_grid.long()  # assumes integer years
        P_tau = P_full[:, idx]  # (B,M)

    B, M = P_tau.shape
    d = z.shape[1]

    # --- 3) dP/dtau via autograd: treat tau as discrete here (annual grid)
    # For true ∂/∂tau you need a continuous tau implementation.
    # Since papers show annual points, a finite difference is acceptable for plotting.
    # We'll do centered FD on the annual grid:
    #   dP/dtau(tau=k) ≈ (P(k+1)-P(k-1))/2
    dP_dtau = torch.zeros_like(P_tau)
    # forward/backward diff at ends
    dP_dtau[:, 0]  = (P_tau[:, 1] - P_tau[:, 0])
    dP_dtau[:, -1] = (P_tau[:, -1] - P_tau[:, -2])
    if M > 2:
        dP_dtau[:, 1:-1] = 0.5 * (P_tau[:, 2:] - P_tau[:, :-2])

    # --- 4) ∇_z P and Hess(P) contraction: 0.5 Tr(Cov * Hess)
    cov = sigma @ sigma.transpose(1, 2)  # (B,d,d)

    gradP = torch.zeros(B, M, d, device=device, dtype=dtype)
    trace_term = torch.zeros(B, M, device=device, dtype=dtype)

    for m in range(M):
        Pm = P_tau[:, m].sum()
        g = torch.autograd.grad(Pm, z, create_graph=True)[0]   # (B,d)
        gradP[:, m, :] = g

        # Build Hessian entries (d is small: 2 or 3)
        # Hess_{ij} = ∂^2 P / ∂z_i ∂z_j
        H = torch.zeros(B, d, d, device=device, dtype=dtype)
        for i in range(d):
            gi = g[:, i].sum()
            Hi = torch.autograd.grad(gi, z, create_graph=True)[0]  # (B,d)
            H[:, i, :] = Hi

        # 0.5 * Tr(Cov * Hess) = 0.5 * sum_{i,j} Cov_{ij} * Hess_{ij}
        trace_term[:, m] = 0.5 * (cov * H).sum(dim=(1,2))

    # --- 5) LP residual
    drift_term = (gradP * mu.unsqueeze(1)).sum(dim=-1)  # (B,M)
    LP = -dP_dtau - (r * P_tau) + drift_term + trace_term  # (B,M)

    # --- 6) Andreasen approx SR
    tau_safe = tau_grid.clamp_min(1e-6).view(1, -1)
    SR_approx = LP / (tau_safe * P_tau * sigma_bar)

    return P_tau, LP, SR_approx


def LP_and_SR_approx_from_model_fast(
    model,
    S_in,                    # (B,8)
    tau_grid,                # (M,) integer maturities in years (1..30)
    sigma_bar=0.006,
):
    """
    Faster SR diagnostic:
      - Detaches ODE part (A_vals, B_vals) from autograd graph
      - Keeps grad only through encoder + G(z,tau) (much cheaper)
      - Uses same LP formula with FD in tau and exact z-derivatives

    Returns:
      P_tau: (B,M)
      LP:    (B,M)
      SR:    (B,M)
    """
    device = S_in.device
    dtype  = S_in.dtype
    tau_grid = tau_grid.to(device=device, dtype=dtype)
    idx = tau_grid.long()

    # 1) Get A,B and params cheaply (no graph)
    with torch.no_grad():
        # one forward to get A,B on full integer grid 0..tau_max
        out = model(S_in)
        P_full = out[2]      # (B, tau_max+1)
        A_vals = out[3]      # (B, tau_max+1)
        B_vals = out[4]      # (B, tau_max+1)
        mu     = out[6]      # (B,d)
        sigma  = out[7]      # (B,d,d)
        r_tilde= out[8].view(-1, 1)  # (B,1)

        # select annual points
        A_tau = A_vals[:, idx]      # (B,M)
        B_tau = B_vals[:, idx]      # (B,M)

    # 2) Build small graph: z -> G -> P
    #    (A_tau and B_tau are treated as constants here)
    z = model.encoder(S_in).requires_grad_(True)  # (B,d)

    # maturity grid for G uses normalized input u in [0,1]
    # your G expects u = tau / tau_max
    tau_max = float(model.tau_max)
    u = (tau_grid / tau_max).to(device=device, dtype=dtype)  # (M,)
    G_tau = model.G(z, u)  # (B,M)  (make sure your DecoderG supports broadcasting like this)

    P_tau = torch.exp(A_tau - B_tau * G_tau)  # (B,M)

    B, M = P_tau.shape
    d = z.shape[1]

    # 3) dP/dtau via finite differences on annual grid
    dP_dtau = torch.zeros_like(P_tau)
    dP_dtau[:, 0]  = (P_tau[:, 1] - P_tau[:, 0])
    dP_dtau[:, -1] = (P_tau[:, -1] - P_tau[:, -2])
    if M > 2:
        dP_dtau[:, 1:-1] = 0.5 * (P_tau[:, 2:] - P_tau[:, :-2])

    # 4) trace term 0.5 Tr(Cov Hess P) with small d (2 or 3) — now cheap
    cov = sigma @ sigma.transpose(1, 2)  # (B,d,d)   (no grad needed)

    gradP = torch.zeros(B, M, d, device=device, dtype=dtype)
    trace_term = torch.zeros(B, M, device=device, dtype=dtype)

    for m in range(M):
        Pm = P_tau[:, m].sum()
        g = torch.autograd.grad(Pm, z, create_graph=True)[0]  # (B,d)
        gradP[:, m, :] = g

        H = torch.zeros(B, d, d, device=device, dtype=dtype)
        for i in range(d):
            gi = g[:, i].sum()
            Hi = torch.autograd.grad(gi, z, create_graph=True)[0]  # (B,d)
            H[:, i, :] = Hi

        trace_term[:, m] = 0.5 * (cov * H).sum(dim=(1, 2))

    # 5) LP residual
    drift_term = (gradP * mu.unsqueeze(1)).sum(dim=-1)  # (B,M)
    LP = -dP_dtau - (r_tilde * P_tau) + drift_term + trace_term

    # 6) Andreasen SR approx
    tau_safe = tau_grid.clamp_min(1e-6).view(1, -1)
    SR = LP / (tau_safe * P_tau * sigma_bar)

    return P_tau, LP, SR


def LP_and_SR_approx_from_model_hpv_old(
    model,
    S_in,                    # (B,8)
    tau_grid,                # (M,) integer maturities in years (1..30)
    sigma_bar=0.006,
):
    """
    Fast SR diagnostic using Hessian-vector products (HVP), no explicit Hessian.

    - Detaches ODE part: A(tau), B(tau) treated as constants
    - Keeps grad only through encoder + G(z,tau) -> P(z,tau)
    - Computes 0.5 * Tr(Cov * Hess P) via sum_j v_j^T Hess(P) v_j, where v_j are columns of sigma
    """
    device = S_in.device
    dtype  = S_in.dtype
    tau_grid = tau_grid.to(device=device, dtype=dtype)
    idx = tau_grid.long()

    # ---- 1) Get A,B and (mu, sigma, r) cheaply (no graph)
    with torch.no_grad():
        out = model(S_in)
        A_vals = out[3]                      # (B, tau_max+1)
        B_vals = out[4]                      # (B, tau_max+1)
        mu     = out[6]                      # (B,d)
        sigma  = out[7]                      # (B,d,d)  (Cholesky L in your code)
        r_tilde= out[8].view(-1, 1)          # (B,1)

        A_tau = A_vals[:, idx]               # (B,M)
        B_tau = B_vals[:, idx]               # (B,M)

    # ---- 2) Build small graph: z -> G -> P
    z = model.encoder(S_in).requires_grad_(True)  # (B,d)

    tau_max = float(model.tau_max)
    u = (tau_grid / tau_max).to(device=device, dtype=dtype)  # (M,)
    G_tau = model.G(z, u)                                     # (B,M)

    P_tau = torch.exp(A_tau - B_tau * G_tau)                  # (B,M)

    Bsz, M = P_tau.shape
    d = z.shape[1]

    # ---- 3) dP/dtau via finite differences on annual grid
    dP_dtau = torch.zeros_like(P_tau)
    dP_dtau[:, 0]  = (P_tau[:, 1] - P_tau[:, 0])
    dP_dtau[:, -1] = (P_tau[:, -1] - P_tau[:, -2])
    if M > 2:
        dP_dtau[:, 1:-1] = 0.5 * (P_tau[:, 2:] - P_tau[:, :-2])

    # ---- 4) grad term and trace term via HVP
    # drift_term = gradP · mu
    # trace_term = 0.5 * sum_j v_j^T Hess(P) v_j  with v_j = sigma[:, :, j]
    drift_term = torch.zeros(Bsz, M, device=device, dtype=dtype)
    trace_term = torch.zeros(Bsz, M, device=device, dtype=dtype)

    # columns of sigma: (B,d,d) -> list of (B,d)
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    for m in range(M):
        # We need gradients per-sample, so do sum but keep batch structure via grad outputs
        Pm = P_tau[:, m]              # (B,)
        Pm_sum = Pm.sum()

        g = torch.autograd.grad(Pm_sum, z, create_graph=True)[0]  # (B,d)

        drift_term[:, m] = (g * mu).sum(dim=1)

        # HVP trick: for each direction v, compute v^T H v as grad(g·v) · v
        # scalar = (g * v).sum() ; grad(scalar, z) gives (B,d); dot with v per sample
        hvp_sum = torch.zeros(Bsz, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()  # scalar (sums over batch and dims)
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]  # (B,d)
            hvp_sum = hvp_sum + (Hg_v * v).sum(dim=1)  # per-sample v^T H v

        trace_term[:, m] = 0.5 * hvp_sum

    # ---- 5) LP residual and SR
    LP = -dP_dtau - (r_tilde * P_tau) + drift_term + trace_term

    tau_safe = tau_grid.clamp_min(1e-6).view(1, -1)
    SR = LP / (tau_safe * P_tau * sigma_bar)

    return P_tau, LP, SR


def LP_and_SR_approx_from_model(
    model,
    S_in,                    # (B,8)
    tau_max=30,
    sigma_bar=0.006,
):
    """
    Computes SR on maturities 1..tau_max, but computes dP/dtau using a grid 0..tau_max
    to avoid boundary artifacts at tau=1 and tau=tau_max.
    Uses HVP for trace term (fast).
    """
    device = S_in.device
    dtype  = S_in.dtype
    Bsz = S_in.shape[0]

    # ---- tau grid including 0
    tau_full = torch.arange(0, tau_max + 1, device=device, dtype=dtype)   # (Mfull,)
    idx_full = tau_full.long()

    # ---- 1) Get A,B and params (no grad)
    with torch.no_grad():
        out = model(S_in)
        A_vals = out[3]                      # (B, tau_max+1)
        B_vals = out[4]                      # (B, tau_max+1)
        mu     = out[6]                      # (B,d)
        sigma  = out[7]                      # (B,d,d)
        r_tilde= out[8].view(-1, 1)          # (B,1)

        A_full = A_vals[:, idx_full]         # (B, tau_max+1)
        B_full = B_vals[:, idx_full]         # (B, tau_max+1)

    # ---- 2) Build small graph: z -> G -> P on tau_full
    z = model.encoder(S_in).requires_grad_(True)  # (B,d)

    u_full = (tau_full / float(model.tau_max)).to(device=device, dtype=dtype)  # (tau_max+1,)
    G_full = model.G(z, u_full)                                                 # (B, tau_max+1)
    P_full = torch.exp(A_full - B_full * G_full)                                # (B, tau_max+1)

    # ---- 3) dP/dtau on the full grid (central inside, one-sided at ends)
    dP_dtau_full = torch.zeros_like(P_full)
    # dt = 1 year between nodes
    dP_dtau_full[:, 0]  = (P_full[:, 1] - P_full[:, 0])          # forward at 0
    dP_dtau_full[:, -1] = (P_full[:, -1] - P_full[:, -2])        # backward at tau_max
    if tau_max >= 2:
        dP_dtau_full[:, 1:-1] = 0.5 * (P_full[:, 2:] - P_full[:, :-2])

    # now slice maturities 1..tau_max (these are what you plot)
    P_tau   = P_full[:, 1:]            # (B, tau_max)
    dP_dtau = dP_dtau_full[:, 1:]      # (B, tau_max)

    d = z.shape[1]
    # columns of sigma
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    drift_term = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)
    trace_term = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)

    # ---- 4) grad + trace term via HVP per maturity
    for m in range(tau_max):
        Pm = P_tau[:, m]             # (B,)
        g = torch.autograd.grad(Pm.sum(), z, create_graph=True)[0]  # (B,d)

        drift_term[:, m] = (g * mu).sum(dim=1)

        hvp_sum = torch.zeros(Bsz, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()  # scalar
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]  # (B,d)
            hvp_sum = hvp_sum + (Hg_v * v).sum(dim=1)

        trace_term[:, m] = 0.5 * hvp_sum

    # ---- 5) LP and SR
    LP = -dP_dtau - (r_tilde * P_tau) + drift_term + trace_term

    tau_plot = torch.arange(1, tau_max + 1, device=device, dtype=dtype).view(1, -1)
    SR = LP / (tau_plot * P_tau * sigma_bar)

    return P_tau, LP, SR, tau_plot.squeeze(0)