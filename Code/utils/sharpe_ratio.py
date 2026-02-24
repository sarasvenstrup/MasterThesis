# Code/utils/sharpe_ratio.py
# FINAL (τ in YEARS everywhere; no [0,1] normalization)

import torch

def final_term_2factor_from_model(
    model,
    S_in,                 # (B,8)
    tau_max=30,           # years
    use_no_grad_AB=True,  # True = detach A,B like your fast SR
):
    """
    Returns your 2-factor 'final_term' residual on τ = 1..tau_max (YEARS).

    Implements (vectorized over batch and maturities):
      final_term = -r - A' + G*B' + B*(G' - mu·∇G - 0.5 Tr(Σ^T Hess(G) Σ))
                   + B^2 * 0.5 Tr(Σ^T (∇G∇G^T) Σ)

    Shapes:
      final_term: (B, tau_max)
      Also returns a dict of intermediate terms for debugging.
    """
    device = S_in.device
    dtype  = S_in.dtype
    Bsz = S_in.shape[0]

    # τ grid 0..T (years), and slice 1..T for outputs
    tau_full = torch.arange(0, tau_max + 1, device=device, dtype=dtype)  # (T+1,)
    tau = tau_full[1:]  # (T,)

    # --- 1) Get A,B,mu,sigma,r and (optionally) keep graph ---
    if use_no_grad_AB:
        with torch.no_grad():
            S_hat, z0, P_full, A_vals, B_vals, G_vals0, mu, sigma, r_tilde = model(S_in)
        # build a small graph only through encoder+G (like your fast functions)
        z = model.encoder(S_in).requires_grad_(True)  # (B,d)
        G_full = model.G(z, tau_full)                 # (B,T+1)
        A_full = A_vals                               # constants
        B_full = B_vals                               # constants
    else:
        # full graph through everything
        S_hat, z, P_full, A_full, B_full, G_full, mu, sigma, r_tilde = model(S_in.requires_grad_(True))

    r = r_tilde.view(-1, 1)  # (B,1)
    d = z.shape[1]

    # --- 2) Finite-difference A', B' on annual grid 0..T ---
    def fd_time_derivative(X_full):  # X_full: (B,T+1)
        dX = torch.zeros_like(X_full)
        dX[:, 0]  = (X_full[:, 1] - X_full[:, 0])
        dX[:, -1] = (X_full[:, -1] - X_full[:, -2])
        if tau_max >= 2:
            dX[:, 1:-1] = 0.5 * (X_full[:, 2:] - X_full[:, :-2])
        return dX

    dA_full = fd_time_derivative(A_full)  # (B,T+1)
    dB_full = fd_time_derivative(B_full)  # (B,T+1)

    # slice τ=1..T
    G = G_full[:, 1:]       # (B,T)
    B = B_full[:, 1:]       # (B,T)
    dA = dA_full[:, 1:]     # (B,T)
    dB = dB_full[:, 1:]     # (B,T)

    # --- 3) Compute ∂τG via FD (since your original dy_dm is effectively G') ---
    dG_full = fd_time_derivative(G_full)   # (B,T+1)
    dG = dG_full[:, 1:]                    # (B,T)

    # --- 4) z-derivatives of G: grad_z G and Hess_z G (via HVP for trace terms) ---
    # sigma assumed (B,d,d); use its columns v_j
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    mu_dot_gradG = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)
    trace_hess   = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)
    trace_gradgrad = torch.zeros(Bsz, tau_max, device=device, dtype=dtype)

    # We loop over maturities (T=30) which is cheap.
    for m in range(tau_max):
        Gm = G[:, m]  # (B,)
        g = torch.autograd.grad(Gm.sum(), z, create_graph=True)[0]  # (B,d)

        # mu · ∇G
        mu_dot_gradG[:, m] = (g * mu).sum(dim=1)

        # 0.5 * sum_j v_j^T Hess(G) v_j  (HVP form)
        hvp_sum = torch.zeros(Bsz, device=device, dtype=dtype)
        # 0.5 * sum_j (v_j·∇G)^2  (since v^T (∇G∇G^T) v = (v·∇G)^2)
        gg_sum = torch.zeros(Bsz, device=device, dtype=dtype)

        for v in sigma_cols:
            gv = (g * v).sum()  # scalar
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]  # (B,d)
            hvp_sum += (Hg_v * v).sum(dim=1)

            vg = (g * v).sum(dim=1)  # (B,)
            gg_sum += vg * vg

        trace_hess[:, m]    = 0.5 * hvp_sum
        trace_gradgrad[:, m] = 0.5 * gg_sum

    # --- 5) Assemble your final_term (vectorized over (B,T)) ---
    # final_term = -r - dA + G*dB + B*(dG - mu·∇G - trace_hess) + B^2*trace_gradgrad
    final_term = (
        -r
        - dA
        + G * dB
        + B * (dG - mu_dot_gradG - trace_hess)
        + (B ** 2) * trace_gradgrad
    )  # (B,T)

    debug = {
        "r": r,
        "dA": dA,
        "dB": dB,
        "G": G,
        "B": B,
        "dG": dG,
        "mu_dot_gradG": mu_dot_gradG,
        "trace_hess": trace_hess,
        "trace_gradgrad": trace_gradgrad,
        "tau": tau,
    }

    return final_term, debug

def approx_sharpe_from_final_term(final_term: torch.Tensor, P_1T: torch.Tensor, sigma_bar: float = 0.006):
    """
    Andreasen-style approx Sharpe:
      SR(τ) = LN(τ) / (τ * N(τ) * sigma_bar)

    Here we use:
      LN(τ) := final_term(τ) * N(τ)
    where N(τ)=P(τ) (discount factor).
    This matches the idea: final_term is the normalized PDE residual; multiplying by N
    gives the funded bond drift numerator scale.
    """
    device = final_term.device
    dtype  = final_term.dtype
    tau = torch.arange(1, final_term.shape[1] + 1, device=device, dtype=dtype).view(1, -1)  # (1,T)

    N = P_1T  # (B,T) discount factors for τ=1..T
    LN = final_term * N
    SR = LN / (tau * N * sigma_bar)
    return SR