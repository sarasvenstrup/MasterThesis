import torch

def _hessian_scalar_wrt_z(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Hessian of scalar y_i w.r.t. z_i for each batch element i.

    y: (B,)  scalar per batch item
    z: (B,d) requires_grad=True
    returns: (B,d,d)
    """
    if y.ndim != 1:
        raise ValueError(f"y must be (B,), got {tuple(y.shape)}")
    if z.ndim != 2:
        raise ValueError(f"z must be (B,d), got {tuple(z.shape)}")

    grads = torch.autograd.grad(y.sum(), z, create_graph=True)[0]  # (B,d)

    rows = []
    d = z.shape[1]
    for j in range(d):
        gj = grads[:, j]  # (B,)
        Hj = torch.autograd.grad(gj.sum(), z, create_graph=True)[0]  # (B,d)
        rows.append(Hj)

    return torch.stack(rows, dim=1)  # (B,d,d)

def sharpe_ratio_zcb_curve(
    model,
    z_in: torch.Tensor,          # (Bz,d)
    tau_grid: torch.Tensor,      # (N,)
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns SR curves: (Bz, N) for each z and each tau in tau_grid.
    Uses model.bond_price_from_z_grid(z, tau_grid) (NO interpolation).
    """
    model.eval()

    if z_in.ndim == 1:
        z_in = z_in.unsqueeze(0)
    z = z_in.detach().clone().requires_grad_(True)  # (Bz,d)
    Bz, d = z.shape

    tau = tau_grid.detach().clone().requires_grad_(True)  # (N,)

    mu, sigma_or_L, r = model.params_from_z(z)
    if r.ndim == 2 and r.shape[1] == 1:
        r = r.squeeze(1)

    if sigma_or_L.ndim == 2:
        L = torch.diag_embed(sigma_or_L)   # (Bz,d,d)
    elif sigma_or_L.ndim == 3:
        L = sigma_or_L
    else:
        raise ValueError(f"sigma_or_L must be (B,d) or (B,d,d), got {tuple(sigma_or_L.shape)}")

    # (Bz,N) no-interp prices
    P = model.bond_price_from_z_grid(z, tau)  # (Bz,N)

    # ∂τ P column-by-column (N is small ~117 so loop is fine)
    dP_dtau_cols = []
    for j in range(tau.numel()):
        # derivative of sum over batch for that column w.r.t tau vector
        g = torch.autograd.grad(P[:, j].sum(), tau, create_graph=True)[0]  # (N,)
        dP_dtau_cols.append(g[j])
    dP_dtau = torch.stack(dP_dtau_cols, dim=0)  # (N,)
    dP_dtau = dP_dtau.unsqueeze(0).repeat(Bz, 1)  # (Bz,N)

    # ∇_z P for each maturity j (loop over N)
    gradP = []
    HessP = []
    for j in range(tau.numel()):
        Pj = P[:, j]  # (Bz,)
        g1 = torch.autograd.grad(Pj.sum(), z, create_graph=True)[0]  # (Bz,d)
        gradP.append(g1)
        H1 = _hessian_scalar_wrt_z(Pj, z)  # (Bz,d,d)
        HessP.append(H1)

    gradP = torch.stack(gradP, dim=1)   # (Bz,N,d)
    HessP = torch.stack(HessP, dim=1)   # (Bz,N,d,d)

    # term_tau = -∂τP
    term_tau = -dP_dtau  # (Bz,N)

    # term_mu = (∇P)^T mu
    term_mu = (gradP * mu.unsqueeze(1)).sum(dim=2)  # (Bz,N)

    # term_trace = 1/2 Tr(L^T H L)
    # compute per maturity
    LT = L.transpose(1, 2)  # (Bz,d,d)
    term_trace_list = []
    for j in range(tau.numel()):
        Hj = HessP[:, j]                 # (Bz,d,d)
        tmp = torch.matmul(Hj, L)        # (Bz,d,d)
        quad = torch.matmul(LT, tmp)     # (Bz,d,d)
        tr = torch.diagonal(quad, dim1=1, dim2=2).sum(dim=1)  # (Bz,)
        term_trace_list.append(0.5 * tr)
    term_trace = torch.stack(term_trace_list, dim=1)  # (Bz,N)

    mu_P = term_tau + term_mu + term_trace  # (Bz,N)

    # vol of P: (∇P)^T L -> (Bz,N,d) then norm over d
    vol_vec = torch.matmul(gradP, L)  # (Bz,N,d)
    vol_price = torch.sqrt(torch.clamp((vol_vec * vol_vec).sum(dim=2), min=eps))  # (Bz,N)

    SR = (mu_P - r.unsqueeze(1) * P) / torch.clamp(vol_price, min=eps)  # (Bz,N)
    return SR.detach()