import torch

def _hessian_scalar_wrt_z(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    if y.ndim != 1:
        raise ValueError(f"y must be (B,), got {tuple(y.shape)}")
    if z.ndim != 2:
        raise ValueError(f"z must be (B,d), got {tuple(z.shape)}")

    grads = torch.autograd.grad(y.sum(), z, create_graph=True)[0]  # (B,d)

    rows = []
    d = z.shape[1]
    for j in range(d):
        gj = grads[:, j]
        Hj = torch.autograd.grad(gj.sum(), z, create_graph=True)[0]  # (B,d)
        rows.append(Hj)

    return torch.stack(rows, dim=1)  # (B,d,d)


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
    z = z_in.detach().clone().requires_grad_(True)  # (Bz,d)
    Bz, d = z.shape

    tau = tau_grid.to(device=z.device, dtype=z.dtype).detach().clone().requires_grad_(True)  # (N,)

    mu, sigma_or_L, r = model.params_from_z(z)
    if r.ndim == 2 and r.shape[1] == 1:
        r = r.squeeze(1)
    if r.ndim != 1:
        raise ValueError(f"r must be (B,), got {tuple(r.shape)}")

    if sigma_or_L.ndim == 2:
        L = torch.diag_embed(sigma_or_L)   # (Bz,d,d)
    elif sigma_or_L.ndim == 3:
        L = sigma_or_L
    else:
        raise ValueError(f"sigma_or_L must be (B,d) or (B,d,d), got {tuple(sigma_or_L.shape)}")

    P = model.bond_price_from_z_grid(z, tau)  # (Bz,N)

    # dP/dtau: grab diagonal of Jacobian of P wrt tau (one element per maturity)
    dP_dtau_cols = []
    for j in range(tau.numel()):
        g = torch.autograd.grad(P[:, j].sum(), tau, create_graph=True, retain_graph=True)[0]  # (N,)
        dP_dtau_cols.append(g[j])
    dP_dtau = torch.stack(dP_dtau_cols, dim=0).unsqueeze(0).repeat(Bz, 1)  # (Bz,N)

    gradP = []
    HessP = []
    for j in range(tau.numel()):
        Pj = P[:, j]  # (Bz,)
        g1 = torch.autograd.grad(Pj.sum(), z, create_graph=True, retain_graph=True)[0]  # (Bz,d)
        gradP.append(g1)
        H1 = _hessian_scalar_wrt_z(Pj, z)  # (Bz,d,d)
        HessP.append(H1)

    gradP = torch.stack(gradP, dim=1)   # (Bz,N,d)
    HessP = torch.stack(HessP, dim=1)   # (Bz,N,d,d)

    term_tau = -dP_dtau
    term_mu = (gradP * mu.unsqueeze(1)).sum(dim=2)

    LT = L.transpose(1, 2)
    term_trace_list = []
    for j in range(tau.numel()):
        Hj = HessP[:, j]
        quad = LT @ (Hj @ L)
        tr = torch.diagonal(quad, dim1=1, dim2=2).sum(dim=1)
        term_trace_list.append(0.5 * tr)
    term_trace = torch.stack(term_trace_list, dim=1)

    mu_P = term_tau + term_mu + term_trace

    vol_vec = torch.matmul(gradP, L)  # (Bz,N,d)
    vol_price = torch.sqrt(torch.clamp((vol_vec * vol_vec).sum(dim=2), min=eps))  # (Bz,N)

    resid = mu_P - r.unsqueeze(1) * P
    SR = resid / torch.clamp(vol_price, min=eps)
    SR = torch.where(vol_price > vol_floor, SR, torch.nan)

    if debug:
        print("min vol_price:", float(vol_price.min()), "max vol_price:", float(vol_price.max()))
        print("min |resid|:", float(resid.abs().min()), "max |resid|:", float(resid.abs().max()))
        print("vol_price first 5 taus:", vol_price[0, :5].detach().cpu().numpy())
        print("SR first 5 taus:", SR[0, :5].detach().cpu().numpy())

    return SR.detach()