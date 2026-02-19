import torch
import numpy as np
from Code.utils.rates import par_swap_from_discount  # <- your existing one
from Code.model.vasicek import Vasicek

def fit_r0_single_curve(
    S_obs: torch.Tensor,
    tenors: list[int],
    kappa=0.5, theta=0.02, sigma=0.01,
    n_steps=800, lr=5e-2
):
    """
    Fit ONLY r0 for a single curve, holding (kappa, theta, sigma) fixed.

    Returns:
      r0_hat (scalar tensor),
      model (Vasicek),
      S_pred (K,),
      loss (scalar)
    """
    device = S_obs.device
    dtype = S_obs.dtype

    model = Vasicek(kappa=kappa, theta=theta, sigma=sigma, device=device, dtype=dtype)

    # good init: start near 1Y observed
    r0 = S_obs[0].detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([r0], lr=lr)
    T_max = max(int(t) for t in tenors)

    S_pred = None
    loss = None

    for _ in range(n_steps):
        opt.zero_grad()
        P = model.discount_curve_annual(r0, T_max=T_max)   # should be (1, T_max)
        if P.dim() == 1:
            P = P.unsqueeze(0)

        S_pred = par_swap_from_discount(P, tenors).squeeze(0)  # (K,)
        loss = torch.mean((S_pred - S_obs) ** 2)

        loss.backward()
        opt.step()

    return r0.detach(), model, S_pred.detach(), loss.detach()


def fit_optionA(curves, tenors, outer_steps=150, inner_steps=200, lr_params=5e-2, lr_r0=5e-2, device="cpu"):
    """
    curves: (N,K) numpy array or tensor of swap rates in decimals
    returns: fitted vasicek model, r0 per date
    """
    tenors = [int(t) for t in tenors]
    T_max = max(tenors)

    Y = torch.tensor(np.asarray(curves), device=device, dtype=torch.float64)  # (N,K)
    N, K = Y.shape

    level = float(torch.mean(Y).item())  # average curve level (decimal)
    model = Vasicek(kappa=0.5, theta=level, sigma=0.01,
                    device=torch.device(device), dtype=torch.float64)

    opt_params = torch.optim.Adam(model.parameters(), lr=lr_params)

    r0s = torch.zeros((N,), device=device, dtype=torch.float64)

    for outer in range(outer_steps):
        # 1) update r0s (per date)
        for i in range(N):
            r0_i = r0s[i].detach().clone().requires_grad_(True)
            opt_r = torch.optim.Adam([r0_i], lr=lr_r0)

            target = Y[i]  # (K,)
            for _ in range(inner_steps):
                opt_r.zero_grad()
                P = model.discount_curve_annual(r0_i, T_max=T_max)      # (Tmax,) -> broadcast to (1,Tmax)
                if P.dim() == 1:
                    P = P.unsqueeze(0)
                S_pred = par_swap_from_discount(P, tenors).squeeze(0)   # (K,)
                loss_i = torch.mean((S_pred - target) ** 2)
                loss_i.backward()
                opt_r.step()

            r0s[i] = r0_i.detach()

        # 2) update global params given r0s
        opt_params.zero_grad()
        P_all = model.discount_curve_annual(r0s, T_max=T_max)          # (N,Tmax)
        S_all = par_swap_from_discount(P_all, tenors)                  # (N,K)
        loss = torch.mean((S_all - Y) ** 2)
        loss.backward()
        opt_params.step()

        if (outer + 1) % 1 == 0:
            print(f"[{outer+1}/{outer_steps}] loss={loss.item():.3e} "
                  f"kappa={model.kappa.item():.4f} theta={model.theta.item():.4f} sigma={model.sigma.item():.4f}")

    return model, r0s
