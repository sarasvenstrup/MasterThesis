import numpy as np
import torch
from Code.Vasicek.vasicek import Vasicek
from Code.utils.rates import par_swap_from_discount


def _to_tensor_curves(curves, device, dtype) -> torch.Tensor:
    Y = torch.as_tensor(np.asarray(curves), device=device, dtype=dtype)
    if Y.dim() != 2:
        raise ValueError(f"curves must be (N,K). Got {tuple(Y.shape)}")
    return Y


def _maybe_percent_to_decimal(Y_np: np.ndarray) -> np.ndarray:
    return Y_np / 100.0 if np.nanmean(Y_np) > 1.0 else Y_np


def fit_r0_single_curve(
    S_obs: torch.Tensor,
    tenors: list[int],
    kappa=0.5, theta=0.02, sigma=0.01,
    n_steps=800, lr=5e-2,
    tol=1e-12
):
    """
    Fit ONLY r0 for a single curve, holding (kappa, theta, sigma) fixed.
    S_obs: (K,)
    """
    tenors = [int(t) for t in tenors]
    device, dtype = S_obs.device, S_obs.dtype

    model = Vasicek(kappa=kappa, theta=theta, sigma=sigma, device=device, dtype=dtype)

    # init near 1Y
    r0 = S_obs[0].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([r0], lr=lr)
    T_max = max(tenors)

    prev = None
    for _ in range(int(n_steps)):
        opt.zero_grad()
        P = model.discount_curve_annual(r0, T_max=T_max)          # (1,Tmax)
        S_pred = par_swap_from_discount(P, tenors).squeeze(0)     # (K,)
        loss = torch.mean((S_pred - S_obs) ** 2)
        loss.backward()
        opt.step()

        # optional early stop
        if prev is not None and torch.abs(prev - loss) < tol:
            break
        prev = loss.detach()

    return r0.detach(), model, S_pred.detach(), loss.detach()


def fit_global_params_and_r0s(
    curves, tenors,
    outer_steps=150,
    inner_steps=200,
    lr_params=5e-2,
    lr_r0=5e-2,
    device="cpu",
    dtype=torch.float64,
    print_every=1
):
    """
    Alternating optimization ("Option A"):
      - Inner: optimize r0_i per curve i given global params
      - Outer: optimize global params given r0 vector

    curves: (N,K) numpy array or tensor, swap rates in decimals (preferred) or percent.
    returns: (model, r0s_tensor, final_loss_tensor)
    """
    tenors = [int(t) for t in tenors]
    T_max = max(tenors)

    Y = _to_tensor_curves(curves, device=device, dtype=dtype)     # (N,K)
    N, K = Y.shape

    # heuristic init
    level = float(torch.mean(Y).item())
    model = Vasicek(kappa=0.5, theta=max(level, 1e-8), sigma=0.01,
                    device=torch.device(device), dtype=dtype)

    opt_params = torch.optim.Adam(model.parameters(), lr=lr_params)
    r0s = torch.zeros((N,), device=device, dtype=dtype)

    loss = None
    for outer in range(int(outer_steps)):
        # ---- (1) update r0s given global params ----
        for i in range(N):
            r0_i = r0s[i].detach().clone().requires_grad_(True)
            opt_r = torch.optim.Adam([r0_i], lr=lr_r0)
            target = Y[i]  # (K,)

            for _ in range(int(inner_steps)):
                opt_r.zero_grad()
                P = model.discount_curve_annual(r0_i, T_max=T_max)      # (1,Tmax)
                S_pred = par_swap_from_discount(P, tenors).squeeze(0)   # (K,)
                loss_i = torch.mean((S_pred - target) ** 2)
                loss_i.backward()
                opt_r.step()

            r0s[i] = r0_i.detach()

        # ---- (2) update global params given r0s ----
        opt_params.zero_grad()
        P_all = model.discount_curve_annual(r0s, T_max=T_max)           # (N,Tmax)
        S_all = par_swap_from_discount(P_all, tenors)                   # (N,K)
        loss = torch.mean((S_all - Y) ** 2)
        loss.backward()
        opt_params.step()

        if print_every and ((outer + 1) % int(print_every) == 0):
            print(
                f"[{outer+1}/{outer_steps}] loss={loss.item():.3e} "
                f"kappa={model.kappa.item():.4f} theta={model.theta.item():.4f} sigma={model.sigma.item():.4f}"
            )

    return model, r0s, loss.detach()
