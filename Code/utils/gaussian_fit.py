# Code/utils/gaussian2f_fit.py
import numpy as np
import torch

from Code.utils.rates import par_swap_from_discount
from Code.model.gaussian import Gaussian2F

def fit_optionA_2f(
    curves,
    tenors,
    outer_steps=10,
    inner_steps=80,
    lr_params=5e-2,
    lr_z=1e-1,
    device="cpu"
):
    """
    curves: (N,K) swap rates in decimals
    returns: fitted model, z_states (N,2)
    """
    tenors = [int(t) for t in tenors]
    T_max = max(tenors)

    Y = torch.tensor(np.asarray(curves), device=device, dtype=torch.float64)  # (N,K)
    N, K = Y.shape

    # init levels: split mean level across two thetas
    level = float(torch.mean(Y).item())
    model = Gaussian2F(
        kappa1=0.5, theta1=max(level/2, 1e-6), sigma1=0.01,
        kappa2=0.1, theta2=max(level/2, 1e-6), sigma2=0.01,
        device=torch.device(device), dtype=torch.float64
    )

    opt_params = torch.optim.Adam(model.parameters(), lr=lr_params)

    # states per date: z=(x,y)
    z = torch.zeros((N, 2), device=device, dtype=torch.float64)

    for outer in range(outer_steps):
        # (1) update z per date
        for i in range(N):
            z_i = z[i].detach().clone().requires_grad_(True)  # (2,)
            opt_z = torch.optim.Adam([z_i], lr=lr_z)

            target = Y[i]  # (K,)
            for _ in range(inner_steps):
                opt_z.zero_grad()
                P = model.discount_curve_annual(z_i, T_max=T_max)            # (1,T)
                S_pred = par_swap_from_discount(P, tenors).squeeze(0)        # (K,)
                loss_i = torch.mean((S_pred - target) ** 2)
                loss_i.backward()
                opt_z.step()

            z[i] = z_i.detach()

        # (2) update global params given z
        opt_params.zero_grad()
        P_all = model.discount_curve_annual(z, T_max=T_max)                  # (N,T)
        S_all = par_swap_from_discount(P_all, tenors)                        # (N,K)
        loss = torch.mean((S_all - Y) ** 2)
        loss.backward()
        opt_params.step()

        if (outer + 1) % 1 == 0:
            print(
                f"[{outer+1}/{outer_steps}] loss={loss.item():.3e} "
                f"k1={model.kappa1.item():.3f} th1={model.theta1.item():.4f} s1={model.sigma1.item():.4f} | "
                f"k2={model.kappa2.item():.3f} th2={model.theta2.item():.4f} s2={model.sigma2.item():.4f}"
            )

    return model, z.detach()
