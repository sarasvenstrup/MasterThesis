# Code/utils/gaussian2f_fit.py
import numpy as np
import torch
import torch.nn as nn

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


class LinearEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True, dtype=dtype).to(device)

    def forward(self, x):
        return self.lin(x)

def fit_optionB_2f(
    curves,                 # (N,K) numpy, decimals
    tenors,                 # list of maturities
    hidden=None,            # reserved if you later want MLP
    n_steps=5000,
    lr=1e-3,
    device="cpu",
    seed=0,
):
    """
    Option B: learn encoder z_t = g(S_t) + global params jointly (end-to-end).
    Returns: model, encoder, z (N,2), final loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tenors = [int(t) for t in tenors]
    T_max = max(tenors)

    Y = torch.tensor(np.asarray(curves), device=device, dtype=torch.float64)  # (N,K)
    N, K = Y.shape

    # init model params
    level = float(torch.mean(Y).item())
    model = Gaussian2F(
        kappa1=0.5, theta1=max(level/2, 1e-6), sigma1=0.01,
        kappa2=0.1, theta2=max(level/2, 1e-6), sigma2=0.01,
        device=torch.device(device), dtype=torch.float64
    )

    # simplest encoder (linear)
    encoder = LinearEncoder(in_dim=K, out_dim=2, device=device, dtype=torch.float64)

    opt = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=lr)

    loss = None
    for step in range(n_steps):
        opt.zero_grad()

        z = encoder(Y)  # (N,2), uses observed swaps as input
        P = model.discount_curve_annual(z, T_max=T_max)           # (N,T)
        S_pred = par_swap_from_discount(P, tenors)                # (N,K)

        loss = torch.mean((S_pred - Y) ** 2)
        loss.backward()
        opt.step()

        if (step + 1) % 500 == 0:
            print(f"[{step+1}/{n_steps}] loss={loss.item():.3e} "
                  f"k1={model.kappa1.item():.3f} k2={model.kappa2.item():.3f}")

    with torch.no_grad():
        z = encoder(Y)

    return model, encoder, z.detach(), loss.detach()