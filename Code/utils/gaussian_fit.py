import numpy as np
import torch
import torch.nn as nn

from Code.utils.rates import par_swap_from_discount
from Code.Vasicek.gaussian import Gaussian2F


def to_decimals(Y: np.ndarray) -> np.ndarray:
    """Convert percent to decimals if needed (heuristic)."""
    return Y / 100.0 if np.nanmean(Y) > 1.0 else Y


def compute_rmse_bps(S_pred: np.ndarray, S_obs: np.ndarray):
    """Return (total_rmse_bps, rmse_by_tenor_bps)."""
    err = (S_pred - S_obs)  # decimals
    total = 1e4 * float(np.sqrt(np.mean(err ** 2)))
    by_tenor = 1e4 * np.sqrt(np.mean(err ** 2, axis=0))
    return total, by_tenor


def fit_optionA_2f(
    curves,
    tenors,
    outer_steps=10,
    inner_steps=80,
    lr_params=5e-2,
    lr_z=1e-1,
    device="cpu",
    dtype=torch.float64,
    print_every=1
):
    """
    Option A: alternating optimization
      - per-date state z_i = (x_i, y_i)
      - global parameters (k1,th1,s1,k2,th2,s2)

    curves: (N,K) swap rates (decimals)
    returns: (model, z_states (N,2), final_loss)
    """
    tenors = [int(t) for t in tenors]
    T_max = max(tenors)

    Y = torch.as_tensor(np.asarray(curves), device=device, dtype=dtype)  # (N,K)
    if Y.dim() != 2:
        raise ValueError(f"curves must be (N,K). Got {tuple(Y.shape)}")
    N, K = Y.shape

    level = float(torch.mean(Y).item())
    model = Gaussian2F(
        kappa1=0.5, theta1=max(level / 2, 1e-8), sigma1=0.01,
        kappa2=0.1, theta2=max(level / 2, 1e-8), sigma2=0.01,
        device=torch.device(device), dtype=dtype
    )

    opt_params = torch.optim.Adam(model.parameters(), lr=lr_params)

    # per-date latent states
    z = torch.zeros((N, 2), device=device, dtype=dtype)

    loss = None
    for outer in range(int(outer_steps)):
        # (1) update z_i per date
        for i in range(N):
            z_i = z[i].detach().clone().requires_grad_(True)  # (2,)
            opt_z = torch.optim.Adam([z_i], lr=lr_z)
            target = Y[i]  # (K,)

            for _ in range(int(inner_steps)):
                opt_z.zero_grad()
                P = model.discount_curve_annual(z_i, T_max=T_max)       # (1,Tmax)
                S_pred = par_swap_from_discount(P, tenors).squeeze(0)   # (K,)
                loss_i = torch.mean((S_pred - target) ** 2)
                loss_i.backward()
                opt_z.step()

            z[i] = z_i.detach()

        # (2) update global params given z
        opt_params.zero_grad()
        P_all = model.discount_curve_annual(z, T_max=T_max)              # (N,Tmax)
        S_all = par_swap_from_discount(P_all, tenors)                    # (N,K)
        loss = torch.mean((S_all - Y) ** 2)
        loss.backward()
        opt_params.step()

        if print_every and ((outer + 1) % int(print_every) == 0):
            print(
                f"[{outer+1}/{outer_steps}] loss={loss.item():.3e} "
                f"k1={model.kappa1.item():.3f} th1={model.theta1.item():.4f} s1={model.sigma1.item():.4f} | "
                f"k2={model.kappa2.item():.3f} th2={model.theta2.item():.4f} s2={model.sigma2.item():.4f}"
            )

    return model, z.detach(), loss.detach()


class LinearEncoder(nn.Module):
    """Simple linear encoder z = W S + b."""
    def __init__(self, in_dim: int, out_dim: int, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True, dtype=dtype).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def fit_optionB_2f(
    curves,                 # (N,K) numpy, decimals preferred
    tenors,                 # list[int]
    n_steps=4000,
    lr=1e-3,
    device="cpu",
    dtype=torch.float64,
    seed=0,
    print_every=500
):
    """
    Option B: end-to-end learn encoder z=g(S) and global params jointly.

    returns: (model, encoder, z (N,2), final_loss)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tenors = [int(t) for t in tenors]
    T_max = max(tenors)

    Y = torch.as_tensor(np.asarray(curves), device=device, dtype=dtype)  # (N,K)
    if Y.dim() != 2:
        raise ValueError(f"curves must be (N,K). Got {tuple(Y.shape)}")
    N, K = Y.shape

    level = float(torch.mean(Y).item())
    model = Gaussian2F(
        kappa1=0.5, theta1=max(level / 2, 1e-8), sigma1=0.01,
        kappa2=0.1, theta2=max(level / 2, 1e-8), sigma2=0.01,
        device=torch.device(device), dtype=dtype
    )

    encoder = LinearEncoder(in_dim=K, out_dim=2, device=device, dtype=dtype)

    opt = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=lr)

    loss = None
    for step in range(int(n_steps)):
        opt.zero_grad()

        z = encoder(Y)                                           # (N,2)
        P = model.discount_curve_annual(z, T_max=T_max)           # (N,Tmax)
        S_pred = par_swap_from_discount(P, tenors)                # (N,K)

        loss = torch.mean((S_pred - Y) ** 2)
        loss.backward()
        opt.step()

        if print_every and ((step + 1) % int(print_every) == 0):
            print(f"[{step+1}/{n_steps}] loss={loss.item():.3e} "
                  f"k1={model.kappa1.item():.3f} k2={model.kappa2.item():.3f}")

    with torch.no_grad():
        z = encoder(Y)

    return model, encoder, z.detach(), loss.detach()
