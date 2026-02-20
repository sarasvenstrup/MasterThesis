# Code/utils/simulate_vasicek.py
import numpy as np
import torch

from Code.model.vasicek import Vasicek
from Code.utils.rates import par_swap_from_discount


def simulate_vasicek_states(
    N: int,
    kappa: float,
    theta: float,
    sigma: float,
    r0: float | None = None,
    dt: float = 1.0 / 252.0,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Simulate an OU/Vasicek short-rate state sequence r_t at N dates
    using the exact discretization.

    Returns:
        r: (N,) tensor of states in decimals (e.g. 0.02 = 2%)
    """
    rng = np.random.default_rng(seed)

    # exact transition: r_{t+dt} = theta + (r_t-theta)*exp(-k dt) + eps
    a = np.exp(-kappa * dt)
    var = (sigma**2) * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa + 1e-12)
    std = np.sqrt(max(var, 0.0))

    r = np.empty(N, dtype=float)
    r[0] = theta if r0 is None else float(r0)

    z = rng.standard_normal(N - 1)
    for t in range(1, N):
        r[t] = theta + (r[t - 1] - theta) * a + std * z[t - 1]

    return torch.tensor(r, device=device, dtype=dtype)


def simulate_vasicek_curves(
    N: int,
    tenors: list[int],
    kappa: float = 0.5,
    theta: float = 0.02,
    sigma: float = 0.01,
    r0: float | None = None,
    dt: float = 1.0 / 252.0,
    noise_std_bps: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    return_states: bool = True,
):
    """
    Simulate N swap curves at the given tenors from a Vasicek model.

    Pipeline:
      simulate r_t states (N,)
        -> discount curves P(r_t, 1..T_max)
        -> par swap rates at `tenors`
        -> optionally add iid measurement noise (in bps)

    Args:
      tenors: list of maturities in years (ints), e.g. [1,2,3,5,10,15,20,30]
      noise_std_bps: std dev of additive noise in basis points (on swap rates)
                     (e.g. 1.0 = 1bp noise). Noise is added in *decimal* units.

    Returns:
      If return_states:
        Y: (N,K) numpy array of swap rates (decimals)
        r: (N,) numpy array of simulated short rates (decimals)
      else:
        Y: (N,K) numpy array
    """
    tenors = [int(t) for t in tenors]
    T_max = max(tenors)

    # 1) simulate states
    r_t = simulate_vasicek_states(
        N=N, kappa=kappa, theta=theta, sigma=sigma, r0=r0, dt=dt,
        seed=seed, device=device, dtype=dtype
    )  # (N,)

    # 2) build model + discount curves
    model = Vasicek(kappa=kappa, theta=theta, sigma=sigma, device=torch.device(device), dtype=dtype)

    with torch.no_grad():
        P = model.discount_curve_annual(r_t, T_max=T_max)      # (N, T_max)
        S = par_swap_from_discount(P, tenors)                  # (N, K)

    Y = S.cpu().numpy()

    # 3) add measurement noise (optional)
    if noise_std_bps and noise_std_bps > 0:
        rng = np.random.default_rng(seed + 12345)
        noise_std_dec = noise_std_bps * 1e-4
        Y = Y + rng.normal(loc=0.0, scale=noise_std_dec, size=Y.shape)

    if return_states:
        return Y, r_t.cpu().numpy()
    return Y


if __name__ == "__main__":
    # quick sanity run
    tenors = [1, 2, 3, 5, 10, 15, 20, 30]
    Y, r = simulate_vasicek_curves(
        N=200,
        tenors=tenors,
        kappa=0.7,
        theta=0.02,
        sigma=0.01,
        dt=1/252,
        noise_std_bps=0.0,
        seed=0,
        device="cpu",
    )
    print("Y shape:", Y.shape, "r shape:", r.shape)
    print("first curve (decimals):", Y[0])
    print("first state r0:", r[0])
