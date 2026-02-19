# Code/model/vasicek.py
import torch
import torch.nn as nn

class Vasicek(nn.Module):
    def __init__(self, kappa=0.5, theta=0.02, sigma=0.01, dtype=torch.float64, device=None):
        super().__init__()
        device = device or torch.device("cpu")

        self.log_kappa = nn.Parameter(torch.log(torch.tensor(kappa, device=device, dtype=dtype)))
        self.log_theta = nn.Parameter(torch.log(torch.tensor(max(theta, 1e-6), device=device, dtype=dtype)))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma, device=device, dtype=dtype)))

    @property
    def kappa(self):
        return torch.exp(self.log_kappa)

    @property
    def theta(self):
        return torch.exp(self.log_theta)

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def zcb_price(self, r0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        r0: (B,) or scalar
        tau: (T,) or scalar
        returns: (B,T)
        """
        k = self.kappa
        s = self.sigma
        th = self.theta

        # robust device/dtype source (always exists):
        dev = self.log_kappa.device
        dty = self.log_kappa.dtype

        r0 = torch.as_tensor(r0, device=dev, dtype=dty)
        tau = torch.as_tensor(tau, device=dev, dtype=dty)

        # force batch dimension on r0
        if r0.dim() == 0:
            r0 = r0.unsqueeze(0)  # (1,)
        elif r0.dim() != 1:
            raise ValueError(f"r0 must be scalar or 1D (B,). Got {tuple(r0.shape)}")

        # force tau to be 1D
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)  # (1,)
        elif tau.dim() != 1:
            raise ValueError(f"tau must be scalar or 1D (T,). Got {tuple(tau.shape)}")

        B = (1.0 - torch.exp(-k * tau)) / (k + 1e-12)  # (T,)
        lnA = (th - (s*s)/(2*k*k)) * (B - tau) - (s*s) * (B*B) / (4*k)
        A = torch.exp(lnA)

        return A.unsqueeze(0) * torch.exp(-B.unsqueeze(0) * r0.unsqueeze(1))

    def discount_curve_annual(self, r0: torch.Tensor, T_max: int) -> torch.Tensor:
        tau = torch.arange(1, T_max + 1, device=self.log_kappa.device, dtype=self.log_kappa.dtype)
        return self.zcb_price(r0, tau)
