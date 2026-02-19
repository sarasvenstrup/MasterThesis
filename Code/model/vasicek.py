import torch
import torch.nn as nn

class Vasicek(nn.Module):
    """
    1-factor Vasicek short rate:
      dr = kappa*(theta - r) dt + sigma dW

    ZCB price:
      P(0,tau) = A(tau) * exp(-B(tau)*r0)
    """

    def __init__(self, kappa=0.5, theta=0.02, sigma=0.01, dtype=torch.float64, device=None):
        super().__init__()
        device = device or torch.device("cpu")

        # store as unconstrained params; enforce positivity via exp
        self._log_kappa = nn.Parameter(torch.log(torch.tensor(float(kappa), device=device, dtype=dtype)))
        self._log_theta = nn.Parameter(torch.log(torch.tensor(float(max(theta, 1e-8)), device=device, dtype=dtype)))
        self._log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma), device=device, dtype=dtype)))

    @property
    def device(self):
        return self._log_kappa.device

    @property
    def dtype(self):
        return self._log_kappa.dtype

    @property
    def kappa(self):
        return torch.exp(self._log_kappa)

    @property
    def theta(self):
        return torch.exp(self._log_theta)

    @property
    def sigma(self):
        return torch.exp(self._log_sigma)

    def zcb_price(self, r0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        r0: (B,) or scalar
        tau: (T,) or scalar
        returns: (B,T)
        """
        r0 = torch.as_tensor(r0, device=self.device, dtype=self.dtype)
        tau = torch.as_tensor(tau, device=self.device, dtype=self.dtype)

        if r0.dim() == 0:
            r0 = r0[None]             # (1,)
        elif r0.dim() != 1:
            raise ValueError(f"r0 must be scalar or (B,). Got {tuple(r0.shape)}")

        if tau.dim() == 0:
            tau = tau[None]           # (1,)
        elif tau.dim() != 1:
            raise ValueError(f"tau must be scalar or (T,). Got {tuple(tau.shape)}")

        k = self.kappa
        th = self.theta
        s = self.sigma

        # B(tau)
        B = (1.0 - torch.exp(-k * tau)) / (k + 1e-12)     # (T,)

        # A(tau) in log-form for stability
        lnA = (th - (s * s) / (2 * k * k)) * (B - tau) - (s * s) * (B * B) / (4 * k)
        A = torch.exp(lnA)                                # (T,)

        # broadcast to (B,T)
        return A[None, :] * torch.exp(-B[None, :] * r0[:, None])

    def discount_curve_annual(self, r0: torch.Tensor, T_max: int) -> torch.Tensor:
        tau = torch.arange(1, int(T_max) + 1, device=self.device, dtype=self.dtype)
        return self.zcb_price(r0, tau)
