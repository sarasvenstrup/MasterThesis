# Code/model/gaussian2f.py
import torch
import torch.nn as nn


class Gaussian2F(nn.Module):
    """
    Two-factor Gaussian affine model as a sum of two independent Vasicek factors:

        x: dx = k1 (th1 - x) dt + s1 dW1
        y: dy = k2 (th2 - y) dt + s2 dW2

    Short rate:
        r = x + y

    Closed-form ZCB:
        P(z, tau) = exp( lnA1(tau)+lnA2(tau) - B1(tau)*x - B2(tau)*y )
    """

    def __init__(
        self,
        kappa1=0.5, theta1=0.02, sigma1=0.01,
        kappa2=0.1, theta2=0.02, sigma2=0.01,
        dtype=torch.float64,
        device=None
    ):
        super().__init__()
        device = device or torch.device("cpu")

        # log-parameters enforce positivity via exp
        self._log_kappa1 = nn.Parameter(torch.log(torch.tensor(float(kappa1), device=device, dtype=dtype)))
        self._log_sigma1 = nn.Parameter(torch.log(torch.tensor(float(sigma1), device=device, dtype=dtype)))
        self._log_theta1 = nn.Parameter(torch.log(torch.tensor(float(max(theta1, 1e-8)), device=device, dtype=dtype)))

        self._log_kappa2 = nn.Parameter(torch.log(torch.tensor(float(kappa2), device=device, dtype=dtype)))
        self._log_sigma2 = nn.Parameter(torch.log(torch.tensor(float(sigma2), device=device, dtype=dtype)))
        self._log_theta2 = nn.Parameter(torch.log(torch.tensor(float(max(theta2, 1e-8)), device=device, dtype=dtype)))

    @property
    def device(self):
        return self._log_kappa1.device

    @property
    def dtype(self):
        return self._log_kappa1.dtype

    @property
    def kappa1(self): return torch.exp(self._log_kappa1)

    @property
    def sigma1(self): return torch.exp(self._log_sigma1)

    @property
    def theta1(self): return torch.exp(self._log_theta1)

    @property
    def kappa2(self): return torch.exp(self._log_kappa2)

    @property
    def sigma2(self): return torch.exp(self._log_sigma2)

    @property
    def theta2(self): return torch.exp(self._log_theta2)

    @staticmethod
    def _lnA_and_B(kappa: torch.Tensor, theta: torch.Tensor, sigma: torch.Tensor, tau: torch.Tensor):
        """
        Single-factor Vasicek components:
          B(tau) = (1 - exp(-k tau))/k
          lnA(tau)= (theta - sigma^2/(2k^2))*(B - tau) - sigma^2*B^2/(4k)
        """
        B = (1.0 - torch.exp(-kappa * tau)) / (kappa + 1e-12)
        lnA = (theta - (sigma * sigma) / (2 * kappa * kappa)) * (B - tau) - (sigma * sigma) * (B * B) / (4 * kappa)
        return lnA, B

    def zcb_price(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        z: (B,2) with z[:,0]=x, z[:,1]=y OR (2,) -> treated as (1,2)
        tau: (T,) or scalar
        returns: (B,T)
        """
        z = torch.as_tensor(z, device=self.device, dtype=self.dtype)
        tau = torch.as_tensor(tau, device=self.device, dtype=self.dtype)

        if z.dim() == 1:
            z = z[None, :]  # (1,2)
        if z.dim() != 2 or z.shape[1] != 2:
            raise ValueError(f"z must be (B,2). Got {tuple(z.shape)}")

        if tau.dim() == 0:
            tau = tau[None]
        if tau.dim() != 1:
            raise ValueError(f"tau must be scalar or (T,). Got {tuple(tau.shape)}")

        x = z[:, 0]  # (B,)
        y = z[:, 1]  # (B,)

        lnA1, B1 = self._lnA_and_B(self.kappa1, self.theta1, self.sigma1, tau)  # (T,)
        lnA2, B2 = self._lnA_and_B(self.kappa2, self.theta2, self.sigma2, tau)  # (T,)
        lnA = lnA1 + lnA2  # (T,)

        return torch.exp(
            lnA[None, :]
            - B1[None, :] * x[:, None]
            - B2[None, :] * y[:, None]
        )

    def discount_curve_annual(self, z: torch.Tensor, T_max: int) -> torch.Tensor:
        tau = torch.arange(1, int(T_max) + 1, device=self.device, dtype=self.dtype)
        return self.zcb_price(z, tau)
