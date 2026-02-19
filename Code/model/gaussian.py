# Code/model/gaussian2f.py
import torch
import torch.nn as nn

class Gaussian2F(nn.Module):
    """
    Two-factor Gaussian affine model (2x Vasicek factors), short rate:
        r_t = x_t + y_t
    Independent factors, closed-form ZCB price:
        P = exp( lnA1+lnA2 - B1*x - B2*y )
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

        # positive params via log
        self.log_kappa1 = nn.Parameter(torch.log(torch.tensor(kappa1, device=device, dtype=dtype)))
        self.log_sigma1 = nn.Parameter(torch.log(torch.tensor(sigma1, device=device, dtype=dtype)))
        self.log_theta1 = nn.Parameter(torch.log(torch.tensor(max(theta1, 1e-6), device=device, dtype=dtype)))

        self.log_kappa2 = nn.Parameter(torch.log(torch.tensor(kappa2, device=device, dtype=dtype)))
        self.log_sigma2 = nn.Parameter(torch.log(torch.tensor(sigma2, device=device, dtype=dtype)))
        self.log_theta2 = nn.Parameter(torch.log(torch.tensor(max(theta2, 1e-6), device=device, dtype=dtype)))

    @property
    def kappa1(self): return torch.exp(self.log_kappa1)

    @property
    def sigma1(self): return torch.exp(self.log_sigma1)

    @property
    def theta1(self): return torch.exp(self.log_theta1)

    @property
    def kappa2(self): return torch.exp(self.log_kappa2)

    @property
    def sigma2(self): return torch.exp(self.log_sigma2)

    @property
    def theta2(self): return torch.exp(self.log_theta2)

    def _lnA_and_B(self, kappa, theta, sigma, tau):
        """
        Single-factor Vasicek components:
          B(tau) = (1 - exp(-k tau))/k
          lnA(tau)= (theta - sigma^2/(2k^2))*(B - tau) - sigma^2*B^2/(4k)
        """
        B = (1.0 - torch.exp(-kappa * tau)) / (kappa + 1e-12)
        lnA = (theta - (sigma*sigma)/(2*kappa*kappa)) * (B - tau) - (sigma*sigma) * (B*B) / (4*kappa)
        return lnA, B

    def zcb_price(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        z: (B,2) with z[:,0]=x, z[:,1]=y  OR (2,) -> will be treated as (1,2)
        tau: (T,) or scalar
        returns: (B,T)
        """
        dev = self.log_kappa1.device
        dty = self.log_kappa1.dtype

        z = torch.as_tensor(z, device=dev, dtype=dty)
        tau = torch.as_tensor(tau, device=dev, dtype=dty)

        if z.dim() == 1:
            z = z.unsqueeze(0)  # (1,2)
        if z.dim() != 2 or z.shape[1] != 2:
            raise ValueError(f"z must be (B,2). Got {tuple(z.shape)}")

        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        if tau.dim() != 1:
            raise ValueError(f"tau must be 1D. Got {tuple(tau.shape)}")

        x = z[:, 0]  # (B,)
        y = z[:, 1]  # (B,)

        lnA1, B1 = self._lnA_and_B(self.kappa1, self.theta1, self.sigma1, tau)  # (T,)
        lnA2, B2 = self._lnA_and_B(self.kappa2, self.theta2, self.sigma2, tau)  # (T,)

        lnA = lnA1 + lnA2  # (T,)

        # broadcast to (B,T)
        return torch.exp(
            lnA.unsqueeze(0)
            - B1.unsqueeze(0) * x.unsqueeze(1)
            - B2.unsqueeze(0) * y.unsqueeze(1)
        )

    def discount_curve_annual(self, z: torch.Tensor, T_max: int) -> torch.Tensor:
        tau = torch.arange(1, T_max + 1, device=self.log_kappa1.device, dtype=self.log_kappa1.dtype)
        return self.zcb_price(z, tau)
