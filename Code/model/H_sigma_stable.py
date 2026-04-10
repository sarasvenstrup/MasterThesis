import math
import torch
import torch.nn as nn

from Code.utils.common import CenteredSoftStep


class HSigmaStable(nn.Module):
    """
    Structurally stable H-network for any latent dimension d.

    Guarantees:
      sigma_i(z) in (sigma_min, sigma_max)  for all i = 1, ..., d
      rho_ij(z)  in (-rho_max, rho_max)     for all pairs i < j, with rho_max < 1

    Smooth bounded parameterization:
      log_sigma_i(z) = log_sigma_min
                       + (log_sigma_max - log_sigma_min) * sigmoid(net_i(z) + offset_i)
      rho_ij(z)      = rho_max * tanh(net_ij(z))

    This is not clamping — it is a smooth reparameterization.

    Properties:
      - sigma_i > 0 and bounded above for all z  ->  diffusion cannot grow arbitrarily
      - |rho_ij| < rho_max < 1                   ->  Cholesky factor is always real
      - Generalises to any latent_dim d
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        bias: bool = False,
        sigma_init: float = 0.3,
        sigma_min: float = 1e-4,
        sigma_max: float = 2,
        rho_max: float = 0.999,
    ):
        super().__init__()

        if not (0.0 < sigma_min < sigma_max):
            raise ValueError("Require 0 < sigma_min < sigma_max.")
        if not (sigma_min < sigma_init < sigma_max):
            raise ValueError("Require sigma_min < sigma_init < sigma_max.")
        if not (0.0 < rho_max < 1.0):
            raise ValueError("Require 0 < rho_max < 1.")

        self.d = int(latent_dim)
        self.n_corr = self.d * (self.d - 1) // 2
        out_dim = self.d + self.n_corr  # d sigmas + d(d-1)/2 correlations

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho_max = float(rho_max)

        self.log_sigma_min = math.log(self.sigma_min)
        self.log_sigma_max = math.log(self.sigma_max)
        self.log_sigma_range = self.log_sigma_max - self.log_sigma_min

        self.net = nn.Sequential(
            nn.Linear(self.d, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )

        # Flat initial surface: zero-initialise final layer weights
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

        # Choose offset so that sigma_i = sigma_init when raw output is zero
        target = (math.log(sigma_init) - self.log_sigma_min) / self.log_sigma_range
        target = min(max(target, 1e-8), 1.0 - 1e-8)
        raw_init = math.log(target / (1.0 - target))  # logit(target)

        self.raw_logsigma_offset = nn.Parameter(torch.full((self.d,), raw_init))

    def forward(self, z: torch.Tensor, return_raw: bool = False):
        """
        z: (B, d) or (d,)

        Returns (default):
          sigmas: (B, d)    — volatilities in (sigma_min, sigma_max)
          rhos:   (B, n_corr) — correlations in (-rho_max, rho_max)

        If return_raw=True:
          dict with intermediate tensors for diagnostics
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        raw = self.net(z)  # (B, d + n_corr)

        # Smoothly bounded log-sigmas
        raw_logsigmas = raw[:, :self.d] + self.raw_logsigma_offset   # (B, d)
        log_sigmas = self.log_sigma_min + self.log_sigma_range * torch.sigmoid(raw_logsigmas)
        sigmas = torch.exp(log_sigmas)                                 # (B, d)

        # Smoothly bounded correlations away from ±1
        if self.n_corr > 0:
            raw_rhos = raw[:, self.d:]                                 # (B, n_corr)
            rhos = self.rho_max * torch.tanh(raw_rhos)                 # (B, n_corr)
        else:
            rhos = raw[:, :0]                                          # (B, 0) empty

        if return_raw:
            return {
                "raw": raw,
                "raw_logsigmas": raw_logsigmas,
                "log_sigmas": log_sigmas,
                "raw_rhos": raw[:, self.d:] if self.n_corr > 0 else None,
            }

        return sigmas, rhos


