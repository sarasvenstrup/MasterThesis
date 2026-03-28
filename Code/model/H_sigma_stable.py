import math
import torch
import torch.nn as nn

from Code.utils.common import CenteredSoftStep


class HSigmaStable(nn.Module):
    """
    Structurally stable 2D H-network.

    Guarantees:
      sigma_i(z) in (sigma_min, sigma_max)
      rho(z)     in (-rho_max, rho_max), with rho_max < 1

    Smooth bounded parameterization:
      log_sigma_i(z) = log_sigma_min
                       + (log_sigma_max - log_sigma_min) * sigmoid(h_i(z) + offset_i)
      rho(z)         = rho_max * tanh(h_rho(z))

    This is not clamping. It is a smooth reparameterization.

    Why this version:
      - closer to the paper, which models log(sigma_i) and atanh(rho)
      - guarantees positive and bounded volatilities
      - guarantees correlation stays away from ±1
      - guarantees the 2D Cholesky factor is always real
    """

    def __init__(
        self,
        hidden_dim: int,
        bias: bool = False,
        sigma_init: float = 0.015,
        sigma_min: float = 1e-4,
        sigma_max: float = 0.20,
        rho_max: float = 0.999,
    ):
        super().__init__()

        if not (0.0 < sigma_min < sigma_max):
            raise ValueError("Require 0 < sigma_min < sigma_max.")
        if not (sigma_min < sigma_init < sigma_max):
            raise ValueError("Require sigma_min < sigma_init < sigma_max.")
        if not (0.0 < rho_max < 1.0):
            raise ValueError("Require 0 < rho_max < 1.")

        self.d = 2
        self.n_corr = 1

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho_max = float(rho_max)

        self.log_sigma_min = math.log(self.sigma_min)
        self.log_sigma_max = math.log(self.sigma_max)
        self.log_sigma_range = self.log_sigma_max - self.log_sigma_min

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 3, bias=bias),  # [raw_logsigma1, raw_logsigma2, raw_rho]
        )

        # Flat initial surface
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

        # Choose offset so that sigma = sigma_init when raw output is zero
        target = (math.log(sigma_init) - self.log_sigma_min) / self.log_sigma_range
        target = min(max(target, 1e-8), 1.0 - 1e-8)
        raw_init = math.log(target / (1.0 - target))  # logit(target)

        self.raw_logsigma_offset = nn.Parameter(torch.full((2,), raw_init))

    def forward(self, z: torch.Tensor, return_raw: bool = False):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        raw = self.net(z)  # (B, 3)

        # Smoothly bounded log-sigmas
        raw_logsigmas = raw[:, :2] + self.raw_logsigma_offset
        log_sigmas = self.log_sigma_min + self.log_sigma_range * torch.sigmoid(raw_logsigmas)
        sigmas = torch.exp(log_sigmas)

        # Smoothly bounded correlation away from ±1
        raw_rho = raw[:, 2:3]
        rhos = self.rho_max * torch.tanh(raw_rho)

        if return_raw:
            return {
                "raw": raw,
                "raw_logsigmas": raw_logsigmas,
                "log_sigmas": log_sigmas,
                "raw_rho": raw_rho,
            }

        return sigmas, rhos