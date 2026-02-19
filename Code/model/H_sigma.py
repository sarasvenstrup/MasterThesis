#%pip install torch
#dbutils.library.restartPython()

# DO NOT RUN THIS FILE

import torch
import torch.nn as nn

from Code.utils.common import CenteredSoftStep # Load the activation function


class HSigma(nn.Module):
    """
    H network from the paper: (2,4,3), no bias, soft step activation.
    Maps z -> (logσ1, logσ2, atanhρ), then transforms.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, bias = False):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 3, bias=bias),
        )

    def forward(self, z, return_raw=False):
        """
        z: (B,2) or (2,)

        Returns (default):
          sigma1, sigma2, rho

        If return_raw=True:
          log_sigma1, log_sigma2, atanh_rho
        """

        if z.dim() == 1:
            z = z.unsqueeze(0)

        raw = self.net(z)   # (B,3)

        if return_raw:
            return raw

        # ----------------------------
        # 1) Volatilities
        # ----------------------------
        log_sigmas = raw[:, :self.d]  # (B,d)
        sigmas = torch.exp(log_sigmas)  # ensure positive

        # ----------------------------
        # 2) Correlations
        # ----------------------------
        atanh_rhos = raw[:, self.d:]  # (B, n_corr)
        rhos = torch.tanh(atanh_rhos)  # ensure in (-1,1)

        return sigmas, rhos  # sigma1, sigma2, rho

