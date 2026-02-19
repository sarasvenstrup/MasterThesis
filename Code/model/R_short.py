# %pip install torch
# dbutils.library.restartPython()

# DO NOT RUN THIS FILE

import torch
import torch.nn as nn

from Code.utils.common import CenteredSoftStep  # Load the activation function


class RShort(nn.Module):
    """
    Risk-free rate network R from the paper: (2,4,1),
    no bias, centered soft step activation.
    Maps z -> r_tilde(z).
    """

    def __init__(self, latent_dim: int, hidden_dim: int, bias=False):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 1, bias=bias),
        )

    def forward(self, z):
        """
        z: (B,2) or (2,)
        Returns: r_tilde (B,1) or (1,)
        """

        if z.dim() == 1:
            z = z.unsqueeze(0)

        r = self.net(z)

        return r





