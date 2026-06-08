import torch
import torch.nn as nn
from Code.utils.common import CenteredSoftStep

class DecoderG(nn.Module):
    """Two-layer MLP decoder: maps (z, τ) → g(z, τ) for a batch of curves and tenors."""

    def __init__(self, latent_dim: int, hidden_dim: int, bias: bool = True):
        """
        Parameters
        ----------
        latent_dim : dimension of the latent vector z.
        hidden_dim : number of hidden units.
        bias       : include bias terms in the linear layers.
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 1, bias=bias),
        )

    def forward(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z   : (B, d) latent vectors.
        tau : (N,)   tenor grid.

        Returns (B, N) decoded values.
        """
        B, d = z.shape
        N = tau.numel()

        tau_in = tau.unsqueeze(0).expand(B, -1)                 # (B,N)
        z_exp  = z.unsqueeze(1).expand(-1, N, -1)               # (B,N,d)
        inp    = torch.cat([z_exp, tau_in.unsqueeze(-1)], dim=-1)  # (B,N,d+1)

        g = self.net(inp.reshape(-1, d + 1)).reshape(B, N)      # (B,N)
        return g
