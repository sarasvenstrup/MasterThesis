import torch
import torch.nn as nn

from Code.utils.common import CenteredSoftStep


class DecoderG(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()

        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 1, bias=bias)
        )

    def forward(self, z: torch.Tensor,
                tau: torch.Tensor) -> torch.Tensor:  # Here it expects two inputs, the latent factors from the encoder output and the maturity list

        B, d = z.shape
        N = tau.numel()

        tau_in = tau.unsqueeze(0).expand(B, -1)

        z_exp = z.unsqueeze(1).expand(-1, N, -1)  # (batch,N,d)
        inp = torch.cat([z_exp, tau_in.unsqueeze(-1)], dim=-1)  # (batch,N,d+1)
        g = self.net(inp.reshape(-1, d + 1)).reshape(B, N)

        return g

