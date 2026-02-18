# %pip install torch
# dbutils.library.restartPython()

# DO NOT RUN THIS FILE

import torch
import torch.nn as nn

from utils.common import CenteredSoftStep


class DecoderG(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, bias: bool = True):
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

        # if z.dim() == 1:
        #    z = z.unsqueeze(0)

        # if not torch.is_tensor(tau):
        #    tau = torch.tensor(tau, dtype=z.dtype, device=z.device)

        # if tau.dim() == 0:
        #    tau = tau.expand(z.shape[0], 1)
        # elif tau.dim() == 1:
        #    tau = tau.unsqueeze(1)

        # Expand tau across batch
        # tau_scaled = tau / self.tau_max   # normalize to [0,1]
        tau_in = tau.unsqueeze(0).expand(B, -1)

        z_exp = z.unsqueeze(1).expand(-1, N, -1)  # (batch,N,d)
        inp = torch.cat([z_exp, tau_in.unsqueeze(-1)], dim=-1)  # (batch,N,d+1)
        g = self.net(inp.reshape(-1, d + 1)).reshape(B, N)

        return g

