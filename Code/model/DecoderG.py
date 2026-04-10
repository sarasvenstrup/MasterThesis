import torch
import torch.nn as nn
from Code.utils.common import CenteredSoftStep

class DecoderG(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 1, bias=bias),
        )

    def forward(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        B, d = z.shape
        N = tau.numel()

        tau_in = tau.unsqueeze(0).expand(B, -1)                 # (B,N)
        z_exp  = z.unsqueeze(1).expand(-1, N, -1)               # (B,N,d)
        inp    = torch.cat([z_exp, tau_in.unsqueeze(-1)], dim=-1)  # (B,N,d+1)

        g = self.net(inp.reshape(-1, d + 1)).reshape(B, N)      # (B,N)
        return g

class DecoderGStable(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 1, bias=bias),
        )

        # Learnable per-nothing floor: G >= softplus(g_floor) + eps
        # Initialised so floor ≈ 0.05 at the start of retraining
        self.g_floor = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        B, d = z.shape
        N = tau.numel()

        tau_in = tau.unsqueeze(0).expand(B, -1)
        z_exp  = z.unsqueeze(1).expand(-1, N, -1)
        inp    = torch.cat([z_exp, tau_in.unsqueeze(-1)], dim=-1)

        g = self.net(inp.reshape(-1, d + 1)).reshape(B, N)

        # Shift up so G > softplus(g_floor) > 0 everywhere
        floor = torch.nn.functional.softplus(self.g_floor)
        return g + floor