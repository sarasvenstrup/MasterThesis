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
    def __init__(self, latent_dim: int, hidden_dim: int, bias: bool = True, g_floor_init: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 1, bias=bias),
        )

        self.g_floor = nn.Parameter(torch.tensor(float(g_floor_init)))

    def forward(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        B, d = z.shape
        N = tau.numel()

        tau_in = tau.unsqueeze(0).expand(B, -1)
        z_exp  = z.unsqueeze(1).expand(-1, N, -1)
        inp    = torch.cat([z_exp, tau_in.unsqueeze(-1)], dim=-1)

        g = self.net(inp.reshape(-1, d + 1)).reshape(B, N)

        floor = torch.nn.functional.softplus(self.g_floor)
        return g + floor