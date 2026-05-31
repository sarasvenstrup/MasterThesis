import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.utils.common import CenteredSoftStep


class DecoderGStable(nn.Module):
    """
    Stable decoder G with two structural safeguards:

      1. Softplus output shift — guarantees G > g_min > 0 for all (z, τ).
         Implements: G = g_min + softplus(MLP(z, τ))
         This directly prevents the G ≈ 0 singularity that causes the ODE
         coefficients α = .../G and β = r/G to blow up, producing P = inf.
         The MLP is otherwise unconstrained: it learns freely, only the
         lower bound is enforced.

      2. Tanh input normalisation on z — bounds the MLP's effective input
         space to (-1, +1)^ell regardless of how far z drifts during
         simulation.  In-distribution z values pass through nearly linearly;
         OOD z values saturate smoothly rather than extrapolating freely.
         Implements: z_in = tanh(z / z_scale)

    Together these target the two failure modes observed in pricing:
      • G → 0 (softplus floor)
      • G unbounded for OOD z (tanh input compression)

    Parameters
    ----------
    latent_dim : int
    hidden_dim : int
    bias       : bool
    g_min      : float
        Positive lower bound on G output.  Default 0.05.  Must be > 0.
        Larger values give stronger ODE stability but reduce G's range.

    Buffers
    -------
    z_scale : (latent_dim,)
        Non-learnable tanh scale.  Initialised to ones; call
        set_z_scale(z_train) after training to calibrate from data.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        bias: bool = True,
        g_min: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.g_min = g_min

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 1, bias=bias),
        )

        self.register_buffer(
            "z_scale",
            torch.ones(latent_dim, dtype=torch.float32),
        )

    def set_z_scale(self, z_train: torch.Tensor) -> None:
        """Set z_scale = std(z_train) per dimension."""
        with torch.no_grad():
            std = z_train.std(dim=0).clamp(min=1e-6)
            self.z_scale.copy_(std.to(self.z_scale.device))

    def forward(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:   (B, latent_dim)
            tau: (N,)
        Returns:
            G:   (B, N)  with G > g_min everywhere.
        """
        B, d = z.shape
        N = tau.numel()

        # 1) Tanh-normalise z input
        z_norm = torch.tanh(z / self.z_scale.clamp(min=1e-6))      # (B, d)

        # 2) Build (B*N, d+1) input tensor
        tau_in = tau.unsqueeze(0).expand(B, -1)                     # (B, N)
        z_exp  = z_norm.unsqueeze(1).expand(-1, N, -1)              # (B, N, d)
        inp    = torch.cat([z_exp, tau_in.unsqueeze(-1)], dim=-1)    # (B, N, d+1)

        # 3) MLP raw output
        raw = self.net(inp.reshape(-1, d + 1)).reshape(B, N)        # (B, N)

        # 4) Softplus shift: G > g_min guaranteed for all inputs
        g = self.g_min + F.softplus(raw)                            # (B, N)
        return g
