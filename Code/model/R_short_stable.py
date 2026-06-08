import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.utils.common import CenteredSoftStep


class RShortStable(nn.Module):
    """
    Stable short-rate network with bounded output.

    Implements: r(z) = r_center + r_scale * tanh(f(z))

    Where:
    - f(z) is a neural network (latent_dim -> hidden_dim -> 1)
    - r_center: learnable mean short rate
    - r_scale: learnable spread (enforced > 0 via softplus)
    - Output is bounded in approximately [r_center - r_scale, r_center + r_scale]

    """

    def __init__(
            self,
            latent_dim: int,
            hidden_dim: int,
            bias: bool = True,
            r_center_init: float = 0.01,
            r_scale_init: float = 0.02,
            min_scale: float = 1e-4,
    ):
        """
        Parameters
        ----------
        latent_dim : int
            Dimension of latent factors.
        hidden_dim : int
            Hidden layer dimension.
        bias : bool
            Whether to use bias in linear layers.
        r_center_init : float
            Initial center of rate distribution (default 1%).
        r_scale_init : float
            Initial half-width of range (default ±2%).
        min_scale : float
            Minimum value for r_scale to ensure positivity.
        """
        super().__init__()
        self.min_scale = min_scale

        # Neural network
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, 1, bias=bias),
        )

        # Learnable center
        self.r_center = nn.Parameter(torch.tensor(float(r_center_init)))

        # Initialize raw_r_scale via inverse softplus so that scale() starts near r_scale_init
        # We want: softplus(raw_r_scale) + min_scale ≈ r_scale_init
        # So: raw_r_scale ≈ inverse_softplus(r_scale_init - min_scale)
        #     inverse_softplus(y) = log(exp(y) - 1) = log(expm1(y))
        init_scale = max(r_scale_init - min_scale, 1e-8)
        raw_init = math.log(math.expm1(init_scale))
        self.raw_r_scale = nn.Parameter(torch.tensor(float(raw_init)))

    def scale(self):
        """
        Compute r_scale from unconstrained parameter.

        Returns
        -------
        r_scale : Always positive, equals softplus(raw_r_scale) + min_scale
        """
        return F.softplus(self.raw_r_scale) + self.min_scale

    def forward(self, z):
        """
        Compute r_tilde(z) with learnable bounded output.

        Parameters
        ----------
        z : (B, latent_dim) or (latent_dim,) latent factors

        Returns
        -------
        r_tilde : (B, 1) short rate, approximately in
                  [r_center - r_scale, r_center + r_scale]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Get unbounded network output
        x = self.net(z)  # (B, 1), unbounded (-∞, +∞)

        # Apply tanh to map to (-1, +1)
        r_normalized = torch.tanh(x)  # (B, 1) in (-1, +1)

        # Scale by learnable r_scale (guaranteed positive)
        r_scale = self.scale()  # scalar, always > 0

        # Final output: bounded in approximately [r_center - r_scale, r_center + r_scale]
        r = self.r_center + r_scale * r_normalized  # (B, 1)

        return r