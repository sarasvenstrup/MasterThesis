import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Code.utils.common import CenteredSoftStep


class HSigmaStable(nn.Module):
    """
    Stable volatility/correlation network with mathematically guaranteed positive sigmas.

    Implements:
        σ_i(z) = σ_min + σ_amp_i * sigmoid(g_i(z))
        ρ_ij(z) = tanh(h_ij(z))

    This design guarantees:
        - σ_i(z) > σ_min > 0 always (positive volatility by construction)
        - σ_i(z) ≤ σ_min + σ_amp_i (bounded above)
        - Smooth gradients via sigmoid/tanh
        - ρ_ij(z) ∈ (-1, 1) (bounded correlation)

    Motivation:
        The original H network uses σ = exp(f(z)), which has no upper bound.
        This causes:
        1. Unbounded shocks as z varies
        2. Decoder failures at large |z|
        3. Numerical instabilities in long-horizon MC

        With sigmoid parameterization, volatility is controlled and matches
        the decoder's safe latent region.

    Paper/Reference:
        Same architecture as original H (2,4,3) for latent_dim=2,
        but with safe output transformation inspired by R_short_stable.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        bias: bool = False,
        sigma_min: float = 0.5,
        sigma_amp_init: float = 0.5,
    ):
        """
        Args:
            latent_dim: Dimension of latent factors (typically 2)
            hidden_dim: Hidden layer dimension (typically 4)
            bias: Whether to use bias in linear layers (default False for stability)
            sigma_min: Floor for all volatilities, must be > 0 (default 0.5)
            sigma_amp_init: Initial amplitude per dimension (default 0.5)

        Guarantees:
            σ_i(z) ∈ (sigma_min, sigma_min + sigma_amp_i] for all z and i
        """
        super().__init__()
        self.d = int(latent_dim)
        self.n_corr = self.d * (self.d - 1) // 2
        self.sigma_min = float(sigma_min)

        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be > 0, got {sigma_min}")

        out_dim = self.d + self.n_corr

        self.net = nn.Sequential(
            nn.Linear(self.d, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )

        # Learnable positive amplitudes per dimension
        # Initialize via inverse softplus to start near sigma_amp_init
        # softplus(x) ≈ log(1 + exp(x)), so inverse is roughly log(exp(y) - 1) = log(expm1(y))
        init_amp = max(float(sigma_amp_init), 1e-8)
        raw_init = math.log(math.expm1(init_amp))
        self.raw_sigma_amps = nn.Parameter(
            torch.full((self.d,), float(raw_init))
        )

    def sigma_amps(self):
        """
        Compute sigma amplitudes from unconstrained parameters.

        Returns:
            sigma_amps: (d,) Always positive
        """
        return F.softplus(self.raw_sigma_amps)

    def forward(self, z: torch.Tensor, return_raw: bool = False):
        """
        Compute σ(z) and ρ(z) with mathematically guaranteed positive volatilities.

        Uses sigmoid for bounded positive sigmas and tanh for bounded correlations.

        Args:
            z: (B,d) or (d,) latent factors
            return_raw: If True, return raw network outputs instead of processed ones

        Returns (default):
            sigmas: (B,d) volatilities, all strictly positive and bounded
            rhos: (B,n_corr) correlations, bounded in (-1, 1)

        If return_raw=True:
            raw: (B, d + n_corr) unbounded network outputs
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Get unbounded network output
        raw = self.net(z)  # (B, d + n_corr)

        if return_raw:
            return raw

        # === 1) Volatilities (guaranteed positive and bounded) ===
        raw_sigmas = raw[:, : self.d]  # (B, d), unbounded from network
        sigma_amps = self.sigma_amps()  # (d,), always positive

        # σ_i(z) = σ_min + σ_amp_i * sigmoid(raw_sigma_i)
        # sigmoid maps (-∞, +∞) → (0, 1)
        # So σ_i ∈ (σ_min, σ_min + σ_amp_i)
        # This is GUARANTEED positive for all z!
        sigmas = (
            self.sigma_min
            + sigma_amps.unsqueeze(0) * torch.sigmoid(raw_sigmas)
        )  # (B, d)

        # === 2) Correlations (in (-1, 1)) ===
        if self.n_corr > 0:
            raw_rhos = raw[:, self.d :]  # (B, n_corr)
            # tanh maps (-∞, +∞) → (-1, 1)
            rhos = torch.tanh(raw_rhos)
        else:
            rhos = raw[:, :0]  # (B, 0) empty

        return sigmas, rhos



