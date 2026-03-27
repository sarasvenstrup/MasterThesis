import torch
import torch.nn as nn

from Code.utils.common import CenteredSoftStep


class HSigma(nn.Module):
    """
    H network from the paper (for d=2): (2,4,3), no bias, soft step activation.
    Generalized to any latent_dim d:
      outputs raw = (log σ_1,...,log σ_d, atanh ρ_12, ρ_13, ..., ρ_(d-1,d))
      length = d + d(d-1)/2
    """

    def __init__(self, latent_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.d = int(latent_dim)                       # <-- FIX: store dimension
        self.n_corr = self.d * (self.d - 1) // 2
        out_dim = self.d + self.n_corr                 # <-- FIX: correct output size

        self.net = nn.Sequential(
            nn.Linear(self.d, hidden_dim, bias=bias),
            CenteredSoftStep(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )

    def forward(self, z: torch.Tensor, return_raw: bool = False):
        """
        z: (B,d) or (d,)

        Returns (default):
          sigmas: (B,d)
          rhos:   (B,n_corr)   (empty tensor if d=1)

        If return_raw=True:
          raw: (B, d + n_corr) = [log_sigmas, atanh_rhos]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        raw = self.net(z)  # (B, d + n_corr)

        if return_raw:
            return raw

        # 1) Volatilities (positive)
        log_sigmas = raw[:, :self.d]          # (B,d)
        sigmas = torch.exp(log_sigmas)

        # 2) Correlations (in (-1,1))
        if self.n_corr > 0:
            atanh_rhos = raw[:, self.d:]  # (B,n_corr)
            rhos = torch.tanh(atanh_rhos)  # tanh naturally bounds to (-1, 1), no need to clamp
        else:
            rhos = raw[:, :0]  # (B,0) empty

        return sigmas, rhos
