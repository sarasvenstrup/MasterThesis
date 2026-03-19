# %pip install torch
# dbutils.library.restartPython()

# DO NOT RUN THIS FILE

import torch
import torch.nn as nn


class KMu(nn.Module):
    """
    Stable linear drift:
        mu(z) = M z + N

    with
        M = -(V^T V + eps I),

    so M is symmetric negative definite and the drift is
    numerically stable by construction.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent state z.
    eps : float
        Small positive constant ensuring strict negative definiteness.
    bias : bool
        Whether to include free intercept N.
    """

    def __init__(self, latent_dim: int, eps: float = 1e-3, bias: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps = eps

        # Free parameter used to build stable M
        self.V = nn.Parameter(torch.zeros(latent_dim, latent_dim))

        if bias:
            self.N = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.N = None

    def stable_matrix(self, device=None, dtype=None):
        if device is None:
            device = self.V.device
        if dtype is None:
            dtype = self.V.dtype

        I = torch.eye(self.latent_dim, device=device, dtype=dtype)
        M = -(self.V.T @ self.V + self.eps * I)
        return M

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        M = self.stable_matrix(device=z.device, dtype=z.dtype)
        mu = z @ M.T

        if self.N is not None:
            mu = mu + self.N

        return mu


class old_KMu(nn.Module):
    """
    Drift network K from the paper.
    Implements: mu(z) = M z + N
    (pure linear mapping, no activation)
    """

    def __init__(self, latent_dim: int, bias=True):
        super().__init__()
        self.lin = nn.Linear(latent_dim, latent_dim, bias=bias)

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.lin(z)  # (B, latent_dim)

