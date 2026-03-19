# %pip install torch
# dbutils.library.restartPython()

# DO NOT RUN THIS FILE

import torch
import torch.nn as nn


class KMu(nn.Module):
    """
    Stable affine linear drift:
        mu(z) = -A z + N

    where
        A = B B^T + eps I

    so the matrix multiplying z is stable by construction,
    while N is left completely free.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent state z.
    eps : float
        Small positive constant to ensure strict stability.
    bias : bool
        If True, include a free affine term N.
    """

    def __init__(
        self,
        latent_dim: int,
        eps: float = 1e-3,
        bias: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps = eps

        # Free parameter used to build stable matrix A = B B^T + eps I
        self.B = nn.Parameter(torch.zeros(latent_dim, latent_dim))

        # Free affine term N
        if bias:
            self.N = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_parameter("N", None)

    def stable_matrix(self, device=None, dtype=None):
        if device is None:
            device = self.B.device
        if dtype is None:
            dtype = self.B.dtype

        I = torch.eye(self.latent_dim, device=device, dtype=dtype)
        A = self.B @ self.B.T + self.eps * I
        return A

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        A = self.stable_matrix(device=z.device, dtype=z.dtype)

        # mu(z) = -A z + N
        mu = -z @ A.T

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

