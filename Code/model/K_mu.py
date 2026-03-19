# %pip install torch
# dbutils.library.restartPython()

# DO NOT RUN THIS FILE

import torch
import torch.nn as nn


class KMu(nn.Module):
    """
    Stable linear drift:
        mu(z) = -A (z - z_star)

    where
        A = B B^T + eps I

    so A is positive definite and the drift is mean-reverting by construction.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent state z.
    eps : float
        Small positive constant to ensure strict stability.
    learn_z_star : bool
        If True, learn the long-run mean z_star.
        If False, z_star is fixed at zero unless set externally.
    init_z_star : torch.Tensor or None
        Optional initial value for z_star, shape (latent_dim,).
    """

    def __init__(
        self,
        latent_dim: int,
        eps: float = 1e-3,
        learn_z_star: bool = True,
        init_z_star: torch.Tensor | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps = eps

        # Free parameter used to build stable matrix A = B B^T + eps I
        self.B = nn.Parameter(torch.zeros(latent_dim, latent_dim))

        if init_z_star is None:
            init_z_star = torch.zeros(latent_dim, dtype=torch.float32)
        else:
            init_z_star = init_z_star.detach().clone().float()
            if init_z_star.numel() != latent_dim:
                raise ValueError(
                    f"init_z_star must have {latent_dim} entries, got {init_z_star.numel()}"
                )

        if learn_z_star:
            self.z_star = nn.Parameter(init_z_star)
        else:
            self.register_buffer("z_star", init_z_star)

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

        # mu(z) = -A (z - z_star)
        mu = -(z - self.z_star) @ A.T
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

