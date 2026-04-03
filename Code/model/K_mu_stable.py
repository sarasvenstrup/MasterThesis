import torch
import torch.nn as nn


class KMuStable_old(nn.Module):
    """
    Stable drift network K from the paper with guaranteed mean-reversion.
    
    Implements: mu(z) = M z + N
    where M = -(V^T V + eps*I) to ensure all eigenvalues are strictly negative.
    
    This parameterization guarantees mean-reversion by construction:
    - V is a learned (latent_dim x latent_dim) matrix
    - M = -(V^T V + eps*I) has all strictly negative eigenvalues
    - N is a learned bias vector
    """

    def __init__(self, latent_dim: int, bias: bool = True, epsilon: float = 1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        
        # Learnable matrix V - initialized orthogonal for stability
        self.V = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.orthogonal_(self.V)
        
        # Learnable bias N - initialized to zero
        if bias:
            self.N = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_parameter('N', None)

    def stable_matrix(self) -> torch.Tensor:
        """
        Compute the stable drift matrix M = -(V^T V + eps*I).
        
        Returns:
            M: (latent_dim, latent_dim) tensor with strictly negative eigenvalues
        """
        VtV = torch.matmul(self.V.t(), self.V)  # (d, d)
        M = -(VtV + self.epsilon * torch.eye(self.latent_dim, device=self.V.device, dtype=self.V.dtype))
        return M

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute mu(z) = M z + N using the stable parameterization of M.
        
        Args:
            z: (B, latent_dim) or (latent_dim,) latent factors
            
        Returns:
            mu: (B, latent_dim) drift
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        M = self.stable_matrix()  # (d, d)
        mu = torch.matmul(z, M.t())  # (B, d) @ (d, d)^T = (B, d)
        
        if self.N is not None:
            mu = mu + self.N.unsqueeze(0)  # (B, d)
        
        return mu




class KMuStable(nn.Module):
    """
    Mean-reverting linear drift with less restrictive stability constraint.

    mu(z) = M z + N
    with M = S - A

    S is skew-symmetric  -> allows rotational / cross-factor effects
    A is positive definite -> ensures symmetric part of M is negative definite

    Therefore M is Hurwitz, so the drift is stable / mean-reverting.
    """

    def __init__(self, latent_dim: int, bias: bool = True, epsilon: float = 1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.epsilon = epsilon

        # unconstrained matrix used to build skew-symmetric part
        self.B = nn.Parameter(torch.zeros(latent_dim, latent_dim))

        # unconstrained matrix used to build positive definite part
        self.L = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.orthogonal_(self.L)
        with torch.no_grad():
            self.L.mul_(0.20)

        if bias:
            self.N = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_parameter("N", None)

    def drift_matrix(self) -> torch.Tensor:
        I = torch.eye(self.latent_dim, device=self.L.device, dtype=self.L.dtype)

        # skew-symmetric part
        S = self.B - self.B.t()

        # positive definite part
        A = self.L @ self.L.t() + self.epsilon * I

        # Hurwitz matrix
        M = S - A
        return M

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)

        M = self.drift_matrix()
        mu = z @ M.t()

        if self.N is not None:
            mu = mu + self.N.unsqueeze(0)

        return mu