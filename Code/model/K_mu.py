import torch.nn as nn

class KMu(nn.Module):
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