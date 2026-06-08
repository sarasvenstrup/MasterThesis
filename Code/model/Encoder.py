import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Linear encoder mapping input_dim → latent_dim with no bias."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, latent_dim, bias=False)

    def forward(self, x):
        return self.lin(x)
