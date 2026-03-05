#%pip install torch
#dbutils.library.restartPython()

# DO NOT RUN THIS FILE

import torch
import torch.nn as nn # import of PyTorch's neural-network layer

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, latent_dim, bias = False)
        # This line defines a linear transformation from R^8 to R^2, and creates the input vector, W and b

    def forward(self, x):
        return self.lin(x)

# This code defines a new neural network class called Encoder, just like in the article where there is no bias.
