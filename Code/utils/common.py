"""Common utilities shared across the thesis codebase: activation functions and plot theme."""

import seaborn as sns
import torch
import torch.nn as nn


# Activation Functions

class CenteredSoftStep(nn.Module):
    """
    ψ(x) = sigmoid(x) - 0.5
    """

    def forward(self, x):
        return torch.sigmoid(x) - 0.5


# Layout & Themes

def set_paper_theme():
    """Apply the paper plot theme and return the custom colour palette."""
    sns.set_theme(
        style="darkgrid",
        context="paper",
        rc={
            "axes.facecolor": "#EAEAF2",
            "figure.facecolor": "white",
            "grid.color": "white",
            "grid.linewidth": 1.0,
            "axes.edgecolor": "0.8",
            "axes.linewidth": 0.8,
        }
    )

    palette = sns.color_palette("tab20b")
    custom_palette = [
        palette[0], palette[1], palette[2], palette[3],
        palette[12], palette[13], palette[14], palette[15],
    ]
    sns.set_palette(custom_palette)
    return custom_palette
