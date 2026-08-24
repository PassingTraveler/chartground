from __future__ import annotations

import torch
from torch import nn


class VisionProjector(nn.Module):
    """SigLIP hidden states -> MiniMind hidden states."""

    def __init__(self, in_dim: int = 768, out_dim: int = 1024, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

