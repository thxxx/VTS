from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int = 1, out_features: int = 256, std: float = 1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(out_features // 2, in_features) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = 2 * math.pi * x @ self.weight.T
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

