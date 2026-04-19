from __future__ import annotations

from math import pi
from typing import Sequence, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .base import Conditioner


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


def time_positional_embedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(dim + 1, out_features),
    )


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        self.features = features
        self.embedding = time_positional_embedding(dim=dim, out_features=features)

    def forward(self, x: Union[Sequence[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        return embedding.view(*shape, self.features)


class NumberConditioner(Conditioner):
    def __init__(self, output_dim: int, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: Union[Sequence[float], Tensor], device: str | torch.device | None = None):
        floats = torch.as_tensor(floats, device=device, dtype=torch.float32).view(-1)
        floats = floats.clamp(self.min_val, self.max_val)
        normalized = (floats - self.min_val) / (self.max_val - self.min_val)
        normalized = normalized.to(next(self.embedder.parameters()).dtype)
        embeds = self.embedder(normalized).unsqueeze(1)
        mask = torch.ones(embeds.shape[0], 1, device=embeds.device, dtype=torch.bool)
        return embeds, mask

