from __future__ import annotations

import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: nn.Module):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, no_bias: bool = False, zero_init_output: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        self.linear_in = GLU(dim, inner_dim, nn.SiLU())
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=not no_bias)
        if zero_init_output:
            nn.init.zeros_(self.linear_out.weight)
            if self.linear_out.bias is not None:
                nn.init.zeros_(self.linear_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.linear_in(x))

