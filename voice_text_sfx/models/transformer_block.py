from __future__ import annotations

import torch
import torch.nn as nn

from .attention import Attention
from .feedforward import FeedForward
from .utils import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        cross_attend: bool = False,
        dim_context: int | None = None,
        zero_init_branch_outputs: bool = True,
        attn_kwargs: dict | None = None,
        ff_kwargs: dict | None = None,
        norm_kwargs: dict | None = None,
    ):
        super().__init__()
        attn_kwargs = attn_kwargs or {}
        ff_kwargs = ff_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.pre_norm = LayerNorm(dim, **norm_kwargs)
        self.self_attn = Attention(dim, dim_heads=dim_heads, zero_init_output=zero_init_branch_outputs, **attn_kwargs)

        self.cross_attn = None
        self.cross_attend_norm = None
        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, **norm_kwargs)
            self.cross_attn = Attention(
                dim,
                dim_heads=dim_heads,
                dim_context=dim_context,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )

        self.ff_norm = LayerNorm(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb=None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.pre_norm(x), mask=mask, rotary_pos_emb=rotary_pos_emb)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)
        x = x + self.ff(self.ff_norm(x))
        return x

