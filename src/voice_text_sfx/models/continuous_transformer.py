from __future__ import annotations

import torch
from torch import nn

from .rotary import RotaryEmbedding
from .transformer_block import TransformerBlock
from .utils import checkpoint


class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        dim_in: int | None = None,
        dim_out: int | None = None,
        dim_heads: int = 64,
        cross_attend: bool = True,
        cond_embed_dim: int | None = None,
        rotary_pos_emb: bool = True,
        zero_init_branch_outputs: bool = True,
        attn_kwargs: dict | None = None,
    ):
        super().__init__()
        self.d_model = dim
        self.depth = depth
        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32)) if rotary_pos_emb else None

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    dim_heads=dim_heads,
                    cross_attend=cross_attend,
                    dim_context=cond_embed_dim,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    attn_kwargs=attn_kwargs or {},
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        prepend_embeds: torch.Tensor | None = None,
        prepend_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq, device = x.shape[0], x.shape[1], x.device
        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length = prepend_embeds.shape[1]
            x = torch.cat((prepend_embeds, x), dim=-2)
            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device=device, dtype=torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device=device, dtype=torch.bool)
                mask = torch.cat((prepend_mask, mask), dim=-1)

        rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1]) if self.rotary_pos_emb is not None else None

        for layer in self.layers:
            x = checkpoint(layer, x, context=context, context_mask=context_mask, rotary_pos_emb=rotary_pos_emb, mask=mask)

        return self.project_out(x)
