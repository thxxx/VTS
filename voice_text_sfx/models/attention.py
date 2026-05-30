from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import einsum, nn

from .utils import apply_rotary_pos_emb, or_reduce

flash_attn_func = None


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        dim_context: int | None = None,
        zero_init_output: bool = True,
        qk_norm: Literal["l2", "ln", "none"] = "l2",
    ):
        super().__init__()
        self.d_model = dim
        self.dim_heads = dim_heads
        self.dim_context = dim_context

        dim_kv = dim_context if dim_context is not None else self.d_model
        self.num_heads = self.d_model // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(self.d_model, self.d_model, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim_kv, dim_kv * 3, bias=False)

        self.to_out = nn.Linear(self.d_model, self.d_model, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm
        if qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1e-6)

        self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse("2.0.0")
        self.sdp_kwargs = {
            "enable_flash": True,
            "enable_math": True,
            "enable_mem_efficient": True,
        }

    def flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        batch, heads, q_len, _, k_len = *q.shape, k.shape[-2]
        kv_heads = k.shape[1]

        if heads != kv_heads:
            heads_per_kv_head = heads // kv_heads
            k = k.repeat_interleave(heads_per_kv_head, dim=1)
            v = v.repeat_interleave(heads_per_kv_head, dim=1)

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)
        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)
        if mask is not None:
            mask = mask.expand(batch, heads, q_len, k_len)

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb=None,
    ) -> torch.Tensor:
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None
        kv_input = context if has_context else x

        if self.dim_context is not None:
            q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            k = rearrange(k, "b n (h d) -> b h n d", h=kv_h)
            v = rearrange(v, "b n (h d) -> b h n d", h=kv_h)
        else:
            q, k, v = self.to_qkv(kv_input).chunk(3, dim=-1)
            q = rearrange(q, "b n (h d) -> b h n d", h=h)
            k = rearrange(k, "b n (h d) -> b h n d", h=kv_h)
            v = rearrange(v, "b n (h d) -> b h n d", h=kv_h)

        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm == "ln":
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb
            q = apply_rotary_pos_emb(q.to(torch.float32), freqs.to(torch.float32)).to(q.dtype)
            k = apply_rotary_pos_emb(k.to(torch.float32), freqs.to(torch.float32)).to(k.dtype)

        input_mask = context_mask if has_context else mask
        masks = []
        if input_mask is not None:
            masks.append(~rearrange(input_mask, "b j -> b 1 1 j"))
        final_attn_mask = ~or_reduce(masks) if masks else None

        if self.use_pt_flash and torch.cuda.is_available():
            out = self.flash_attn(q, k, v, mask=final_attn_mask)
        else:
            if h != kv_h:
                heads_per_kv_head = h // kv_h
                k = k.repeat_interleave(heads_per_kv_head, dim=1)
                v = v.repeat_interleave(heads_per_kv_head, dim=1)

            scale = 1.0 / (q.shape[-1] ** 0.5)
            dots = einsum("b h i d, b h j d -> b h i j", q, k) * scale
            if final_attn_mask is not None:
                dots = dots.masked_fill(~final_attn_mask, -torch.finfo(dots.dtype).max)
            attn = F.softmax(dots, dim=-1, dtype=torch.float32).to(dots.dtype)
            out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if mask is not None:
            out = out.masked_fill(~rearrange(mask, "b n -> b n 1"), 0.0)
        return out

