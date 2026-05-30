from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: int = 512,
        interpolation_factor: float = 1.0,
        base: int = 10000,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward_from_seq_len(self, seq_len: int):
        return self.forward(torch.arange(seq_len, device=self.inv_freq.device))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, t: torch.Tensor):
        t = t.to(torch.float32) / self.interpolation_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if self.scale is None:
            return freqs, 1.0

        seq_len = t.shape[0]
        power = (torch.arange(seq_len, device=t.device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale

