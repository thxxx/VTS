from __future__ import annotations

import copy

import torch
import torch.nn as nn
from einops import rearrange

from ..conditioning.time import NumberConditioner
from ..conditioning.timestep import FourierFeatures
from .continuous_transformer import ContinuousTransformer


class VoiceConditionedDiffusionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        num_heads: int,
        input_concat_dim: int,
        global_cond_type: str,
        latent_channels: int,
        config: dict,
        device: str | torch.device,
        voice_dim: int | None = None,
        voice_cond_type: str = "prepend",
    ):
        super().__init__()
        self.global_cond_type = global_cond_type
        self.voice_cond_dim = voice_dim
        self.voice_cond_type = voice_cond_type

        dim_in = latent_channels + input_concat_dim
        dim_per_head = d_model // num_heads

        default_config = copy.deepcopy(config["model"])
        default_config["dim"] = d_model
        default_config["depth"] = depth
        default_config["dim_heads"] = dim_per_head
        default_config["dim_in"] = dim_in
        default_config["dim_out"] = latent_channels
        default_config["attn_kwargs"] = {"qk_norm": "l2"}
        self.model = ContinuousTransformer(**default_config)

        self.timing_start_conditioner = NumberConditioner(**config["timing_config"])
        self.timing_total_conditioner = NumberConditioner(**config["timing_config"])

        self.cross_attention_inputs_keys = config["cross_attn_cond_keys"]
        self.global_cond_keys = config["global_cond_keys"]

        self.cond_token_dim = config["cond_token_dim"]
        self.global_cond_dim = config["global_cond_dim"]
        self.to_cond_embed = None
        self.to_global_embed = None
        self.to_voice_embed = None

        if self.cond_token_dim > 0:
            cond_embed_dim = self.cond_token_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(self.cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            ).to(device)

        if self.global_cond_dim > 0:
            self.to_global_embed = nn.Sequential(
                nn.Linear(self.global_cond_dim, d_model, bias=False),
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=False),
            ).to(device)

        if self.voice_cond_dim and self.voice_cond_dim > 0:
            voice_embed_dim = d_model if self.voice_cond_type == "prepend" else self.cond_token_dim
            self.to_voice_embed = nn.Sequential(
                nn.Linear(self.voice_cond_dim, voice_embed_dim * 4, bias=False),
                nn.SiLU(),
                nn.Linear(voice_embed_dim * 4, voice_embed_dim, bias=False),
            ).to(device)

        self.fourier = FourierFeatures(1, 256)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(256, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(latent_channels, latent_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def get_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, seconds_start, seconds_total):
        start_emb, start_mask = self.timing_start_conditioner(seconds_start, device=input_ids.device)
        total_emb, total_mask = self.timing_total_conditioner(seconds_total, device=input_ids.device)

        conditioning_tensors = {
            "prompt": (input_ids, attention_mask),
            "seconds_start": (start_emb, start_mask),
            "seconds_total": (total_emb, total_mask),
        }

        cross_attention_inputs = []
        cross_attention_masks = []
        for key in self.cross_attention_inputs_keys:
            cross_attn_in, cross_attn_mask = conditioning_tensors[key]
            cross_attention_inputs.append(cross_attn_in)
            cross_attention_masks.append(cross_attn_mask)

        cross_attention_inputs = torch.cat(cross_attention_inputs, dim=1)
        cross_attention_masks = torch.cat(cross_attention_masks, dim=1).to(torch.bool)

        global_cond = None
        if self.global_cond_keys:
            global_conds = []
            for key in self.global_cond_keys:
                global_cond_input = conditioning_tensors[key][0]
                if global_cond_input.ndim == 3:
                    global_cond_input = global_cond_input.squeeze(1)
                global_conds.append(global_cond_input)
            global_cond = torch.cat(global_conds, dim=-1)

        return cross_attention_inputs, cross_attention_masks, global_cond

    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        cross_attention_inputs: torch.Tensor | None = None,
        cross_attention_masks: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        voice_cond: torch.Tensor | None = None,
        prepend_inputs: torch.Tensor | None = None,
        prepend_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.to_cond_embed is not None and cross_attention_inputs is not None:
            cross_attention_inputs = self.to_cond_embed(cross_attention_inputs)
        if self.to_global_embed is not None and global_cond is not None:
            global_cond = self.to_global_embed(global_cond)

        timestep_embed = self.to_timestep_embed(self.fourier(t.view(-1, 1)))
        global_cond = timestep_embed if global_cond is None else global_cond + timestep_embed

        voice_embed = None
        if voice_cond is not None and self.to_voice_embed is not None:
            voice_embed = self.to_voice_embed(voice_cond)

        if self.voice_cond_type == "cross" and voice_embed is not None:
            cross_attention_inputs = torch.cat((cross_attention_inputs, voice_embed), dim=1)
            voice_mask = torch.ones(
                (cross_attention_masks.shape[0], voice_embed.shape[1]),
                device=cross_attention_masks.device,
                dtype=torch.bool,
            )
            cross_attention_masks = torch.cat((cross_attention_masks, voice_mask), dim=1)

        if self.global_cond_type == "prepend":
            prepend_inputs = global_cond.unsqueeze(1) if global_cond.ndim == 2 else global_cond
            if self.voice_cond_type == "prepend" and voice_embed is not None:
                prepend_inputs = torch.cat((voice_embed, prepend_inputs), dim=1)
            prepend_mask = torch.ones((x.shape[0], prepend_inputs.shape[1]), device=x.device, dtype=torch.bool)

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")
        output = self.model(
            x,
            mask=mask,
            context=cross_attention_inputs,
            context_mask=cross_attention_masks,
            prepend_embeds=prepend_inputs,
            prepend_mask=prepend_mask,
        )

        prepend_length = prepend_inputs.shape[1] if prepend_inputs is not None else 0
        output = rearrange(output, "b t c -> b c t")
        if prepend_length > 0:
            output = output[:, :, prepend_length:]
        return self.postprocess_conv(output) + output

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seconds_start,
        seconds_total,
        voice_cond: torch.Tensor | None = None,
        cfg_dropout_prob: float = 0.1,
        cfg_scale: float | None = None,
        prepend_inputs: torch.Tensor | None = None,
        scale_phi: float = 0.75,
    ) -> torch.Tensor:
        cross_attention_inputs, cross_attention_masks, global_cond = self.get_context(
            input_ids, attention_mask, seconds_start, seconds_total
        )

        if cfg_dropout_prob > 0 and cross_attention_inputs is not None:
            null_embed = torch.zeros_like(cross_attention_inputs)
            dropout_mask = torch.bernoulli(
                torch.full((cross_attention_inputs.shape[0], 1, 1), cfg_dropout_prob, device=cross_attention_inputs.device)
            ).to(torch.bool)
            cross_attention_inputs = torch.where(dropout_mask, null_embed, cross_attention_inputs)

        if cfg_dropout_prob == 0.0 and cfg_scale is not None:
            cond_output = self._forward(
                x,
                t,
                mask=mask,
                cross_attention_inputs=cross_attention_inputs,
                cross_attention_masks=cross_attention_masks,
                global_cond=global_cond,
                voice_cond=voice_cond,
                prepend_inputs=prepend_inputs,
                prepend_mask=None,
            )

            null_cross = torch.zeros_like(cross_attention_inputs)
            uncond_output = self._forward(
                x,
                t,
                mask=mask,
                cross_attention_inputs=null_cross,
                cross_attention_masks=cross_attention_masks,
                global_cond=global_cond,
                voice_cond=None,
                prepend_inputs=prepend_inputs,
                prepend_mask=None,
            )

            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True).clamp_min(1e-6)
                return scale_phi * (cfg_output * (cond_out_std / out_cfg_std)) + (1 - scale_phi) * cfg_output
            return cfg_output

        return self._forward(
            x,
            t,
            mask=mask,
            cross_attention_inputs=cross_attention_inputs,
            cross_attention_masks=cross_attention_masks,
            global_cond=global_cond,
            voice_cond=voice_cond,
            prepend_inputs=prepend_inputs,
            prepend_mask=None,
        )
