from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake_beta(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


def get_activation(activation: str, channels: int | None = None) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        if channels is None:
            raise ValueError("channels is required for snake activation")
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, use_snake: bool = False):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", channels=in_channels),
            WNConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        use_snake: bool = False,
    ):
        super().__init__()
        c_mults = [1] + (c_mults or [1, 2, 4, 8])
        strides = strides or [2, 4, 8, 8]
        layers = [WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1):
            layers.append(EncoderBlock(c_mults[i] * channels, c_mults[i + 1] * channels, stride=strides[i], use_snake=use_snake))
        layers.extend(
            [
                get_activation("snake" if use_snake else "elu", channels=c_mults[-1] * channels),
                WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=1),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        use_snake: bool = False,
        final_tanh: bool = True,
    ):
        super().__init__()
        c_mults = [1] + (c_mults or [1, 2, 4, 8])
        strides = strides or [2, 4, 8, 8]
        layers = [WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(DecoderBlock(c_mults[i] * channels, c_mults[i - 1] * channels, stride=strides[i - 1], use_snake=use_snake))
        layers.extend(
            [
                get_activation("snake" if use_snake else "elu", channels=c_mults[0] * channels),
                WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def vae_sample(mean: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


class VAEBottleneck(nn.Module):
    is_discrete = False

    def encode(self, x: torch.Tensor, return_info: bool = False, **kwargs):
        mean, scale = x.chunk(2, dim=1)
        latents, kl = vae_sample(mean, scale)
        if return_info:
            return latents, {"kl": kl}
        return latents

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        downsampling_ratio: int,
        sample_rate: int,
        io_channels: int = 2,
        bottleneck: nn.Module | None = None,
        soft_clip: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.io_channels = io_channels
        self.bottleneck = bottleneck
        self.soft_clip = soft_clip

    def encode(self, audio: torch.Tensor, return_info: bool = False, **kwargs):
        latents = self.encoder(audio)
        if self.bottleneck is not None:
            latents, info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            if return_info:
                return latents, info
        elif return_info:
            return latents, {}
        return latents

    def encode_audio(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encode(audio, **kwargs)

    def decode(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)
        audio = self.decoder(latents)
        if self.soft_clip:
            audio = torch.tanh(audio)
        return audio


def create_encoder_from_config(encoder_config: dict[str, Any]) -> nn.Module:
    encoder_type = encoder_config["type"]
    if encoder_type != "oobleck":
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    encoder = OobleckEncoder(**encoder_config["config"])
    if not encoder_config.get("requires_grad", True):
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder


def create_decoder_from_config(decoder_config: dict[str, Any]) -> nn.Module:
    decoder_type = decoder_config["type"]
    if decoder_type != "oobleck":
        raise ValueError(f"Unsupported decoder type: {decoder_type}")
    decoder = OobleckDecoder(**decoder_config["config"])
    if not decoder_config.get("requires_grad", True):
        for param in decoder.parameters():
            param.requires_grad = False
    return decoder


def create_autoencoder_from_config(config: dict[str, Any]) -> AudioAutoencoder:
    encoder = create_encoder_from_config(config["encoder"])
    decoder = create_decoder_from_config(config["decoder"])
    bottleneck_cfg = config.get("bottleneck")
    bottleneck = VAEBottleneck() if bottleneck_cfg and bottleneck_cfg.get("type") == "vae" else None
    return AudioAutoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config["latent_dim"],
        downsampling_ratio=config["downsampling_ratio"],
        sample_rate=config["sample_rate"],
        io_channels=config["io_channels"],
        bottleneck=bottleneck,
        soft_clip=config["decoder"].get("soft_clip", False),
    )

