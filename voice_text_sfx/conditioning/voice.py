from __future__ import annotations

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange


class VoiceConditionExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        n_chroma: int = 12,
        radix2_exp: int = 14,
        rms_repeats: int = 4,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.rms_repeats = rms_repeats
        self.winlen = 2 ** radix2_exp
        self.nfft = self.winlen
        self.winhop = self.winlen // 6

        fbanks = librosa.filters.chroma(
            sr=sample_rate,
            n_fft=self.nfft,
            tuning=0,
            n_chroma=n_chroma,
        )
        self.register_buffer("fbanks", torch.from_numpy(fbanks).float())

        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.nfft,
            win_length=self.winlen,
            hop_length=self.winhop,
            power=2,
            center=True,
            pad=0,
            normalized=True,
        )

    @property
    def output_dim(self) -> int:
        return self.n_chroma + self.rms_repeats

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mono = self._mono_like_original(audio)

        spec = self.spec(mono)
        raw_chroma = torch.einsum("cf,bft->bct", self.fbanks.to(spec.device, spec.dtype), spec)
        norm_chroma = F.normalize(raw_chroma, p=float("inf"), dim=1, eps=1e-6)

        rms = self._rms_like_original(mono, target_frames=norm_chroma.shape[-1], dtype=norm_chroma.dtype)

        total = torch.cat((rms, norm_chroma), dim=1)
        return rearrange(total, "b d t -> b t d")

    @staticmethod
    def _mono_like_original(audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3 and audio.shape[1] == 2:
            mono = (audio[:, 0, :] + audio[:, 1, :]) / 2
        elif audio.dim() == 3 and audio.shape[1] == 1:
            mono = audio[:, 0, :]
        elif audio.dim() == 2 and audio.shape[0] == 2:
            mono = (audio[0, :] + audio[1, :]) / 2
        elif audio.dim() == 2:
            mono = audio
        elif audio.dim() == 1:
            mono = audio
        else:
            raise ValueError(f"Unsupported audio shape: {tuple(audio.shape)}")

        if mono.dim() == 1:
            mono = mono.unsqueeze(0)
        return mono

    def _rms_like_original(self, mono: torch.Tensor, target_frames: int, dtype: torch.dtype) -> torch.Tensor:
        rms = librosa.feature.rms(y=mono.detach().float().cpu().numpy())
        hop = rms.shape[-1] // target_frames

        pooled = []
        for i in range(target_frames):
            start = max(i * hop - 3, 0)
            end = (i + 1) * hop + 3
            pooled.append(np.sum(rms[:, :, start:end], axis=-1, keepdims=True))

        qrmss = torch.as_tensor(np.array(pooled), device=mono.device, dtype=dtype)
        qrmss = qrmss.squeeze(-1).squeeze(-1)
        qrmss = rearrange(qrmss, "t b -> b t").unsqueeze(1).expand(-1, self.rms_repeats, -1)
        qrmss = (qrmss - qrmss.min()) / (qrmss.max() - qrmss.min())
        return qrmss


def make_voice_condition(audio: torch.Tensor, sample_rate: int = 44100) -> torch.Tensor:
    extractor = VoiceConditionExtractor(sample_rate=sample_rate).to(audio.device)
    return extractor(audio)
