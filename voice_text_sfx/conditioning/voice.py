from __future__ import annotations

import librosa
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
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if audio.dim() == 2:
            mono = audio
        elif audio.dim() == 3:
            mono = audio.mean(dim=1)
        else:
            raise ValueError(f"Unsupported audio shape: {tuple(audio.shape)}")

        spec = self.spec(mono)
        raw_chroma = torch.einsum("cf,bft->bct", self.fbanks.to(spec.device, spec.dtype), spec)
        norm_chroma = F.normalize(raw_chroma, p=float("inf"), dim=1, eps=1e-6)

        rms = self._frame_rms(mono, target_frames=norm_chroma.shape[-1])
        rms = self._min_max_normalize(rms).unsqueeze(1).expand(-1, self.rms_repeats, -1)

        total = torch.cat((rms, norm_chroma), dim=1)
        return rearrange(total, "b d t -> b t d")

    def _frame_rms(self, mono: torch.Tensor, target_frames: int) -> torch.Tensor:
        rms = torch.sqrt(
            F.avg_pool1d(
                mono.unsqueeze(1).pow(2),
                kernel_size=self.winlen,
                stride=self.winhop,
                padding=self.winlen // 2,
            ).clamp_min(1e-8)
        ).squeeze(1)
        if rms.shape[-1] != target_frames:
            rms = F.interpolate(rms.unsqueeze(1), size=target_frames, mode="linear", align_corners=False).squeeze(1)
        return rms

    @staticmethod
    def _min_max_normalize(values: torch.Tensor) -> torch.Tensor:
        min_val = values.amin(dim=-1, keepdim=True)
        max_val = values.amax(dim=-1, keepdim=True)
        return (values - min_val) / (max_val - min_val + 1e-8)


def make_voice_condition(audio: torch.Tensor, sample_rate: int = 44100) -> torch.Tensor:
    extractor = VoiceConditionExtractor(sample_rate=sample_rate).to(audio.device)
    return extractor(audio)

