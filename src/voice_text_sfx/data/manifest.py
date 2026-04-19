from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


def _load_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open() as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def _resample(audio: torch.Tensor, src_sr: int, target_sr: int) -> torch.Tensor:
    if src_sr == target_sr:
        return audio
    return torchaudio.functional.resample(audio, src_sr, target_sr)


def _match_channels(audio: torch.Tensor, channels: int) -> torch.Tensor:
    if audio.shape[0] == channels:
        return audio
    if channels == 1:
        return audio.mean(dim=0, keepdim=True)
    if audio.shape[0] == 1 and channels == 2:
        return audio.repeat(2, 1)
    if audio.shape[0] > channels:
        return audio[:channels]
    repeats = channels // audio.shape[0] + int(channels % audio.shape[0] != 0)
    return audio.repeat(repeats, 1)[:channels]


def _crop_or_pad(audio: torch.Tensor, target_samples: int, random_crop: bool = True, crop_start: int | None = None):
    length = audio.shape[-1]
    if length > target_samples:
        if crop_start is None:
            max_start = length - target_samples
            crop_start = random.randint(0, max_start) if random_crop else 0
        audio = audio[..., crop_start : crop_start + target_samples]
        return audio, crop_start

    if length < target_samples:
        pad = target_samples - length
        audio = torch.nn.functional.pad(audio, (0, pad))
    return audio, 0 if crop_start is None else crop_start


class VoiceTextSFXDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 44100,
        channels: int = 2,
        segment_seconds: float = 3.0,
        random_crop: bool = True,
    ):
        self.rows = _load_manifest(manifest_path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.segment_seconds = segment_seconds
        self.segment_samples = int(segment_seconds * sample_rate)
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        audio_path = Path(row["audio_path"])
        conditioning_audio_path = Path(row.get("conditioning_audio_path") or row["audio_path"])

        target_audio, target_sr = torchaudio.load(audio_path)
        target_audio = _resample(target_audio, target_sr, self.sample_rate)
        target_audio = _match_channels(target_audio, self.channels)
        target_audio, crop_start = _crop_or_pad(target_audio, self.segment_samples, random_crop=self.random_crop)

        conditioning_audio, conditioning_sr = torchaudio.load(conditioning_audio_path)
        conditioning_audio = _resample(conditioning_audio, conditioning_sr, self.sample_rate)

        if conditioning_audio_path.resolve() == audio_path.resolve():
            conditioning_audio, _ = _crop_or_pad(conditioning_audio, self.segment_samples, random_crop=False, crop_start=crop_start)
        else:
            conditioning_audio, _ = _crop_or_pad(conditioning_audio, self.segment_samples, random_crop=self.random_crop)

        seconds_start = float(row.get("seconds_start") or 0.0)
        seconds_total = float(row.get("seconds_total") or (target_audio.shape[-1] / self.sample_rate))

        return {
            "audio": target_audio,
            "conditioning_audio": conditioning_audio,
            "caption": row["caption"],
            "seconds_start": torch.tensor(seconds_start, dtype=torch.float32),
            "seconds_total": torch.tensor(seconds_total, dtype=torch.float32),
            "audio_path": str(audio_path),
            "conditioning_audio_path": str(conditioning_audio_path),
        }

