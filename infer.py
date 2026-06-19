#!/usr/bin/env python3
"""Run the VTS inference path locally, without RunPod or Supabase."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


log("starting infer.py; importing dependencies")

import numpy as np
import soundfile as sf
import torch
import torchaudio
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer

log("base dependencies imported")


PROJECT_ROOT = Path(__file__).resolve().parent
VTS_ROOT = PROJECT_ROOT / "vts"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "dynamic_v3_0415.ckpt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_vts_outputs"
BUCKET_LENGTHS = (86, 192, 384, 768, 1536)

sys.path.insert(0, str(VTS_ROOT))

log(f"importing VTS modules from {VTS_ROOT}")
from model.module_voice import VTSModule  # noqa: E402
from torchode.interface import solve_ivp  # noqa: E402
from utils.utils import get_dynamic, span_mask_strided  # noqa: E402
from vocos_custom import get_voco  # noqa: E402
log("VTS modules imported")


def cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:
        print(f"CUDA is visible but unusable, falling back to CPU: {exc}", file=sys.stderr)
        return False
    return True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if cuda_is_usable() else "cpu")
    if requested == "cuda" and not cuda_is_usable():
        raise RuntimeError("CUDA was requested, but CUDA is not usable in this environment.")
    return torch.device(requested)


def download_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.exists():
        log(f"checkpoint found: {checkpoint_path}")
        return checkpoint_path

    log(f"checkpoint missing, downloading to: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or None
    downloaded = hf_hub_download(
        repo_id="Daniel777/textalignment",
        filename="dynamic_v3_0415.ckpt",
        local_dir=str(checkpoint_path.parent),
        local_dir_use_symlinks=False,
        token=token,
    )
    downloaded_path = Path(downloaded).resolve()
    if downloaded_path != checkpoint_path and downloaded_path.exists():
        return downloaded_path
    return checkpoint_path


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    log(f"reading input audio: {path}")
    data, sample_rate = sf.read(path, always_2d=True)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    data = np.asarray(data, dtype=np.float32)
    data = np.clip(data, -1.0, 1.0)
    if data.shape[1] > 2:
        data = data[:, :2]
    log(f"input audio loaded: shape={data.shape}, sample_rate={sample_rate}")
    return data, int(sample_rate)


def to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def target_bucket(length: int) -> int:
    for bucket in BUCKET_LENGTHS:
        if length <= bucket:
            return bucket
    return length


def fit_time(tensor: Tensor, length: int, value: float | bool = 0.0) -> Tensor:
    current = tensor.shape[1]
    if current == length:
        return tensor
    if current > length:
        return tensor[:, :length]
    pad_value = bool(value) if tensor.dtype == torch.bool else float(value)
    return F.pad(tensor, (0, 0, 0, length - current), value=pad_value)


def fit_mask(mask: Tensor, length: int) -> Tensor:
    current = mask.shape[1]
    if current == length:
        return mask
    if current > length:
        return mask[:, :length]
    return F.pad(mask, (0, length - current), value=False)


def prepare_dynamic(audio: np.ndarray, sample_rate: int) -> Tensor:
    log("extracting dynamic conditioning from input audio")
    merged = audio[None, :, :]
    max_len = int((merged.shape[1] / sample_rate) * 21.5 + 0.5)
    waveform = rearrange(torch.tensor(merged), "b t c -> b c t").float()
    dynamic = get_dynamic(waveform, max_len=max_len)
    if dynamic.ndim == 2:
        dynamic = dynamic.unsqueeze(0)
    log(f"dynamic ready: shape={tuple(dynamic.shape)}")
    return dynamic


class LocalVTSInfer:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        steps: int = 64,
        alpha: float = 3.0,
    ):
        self.device = device
        log(f"loading checkpoint on device={device}: {checkpoint}")
        try:
            self.model = VTSModule.load_from_checkpoint(
                str(checkpoint),
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            self.model = VTSModule.load_from_checkpoint(str(checkpoint), map_location=device)
        log("checkpoint loaded")
        log(f"moving model to {device}")
        self.model.to(device)
        self.model.eval()

        log(f"loading vocoder: {self.model.voco_type}")
        self.voco = get_voco(self.model.voco_type).to(device)
        log(
            "vocoder ready: "
            f"type={self.model.voco_type}, sr={self.voco.sampling_rate}, latent_dim={self.voco.latent_dim}"
        )

        log("loading tokenizer: google/flan-t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.tokenizer.padding_side = "right"
        log("tokenizer ready")

        self.steps = steps
        self.alpha = alpha

    @property
    def sampling_rate(self) -> int:
        return int(self.model.sampling_rate)

    def autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda")
        return contextlib.nullcontext()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        log(f"encoding text: batch={len(texts)}")
        batch_encoding = self.tokenizer(
            [text + self.tokenizer.eos_token for text in texts],
            add_special_tokens=False,
            return_tensors="pt",
            max_length=127,
            truncation="longest_first",
            padding="max_length",
        )
        input_ids = batch_encoding.input_ids.to(self.device)
        attention_mask = batch_encoding.attention_mask.to(self.device) > 0
        with torch.autocast(device_type=self.device.type, enabled=False):
            text_emb = self.model.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        log(f"text encoded: shape={tuple(text_emb.shape)}")
        return text_emb, attention_mask

    @torch.no_grad()
    def generate(
        self,
        texts: list[str],
        duration: float,
        cfg_score: float = 3.0,
        voice_enc: Tensor | None = None,
    ) -> list[np.ndarray]:
        with self.autocast_context():
            log("generate branch started")
            text_emb, text_mask = self.encode_text(texts)
            batch_size = text_emb.shape[0]

            target_len = round(self.sampling_rate * duration)
            latent_len = self.voco.encode_length(target_len)
            bucket_len = target_bucket(int(latent_len))
            log(
                "generate lengths: "
                f"duration={duration:.3f}s, target_len={target_len}, "
                f"latent_len={latent_len}, bucket_len={bucket_len}, steps={self.steps}"
            )

            audio_mask = torch.ones(
                batch_size,
                latent_len,
                dtype=torch.bool,
                device=self.device,
            )
            audio_context = torch.zeros(
                batch_size,
                latent_len,
                self.voco.latent_dim,
                device=self.device,
            )
            if voice_enc is None:
                voice_enc = torch.zeros(batch_size, latent_len, 12, device=self.device)
            else:
                voice_enc = voice_enc.to(self.device)

            audio_mask = fit_mask(audio_mask, bucket_len)
            audio_context = fit_time(audio_context, bucket_len)
            voice_enc = fit_time(voice_enc, bucket_len)

            def fn(t: Tensor, y: Tensor) -> Tensor:
                return self.model.vts.cfg(
                    w=y,
                    context=audio_context,
                    times=t,
                    alpha=cfg_score,
                    mask=audio_mask,
                    phoneme_emb=text_emb,
                    phoneme_mask=text_mask,
                    voice_enc=voice_enc,
                )

            y0 = torch.randn_like(audio_context)
            t = torch.linspace(0, 1, self.steps, device=self.device)
            t = repeat(t, "n -> b n", b=batch_size)
            log("starting ODE solve for generate")
            sol = solve_ivp(fn, y0, t, method_class=self.model.method)
            log("ODE solve finished for generate; decoding")
            sampled_audio = sol.ys[-1]
            sample = self.voco.decode(sampled_audio)[:, :target_len]
            sample = sample / sample.abs().amax(dim=1, keepdim=True).clamp_min(1)
            log("generate branch finished")
            return [audio for audio in sample.detach().cpu().numpy().astype(np.float32)]

    @torch.no_grad()
    def variation(
        self,
        audios: list[np.ndarray],
        texts: list[str],
        duration: float,
        corrupt: float,
        sample_rate: int,
        cfg_score: float = 3.0,
        voice_enc: Tensor | None = None,
    ) -> list[np.ndarray]:
        with self.autocast_context():
            log("variation branch started")
            text_emb, text_mask = self.encode_text(texts)
            batch_size = text_emb.shape[0]

            float_audios = [audio.astype(np.float32) / np.iinfo(audio.dtype).max for audio in audios]
            audio_tensor = torch.from_numpy(np.stack(float_audios, axis=0)).to(self.device).float()
            audio_tensor = audio_tensor.transpose(1, 2)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.contiguous(),
                orig_freq=sample_rate,
                new_freq=self.voco.sampling_rate,
            )
            audio_tensor = audio_tensor.transpose(1, 2)

            if audio_tensor.shape[2] == 1:
                audio_tensor = audio_tensor.repeat(1, 1, 2)
            elif audio_tensor.shape[2] > 2:
                audio_tensor = audio_tensor[:, :, :2]

            target_len = audio_tensor.shape[1]
            latent_len = self.voco.encode_length(target_len)
            bucket_len = target_bucket(int(latent_len))
            log(
                "variation lengths: "
                f"duration={duration:.3f}s, target_len={target_len}, "
                f"latent_len={latent_len}, bucket_len={bucket_len}, steps={self.steps}, corrupt={corrupt}"
            )

            log("encoding input audio to vocoder latent")
            audio_enc = self.voco.encode(audio_tensor)
            audio_mask = torch.ones(batch_size, latent_len, dtype=torch.bool, device=self.device)
            audio_context = torch.zeros(
                batch_size,
                latent_len,
                self.voco.latent_dim,
                device=self.device,
            )
            if voice_enc is None:
                voice_enc = torch.zeros(batch_size, latent_len, 12, device=self.device)
            else:
                voice_enc = voice_enc.to(self.device)

            audio_enc = fit_time(audio_enc, bucket_len)
            audio_mask = fit_mask(audio_mask, bucket_len)
            audio_context = fit_time(audio_context, bucket_len)
            voice_enc = fit_time(voice_enc, bucket_len)

            sigma = 1e-3
            c = 1.0 - corrupt
            noised_enc = (audio_enc * c) + torch.randn_like(audio_enc) * (1 - (1 - sigma) * c)

            def fn(t: Tensor, y: Tensor) -> Tensor:
                return self.model.vts.cfg(
                    w=y,
                    context=audio_context,
                    times=t,
                    alpha=cfg_score,
                    mask=audio_mask,
                    phoneme_emb=text_emb,
                    phoneme_mask=text_mask,
                    voice_enc=voice_enc,
                )

            t = torch.linspace(c, 1.0, self.steps, device=self.device)
            t = repeat(t, "n -> b n", b=batch_size)
            log("starting ODE solve for variation")
            sol = solve_ivp(fn, noised_enc, t, method_class=self.model.method)
            log("ODE solve finished for variation; decoding")
            sampled_audio = sol.ys[-1]
            new_target_len = round(self.sampling_rate * duration)
            sample = self.voco.decode(sampled_audio)[:, :new_target_len]
            sample = sample / sample.abs().amax(dim=1, keepdim=True).clamp_min(1)
            log("variation branch finished")
            return [audio for audio in sample.detach().cpu().numpy().astype(np.float32)]


def temperature_branch(temperature: float) -> str:
    if temperature < 0.6:
        return "low"
    if temperature < 0.8:
        return "medium"
    return "high"


def normalize_texts(texts: list[str], num_samples: int, alpha: float) -> tuple[list[str], float]:
    texts = texts or [""]
    if len(texts) == 1 and num_samples > 1:
        texts = texts * num_samples
    cfg_score = 0.0 if len(texts) == 1 and texts[0] == "" else alpha
    return texts, cfg_score


def next_sample_index(output_dir: Path) -> int:
    max_index = -1
    if output_dir.is_dir():
        for path in output_dir.glob("sample_*.wav"):
            suffix = path.stem.removeprefix("sample_")
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return max_index + 1

def save_outputs(outputs: list[np.ndarray], output_dir: Path, sample_rate: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for offset, audio in enumerate(outputs):
        output_path = output_dir / f"sample_{start_index + offset:02d}.wav"
        log(f"writing output: {output_path}")
        sf.write(output_path, audio, sample_rate)
        paths.append(str(output_path.resolve()))
    return paths


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = Path(args.checkpoint)
    input_audio = Path(args.input_audio) if args.input_audio else None
    cuda_usable = cuda_is_usable()
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_usable": cuda_usable,
        "device_auto_would_use": "cuda" if cuda_usable else "cpu",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        "input_audio": str(input_audio.resolve()) if input_audio else None,
        "input_audio_exists": input_audio.exists() if input_audio else None,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.seed is not None:
        log(f"setting seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if cuda_is_usable():
            torch.cuda.manual_seed_all(args.seed)

    if args.diagnose:
        return diagnose(args)

    checkpoint = download_checkpoint(Path(args.checkpoint))
    if args.download_only:
        return {"checkpoint": str(checkpoint), "download_only": True}

    input_audio = Path(args.input_audio)
    source_audio, source_sr = read_audio(input_audio)
    duration = args.duration if args.duration is not None else source_audio.shape[0] / source_sr
    texts, cfg_score = normalize_texts(args.text, args.num_samples, args.alpha)
    log(f"text prompts: {texts}")
    log(f"cfg_score={cfg_score}")

    device = resolve_device(args.device)
    log(f"selected device: {device}")
    infer = LocalVTSInfer(
        checkpoint=checkpoint,
        device=device,
        steps=args.steps,
        alpha=args.alpha,
    )

    dynamic = prepare_dynamic(source_audio, source_sr)
    branch = temperature_branch(args.temperature)
    log(f"temperature={args.temperature} -> branch={branch}")

    if branch == "low":
        dynamic = span_mask_strided(dynamic, 1, 2)
        dynamic[:, :, :4] = 0.0
        dynamic = dynamic.to(device).expand(len(texts), -1, -1)
        outputs = infer.generate(texts, duration, cfg_score=cfg_score, voice_enc=dynamic)
    elif branch == "medium":
        dynamic = dynamic.to(device).expand(len(texts), -1, -1)
        outputs = infer.generate(texts, duration, cfg_score=cfg_score, voice_enc=dynamic)
    else:
        audio_int16 = to_int16(source_audio)
        audios = [audio_int16.copy() for _ in texts]
        dynamic = dynamic.to(device).expand(len(texts), -1, -1)
        outputs = infer.variation(
            audios=audios,
            texts=texts,
            duration=duration,
            corrupt=args.temperature,
            sample_rate=source_sr,
            cfg_score=cfg_score,
            voice_enc=dynamic,
        )

    output_paths = save_outputs(outputs, Path(args.output_dir), infer.sampling_rate)
    log("all outputs written")
    return {
        "branch": branch,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "duration": duration,
        "input_audio": str(input_audio.resolve()),
        "output_paths": output_paths,
        "sample_rate": infer.sampling_rate,
        "temperature": args.temperature,
        "texts": texts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local VTS inference for voice-audio + text-to-sound generation.",
    )
    parser.add_argument("--input-audio", help="Path to the conditioning voice/audio file.")
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text prompt. Repeat this flag to generate multiple prompts.",
    )
    parser.add_argument("--num-samples", type=int, default=1, help="Repeat one prompt N times.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--duration", type=float, default=None, help="Default: input audio duration.")
    parser.add_argument(
        "--checkpoint",
        "--model-path",
        dest="checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to dynamic_v3_0415.ckpt. --model-path is an alias.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print environment/checkpoint/input diagnostics and exit before loading the model.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download/verify the VTS checkpoint, then exit.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.temperature <= 1.0:
        parser.error("--temperature must be between 0.0 and 1.0.")
    if not args.download_only and not args.input_audio:
        parser.error("--input-audio is required unless --download-only is used.")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
