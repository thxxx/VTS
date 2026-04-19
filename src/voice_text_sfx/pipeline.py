from __future__ import annotations

import copy
from pathlib import Path

import torch
import torchaudio

from .autoencoder import create_autoencoder_from_config
from .conditioning import T5Conditioner, VoiceConditionExtractor
from .config import (
    get_default_autoencoder_config,
    get_default_backbone_config,
    get_default_transformer_init,
)
from .inference import generate_audio
from .models import VoiceConditionedDiffusionTransformer


def _extract_state_dict(checkpoint: dict | torch.Tensor, preferred_keys: tuple[str, ...] = ("model", "state_dict", "autoencoder")):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in preferred_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _strip_prefixes(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def load_module_checkpoint(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    preferred_keys: tuple[str, ...] = ("model", "state_dict"),
    strip_prefixes: tuple[str, ...] = ("module.",),
    strict: bool = True,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint, preferred_keys=preferred_keys)
    state_dict = _strip_prefixes(state_dict, prefixes=strip_prefixes)
    module.load_state_dict(state_dict, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def build_components(
    device: str | torch.device = "cuda",
    *,
    model_config: dict | None = None,
    autoencoder_config: dict | None = None,
    transformer_init: dict | None = None,
):
    device = torch.device(device)
    backbone_config = copy.deepcopy(model_config or get_default_backbone_config())
    ae_config = copy.deepcopy(autoencoder_config or get_default_autoencoder_config())
    transformer_init = copy.deepcopy(transformer_init or get_default_transformer_init())

    model = VoiceConditionedDiffusionTransformer(
        config=backbone_config,
        device=device,
        **transformer_init,
    ).to(device)

    autoencoder = create_autoencoder_from_config(ae_config).to(device)
    text_conditioner = T5Conditioner(**backbone_config["text_conditioner_config"]).to(device)
    voice_extractor = VoiceConditionExtractor(sample_rate=ae_config["sample_rate"]).to(device)

    autoencoder.eval()
    text_conditioner.eval()
    voice_extractor.eval()

    return model, autoencoder, text_conditioner, voice_extractor


class VoiceTextSFXPipeline:
    def __init__(self, model, autoencoder, text_conditioner, voice_extractor, device: str | torch.device = "cuda"):
        self.model = model
        self.autoencoder = autoencoder
        self.text_conditioner = text_conditioner
        self.voice_extractor = voice_extractor
        self.device = torch.device(device)

    @classmethod
    def from_checkpoints(
        cls,
        *,
        model_checkpoint: str | Path,
        autoencoder_checkpoint: str | Path,
        device: str | torch.device = "cuda",
        model_config: dict | None = None,
        autoencoder_config: dict | None = None,
        transformer_init: dict | None = None,
    ):
        model, autoencoder, text_conditioner, voice_extractor = build_components(
            device=device,
            model_config=model_config,
            autoencoder_config=autoencoder_config,
            transformer_init=transformer_init,
        )

        load_module_checkpoint(
            model,
            model_checkpoint,
            preferred_keys=("model", "state_dict"),
            strip_prefixes=("module.",),
            strict=True,
        )
        load_module_checkpoint(
            autoencoder,
            autoencoder_checkpoint,
            preferred_keys=("autoencoder", "model", "state_dict"),
            strip_prefixes=("module.", "model."),
            strict=False,
        )

        model.eval()
        autoencoder.eval()
        return cls(model, autoencoder, text_conditioner, voice_extractor, device=device)

    def generate(
        self,
        *,
        prompt_audio: torch.Tensor,
        text: str,
        duration: float = 3.0,
        steps: int = 100,
        cfg_scale: float = 6.0,
    ) -> torch.Tensor:
        with torch.no_grad():
            voice_cond = self.voice_extractor(prompt_audio.to(self.device))
            return generate_audio(
                self.model,
                self.autoencoder,
                self.text_conditioner,
                text=text,
                voice_cond=voice_cond,
                steps=steps,
                cfg_scale=cfg_scale,
                duration=duration,
                sample_rate=self.autoencoder.sample_rate,
                batch_size=voice_cond.shape[0],
                device=self.device,
            )

    def generate_from_audio_file(
        self,
        *,
        prompt_audio_path: str | Path,
        text: str,
        duration: float = 3.0,
        steps: int = 100,
        cfg_scale: float = 6.0,
    ) -> torch.Tensor:
        audio, sr = torchaudio.load(prompt_audio_path)
        if sr != self.autoencoder.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.autoencoder.sample_rate)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        return self.generate(
            prompt_audio=audio,
            text=text,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
        )

    @staticmethod
    def save_audio(audio: torch.Tensor, output_path: str | Path, sample_rate: int = 44100):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), audio.detach().cpu().squeeze(0), sample_rate)

