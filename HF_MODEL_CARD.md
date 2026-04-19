---
license: mit
library_name: pytorch
tags:
  - audio-generation
  - sound-effects
  - voice-conditioned
  - text-conditioned
  - pytorch
---

# VTS

![VTS overview](./Thumbnail.png)

VTS (Voice To Sound) generates sound effects from:

- a short vocal sketch
- a text prompt

This repository hosts the pretrained checkpoint files for the older `voice_cond` VTS pipeline.

## Files

- `model_voice_1030_24.pth`: main diffusion checkpoint
- `vae_weight.pth`: VAE checkpoint used for decoding

## Download

```bash
pip install -U "huggingface_hub"
hf download Daniel777/VTS model_voice_1030_24.pth vae_weight.pth --local-dir ./checkpoints
```

## Usage

Use these checkpoints with the companion `voice_text_sfx` codebase.

```bash
python3 scripts/infer.py \
  --model-ckpt ./checkpoints/model_voice_1030_24.pth \
  --ae-ckpt ./checkpoints/vae_weight.pth \
  --prompt-audio /path/to/prompt.wav \
  --text "glassy swipe with rising pitch" \
  --output /tmp/generated.wav \
  --duration 3.0 \
  --steps 100 \
  --cfg-scale 6.0 \
  --device cuda
```

## Notes

- This checkpoint matches the older `voice_cond` path.
- It is not a drop-in checkpoint for later `script_embed` or `voice_prompt` variants.
- This is a research checkpoint, not a packaged Hugging Face Inference API model.

## SHA256

- `model_voice_1030_24.pth`: `a061bfb5e4fca61d8857c3056245304d0a421b55d4f86deca3b47442b08f5287`
- `vae_weight.pth`: `45e2d5ab17e5bbb22dc533cd70798bb4ed96dbbe3487f6f20f5528fc9915558e`
