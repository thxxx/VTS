## 🌟 Why This Project Exists

**Describing sound with text alone is surprisingly hard.**

Try picking a sound in your head(e.g., Minecraft chest opening or Creeper exploding). **Can you describe the sound directly as text?** At best, you can only describe the situation.

That is why sound-design meetings often turn into a brief beatboxing session(🔫 pew pew, 💥 boom) When words stop being precise enough, people make the sound with their mouths.

VTS turns that behavior into a new interface. **Instead of relying on text alone, you can give the model a short vocal sketch together with a text prompt.**

`The voice carries timing, contour, and feel; the text keeps the generation anchored to intent.`

# VTS (Voice To Sound)

`Describing a sound with text is hard.` You can hear it in your head immediately, but the moment you try to write it down, it usually turns into vague words or bad beatboxing.

VTS lets you do the obvious thing instead: sketch the sound with your voice, add a short text prompt, and generate a sound effect from both.

<img src="./Thumbnail.png" alt="VTS overview" width="100%" />

Generate sound effects from:

- a short voice sketch (`pshh-kting, clank`)
- a text prompt

If you have ever typed "metallic sci-fi impact with a short tail" and then immediately made a much more useful `pshh-kting` sound with your mouth, this repo is for you.

## ✨ Highlights

- **Model checkpoint:** uses the VTS checkpoint `dynamic_v3_0415.ckpt`.
- **Architecture:** voice/text-conditioned VTS inference with a frozen `google/flan-t5-base` text encoder, local model code, local ODE solver, and local vocoder code.
- **Conditioning:** combines a short **vocal/audio sketch** with a **text prompt**; voice conditioning uses dynamic features derived from spectral centroid, RMS, and chroma-index signals.
- **Audio latent setup:** input audio is converted into frame-level dynamic conditioning at roughly 21.5 fps; high-temperature variation also encodes the input audio into the vocoder latent space.
- **Generation length:** defaults to the input audio duration; `--duration` is configurable, but the checkpoint is tuned around short SFX clips.
- **Sampling:** ODE-based sampling through the local `vts/torchode` implementation, typically with 64 steps and CFG scale `--alpha 3.0`.

## Demo

_[Demo link](https://spicy-pufferfish-699.notion.site/VTS-347cf95761f480f19dc0eb790e1467af?source=copy_link)_

## Repository Scope

This repository is an inference-only VTS package. It contains the local inference entrypoint and the model/runtime code needed by that entrypoint.

```text
infer.py                       # local inference entrypoint
local_vts_infer.py             # same inference script, kept for compatibility
vts/model/                     # VTS model code
vts/torchode/                  # local ODE solver used by inference
vts/utils/                     # dynamic feature extraction and masking
vts/vocos_custom/              # vocoder encode/decode code
requirements.txt
setup_vts_local.sh
```

Training code, datasets, notebooks, RunPod handler, Supabase upload code, OpenAI prompt rewriting code, and private samples are intentionally excluded.

## Checkpoints

Pretrained checkpoints are available on Hugging Face:

- [https://huggingface.co/Daniel777/VTS](https://huggingface.co/Daniel777/VTS)

Download:

```bash
pip install -U "huggingface_hub"
hf download Daniel777/VTS dynamic_v3_0415.ckpt --local-dir ./checkpoints
```

Recommended local layout:

```text
checkpoints/
  dynamic_v3_0415.ckpt
```

The inference script also downloads `dynamic_v3_0415.ckpt` automatically if the default checkpoint path is missing. If the Hugging Face repo requires authentication, set `HF_TOKEN` before running inference.

```bash
export HF_TOKEN=hf_...
python -u infer.py --download-only
```

## Installation

Create a fresh environment first. The provided setup script creates `.venv-vts`, installs the pinned local inference requirements, and verifies the VTS imports.

```bash
cd vts_inference
bash setup_vts_local.sh
source .venv-vts/bin/activate
```

The provided requirements pin the versions used by the working local inference path:

```text
torch==2.4.1+cu124
torchaudio==2.4.1+cu124
transformers==4.47.0
```

If your CUDA driver requires another PyTorch build, install the matching `torch` and `torchaudio` wheels first, then install the rest of the requirements.

## Quick Start

### Inference

You need:

- the VTS checkpoint `dynamic_v3_0415.ckpt`
- a prompt audio clip for voice conditioning
- a text prompt

```bash
python -u infer.py \
  --input-audio ./examples/voice.wav \
  --text "scifi cannon charging and shooting" \
  --temperature 0.7 \
  --model-path ./checkpoints/dynamic_v3_0415.ckpt \
  --output-dir ./outputs \
  --duration 3.0 \
  --steps 64 \
  --device cuda
```

For a quick smoke test, reduce the ODE steps:

```bash
python -u infer.py \
  --input-audio ./examples/voice.wav \
  --text "scifi cannon charging and shooting" \
  --temperature 0.7 \
  --model-path ./checkpoints/dynamic_v3_0415.ckpt \
  --output-dir ./outputs \
  --device cuda \
  --steps 16
```

To inspect the environment before loading the model:

```bash
python -u infer.py \
  --diagnose \
  --input-audio ./examples/voice.wav \
  --model-path ./checkpoints/dynamic_v3_0415.ckpt
```

### Training

This repository is inference-only. Training scripts and dataset manifests are not included in this package.

## Temperature Guide

`--temperature` does more than control randomness in this VTS inference path. It selects how strongly the input audio is reused.

For normal inference, use `--temperature 0.7`. This keeps the original dynamic conditioning from the input audio and runs the standard `generate` path.

| Temperature      | Path        | Input audio usage                                                                                                                                            |
| ---------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `< 0.6`          | `generate`  | Extract dynamic features, weaken them with strided masking, zero the first 4 channels, then condition generation on the weakened dynamic tensor.             |
| `0.6 <= t < 0.8` | `generate`  | Extract dynamic features and condition generation on the full dynamic tensor.                                                                                |
| `>= 0.8`         | `variation` | Extract dynamic features, encode the input audio into the vocoder latent space, mix that latent with noise, then continue generation from the noised latent. |

The accepted range is `0.0` to `1.0`. The default in `infer.py` is `0.8`, which enters the `variation` branch. If you want text + vocal-sketch generation without latent variation from the input waveform, pass a value such as `0.7`.

## Inference Notes

- The current inference path uses the same dynamic feature extractor as the VTS voice-conditioning path.
- The prompt audio is converted into a conditioning tensor before sampling.
- The input audio is loaded with `soundfile` as a 2D waveform.
- The dynamic tensor has 12 channels derived from spectral centroid, RMS, and chroma-index features.
- The dynamic tensor is passed to the model as `voice_enc`.
- In all modes, the text prompt is encoded with `google/flan-t5-base`.
- For `variation`, the input audio is also encoded by the vocoder and used as the latent starting point.
- The vocoder implementation lives in `vts/vocos_custom` and is required for both latent encoding and waveform decoding.
- Default sampling uses 64 ODE steps. Use `--steps` to change it.
- If `--duration` is omitted, output duration follows the input audio duration.
- Typical values:
  - `temperature=0.7`
  - `steps=64`
  - `alpha=3.0`

## Output

Generated WAV files are written to the output directory:

```text
outputs/sample_00.wav
outputs/sample_01.wav
...
```

The script also prints a JSON summary containing the selected branch, device, prompt, sample rate, and output file paths.

## 🤝 Acknowledgements

- Thanks to [OptimizerAI](https://www.linkedin.com/company/optimizerai/). I worked on this project while I was at OptimizerAI.

## 📅 Next Plan

- I’m going to adapt this for voice-conditioned music generation.

## License

MIT License. See [LICENSE](./LICENSE).

If you have any other questions, please contact me at daniel@matchharper.com.
