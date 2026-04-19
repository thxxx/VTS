from __future__ import annotations

import argparse

from voice_text_sfx.pipeline import VoiceTextSFXPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run voice+text conditioned sound effect inference")
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = VoiceTextSFXPipeline.from_checkpoints(
        model_checkpoint=args.model_ckpt,
        autoencoder_checkpoint=args.ae_ckpt,
        device=args.device,
    )
    audio = pipeline.generate_from_audio_file(
        prompt_audio_path=args.prompt_audio,
        text=args.text,
        duration=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
    )
    pipeline.save_audio(audio, args.output, sample_rate=pipeline.autoencoder.sample_rate)


if __name__ == "__main__":
    main()
