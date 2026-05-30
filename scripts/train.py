from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from voice_text_sfx.config import get_default_autoencoder_config, get_default_backbone_config, get_default_training_config
from voice_text_sfx.data import VoiceTextSFXDataset
from voice_text_sfx.pipeline import build_components, load_module_checkpoint
from voice_text_sfx.training import calculate_targets, get_alphas_sigmas


def parse_args():
    defaults = get_default_training_config()
    parser = argparse.ArgumentParser(description="Train voice+text conditioned latent diffusion for sound effects")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest")
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=defaults["num_epochs"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--segment-seconds", type=float, default=defaults["segment_seconds"])
    parser.add_argument("--cfg-dropout-prob", type=float, default=defaults["cfg_dropout_prob"])
    parser.add_argument("--voice-conditioning-prob", type=float, default=defaults["voice_conditioning_prob"])
    parser.add_argument("--save-every", type=int, default=defaults["save_every"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(manifest: str, batch_size: int, num_workers: int, segment_seconds: float, sample_rate: int, shuffle: bool):
    dataset = VoiceTextSFXDataset(
        manifest_path=manifest,
        sample_rate=sample_rate,
        channels=2,
        segment_seconds=segment_seconds,
        random_crop=shuffle,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def compute_batch_loss(
    batch,
    *,
    model,
    autoencoder,
    text_conditioner,
    voice_extractor,
    device: torch.device,
    cfg_dropout_prob: float,
    voice_conditioning_prob: float,
):
    audio = batch["audio"].to(device)
    conditioning_audio = batch["conditioning_audio"].to(device)
    captions = list(batch["caption"])
    seconds_start = batch["seconds_start"].to(device)
    seconds_total = batch["seconds_total"].to(device)

    with torch.no_grad():
        z_0 = autoencoder.encode_audio(audio)
        input_ids, attention_mask = text_conditioner(captions, device=device)
        voice_cond = voice_extractor(conditioning_audio)

    if random.random() > voice_conditioning_prob:
        voice_cond = None

    t = torch.sigmoid(torch.randn(z_0.shape[0], device=device))
    alphas, sigmas = get_alphas_sigmas(t)
    alphas = alphas[:, None, None]
    sigmas = sigmas[:, None, None]
    noise = torch.randn_like(z_0)
    noised_inputs = z_0 * alphas + noise * sigmas
    targets = calculate_targets(noise, z_0, alphas, sigmas, "v")

    use_autocast = device.type == "cuda"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
        output = model(
            x=noised_inputs,
            t=t,
            mask=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            seconds_start=seconds_start,
            seconds_total=seconds_total,
            voice_cond=voice_cond,
            cfg_dropout_prob=cfg_dropout_prob,
            cfg_scale=None,
        )
        loss = F.mse_loss(output, targets)
    return loss


def evaluate(loader, *, model, autoencoder, text_conditioner, voice_extractor, device: torch.device):
    if loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss = compute_batch_loss(
                batch,
                model=model,
                autoencoder=autoencoder,
                text_conditioner=text_conditioner,
                voice_extractor=voice_extractor,
                device=device,
                cfg_dropout_prob=0.0,
                voice_conditioning_prob=1.0,
            )
            total += loss.item()
            count += 1
    return total / max(count, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    autoencoder_config = get_default_autoencoder_config()
    sample_rate = autoencoder_config["sample_rate"]

    model, autoencoder, text_conditioner, voice_extractor = build_components(device=device)
    load_module_checkpoint(
        autoencoder,
        args.ae_ckpt,
        preferred_keys=("autoencoder", "model", "state_dict"),
        strip_prefixes=("module.", "model."),
        strict=False,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_module_checkpoint(
            model,
            args.resume,
            preferred_keys=("model", "state_dict"),
            strip_prefixes=("module.",),
            strict=True,
        )
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    train_loader = make_loader(
        args.train_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_seconds=args.segment_seconds,
        sample_rate=sample_rate,
        shuffle=True,
    )
    valid_loader = None
    if args.valid_manifest:
        valid_loader = make_loader(
            args.valid_manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            segment_seconds=args.segment_seconds,
            sample_rate=sample_rate,
            shuffle=False,
        )

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_batch_loss(
                batch,
                model=model,
                autoencoder=autoencoder,
                text_conditioner=text_conditioner,
                voice_extractor=voice_extractor,
                device=device,
                cfg_dropout_prob=args.cfg_dropout_prob,
                voice_conditioning_prob=args.voice_conditioning_prob,
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

        train_loss = running_loss / max(len(train_loader), 1)
        valid_loss = evaluate(
            valid_loader,
            model=model,
            autoencoder=autoencoder,
            text_conditioner=text_conditioner,
            voice_extractor=voice_extractor,
            device=device,
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "backbone_config": get_default_backbone_config(),
            "autoencoder_config": autoencoder_config,
        }
        torch.save(checkpoint, output_dir / "model_latest.pt")
        if epoch % args.save_every == 0:
            torch.save(checkpoint, output_dir / f"model_epoch_{epoch:04d}.pt")

        message = f"epoch={epoch} train_loss={train_loss:.6f}"
        if valid_loss is not None:
            message += f" valid_loss={valid_loss:.6f}"
        print(message, flush=True)


if __name__ == "__main__":
    main()
