from __future__ import annotations

import copy

DEFAULT_AUTOENCODER_CONFIG = {
    "encoder": {
        "type": "oobleck",
        "requires_grad": False,
        "config": {
            "in_channels": 2,
            "channels": 128,
            "c_mults": [1, 2, 4, 8, 16],
            "strides": [2, 4, 4, 8, 8],
            "latent_dim": 128,
            "use_snake": True,
        },
    },
    "decoder": {
        "type": "oobleck",
        "requires_grad": False,
        "config": {
            "out_channels": 2,
            "channels": 128,
            "c_mults": [1, 2, 4, 8, 16],
            "strides": [2, 4, 4, 8, 8],
            "latent_dim": 64,
            "use_snake": True,
            "final_tanh": False,
        },
    },
    "bottleneck": {"type": "vae"},
    "latent_dim": 64,
    "downsampling_ratio": 2048,
    "io_channels": 2,
    "sample_rate": 44100,
}

DEFAULT_DIFFUSION_BACKBONE_CONFIG = {
    "text_conditioner_config": {
        "output_dim": 768,
        "t5_model_name": "t5-base",
        "max_length": 128,
    },
    "timing_config": {
        "output_dim": 768,
        "min_val": 0,
        "max_val": 512,
    },
    "model": {
        "cond_embed_dim": 768,
    },
    "cond_token_dim": 768,
    "project_cond_tokens": False,
    "global_cond_dim": 1536,
    "cross_attn_cond_keys": ["prompt", "seconds_start", "seconds_total"],
    "global_cond_keys": ["seconds_start", "seconds_total"],
}

DEFAULT_TRANSFORMER_INIT = {
    "d_model": 1536,
    "depth": 24,
    "num_heads": 24,
    "input_concat_dim": 0,
    "global_cond_type": "prepend",
    "latent_channels": 64,
    "voice_dim": 16,
    "voice_cond_type": "prepend",
}

DEFAULT_TRAINING_CONFIG = {
    "sample_rate": 44100,
    "segment_seconds": 3.0,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 10,
    "num_workers": 4,
    "cfg_dropout_prob": 0.1,
    "voice_conditioning_prob": 0.5,
    "diffusion_objective": "v",
    "save_every": 1,
}


def get_default_autoencoder_config() -> dict:
    return copy.deepcopy(DEFAULT_AUTOENCODER_CONFIG)


def get_default_backbone_config() -> dict:
    return copy.deepcopy(DEFAULT_DIFFUSION_BACKBONE_CONFIG)


def get_default_transformer_init() -> dict:
    return copy.deepcopy(DEFAULT_TRANSFORMER_INIT)


def get_default_training_config() -> dict:
    return copy.deepcopy(DEFAULT_TRAINING_CONFIG)

