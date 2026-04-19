from __future__ import annotations

import math

import torch


def get_alphas_sigmas(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def calculate_targets(noise: torch.Tensor, z_0: torch.Tensor, alphas: torch.Tensor, sigmas: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == "v":
        return noise * alphas - z_0 * sigmas
    if objective == "rectified_flow":
        return noise - z_0
    raise ValueError(f"Unknown objective: {objective}")
