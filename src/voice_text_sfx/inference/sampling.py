from __future__ import annotations

import k_diffusion as K
import torch
import torchsde
from tqdm.auto import trange


class BatchedBrownianTree:
    def __init__(self, x: torch.Tensor, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    def __init__(self, x: torch.Tensor, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_dpmpp_3m_sde(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    cross_attention_inputs: torch.Tensor,
    cross_attention_masks: torch.Tensor,
    seconds_start: torch.Tensor,
    seconds_total: torch.Tensor,
    cfg_scale: float = 6.0,
    callback=None,
    disable: bool | None = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    voice_cond: torch.Tensor | None = None,
):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(
            x,
            sigmas[i] * s_in,
            mask=None,
            input_ids=cross_attention_inputs,
            attention_mask=cross_attention_masks,
            seconds_start=seconds_start,
            seconds_total=seconds_total,
            cfg_dropout_prob=0.0,
            cfg_scale=cfg_scale,
            voice_cond=voice_cond,
        )

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1

    return x


@torch.no_grad()
def generate_audio(
    model,
    ae,
    text_conditioner,
    text: str,
    voice_cond: torch.Tensor,
    *,
    steps: int = 100,
    cfg_scale: float = 6.0,
    duration: float = 3.0,
    sample_rate: int = 44100,
    batch_size: int = 1,
    device: str | torch.device = "cuda",
    disable: bool = False,
):
    device = torch.device(device)
    latent_channels = ae.latent_dim
    latent_length = int(sample_rate * duration // ae.downsampling_ratio)
    noise = torch.randn([batch_size, latent_channels, latent_length], device=device)

    input_ids, attention_mask = text_conditioner([text], device=device)
    seconds_start = torch.zeros(batch_size, device=device)
    seconds_total = torch.full((batch_size,), float(duration), device=device)
    cross_attention_inputs, cross_attention_masks, _ = model.get_context(input_ids, attention_mask, seconds_start, seconds_total)

    sigmas = K.sampling.get_sigmas_polyexponential(steps, 0.3, 500, rho=1.0, device=device)
    denoiser = K.external.VDenoiser(model)

    x = noise * sigmas[0]
    use_autocast = device.type == "cuda"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
        out = sample_dpmpp_3m_sde(
            denoiser,
            x,
            sigmas,
            cross_attention_inputs,
            cross_attention_masks,
            seconds_start,
            seconds_total,
            disable=disable,
            cfg_scale=cfg_scale,
            callback=None,
            voice_cond=voice_cond,
        )

    return ae.decode(out)
