from __future__ import annotations

import logging
import typing as tp
import warnings

import torch

from .base import Conditioner


class T5Conditioner(Conditioner):
    T5_MODELS = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]

    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
        self,
        output_dim: int,
        t5_model_name: str = "t5-base",
        max_length: int = 128,
        enable_grad: bool = False,
        project_out: bool = False,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)

        from transformers import AutoTokenizer, T5EncoderModel

        self.max_length = max_length
        self.enable_grad = enable_grad
        self.t5_model_name = t5_model_name

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                self.model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad)
            finally:
                logging.disable(previous_level)

    def forward(self, texts: tp.Sequence[str], device: tp.Union[str, torch.device]) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
        use_autocast = device.type == "cuda"
        autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            with torch.set_grad_enabled(self.enable_grad):
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask
