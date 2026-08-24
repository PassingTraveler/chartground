"""Frozen SigLIP2 wrapper with optional lazy loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class VisionBatch:
    pixel_values: torch.Tensor


class FrozenSigLIP2:
    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load(self) -> "FrozenSigLIP2":
        try:
            from transformers import SiglipImageProcessor, SiglipVisionModel
        except ImportError as e:
            raise RuntimeError("vision encoder requires transformers; install proj_2/requirements.txt") from e
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"SigLIP2 checkpoint not found: {self.model_path}")
        self.model = SiglipVisionModel.from_pretrained(self.model_path).to(self.device, dtype=self.dtype).eval()
        self.processor = SiglipImageProcessor.from_pretrained(self.model_path)
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    def preprocess(self, images) -> torch.Tensor:
        if self.processor is None:
            self.load()
        batch = self.processor(images=images, return_tensors="pt")
        return batch["pixel_values"].to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            self.load()
        return self.model(pixel_values=pixel_values.to(self.device, dtype=self.dtype)).last_hidden_state

    @property
    def hidden_size(self) -> int:
        if self.model is None:
            self.load()
        return int(self.model.config.hidden_size)

