"""Frozen Whisper encoder wrapper (encoder-only; decoder weights unused).

Mirrors model/vision_encoder.py FrozenSigLIP2. The whisper encoder runs at
50 frames/second (mel 100Hz -> conv stride 2), so a 5s clip yields ~250
frames; the caller compresses to the fixed audio-token budget with
`project_vision_features` (linear interpolate then projector).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class FrozenWhisperEncoder:
    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = None
        self.extractor = None

    def load(self) -> "FrozenWhisperEncoder":
        try:
            from transformers import WhisperFeatureExtractor, WhisperModel
        except ImportError as e:
            raise RuntimeError("audio encoder requires transformers; install proj_2/requirements.txt") from e
        self.model = WhisperModel.from_pretrained(self.model_path).to(self.device, dtype=self.dtype).eval()
        self.extractor = WhisperFeatureExtractor.from_pretrained(self.model_path)
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    def preprocess(self, wavs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mel features for a list of float32 mono 16k waveforms.

        The whisper encoder in transformers >= 5.x asserts 3000 mel frames
        (30s), and the feature extractor's DEFAULT behavior pads every clip to
        3000 — measured: no padding args -> (1, 80, 3000) for a 3s clip.
        Explicit padding args misbehaved (returned 18 frames), so the default
        is used. Short clips therefore cost the same as a full 30s pass.
        """
        if self.extractor is None:
            self.load()
        batch = self.extractor(wavs, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        return (batch["input_features"].to(self.device, dtype=self.dtype),
                batch["attention_mask"].to(self.device) if batch.get("attention_mask") is not None else None)

    @torch.no_grad()
    def encode(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """[B, F, d_model] encoder hidden states (whisper-small: d_model=768)."""
        if self.model is None:
            self.load()
        return self.model.encoder(input_features.to(self.device, dtype=self.dtype),
                                  attention_mask=attention_mask).last_hidden_state

    @property
    def hidden_size(self) -> int:
        if self.model is None:
            self.load()
        return int(self.model.config.d_model)
