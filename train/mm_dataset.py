from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover
    Tokenizer = Any

from model.mm_template import MMTemplate


def token_id(tokenizer: Tokenizer, token: str, default: int) -> int:
    value = tokenizer.token_to_id(token)
    return default if value is None else int(value)


class MMDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: Tokenizer, max_seq_len: int = 1024,
                 image_processor=None, image_token_len: int = 64, audio_token_len: int = 256):
        self.path = Path(path)
        self.rows = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.image_processor = image_processor
        self.image_token_len = image_token_len
        self.audio_token_len = audio_token_len
        self.template = MMTemplate()
        self.image_token_id = token_id(tokenizer, self.template.image_token, 12003)
        self.video_token_id = token_id(tokenizer, self.template.video_token, 12004)
        self.audio_start_id = token_id(tokenizer, self.template.audio_start, 12005)
        self.audio_pad_id = token_id(tokenizer, self.template.audio_pad, 12007)
        self.pad_id = token_id(tokenizer, "<|endoftext|>", 0)
        # Decoded pixel tensors cache (image paths are stable across epochs;
        # re-decoding tens of thousands of JPEGs every epoch is pure waste).
        self._pixel_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._pixel_cache_limit = 2048

    def __len__(self):
        return len(self.rows)

    def _asset_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.path.parent / path

    def _encode(self, row: dict) -> dict:
        modality = row.get("modality", "text")
        prompt = row.get("prompt") or row.get("messages", [{}])[0].get("content", "")
        answer = row.get("answer") or row.get("messages", [{}, {"content": ""}])[-1].get("content", "")
        if modality == "video" or row.get("video"):
            raise NotImplementedError(
                f"video modality is not supported yet (example {row.get('example_id')}); "
                "remove video rows from the manifest until the single-image milestone is done"
            )
        if modality == "audio" or row.get("audio"):
            include_q = bool((row.get("metadata") or {}).get("include_question", False))
            prompt_text = self.template.prompt_audio(prompt, 1, tokens_per_image=self.image_token_len,
                                                     n_audio_tokens=self.audio_token_len, include_question=include_q)
        elif modality == "image" or row.get("image"):
            prompt_text = self.template.prompt(prompt, "image", 1, tokens_per_image=self.image_token_len)
        else:
            prompt_text = self.template.prompt(prompt, "text")
        target = self.template.target(answer)
        # Encode prompt and target separately, then concatenate: a single
        # encode(prompt + target) can merge boundary tokens and shift labels.
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False).ids
        target_ids = self.tokenizer.encode(target, add_special_tokens=False).ids
        if len(prompt_ids) >= self.max_seq_len:
            # A prompt alone fills the budget; trim it so the sample still
            # produces labels instead of being silently all -100.
            prompt_ids = prompt_ids[: self.max_seq_len - 1]
        target_ids = target_ids[: max(0, self.max_seq_len - len(prompt_ids))]
        full_ids = prompt_ids + target_ids
        pad = self.max_seq_len - len(full_ids)
        input_ids = full_ids + [self.pad_id] * pad
        labels = [-100] * len(prompt_ids) + target_ids + [-100] * (self.max_seq_len - len(prompt_ids) - len(target_ids))
        image_path = row.get("image")
        pixel_values = None
        if image_path and self.image_processor is not None:
            asset = self._asset_path(image_path)
            if not asset.is_file():
                raise FileNotFoundError(f"missing image for example {row.get('example_id')}: {asset}")
            key = str(asset)
            cached = self._pixel_cache.get(key)
            if cached is not None:
                pixel_values = cached
            else:
                with Image.open(asset).convert("RGB") as image:
                    pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
                self._pixel_cache[key] = pixel_values
                if len(self._pixel_cache) > self._pixel_cache_limit:
                    self._pixel_cache.popitem(last=False)
        audio_path = row.get("audio_features")
        audio_features = None
        if audio_path:
            asset = self._asset_path(audio_path)
            if not asset.is_file():
                raise FileNotFoundError(f"missing audio features for example {row.get('example_id')}: {asset}")
            # Precomputed whisper features (data/precompute_audio_features.py);
            # npy loads are ~1ms, cheap enough to skip caching.
            audio_features = torch.from_numpy(np.ascontiguousarray(np.load(asset), dtype=np.float32))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
            "audio_features": audio_features,
            "row": row,
        }

    def __getitem__(self, index: int) -> dict:
        return self._encode(self.rows[index])


def collate_mm(batch: list[dict]) -> dict:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    images = [x["pixel_values"] for x in batch]
    if any(x is not None for x in images):
        if not all(x is not None for x in images):
            # Keep tensor shapes aligned; text-only examples receive a zero image
            # and are protected by the absence of image marker tokens.
            ref = next(x for x in images if x is not None)
            images = [x if x is not None else torch.zeros_like(ref) for x in images]
        pixel_values = torch.stack(images)
    else:
        pixel_values = None
    audios = [x["audio_features"] for x in batch]
    if any(x is not None for x in audios):
        if not all(x is not None for x in audios):
            ref = next(x for x in audios if x is not None)
            audios = [x if x is not None else torch.zeros_like(ref) for x in audios]
        audio_features = torch.stack(audios)
    else:
        audio_features = None
    return {"input_ids": input_ids, "labels": labels, "pixel_values": pixel_values,
            "audio_features": audio_features, "rows": [x["row"] for x in batch]}


def project_vision_features(raw: torch.Tensor, projector, target_tokens: int = 64) -> torch.Tensor:
    """Normalize a vision encoder's token count to the marker budget."""
    if raw.size(1) != target_tokens:
        raw = torch.nn.functional.interpolate(raw.transpose(1, 2), size=target_tokens, mode="linear", align_corners=False).transpose(1, 2)
    return projector(raw)


def encode_prompt_row(row: dict, tokenizer: Tokenizer, max_seq_len: int = 1024,
                      image_token_len: int = 64, audio_token_len: int = 256) -> tuple[torch.Tensor, int]:
    """Encode only the user prompt for generation/RL."""
    template = MMTemplate()
    modality = row.get("modality", "text")
    question = row.get("prompt") or row.get("messages", [{}])[0].get("content", "")
    if modality == "video" or row.get("video"):
        raise NotImplementedError(
            f"video modality is not supported yet (example {row.get('example_id')})"
        )
    if modality == "audio" or row.get("audio"):
        # TODO(audio-RL): the caller must also load and inject the precomputed
        # audio features; only the text template is assembled here.
        include_q = bool((row.get("metadata") or {}).get("include_question", False))
        text = template.prompt_audio(question, 1, tokens_per_image=image_token_len,
                                     n_audio_tokens=audio_token_len, include_question=include_q)
    elif modality == "image" or row.get("image"):
        text = template.prompt(question, "image", 1, tokens_per_image=image_token_len)
    else:
        text = template.prompt(question, "text")
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:max_seq_len]
    return torch.tensor([ids], dtype=torch.long), len(ids)
