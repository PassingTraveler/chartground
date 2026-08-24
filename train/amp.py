"""Small, explicit mixed-precision helpers shared by training entry points."""
from __future__ import annotations

from contextlib import nullcontext

import torch


def configure_precision(device: torch.device, name: str):
    """Return (autocast_dtype, optional GradScaler) for a training device."""
    if name not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported precision: {name}")
    if device.type != "cuda" or name == "fp32":
        return None, None
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested, but this CUDA/PyTorch build does not support it")
        return torch.bfloat16, None
    return torch.float16, torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)
