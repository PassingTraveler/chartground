"""Central configuration for the image-first MiniMind multimodal project."""
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ONE = ROOT / "assets" / "project1"
LEGACY_PROJECT_ONE = ROOT.parent / "upload_proj_1"


def project1_root() -> Path:
    """Prefer assets copied into proj_2; keep a read-only legacy fallback."""
    required = PROJECT_ONE / "tokenizer" / "best_mm.json"
    return PROJECT_ONE if required.exists() else LEGACY_PROJECT_ONE


@dataclass
class ModelConfig:
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 8
    intermediate_size: int = 2048
    vocab_size: int = 12014
    max_seq_len: int = 1024
    image_token_len: int = 64
    image_hidden_size: int = 768
    projector_hidden_size: int = 1024
    vision_model_path: str = "model/siglip2-base-patch16-256"
    audio_token_len: int = 256
    # Whisper d_model: base=512, small=768 — set to whatever encoder is used.
    audio_hidden_size: int = 768
    whisper_model: str = "model/whisper-small"


@dataclass
class DataMixConfig:
    general_image_ratio: float = 0.40
    text_math_ratio: float = 0.13
    visual_math_ratio: float = 0.22
    text_replay_ratio: float = 0.15
    audio_math_ratio: float = 0.10

    def validate(self) -> None:
        total = (self.general_image_ratio + self.text_math_ratio + self.visual_math_ratio
                 + self.text_replay_ratio + self.audio_math_ratio)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"data mix must sum to 1.0, got {total}")


def default_paths() -> dict[str, Path]:
    project_one = project1_root()
    return {
        "data": ROOT / "data",
        "generated": ROOT / "data" / "generated",
        "clean": ROOT / "data" / "clean",
        "out": ROOT / "out",
        "project1": project_one,
        "tokenizer": project_one / "tokenizer" / "best_mm.json",
        "pretrain": project_one / "out" / "pretrain_02b" / "pretrain_h1024_l24.pth",
        "math_sft": project_one / "out" / "agent_sft_v3" / "sft_h1024_l24.pth",
    }
