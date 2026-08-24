from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Example:
    example_id: str
    modality: str
    prompt: str
    answer: str
    image: str | None = None
    video: str | None = None
    audio: str | None = None
    audio_features: str | None = None
    split: str = "train"
    source: str = "unknown"
    task: str = "vqa"
    visual_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["messages"] = [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.answer},
        ]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in allowed})


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any] | Example], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") if append else path.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = row.to_dict() if isinstance(row, Example) else row
            obj = dict(obj)
            # Keep manifests portable across Windows development and Linux
            # training hosts. Consumers resolve relative assets against the
            # manifest directory.
            for field in ("image", "video", "audio", "audio_features"):
                value = obj.get(field)
                if value:
                    asset = Path(value)
                    if asset.is_absolute():
                        try:
                            obj[field] = Path(os.path.relpath(asset, path.parent)).as_posix()
                        except ValueError:
                            # Different drives (Windows) cannot be relativized;
                            # retaining the absolute path is safer than corrupting it.
                            obj[field] = str(asset)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {e}") from e
    return rows
