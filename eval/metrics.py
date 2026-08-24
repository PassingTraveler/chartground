from __future__ import annotations

import math
import random
import re
from typing import Iterable


def normalize_answer(text: str) -> str:
    text = text.strip().replace(",", "").replace("，", "")
    match = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return match[-1] if match else text


def exact_acc(preds: Iterable[str], golds: Iterable[str]) -> float:
    pairs = [(normalize_answer(p), normalize_answer(g)) for p, g in zip(preds, golds)]
    return sum(p == g for p, g in pairs) / max(len(pairs), 1)


def bootstrap_ci(values: list[float], seed: int = 0, rounds: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def paired_gain(with_image: list[str], without_image: list[str], golds: list[str]) -> dict:
    a = [float(normalize_answer(p) == normalize_answer(g)) for p, g in zip(with_image, golds)]
    b = [float(normalize_answer(p) == normalize_answer(g)) for p, g in zip(without_image, golds)]
    deltas = [x - y for x, y in zip(a, b)]
    ci = bootstrap_ci(deltas)
    return {"with_image_acc": sum(a) / max(len(a), 1), "without_image_acc": sum(b) / max(len(b), 1), "delta_vis": sum(deltas) / max(len(deltas), 1), "ci_low": ci[0], "ci_high": ci[1]}

