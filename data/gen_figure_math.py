"""Generate visual-math images with deterministic, executable ground truth.

The first version intentionally uses Pillow only. This keeps data generation
available on a fresh machine before matplotlib/torch are installed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from .schema import Example, write_jsonl
except ImportError:  # direct script execution
    from schema import Example, write_jsonl


W, H = 256, 256


def font(size: int = 12):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), "white")
    return im, ImageDraw.Draw(im)


def draw_axes(draw: ImageDraw.ImageDraw, readable: bool = False) -> int:
    """Draw x/y axes; return the y-axis x position so callers can lay out
    their marks to its right. `readable` is the legibility variant used for
    the discriminative experiment: bigger tick labels, longer ticks and
    horizontal grid lines so bar-top / point readings can be aligned to the
    y scale visually."""
    if readable:
        ax_x = 40
        draw.line((ax_x, 20, ax_x, 220), fill="black", width=2)
        draw.line((ax_x, 220, 238, 220), fill="black", width=2)
        for value in range(0, 101, 10):
            y = 220 - int(value * 1.8)
            major = value % 20 == 0
            draw.line((ax_x - (6 if major else 4), y, ax_x, y), fill="black", width=1)
            draw.line((ax_x, y, 238, y), fill="#DDDDDD", width=1)
            if major:
                draw.text((6, y - 9), str(value), fill="black", font=font(14))
        return ax_x
    draw.line((28, 20, 28, 220), fill="black", width=2)
    draw.line((28, 220, 238, 220), fill="black", width=2)
    for value in range(0, 101, 20):
        y = 220 - int(value * 1.8)
        draw.line((24, y, 28, y), fill="black", width=1)
        draw.text((1, y - 5), str(value), fill="black", font=font(8))
    return 28


def make_bar(rng: random.Random, readable: bool = False, coarse: bool = False) -> tuple[Image.Image, str, str, dict]:
    # 2026-08-21 题型重设计：精确读数（差值）超出 0.25B+SigLIP256 能力边界
    # （四组判别实验证实），改为找最值/比较——与 grid 检测同级别的粗粒度
    # 视觉任务（柱顶相对高低），不需要与 y 轴刻度对齐。柱值两两不同保证
    # 最值唯一。
    # coarse: quantize to multiples of 10 (18px apart at 1.8px/unit) so a
    # 16px SigLIP patch can separate adjacent values — discriminative probe
    # for whether patch-16 encoding is the resolution bottleneck.
    pool = range(20, 91, 10) if coarse else range(20, 91)
    values = rng.sample(pool, 4)
    im, draw = canvas()
    ax_x = draw_axes(draw, readable)
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for i, (v, label, color) in enumerate(zip(values, labels, colors)):
        x = (ax_x + 16 if readable else 48) + i * (42 if readable else 45)
        y = 220 - int(v * 1.8)
        draw.rectangle((x, y, x + (24 if readable else 26), 220), fill=color, outline="black")
        draw.text((x + 4, 224), label, fill="black", font=font(12 if readable else 10))
    ask_max = rng.random() < 0.5
    order = sorted(range(4), key=lambda i: values[i])  # 柱高从低到高的季度下标
    # CoT：相对高度比较链（从低到高相邻比较，不出现数值——数值是读不出的，
    # 相对高低是可以看的）。尾部"第X季度"含答案数字，normalize/smooth_reward
    # 取最后一个数字天然兼容。
    chain = "；".join(f"第{order[k] + 1}季度的柱比第{order[k + 1] + 1}季度的柱低" for k in range(3))
    if ask_max:
        q = "图中四个季度的销量分别由四根柱表示。销量最高的是第几个季度？只输出编号（1到4）。"
        ans_i = order[-1]
        cot = f"比较柱高：{chain}。第{ans_i + 1}季度的柱最高，所以答案是{ans_i + 1}。"
        prog = "argmax"
    else:
        q = "图中四个季度的销量分别由四根柱表示。销量最低的是第几个季度？只输出编号（1到4）。"
        ans_i = order[0]
        cot = f"比较柱高：{chain}。第{ans_i + 1}季度的柱最低，所以答案是{ans_i + 1}。"
        prog = "argmin"
    return im, q, cot, {"type": "bar", "values": values, "program": prog, "cot": "v2"}


def make_line(rng: random.Random, readable: bool = False, coarse: bool = False) -> tuple[Image.Image, str, str, dict]:
    # 同 make_bar：题型改为找最值/比较（数据点相对高低），不再要求精确读数。
    pool = range(10, 81, 10) if coarse else range(10, 81)
    values = rng.sample(pool, 4)
    im, draw = canvas()
    ax_x = draw_axes(draw, readable)
    pts = []
    r = 5 if readable else 4
    for i, v in enumerate(values):
        x = (ax_x + 16 if readable else 48) + i * (48 if readable else 52)
        y = 220 - int(v * 1.8)
        pts.append((x, y))
        draw.ellipse((x - r, y - r, x + r, y + r), fill="#C00000")
        draw.text((x - 8, 224), f"M{i + 1}", fill="black", font=font(12 if readable else 10))
    draw.line(pts, fill="#C00000", width=3)
    ask_max = rng.random() < 0.5
    order = sorted(range(4), key=lambda i: values[i])  # 数值从低到高的月份下标
    chain = "；".join(f"第{order[k] + 1}个月的点比第{order[k + 1] + 1}个月的点低" for k in range(3))
    if ask_max:
        q = "折线图给出了四个月的数值。数值最高的是第几个月？只输出编号（1到4）。"
        ans_i = order[-1]
        cot = f"比较各点高低：{chain}。第{ans_i + 1}个月的点最高，所以答案是{ans_i + 1}。"
        prog = "argmax"
    else:
        q = "折线图给出了四个月的数值。数值最低的是第几个月？只输出编号（1到4）。"
        ans_i = order[0]
        cot = f"比较各点高低：{chain}。第{ans_i + 1}个月的点最低，所以答案是{ans_i + 1}。"
        prog = "argmin"
    return im, q, cot, {"type": "line", "values": values, "program": prog, "cot": "v2"}


def make_grid(rng: random.Random, readable: bool = False, coarse: bool = False) -> tuple[Image.Image, str, str, dict]:
    # 2026-08-20 泄露教训：2×2 只有 16 种对角线配置、3×3 只有 512 种——
    # eval 与 train 必然撞图，模型可以纯记忆作答（泄露集 acc 0.836 vs
    # 干净集 0.050）。改用 4/5/6：配置空间 65536/33M/68B，配合
    # generate() 的 train 哈希去重后，撞图概率可忽略。
    n = rng.choice([4, 5, 6])
    im, draw = canvas()
    left, top, size = 32, 32, 180
    step = size // n
    for i in range(n + 1):
        draw.line((left + i * step, top, left + i * step, top + size), fill="black", width=2)
        draw.line((left, top + i * step, left + size, top + i * step), fill="black", width=2)
    # Each diagonal splits its cell into two basic triangles; cells without a
    # diagonal contribute none. Ground truth is exactly 2 * number of
    # diagonal cells (verified against an independent pixel/geometry count).
    diag_cells = []
    for r in range(n):
        for c in range(n):
            if (r + c + rng.randint(0, 1)) % 2 == 0:
                x, y = left + c * step, top + r * step
                draw.line((x, y, x + step, y + step), fill="#4472C4", width=2)
                diag_cells.append((r, c))
    triangles = 2 * len(diag_cells)
    q = f"这是一个 {n}×{n} 网格图。请数出图中蓝色对角线参与形成的基本三角形数量，只输出数值。"
    # CoT: per-row diagonal counts -> per-row triangle counts -> sum. The row
    # sweep is exactly the program a counter needs; teaching it as text is
    # the whole point of the CoT variant (0.25B cannot memorize 33M configs).
    row_ordinals = "一二三四五六七八"
    per_row = [0] * n
    for r, _ in diag_cells:
        per_row[r] += 1
    parts = [f"第{row_ordinals[r]}行有{c}格带对角线，形成{2 * c}个三角形" for r, c in enumerate(per_row)]
    tri_nums = [2 * c for c in per_row]
    cot = (("；".join(parts)) + f"。总数为{'加'.join(map(str, tri_nums))}等于{triangles}，"
           f"所以答案是{triangles}。")
    return im, q, cot, {"type": "grid", "n": n, "diagonal_cells": diag_cells,
                        "program": f"2*{len(diag_cells)}", "cot": "v1"}


def make_bar_paircmp(rng: random.Random, readable: bool = False, coarse: bool = False) -> tuple[Image.Image, str, str, dict]:
    # v5 新题型：两柱定位比较（只比较两个点名柱子的相对高低，比 argmax 的
    # 全序链更浅）。答案空间 4，尾数字兼容；无数值读数。
    pool = range(20, 91, 10) if coarse else range(20, 91)
    values = rng.sample(pool, 4)
    im, draw = canvas()
    ax_x = draw_axes(draw, readable)
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for i, (v, label, color) in enumerate(zip(values, labels, colors)):
        x = (ax_x + 16 if readable else 48) + i * (42 if readable else 45)
        y = 220 - int(v * 1.8)
        draw.rectangle((x, y, x + (24 if readable else 26), 220), fill=color, outline="black")
        draw.text((x + 4, 224), label, fill="black", font=font(12 if readable else 10))
    i, j = rng.sample(range(4), 2)
    lo, hi = (i, j) if values[i] < values[j] else (j, i)
    q = f"图中四个季度的销量分别由四根柱表示。第{lo + 1}季度和第{hi + 1}季度的销量谁更高？只输出编号（1到4）。"
    cot = (f"比较第{lo + 1}季度和第{hi + 1}季度的柱高：第{lo + 1}季度的柱比第{hi + 1}季度的柱低。"
           f"第{hi + 1}季度的柱更高，所以答案是{hi + 1}。")
    return im, q, cot, {"type": "bar_paircmp", "values": values, "program": f"paircmp({lo},{hi})", "cot": "v5"}


def make_bar_groupsum(rng: random.Random, readable: bool = False, coarse: bool = False) -> tuple[Image.Image, str, str, dict]:
    # v5 新题型：两组季度销量总和比较（4 选 1）。选项构造保证唯一正解：
    # 最高的两根柱组成正解组，其余选项各替换一个成员后严格变小，所以
    # CoT 可以用两次两两柱高比较证明（无数值读数）。
    pool = range(20, 91, 10) if coarse else range(20, 91)
    values = rng.sample(pool, 4)
    im, draw = canvas()
    ax_x = draw_axes(draw, readable)
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for i, (v, label, color) in enumerate(zip(values, labels, colors)):
        x = (ax_x + 16 if readable else 48) + i * (42 if readable else 45)
        y = 220 - int(v * 1.8)
        draw.rectangle((x, y, x + (24 if readable else 26), 220), fill=color, outline="black")
        draw.text((x + 4, 224), label, fill="black", font=font(12 if readable else 10))
    order = sorted(range(4), key=lambda i: values[i])  # 柱高从低到高的季度下标
    a1, a2 = order[2], order[3]  # 两根最高的柱
    b1, b2 = order[0], order[1]
    options = [(a1, a2), (a2, b2), (a1, b1), (b1, b2)]
    rng.shuffle(options)
    correct = options.index((a1, a2)) + 1
    opt_text = "；".join(f"{k}：第{u + 1}季度和第{v + 1}季度" for k, (u, v) in enumerate(options, start=1))
    q = (f"图中四个季度的销量分别由四根柱表示。以下哪一组两个季度的销量总和最大？"
         f"{opt_text}。只输出组编号（1到4）。")
    cot = (f"第{a1 + 1}季度的柱比第{b1 + 1}季度的柱高，第{a2 + 1}季度的柱比第{b2 + 1}季度的柱高；"
           f"第{a1 + 1}季度和第{a2 + 1}季度是两根最高的柱，它们的总和最大。"
           f"对应第{correct}组，所以答案是{correct}。")
    return im, q, cot, {"type": "bar_groupsum", "values": values, "program": f"groupsum({a1},{a2})", "cot": "v5"}


def make_line_turns(rng: random.Random, readable: bool = False, coarse: bool = False) -> tuple[Image.Image, str, str, dict]:
    # v5 新题型：数相邻下降段数（0-3）。三个相邻符号判断 + 计数，无数值读数。
    pool = range(10, 81, 10) if coarse else range(10, 81)
    values = rng.sample(pool, 4)
    im, draw = canvas()
    ax_x = draw_axes(draw, readable)
    pts = []
    r = 5 if readable else 4
    for i, v in enumerate(values):
        x = (ax_x + 16 if readable else 48) + i * (48 if readable else 52)
        y = 220 - int(v * 1.8)
        pts.append((x, y))
        draw.ellipse((x - r, y - r, x + r, y + r), fill="#C00000")
        draw.text((x - 8, 224), f"M{i + 1}", fill="black", font=font(12 if readable else 10))
    draw.line(pts, fill="#C00000", width=3)
    diffs = [values[k + 1] - values[k] for k in range(3)]
    parts = [f"第{k + 1}个月到第{k + 2}个月{'上升' if d > 0 else '下降'}" for k, d in enumerate(diffs)]
    n_fall = sum(1 for d in diffs if d < 0)
    q = "折线图给出了四个月的数值。从第1个月到第4个月，相邻两个月之间数值下降的有几对？只输出 0 到 3 的整数。"
    cot = "；".join(parts) + f"。下降的相邻对一共有{n_fall}对，所以答案是{n_fall}。"
    return im, q, cot, {"type": "line_turns", "values": values, "program": f"falls={n_fall}", "cot": "v5"}


def make_bar_gap(rng: random.Random, readable: bool = False, coarse: bool = False) -> tuple[Image.Image, str, str, dict]:
    # 间距受控探针：柱值两两间距 >= 10 units（>=18px），与 make_bar 同题型
    # 同渲染。gap 分层分析在非受控集上未见单调关系（bar 大间距行仍 0.250）
    # ——受控集上再确认"间距"是否真的不是瓶颈。
    pool = range(20, 91)
    while True:
        values = rng.sample(pool, 4)
        if min(abs(values[i] - values[j]) for i in range(4) for j in range(i + 1, 4)) >= 10:
            break
    im, draw = canvas()
    ax_x = draw_axes(draw, readable)
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for i, (v, label, color) in enumerate(zip(values, labels, colors)):
        x = (ax_x + 16 if readable else 48) + i * (42 if readable else 45)
        y = 220 - int(v * 1.8)
        draw.rectangle((x, y, x + (24 if readable else 26), 220), fill=color, outline="black")
        draw.text((x + 4, 224), label, fill="black", font=font(12 if readable else 10))
    ask_max = rng.random() < 0.5
    order = sorted(range(4), key=lambda i: values[i])
    chain = "；".join(f"第{order[k] + 1}季度的柱比第{order[k + 1] + 1}季度的柱低" for k in range(3))
    if ask_max:
        q = "图中四个季度的销量分别由四根柱表示。销量最高的是第几个季度？只输出编号（1到4）。"
        ans_i = order[-1]
        cot = f"比较柱高：{chain}。第{ans_i + 1}季度的柱最高，所以答案是{ans_i + 1}。"
        prog = "argmax"
    else:
        q = "图中四个季度的销量分别由四根柱表示。销量最低的是第几个季度？只输出编号（1到4）。"
        ans_i = order[0]
        cot = f"比较柱高：{chain}。第{ans_i + 1}季度的柱最低，所以答案是{ans_i + 1}。"
        prog = "argmin"
    return im, q, cot, {"type": "bar_gap", "values": values, "program": prog, "cot": "v2", "gap": ">=10"}


MAKERS = {"bar": make_bar, "line": make_line, "grid": make_grid,
          "bar_paircmp": make_bar_paircmp, "bar_groupsum": make_bar_groupsum,
          "line_turns": make_line_turns, "bar_gap": make_bar_gap}


def _png_hash(image: Image.Image) -> str:
    """Pixel-content hash: independent of PNG encoder/version, so it compares
    the same logical chart even when the train images were written by a
    different PIL build."""
    return hashlib.md5(image.convert("RGB").tobytes()).hexdigest()


def generate(count: int, out_dir: Path, manifest: Path, seed: int, split: str,
             dedup_against: Path | None = None, max_retries: int = 200,
             kinds: tuple[str, ...] = ("bar", "line", "grid"),
             readable: bool = False, coarse: bool = False) -> int:
    """Generate `count` rows; with `dedup_against`, every chart (pixel hash)
    that collides with the reference manifest is regenerated so the eval set
    never repeats a training chart. Also rejects intra-set duplicates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    forbidden: set[str] = set()
    if dedup_against is not None:
        base = dedup_against.parent
        for line in dedup_against.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            p = Path(row["image"])
            if not p.is_absolute():
                p = base / p
            try:
                with Image.open(p) as im:
                    forbidden.add(_png_hash(im))
            except (FileNotFoundError, OSError):
                continue
    rng = random.Random(seed)
    rows = []
    seen: set[str] = set()
    skipped = 0
    for i in range(count):
        kind = kinds[i % len(kinds)]
        for _ in range(max_retries):
            image, prompt, answer, meta = MAKERS[kind](rng, readable, coarse)
            h = _png_hash(image)
            if h not in forbidden and h not in seen:
                break
            skipped += 1
        else:
            raise RuntimeError(f"failed to find a non-colliding {kind} after {max_retries} retries")
        seen.add(h)
        example_id = f"figure_{split}_{seed}_{i:06d}"
        image_path = out_dir / f"{example_id}.png"
        image.save(image_path, format="PNG", optimize=True)
        rows.append(Example(
            example_id=example_id,
            modality="image",
            image=str(image_path.resolve()),
            prompt=prompt,
            answer=answer,
            split=split,
            source="programmatic_pillow",
            task="visual_math",
            visual_required=True,
            metadata={**meta, "seed": seed, "render": "readable" if readable else "base"},
        ))
    if skipped:
        print(f"dedup: regenerated {skipped} colliding charts (forbidden={len(forbidden)})")
    return write_jsonl(manifest, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/generated/figure_math/images"))
    ap.add_argument("--manifest", type=Path, default=Path("data/generated/figure_math/train.jsonl"))
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--split", choices=["train", "eval", "ood"], default="train")
    ap.add_argument("--dedup-against", type=Path, default=None,
                    help="reference manifest whose chart images the new set must not repeat (content hash)")
    ap.add_argument("--kinds", default="bar,line,grid",
                    help="comma-separated chart kinds to rotate over")
    ap.add_argument("--readable", action="store_true",
                    help="legibility variant: 14px tick labels + grid lines (discriminative experiment)")
    ap.add_argument("--coarse", action="store_true",
                    help="quantize bar/line values to multiples of 10 (resolution probe)")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    assert set(kinds) <= set(MAKERS), f"unknown kinds: {kinds}"
    n = generate(args.count, args.out_dir, args.manifest, args.seed, args.split,
                 args.dedup_against, kinds=kinds, readable=args.readable, coarse=args.coarse)
    print(f"generated={n} manifest={args.manifest}")


if __name__ == "__main__":
    main()
