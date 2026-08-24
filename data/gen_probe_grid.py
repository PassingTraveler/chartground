#!/usr/bin/env python3
"""探针网格：把 groupsum/argmax 失败（≈chance）定位到具体层级。

11 格 × 200 行，SFT v5 零训练 with-image 评估。三轴交叉：
  - 柱数：2 / 3 / 4
  - 题面形式：点名比较（paircmp 式）vs argmax/argmin 式 vs 选项式（groupsum 原形）
  - 运算：比高（单比较/全序/排序）vs 求和比较（两两和 vs 和对单柱 vs 选项组）

假设（按 4 柱数据）：
  H1 题面形式：paircmp 1.000 vs argmax 0.270——点名单比较浅于全序；
  H2 柱数/步数：2/3 柱 argmax 若高，则 4 柱全序（3 步链）是瓶颈；
  H3 求和：需要绝对数值读数（§4.2 边界）——sum 类即使点名形式也应 ≈chance；
  H4 排序：第二高需要更深的序（前二），应介于 argmax 与 paircmp 之间。

用法：python -m data.gen_probe_grid（CPU，分钟级）。manifest 按格拆分，
image 字段相对各格 manifest 目录解析（images/{id}.png）。
"""
import json
import random
from pathlib import Path

from data import gen_figure_math as gf

CN = {2: "两", 3: "三", 4: "四"}
COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]


def render_bars(values, path):
    n = len(values)
    im, draw = gf.canvas()
    ax_x = gf.draw_axes(draw)
    for i, v in enumerate(values):
        x = 48 + i * 45
        y = 220 - int(v * 1.8)
        draw.rectangle((x, y, x + 26, 220), fill=COLORS[i], outline="black")
        draw.text((x + 4, 224), f"Q{i + 1}", fill="black", font=gf.font(10))
    im.save(path)


def cell_n_argmax(rng, n, ask_max=True):
    values = rng.sample(range(20, 91), n)
    order = sorted(range(n), key=lambda i: values[i])
    ans_i = order[-1] if ask_max else order[0]
    word = "最高" if ask_max else "最低"
    q = (f"图中{CN[n]}个季度的销量分别由{CN[n]}根柱表示。"
         f"销量{word}的是第几个季度？只输出编号（1到{n}）。")
    chain = "；".join(f"第{order[k] + 1}季度的柱比第{order[k + 1] + 1}季度的柱低" for k in range(n - 1))
    cot = f"比较柱高：{chain}。第{ans_i + 1}季度的柱{word}，所以答案是{ans_i + 1}。"
    prog = "argmax" if ask_max else "argmin"
    return q, cot, {"type": f"probe_n{n}_{prog}", "n_bars": n, "form": prog,
                    "values": values, "program": prog, "seed": 20260822}


def cell_n_named(rng, n):
    values = rng.sample(range(20, 91), n)
    i, j = rng.sample(range(n), 2)
    lo, hi = (i, j) if values[i] < values[j] else (j, i)
    q = (f"图中{CN[n]}个季度的销量分别由{CN[n]}根柱表示。"
         f"第{lo + 1}季度和第{hi + 1}季度的销量谁更高？只输出编号（1到{n}）。")
    cot = (f"比较第{lo + 1}季度和第{hi + 1}季度的柱高：第{lo + 1}季度的柱比第{hi + 1}季度的柱低。"
           f"第{hi + 1}季度的柱更高，所以答案是{hi + 1}。")
    return q, cot, {"type": f"probe_n{n}_named", "n_bars": n, "form": "named",
                    "values": values, "program": f"paircmp({lo},{hi})", "seed": 20260822}


def cell_n4_2nd(rng):
    values = rng.sample(range(20, 91), 4)
    order = sorted(range(4), key=lambda i: values[i])
    ans_i = order[-2]
    q = "图中四个季度的销量分别由四根柱表示。销量第二高的是第几个季度？只输出编号（1到4）。"
    chain = "；".join(f"第{order[k] + 1}季度的柱比第{order[k + 1] + 1}季度的柱低" for k in range(3))
    cot = f"比较柱高：{chain}。第{ans_i + 1}季度的柱第二高，所以答案是{ans_i + 1}。"
    return q, cot, {"type": "probe_n4_2nd", "n_bars": 4, "form": "arg2nd",
                    "values": values, "program": "arg2nd", "seed": 20260822}


def cell_sum2(rng):
    while True:
        values = rng.sample(range(20, 91), 4)
        if values[0] + values[1] != values[2] + values[3]:
            break
    s12, s34 = values[0] + values[1], values[2] + values[3]
    ans = 1 if s12 > s34 else 2
    q = ("第1季度和第2季度的销量之和，与第3季度和第4季度的销量之和，哪个更大？"
         "只输出编号（1或2）。")
    cot = f"答案是{ans}。"
    return q, cot, {"type": "probe_n4_sum2", "n_bars": 4, "form": "sum2_named",
                    "values": values, "program": f"sum(1,2)={s12} vs sum(3,4)={s34}", "seed": 20260822}


def cell_sum2v1(rng):
    while True:
        values = rng.sample(range(20, 91), 4)
        if values[0] + values[1] != values[2]:
            break
    s12, v3 = values[0] + values[1], values[2]
    ans = 1 if s12 > v3 else 2
    q = ("第1季度和第2季度的销量之和，与第3季度的销量，哪个更大？"
         "只输出编号（1或2）。")
    cot = f"答案是{ans}。"
    return q, cot, {"type": "probe_n4_sum2v1", "n_bars": 4, "form": "sum2v1_named",
                    "values": values, "program": f"sum(1,2)={s12} vs bar(3)={v3}", "seed": 20260822}


def cell_groupsum(rng):
    values = rng.sample(range(20, 91), 4)
    order = sorted(range(4), key=lambda i: values[i])
    a1, a2 = order[2], order[3]
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
    return q, cot, {"type": "probe_n4_groupsum", "n_bars": 4, "form": "groupsum_options",
                    "values": values, "program": f"groupsum({a1},{a2})", "seed": 20260822}


def generate(count=200):
    out_dir = Path("data/generated/probe_grid")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for name, fn, n in [
        ("n2_argmax", lambda rng: cell_n_argmax(rng, 2, True), 2),
        ("n2_named", lambda rng: cell_n_named(rng, 2), 2),
        ("n3_argmax", lambda rng: cell_n_argmax(rng, 3, True), 3),
        ("n3_named", lambda rng: cell_n_named(rng, 3), 3),
        ("n4_argmax", lambda rng: cell_n_argmax(rng, 4, True), 4),
        ("n4_argmin", lambda rng: cell_n_argmax(rng, 4, False), 4),
        ("n4_named", lambda rng: cell_n_named(rng, 4), 4),
        ("n4_2nd", cell_n4_2nd, 4),
        ("n4_sum2", cell_sum2, 4),
        ("n4_sum2v1", cell_sum2v1, 4),
        ("n4_groupsum", cell_groupsum, 4),
    ]:
        rng = random.Random(20260822 + len(cells))
        rows = []
        for i in range(count):
            eid = f"probe_{name}_{i:03d}"
            q, cot, meta = fn(rng)
            img_path = img_dir / f"{eid}.png"
            render_bars(meta["values"], img_path)
            rows.append({
                "example_id": eid,
                "modality": "image",
                "image": f"images/{eid}.png",
                "prompt": q,
                "answer": cot,
                "split": "eval",
                "source": "probe_grid",
                "task": "visual_math",
                "visual_required": True,
                "metadata": meta,
            })
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[probe] {name}: {count} rows, sample q: {rows[0]['prompt'][:60]!r}")
    print(f"[probe] done -> {out_dir}")


if __name__ == "__main__":
    generate()
