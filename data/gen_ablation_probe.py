#!/usr/bin/env python3
"""消融探针：argmax 失败是"中文序数语言映射"还是"argmax 操作本身"？

对照格（已出分）：
  n2_argmax 0.505 / n4_argmax 0.295 / n4_argmin 0.255 —— 问法"销量最高的是第几个季度"
消融格（本文件）：
  n2_whichbar_max / n4_whichbar_max / n4_whichbar_min —— 同样的图、同样的上下文句、
  同样的答案编号，唯一区别是把"第几个季度"换成"哪根柱"。

若 whichbar 族显著高于 argmax 族 → 失败在中文序数指称（"第X季度"绑定），课程走语言改写；
若 whichbar 族同样 chance → 失败在 argmax/全序操作本身，课程走操作分解（named 比较 + 显式转换 CoT）。
"""
import json
import random
from pathlib import Path

from data import gen_probe_grid as gp

CN = {2: "两", 3: "三", 4: "四"}


def cell_whichbar(rng, n, ask_max=True):
    values = rng.sample(range(20, 91), n)
    order = sorted(range(n), key=lambda i: values[i])
    ans_i = order[-1] if ask_max else order[0]
    word = "最高" if ask_max else "最低"
    q = (f"图中{CN[n]}根柱分别表示{CN[n]}个季度的销量。"
         f"哪根柱{word}？只输出编号（1到{n}）。")
    cot = f"第{ans_i + 1}根柱{word}，所以答案是{ans_i + 1}。"
    prog = "whichbar_max" if ask_max else "whichbar_min"
    return q, cot, {"type": f"ablation_n{n}_{prog}", "n_bars": n, "form": prog,
                    "values": values, "program": prog, "seed": 20260822}


def generate(count=200):
    out_dir = Path("data/generated/probe_ablation")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        ("n2_whichbar_max", lambda rng: cell_whichbar(rng, 2, True)),
        ("n4_whichbar_max", lambda rng: cell_whichbar(rng, 4, True)),
        ("n4_whichbar_min", lambda rng: cell_whichbar(rng, 4, False)),
    ]
    for name, fn in cells:
        rng = random.Random(20260822 + len(cells) + 100)
        rows = []
        for i in range(count):
            eid = f"ablation_{name}_{i:03d}"
            q, cot, meta = fn(rng)
            img_path = img_dir / f"{eid}.png"
            gp.render_bars(meta["values"], img_path)
            rows.append({
                "example_id": eid,
                "modality": "image",
                "image": f"images/{eid}.png",
                "prompt": q,
                "answer": cot,
                "split": "eval",
                "source": "probe_ablation",
                "task": "visual_math",
                "visual_required": True,
                "metadata": meta,
            })
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ablation] {name}: {count} rows, sample q: {rows[0]['prompt'][:60]!r}")
    print(f"[ablation] done -> {out_dir}")


if __name__ == "__main__":
    generate()
