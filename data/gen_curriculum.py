#!/usr/bin/env python3
"""课程化题型数据（为课程 SFT 备料，纯 CPU）。

探针结论驱动的两条课程（见 gen_probe_grid.py 12 格矩阵）：
  A. 序数绑定课程（ordinal curriculum）—— argmax/argmin/第二高 全部 ≈chance，
     但 named 点名单比较全 1.000。课程 CoT 把"全序"分解为逐对 named 比较
     + 显式序数转换步骤（"第X季度最高 → 答案是X"），从 n=2 递进到 n=4。
  B. sum dominance 课程 —— sum 类模板坍缩（n4_sum2 0.030、185/200 末位"3"）。
     课程只保留 dominance 成立的子集（高的一对逐柱高于低的一对），
     CoT 用逐对柱高比较论证，不需要读绝对数值。

输出 data/generated/curriculum/{train,eval}.jsonl（每格 eval 留 200 行作未来零训练探针）。
用法：python -m data.gen_curriculum
"""
import json
import random
from pathlib import Path

from data import gen_probe_grid as gp

CN = {2: "两", 3: "三", 4: "四"}


def _row(eid, q, cot, meta, split):
    return {
        "example_id": eid,
        "modality": "image",
        "image": f"images/{eid}.png",
        "prompt": q,
        "answer": cot,
        "split": split,
        "source": "curriculum",
        "task": "visual_math",
        "visual_required": True,
        "metadata": meta,
    }


# ---------- A. 序数绑定课程 ----------

def a_argmax_bridge(rng, n, ask_max=True):
    """argmax 问法 + 桥接 CoT：逐对 named 比较后显式转换到序数答案。"""
    values = rng.sample(range(20, 91), n)
    order = sorted(range(n), key=lambda i: values[i])
    ans_i = order[-1] if ask_max else order[0]
    word = "最高" if ask_max else "最低"
    q = (f"图中{CN[n]}个季度的销量分别由{CN[n]}根柱表示。"
         f"销量{word}的是第几个季度？只输出编号（1到{n}）。")
    # order 升序：order[k] 总比 order[k+1] 低（与探针 argmax 训练形式同源）
    chain = "；".join(f"第{order[k] + 1}季度的柱比第{order[k + 1] + 1}季度的柱低"
                     for k in range(n - 1))
    cot = (f"比较柱高：{chain}。"
           f"所以第{ans_i + 1}季度的销量{word}，销量{word}的是第{ans_i + 1}季度，答案是{ans_i + 1}。")
    prog = f"argmax_bridge_{n}" if ask_max else f"argmin_bridge_{n}"
    return q, cot, {"type": f"curriculum_{prog}", "n_bars": n, "form": prog,
                    "values": values, "program": prog}


def a_2nd_bridge(rng):
    """第二高：先逐对比较排全序，再显式指出第二高。"""
    values = rng.sample(range(20, 91), 4)
    order = sorted(range(4), key=lambda i: values[i])
    ans_i = order[-2]
    q = "图中四个季度的销量分别由四根柱表示。销量第二高的是第几个季度？只输出编号（1到4）。"
    chain = "；".join(f"第{order[k] + 1}季度的柱比第{order[k + 1] + 1}季度的柱低" for k in range(3))
    cot = (f"比较柱高：{chain}。"
           f"第{order[3] + 1}季度的柱最高，第{ans_i + 1}季度的柱比其余两根柱都高、只比第{order[3] + 1}季度的柱低，"
           f"所以销量第二高的是第{ans_i + 1}季度，答案是{ans_i + 1}。")
    return q, cot, {"type": "curriculum_2nd_bridge", "n_bars": 4, "form": "2nd_bridge",
                    "values": values, "program": "arg2nd_bridge"}


# ---------- B. sum dominance 课程 ----------

def b_sum2_dom(rng):
    """(Q1+Q2) vs (Q3+Q4)，只保留 dominance 成立子集；CoT 用逐对柱高论证。"""
    while True:
        values = rng.sample(range(20, 91), 4)
        a12, a34 = values[0] + values[1], values[2] + values[3]
        if a12 == a34:
            continue
        if a12 > a34 and values[0] > values[2] and values[1] > values[3]:
            ans, (hi, lo) = 1, ((0, 2), (1, 3))
            break
        if a34 > a12 and values[2] > values[0] and values[3] > values[1]:
            ans, (hi, lo) = 2, ((2, 0), (3, 1))
            break
    q = ("第1季度和第2季度的销量之和，与第3季度和第4季度的销量之和，哪个更大？"
         "只输出编号（1或2）。")
    cot = (f"第{hi[0] + 1}季度的柱比第{lo[0] + 1}季度的柱高，第{hi[1] + 1}季度的柱比第{lo[1] + 1}季度的柱高；"
           f"两根都更高，所以第{hi[0] + 1}季度和第{hi[1] + 1}季度的销量之和更大，答案是{ans}。")
    return q, cot, {"type": "curriculum_sum2_dom", "n_bars": 4, "form": "sum2_dom",
                    "values": values, "program": f"sum2_dom({hi[0]}+{hi[1]})"}


def b_sum2tall_dom(rng):
    """两根最高柱之和 vs 两根最低柱之和（组序随机），dominance 天然成立。"""
    values = rng.sample(range(20, 91), 4)
    order = sorted(range(4), key=lambda i: values[i])
    t1, t2 = order[3], order[2]
    b1, b2 = order[0], order[1]
    if rng.random() < 0.5:
        pair_hi, pair_lo, ans = (t1, t2), (b1, b2), 1
    else:
        pair_hi, pair_lo, ans = (b1, b2), (t1, t2), 2
    q = (f"第{pair_hi[0] + 1}季度和第{pair_hi[1] + 1}季度的销量之和，与"
         f"第{pair_lo[0] + 1}季度和第{pair_lo[1] + 1}季度的销量之和，哪个更大？"
         f"只输出编号（1或2）。")
    if ans == 1:
        hi, lo = pair_hi, pair_lo
    else:
        hi, lo = pair_lo, pair_hi
    cot = (f"第{hi[0] + 1}季度的柱比第{lo[0] + 1}季度的柱高，第{hi[1] + 1}季度的柱比第{lo[1] + 1}季度的柱高；"
           f"两根都更高，所以第{hi[0] + 1}季度和第{hi[1] + 1}季度的销量之和更大，答案是{ans}。")
    return q, cot, {"type": "curriculum_sum2tall_dom", "n_bars": 4, "form": "sum2tall_dom",
                    "values": values, "program": f"sum2tall_dom({hi[0]}+{hi[1]})"}


CELLS = [
    ("n2_argmax_bridge", lambda rng: a_argmax_bridge(rng, 2, True), 1500),
    ("n3_argmax_bridge", lambda rng: a_argmax_bridge(rng, 3, True), 1500),
    ("n4_argmax_bridge", lambda rng: a_argmax_bridge(rng, 4, True), 2000),
    ("n4_argmin_bridge", lambda rng: a_argmax_bridge(rng, 4, False), 2000),
    ("n4_2nd_bridge", a_2nd_bridge, 2000),
    ("n4_sum2_dom", b_sum2_dom, 2000),
    ("n4_sum2tall_dom", b_sum2tall_dom, 2000),
]
EVAL_PER_CELL = 200


def generate():
    out_dir = Path("data/generated/curriculum")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = [], []
    for name, fn, n_train in CELLS:
        rng = random.Random(20260822 + len(CELLS) + 777)
        n_eval = EVAL_PER_CELL
        for split, count, rows_out in (("train", n_train, train_rows), ("eval", n_eval, eval_rows)):
            for i in range(count):
                eid = f"curriculum_{name}_{split}_{i:04d}"
                q, cot, meta = fn(rng)
                img_path = img_dir / f"{eid}.png"
                gp.render_bars(meta["values"], img_path)
                rows_out.append(_row(eid, q, cot, meta, split))
        print(f"[curriculum] {name}: train={n_train} eval={n_eval}")
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        random.Random(20260822).shuffle(rows)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[curriculum] done -> {out_dir} (train={len(train_rows)} eval={len(eval_rows)})")


if __name__ == "__main__":
    generate()
