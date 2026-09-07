#!/usr/bin/env python
"""ChartGround 的零依赖视觉证据回放 Demo。

页面只回放仓库内已保存的评测样本和预测，不伪造在线多模态推理。
真实推理仍由 scripts/demo.py 与 demo_audio_math.py 承担。

运行：python demo/app.py --port 7862
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "generated" / "curriculum" / "eval.jsonl"
PRED_PATH = ROOT / "out" / "curriculum_eval_v7" / "pred.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_evidence() -> tuple[list[dict], dict[str, Path]]:
    """对齐 v7 保存预测与课程评测图像，保留正确和错误样例。"""
    eval_rows = {row["example_id"]: row for row in read_jsonl(EVAL_PATH)}
    evidence: list[dict] = []
    images: dict[str, Path] = {}
    for row in read_jsonl(PRED_PATH):
        base = eval_rows.get(row["example_id"])
        if base is None:
            continue
        image = (EVAL_PATH.parent / base["image"]).resolve()
        if not image.is_file():
            continue
        idx = len(evidence)
        evidence.append({
            "id": str(idx),
            "example_id": row["example_id"],
            "type": (base.get("metadata") or {}).get("type", "visual_math"),
            "question": row["prompt"],
            "gold": row["answer"],
            "prediction": row["prediction"],
            "correct": row["prediction"].strip() == row["answer"].strip(),
            "visual_required": bool(base.get("visual_required")),
        })
        images[str(idx)] = image
    # 按正确、错误交错排列，面试展示不只挑选成功案例。
    good = [x for x in evidence if x["correct"]]
    bad = [x for x in evidence if not x["correct"]]
    ordered = []
    while good or bad:
        if good:
            ordered.append(good.pop(0))
        if bad:
            ordered.append(bad.pop(0))
    remapped_images = {str(i): images[row["id"]] for i, row in enumerate(ordered)}
    for i, row in enumerate(ordered):
        row["id"] = str(i)
    return ordered, remapped_images


EVIDENCE, IMAGES = load_evidence()


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChartGround · 视觉数学 Demo</title><style>
 :root{--ink:#10272a;--muted:#60777a;--accent:#007f7a;--line:#dce8e6;--paper:#f4fbf9;--good:#e7f8ef;--bad:#fff0eb}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,"Microsoft YaHei",sans-serif}.shell{max-width:1120px;margin:auto;padding:34px 24px 54px}.eyebrow{color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}h1{font-size:clamp(29px,5vw,46px);line-height:1.08;margin:11px 0;letter-spacing:-.04em}.lead{font-size:17px;color:var(--muted);max-width:790px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric,.card{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 10px 25px rgba(16,53,49,.045)}.metric{padding:15px}.metric b{color:var(--accent);font-size:24px;display:block}.metric span,.sub{font-size:12px;color:var(--muted)}.grid{display:grid;grid-template-columns:.83fr 1.17fr;gap:16px}.card{padding:20px}h2{font-size:20px;margin:0 0 3px}select,button{font:inherit}select{width:100%;padding:10px;border:1px solid #cfdfdc;border-radius:10px;background:white}button{margin-top:11px;padding:9px 12px;border:0;border-radius:10px;background:var(--accent);color:white;font-weight:700;cursor:pointer}.stage{margin-top:12px;background:#f5faf9;border-radius:13px;border:1px dashed #c5ddda;padding:12px;min-height:240px;display:grid;place-items:center}.stage img{max-width:100%;max-height:360px;border-radius:10px;box-shadow:0 5px 18px #0e2a291a}.answer{padding:12px;border-radius:12px;background:#f4f7f7;margin-top:12px}.answer b{display:block;font-size:12px;color:var(--muted);margin-bottom:3px}.badge{display:inline-block;margin-top:10px;padding:3px 8px;border-radius:99px;font-weight:700;font-size:12px}.ok{background:var(--good);color:#167047}.bad{background:var(--bad);color:#b44720}.note{margin:20px 0;padding:12px 14px;border-left:4px solid #3baaa3;background:#e9f7f4;border-radius:7px;color:#285a56}.command{padding:12px;background:#f1f6f5;border-radius:10px;overflow:auto;color:#365856}.command+ .command{margin-top:9px}code{font-family:ui-monospace,Consolas,monospace}footer{color:var(--muted);font-size:12px;margin-top:20px}@media(max-width:760px){.metrics{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.shell{padding:24px 15px}}
</style></head><body><main class="shell"><div class="eyebrow">ChartGround · saved evaluation playback</div><h1>把“看见了图”<br>拆成可核验的视觉增益。</h1><p class="lead">这不是截图拼贴：页面从课程评测集、保存预测和原始图像中读取同一条样本。可切换图片可见性，直观看到任务为何必须依赖视觉输入。</p><div class="note">严谨口径：SFT v6 在视觉数学新题型取得 paired Δvis=0.413（95% CI [0.357, 0.471]）。下方回放为 <code>curriculum_eval_v7</code> 保存预测，包含错误样例，不把 v6 指标冒充成 v7 结果。</div><section class="metrics"><div class="metric"><b>+0.413</b><span>SFT v6 paired Δvis</span></div><div class="metric"><b>[.357, .471]</b><span>Bootstrap 95% CI</span></div><div class="metric"><b>3 模态</b><span>文本 / 图像 / 语音题面</span></div><div class="metric"><b>4×3090</b><span>DDP 混合精度训练</span></div></section><section class="grid"><article class="card"><h2>真实样本选择</h2><p class="sub">SFT v7 课程评测的模型输出回放；选择器按正确/错误交错排列。</p><select id="sample"></select><button id="toggle">隐藏图片：模拟无视觉线索</button><div class="stage" id="stage"></div></article><article class="card"><h2>评测记录</h2><p class="sub" id="meta"></p><div class="answer"><b>题目</b><span id="question"></span></div><div class="answer"><b>保存的 SFT v7 预测</b><span id="prediction"></span></div><div class="answer"><b>标准答案</b><span id="gold"></span></div><span id="badge" class="badge"></span><h2 style="margin-top:24px">接入实时多模态推理</h2><p class="sub">需要 MM checkpoint、SigLIP2 与 tokenizer；该入口真实加载模型。</p><div class="command"><code>python scripts/demo.py --checkpoint out/mm_sft_v6/mm_sft.pt --image &lt;chart.png&gt; --question "&lt;问题&gt;" --vision-model &lt;siglip2_path&gt;</code></div><p class="sub" style="margin-top:14px">语音 + 图表闭环已提供自包含 HTML 生成器：</p><div class="command"><code>python demo/demo_audio_math.py --reuse-json --out-json demo/out/demo_audio_math_v5.json</code></div></article></section><footer>离线页面只回放真实保存产物；实时推理由项目原有命令显式启动。图像隐藏只用于说明视觉必要性，并不替代 “无图” 条件下的正式推理评测。</footer></main><script>
let rows=[],visible=true;const $=s=>document.querySelector(s);const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');async function init(){rows=await(await fetch('/api/samples')).json();rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${r.correct?'✓':'✗'} ${r.type} · ${r.question}`;$('select').append(o)});$('select').onchange=show;$('#toggle').onclick=()=>{visible=!visible;show()};show()}function show(){const r=rows[$('select').value||0];$('#meta').textContent=`ID: ${r.example_id} · visual_required=${r.visual_required}`;$('#question').textContent=r.question;$('#prediction').textContent=r.prediction;$('#gold').textContent=r.gold;const b=$('#badge');b.className=`badge ${r.correct?'ok':'bad'}`;b.textContent=r.correct?'✓ 保存预测与标准答案一致':'✗ 保存预测与标准答案不一致（保留作失败案例）';$('#stage').innerHTML=visible?`<img src="/asset/${r.id}" alt="${esc(r.question)}">`:'<div class="sub">图片已隐藏。此题的关键数值只在图像中；正式 “无图” 对照需要另行运行同一模型与 prompt 的评测协议。</div>';$('#toggle').textContent=visible?'隐藏图片：模拟无视觉线索':'显示原始评测图像'}init();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/samples":
            return self._send(json.dumps(EVIDENCE, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        if path == "/health":
            return self._send(json.dumps({"ok": True, "samples": len(EVIDENCE)}).encode(), "application/json")
        if path.startswith("/asset/"):
            key = unquote(path.removeprefix("/asset/"))
            image = IMAGES.get(key)
            if image is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
            return self._send(image.read_bytes(), content_type)
        if path == "/":
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        self.send_error(HTTPStatus.NOT_FOUND)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChartGround evidence demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ChartGround demo: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
