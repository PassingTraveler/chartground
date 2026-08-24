"""语音数学问答演示：听中文题面 + 看图表 -> 说出答案（语音输出闭环）。

对每个样例展示完整链路：
  1. 合成语音题面（wav，可播放）
  2. whisper-small 转写（模型实际"听到"的内容，与题目原文对照）
  3. 音频臂答案（whisper 特征 + 图表，无文字题面）
  4. 🔊 语音答案（ChatTTS 朗读音频臂回答；data/gen_answer_tts.py 在
     omni-tts env 离线合成，--answer-wav-dir 指向其输出目录时嵌入播放条）
  5. 文字臂答案（对照组：文字题面 + 图表）
  6. 标准答案 + 对错判定

输出自包含 HTML（音频/图片全部 base64 内嵌，可直接在浏览器打开）。
--reuse-json：跳过模型推理，从 out-json 重渲染（合成答案语音后二次渲染用）。
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

# 推理依赖全部延迟加载：--reuse-json 渲染路径只做 JSON/soundfile/base64
# 拼装，不应要求 torch/numpy（例如在轻量 env 里重渲染最终 HTML）。
try:
    import numpy as np
    import torch
    from PIL import Image
    from tokenizers import Tokenizer
    from eval.metrics import normalize_answer
    from eval.predict_mm import load_checkpoint
    from model.mm_template import MMTemplate
    from model.projector import VisionProjector
    from model.vision_encoder import FrozenSigLIP2
    from train.mm_dataset import project_vision_features
    _HEAVY_ERR: Exception | None = None
except ImportError as _e:
    _HEAVY_ERR = _e

TYPE_LABELS = {"bar": "柱状图", "line": "折线图", "grid": "网格图"}


def sample_rows(rows: list[dict], n: int, seed: int, correct_ids: set[str] | None,
                wrong_ids: set[str] | None) -> list[dict]:
    """按图类型均衡采样 n 行；优先展示音频臂答对的样本（correct_ids），
    混入少量错例（wrong_ids）保证真实。"""
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault((r.get("metadata") or {}).get("type", "?"), []).append(r)
    rng = np.random.default_rng(seed)
    def draw(pool, key):
        pool = [r for r in pool if r["example_id"] in key] or pool
        r = pool[int(rng.integers(len(pool)))]
        return r
    picked = []
    types = list(by_type)
    # 主体：答对的样本；最后 1-2 个：答错的样本。
    wrong_wanted = 0 if wrong_ids is None else min(2, max(1, n // 6))
    while len(picked) < n - wrong_wanted and types:
        for t in types:
            pool = by_type[t]
            if pool and len(picked) < n - wrong_wanted:
                r = draw(pool, correct_ids) if correct_ids else pool[int(rng.integers(len(pool)))]
                by_type[t].remove(r)
                picked.append(r)
        types = [t for t in types if by_type[t]]
    if wrong_wanted:
        remaining = [r for t in types for r in by_type[t]]
        wrongs = [r for r in remaining if r["example_id"] in (wrong_ids or set())]
        if not wrongs:
            wrongs = remaining
        for r in wrongs[:wrong_wanted]:
            picked.append(r)
    return picked


def transcribe(whisper, extractor, wtok, wav: np.ndarray, device: torch.device) -> str:
    """whisper-small 转写（展示"听到的内容"）。"""
    feats = extractor(wav, sampling_rate=16000, return_tensors="pt")["input_features"].to(device, dtype=torch.float32)
    with torch.no_grad():
        out = whisper.generate(feats, max_new_tokens=128)
    return wtok.decode(out[0], skip_special_tokens=True)


def answer_with(model, projector, audio_projector, tokenizer, template, vision, cfg, device,
                question: str, image_path: Path, audio_feats: np.ndarray | None, use_audio: bool) -> str:
    """一条推理：音频臂（use_audio=True，无文字题面）或文字臂。"""
    if use_audio:
        text = template.prompt_audio(question, 1, tokens_per_image=cfg.image_token_len,
                                     n_audio_tokens=cfg.audio_token_len, include_question=False)
    else:
        text = template.prompt(question, "image", 1, tokens_per_image=cfg.image_token_len)
    features = None
    audio_features = None
    with Image.open(image_path).convert("RGB") as img:
        pixels = vision.processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        raw = vision.encode(pixels)
        features = project_vision_features(raw, projector, cfg.image_token_len)
        if use_audio:
            feats = torch.from_numpy(np.ascontiguousarray(audio_feats, dtype=np.float32)).unsqueeze(0).to(device)
            audio_features = project_vision_features(feats, audio_projector, cfg.audio_token_len)
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False).ids], dtype=torch.long, device=device)
        output = model.generate(ids, vision_features=features, audio_features=audio_features,
                                max_new_tokens=128, eos_id=2)
    return tokenizer.decode(output[0, ids.size(1):].tolist()).strip()


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/generated/figure_audio_v3/eval.jsonl"))
    ap.add_argument("--checkpoint", type=Path, default=Path("out/mm_sft_v4/mm_sft.pt"))
    ap.add_argument("--vision-model", default="model/siglip2-base-patch16-256")
    ap.add_argument("--tokenizer", type=Path, default=Path("assets/project1/tokenizer/best_mm.json"))
    ap.add_argument("--whisper-model", default="openai/whisper-small")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--audio-preds", type=Path, default=Path("out/eval_audio_mm_sft_v4_v3/pred_with_audio.jsonl"),
                    help="已有音频臂预测（优先挑选答对样本展示；干净 eval 的预测，无泄露行）")
    ap.add_argument("--text-preds", type=Path, default=Path("out/eval_audio_mm_sft_v4_v3/pred_with_text.jsonl"))
    ap.add_argument("--delta-vis", type=float, default=0.166,
                    help="干净评估的 Δvis 参考值（SFT v4，比较题型）")
    ap.add_argument("--delta-audio", type=float, default=-0.018,
                    help="干净评估的 Δaudio 参考值（SFT v4，比较题型）")
    ap.add_argument("--out-html", type=Path, default=Path("demo/out/demo_audio_math_v5.html"))
    ap.add_argument("--out-json", type=Path, default=Path("demo/out/demo_audio_math_v5.json"))
    ap.add_argument("--reuse-json", action="store_true",
                    help="跳过推理，从 --out-json 读取结果重渲染（答案语音合成后二次渲染用）")
    ap.add_argument("--answer-wav-dir", type=Path, default=Path("demo/out/answer_wavs"),
                    help="gen_answer_tts.py 输出的语音答案目录；存在对应 wav 时内嵌播放条")
    args = ap.parse_args()

    if args.reuse_json:
        results = json.loads(args.out_json.read_text(encoding="utf-8"))
        _finalize(args, results)
        return

    if _HEAVY_ERR is not None:
        raise SystemExit(f"推理依赖缺失（渲染可用 --reuse-json）：{_HEAVY_ERR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model, projector_state, audio_projector_state = load_checkpoint(args.checkpoint, len(tokenizer.get_vocab()), device)
    model.eval()
    cfg = model.config
    projector = VisionProjector(cfg.image_hidden_size if hasattr(cfg, "image_hidden_size") else 768,
                                cfg.hidden_size, 1024).to(device)
    projector.load_state_dict(projector_state, strict=False)
    projector.eval()
    audio_projector = VisionProjector(768, cfg.hidden_size, 1024).to(device)
    if audio_projector_state is None:
        raise SystemExit("checkpoint 缺少 audio_projector；请用含音频的 SFT/GRPO checkpoint")
    audio_projector.load_state_dict(audio_projector_state, strict=False)
    audio_projector.eval()
    vision = FrozenSigLIP2(args.vision_model, device=device, dtype=torch.float32).load()
    template = MMTemplate()

    # whisper 完整模型（encoder+decoder）只用于转写展示；模型本身走预存 npy 特征。
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
    whisper = WhisperForConditionalGeneration.from_pretrained(args.whisper_model).to(device, dtype=torch.float32).eval()
    extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_model)
    wtok = WhisperTokenizer.from_pretrained(args.whisper_model)

    rows = [json.loads(x) for x in args.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    correct_ids: set[str] | None = None
    wrong_ids: set[str] | None = None
    if args.audio_preds.exists():
        preds = [json.loads(x) for x in args.audio_preds.read_text(encoding="utf-8").splitlines() if x.strip()]
        correct_ids = {p["example_id"] for p in preds
                       if normalize_answer(p["prediction"]) == normalize_answer(p["answer"])}
        wrong_ids = {p["example_id"] for p in preds if p["example_id"] not in correct_ids}
    rows = sample_rows(rows, args.rows, args.seed, correct_ids, wrong_ids)
    import soundfile as sf

    results = []
    for i, row in enumerate(rows):
        eid = row["example_id"]
        wav_path = Path(row["audio"])
        if not wav_path.is_absolute():
            wav_path = args.manifest.parent / wav_path
        img_path = Path(row["image"])
        if not img_path.is_absolute():
            img_path = args.manifest.parent / img_path
        feat_path = Path(row["audio_features"])
        if not feat_path.is_absolute():
            feat_path = args.manifest.parent / feat_path
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        heard = transcribe(whisper, extractor, wtok, wav.astype(np.float32), device)
        audio_feats = np.load(feat_path)
        pred_audio = answer_with(model, projector, audio_projector, tokenizer, template, vision, cfg, device,
                                 row["prompt"], img_path, audio_feats, use_audio=True)
        pred_text = answer_with(model, projector, audio_projector, tokenizer, template, vision, cfg, device,
                                row["prompt"], img_path, None, use_audio=False)
        gold = str(row["answer"])
        results.append({
            "example_id": eid,
            "type": (row.get("metadata") or {}).get("type", "?"),
            "question_text": row["prompt"],
            "heard": heard,
            "audio_sec": round(float(len(wav)) / sr, 1),
            "pred_audio": pred_audio,
            "pred_text": pred_text,
            "gold": gold,
            "audio_correct": normalize_answer(pred_audio) == normalize_answer(gold),
            "text_correct": normalize_answer(pred_text) == normalize_answer(gold),
            "image_b64": b64(img_path),
            "wav_b64": b64(wav_path),
        })
        print(f"[demo] {i + 1}/{len(rows)} {eid} {TYPE_LABELS.get(results[-1]['type'], '?')} "
              f"audio={'✓' if results[-1]['audio_correct'] else '✗'} text={'✓' if results[-1]['text_correct'] else '✗'} "
              f"| {pred_audio} vs gold {gold}", flush=True)

    _finalize(args, results)


def _finalize(args, results: list[dict]) -> None:
    """嵌入语音答案（若已合成）并写出 JSON/HTML。"""
    if args.answer_wav_dir and args.answer_wav_dir.is_dir():
        import soundfile as sf
        for r in results:
            wav = args.answer_wav_dir / f"{r['example_id']}.wav"
            if wav.exists():
                r["answer_wav_b64"] = b64(wav)
                r["answer_wav_sec"] = round(sf.info(wav).duration, 1)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for r in results if r["audio_correct"])
    args.out_html.write_text(render_html(args.checkpoint, results, n_ok, args.delta_vis, args.delta_audio), encoding="utf-8")
    print(f"[demo] 音频臂 {n_ok}/{len(results)} 正确；HTML -> {args.out_html}")


def render_html(ckpt: Path, rows: list[dict], n_ok: int, delta_vis: float,
                delta_audio: float | None) -> str:
    cards = []
    for r in rows:
        t = TYPE_LABELS.get(r["type"], r["type"])
        badge = lambda ok: f'<span class="badge {"ok" if ok else "bad"}">{"✓ 正确" if ok else "✗ 错误"}</span>'
        ans_audio = ""
        if r.get("answer_wav_b64"):
            ans_audio = (f'<div class="audio-box ans-audio"><audio controls '
                         f'src="data:audio/wav;base64,{r["answer_wav_b64"]}"></audio>'
                         f'<p class="meta">🔊 语音答案（ChatTTS 朗读模型回答，'
                         f'{r.get("answer_wav_sec", 0)}s）</p></div>')
        cards.append(f"""
<div class="card">
  <div class="card-head"><span class="chip">{t}</span><span class="eid">{r["example_id"]}</span></div>
  <div class="media">
    <div class="audio-box">
      <audio controls src="data:audio/wav;base64,{r["wav_b64"]}"></audio>
      <p class="meta">语音题面（{r["audio_sec"]}s，ChatTTS 合成）</p>
    </div>
    <img class="chart" src="data:image/png;base64,{r["image_b64"]}" alt="图表">
  </div>
  <div class="qa">
    <p class="q"><b>题目原文</b>：{r["question_text"]}</p>
    <p class="q heard"><b>whisper 转写（听到的，仅展示）</b>：{r["heard"]}<br>
    <span class="meta">注：模型直接使用 whisper 编码特征（不经过文字转写）；转写仅供参考，有同音字噪声属正常。</span></p>
    <p class="ans"><b>🎧 音频臂答案</b>：<span class="pred">{r["pred_audio"]}</span> {badge(r["audio_correct"])}</p>
    {ans_audio}
    <p class="ans"><b>✍️ 文字臂答案</b>：<span class="pred">{r["pred_text"]}</span> {badge(r["text_correct"])}</p>
    <p class="ans"><b>✅ 标准答案</b>：{r["gold"]}</p>
  </div>
</div>""")
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>语音数学问答 Demo — {ckpt}</title>
<style>
  body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f6fa; margin: 0; padding: 24px; color: #222; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
  .stats {{ display: inline-block; background: #fff; border-radius: 8px; padding: 10px 16px; margin-right: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .stats b {{ font-size: 18px; }}
  .card {{ background: #fff; border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .chip {{ background: #eef1f8; color: #40507a; border-radius: 999px; padding: 3px 10px; font-size: 12px; }}
  .eid {{ color: #999; font-size: 11px; font-family: monospace; }}
  .media {{ display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
  .audio-box {{ min-width: 280px; }}
  .meta {{ color: #999; font-size: 12px; margin: 6px 0 0; }}
  .chart {{ max-height: 220px; border: 1px solid #eee; border-radius: 8px; }}
  .qa {{ margin-top: 12px; }}
  .q, .ans {{ margin: 6px 0; font-size: 14px; }}
  .heard {{ color: #555; }}
  .pred {{ font-weight: 600; color: #1a3a8f; }}
  .badge {{ border-radius: 6px; padding: 2px 8px; font-size: 12px; margin-left: 6px; }}
  .badge.ok {{ background: #e6f6ec; color: #1a7f42; }}
  .badge.bad {{ background: #fdecec; color: #b3261e; }}
</style>
</head>
<body>
<h1>🎧 语音数学问答 Demo</h1>
<p class="sub">听中文题面 + 看图表 → 说出答案（语音输出闭环）。模型：{ckpt}（0.25B 多模态 LLM + whisper-small 特征 + AudioProjector → ChatTTS 语音回答）</p>
<div>
  <span class="stats">演示抽样音频臂正确率 <b>{n_ok}/{len(rows)}</b></span>
  <span class="stats">干净评估 Δaudio <b>{("进行中（v2 音频）" if delta_audio is None else f"{delta_audio:+.3f}")}</b></span>
  <span class="stats">干净评估 Δvis <b>{delta_vis:+.3f}</b></span>
</div>
{''.join(cards)}
</body>
</html>"""


if __name__ == "__main__":
    main()
