"""Conservative, executable visual GRPO/REINFORCE-style loop.

It uses group-relative rewards and a frozen reference model. The rollout path
uses the MM model's KV cache and reuses one projected image embedding for the
whole group. This is intentionally a single-GPU, auditable baseline; it is not
presented as a full DAPO reproduction or a distributed rollout engine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.mm_minimind import MMConfig, MMForCausalLM
from model.projector import VisionProjector
from model.vision_encoder import FrozenSigLIP2
from train.mm_dataset import encode_prompt_row, project_vision_features
from train.amp import autocast_context, configure_precision


def smooth_reward(text: str, gold: str) -> float:
    """Smooth group-relative reward.

    Anti-hack (project-one lesson): only the LAST number in the response
    counts, so padding the tail with numbers gains nothing. Exact match is
    1.0; otherwise a distance partial credit of at most 0.3
    (1 - |pred-gold|/max(|gold|,1)). No number at all -> 0.0. Answers here
    are pure integers, so numeric equality is the right comparison.
    v5 lesson: the answer field now carries CoT text (ends with the answer
    number), so gold must go through the SAME last-number extraction — the
    old float(gold) raised ValueError on CoT and silently zeroed EVERY
    reward (only visible once the GRPO mix switched to CoT answers).
    """
    cleaned = text.replace(",", "").replace("，", "").replace("−", "-")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return 0.0
    try:
        pred = float(nums[-1])
        gold_cleaned = str(gold).replace(",", "").replace("，", "").replace("−", "-")
        gold_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", gold_cleaned)
        gold_v = float(gold_nums[-1] if gold_nums else gold_cleaned)
    except ValueError:
        return 0.0
    if pred == gold_v:
        return 1.0
    dist = abs(pred - gold_v) / max(abs(gold_v), 1.0)
    return 0.3 * (1.0 - min(dist, 1.0))


def group_advantage(rewards: torch.Tensor, eps: float = 0.1, clip: float = 3.0) -> torch.Tensor:
    """Group-relative advantage with a std floor AND a +/-clip.

    The floor alone is not enough: with a near-uniform group (std ~0.02 after
    clamping the denominator) advantages blew up to +/-45 in project one;
    clipping keeps the surrogate stable (project-one rl_utils.group_advantage).
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return ((rewards - mean) / std.clamp(min=eps)).clamp(-clip, clip)


def cosine_lr(step: int, total_steps: int, lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def token_logprobs(model, sequences: torch.Tensor, prompt_len: int, vision_features: torch.Tensor,
                   audio_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of the generated tokens with eos-padding masked out.

    Sequences are padded to the group max length with eos(2). Including the
    padding in the sum pollutes the advantage/KL (measured: a 12-token sample
    padded to 20 gained ~65 nats of fake signal). Returns
    (masked per-token logp [B, T], per-token mask [B, T_gen]).
    """
    logits, _, _ = model(sequences, vision_features=vision_features, audio_features=audio_features)
    logp = F.log_softmax(logits[:, :-1], -1)
    selected = logp.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    # The chat template carries its own <|im_end|> (id 2) mid-prompt, so the
    # stop token must be searched only inside the generated span. Padding is
    # also id 2, but it only ever follows the real eos (a no-eos sample always
    # fills max_new_tokens, so it is never the padded one). Searching the full
    # sequence clamped ends to prompt_len, zeroing the mask, the loss, the
    # KL, and every gradient: the checkpoint came out bit-identical to its
    # SFT input and GRPO silently no-oped.
    gen_span = sequences[:, prompt_len:]
    eos_in_gen = gen_span == 2
    has_eos = eos_in_gen.any(dim=1)
    ends = torch.where(
        has_eos,
        prompt_len + eos_in_gen.float().argmax(dim=1).long(),
        torch.full((sequences.size(0),), sequences.size(1), device=sequences.device, dtype=torch.long),
    )
    seq_mask = torch.arange(sequences.size(1), device=sequences.device).unsqueeze(0) < ends.unsqueeze(1)
    # Both returns must share the generated-token width [B, T_gen];
    # `selected` columns prompt_len-1..T-2 are sequence positions
    # prompt_len..T-1, which is exactly what `seq_mask[:, prompt_len:]` covers.
    gen = seq_mask[:, prompt_len:]
    return selected[:, prompt_len - 1:] * gen, gen


def load_mm_checkpoint(path: Path, tokenizer_size: int, device: torch.device):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("model", raw)
    cfg_data = raw.get("config", {})
    cfg = MMConfig(vocab_size=tokenizer_size, **{k: v for k, v in cfg_data.items() if k in MMConfig.__dataclass_fields__ and k != "vocab_size"})
    model = MMForCausalLM(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[rl-init] missing={len(missing)} unexpected={len(unexpected)}")
    proj_in = int(cfg_data.get("image_hidden_size", 768)) if isinstance(cfg_data, dict) else 768
    projector = VisionProjector(proj_in, cfg.hidden_size, 1024).to(device)
    if "projector" in raw:
        projector.load_state_dict(raw["projector"], strict=False)
    # GRPO v3: audio rows roll out on the whisper-feature span, so the audio
    # projector (out/align_audio/audio_projector.pt, re-aligned inside the
    # mixed SFT) rides along in the same checkpoint.
    audio_projector = VisionProjector(768, cfg.hidden_size, 1024).to(device)
    if raw.get("audio_projector") is not None:
        audio_projector.load_state_dict(raw["audio_projector"], strict=False)
        print("[rl-init] audio_projector loaded from checkpoint")
    return model, projector, audio_projector, cfg


def rollout_for_row(row, args, policy, projector, reference_projector, audio_projector,
                    reference_audio_projector, vision, tokenizer, cfg, device, amp_dtype) -> dict:
    """Encode one row's prompt+image and sample a group of generations.

    Audio rows (modality=="audio") feed the precomputed whisper-feature npy
    through the audio projector instead of the visual projector; the chart is
    still encoded (audio math questions put the numbers in the figure), so
    `visual` is computed for every row and audio rows additionally carry
    `audio`/`reference_audio` in the returned dict.

    Returns samples/batch/texts/rewards (post-overlong-shaping)/visual/
    reference_visual/audio/reference_audio/prompt_len so the caller can either
    update on the group or re-roll a different row.
    """
    modality = row.get("modality", "text")
    is_audio = modality == "audio" or row.get("audio_features") is not None
    prompt_ids, prompt_len = encode_prompt_row(row, tokenizer, cfg.max_position_embeddings,
                                               cfg.image_token_len, cfg.audio_token_len)
    prompt_ids = prompt_ids.to(device)
    image_path = Path(row["image"])
    if not image_path.is_absolute():
        image_path = args.data.parent / image_path
    with Image.open(image_path).convert("RGB") as image:
        pixels = vision.processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        raw = vision.encode(pixels)
    visual = project_vision_features(raw, projector, cfg.image_token_len)
    with torch.no_grad():
        reference_visual = project_vision_features(raw, reference_projector, cfg.image_token_len)
    audio, reference_audio = None, None
    if is_audio:
        npy = Path(row["audio_features"])
        if not npy.is_absolute():
            npy = args.data.parent / npy
        # npy is [256, d]; interpolate needs a batch dim like vision's [1, T, d].
        feats = torch.from_numpy(np.ascontiguousarray(np.load(npy), dtype=np.float32)).unsqueeze(0).to(device)
        # The policy projection keeps its graph (the update pass reuses the
        # projected features through token_logprobs, so the audio projector
        # must stay attached for gradients — mirroring the visual branch); the
        # reference projection is frozen and must not.
        audio = project_vision_features(feats, audio_projector, cfg.audio_token_len)
        with torch.no_grad():
            reference_audio = project_vision_features(feats, reference_audio_projector, cfg.audio_token_len)
    samples = []
    for _ in range(args.group_size):
        with autocast_context(device, amp_dtype):
            seq = policy.generate(prompt_ids, vision_features=visual.detach(),
                                  audio_features=None if audio is None else audio.detach(),
                                  max_new_tokens=args.max_new_tokens,
                                  eos_id=2, temperature=args.temperature, top_k=args.top_k)
        samples.append(seq[0])
    max_len = max(x.numel() for x in samples)
    batch = torch.stack([F.pad(x, (0, max_len - x.numel()), value=2) for x in samples])
    texts = [tokenizer.decode(x[prompt_len:].tolist()) for x in samples]
    raw_rewards = [smooth_reward(t, row["answer"]) for t in texts]
    # Overlong shaping: samples that hit the generation cap without an eos are
    # truncated; halve the reward ONLY when wrong so the model is pushed toward
    # concise answers without punishing a correct-but-verbose one (project-one
    # v2 lesson).
    for i, seq in enumerate(samples):
        if seq.numel() - prompt_len >= args.max_new_tokens and raw_rewards[i] < 1.0:
            raw_rewards[i] *= 0.5
    return {"samples": samples, "batch": batch, "texts": texts, "rewards": raw_rewards,
            "visual": visual, "reference_visual": reference_visual,
            "audio": audio, "reference_audio": reference_audio, "prompt_len": prompt_len}


def main() -> None:
    ap = argparse.ArgumentParser(description="visual math group-relative RL")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="MM-SFT checkpoint containing model and projector(s)")
    ap.add_argument("--vision-model", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--save-dir", type=Path, default=Path("out/mm_grpo"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50, help="sampling top-k in logit space; 0 disables")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--save-interval", type=int, default=100, help="checkpoint every N steps as mm_grpo_step{n}.pt")
    ap.add_argument("--resample-attempts", type=int, default=1, help="max rows tried per step when the group reward is uniform (v5 lesson: >1 multiplies per-step cost on small answer spaces)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = ap.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("train_mm_grpo.py is intentionally single-GPU; use train_mm_sft.py with torchrun for 4-card training")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, scaler = configure_precision(device, args.precision)
    print(f"[mm_grpo] device={device} precision={args.precision}")
    policy, projector, audio_projector, cfg = load_mm_checkpoint(args.checkpoint, len(tokenizer.get_vocab()), device)
    reference, reference_projector, reference_audio_projector, _ = load_mm_checkpoint(args.checkpoint, len(tokenizer.get_vocab()), device)
    reference.eval(); projector.eval(); reference_projector.eval()
    audio_projector.eval(); reference_audio_projector.eval()
    for p in reference.parameters(): p.requires_grad_(False)
    for p in reference_projector.parameters(): p.requires_grad_(False)
    for p in reference_audio_projector.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(projector.parameters())
                                  + list(audio_projector.parameters()), lr=args.lr)
    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("image") and r.get("answer") is not None]
    if not rows:
        raise SystemExit("RL data has no image examples")
    vision = FrozenSigLIP2(args.vision_model, device=device, dtype=torch.float32).load()
    policy.train(); projector.train()
    # Shuffled row order with a cursor: adjacent steps visit different types
    # (bar/line/grid are interleaved in the source), and uniform groups re-roll
    # the next rows instead of burning the step.
    order = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(order)
    cursor = 0
    uniform_skips = 0
    for step in range(args.steps):
        for pg in optimizer.param_groups:
            pg["lr"] = cosine_lr(step, args.steps, args.lr, args.warmup_steps)
        result = None
        attempt = 0
        while True:
            row = rows[order[cursor % len(order)]]
            cursor += 1
            result = rollout_for_row(row, args, policy, projector, reference_projector, audio_projector,
                                     reference_audio_projector, vision, tokenizer, cfg, device, amp_dtype)
            attempt += 1
            if len(set(result["rewards"])) > 1 or attempt >= args.resample_attempts:
                break
        rewards = torch.tensor(result["rewards"], dtype=torch.float32, device=device)
        if torch.all(rewards == rewards[0]):
            uniform_skips += 1
            if step % 10 == 0:
                print(f"[mm_grpo] step={step} skipped uniform reward={rewards[0].item():.3f} after {attempt} attempts")
            # Checkpoint on the boundary even when this step is a no-op
            # (v7 lesson: the save check below is unreachable on uniform
            # steps, so step200 was never written and the rotation stalled).
            if (step + 1) % args.save_interval == 0 or step + 1 == args.steps:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": policy.state_dict(), "projector": projector.state_dict(),
                            "audio_projector": audio_projector.state_dict(),
                            "config": vars(cfg), "step": step + 1},
                           args.save_dir / f"mm_grpo_step{step + 1}.pt")
                print(f"[mm_grpo] saved=mm_grpo_step{step + 1}.pt (uniform_skips={uniform_skips}, skipped step)")
            continue
        batch = result["batch"]
        prompt_len = result["prompt_len"]
        visual_batch = result["visual"].expand(args.group_size, -1, -1).contiguous()
        reference_visual_batch = result["reference_visual"].expand(args.group_size, -1, -1).contiguous()
        audio_batch = None
        reference_audio_batch = None
        if result["audio"] is not None:
            audio_batch = result["audio"].expand(args.group_size, -1, -1).contiguous()
            reference_audio_batch = result["reference_audio"].expand(args.group_size, -1, -1).contiguous()
        # Group-relative advantage with a std floor AND a +/-clip (project-one
        # rl_utils.group_advantage: the floor alone let std~0.02 groups reach
        # advantages of +/-45 and destabilize the surrogate).
        adv = group_advantage(rewards)
        with autocast_context(device, amp_dtype):
            new_lp_t, gen_mask = token_logprobs(policy, batch, prompt_len, visual_batch, audio_batch)
            with torch.no_grad():
                # old_lp = rollout-policy probability of the sampled sequences
                # (the policy has not been updated since sampling).
                old_lp_t, _ = token_logprobs(policy, batch, prompt_len, visual_batch, audio_batch)
                ref_lp_t, _ = token_logprobs(reference, batch, prompt_len, reference_visual_batch, reference_audio_batch)
        if not torch.isfinite(new_lp_t).all() or not torch.isfinite(old_lp_t).all() or not torch.isfinite(ref_lp_t).all():
            raise FloatingPointError("non-finite log probability")
        # GRPO/DAPO-style clip-higher surrogate, per generated token:
        # positive advantages are clipped at 1+eps, negative are not clipped.
        eps = 0.2
        adv_t = adv.unsqueeze(1).detach()
        ratios = (new_lp_t - old_lp_t.detach()).exp()
        clipped = torch.where(adv_t > 0, ratios.clamp(max=1.0 + eps), ratios)
        gen_tokens = gen_mask.sum(-1).clamp(min=1)
        pg_loss = -(clipped * adv_t * gen_mask).sum(-1) / gen_tokens
        # Per-token KL, normalized by real generated length (not padded length).
        kl_t = (new_lp_t - ref_lp_t.detach()) * gen_mask
        kl_per_seq = kl_t.sum(-1) / gen_tokens
        kl = kl_per_seq.mean()
        loss = pg_loss.mean() + args.kl_beta * kl
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        clip_params = list(policy.parameters()) + list(projector.parameters())
        if audio_projector is not None:
            clip_params += list(audio_projector.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer); scaler.update()
        rtype = (row.get("metadata") or {}).get("type", "?")
        if step % 5 == 0:
            print(f"[mm_grpo] step={step} type={rtype} loss={loss.item():.5f} reward={rewards.mean().item():.3f} kl={kl.item():.5f}")
        if (step + 1) % args.save_interval == 0 or step + 1 == args.steps:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": policy.state_dict(), "projector": projector.state_dict(),
                        "audio_projector": audio_projector.state_dict(),
                        "config": vars(cfg), "step": step + 1},
                       args.save_dir / f"mm_grpo_step{step + 1}.pt")
            print(f"[mm_grpo] saved=mm_grpo_step{step + 1}.pt (uniform_skips={uniform_skips})")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(), "projector": projector.state_dict(),
                "audio_projector": audio_projector.state_dict(),
                "config": vars(cfg), "step": args.steps}, args.save_dir / "mm_grpo.pt")
    print(f"[mm_grpo] saved={args.save_dir / 'mm_grpo.pt'} (uniform_skips={uniform_skips})")


if __name__ == "__main__":
    main()
