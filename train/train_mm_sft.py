from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config import ModelConfig
from model.mm_minimind import MMConfig, MMForCausalLM
from model.projector import VisionProjector
from model.vision_encoder import FrozenSigLIP2
from train.mm_dataset import MMDataset, collate_mm, project_vision_features
from train.amp import autocast_context, configure_precision


def init_dist():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1
    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def main() -> None:
    ap = argparse.ArgumentParser(description="mixed image/text multimodal SFT")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vision-model", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--save-dir", type=Path, default=Path("out/mm_sft"))
    ap.add_argument("--projector", type=Path, help="optional projector checkpoint from train_align.py")
    ap.add_argument("--audio-projector", type=Path,
                    help="optional audio projector from train_align_audio.py; enables audio features")
    ap.add_argument("--audio-hidden-size", type=int, default=768,
                    help="whisper-small encoder d_model (512 for whisper-base); precomputed npy must match")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--freeze-llm", action="store_true")
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = ap.parse_args()
    rank, world = init_dist()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype, scaler = configure_precision(device, args.precision)
    if rank == 0:
        print(f"[mm_sft] world={world} device={device} precision={args.precision}", flush=True)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cfg = ModelConfig()
    image_id = tokenizer.token_to_id("<|image_pad|>")
    video_id = tokenizer.token_to_id("<|video_pad|>")
    audio_id = tokenizer.token_to_id("<|audio_pad|>")
    mm_cfg = MMConfig(
        vocab_size=len(tokenizer.get_vocab()),
        max_position_embeddings=cfg.max_seq_len,
        image_token_id=12003 if image_id is None else int(image_id),
        video_token_id=12004 if video_id is None else int(video_id),
        audio_token_id=12005 if audio_id is None else int(audio_id),
        image_token_len=cfg.image_token_len,
        audio_token_len=cfg.audio_token_len,
    )
    model = MMForCausalLM.from_project1_checkpoint(mm_cfg, str(args.checkpoint), device)
    vision = FrozenSigLIP2(args.vision_model, device=device, dtype=torch.float32).load()
    projector = VisionProjector(vision.hidden_size, cfg.hidden_size, cfg.projector_hidden_size).to(device)
    if args.projector:
        projector.load_state_dict(torch.load(args.projector, map_location="cpu", weights_only=False), strict=False)
    audio_projector = None
    if args.audio_projector:
        audio_projector = VisionProjector(args.audio_hidden_size, cfg.hidden_size,
                                          cfg.projector_hidden_size).to(device)
        audio_projector.load_state_dict(torch.load(args.audio_projector, map_location="cpu", weights_only=False),
                                        strict=False)
        if rank == 0:
            print(f"[mm_sft] audio projector loaded from {args.audio_projector}", flush=True)
    if args.freeze_llm:
        for p in model.parameters():
            p.requires_grad_(False)
    trainable = list(model.parameters()) + list(projector.parameters())
    if audio_projector is not None:
        trainable += list(audio_projector.parameters())
    opt = torch.optim.AdamW([p for p in trainable if p.requires_grad], lr=args.lr, weight_decay=0.01)
    ds = MMDataset(args.data, tokenizer, cfg.max_seq_len, vision.processor, cfg.image_token_len,
                   cfg.audio_token_len)
    sampler = DistributedSampler(ds, shuffle=True) if world > 1 else None
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                        collate_fn=collate_mm, num_workers=2, pin_memory=True)
    if world > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train(); projector.train(); opt.zero_grad(set_to_none=True)
    step = 0
    while step < args.steps:
        if sampler:
            sampler.set_epoch(step)
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            features = None
            if batch["pixel_values"] is not None:
                with torch.no_grad():
                    raw = vision.encode(batch["pixel_values"].to(device, non_blocking=True))
                features = project_vision_features(raw, projector, cfg.image_token_len)
            audio_features = None
            if audio_projector is not None and batch["audio_features"] is not None:
                # Precomputed [B, 256, 768] npy features (already interpolated
                # to the audio token budget by precompute_audio_features.py).
                raw_audio = batch["audio_features"].to(device, non_blocking=True)
                audio_features = project_vision_features(raw_audio, audio_projector, cfg.audio_token_len)
            with autocast_context(device, amp_dtype):
                _, loss, _ = model(ids, labels=labels, vision_features=features,
                                   audio_features=audio_features)
                scaled_loss = loss / args.grad_accum
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
            if world > 1:
                # The projector is intentionally kept outside the DDP-wrapped
                # language model; synchronize its gradients explicitly. Every
                # rank must call all_reduce with the same tensor sequence, so
                # text-only batches (projector untouched, grads None) reduce
                # zeros instead of skipping the collective: a skipped rank
                # deadlocks NCCL for the ranks that do call it.
                for p in projector.parameters():
                    g = p.grad if p.grad is not None else torch.zeros_like(p)
                    dist.all_reduce(g, op=dist.ReduceOp.SUM)
                    if p.grad is not None:
                        p.grad.copy_(g.div_(world))
                if audio_projector is not None:
                    # Same collective pattern: audio rows are rare (10% of the
                    # mix), so audio-projector grads are mostly None — reducing
                    # zeros keeps every rank's NCCL call sequence identical.
                    for p in audio_projector.parameters():
                        g = p.grad if p.grad is not None else torch.zeros_like(p)
                        dist.all_reduce(g, op=dist.ReduceOp.SUM)
                        if p.grad is not None:
                            p.grad.copy_(g.div_(world))
            if (step + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if scaler is None:
                    opt.step()
                else:
                    scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and (step % 10 == 0 or step == 1):
                print(f"[mm_sft] step={step} loss={loss.item():.5f}", flush=True)
            if step >= args.steps:
                break
    if rank == 0:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if isinstance(model, DDP) else model
        state = {"model": raw_model.state_dict(), "projector": projector.state_dict(), "config": vars(mm_cfg)}
        if audio_projector is not None:
            state["audio_projector"] = audio_projector.state_dict()
        torch.save(state, args.save_dir / "mm_sft.pt")
        print(f"[mm_sft] saved={args.save_dir / 'mm_sft.pt'}")
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
