"""A MiniMind-compatible causal LM with projected visual token injection.

The parameter names intentionally match the project-one MiniMind checkpoint:
model.embed_tokens, model.layers.*, model.norm and lm_head. This makes the
pretrain_02b checkpoint reusable without resizing the vocabulary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MMConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    vocab_size: int = 12014
    intermediate_size: int = 2048
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True
    image_token_id: int = 12003
    video_token_id: int = 12004
    image_token_len: int = 64
    audio_token_id: int = 12005
    audio_token_len: int = 256


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match project-one's exact computation (variance in the input dtype);
        # the pretrained weights were trained with this behavior.
        w = self.weight.to(x.dtype)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w


def precompute_freqs_cis(dim: int, end: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([torch.cos(freqs), torch.cos(freqs)], -1), torch.cat([torch.sin(freqs), torch.sin(freqs)], -1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: MMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        past_len = past_key_value[0].size(2) if past_key_value is not None else 0
        c = cos[past_len:past_len + t].to(q.dtype).unsqueeze(0).unsqueeze(-2)
        s = sin[past_len:past_len + t].to(q.dtype).unsqueeze(0).unsqueeze(-2)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        if past_len == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Query position i may attend to keys <= past_len+i.
            total = k.size(2)
            allowed = torch.arange(total, device=x.device)[None, :] <= (past_len + torch.arange(t, device=x.device))[:, None]
            mask = torch.zeros((t, total), dtype=q.dtype, device=x.device)
            mask.masked_fill_(~allowed, torch.finfo(q.dtype).min)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = self.o_proj(out.transpose(1, 2).contiguous().view(b, t, -1))
        return out, ((k, v) if use_cache else None)


class FeedForward(nn.Module):
    def __init__(self, config: MMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMindBlock(nn.Module):
    def __init__(self, config: MMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x, cos, sin, past_key_value=None, use_cache=False):
        a, present = self.self_attn(self.input_layernorm(x), cos, sin, past_key_value, use_cache)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class MiniMindModel(nn.Module):
    def __init__(self, config: MMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        cos, sin = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False):
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        past_key_values = past_key_values or [None] * len(self.layers)
        past_len = 0 if past_key_values[0] is None else past_key_values[0][0].size(2)
        if past_len + x.size(1) > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {past_len + x.size(1)} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )
        presents = []
        for layer, past in zip(self.layers, past_key_values):
            x, present = layer(x, self.freqs_cos, self.freqs_sin, past, use_cache)
            if use_cache:
                presents.append(present)
        return self.norm(x), (presents if use_cache else None)


class MMForCausalLM(nn.Module):
    def __init__(self, config: MMConfig):
        super().__init__()
        self.config = config
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def _inject_modalities(self, input_ids: torch.Tensor, embeds: torch.Tensor,
                           vision_features: torch.Tensor | None, audio_features: torch.Tensor | None) -> torch.Tensor:
        """Overwrite the embedding rows at each modality's marker token.

        Vision and audio are injected by the same mechanism: the marker
        position must equal the number of projector frames, and text-only
        samples sharing a batch (zero-filled collate features) are skipped.
        """
        if vision_features is not None:
            marker = (input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)
            for b in range(input_ids.size(0)):
                positions = marker[b].nonzero(as_tuple=False).flatten()
                # Text-only samples may share a batch with image samples. Their
                # zero-filled pixel tensor is only a collate convenience and must
                # not be treated as a request to inject visual tokens.
                if positions.numel() == 0:
                    continue
                feats = vision_features[b]
                if feats.dim() == 3:
                    feats = feats.reshape(-1, feats.size(-1))
                if positions.numel() != feats.size(0):
                    video = (input_ids[b] == self.config.video_token_id)
                    if video.any():
                        raise NotImplementedError(
                            f"video modality is not implemented yet: {video.sum()} video markers found "
                            f"but the projector only produces {feats.size(0)} features. "
                            "Remove video rows from the manifest or implement frame-level features."
                        )
                    raise ValueError(f"visual marker/feature mismatch at batch {b}: markers={positions.numel()} features={feats.size(0)}")
                embeds[b, positions] = feats.to(embeds.dtype)
        if audio_features is not None:
            marker = input_ids == self.config.audio_token_id
            for b in range(input_ids.size(0)):
                positions = marker[b].nonzero(as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                feats = audio_features[b]
                if feats.dim() == 3:
                    feats = feats.reshape(-1, feats.size(-1))
                if positions.numel() != feats.size(0):
                    raise ValueError(f"audio marker/feature mismatch at batch {b}: markers={positions.numel()} features={feats.size(0)}")
                embeds[b, positions] = feats.to(embeds.dtype)
        return embeds

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                vision_features: torch.Tensor | None = None, audio_features: torch.Tensor | None = None,
                past_key_values=None, use_cache: bool = False):
        embeds = self.model.embed_tokens(input_ids)
        if past_key_values is None or all(p is None for p in past_key_values):
            embeds = self._inject_modalities(input_ids, embeds, vision_features, audio_features)
        hidden, presents = self.model(inputs_embeds=embeds, past_key_values=past_key_values, use_cache=use_cache)
        # Outside autocast, align activation and weight dtypes explicitly;
        # under autocast (bf16/fp16 training) let autocast pick the matmul
        # dtype so the large vocab-space matmul actually accelerates.
        if not torch.is_autocast_enabled():
            hidden = hidden.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss, presents

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, vision_features: torch.Tensor | None = None,
                 audio_features: torch.Tensor | None = None,
                 max_new_tokens: int = 128, eos_id: int = 2, temperature: float = 0.0, top_k: int = 0) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            seq = input_ids
            logits, _, past = self(seq, vision_features=vision_features, audio_features=audio_features, use_cache=True)
            for _ in range(max_new_tokens):
                next_logits = logits[:, -1]
                if temperature and temperature > 0:
                    # top-k in logit space (0 disables), matching the
                    # project-one rollout sampler.
                    if top_k and top_k > 0:
                        k = min(top_k, next_logits.size(-1))
                        thresh = torch.topk(next_logits, k, dim=-1).values[..., -1, None]
                        next_logits = next_logits.masked_fill(next_logits < thresh, float("-inf"))
                    next_id = torch.multinomial(F.softmax(next_logits / temperature, -1), 1)
                else:
                    next_id = next_logits.argmax(-1, keepdim=True)
                seq = torch.cat([seq, next_id], -1)
                if torch.all(next_id == eos_id):
                    break
                if seq.size(1) >= self.config.max_position_embeddings:
                    # Truncate instead of letting the next forward raise.
                    break
                logits, _, past = self(next_id, past_key_values=past, use_cache=True)
            return seq
        finally:
            self.train(was_training)

    @classmethod
    def from_project1_checkpoint(cls, config: MMConfig, checkpoint: str, device: str | torch.device = "cpu"):
        model = cls(config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init] loaded={checkpoint} missing={len(missing)} unexpected={len(unexpected)}")
        return model.to(device)
