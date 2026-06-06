"""
Lasmoid — attention.py (Phase 0 SOLiD Refactoring)
=================================================================
Shared Attention interface with CSA, HCA, MLA implementations.

After Phase 0 Batch 3 (model.py refactoring), temporary `from model import ...`
imports below will be cleaned up to import from _common.py instead.
"""

import math
from functools import lru_cache
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

try:
    from ._common import Linear, RMSNorm, apply_rotary_emb
    from .kernel import sparse_attn
    from .config import ModelArgs
    from .compressor import Compressor
    from .attention_indexer import Indexer
except ImportError:
    from _common import Linear, RMSNorm, apply_rotary_emb
    from kernel import sparse_attn
    from config import ModelArgs
    from compressor import Compressor
    from attention_indexer import Indexer


# ══════════════════════════════════════════════════════════════════════
# RELOCATED HELPERS  (moved from model.py to prevent circular imports)
# ══════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=4)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int = 0,
    base: float = 10000.0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        lf = (torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn)
        return torch.clamp(lf, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


@lru_cache(maxsize=4)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1
        )
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size)
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(maxsize=4)
def get_compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int
):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


# ══════════════════════════════════════════════════════════════════════
# SHARED ATTENTION INTERFACE
# ══════════════════════════════════════════════════════════════════════


class Attention(nn.Module, ABC):
    """Shared interface for all attention variants (CSA, HCA, MLA, Hybrid)."""

    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__()

    @abstractmethod
    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0, **kwargs
    ) -> torch.Tensor: ...

    def resize_buffers(self, bsz: int, device: torch.device) -> None:
        pass

    def reset_cache(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════
# MULTI-HEAD LATENT ATTENTION (MLA)
# ══════════════════════════════════════════════════════════════════════


class MLAAttention(Attention):
    """Multi-head Latent Attention (MLA) with sliding window + optional concept fusion."""

    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__(args, layer_id)
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.nope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.eps = args.norm_eps
        self.attn_logits_soft_cap = getattr(args, "attn_logits_soft_cap", None)

        # Low-rank Q projection
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)

        # Latent KV compression
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # Grouped O projection (from DeepSeek-V4)
        heads_per_group = self.n_heads // self.n_groups
        self.wo_a = Linear(
            heads_per_group * self.head_dim,
            self.n_groups * self.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, self.dim)

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.softmax_scale = self.head_dim**-0.5

        # Sliding window cache
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.window_size, self.head_dim),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        concept_db: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        win = self.kv_cache.shape[1]

        # 1. Project Query
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_nope, q_rope = q[..., : -self.rope_head_dim], q[..., -self.rope_head_dim :]
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # 2. Compress KV
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = (
            kv[..., : -self.rope_head_dim],
            kv[..., -self.rope_head_dim :],
        )
        kv_rope = apply_rotary_emb(kv_rope, freqs_cis)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        # 3. Sliding window KV cache update
        if B > self.kv_cache.shape[0]:
            new_cache = torch.zeros(
                B,
                win,
                self.head_dim,
                device=self.kv_cache.device,
                dtype=self.kv_cache.dtype,
            )
            new_cache[: self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_cache, persistent=False)

        if start_pos == 0:
            self.kv_cache.detach_().zero_()
            if N <= win:
                self.kv_cache[:B, :N] = kv
            else:
                cutoff = N % win
                self.kv_cache[:B, cutoff:win], self.kv_cache[:B, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
        else:
            slot = start_pos % win
            self.kv_cache[:B, slot] = kv[:, 0]

        K_cache = kv if start_pos == 0 else self.kv_cache[:B]

        # Fuse retrieved concept states directly into KV sequence attention space
        if concept_db is not None:
            concept_kv = self.wkv(concept_db)
            concept_kv = self.kv_norm(concept_kv)
            concept_kv = torch.cat(
                [
                    concept_kv[..., : -self.rope_head_dim],
                    concept_kv[..., -self.rope_head_dim :],
                ],
                dim=-1,
            )

            K_combined = torch.cat([K_cache, concept_kv], dim=1)
        else:
            K_combined = K_cache

        # Attention scaling
        q_t = q.transpose(1, 2)
        is_causal = (
            start_pos == 0 and N > 1 and concept_db is None and cu_seqlens is None
        )
        attn_mask = None

        if start_pos == 0 and N > 1:
            if concept_db is not None or cu_seqlens is not None:
                Seq_combined = K_combined.size(1)
                Seq_token = K_cache.size(1)

                mask = torch.ones(B, N, Seq_combined, dtype=torch.bool, device=x.device)
                causal_mask = torch.triu(
                    torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1
                )
                mask[:, :, :N] = causal_mask.unsqueeze(0)

                if Seq_token > N:
                    mask[:, :, N:Seq_token] = True
                if Seq_combined > Seq_token:
                    mask[:, :, Seq_token:] = False

                if cu_seqlens is not None:
                    for b in range(B):
                        doc_boundaries = cu_seqlens[b]
                        for i in range(len(doc_boundaries) - 1):
                            start_idx = doc_boundaries[i].item()
                            end_idx = doc_boundaries[i + 1].item()
                            if start_idx >= N:
                                continue
                            mask[b, start_idx:end_idx, :start_idx] = True
                            if end_idx < N:
                                mask[b, start_idx:end_idx, end_idx:N] = True

                attn_mask = mask

        kv_h = K_combined.unsqueeze(1).expand(-1, self.n_heads, -1, -1).to(q_t.dtype)

        if attn_mask is not None or self.attn_logits_soft_cap is not None:
            scores = (
                torch.matmul(q_t.float(), kv_h.transpose(-2, -1).float())
                * self.softmax_scale
            )
            if self.attn_logits_soft_cap is not None:
                scores = (
                    torch.tanh(scores / self.attn_logits_soft_cap)
                    * self.attn_logits_soft_cap
                )

            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask.unsqueeze(1), -10000.0)
            elif is_causal:
                causal_mask = torch.triu(
                    torch.ones(N, kv_h.size(-2), dtype=torch.bool, device=x.device),
                    diagonal=1,
                )
                scores = scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(1), -10000.0
                )

            probs = torch.softmax(scores, dim=-1).to(q_t.dtype)
            attn_out = torch.matmul(probs, kv_h)
        else:
            attn_out = F.scaled_dot_product_attention(
                q_t,
                kv_h,
                kv_h,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.softmax_scale,
            )

        attn_out_perm = attn_out.transpose(1, 2)
        o = attn_out_perm.reshape(B, N, self.n_groups, -1)
        wo_a_w = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a_w.float())
        out = self.wo_b(o.flatten(2).to(x.dtype))
        return out


# ══════════════════════════════════════════════════════════════════════
# COMPRESSED SPARSE ATTENTION (CSA)  —  with learned Indexer
# ══════════════════════════════════════════════════════════════════════


class CSAAttention(Attention):
    """Multi-head Latent Attention with sliding window + CSA compression (use_indexer=True)."""

    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__(args, layer_id)
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.sliding_window_size
        self.compress_ratio = args.csa_compression_ratio
        self.eps = args.norm_eps
        self.attn_logits_soft_cap = getattr(args, "attn_logits_soft_cap", None)

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # Qwen-style Gated Attention (NeurIPS 2025 Best Paper)
        self.wg = Linear(self.dim, self.n_heads * self.head_dim)

        heads_per_group = self.n_heads // self.n_groups
        self.wo_a = Linear(
            heads_per_group * self.head_dim,
            self.n_groups * self.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        # Semantic Connection Pairformer Projections and Caches removed (Issue 1)

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            self.indexer = Indexer(args, self.compress_ratio)
        else:
            self.compressor = None
            self.indexer = None

        kv_cache_size = self.window_size + (
            max(1, args.max_seq_len // self.compress_ratio)
            if self.compress_ratio
            else 0
        )
        max_batch_size_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size_comp, kv_cache_size, self.head_dim),
            persistent=False,
        )

        if self.compress_ratio:
            original_seq_len = args.original_seq_len
            rope_theta = getattr(args, "compress_rope_theta", 40000.0)
        else:
            original_seq_len = 0
            rope_theta = args.rope_theta

        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len + 1024,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis_layer = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Dynamic resizing of attention kv_cache for larger batch size (e.g. GRPO)
        if bsz > self.kv_cache.shape[0]:
            device = x.device
            new_kv_cache = torch.zeros(
                bsz,
                self.kv_cache.shape[1],
                self.kv_cache.shape[2],
                device=device,
                dtype=self.kv_cache.dtype,
            )
            new_kv_cache[: self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_kv_cache, persistent=False)

            if self.compress_ratio:
                self.compressor.kv_cache = self.kv_cache[:, win:]
                if self.indexer is not None:
                    self.indexer.resize_buffers(bsz, device=device)
                self.compressor.resize_buffers(bsz, device=device)

        if self.compress_ratio:
            if self.compressor.kv_cache is None:
                self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # q
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_rope = apply_rotary_emb(q[..., -rd:].contiguous(), freqs_cis_layer)
        q_nope = q[..., :-rd].contiguous()
        q = torch.cat([q_nope, q_rope], dim=-1)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_rope = apply_rotary_emb(kv[..., -rd:].contiguous(), freqs_cis_layer)
        kv_nope = kv[..., :-rd].contiguous()
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)

        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                # HCA dynamic index generation (no indexer)
                cache_cap = self.kv_cache.shape[1] - win
                cache_len = max(
                    1,
                    min(self.compressor.cache_write_ptr[:bsz].max().item(), cache_cap),
                )
                if start_pos == 0:
                    fired_positions = self.compressor.fired_indices_buf[
                        :bsz, :cache_len
                    ]
                    query_positions = torch.arange(seqlen, device=x.device).view(
                        1, seqlen, 1
                    )
                    mask = fired_positions.unsqueeze(1) > query_positions
                    matrix = (
                        torch.arange(cache_len, device=x.device)
                        .view(1, 1, cache_len)
                        .expand(bsz, seqlen, -1)
                    )
                    compress_topk_idxs = torch.where(mask, -1, matrix + offset)
                else:
                    compress_topk_idxs = (
                        torch.arange(cache_len, device=x.device)
                        .view(1, 1, cache_len)
                        .expand(bsz, seqlen, -1)
                        + offset
                    )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                compressor_out = self.compressor(x, start_pos)
                if compressor_out is not None:
                    if isinstance(compressor_out, tuple):
                        kv_compress, event_prob = compressor_out
                        self._last_event_prob = event_prob
                    else:
                        kv_compress = compressor_out
                    kv = torch.cat([kv, kv_compress], dim=1)

            topk_idxs = torch.clamp(topk_idxs, min=-1, max=kv.size(1) - 1)
            o = sparse_attn(
                q,
                kv,
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
                soft_cap=self.attn_logits_soft_cap,
            )
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)

            topk_idxs = torch.clamp(topk_idxs, min=-1, max=self.kv_cache.size(1) - 1)
            o = sparse_attn(
                q,
                self.kv_cache[:bsz],
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
                soft_cap=self.attn_logits_soft_cap,
            )

        # Apply Qwen-style head-specific sigmoid gate (NeurIPS 2025 Best Paper)
        g = torch.sigmoid(self.wg(x)).unflatten(-1, (self.n_heads, self.head_dim))
        o = o * g

        # o
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float())
        x = self.wo_b(o.flatten(2).to(x.dtype))
        return x

    def resize_buffers(self, bsz: int, device: torch.device) -> None:
        if bsz > self.kv_cache.shape[0]:
            new_kv_cache = torch.zeros(
                bsz,
                self.kv_cache.shape[1],
                self.kv_cache.shape[2],
                device=device,
                dtype=self.kv_cache.dtype,
            )
            new_kv_cache[: self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_kv_cache, persistent=False)
            if self.compress_ratio:
                self.compressor.kv_cache = self.kv_cache[:, self.window_size :]
                if self.indexer is not None:
                    self.indexer.resize_buffers(bsz, device=device)
                self.compressor.resize_buffers(bsz, device=device)

    def reset_cache(self) -> None:
        self.kv_cache.detach_().zero_()


# ══════════════════════════════════════════════════════════════════════
# HEAVILY COMPRESSED ATTENTION (HCA)  —  without learned Indexer
# ══════════════════════════════════════════════════════════════════════


class HCAAttention(Attention):
    """Multi-head Latent Attention with sliding window + HCA compression (no indexer, higher ratio)."""

    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__(args, layer_id)
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.sliding_window_size
        self.compress_ratio = args.hca_compression_ratio
        self.eps = args.norm_eps
        self.attn_logits_soft_cap = getattr(args, "attn_logits_soft_cap", None)

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # Qwen-style Gated Attention (NeurIPS 2025 Best Paper)
        self.wg = Linear(self.dim, self.n_heads * self.head_dim)

        heads_per_group = self.n_heads // self.n_groups
        self.wo_a = Linear(
            heads_per_group * self.head_dim,
            self.n_groups * self.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        # Semantic Connection Pairformer Projections and Caches removed (Issue 1)

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            self.indexer = None  # HCA: no learned sparse indexer
        else:
            self.compressor = None
            self.indexer = None

        kv_cache_size = self.window_size + (
            max(1, args.max_seq_len // self.compress_ratio)
            if self.compress_ratio
            else 0
        )
        max_batch_size_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size_comp, kv_cache_size, self.head_dim),
            persistent=False,
        )

        if self.compress_ratio:
            original_seq_len = args.original_seq_len
            rope_theta = getattr(args, "compress_rope_theta", 40000.0)
        else:
            original_seq_len = 0
            rope_theta = args.rope_theta

        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len + 1024,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis_layer = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Dynamic resizing of attention kv_cache for larger batch size (e.g. GRPO)
        if bsz > self.kv_cache.shape[0]:
            device = x.device
            new_kv_cache = torch.zeros(
                bsz,
                self.kv_cache.shape[1],
                self.kv_cache.shape[2],
                device=device,
                dtype=self.kv_cache.dtype,
            )
            new_kv_cache[: self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_kv_cache, persistent=False)

            if self.compress_ratio:
                self.compressor.kv_cache = self.kv_cache[:, win:]
                self.compressor.resize_buffers(bsz, device=device)

        if self.compress_ratio:
            if self.compressor.kv_cache is None:
                self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis

        # q
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_rope = apply_rotary_emb(q[..., -rd:].contiguous(), freqs_cis_layer)
        q_nope = q[..., :-rd].contiguous()
        q = torch.cat([q_nope, q_rope], dim=-1)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_rope = apply_rotary_emb(kv[..., -rd:].contiguous(), freqs_cis_layer)
        kv_nope = kv[..., :-rd].contiguous()
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)

        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            # HCA dynamic index path (no learned indexer)
            cache_cap = self.kv_cache.shape[1] - win
            cache_len = max(
                1, min(self.compressor.cache_write_ptr[:bsz].max().item(), cache_cap)
            )
            if start_pos == 0:
                fired_positions = self.compressor.fired_indices_buf[:bsz, :cache_len]
                query_positions = torch.arange(seqlen, device=x.device).view(
                    1, seqlen, 1
                )
                mask = fired_positions.unsqueeze(1) > query_positions
                matrix = (
                    torch.arange(cache_len, device=x.device)
                    .view(1, 1, cache_len)
                    .expand(bsz, seqlen, -1)
                )
                compress_topk_idxs = torch.where(mask, -1, matrix + offset)
            else:
                compress_topk_idxs = (
                    torch.arange(cache_len, device=x.device)
                    .view(1, 1, cache_len)
                    .expand(bsz, seqlen, -1)
                    + offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                compressor_out = self.compressor(x, start_pos)
                if compressor_out is not None:
                    if isinstance(compressor_out, tuple):
                        kv_compress, event_prob = compressor_out
                        self._last_event_prob = event_prob
                    else:
                        kv_compress = compressor_out
                    kv = torch.cat([kv, kv_compress], dim=1)

            topk_idxs = torch.clamp(topk_idxs, min=-1, max=kv.size(1) - 1)
            o = sparse_attn(
                q,
                kv,
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
                soft_cap=self.attn_logits_soft_cap,
            )
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)

            topk_idxs = torch.clamp(topk_idxs, min=-1, max=self.kv_cache.size(1) - 1)
            o = sparse_attn(
                q,
                self.kv_cache[:bsz],
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
                soft_cap=self.attn_logits_soft_cap,
            )

        # Apply Qwen-style head-specific sigmoid gate (NeurIPS 2025 Best Paper)
        g = torch.sigmoid(self.wg(x)).unflatten(-1, (self.n_heads, self.head_dim))
        o = o * g

        # o
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float())
        x = self.wo_b(o.flatten(2).to(x.dtype))
        return x

    def resize_buffers(self, bsz: int, device: torch.device) -> None:
        if bsz > self.kv_cache.shape[0]:
            new_kv_cache = torch.zeros(
                bsz,
                self.kv_cache.shape[1],
                self.kv_cache.shape[2],
                device=device,
                dtype=self.kv_cache.dtype,
            )
            new_kv_cache[: self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_kv_cache, persistent=False)
            if self.compress_ratio:
                self.compressor.kv_cache = self.kv_cache[:, self.window_size :]
                self.compressor.resize_buffers(bsz, device=device)

    def reset_cache(self) -> None:
        self.kv_cache.detach_().zero_()


# ══════════════════════════════════════════════════════════════════════
# HYBRID SLIDING GLOBAL  —  Phase 3 stub
# ══════════════════════════════════════════════════════════════════════


class HybridSlidingGlobal(Attention):
    """Gemma-4 hybrid sliding window + global attention.  Phase 3 stub."""

    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__(args, layer_id)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("Hybrid attention — Phase 3")


# ══════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ══════════════════════════════════════════════════════════════════════

__all__ = [
    "precompute_freqs_cis",
    "get_window_topk_idxs",
    "get_compress_topk_idxs",
    "Attention",
    "MLAAttention",
    "CSAAttention",
    "HCAAttention",
    "HybridSlidingGlobal",
]
