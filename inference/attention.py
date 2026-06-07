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
    from .kv_cache import AdaptiveQuantizedKVCache
except ImportError:
    from _common import Linear, RMSNorm, apply_rotary_emb
    from kernel import sparse_attn
    from config import ModelArgs
    from compressor import Compressor
    from attention_indexer import Indexer
    from kv_cache import AdaptiveQuantizedKVCache

try:
    from ._layers import QKNorm
except ImportError:
    from _layers import QKNorm

try:
    from .debug import push_attention_debug
except ImportError:
    push_attention_debug = None


# ══════════════════════════════════════════════════════════════════════
# RELOCATED HELPERS  (moved from model.py to prevent circular imports)
# ══════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=8)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int = 4096,
    base: float = 1000000.0,
    factor: float = 32.0,
    beta_fast: int = 64,
    beta_slow: int = 2,
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


class DualRoPECache(nn.Module):
    """
    Two independent RoPE frequency bands.

    - LOCAL:  base_frequency = 10,000  (standard, nearby positions)
    - GLOBAL: base_frequency = 1,000,000 (extended, 2M-range positions)
    """

    LOCAL_BASE = 10_000
    GLOBAL_BASE = 1_000_000

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        original_seq_len: int = 4096,
        factor: float = 32.0,
        beta_fast: int = 64,
        beta_slow: int = 2,
        local_base: Optional[int] = None,
        global_base: Optional[int] = None,
    ):
        super().__init__()
        l_base = local_base or self.LOCAL_BASE
        g_base = global_base or self.GLOBAL_BASE
        self.register_buffer(
            "local_freqs",
            precompute_freqs_cis(
                dim,
                max_seq_len,
                original_seq_len,
                base=l_base,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            ),
            persistent=False,
        )
        self.register_buffer(
            "global_freqs",
            precompute_freqs_cis(
                dim,
                max_seq_len,
                original_seq_len,
                base=g_base,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            ),
            persistent=False,
        )

    def get_frequencies(self, use_global: bool = False) -> torch.Tensor:
        return self.global_freqs if use_global else self.local_freqs


@lru_cache(maxsize=4)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos > 0 and seqlen > 1:
        rows = []
        for j in range(seqlen):
            p = start_pos + j
            if p >= window_size - 1:
                idx = p % window_size
                row_j = torch.cat(
                    [torch.arange(idx + 1, window_size), torch.arange(0, idx + 1)],
                    dim=0,
                )
            else:
                row_j = F.pad(torch.arange(p + 1), (0, window_size - p - 1), value=-1)
            rows.append(row_j)
        matrix = torch.stack(rows, dim=0)
    elif start_pos >= window_size - 1:
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
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)
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
        self._cache_valid_len = 0  # Track how many valid positions are in the cache

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
        **kwargs,
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
            self._cache_valid_len = min(N, win)
        else:
            write_start = start_pos % win
            if write_start + N <= win:
                self.kv_cache[:B, write_start : write_start + N] = kv
            else:
                part1_len = win - write_start
                part2_len = N - part1_len
                self.kv_cache[:B, write_start:win] = kv[:, :part1_len]
                self.kv_cache[:B, 0:part2_len] = kv[:, part1_len:]
            self._cache_valid_len = min(start_pos + N, win)

        if start_pos == 0:
            K_cache = kv
        else:
            # Only attend to valid (filled) cache positions to avoid attending to zeros.
            # When cache is full (start_pos + N >= win), use the entire circular buffer.
            valid = self._cache_valid_len
            if valid >= win:
                K_cache = self.kv_cache[:B]
            else:
                K_cache = self.kv_cache[:B, :valid]

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
        self.layer_id = layer_id
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
            self.compressor.layer_id = self.layer_id
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
        use_opt = (
            getattr(args, "use_fp8_kv", False)
            or getattr(args, "use_turboquant", False)
            or getattr(args, "use_kv_eviction", False)
            or getattr(args, "use_compaction", False)
            or (getattr(args, "frac_shared_layers", 0.0) > 0.0)
        )
        if use_opt:
            self.kv_cache = AdaptiveQuantizedKVCache(
                max_batch=max_batch_size_comp,
                max_seq=kv_cache_size,
                head_dim=self.head_dim,
                args=args,
                dtype=torch.bfloat16,
            )
        else:
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

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        r_step: int = 0,
        **kwargs,
    ):
        bsz, seqlen, _ = x.size()
        freqs_cis_layer = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Dynamic resizing of attention kv_cache for larger batch size (e.g. GRPO)
        if bsz > self.kv_cache.shape[0]:
            if isinstance(self.kv_cache, AdaptiveQuantizedKVCache):
                self.kv_cache.resize(bsz)
            else:
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
                    self.indexer.resize_buffers(bsz, device=x.device)
                self.compressor.resize_buffers(bsz, device=x.device)

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

        if isinstance(self.kv_cache, AdaptiveQuantizedKVCache):
            self.kv_cache.set_queries(q.mean(dim=2))

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_rope = apply_rotary_emb(kv[..., -rd:].contiguous(), freqs_cis_layer)
        kv_nope = kv[..., :-rd].contiguous()
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        # ── Run Compressor/Caching First if start_pos == 0 ──
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                compressor_out = self.compressor(x, start_pos, r_step=r_step)
                if compressor_out is not None:
                    if isinstance(compressor_out, tuple):
                        kv_compress, event_prob = compressor_out
                        self._last_event_prob = event_prob
                    else:
                        kv_compress = compressor_out
                    kv = torch.cat([kv, kv_compress], dim=1)

        # ── Compute indices after Compressor has run ──
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)

        if self.compress_ratio:
            offset = seqlen if start_pos == 0 else win
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

        if isinstance(self.kv_cache, AdaptiveQuantizedKVCache):
            topk_idxs = self.kv_cache.filter_topk_idxs(topk_idxs, start_pos, win)

        # ── Run Compressor/Caching if start_pos > 0 ──
        if start_pos > 0:
            write_start = start_pos % win
            if write_start + seqlen <= win:
                self.kv_cache[:bsz, write_start : write_start + seqlen] = kv
            else:
                part1_len = win - write_start
                part2_len = seqlen - part1_len
                self.kv_cache[:bsz, write_start:win] = kv[:, :part1_len]
                self.kv_cache[:bsz, 0:part2_len] = kv[:, part1_len:]
            if self.compress_ratio:
                self.compressor(x, start_pos, r_step=r_step)

            topk_idxs = torch.clamp(topk_idxs, min=-1, max=self.kv_cache.size(1) - 1)
            o = sparse_attn(
                q,
                self.kv_cache[:bsz],
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
                soft_cap=self.attn_logits_soft_cap,
            )
        else:
            topk_idxs = torch.clamp(topk_idxs, min=-1, max=kv.size(1) - 1)
            o = sparse_attn(
                q,
                kv,
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
            if isinstance(self.kv_cache, AdaptiveQuantizedKVCache):
                self.kv_cache.resize(bsz)
            else:
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
                    self.indexer.resize_buffers(bsz, device=x.device)
                self.compressor.resize_buffers(bsz, device=x.device)

    def reset_cache(self) -> None:
        self.kv_cache.detach_().zero_()


# ══════════════════════════════════════════════════════════════════════
# HEAVILY COMPRESSED ATTENTION (HCA)  —  without learned Indexer
# ══════════════════════════════════════════════════════════════════════


class HCAAttention(Attention):
    """Multi-head Latent Attention with sliding window + HCA compression (no indexer, higher ratio)."""

    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__(args, layer_id)
        self.layer_id = layer_id
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
            self.compressor.layer_id = self.layer_id
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
        use_opt = (
            getattr(args, "use_fp8_kv", False)
            or getattr(args, "use_turboquant", False)
            or getattr(args, "use_kv_eviction", False)
            or getattr(args, "use_compaction", False)
            or (getattr(args, "frac_shared_layers", 0.0) > 0.0)
        )
        if use_opt:
            self.kv_cache = AdaptiveQuantizedKVCache(
                max_batch=max_batch_size_comp,
                max_seq=kv_cache_size,
                head_dim=self.head_dim,
                args=args,
                dtype=torch.bfloat16,
            )
        else:
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

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        r_step: int = 0,
        **kwargs,
    ):
        bsz, seqlen, _ = x.size()
        freqs_cis_layer = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Dynamic resizing of attention kv_cache for larger batch size (e.g. GRPO)
        if bsz > self.kv_cache.shape[0]:
            if isinstance(self.kv_cache, AdaptiveQuantizedKVCache):
                self.kv_cache.resize(bsz)
            else:
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
                self.compressor.resize_buffers(bsz, device=x.device)

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

        if isinstance(self.kv_cache, AdaptiveQuantizedKVCache):
            self.kv_cache.set_queries(q.mean(dim=2))

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_rope = apply_rotary_emb(kv[..., -rd:].contiguous(), freqs_cis_layer)
        kv_nope = kv[..., :-rd].contiguous()
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        # ── Run Compressor/Caching First if start_pos == 0 ──
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                compressor_out = self.compressor(x, start_pos, r_step=r_step)
                if compressor_out is not None:
                    if isinstance(compressor_out, tuple):
                        kv_compress, event_prob = compressor_out
                        self._last_event_prob = event_prob
                    else:
                        kv_compress = compressor_out
                    kv = torch.cat([kv, kv_compress], dim=1)

        # ── Compute indices after Compressor has run ──
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)

        if self.compress_ratio:
            offset = seqlen if start_pos == 0 else win
            # HCA dynamic index path (no learned indexer)
            cache_cap = self.kv_cache.shape[1] - win
            ptr_val = self.compressor.cache_write_ptr[:bsz].max().item()
            if push_attention_debug is not None:
                push_attention_debug(
                    layer_id=self.layer_id,
                    bsz=bsz,
                    cache_write_ptr=self.compressor.cache_write_ptr[:bsz].tolist(),
                    ptr_val=ptr_val,
                    cache_cap=cache_cap,
                )
            cache_len = max(1, min(ptr_val, cache_cap))
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

        if isinstance(self.kv_cache, AdaptiveQuantizedKVCache):
            topk_idxs = self.kv_cache.filter_topk_idxs(topk_idxs, start_pos, win)

        # ── Run Compressor/Caching if start_pos > 0 ──
        if start_pos > 0:
            write_start = start_pos % win
            if write_start + seqlen <= win:
                self.kv_cache[:bsz, write_start : write_start + seqlen] = kv
            else:
                part1_len = win - write_start
                part2_len = seqlen - part1_len
                self.kv_cache[:bsz, write_start:win] = kv[:, :part1_len]
                self.kv_cache[:bsz, 0:part2_len] = kv[:, part1_len:]
            if self.compress_ratio:
                self.compressor(x, start_pos, r_step=r_step)

            topk_idxs = torch.clamp(topk_idxs, min=-1, max=self.kv_cache.size(1) - 1)
            o = sparse_attn(
                q,
                self.kv_cache[:bsz],
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
                soft_cap=self.attn_logits_soft_cap,
            )
        else:
            topk_idxs = torch.clamp(topk_idxs, min=-1, max=kv.size(1) - 1)
            o = sparse_attn(
                q,
                kv,
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
            if isinstance(self.kv_cache, AdaptiveQuantizedKVCache):
                self.kv_cache.resize(bsz)
            else:
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
                self.compressor.resize_buffers(bsz, device=x.device)

    def reset_cache(self) -> None:
        self.kv_cache.detach_().zero_()


# ══════════════════════════════════════════════════════════════════════
# HYBRID SLIDING GLOBAL  —  Phase 3 stub
# ══════════════════════════════════════════════════════════════════════


class HybridSlidingGlobal(Attention):
    """Gemma-4 hybrid sliding window + global attention.

    Splits heads into *local* (sliding window, 10K RoPE) and *global*
    (full context, 1M RoPE) groups within a single layer.  Uses standard
    multi-head QKV projections (not MLA low-rank).  Optional QK norm
    with per-head learned scale for training stability.
    """

    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__(args, layer_id)
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.eps = args.norm_eps
        self.attn_logits_soft_cap = getattr(args, "attn_logits_soft_cap", None)

        attn_cfg = args.attention_config
        self.global_heads = attn_cfg.global_heads
        self.local_heads = self.n_heads - self.global_heads
        assert self.local_heads > 0, "Hybrid needs at least 1 local head"
        assert self.global_heads > 0, "Hybrid needs at least 1 global head"
        self.window_size = getattr(args, "sliding_window_size", 512)
        self.global_key_size = attn_cfg.global_key_size
        self.k_eq_v_global = attn_cfg.k_eq_v_global
        self.qk_norm_with_scale = attn_cfg.qk_norm_with_scale
        self.local_base = attn_cfg.local_base_frequency
        self.global_base = attn_cfg.global_base_frequency

        # ── QKV projections (standard MHA, not MLA) ──
        self.wq = Linear(self.dim, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.wv = Linear(self.dim, self.head_dim)

        self.wq_g = Linear(self.dim, self.global_heads * self.head_dim)
        self.wk_g = Linear(self.dim, self.global_heads * self.head_dim)
        if not self.k_eq_v_global:
            self.wv_g = Linear(self.dim, self.global_heads * self.head_dim)

        # ── QK norm + per-head scale ──
        if self.qk_norm_with_scale:
            self.q_norm = QKNorm(self.head_dim, self.eps)
            self.k_norm = QKNorm(self.head_dim, self.eps)
            self.q_norm_g = QKNorm(self.head_dim, self.eps)
            self.k_norm_g = QKNorm(self.head_dim, self.eps)
            self.qk_head_scale = nn.Parameter(
                torch.ones(self.n_heads, dtype=torch.float32)
            )

        # ── Output projection (grouped low-rank, matching MLA) ──
        heads_per_group = self.n_heads // self.n_groups
        self.wo_a = Linear(
            heads_per_group * self.head_dim,
            self.n_groups * self.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        # ── KV caches ──
        max_batch = args.max_batch_size * args.num_residual_streams
        self.register_buffer(
            "local_k_cache",
            torch.zeros(max_batch, self.window_size, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "local_v_cache",
            torch.zeros(max_batch, self.window_size, self.head_dim),
            persistent=False,
        )
        gcache = min(args.max_seq_len, self.global_key_size)
        ghdim = self.global_heads * self.head_dim
        self.register_buffer(
            "global_k_cache",
            torch.zeros(max_batch, gcache, ghdim),
            persistent=False,
        )
        self.register_buffer(
            "global_v_cache",
            torch.zeros(max_batch, gcache, ghdim),
            persistent=False,
        )
        self.register_buffer(
            "global_write_ptr", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self._local_cache_valid_len = 0  # Track valid local cache positions

        # ── Dual RoPE ──
        self.rope_cache = DualRoPECache(
            self.rope_head_dim,
            args.max_seq_len + 1024,
            original_seq_len=0,
            factor=1.0,
            beta_fast=0,
            beta_slow=0,
            local_base=self.local_base,
            global_base=self.global_base,
        )

    def _sliding_attend(self, q_local: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Sliding window MHA for local heads.  q_local: (B, N, Lh, Dh)."""
        B, N, Lh, Dh = q_local.shape
        win = self.window_size

        if start_pos == 0:
            if N <= win:
                self.local_k_cache[:B, :N] = self._local_k
                self.local_v_cache[:B, :N] = self._local_v
            else:
                cutoff = N % win
                k_chunks = self._local_k[:, -win:].split([win - cutoff, cutoff], dim=1)
                v_chunks = self._local_v[:, -win:].split([win - cutoff, cutoff], dim=1)
                self.local_k_cache[:B, cutoff:win] = k_chunks[0]
                self.local_k_cache[:B, :cutoff] = k_chunks[1]
                self.local_v_cache[:B, cutoff:win] = v_chunks[0]
                self.local_v_cache[:B, :cutoff] = v_chunks[1]
            self._local_cache_valid_len = min(N, win)

            K = self.local_k_cache[:B, : min(N, win)].unsqueeze(1)
            V = self.local_v_cache[:B, : min(N, win)].unsqueeze(1)
            is_causal = True
            mask = None
        else:
            slot = start_pos % win
            self.local_k_cache[:B, slot] = self._local_k[:, 0]
            self.local_v_cache[:B, slot] = self._local_v[:, 0]
            self._local_cache_valid_len = min(start_pos + 1, win)

            # Only attend to valid positions to avoid attending to zero-filled slots
            valid = self._local_cache_valid_len
            if valid >= win:
                K = self.local_k_cache[:B].unsqueeze(1)
                V = self.local_v_cache[:B].unsqueeze(1)
            else:
                K = self.local_k_cache[:B, :valid].unsqueeze(1)
                V = self.local_v_cache[:B, :valid].unsqueeze(1)
            is_causal = False
            mask = None

        # MHA: expand K/V across local heads
        K = K.expand(-1, Lh, -1, -1).to(q_local.dtype)
        V = V.expand(-1, Lh, -1, -1).to(q_local.dtype)
        q_t = q_local.transpose(1, 2)

        if self.attn_logits_soft_cap is not None:
            scores = torch.matmul(q_t.float(), K.transpose(-2, -1).float())
            scores = scores * self.softmax_scale
            if isinstance(self._local_scale, torch.Tensor):
                scores = scores * self._local_scale.to(scores.dtype)
            else:
                scores = scores * self._local_scale
            scores = (
                torch.tanh(scores / self.attn_logits_soft_cap)
                * self.attn_logits_soft_cap
            )
            if mask is not None:
                scores = scores.masked_fill(mask, -10000.0)
            probs = torch.softmax(scores, dim=-1).to(q_t.dtype)
            out = torch.matmul(probs, V)
        else:
            out = F.scaled_dot_product_attention(
                q_t,
                K,
                V,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.softmax_scale,
            )
        return out.transpose(1, 2)

    def _global_attend(self, q_global: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Full-context MHA for global heads.  q_global: (B, N, Gh, Dh)."""
        B, N, Gh, Dh = q_global.shape
        gcache = self.global_k_cache.shape[1]
        ptr = int(self.global_write_ptr.item())

        K_g = self._global_k.reshape(B, N, Gh, Dh)
        V_g = self._global_v.reshape(B, N, Gh, Dh)

        if start_pos == 0:
            # Prefill: write up to gcache tokens
            write_len = min(N, gcache)
            self.global_k_cache[:B, :write_len] = K_g[:, :write_len].reshape(
                B, write_len, Gh * Dh
            )
            self.global_v_cache[:B, :write_len] = V_g[:, :write_len].reshape(
                B, write_len, Gh * Dh
            )
            self.global_write_ptr[0] = write_len
        else:
            # Autoregressive: append one token (circular eviction)
            slot = ptr % gcache
            self.global_k_cache[:B, slot] = K_g[:, 0].reshape(B, Gh * Dh)
            self.global_v_cache[:B, slot] = V_g[:, 0].reshape(B, Gh * Dh)
            self.global_write_ptr[0] = (ptr + 1) % (
                2 * gcache
            )  # allow overflow tracking

        # Full KV for attention: use up to min(ptr, gcache) cached tokens
        avail = min(ptr if start_pos > 0 else N, gcache)
        if start_pos > 0 and N == 1:
            # Single step decode: use only the valid portion of the global cache
            valid_global = min(ptr + 1, gcache)
            K_cache = self.global_k_cache[:B, :valid_global].reshape(
                B, valid_global, Gh, Dh
            )
            V_cache = self.global_v_cache[:B, :valid_global].reshape(
                B, valid_global, Gh, Dh
            )
            is_causal = False
            mask = None
        else:
            K_cache = self.global_k_cache[:B, :avail].reshape(B, avail, Gh, Dh)
            V_cache = self.global_v_cache[:B, :avail].reshape(B, avail, Gh, Dh)
            is_causal = start_pos == 0 and N > 1
            mask = None

        K_c = K_cache.permute(0, 2, 1, 3).to(q_global.dtype)
        V_c = V_cache.permute(0, 2, 1, 3).to(q_global.dtype)
        q_t = q_global.transpose(1, 2)

        if self.attn_logits_soft_cap is not None:
            scores = torch.matmul(q_t.float(), K_c.transpose(-2, -1).float())
            scores = scores * self.softmax_scale
            if isinstance(self._global_scale, torch.Tensor):
                scores = scores * self._global_scale.to(scores.dtype)
            else:
                scores = scores * self._global_scale
            scores = (
                torch.tanh(scores / self.attn_logits_soft_cap)
                * self.attn_logits_soft_cap
            )
            if mask is not None:
                scores = scores.masked_fill(mask, -10000.0)
            probs = torch.softmax(scores, dim=-1).to(q_t.dtype)
            out = torch.matmul(probs, V_c)
        else:
            out = F.scaled_dot_product_attention(
                q_t,
                K_c,
                V_c,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.softmax_scale,
            )
        return out.transpose(1, 2)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0, **kwargs
    ) -> torch.Tensor:
        B, N, _ = x.shape
        Dh = self.head_dim
        rd = self.rope_head_dim
        nope = self.nope_head_dim

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q_g = self.wq_g(x)
        k_g = self.wk_g(x)
        if not self.k_eq_v_global:
            v_g = self.wv_g(x)
        else:
            v_g = k_g

        q = q.unflatten(-1, (self.n_heads, Dh))
        q_g = q_g.unflatten(-1, (self.global_heads, Dh))

        local_freqs = self.rope_cache.get_frequencies(use_global=False)[
            start_pos : start_pos + N
        ]
        global_freqs = self.rope_cache.get_frequencies(use_global=True)[
            start_pos : start_pos + N
        ]

        q_nope, q_rope = q[..., :nope], q[..., nope:]
        q_rope = apply_rotary_emb(q_rope.contiguous(), local_freqs)
        q = torch.cat([q_nope, q_rope], dim=-1)

        q_g_nope, q_g_rope = q_g[..., :nope], q_g[..., nope:]
        q_g_rope = apply_rotary_emb(q_g_rope.contiguous(), global_freqs)
        q_g = torch.cat([q_g_nope, q_g_rope], dim=-1)

        k_rope = apply_rotary_emb(k[..., -rd:].contiguous(), local_freqs)
        k = torch.cat([k[..., :-rd], k_rope], dim=-1)
        self._local_k = k
        self._local_v = v

        k_g = k_g.unflatten(-1, (self.global_heads, Dh))
        k_g_rope = apply_rotary_emb(
            k_g[..., -rd:].contiguous(),
            global_freqs,
        )
        k_g = torch.cat([k_g[..., :nope], k_g_rope], dim=-1)
        self._global_k = k_g.reshape(B, N, self.global_heads * Dh)
        self._global_v = (
            v_g if v_g.dim() == 3 else v_g.reshape(B, N, self.global_heads * Dh)
        )

        if self.qk_norm_with_scale:
            q_2d = q.reshape(B * N, self.n_heads, Dh)
            k_2d = self._local_k.reshape(B * N, Dh)
            q_2d = self.q_norm(q_2d.reshape(-1, Dh)).reshape(B * N, self.n_heads, Dh)
            k_2d = self.k_norm(k_2d.reshape(-1, Dh)).reshape(B, N, Dh)
            q = q_2d.reshape(B, N, self.n_heads, Dh)
            self._local_k = k_2d

            q_g_2d = q_g.reshape(B * N, self.global_heads, Dh)
            q_g_2d = self.q_norm_g(q_g_2d.reshape(-1, Dh)).reshape(
                B * N, self.global_heads, Dh
            )
            k_g_2d = self._global_k.reshape(B * N, self.global_heads * Dh)
            k_g_2d = self.k_norm_g(k_g_2d.reshape(-1, Dh)).reshape(
                B, N, self.global_heads * Dh
            )
            q_g = q_g_2d.reshape(B, N, self.global_heads, Dh)
            self._global_k = k_g_2d

            # Per-head scales
            local_scale = self.qk_head_scale[: self.local_heads]
            global_scale = self.qk_head_scale[self.local_heads :]
        else:
            local_scale = 1.0
            global_scale = 1.0

        if isinstance(local_scale, torch.Tensor):
            local_scale = local_scale.reshape(1, -1, 1, 1)
            global_scale = global_scale.reshape(1, -1, 1, 1)
        self._local_scale = local_scale
        self._global_scale = global_scale

        # 4. Attend
        Lh = self.local_heads
        Gh = self.global_heads
        q_local = q[:, :, :Lh]
        q_global = q_g

        local_out = self._sliding_attend(q_local, start_pos)
        global_out = self._global_attend(q_global, start_pos)

        out = torch.cat([local_out, global_out], dim=2)
        out = out.reshape(B, N, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", out.float(), wo_a.float())
        out = self.wo_b(out.flatten(2).to(x.dtype))
        return out

    def resize_buffers(self, bsz: int, device: torch.device) -> None:
        for name in (
            "local_k_cache",
            "local_v_cache",
            "global_k_cache",
            "global_v_cache",
        ):
            buf = getattr(self, name)
            if bsz > buf.shape[0]:
                new = torch.zeros(bsz, *buf.shape[1:], device=x.device, dtype=buf.dtype)
                new[: buf.shape[0]] = buf
                self.register_buffer(name, new, persistent=False)

    def reset_cache(self) -> None:
        self.local_k_cache.detach_().zero_()
        self.local_v_cache.detach_().zero_()
        self.global_k_cache.detach_().zero_()
        self.global_v_cache.detach_().zero_()
        self.global_write_ptr[0] = 0
        self._local_cache_valid_len = 0


# ══════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ══════════════════════════════════════════════════════════════════════

__all__ = [
    "precompute_freqs_cis",
    "DualRoPECache",
    "get_window_topk_idxs",
    "get_compress_topk_idxs",
    "Attention",
    "MLAAttention",
    "CSAAttention",
    "HCAAttention",
    "HybridSlidingGlobal",
]
