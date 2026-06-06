import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._common import (
        Linear,
        RMSNorm,
        apply_rotary_emb,
        default_dtype,
        block_size,
        scale_fmt,
        scale_dtype,
        set_dtype,
    )
    from .config import ModelArgs
except ImportError:
    from _common import (
        Linear,
        RMSNorm,
        apply_rotary_emb,
        default_dtype,
        block_size,
        scale_fmt,
        scale_dtype,
        set_dtype,
    )
    from config import ModelArgs


# ══════════════════════════════════════════════════════════════════════
# COMPRESSION MODE (P1.3: Adaptive Compression Gating)
# ══════════════════════════════════════════════════════════════════════


class CompressionMode:
    """Compression mode constants for adaptive gating."""

    NORMAL: str = "normal"  # ratio=4, full quality
    HYPER: str = "hyper"  # ratio=128, triggered >300K tokens
    EMERGENCY: str = "emergency"  # ratio=512, triggered >1.5M tokens


class AdaptiveCompressorGate(nn.Module):
    """
    Automatically selects compression mode based on context length.

    Modes:
    - NORMAL:    ratio=4, standard CSA compression, full quality
    - HYPER:     ratio=128, HCA + KV quantization + eviction, triggered at >300K tokens
    - EMERGENCY: ratio=512, aggressive compaction, triggered at >1.5M tokens

    Per-layer ratios (DeepSeek-V4-Pro pattern):
    - Early layers: low ratio (preserve detail for downstream)
    - Middle layers: moderate ratio (balanced)
    - Late layers: high ratio (semantic compression sufficient)
    """

    THRESHOLD_NORMAL_TO_HYPER: int = 300_000
    THRESHOLD_HYPER_TO_EMERGENCY: int = 1_500_000

    def __init__(self, n_layers: int, default_ratio: int = 4):
        super().__init__()
        self.ratios = nn.Parameter(
            torch.ones(n_layers) * default_ratio, requires_grad=False
        )
        self.default_ratio = default_ratio

    def get_mode(self, current_seqlen: int) -> str:
        if current_seqlen > self.THRESHOLD_HYPER_TO_EMERGENCY:
            return CompressionMode.EMERGENCY
        elif current_seqlen > self.THRESHOLD_NORMAL_TO_HYPER:
            return CompressionMode.HYPER
        else:
            return CompressionMode.NORMAL

    def get_ratio_for_mode(self, mode: str) -> int:
        if mode == CompressionMode.EMERGENCY:
            return 512
        elif mode == CompressionMode.HYPER:
            return 128
        else:
            return self.default_ratio

    def forward(self, layer_id: int, current_seqlen: int = 0) -> int:
        mode = self.get_mode(current_seqlen)
        # HYPER/EMERGENCY modes force aggressive uniform ratio across all layers
        if mode != CompressionMode.NORMAL:
            return self.get_ratio_for_mode(mode)
        # NORMAL mode: use per-layer ratios for differentiated compression profiles
        if layer_id < len(self.ratios):
            return int(self.ratios[layer_id].item())
        return self.default_ratio


# ══════════════════════════════════════════════════════════════════════
# ENHANCED EVENT DETECTOR (P1.4: Multi-Scale Boundary Detection)
# ══════════════════════════════════════════════════════════════════════


class EnhancedEventDetector(nn.Module):
    """
    Multi-scale event boundary detector for semantic compression.

    Combines three scoring mechanisms:
    - Local features: single-token boundary signals via Linear(dim, 1)
    - Window features: aggregated over +/-16 token window via Conv1d
    - Global features: segment-level topic shift detection via cross-attention

    All three scores are fused via a learned linear combination → softplus.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.local_proj = nn.Linear(dim, 1)
        self.window_conv = nn.Conv1d(dim, max(1, dim // 4), kernel_size=33, padding=16)
        self.window_proj = nn.Linear(max(1, dim // 4), 1)
        self.global_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.fusion = nn.Linear(3, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns multi-scale boundary probability: [B, S, 1], already softplus'd."""
        local = self.local_proj(h)  # [B, S, 1]
        window_feat = self.window_conv(h.transpose(1, 2)).transpose(
            1, 2
        )  # [B, S, dim//4]
        window = self.window_proj(window_feat)  # [B, S, 1]
        global_scores, _ = self.global_attn(h, h, h)  # [B, S, dim]
        global_scores = global_scores.mean(dim=-1, keepdim=True)  # [B, S, 1]
        combined = torch.cat([local, window, global_scores], dim=-1)  # [B, S, 3]
        return F.softplus(self.fusion(combined))  # [B, S, 1]


class Compressor(nn.Module):
    """
    Continuous Integrate-and-Fire (CIF) Semantic Event Compressor.
    Achieves dynamic 100X+ KV compression by dynamically pooling tokens based on semantic
    event boundaries, mirroring biological episodic memory formation.
    """

    def __init__(
        self,
        args: ModelArgs,
        compress_ratio: int = 4,
        head_dim: int = 48,
        rotate: bool = False,
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate

        # Enhanced multi-scale event boundary detector (P1.4)
        self.event_detector = EnhancedEventDetector(args.dim)

        # Adaptive compression gate for context-length-aware ratio selection (P1.3)
        self.gate = AdaptiveCompressorGate(
            n_layers=args.n_layers, default_ratio=compress_ratio
        )

        # Pooling projections
        coff = 1 + self.overlap
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.ape, 0.0, 0.02)

        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache
        self.freqs_cis: torch.Tensor = None

        # State buffers for autoregressive integration
        max_batch_size_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer(
            "kv_accumulator",
            torch.zeros(max_batch_size_comp, self.head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "gate_accumulator",
            torch.zeros(max_batch_size_comp, self.head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "fire_threshold",
            torch.zeros(max_batch_size_comp, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "cache_write_ptr",
            torch.zeros(max_batch_size_comp, dtype=torch.long),
            persistent=False,
        )

        # Fired indices buffer for causal boundary masking
        cache_cap = max(1, args.max_seq_len // compress_ratio)
        self.register_buffer(
            "fired_indices_buf",
            torch.zeros(max_batch_size_comp, cache_cap, dtype=torch.long),
            persistent=False,
        )

    def resize_buffers(self, bsz: int, device: Optional[torch.device] = None):
        if bsz > self.kv_accumulator.shape[0]:
            if device is None:
                device = self.kv_accumulator.device
            cache_cap = self.kv_cache.shape[1] if self.kv_cache is not None else 16
            self.register_buffer(
                "kv_accumulator",
                torch.zeros(bsz, self.head_dim, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "gate_accumulator",
                torch.zeros(bsz, self.head_dim, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "fire_threshold",
                torch.zeros(bsz, 1, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "cache_write_ptr",
                torch.zeros(bsz, dtype=torch.long, device=device),
                persistent=False,
            )
            self.register_buffer(
                "fired_indices_buf",
                torch.zeros(bsz, cache_cap, dtype=torch.long, device=device),
                persistent=False,
            )

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        self.resize_buffers(bsz, device=x.device)

        ratio, overlap, d, rd = (
            self.compress_ratio,
            self.overlap,
            self.head_dim,
            self.rope_head_dim,
        )
        dtype = x.dtype
        x_float = x.float()

        # Single multi-scale event boundary computation (P1.4)
        raw_alpha = self.event_detector(x_float)  # [B, S, 1]

        # Query adaptive gate for context-length-aware compression ratio (P1.3)
        total_seqlen = start_pos + seqlen
        mode = self.gate.get_mode(total_seqlen)
        ratio_mult = 1.0
        if mode != CompressionMode.NORMAL:
            mode_ratio = self.gate.get_ratio_for_mode(mode)
            ratio_mult = mode_ratio / self.compress_ratio

        kv = self.wkv(x_float)
        score = self.wgate(x_float)

        if start_pos == 0:
            # ─────────────────────────────────────────────────────────
            # VECTORIZED INTEGRATE-AND-FIRE (P1.2: Parallel Prefix Scan)
            # ─────────────────────────────────────────────────────────
            with torch.no_grad():
                self.kv_accumulator.zero_()
                self.gate_accumulator.zero_()
                self.fire_threshold.zero_()
                self.cache_write_ptr.zero_()
                self.fired_indices_buf.zero_()

            # 1. Scale alpha by mode ratio for adaptive compression (P1.3)
            # HYPER/EMERGENCY modes amplify ratio_mult → fewer fires → higher compression
            alpha = raw_alpha / ratio_mult  # [B, S, 1]

            # 2. Calculate gate scores and weighted KV
            gate_scores = F.softplus(score)  # [B, S, D]
            weighted_kv = kv * gate_scores  # [B, S, D]

            # 3. Parallel prefix scan: cumulative sum of boundary probabilities
            cum_alpha = torch.cumsum(alpha, dim=1)  # [B, S, 1]

            # 4. Map each position to its fire bucket
            # cum_alpha_prev[s] = cum_alpha[s-1] (0 for s=0)
            # fire_bucket[s] = floor(cum_alpha_prev[s]) — which fire position s feeds into
            cum_alpha_prev = F.pad(cum_alpha[:, :-1], (0, 0, 1, 0), value=0.0)
            fire_bucket = torch.floor(cum_alpha_prev).long()  # [B, S, 1]

            # 5. Determine number of fires across all batch items
            # Use global max of fire_bucket (not just last position) to ensure all
            # scatter_add/reduce_ target indices are in range
            num_fires = max(1, int(fire_bucket.max().item()) + 1)

            # 6. Scatter-add weighted KV and gate scores into fire buckets
            fire_idx = fire_bucket.expand(-1, -1, d)  # [B, S, D]
            fire_kv = torch.zeros(
                bsz, num_fires, d, device=x.device, dtype=weighted_kv.dtype
            )
            fire_gate = torch.zeros(
                bsz, num_fires, d, device=x.device, dtype=gate_scores.dtype
            )

            # Each position's full weighted contribution goes to its fire bucket
            # (Boundary split approximation: O(alpha_minor) error, negligible in practice)
            fire_kv.scatter_add_(1, fire_idx, (weighted_kv * alpha)[..., :d])
            fire_gate.scatter_add_(1, fire_idx, (gate_scores * alpha)[..., :d])

            # 7. Compute emitted KVs — each fire bucket emits one compressed token
            kv_out = torch.zeros(bsz, num_fires, d, device=x.device, dtype=dtype)
            nonzero_mask = fire_gate.abs().sum(dim=-1, keepdim=True) > 1e-10
            kv_out_nonzero = fire_kv / (fire_gate + 1e-6)
            kv_out = torch.where(nonzero_mask, kv_out_nonzero.to(dtype), kv_out)

            # 8. Compute fired indices (first position of each fire per batch)
            # Find the minimum s for each fire bucket via scatter_reduce_ amin
            s_range = (
                torch.arange(seqlen, device=x.device)
                .view(1, seqlen, 1)
                .expand(bsz, -1, -1)
            )  # [B, S, 1]
            min_pos_per_fire = torch.full(
                (bsz, num_fires, 1), seqlen, device=x.device, dtype=torch.float32
            )
            min_pos_per_fire.scatter_reduce_(
                1, fire_bucket, s_range.float(), reduce="amin", include_self=False
            )
            fired_indices_tensor = min_pos_per_fire.squeeze(-1).long()  # [B, num_fires]

            # Handle zero-fire edge case: force one fire at end of sequence
            zero_fire_mask = (fire_bucket[:, -1, 0] < 0) | (
                fired_indices_tensor.sum(dim=-1) == 0
            )
            if zero_fire_mask.any():
                for b_idx in torch.where(zero_fire_mask)[0]:
                    b_val = b_idx.item()
                    kv_out[b_val, 0] = (
                        weighted_kv[b_val, -1, :d] / (gate_scores[b_val, -1, :d] + 1e-6)
                    ).to(dtype)
                    fired_indices_tensor[b_val, 0] = seqlen - 1

            # Compute carryover remainder state for autoregressive phase
            remainder = cum_alpha[:, -1:] - torch.floor(cum_alpha[:, -1:])  # [B, 1, 1]
            # Accumulate weighted residual KV from the last fire bucket
            # Note: weighted_kv has coff*d dims, accumulator stores only d dims
            last_fire_mask = (fire_bucket == fire_bucket[:, -1:]).float()  # [B, S, 1]
            residual_weight = last_fire_mask * alpha
            accum_kv = (residual_weight * weighted_kv[..., :d]).sum(dim=1)  # [B, D]
            accum_gate = (residual_weight * gate_scores[..., :d]).sum(dim=1)  # [B, D]
            accum_prob = remainder.squeeze(-1)  # [B, 1]

            # 5. Apply RoPE to the compressed semantic nodes
            freqs_cis = self.freqs_cis[:num_fires]
            kv_rope = apply_rotary_emb(kv_out[..., -rd:].contiguous(), freqs_cis)
            kv_nope = kv_out[..., :-rd].contiguous()

            # Bypass activation quantization
            kv_out = torch.cat([kv_nope, kv_rope], dim=-1).contiguous()

            # 6. Write to cache
            cache_cap = self.kv_cache.shape[1]
            write_len = min(num_fires, cache_cap)

            with torch.no_grad():
                self.kv_cache[:bsz, :write_len] = kv_out[:, :write_len].detach()
                self.fired_indices_buf[:bsz, :write_len] = fired_indices_tensor[
                    :, :write_len
                ]
                self.cache_write_ptr[:bsz] = write_len

                # 7. Carry over the incomplete remainder to the autoregressive state buffers
                self.kv_accumulator[:bsz] = accum_kv.detach()
                self.gate_accumulator[:bsz] = accum_gate.detach()
                self.fire_threshold[:bsz] = accum_prob.detach()

            # Return kv_out and event_prob (sigmoid-bounded) for CIF loss (only in training/prefill)
            if self.training:
                return kv_out, torch.sigmoid(raw_alpha)
            return kv_out
        else:
            # Autoregressive generation phase
            # Compute bounded event probability for step-wise fire accumulation
            event_prob = torch.sigmoid(raw_alpha)  # [B, S, 1]
            # Retrieve current state from buffers (detached to break autograd graph)
            kv_acc = self.kv_accumulator[:bsz].detach().clone()
            gate_acc = self.gate_accumulator[:bsz].detach().clone()
            fire_th = self.fire_threshold[:bsz].detach().clone()

            prob = (
                event_prob[:, 0, :] / ratio_mult
                if ratio_mult > 1
                else event_prob[:, 0, :]
            )
            fire_th = fire_th + prob

            # Use only head_dim dimensions for accumulation (compressed space)
            current_gate_score = F.softplus(score[:, 0, :d])
            kv_acc = kv_acc + kv[:, 0, :d] * current_gate_score
            gate_acc = gate_acc + current_gate_score

            fire_mask = (fire_th >= 1.0).squeeze(-1)

            if fire_mask.any():
                fired_kv = kv_acc.clone()
                fired_kv[fire_mask] /= gate_acc[fire_mask] + 1e-6
                fired_kv = fired_kv[..., :d]

                # Apply normalization
                fired_kv = self.norm(fired_kv.to(dtype))

                # Apply rotary embedding based on current cache write pointer index
                kv_out_single = fired_kv.unsqueeze(1)

                # Vectorized lookup of freqs_cis per batch item
                batch_ptrs = self.cache_write_ptr[:bsz]
                freqs_cis = self.freqs_cis[batch_ptrs].unsqueeze(1)  # [B, 1, rd // 2]
                kv_rope = apply_rotary_emb(
                    kv_out_single[:, :, -rd:].contiguous(), freqs_cis
                )
                kv_nope = kv_out_single[:, :, :-rd].contiguous()

                # Bypass activation quantization
                kv_out = torch.cat([kv_nope, kv_rope], dim=-1).contiguous()

                # Write to cache with circular buffer index wrapping
                cache_cap = self.kv_cache.shape[1]
                with torch.no_grad():
                    fired_batch_indices = torch.where(fire_mask)[0]
                    for b in fired_batch_indices:
                        b_val = b.item()
                        slot = self.cache_write_ptr[b_val].item() % cache_cap
                        self.kv_cache[b_val, slot] = kv_out[b_val, 0].detach()
                        self.fired_indices_buf[b_val, slot] = start_pos
                        self.cache_write_ptr[b_val] += 1

                # Reset accumulators for fired sequences
                kv_acc = torch.where(fire_mask.unsqueeze(-1), 0.0, kv_acc)
                gate_acc = torch.where(fire_mask.unsqueeze(-1), 0.0, gate_acc)
                fire_th = torch.where(fire_mask.unsqueeze(-1), fire_th - 1.0, fire_th)

                with torch.no_grad():
                    self.kv_accumulator[:bsz] = kv_acc.detach()
                    self.gate_accumulator[:bsz] = gate_acc.detach()
                    self.fire_threshold[:bsz] = fire_th.detach()
                return kv_out
            else:
                with torch.no_grad():
                    self.kv_accumulator[:bsz] = kv_acc.detach()
                    self.gate_accumulator[:bsz] = gate_acc.detach()
                    self.fire_threshold[:bsz] = fire_th.detach()
            return None
