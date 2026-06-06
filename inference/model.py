"""
Lasmod — import-only coordinator (Phase 0 SOLiD Refactoring)
==============================================================
All class/function definitions live in focused SOLiD modules below.
This file re-exports everything and keeps the self-test.
"""

# ── Core ────────────────────────────────────────────────────────────────────
from contextlib import contextmanager
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# ── Kernel ──────────────────────────────────────────────────────────────────
try:
    from .kernel import (
        act_quant,
        fp4_act_quant,
        fp4_gemm,
        fp8_gemm,
        hc_split_sinkhorn,
        sparse_attn,
        weight_dequant,
    )
except ImportError:
    from kernel import (
        act_quant,
        fp4_act_quant,
        fp4_gemm,
        fp8_gemm,
        hc_split_sinkhorn,
        sparse_attn,
        weight_dequant,
    )

# ── Common Utilities ────────────────────────────────────────────────────────
try:
    from ._common import (
        default_dtype,
        scale_fmt,
        scale_dtype,
        block_size,
        fp4_block_size,
        set_dtype,
        RMSNorm,
        Linear,
        apply_rotary_emb,
    )
except ImportError:
    from _common import (
        default_dtype,
        scale_fmt,
        scale_dtype,
        block_size,
        fp4_block_size,
        set_dtype,
        RMSNorm,
        Linear,
        apply_rotary_emb,
    )

# ── Config ──────────────────────────────────────────────────────────────────
try:
    from .config import ModelArgs
except ImportError:
    from config import ModelArgs  # type: ignore[assignment]

# ── Vector Quantizers ───────────────────────────────────────────────────────
try:
    from .vq import VectorQuantizer, ResidualVQ
except ImportError:
    from vq import VectorQuantizer, ResidualVQ  # type: ignore[import]

# ── Concept Memory ──────────────────────────────────────────────────────────
try:
    from .concept_memory import ElasticSparseConceptMemory
except ImportError:
    from concept_memory import ElasticSparseConceptMemory  # type: ignore[import]

# ── Transformer Block ───────────────────────────────────────────────────────
try:
    from .block import LasmoidBlock
except ImportError:
    from block import LasmoidBlock  # type: ignore[import]

# ── Attention ───────────────────────────────────────────────────────────────
try:
    from .attention import (
        CSAAttention,
        HCAAttention,
        MLAAttention,
        precompute_freqs_cis,
    )
except ImportError:
    from attention import CSAAttention, HCAAttention, MLAAttention, precompute_freqs_cis

# ── MHC (Manifold Hyper-Connections) ────────────────────────────────────────
try:
    from .mhc import ManifoldConstrainedHyperConnection
except ImportError:
    from mhc import ManifoldConstrainedHyperConnection

# ── MoE ─────────────────────────────────────────────────────────────────────
try:
    from .moe import DeepSeekMoE, Gate
except ImportError:
    from moe import DeepSeekMoE, Gate

# ── SSM ─────────────────────────────────────────────────────────────────────
try:
    from .ssm import StateSpaceRecurrence
except ImportError:
    from ssm import StateSpaceRecurrence

# ── MTP (Multi-Token Prediction) ────────────────────────────────────────────
try:
    from .mtp import MTPBlock
except ImportError:
    from mtp import MTPBlock

# ── Main Model ──────────────────────────────────────────────────────────────
try:
    from .lasmoid import Lasmoid
except ImportError:
    from lasmoid import Lasmoid  # type: ignore[import]

# ── Loss Functions ──────────────────────────────────────────────────────────
try:
    from .loss import compute_loss, compute_grpo_loss
except ImportError:
    from loss import compute_loss, compute_grpo_loss  # type: ignore[import]

# ── Compressor ──────────────────────────────────────────────────────────────
try:
    from .compressor import Compressor
except ImportError:
    from compressor import Compressor  # type: ignore[import]

# ── KV Cache ────────────────────────────────────────────────────────────────
try:
    from .kv_cache import KVCache, SlidingWindowKVCache
except ImportError:
    from kv_cache import KVCache, SlidingWindowKVCache


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(" Lasmoid — Self-Test")
    print("=" * 60)

    args = ModelArgs()
    args.max_seq_len = 64
    args.max_batch_size = 2

    model = Lasmoid(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total:,}")

    x_enc = torch.randint(0, args.vocab_size, (2, 32))
    x_dec = torch.randint(0, args.vocab_size, (2, 32))

    logits, mtp_logits, c_db, mem, routings, idxs, adjs, event_probs = model(
        x_enc, x_dec
    )
    print(f"  logits       : {logits.shape}")
    print(f"  mtp_logits   : {mtp_logits.shape if mtp_logits is not None else None}")
    print(f"  concept_db   : {c_db.shape}")
    print(f"  memory       : {mem.shape}")
    print(f"  routings     : {[r.shape for r in routings]}")
    print(f"  adjs         : {[a.shape for a in adjs]}")
    print(
        f"  event_probs  : {len(event_probs)} layers"
        + (f", {event_probs[0].shape}" if event_probs else "")
    )

    # Test compute_loss with event_probs
    targets = torch.randint(0, args.vocab_size, (2, 32))
    loss = compute_loss(
        logits,
        targets,
        routings,
        [torch.tensor(0.0)] * len(routings),
        adjs,
        event_probs,
    )
    print(f"  total_loss   : {loss.item():.4f}")

    # Verify superhuman alignment head and attribute steering
    print(f"  predicted attributes: {model.last_predicted_attributes.shape}")
    assert model.last_predicted_attributes.shape == (
        2,
        32,
        len(args.steering_attributes),
    )

    steering_val = {"creativity": 1.5, "scientific_rigor": -0.5}
    logits_steered, *_ = model(x_enc, x_dec, steering_vector=steering_val)
    print(f"  steered logits       : {logits_steered.shape}")
    assert logits_steered.shape == logits.shape

    # Test generation with steering vector
    generated = model.generate(
        x_dec[:, :10], max_new_tokens=5, steering_vector=steering_val
    )
    print(f"  generated (steered)  : {generated.shape}")
    assert generated.shape == (2, 15)

    print("\nALL MODULES TESTED & READY ✓")
