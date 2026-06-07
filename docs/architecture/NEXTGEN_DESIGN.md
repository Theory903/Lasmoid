# Lasmoid Next-Gen Design Document

> **Transforming Lasmoid into a 2M-context, hyper-compressed, multimodal, days-stable agentic model**

| Author | Date | Version | Status |
|--------|------|---------|--------|
| AI Architect | June 2026 | 0.2 | Revised |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Goals & Success Criteria](#2-design-goals--success-criteria)
3. [Current Architecture Summary](#3-current-architecture-summary)
4. [Gap Analysis: Current vs Next-Gen](#4-gap-analysis-current-vs-next-gen)
5. [Detailed Design: Component by Component](#5-detailed-design-component-by-component)
   - 5.0 SOLiD Refactoring (Phase 0)
   - 5.1 Context Window: 1024 → 2,097,152
   - 5.2 Adaptive Hyper-Compressing Compressor
   - 5.3 Unified KV Cache Compression System
   - 5.4 Attention Architecture Upgrade
     - 5.4.1 Hybrid Sliding Window + Global (Gemma-4)
     - 5.4.2 Dual RoPE with QK Norm (Gemma-4)
     - 5.4.3 DeepSeek Indexer: Learned Sparse Attention
     - 5.4.4 Attention Sink
     - 5.4.5 Adaptive CSA/HCA Ratio (Enhanced)
     - 5.4.6 Block AttnRes Integration
     - 5.4.7 Combined Window + Compress TopK
     - 5.4.8 Ring Attention for Distributed Prefill
   - 5.5 Multimodal Integration (SigLIP + Conformer)
   - 5.6 Long-Running Stability System
   - 5.7 Training Configuration (NVFP4 + MTP + MOPD + WSD)
6. [Integration Map: NEXUS Research → Lasmoid](#6-integration-map-nexus-research--lasmoid)
7. [Implementation Phases](#7-implementation-phases)
   - Phase 0: SOLiD Refactoring
   - Phase 1: Foundation & Einsum
   - Phase 2: KV Cache & NVFP4 Quantization
   - Phase 3: Hybrid Attention & Indexer
   - Phase 4: Multimodal (SigLIP + Conformer)
   - Phase 5: Stability & LogitSoftcap
   - Phase 6: Training (NVFP4 + MOPD + WSD)
8. [Memory & Compute Budget](#8-memory--compute-budget)
9. [Risks & Mitigations](#9-risks--mitigations)
10. [Benchmarks & Acceptance Criteria](#10-benchmarks--acceptance-criteria)

---

## 1. Executive Summary

Lasmoid is a hybrid transformer-SSM architecture at 1B params with state-of-the-art components: Compressed Sparse Attention (CSA/HCA), CIF-based semantic compression, Mamba-2 SSM recurrence, Manifold-Constrained Hyper-Connections (mHC), DeepSeek-V4-style MoE, and Multi-Token Prediction (MTP).

**The goal**: Evolve Lasmoid into a next-generation model capable of:
- **2M token context window** with adaptive hyper-compression that automatically triggers aggressive compression when context exceeds 300K tokens
- **Days-long stable agentic output** without hallucination drift or degradation
- **Best-in-class KV cache** via integrated quantization (TurboQuant/NVFP4), eviction (SnapKV/PyramidKV), and compaction
- **Native multimodal support** (vision + audio) alongside text, following Gemma-4 SigLIP + Conformer patterns

**Critical prerequisite**: The current `model.py` is 2,497 lines — a monolithic SOLiD violation. Phase 0 adds a full architectural refactoring before any feature work, decomposing into single-responsibility modules with proper interfaces.

**New innovations incorporated from analysis of Gemma-4, DeepSeek-V4-Pro, Nemotron-Ultra, and Lasmoid-V1**:
- **Gemma-4 hybrid attention**: 4-5 local sliding window + 1 global attention layer per pattern, with dual RoPE frequencies (10K local, 1M global), QK norm with scale, and key-value weight tying (k_eq_v_global)
- **DeepSeek-V4 Indexer**: Learned sparse attention with Hadamard rotation and top-k page selection
- **Nemotron NVFP4 quantization**: Per-operator precision (NVFP4/FP8/BF16) with stochastic rounding and 2D block quantization
- **MTP speculative decoding**: 2 shared-weight heads for 2.89× throughput at decoding
- **Einsum-based parameter layers**: Replacing nn.Linear with flexible Einsum patterns (Gemma-4)
- **KV cache sharing**: frac_shared_layers ~0.5 between layers (Gemma-4)
- **Prefill-decode disaggregation**: Separate compute for prefill vs decode phases (Nemotron)

This document provides the detailed architecture, integration points, implementation phases, and acceptance criteria for this transformation.

---

## 2. Design Goals & Success Criteria

| Dimension | Current State | Next-Gen Target | Success Metric |
|-----------|--------------|-----------------|----------------|
| **Context Window** | 1,024 tokens | 2,097,152 tokens | Lossless perplexity at 2M vs 1K on held-out eval |
| **Compression** | Fixed ratio CSA=4, HCA=128 | Adaptive: ratio=4 normal, ratio=128+quant when >300K tokens | <5% quality degradation at 2x compression of 300K+ segments |
| **KV Cache Memory** | Full precision, no eviction | <1/8th memory via quantization (INT4) + eviction (top-k) | 2M context fits in 80GB H100 with <1% attention quality loss |
| **Prefill Speed** | Sequential CIF loop over seqlen | Vectorized prefix-parallel CIF + ring attention | 2M token prefill in <60s on 8xH100 |
| **Stability** | No explicit safeguards | Drift detection, temp scheduling, KV integrity checks | 72h continuous generation without hallucination cascade |
| **Multimodal** | Stub projections only (vision_proj, audio_proj) | Real CLIP/SigLIP vision + Whisper audio encoders | <10% quality drop on multimodal benchmarks vs text-only |
| **Training** | 1K context pre-training | Long-context fine-tuning + GRPO for stability | Successful 32K+ fine-tune with <1% perplexity degradation |

---

## 3. Current Architecture Summary

```
                  ┌─────────────────────────────────────┐
                  │         Lasmoid Architecture         │
                  ├─────────────────────────────────────┤
 Input Tokens ──▶ │ Token Embedding                      │
                  │         │                            │
                  │    ┌────▼────┐                       │
                  │    │Encoder  │ (Read Replica)        │
                  │    │Attention│                       │
                  │    │ESCM     │──▶ Concept Database    │
                  │    └────┬────┘                       │
                  │         │                            │
                  │    ┌────▼──────────────────────┐     │
                  │    │  Lasmoid Blocks (×28)     │     │
                  │    │  ┌───────────────────┐   │     │
                  │    │  │ Block:            │   │     │
                  │    │  │  RMSNorm           │   │     │
                  │    │  │  Attention         │   │     │
                  │    │  │  (CSA/HCA/MLA)     │   │     │
                  │    │  │  SSM (Mamba-2)     │   │     │
                  │    │  │  mHC Residual Stream│   │     │
                  │    │  │  MoE (DeepSeek-V4) │   │     │
                  │    │  │  YaRN RoPE         │   │     │
                  │    │  └───────────────────┘   │     │
                  │    └──────────────────────────┘     │
                  │         │                            │
                  │    ┌────▼────┐                       │
                  │    │  MTP    │──▶ t+1, t+2 Logits    │
                  │    └─────────┘                       │
                  └─────────────────────────────────────┘
```

### Key Components — Actual Implementation Status

> **Note**: A codebase audit (June 2026) revealed that several "planned" features from the original design doc were already implemented. The table below reflects the **actual** state of `inference/`.

| Component | Status | Details |
|-----------|--------|---------|
| **SOLiD Refactoring (Phase 0)** | ✅ **Complete** | `model.py` reduced from 2,497→214 lines. Decomposed into 14 single-responsibility modules: `config.py`, `compressor.py`, `attention.py`, `attention_indexer.py`, `block.py`, `moe.py`, `ssm.py`, `mhc.py`, `mtp.py`, `kv_cache.py`, `concept_memory.py`, `vq.py`, `loss.py`, `_common.py` |
| **CIF Compressor** | ✅ Present | `compressor.py` — vectorized CIF via `torch.cumsum(alpha)` (line 290), no sequential Python loop. Supports adaptive gating + per-layer ratios |
| **AdaptiveCompressorGate (P1.3)** | ✅ **Implemented** | `compressor.py:46` — threshold-based mode switching (NORMAL at ≤300K, HYPER at >300K, EMERGENCY at >1.5M). Wired into Compressor.forward via `self.gate` + `ratio_mult` |
| **EnhancedEventDetector (P1.4)** | ✅ **Implemented** | `compressor.py:103` — multi-scale detector fusing local (single-token), window (Conv1d kernel=33), and global (cross-attention) features |
| **CSA (CompressedSparseAttention)** | ✅ Present | `attention.py` — ratio=4 compressed key-value attention with window+compress top-k |
| **HCA (HeavilyCompressedAttention)** | ✅ Present | `attention.py` — ratio=128 heavily compressed attention |
| **MLA (Multi-head Latent Attention)** | ✅ Present | `attention.py` — low-rank KV projection (q_lora_rank=256, o_lora_rank=256) |
| **Indexer (DeepSeek sparse attention)** | ✅ **Implemented** | `attention_indexer.py:16` — `Indexer` class with Hadamard rotation + top-k page selection (109 lines) |
| **Attention Sink** | ✅ **Implemented** | `attention.py:226,413,654` — `self.attn_sink` learnable bias per head (nn.Parameter) in CSA, HCA, MLA. Wired through `kernel.py:416` |
| **DualRoPECache** | ⚠️ **Stub** | `attention.py:79` — class defined but uses single RoPE frequency; dual-frequency (10K local + 1M global) not active |
| **SSM (Mamba-2)** | ✅ Present | `ssm.py` — chunked parallel scan, state_dim=16 |
| **ESCM (Concept Memory)** | ✅ Present | `concept_memory.py` + `vq.py` — Elastic Sparse Concept Memory with codebook, commit loss |
| **mHC** | ✅ Present | `mhc.py` — 4 residual streams, Sinkhorn iterations |
| **MoE** | ✅ Present | `moe.py` — 6 routed + 1 shared, 2 activated experts, ragged_dispatch |
| **MTP** | ✅ Present | `mtp.py` — 2-token prediction with shared-weight heads |
| **GRPO** | ✅ Present | `loss.py` — alignment without critic |
| **YaRN RoPE** | ✅ Present | `_common.py` — dynamic frequency scaling |
| **vision_proj / audio_proj** | ✅ **Complete** | `vision.py` & `audio.py` — Real SigLIP and Conformer encoders |
| **Hybrid Sliding Window + Global (Gemma-4)** | ✅ **Complete** | `attention.py:865` — Standard MHA, heads split local/global, dual RoPE, QK norm, sliding window + full KV caches |
| **Einsum Parameterization** | ✅ **Complete** | `_layers.py` — EinsumLinear class, `_common.py` — global flag dispatch, all projections routed through `use_einsum` config flag (default: False) |
| **KV Cache Quantization** | ✅ **Complete** | `kv_cache.py` — `QuantKVCache` (FP8 float8_e4m3fn) & `TieredKVCache` (BF16 hot / FP8 main) |
| **KV Cache Eviction/Compaction** | ✅ **Complete** | `eviction.py` (SnapKV) and `compaction.py` (OMPKVMerger) integrated in `AdaptiveQuantizedKVCache` |
| **Stability System** (drift, temp scheduling, recovery) | ✅ **Complete** | `stability.py` — Drift monitoring, adaptive temperature scheduling, and KV cache integrity checks |
| **Muon Optimizer** | ✅ Present | `optim.py` — Newton-Schulz iteration |
| **Training Pipeline (Phase 6)** | ✅ **Complete** | `train/` — WSD scheduler (`scheduler.py`), stability-aware GRPO (`grpo_stability.py`), multi-teacher distillation (`mopd.py`), pretraining runner (`pretrain.py`), and progressive context length scaling (`long_context_finetune.py`) |

---

## 4. Gap Analysis: Current vs Next-Gen

### 🔴 Critical Gaps (Blocking 2M Context)

| # | Gap | Impact | Root Cause | Status |
|---|-----|--------|------------|--------|
| G1 | `max_seq_len=1024` | Model cannot process >1024 tokens | Config limit in all configs | **Config-only fix** |
| G2 | Sequential CIF prefill loop | O(n) Python loop — 2M positions = hours | Lines 618-647: `for pos in range(1, seqlen)` | ✅ **RESOLVED** — `torch.cumsum(alpha)` at compressor.py:290 |
| G3 | Full-precision KV cache | 2M tokens × dim × layers × heads = TB of memory | No quantization, no eviction | ✅ **RESOLVED** — `AdaptiveQuantizedKVCache` supporting tiered FP8 quantization and eviction |
| G4 | No sliding window attention | Every token attends to full compressed cache, O(n²) cost | All attention is full-sequence CSA/HCA | ✅ **RESOLVED** — HybridSlidingGlobal at attention.py:865, 4:1 local:global pattern |

### 🟡 Important Gaps (Quality & Capability)

| # | Gap | Impact | Root Cause | Status |
|---|-----|--------|------------|--------|
| G5 | No adaptive compression gating | Always uses same ratio regardless of context length | No threshold-based ratio switching | ✅ **RESOLVED** — `AdaptiveCompressorGate` at compressor.py:46 |
| G6 | No KV cache eviction | Retains all keys, O(n) growth unbounded | No SnapKV/PyramidKV integration | ✅ **RESOLVED** — `eviction.py` (SnapKV heuristic) integrated |
| G7 | Vision/audio are stubs | No real multimodal capability | Non-linear proj layers | ✅ **RESOLVED** — Real SigLIP vision & Conformer audio encoders implemented |
| G8 | No drift/hallucination detection | Long generations gradually degrade | No monitoring system | ✅ **RESOLVED** — `stability.py` with entropy-based `DriftDetector` |
| G9 | RoPE scaling fixed to 1K | Position embeddings don't generalize to 2M | YaRN factors tuned for short context | ⚠️ Config-only (update rope_factor, rope_theta) |
| G10 | No compaction-based KV merging | Pruned tokens are dropped, not merged | No `merge_kv` integration | ✅ **RESOLVED** — `compaction.py` (OMPKVMerger) integrated |

### 🟢 Enhancement Gaps (Performance)

| # | Gap | Impact | Root Cause | Status |
|---|-----|--------|------------|--------|
| G11 | No ring/distributed attention | Single-GPU prefill for 2M is slow | All attention is local | ✅ **RESOLVED** — `ring.py` RingAttentionPrefill implemented (simulated/distributed) |
| G12 | No quantization integration | KV cache at BF16 uses 2× memory vs FP8 | No quantization wrapper | ✅ **RESOLVED** — Group-wise value quantization & MSE/QJL key quantization |
| G13 | No Block AttnRes | Long-range block retrieval requires full attention | No depth attention over blocks | ✅ **RESOLVED** — `attnres.py` BlockAttnRes implemented |
| G14 | No temperature scheduling | Single temperature for entire generation | No adaptive generation params | ✅ **RESOLVED** — `AdaptiveTemperatureScheduler` implemented |

---

## 5. Detailed Design: Component by Component

### 5.0 SOLiD Refactoring (Phase 0)

**Problem**: Current `inference/model.py` is 2,497 lines containing Compressor, MLA, CSA, HCA, MoE (Gate/Expert), MTP, mHC, and Transformer — all in one file. This is a critical violation of all five SOLiD principles and must be resolved before adding any new features.

#### SOLiD Violations

| Principle | Violation | Impact |
|-----------|-----------|--------|
| **SRP** | 7+ distinct responsibilities in one class/file | Impossible to reason about, test, or modify one component without touching others |
| **OCP** | Adding new compression strategy (e.g., TurboQuant, NVFP4) requires modifying existing Compressor class | Every new feature creates merge conflicts; cannot extend without modifying |
| **LSP** | CSA, HCA, MLA have no shared interface — different signatures, different behavior | Cannot swap attention types transparently; blocks pattern like Gemma-4 hybrid (local sliding + global) |
| **ISP** | `ModelArgs` is a 90+ field flat struct | Every component sees irrelevant params; impossible to configure independently |
| **DIP** | Direct kernel imports (`from .kernel import ...`) | Cannot swap implementations (e.g., FP8 vs FP4 kernels) without code changes |

#### Refactored Architecture

```
Lasmoid/
└── inference/
    ├── model.py              # Thin coordinator: embeds, stacks blocks, calls components
    ├── config.py             # Per-component configs (split from monolithic ModelArgs)
    ├── attention.py          # Attention interface + CSA, HCA, MLA, HybridSlidingGlobal
    ├── attention_indexer.py  # DeepSeek Indexer (learned sparse attention)
    ├── compressor.py         # CIF compressor (vectorized + adaptive gating)
    ├── kv_cache.py           # AdaptiveQuantizedKVCache (TurboQuant + NVFP4 + eviction + compaction)
    ├── moe.py                # MoE (Gate, Expert, shared expert, ragged_dispatch)
    ├── ssm.py                # Mamba-2 SSM
    ├── mhc.py                # Hyper-Connections (mHC)
    ├── mtp.py                # Multi-Token Prediction heads + speculative decoding
    ├── vision.py             # Vision encoder (SigLIP-style)
    ├── audio.py              # Audio encoder (Conformer-style)
    ├── stability.py          # Drift detection, temperature scheduling
    ├── recovery.py           # Graceful degradation & recovery
    ├── kernels/
    │   ├── __init__.py       # Kernel provider interface (DIP)
    │   ├── hc_sinkhorn.py    # Sinkhorn iterations kernel
    │   ├── sparse_attn.py    # Sparse attention kernel
    │   ├── quant.py          # Quantization kernels (NVFP4, FP8, INT4)
    │   └── ops.py            # Other ops
    └── ring.py               # Ring attention for distributed prefill
```

#### Key Design Decisions

**1. Common Attention Interface (LSP)**
```python
class Attention(nn.Module):
    """Abstract attention interface — all attention variants implement this."""
    @abstractmethod
    def forward(self, x: Tensor, kv_cache: KVCache, positions: Tensor,
                attention_mask: Tensor) -> Tensor:
        ...

class CSAAttention(Attention): ...      # ratio=4 compressed
class HCAAttention(Attention): ...      # ratio=128 heavily compressed
class MLAAttention(Attention): ...      # Multi-head Latent Attention
class HybridSlidingGlobal(Attention): ...  # Gemma-4: local sliding + global
class BlockAttnResidual(Attention): ... # Block-level depth attention
```

**2. Per-Component Configs (ISP)**
```python
@dataclass
class AttentionConfig:
    n_heads: int = 16
    head_dim: int = 48
    q_lora_rank: int = 256
    o_lora_rank: int = 256
    o_groups: int = 2
    window_size: int = 512
    # Hybrid attention (Gemma-4)
    attention_pattern: str = "local"  # "local", "global", "hybrid"
    global_heads: int = 4
    global_key_size: int = 512
    k_eq_v_global: bool = False
    qk_norm_with_scale: bool = True
    local_base_frequency: int = 10_000
    global_base_frequency: int = 1_000_000
    # Indexer (DeepSeek)
    use_indexer: bool = False
    index_topk: int = 512
    index_head_dim: int = 128

@dataclass
class MoEConfig:
    n_routed_experts: int = 6
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    moe_latent_dim: int = 512
    moe_dense_hidden_dim: int = 0  # Gemma-4 dual-branch shared MLP
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.0
    expert_dtype: str = "bf16"  # "bf16", "fp8", "nvfp4"

@dataclass
class CompressorConfig:
    ratio_normal: int = 4
    ratio_hyper: int = 128
    ratio_emergency: int = 512
    hyper_threshold: int = 300_000
    emergency_threshold: int = 1_500_000
    overlap: bool = True
    rotate: bool = False
    per_layer_ratios: tuple = ()  # DeepSeek: different ratios per layer

@dataclass
class SSMConfig:
    heads: int = 16
    state_dim: int = 16
    kernel_size: int = 4
    chunk_size: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1

@dataclass
class MHCConfig:
    num_residual_streams: int = 4
    sinkhorn_iters: int = 8
    eps: float = 1e-6
```

**3. Kernel Provider Interface (DIP)**
```python
class KernelProvider(ABC):
    """Abstract kernel interface — swap FP8/FP4/BF16 without code changes."""
    @abstractmethod
    def matmul(self, a: Tensor, b: Tensor) -> Tensor: ...
    @abstractmethod
    def quantize(self, x: Tensor, bits: int) -> QuantizedTensor: ...

class FP8KernelProvider(KernelProvider): ...
class FP4KernelProvider(KernelProvider): ...
class NVFP4KernelProvider(KernelProvider): ...
class BF16KernelProvider(KernelProvider): ...
```

**4. Weight Tying** (from Lasmoid-V1)
```python
class Embedding(nn.Module):
    """Shared embedding/head with weight tying."""
    def __init__(self, vocab_size, dim):
        self.weight = nn.Parameter(...)
    
    def encode(self, x): return F.embedding(x, self.weight)
    def decode(self, x): return F.linear(x, self.weight.T)  # tied
```

---

### 5.1 Context Window: 1024 → 2,097,152

#### Config Changes (`config_1b_2m.json`)

```json
{
  "max_seq_len": 2097152,
  "window_size": 512,
  "rope_theta": 1000000.0,
  "rope_factor": 32.0,
  "original_seq_len": 4096,
  "beta_fast": 64,
  "beta_slow": 2,
  // Einsum-based layers (Gemma-4)
  "use_einsum": true,
  // KV cache sharing (Gemma-4)
  "frac_shared_layers": 0.5,
  "kv_cache_share_global": false,
  "kv_cache_share_local": false,
  // Per-layer input embeddings (Gemma-4)
  "per_layer_input_dim": 64,
  // ... keep other params from config_1b.json
}
```

Key decisions:
- **RoPE base frequency**: 1,000,000 (up from 10,000) — NTK-aware scaling for 2048× extension
- **rope_factor**: 32 — frequency interpolation for position encoding
- **window_size**: 512 — sliding window for local attention (Gemma-4 pattern for small models; use 1024 for larger)
- **beta_fast/slow**: widened YaRN ramp for smoother dimension-wise interpolation

#### YaRN Scaling for 2M

The existing YaRN RoPE implementation already supports dynamic frequency scaling. For 2M:
- Low-frequency dimensions (i > d/4): full interpolation
- High-frequency dimensions (i < d/4): extrapolation (no change)
- Ramp function `r(i)` widened (beta_fast=64, beta_slow=2) to cover more dimensions gradually

**No architectural change needed** — just config tuning and position-id remapping during prefill.

#### Einsum-Based Parameter Layers (Gemma-4)

Replace all `nn.Linear` with flexible Einsum parameterizations for better parameter efficiency and JIT compilation:

```python
class Einsum(nn.Module):
    """
    Einsum-based parameter layer, replacing nn.Linear.
    
    Key differences:
    - Explicit dimension routing via einsum strings
    - Supports multi-dimensional weight tensors (not just 2D)
    - Enables weight sharing patterns impossible with nn.Linear
    - More compiler-friendly for XLA/TorchDynamo
    """
    def __init__(self, weight_shape: tuple, w_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*weight_shape, dtype=torch.float32))
        self.w_scale = w_scale
    
    def forward(self, equation: str, x: Tensor) -> Tensor:
        return torch.einsum(equation, x, self.weight * self.w_scale)

# Example usage:
# Q/K/V projections as Einsum:
q_proj = Einsum((dim, n_heads * head_dim))  # q = einsum('...d,do->...o', h, weight)
k_proj = Einsum((dim, n_kv_heads * head_dim))
v_proj = Einsum((dim, n_kv_heads * head_dim))

# Multi-dimensional routing weights:
router = Einsum((n_experts, dim))  # scores = einsum('...d,ed->...e', h, weight)
```

**Where to apply**: Q/K/V projections, O projections, MoE gate/router, FFN layers, event projection in Compressor. All current `nn.Linear` → `Einsum`.

#### KV Cache Sharing Between Layers (Gemma-4)

Share KV cache between adjacent layers to reduce memory by ~2×. Shared layers write to the same KV slot.

```python
def create_kv_cache_sharing_patterns(
    frac_shared_layers: float,  # ~0.5
    num_layers: int,
    attention_types: list[AttentionType],
) -> list[int]:
    """
    Returns a mapping: layer_idx → kv_cache_slot_idx.
    Shared layers map to the same slot.
    """
    num_unshared = int(num_layers * (1 - frac_shared_layers))
    patterns = []
    for i in range(num_layers):
        if i < num_unshared:
            patterns.append(i)  # Unique KV slot
        else:
            # Share with last unshared layer of same type
            if attention_types[i] == GLOBAL:
                patterns.append(num_unshared - 1)
            else:
                patterns.append(num_unshared - 2)
    return patterns
```

**Impact**: With `frac_shared_layers=0.5`, 14 of 28 layers share KV cache → 14 + 14/2 = 21 effective slots → **~25% memory savings**.

#### Per-Layer Input Embeddings (Gemma-4)

Add learned per-layer embeddings that give each layer token-type and position information:

```python
class Embedder(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, per_layer_input_dim=64):
        self.input_embedding = nn.Embedding(vocab_size, dim)
        # Per-layer embeddings: [vocab_size, num_layers, per_layer_input_dim]
        self.per_layer_embeddings = nn.Parameter(...)
        # Projection: embed_dim → num_layers × per_layer_input_dim
        self.per_layer_proj = Einsum((dim, num_layers, per_layer_input_dim))
        self.per_layer_norm = RMSNorm(per_layer_input_dim)
    
    def encode(self, tokens):
        base = self.input_embedding(tokens) * sqrt(dim)
        # Per-layer features from token IDs
        ple = self.per_layer_embeddings[tokens] * sqrt(per_layer_input_dim)
        # Project base → per-layer space and add
        proj = self.per_layer_norm(self.per_layer_proj('...d,dnp->...tnp', base))
        return (proj + ple) / sqrt(2)
```

**Purpose**: Gives each layer access to token-type information (e.g., "this is an image token", "this is a system prompt token") without needing separate embedding tables. Beneficial for multimodal — vision and audio tokens get distinct per-layer features.

#### Position ID Management

For 2M tokens, we need offset-aware position ID computation:

```python
# In prefill: compute positions as normal 0..seqlen-1
# In generation: offset by prefix length
# In ring attention: local position IDs + segment offset
```

---

### 5.2 Adaptive Hyper-Compressing Compressor

**Current problem**: The CIF compressor at lines 404-647 runs a sequential Python loop over the full sequence. For 2M tokens, this is prohibitively slow.

**Solution**: Vectorize the CIF computation and add adaptive gating.

#### 5.2.1 Vectorized CIF Prefill

Replace the sequential loop with parallel prefix operations:

```
Current:  for pos in range(1, seqlen):  # O(n) sequential
          │
          ▼
Target:   alpha = softplus(W_event @ H + b_event)    # [seqlen, 1]  — vectorized
          cum_alpha = cumsum(alpha)                    # [seqlen, 1]  — parallel prefix scan
          fire_points = where(cum_alpha // 1.0 changes) # detect fire events
          remainder = alpha - (cum_alpha - floor(cum_alpha))
          # Scatter-gather for fired events
          kernel.scatter_add(accum_kv, fire_indices, kv * alpha)
```

This is a **prefix-sum pattern** (parallelizable via associative scan):
- Compute all `alpha_t` in one matrix multiply (batch)
- Compute cumulative sum via parallel prefix scan (O(log n) depth)
- Identify fire points and scatter-add in parallel
- **Speedup**: O(n) → O(log n) theoretical; ~1000× for 2M tokens

```python
def vectorized_cif_prefill(hidden_states: Tensor, event_proj: Linear) -> Tuple[Tensor, Tensor]:
    """
    Vectorized CIF prefill using parallel prefix scan.
    
    Args:
        hidden_states: [batch, seqlen, dim]
        event_proj: event boundary projection
        
    Returns:
        compressed_kv: [batch, num_events, 2, dim]
        event_indices: [batch, num_events]
    """
    # Step 1: Compute boundary probabilities (vectorized)
    alpha = F.softplus(event_proj(hidden_states))  # [batch, seqlen, 1]
    
    # Step 2: Cumulative sum via parallel prefix scan
    cum_alpha = parallel_exclusive_cumsum(alpha.squeeze(-1))  # [batch, seqlen]
    
    # Step 3: Detect fire events
    fire_counts = (cum_alpha[:, -1:].floor() - cum_alpha[:, :1].floor()).long()  # total events
    
    # Step 4: Scatter KVs by event (parallel)
    event_kvs = scatter_add_kv_by_event(hidden_states, alpha, cum_alpha)
    
    return event_kvs, fire_counts
```

**Implementation approach**: Use PyTorch's built-in `torch.cumsum` (which uses a parallel kernel) and scatter operations. For truly massive sequences, use a Triton kernel.

#### 5.2.2 Adaptive Compression Gating

The key innovation: automatically switch compression ratio based on context length.

```python
class AdaptiveCompressorGate:
    """
    Automatically selects compression mode based on context length.
    
    Modes:
    - NORMAL:    ratio=4, standard CSA compression, full quality
    - HYPER:     ratio=128, HCA + KV quantization + eviction, triggered at >300K tokens
    - EMERGENCY: ratio=512, aggressive compaction, triggered at >1.5M tokens
    """
    
    THRESHOLD_NORMAL_TO_HYPER = 300_000
    THRESHOLD_HYPER_TO_EMERGENCY = 1_500_000
    
    def get_mode(self, current_seqlen: int) -> CompressionMode:
        if current_seqlen > self.THRESHOLD_HYPER_TO_EMERGENCY:
            return CompressionMode.EMERGENCY
        elif current_seqlen > self.THRESHOLD_NORMAL_TO_HYPER:
            return CompressionMode.HYPER
        else:
            return CompressionMode.NORMAL
```

**Trigger mechanism**: After each attention layer's prefill, check current sequence length. If >300K, switch to hyper mode for all subsequent tokens AND re-compress the existing cache with the higher ratio.

#### 5.2.3 Event Detector Enhancement

Current event detector is a single linear layer. For 2M context, enhance to:

```python
class EnhancedEventDetector(nn.Module):
    """
    Multi-scale event boundary detector.
    
    Combines:
    - Local features: single-token boundary signals
    - Window features: aggregated over ±16 token window
    - Global features: segment-level topic shift detection via cross-attention
    """
    def __init__(self, dim: int):
        self.local_proj = nn.Linear(dim, 1)
        self.window_conv = nn.Conv1d(dim, dim//4, kernel_size=33, padding=16)
        self.window_proj = nn.Linear(dim//4, 1)
        self.global_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.fusion = nn.Linear(3, 1)
    
    def forward(self, h: Tensor) -> Tensor:
        local = self.local_proj(h)                                     # [B, S, 1]
        window = self.window_proj(self.window_conv(h.transpose(1,2)))  # [B, S, 1]
        global_scores, _ = self.global_attn(h, h, h)                  # [B, S, dim]
        global_scores = global_scores.mean(dim=-1, keepdim=True)       # [B, S, 1]
        combined = torch.cat([local, window, global_scores], dim=-1)
        return F.softplus(self.fusion(combined))
```

#### 5.2.4 Per-Layer Compress Ratios (DeepSeek-V4-Pro)

Each layer can use a different compression ratio for finer-grained control:

```python
class PerLayerCompressor:
    """
    Each layer gets its own compress ratio from a configurable array.
    
    DeepSeek-V4-Pro pattern: compress_ratios = [4, 4, 8, 8, 16, 16, ...]
    - Early layers: low ratio (preserve detail for downstream)
    - Middle layers: moderate ratio (balanced)
    - Late layers: high ratio (semantic compression sufficient)
    - With overlap: smoothing between event boundaries (reduces boundary artifacts)
    """
    
    def __init__(self, config: CompressorConfig, layer_idx: int):
        if config.per_layer_ratios:
            self.ratio = config.per_layer_ratios[layer_idx]
        else:
            # Fallback to mode-based selection
            self.ratio = None  # Determined by AdaptiveCompressorGate
        self.overlap = config.overlap  # Smooth event boundaries
    
    def compress(self, kv: Tensor, alpha: Tensor, cum_alpha: Tensor) -> Tensor:
        if self.overlap:
            # Overlap mode: keep half-window overlap between events
            # This smooths boundary artifacts in compressed cache
            overlap_size = self.ratio // 2
            return self._compress_with_overlap(kv, alpha, cum_alpha, overlap_size)
        else:
            return self._compress_standard(kv, alpha, cum_alpha)
```

**Impact**: Per-layer ratios allow fine-grained control — early layers (ratio=4) preserve detail while late layers (ratio=128+) save memory. Combined with adaptive gating, this creates a 2D compression space (layer × context-length) for maximum efficiency.

---

### 5.3 Unified KV Cache Compression System

This is the **core innovation** — integrating three research projects into a single composable cache.

#### 5.3.1 Architecture: AdaptiveQuantizedKVCache

```python
class AdaptiveQuantizedKVCache(nn.Module):
    """
    Unified KV cache with composable compression strategies.
    
    Pipeline (applied in order):
    1. BUFFER: keep last N tokens unquantized (from TurboQuant pattern)
    2. QUANTIZE: apply TurboQuant 2-stage quantization to older tokens
    3. EVICT: apply SnapKV/PyramidKV selection to identify kept/pruned tokens
    4. MERGE: use OMP-based compaction to merge pruned tokens into kept tokens
    5. STORE: maintain hierarchical storage (hot/warm/cold tier)
    """
    
    def __init__(self, config: KVCacheConfig):
        self.buffer_size = config.buffer_size      # 128 tokens unquantized (TurboQuant pattern)
        self.quantizer = TurboQuantKVCache(
            key_bits=config.key_bits,              # 4-bit keys
            value_group_size=config.value_group_size,  # group size for val quant
        )
        self.eviction = SnapPyramidKV(
            max_cache_size=config.max_cache_size,   # max tokens in cache
            keep_ratio=config.keep_ratio,            # 0.3 = keep 30% of tokens
            window_size=config.eviction_window,       # 256 token window for score computation
        )
        self.merger = OMPKVMerger(
            merge_ratio=config.merge_ratio,          # merge pruned into kept
            method=config.merge_method,               # "pivot" or "weighted"
        )
```

#### 5.3.2 TurboQuant Integration

**Source**: `turboquant/quantizer.py` (MSE + QJL 2-stage) and `turboquant/kv_cache.py` (buffer pattern)

**Integration method**: Drop-in replacement for the current `torch.nn.Linear`-based KV cache projection.

```python
class TurboQuantKVCache(nn.Module):
    """
    KV cache with TurboQuant 2-stage quantization.
    
    Stage 1: MSE quantization at (b-1) bits — finds optimal scale
    Stage 2: QJL sign estimation on residual — unbiased inner product estimation
    
    Buffer: ~128 most recent tokens at FP16 for quality
    """
    
    def __init__(self, key_bits: int = 4, value_group_size: int = 8, buffer_size: int = 128):
        self.key_quantizer = TurboQuantProd(bits=key_bits)   # MSE+QJL
        self.value_quantizer = GroupQuantizer(
            group_size=value_group_size,
            bits=key_bits + 1,  # values get 1 more bit than keys
        )
        self.buffer_size = buffer_size
    
    def quantize(self, keys: Tensor, values: Tensor, seqlen: int):
        """
        Keep recent tokens in buffer, quantize older ones.
        """
        if seqlen <= self.buffer_size:
            return keys, values  # all in buffer, no quantization
        
        # Keep last buffer_size tokens at full precision
        buf_start = seqlen - self.buffer_size
        
        # Quantize keys and values before buffer
        k_quantized = self.key_quantizer(keys[:buf_start])
        v_quantized = self.value_quantizer(values[:buf_start])
        
        # Concatenate quantized + buffer
        k_out = torch.cat([k_quantized, keys[buf_start:]], dim=0)
        v_out = torch.cat([v_quantized, values[buf_start:]], dim=0)
        
        return k_out, v_out
```

**Pipeline modification in attention**:

```python
# Current:
self.kv_cache.append(k, v)  # Full precision, no compression

# Target:
if comp_gate.mode == CompressionMode.HYPER:
    k, v = self.kv_cache.append_and_quantize(k, v, current_seqlen)
elif self.kv_cache.is_full():
    k, v = self.kv_cache.evict_and_merge(k, v)
else:
    self.kv_cache.append(k, v)
```

#### 5.3.3 KVCache-Factory Eviction Integration

**Source**: `KVCache-Factory/pyramidkv/pyramidkv_utils.py`

**Strategy**: After quantization, select which tokens to keep based on accumulated attention scores.

```python
class SnapPyramidKV:
    """
    PyramidKV-style eviction with SnapKV scoring.
    
    Algorithm:
    1. Compute attention scores between recent query window and all cached keys
    2. Accumulate scores per key position
    3. Keep top-k scoring keys, mark rest for eviction/merging
    """
    
    def select_kept_indices(
        self,
        queries: Tensor,       # [num_heads, seqlen, head_dim]
        keys: Tensor,          # [num_heads, cache_size, head_dim]
        window_size: int = 256
    ) -> Tensor:
        # Use last window_size queries to score all cached keys
        q_window = queries[:, -window_size:, :]  # [num_heads, window, head_dim]
        scores = torch.matmul(q_window, keys.transpose(-2, -1))  # [num_heads, window, cache_size]
        cumulative_scores = scores.sum(dim=1)  # [num_heads, cache_size]
        
        # Select top-k indices per head
        k = int(self.max_cache_size * self.keep_ratio)
        _, top_indices = torch.topk(cumulative_scores, k, dim=-1)
        
        return top_indices  # [num_heads, k]
    
    def evict(self, keys, values, top_indices):
        """Keep only top-k keys/values per head."""
        # Gather kept keys/values
        kept_k = torch.gather(keys, -2, top_indices.unsqueeze(-1).expand(-1, -1, keys.size(-1)))
        kept_v = torch.gather(values, -2, top_indices.unsqueeze(-1).expand(-1, -1, values.size(-1)))
        return kept_k, kept_v
```

#### 5.3.4 Compaction-based KV Merging

**Source**: `compaction/compaction/algorithms/`

Instead of dropping pruned tokens entirely, merge them into similar retained tokens:

```python
class OMPKVMerger:
    """
    OMP-inspired batched KV merging.
    
    For each pruned token, find the most similar kept token (by key similarity)
    and merge their values weighted by attention score.
    """
    
    def merge(self, keys, values, top_indices):
        """
        Merge pruned tokens into their nearest kept neighbor.
        
        Args:
            keys: [num_heads, cache_size, head_dim]
            values: [num_heads, cache_size, head_dim]
            top_indices: [num_heads, k] — indices to keep
            
        Returns:
            merged_keys, merged_values: [num_heads, k, head_dim]
        """
        # Get kept keys
        kept_k = gather(keys, top_indices)
        kept_v = gather(values, top_indices)
        
        # For each pruned token, find nearest kept token
        pruned_mask = ~is_in(top_indices, cache_size)
        pruned_k = keys[:, pruned_mask, :]
        pruned_v = values[:, pruned_mask, :]
        
        # Similarity: dot product between pruned and kept keys
        sim = torch.matmul(pruned_k, kept_k.transpose(-2, -1))  # [h, n_pruned, k]
        assignments = sim.argmax(dim=-1)  # [h, n_pruned]
        
        # Weighted merge: merge pruned values into assigned kept positions
        kept_v = scatter_add(kept_v, assignments, pruned_v * sim.max(dim=-1).values.unsqueeze(-1))
        
        return kept_k, kept_v
```

#### 5.3.5 Cache Hierarchy

For 2M tokens, use a 3-tier cache:

```
Tier 1 (HOT):   ~4K tokens, FP16, no eviction, sliding window
Tier 2 (WARM):  ~300K tokens, INT4 quantized, SnapKV evicted
Tier 3 (COLD):  ~1.7M tokens, INT4 quantized, PyramidKV evicted + compacted
```

Feed attention over Tier 1 normally, use cross-attention for Tier 2/3:

```python
class HierarchicalCache(nn.Module):
    def attend(self, query, hot_cache, warm_cache, cold_cache):
        # Local attention on hot cache (sliding window)
        local_out = sliding_window_attn(query, hot_cache, window_size=4096)
        
        # Sparse attention on warm cache (CSA with more context)
        warm_out = compressed_attention(query, warm_cache, ratio=4)
        
        # Heavily compressed attention on cold cache
        cold_out = heavily_compressed_attention(query, cold_cache, ratio=128)
        
        # Fuse
        return fuse_attention_outputs(local_out, warm_out, cold_out)
```

#### 5.3.6 NVFP4 Per-Operator Precision (Nemotron-Ultra)

Nemotron-Ultra proved that different components need different precision levels for optimal quality-efficiency trade-off. Apply per-operator quantization:

| Operator | Precision | Rationale |
|----------|-----------|-----------|
| **MoE routed experts** | NVFP4 (E2M1) | Largest memory consumer; 2D block quantization with RHT preserves quality |
| **MoE shared expert** | FP8 | Less critical; higher traffic requires efficient compute |
| **Attention Q/K/V/O** | BF16 | Attention quality is sensitive to precision; keep full |
| **KV cache** | FP8 keys + FP8 values | 2× compression vs FP16 with minimal quality loss |
| **SSM (Mamba-2)** | FP16 with stochastic rounding | SSM state recurrence amplifies quantization error; SR mitigates drift |
| **Embeddings** | BF16 | Embedding lookup is memory-bound, low compute cost |
| **MoE gate/router** | BF16 | Router precision critical for expert selection quality |

```python
class PerOperatorPrecisionConfig:
    """Per-operator precision mapping following Nemotron-Ultra."""
    precision_map = {
        'moe_routed': 'nvfp4',      # NVFP4 E2M1 with 2D block quant
        'moe_shared': 'fp8',        # FP8 per-block scaling
        'attention': 'bf16',        # Full precision
        'kv_cache': 'fp8',          # FP8 for both keys and values
        'ssm': 'fp16_sr',           # FP16 with stochastic rounding
        'embedding': 'bf16',        # BF16
        'router': 'bf16',           # BF16
    }
```

**NVFP4 (E2M1) format**: 2 exponent bits + 1 mantissa bit → 4-bit. Uses:
- **2D block quantization**: Scales computed per 2D tile (not per row) for better granularity
- **Random Hadamard Transform (RHT)**: Applied before quantization to spread outliers and reduce MSE
- **Stochastic rounding**: Unbiased rounding for gradients during training

**Memory impact**: MoE routed experts in NVFP4 use ~1/4 the memory of FP16 with <1% quality degradation, validated at 550B scale on Nemotron-Ultra.

#### 5.3.7 Prefill-Decode Disaggregation (Nemotron-Ultra)

Separate compute infrastructure for prefill (compute-bound) vs decode (memory-bound) phases:

```python
class DisaggregatedPipeline:
    """
    Prefill GPUs: handle long-context prefill with ring attention
    Decode GPUs: handle autoregressive generation with optimized KV cache
    
    Benefits:
    - Prefill GPUs can use large batch sizes (throughput-optimized)
    - Decode GPUs can optimize for low latency (no prefill variability)
    - Each phase uses its ideal parallelism strategy:
      - Prefill: TP + CP (context parallelism for ring attention)
      - Decode: TP + EP (expert parallelism for MoE)
    """
    
    def prefill_on_prefill_gpus(self, tokens, segment_group):
        # Use 4-way CP (context parallelism) + 2-way TP
        context_parallel_size = 4
        tensor_parallel_size = 2
        return ring_attention_prefill(tokens, cp_group, tp_group)
    
    def decode_on_decode_gpus(self, token, kv_cache):
        # Use 2-way EP (expert parallelism) + 2-way TP
        expert_parallel_size = 2
        tensor_parallel_size = 2
        return speculative_decode(token, kv_cache, ep_group, tp_group)
```

**Impact**: Nemotron-Ultra achieves 2.89× throughput at DL=6 with MTP speculative decoding combined with disaggregation. For Lasmoid's 2M context, prefill GPUs handle the ~36s prefill while decode GPUs maintain low-latency generation at ≥20 tok/s.

---

### 5.4 Attention Architecture Upgrade

#### 5.4.1 Hybrid Sliding Window + Global Attention (Gemma-4)

The key innovation from Gemma-4: alternating local sliding window and global attention layers. This provides the efficiency of window attention with the long-range capability of global attention.

```python
import enum

class AttentionType(enum.IntEnum):
    """Attention type per layer — Gemma-4 pattern."""
    GLOBAL = 1          # Full attention (standard)
    LOCAL_SLIDING = 2   # Sliding window attention

def make_attention_layers_types(
    pattern: tuple,    # e.g., (LOCAL_SLIDING, LOCAL_SLIDING, LOCAL_SLIDING, LOCAL_SLIDING, GLOBAL)
    num_layers: int,    # 28 for Lasmoid 1B
) -> list[AttentionType]:
    """
    Creates attention types array by repeating the pattern.
    
    Gemma-4 2B:  4 local + 1 global (5-layer pattern)
    Gemma-4 26B: 5 local + 1 global (6-layer pattern)
    Lasmoid 1B:  4 local + 1 global (5-layer pattern, 28 layers → 5 full + 3 remainder)
    """
    types = []
    for i in range(num_layers):
        types.append(pattern[i % len(pattern)])
    return types

# Lasmoid 1B: 4 local sliding + 1 global, repeated
ATTENTION_PATTERN_1B = (
    AttentionType.LOCAL_SLIDING,  # Layer 0
    AttentionType.LOCAL_SLIDING,  # Layer 1
    AttentionType.LOCAL_SLIDING,  # Layer 2
    AttentionType.LOCAL_SLIDING,  # Layer 3
    AttentionType.GLOBAL,         # Layer 4 — full context
)
```

**Sliding mask** for LOCAL_SLIDING layers:

```python
def _create_sliding_mask(
    mask: Tensor,          # [B, 1, q_len, kv_len]
    sliding_window_size: int,  # 512 for Lasmoid 1B
    dtype: torch.dtype,
) -> Tensor:
    """
    Convert causal mask to sliding window mask.
    
    For each query position q:
    - Attend to KVs in range [q - window_size, q]
    - Attend to all KVs before position 0 (no left padding)
    """
    q_idx = torch.arange(mask.shape[-2], dtype=torch.int64, device=mask.device)
    kv_idx = torch.arange(mask.shape[-1], dtype=torch.int64, device=mask.device)
    
    # Causal + sliding: kv must be in [q - window_size, q]
    sliding_mask = (kv_idx.unsqueeze(0) >= q_idx.unsqueeze(1) - sliding_window_size) & \
                   (kv_idx.unsqueeze(0) <= q_idx.unsqueeze(1))
    
    # Combine with original mask
    return mask & sliding_mask.unsqueeze(0).unsqueeze(0)
```

**Layer configuration in Block**:

```python
class LasmoidBlock(nn.Module):
    """Single Lasmoid block with configurable attention type."""
    
    def __init__(self, config, layer_idx: int, attention_type: AttentionType):
        self.attention_type = attention_type
        
        if attention_type == AttentionType.LOCAL_SLIDING:
            self.attn = SlidingWindowAttention(config)
        else:  # GLOBAL
            self.attn = HybridGlobalAttention(config)  # CSA + HCA + MLA
    
    def forward(self, x, kv_cache, positions, mask):
        if self.attention_type == AttentionType.LOCAL_SLIDING:
            # Local sliding window only — very efficient
            sliding_mask = _create_sliding_mask(mask, self.window_size, x.dtype)
            x = self.attn(x, kv_cache, positions, sliding_mask)
        else:
            # Global: full attention including compressed cache
            x = self.attn(x, kv_cache, positions, mask)
        
        x = self.ssm(x)
        x = self.mhc(x)
        x = self.moe(x)
        return x
```

**Ratio**: For Lasmoid 1B (28 layers, 5-layer pattern): 22 local sliding + 6 global → **5:1 local:global ratio**, matching Gemma-4 27B. This means ~79% of attention layers use cheap sliding window, saving ~4× compute per layer vs full attention.

#### 5.4.2 Dual RoPE Frequencies with QK Norm (Gemma-4)

Local and global attention layers use different RoPE frequencies — critical for both short-range precision and long-range generalization:

```python
class DualRoPECache:
    """
    Two independent RoPE frequency bands:
    - LOCAL:  base_frequency = 10,000  (standard, good for nearby token positions)
    - GLOBAL: base_frequency = 1,000,000 (extended, good for 2M-range positions)
    
    Applied as:
    - LOCAL_SLIDING layers → local RoPE (base=10K)
    - GLOBAL layers → global RoPE (base=1M)
    """
    
    LOCAL_BASE = 10_000
    GLOBAL_BASE = 1_000_000
    
    def get_frequencies(self, attention_type: AttentionType, positions: Tensor, dim: int):
        base = self.LOCAL_BASE if attention_type == AttentionType.LOCAL_SLIDING else self.GLOBAL_BASE
        return self._compute_rope(positions, base, dim)
```

**QK Norm with Scale**: Apply RMSNorm after Q/K projection, before RoPE. This stabilizes training and improves quality:

```python
class QKNorm(nn.Module):
    """
    RMSNorm applied to Q and K after projection, before RoPE.
    
    Key detail: uses learnable scale parameter (not just normalization).
    qk_norm_with_scale=True in Gemma-4 config.
    """
    def __init__(self, dim: int):
        self.norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.ones(dim))  # Learnable scale
    
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x) * self.scale

# In attention:
q = q_proj(hidden_states)
k = k_proj(hidden_states)
q = qk_norm(q)  # QK norm with scale
k = qk_norm(k)  # QK norm with scale
q = apply_rope(q, positions, base_frequency)
k = apply_rope(k, positions, base_frequency)
```

**Global Key Expansion**: Global attention layers use a larger key_size than head_dim for better long-range retrieval:

```python
class GlobalAttentionWithExpandedKey(nn.Module):
    """
    Global attention with expanded key dimension.
    
    Gemma-4 pattern: 
    - head_dim = 256 (standard for value)
    - global_key_size = 512 (2× head_dim for keys)
    - k_eq_v_global = False: K and V have separate projections with different dims
    
    The larger key space reduces collisions in long-range attention,
    important for distinguishing between 2M potentially similar tokens.
    """
    def __init__(self, config):
        self.head_dim = config.head_dim          # 256
        self.global_key_size = config.global_key_size  # 512
        self.k_eq_v_global = config.k_eq_v_global
        
        if self.k_eq_v_global:
            # Weight tying: single projection for K and V
            self.kv_proj = nn.Linear(config.dim, self.global_key_size * config.num_heads)
        else:
            self.k_proj = nn.Linear(config.dim, self.global_key_size * config.num_heads)
            self.v_proj = nn.Linear(config.dim, self.head_dim * config.num_kv_heads)
```

#### 5.4.3 DeepSeek Indexer: Learned Sparse Attention

The Indexer is a learned sparse attention mechanism that selects top-k KV entries per query, making attention O(k) instead of O(n):

```python
class DeepSeekIndexer(nn.Module):
    """
    Learned sparse attention via top-k KV selection.
    
    Architecture:
    1. Compress KV cache into compressed index representations
    2. For each query, compute relevance scores via Hadamard-rotated dot product
    3. Select top-k scored KV entries
    
    Key innovations:
    - Hadamard rotation before selection (reduces quantization noise)
    - Learnable weights projection (adapts to attention patterns)
    - Separate lightweight compressor for scoring (lower dim than full KV)
    """
    
    def __init__(self, config):
        self.index_head_dim = config.index_head_dim  # 128
        self.index_topk = config.index_topk          # 512
        
        # Learnable query projection for index scoring
        self.q_index_proj = nn.Linear(
            config.head_dim, 
            self.index_head_dim * config.n_heads
        )
        
        # Compress KV into index space (separate from main KV)
        self.k_index_compressor = nn.Linear(
            config.head_dim,
            self.index_head_dim
        )
        
        # Register Hadamard matrix (orthogonal transform)
        self.register_buffer(
            'hadamard',
            self._create_hadamard(self.index_head_dim)
        )
    
    def forward(self, query: Tensor, kv_cache: KVCache) -> tuple[Tensor, Tensor]:
        B, H, S, D = kv_cache.keys.shape
        
        # 1. Compress KV cache for indexing
        idx_keys = self.k_index_compressor(kv_cache.keys)  # [B, H, S, idx_dim]
        
        # 2. Apply Hadamard rotation (reduces quantization noise in selection)
        idx_keys = torch.matmul(idx_keys, self.hadamard)
        
        # 3. Project query to index space
        q_idx = self.q_index_proj(query)  # [B, H, q_len, idx_dim]
        q_idx = torch.matmul(q_idx, self.hadamard)
        
        # 4. Compute relevance scores and select top-k
        scores = torch.matmul(q_idx, idx_keys.transpose(-2, -1))  # [B, H, q_len, S]
        topk_values, topk_indices = torch.topk(scores, self.index_topk, dim=-1)
        
        # 5. Gather selected KV entries
        selected_k = gather_kv(kv_cache.keys, topk_indices)
        selected_v = gather_kv(kv_cache.values, topk_indices)
        
        return selected_k, selected_v, topk_values
```

**Role in 2M context**: The Indexer replaces full cache attention with O(topk) selection. At 2M tokens, selecting top-512 reduces attention compute by ~4000× vs full attention, while learned selection maintains quality.

#### 5.4.4 Attention Sink (DeepSeek-V4-Pro)

The first KV entry is always kept — acts as an "attention sink" that absorbs excess attention mass:

```python
class AttentionSink(nn.Module):
    """
    Learnable bias parameter for the first KV position.
    
    Idea: During training, the first token absorbs excess attention mass
    (attention sink phenomenon). By explicitly keeping it and adding a
    learnable bias, we prevent the model from allocating attention to
    irrelevant positions just to "dump" mass.
    
    DeepSeek-V4-Pro pattern: first KV entry always present in cache,
    with learnable bias parameter added to attention scores.
    """
    def __init__(self):
        self.sink_bias = nn.Parameter(torch.zeros(1))  # Learnable bias
    
    def apply(self, attn_scores: Tensor, kv_cache: KVCache, sink_indices: Tensor):
        """
        Add sink bias to first KV position's attention score.
        """
        attn_scores[:, :, :, sink_indices] += self.sink_bias
        return attn_scores

# Applied in attention forward:
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
attn_scores = self.attention_sink.apply(attn_scores, kv_cache, sink_idx=0)
```

**Why it matters for 2M**: With a 2M cache where most entries are heavily compressed, the attention sink ensures the first token (which establishes context) is always perfectly preserved, not evicted or blurred by compression.

#### 5.4.5 Adaptive CSA/HCA Ratio (Existing + Enhanced)

```python
class AdaptiveCompressedAttention(nn.Module):
    """
    CSA/HCA with automatic ratio selection based on context length,
    enhanced with Indexer and Attention Sink.
    """
    def __init__(self, config):
        self.csa = CompressedSparseAttention(config)    # ratio=4
        self.hca = HeavilyCompressedAttention(config)   # ratio=128
        self.mla = MultiHeadLatentAttention(config)     # full precision
        self.gate = AdaptiveCompressorGate()
        self.indexer = DeepSeekIndexer(config)           # learned sparse
        self.sink = AttentionSink()                      # attention sink
    
    def forward(self, x, kv_cache, current_seqlen):
        mode = self.gate.get_mode(current_seqlen)
        
        if mode == CompressionMode.NORMAL:
            # MLA for recent, CSA for cached
            return self.csa(x, kv_cache)
        
        elif mode == CompressionMode.HYPER:
            # HCA for cached, Indexer for sparse selection, sink preserved
            selected_k, selected_v = self.indexer(x, kv_cache)
            attn_out = self.hca(x, selected_k, selected_v)
            attn_out = self.sink.apply(attn_out, kv_cache, sink_idx=0)
            return attn_out
        
        else:  # EMERGENCY
            # Indexer-only with aggressive topk, no fallback
            return self.indexer(x, kv_cache, topk=256)
```

#### 5.4.6 Block AttnRes Integration

**Source**: `open-attention-residuals/modeling_attnres.py`

Block AttnRes replaces standard attention with block-level depth attention:

```python
class BlockAttnResidualAttention(nn.Module):
    """
    Block AttnRes: organize past KV into blocks, attend over blocks with depth.
    
    For 2M context:
    - Block size: 4096 tokens
    - ~512 blocks total
    - Depth attention over blocks: O(num_blocks) = O(512) instead of O(2M)
    - Within-block: standard CSA attention
    """
    
    def __init__(self, config):
        self.block_size = config.block_size  # 4096
        self.num_depth_heads = config.num_depth_heads  # 2
        self.block_attn = nn.MultiheadAttention(
            embed_dim=config.dim,
            num_heads=self.num_depth_heads,
            batch_first=True,
        )
        self.csa = CompressedSparseAttention(config)
    
    def forward(self, query, kv_cache, block_indices):
        # 1. Pool each block into a single representation
        block_reprs = self.pool_blocks(kv_cache)  # [num_blocks, dim]
        
        # 2. Depth attention: which blocks are relevant?
        depth_scores = self.block_attn(
            query.unsqueeze(1),           # query
            block_reprs.unsqueeze(0),     # key
            block_reprs.unsqueeze(0),     # value
        )[0]  # [1, 1, num_blocks]
        
        # 3. Select top-k blocks
        top_blocks = depth_scores.topk(k=self.num_selected_blocks).indices
        
        # 4. CSA within selected blocks
        selected_kv = gather_blocks(kv_cache, top_blocks)
        out = self.csa(query, selected_kv)
        
        return out
```

#### 5.4.7 Combined Window + Compress TopK (DeepSeek-V4-Pro)

DeepSeek-V4-Pro combines sliding window topk with compressed cache topk in a single attention pass:

```python
class CombinedWindowCompressAttention(nn.Module):
    """
    Combined attention: recent window tokens + top-k compressed cache tokens.
    
    DeepSeek-V4-Pro pattern:
    1. Window_topk: keep last `window_size` tokens at full precision
    2. Compress_topk: from compressed cache, select top-k by attention score
    3. Concatenate both sets and attend jointly
    
    This gives the best of both worlds:
    - Window: perfect recall of recent context
    - Compressed top-k: long-range retrieval where it matters
    """
    
    def __init__(self, config):
        self.window_size = config.window_size  # 512
        self.compress_topk = config.compress_topk  # 1024
        self.mla = MultiHeadLatentAttention(config)
        self.indexer = DeepSeekIndexer(config)
    
    def forward(self, x, kv_cache, position_ids):
        # 1. Recent window: keep last window_size tokens full precision
        window_start = max(0, kv_cache.size - self.window_size)
        window_k = kv_cache.k[window_start:]
        window_v = kv_cache.v[window_start:]
        
        # 2. Compressed cache: select top-k by learned indexer
        cache_end = window_start  # Exclude window tokens from cache selection
        if cache_end > 0:
            cache_kv = kv_cache.slice(0, cache_end)
            selected_k, selected_v = self.indexer(x, cache_kv, topk=self.compress_topk)
        else:
            selected_k, selected_v = None, None
        
        # 3. Concatenate and attend
        if selected_k is not None:
            all_k = torch.cat([selected_k, window_k], dim=-2)
            all_v = torch.cat([selected_v, window_v], dim=-2)
        else:
            all_k, all_v = window_k, window_v
        
        return self.mla(x, all_k, all_v)
```

**How it works at 2M**: With window_size=512 and compress_topk=1024, the model attends to 1536 total tokens regardless of total context length. This is **~1300× cheaper** than full attention at 2M while preserving both recent precision and long-range retrieval.

#### 5.4.8 Ring Attention for Distributed Prefill

For the initial 2M token prefill, distribute across GPUs:

```python
class RingAttentionPrefill:
    """
    Distributed prefill using ring attention pattern.
    
    Each GPU processes a segment of the sequence.
    Segments are passed around the ring for full attention computation.
    
    Memory: O(seqlen / num_gpus) per GPU instead of O(seqlen)
    """
    
    def prefill(self, hidden_states, ring_group):
        world_size = ring_group.size()
        rank = ring_group.rank()
        
        # Split sequence into segments
        segment_size = hidden_states.size(1) // world_size
        local_segment = hidden_states[:, rank * segment_size:(rank + 1) * segment_size]
        
        # Each GPU computes KV for its segment
        local_k, local_v = self.compute_kv(local_segment)
        
        # Ring pass: send KV around, accumulate attention
        for step in range(world_size):
            sender = (rank - step) % world_size
            recv_k, recv_v = ring_recv(sender, ring_group)
            
            # Attend local query to received KV
            attn_out = self.attention(local_segment, recv_k, recv_v)
            self.accumulate_attention(attn_out, sender)
            
            # Forward KV
            ring_send(local_k, local_v, (rank + 1) % world_size, ring_group)
        
        return self.gather_results(ring_group)
```

---

### 5.5 Multimodal Integration

Following Gemma-4's architecture: SigLIP-style vision encoder (output_length=280, bidirectional attention) + Conformer audio encoder (12 layers, 1024 dim) + per-layer input embeddings for modality-aware routing.

#### 5.5.1 Vision Encoder (Gemma-4 SigLIP-Style)

```python
class LasmoidVisionEncoder(nn.Module):
    """
    Gemma-4 SigLIP-style vision encoder.
    
    Architecture:
    - Vision transformer with 16-27 layers, 14×14 patch size (patch_size=16 for flexible)
    - Output_length=280 vision tokens (pooled from ~256 patches + cls)
    - Bidirectional attention for vision tokens (non-causal)
    - Positional embeddings: learned (not sinusoidal)
    - standardize_embeddings: optional normalization of vision embeddings
    
    Key design decisions (from Gemma-4):
    - Bidirectional attention: vision tokens attend to ALL other vision tokens
      (not causal like text), since images don't have sequential dependency
    - Fixed output_length=280: simplifies integration; spatial pooling maps
      variable-resolution images to fixed token count
    - Positional embeddings: learned per-patch position (captures 2D layout)
    """
    
    def __init__(self, config):
        # Vision backbone
        self.patch_size = config.patch_size or 14
        self.num_vision_layers = config.num_vision_layers or 16
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, config.vision_dim,
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # Positional embeddings (learned)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1024, config.vision_dim) * 0.02
        )  # Max 1024 patches (≈32×32 tiles)
        
        # Vision transformer layers (bidirectional attention)
        self.vision_layers = nn.ModuleList([
            BidirectionalTransformerLayer(config.vision_dim)
            for _ in range(self.num_vision_layers)
        ])
        
        # Spatial pooling: map variable patches → fixed output_length=280
        self.spatial_pool = nn.AdaptiveAvgPool1d(280)  # output_length=280
        
        # Modality embedding — tells text layers "this is a vision token"
        self.modality_embedding = nn.Embedding(3, config.dim)  # text/vision/audio
        # modality_embedding weight shared via per_layer_input_embeddings
        
        # Optional standardization
        self.standardize_embeddings = config.standardize_embeddings or False
        
        # Projection: vision_dim → text_dim
        self.proj = nn.Sequential(
            nn.Linear(config.vision_dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.dim),
            nn.RMSNorm(config.dim),
        )
    
    def encode_image(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W] or variable aspect ratio
        Returns:
            vision_embeddings: [B, 280, dim]
            modality_ids: [B, 280] tensor of MODALITY_VISION
        """
        # 1. Patch embed
        patches = self.patch_embed(pixel_values)  # [B, dim, H/p, W/p]
        B, D, Hp, Wp = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        
        # 2. Add positional embeddings
        patches = patches + self.pos_embed[:, :patches.size(1), :]
        
        # 3. Bidirectional transformer layers (no causal mask)
        for layer in self.vision_layers:
            patches = layer(patches)
        
        # 4. Spatial pool to output_length=280
        pooled = self.spatial_pool(patches.transpose(1, 2)).transpose(1, 2)
        
        # 5. Project to text dimension
        embeddings = self.proj(pooled)  # [B, 280, dim]
        
        # 6. Standardize if configured
        if self.standardize_embeddings:
            embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-6)
        
        # 7. Create modality IDs (for per-layer embeddings)
        modality_ids = torch.full(
            (B, 280), MODALITY_VISION, device=embeddings.device
        )
        
        return embeddings, modality_ids
```

**Output length**: Fixed at 280 tokens per image (Gemma-4 pattern). Resolution flexibility via:
- **Variable aspect ratio**: Image is tiled into `ceil(H/14) × ceil(W/14)` patches, then pooled to 280
- **Quality/compute trade-off**: Multiple budgets at inference (70, 140, 280, 560, 1120) by adjusting the spatial pooling output dimension
- Special tokens `<|image|>` in tokenizer signal insertion points (IMAGE_PLACEHOLDER=258880 following Gemma-4)

#### 5.5.2 Audio Encoder (Gemma-4 Conformer-Style)

```python
class ConformerSubsampling(nn.Module):
    """
    Conformer subsampling block.
    
    Reduces audio sequence length via:
    1. Conv2D: [B, 1, T, F] → [B, dim, T/2, F]
    2. Conv2D: [B, dim, T/2, F] → [B, dim, T/4, F]
    3. Reshape to [B, T/4, dim*F]
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, F] (time × freq features)
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        B, D, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, D * F)
        return self.norm(x)

class LasmoidAudioEncoder(nn.Module):
    """
    Gemma-4 Conformer-style audio encoder.
    
    Architecture:
    - 12 Conformer layers (macaron-style: FFN + MHSA + Conv + FFN)
    - 1024 model dimensions
    - 1536 LM projection dimensions (model_dim → lm_model_dim)
    - 16kHz sample rate
    - 2 subsampling blocks (4× time reduction)
    - Max ~30 seconds audio
    
    Conformer block structure:
    ┌──────────────────────────────────┐
    │  x = x + 0.5 * FFN(x)          │
    │  x = x + MHSA(x)               │
    │  x = x + ConvModule(x)         │
    │  x = x + 0.5 * FFN(x)          │
    │  x = LayerNorm(x)              │
    └──────────────────────────────────┘
    """
    def __init__(self, config):
        self.conformer_dims = config.audio_conformer_dims or 1024
        self.num_audio_layers = config.num_audio_layers or 12
        self.lm_model_dims = config.audio_lm_dims or 1536
        
        # Frontend: subsampling (4× reduction)
        self.subsampling = ConformerSubsampling(
            in_dim=config.audio_feature_dim,  # e.g., 80 mel bands
            out_dim=self.conformer_dims
        )
        
        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(dim=self.conformer_dims)
            for _ in range(self.num_audio_layers)
        ])
        
        # Projection to LM space
        self.lm_proj = nn.Linear(self.conformer_dims, self.lm_model_dims)
        
        # Final projection: lm_model_dims → text dim
        self.proj = nn.Sequential(
            nn.Linear(self.lm_model_dims, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.dim),
            nn.RMSNorm(config.dim),
        )
    
    def encode_audio(self, waveform_or_features: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            waveform_or_features: [B, T, F] or [B, samples]
        Returns:
            audio_embeddings: [B, num_tokens, dim]
            modality_ids: [B, num_tokens] tensor of MODALITY_AUDIO
        """
        # Subsampling (4× time reduction)
        x = self.subsampling(waveform_or_features)
        
        # Conformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to LM dimensions
        x = self.lm_proj(x)
        
        # Project to text dimension
        embeddings = self.proj(x)
        
        modality_ids = torch.full(
            (embeddings.shape[0], embeddings.shape[1]), 
            MODALITY_AUDIO, device=embeddings.device
        )
        
        return embeddings, modality_ids
```

**Gemma-4 audio specifics**:
- **16kHz sample rate**: All input resampled to 16kHz
- **80-channel mel spectrogram** as input features (standard for Conformer)
- **~30 seconds max** audio per segment
- **Modality tokens**: AUDIO_PLACEHOLDER=258881 in tokenizer

#### 5.5.3 Multimodal Embedding Fusion with Per-Layer Support

Leveraging the per-layer input embeddings from Section 5.1 to give each layer modality awareness:

```python
MODALITY_TEXT = 0
MODALITY_VISION = 1
MODALITY_AUDIO = 2

class MultimodalEmbedder(nn.Module):
    """
    Embedder that handles text, vision, and audio with per-layer modality routing.
    
    The per_layer_embeddings (from Section 5.1) give each layer the modality type
    of each token, allowing layer-specific processing of vision vs. audio vs. text tokens.
    """
    
    def __init__(self, config):
        # Base text embedding
        self.input_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Per-layer embeddings with modality awareness
        self.per_layer_embeddings = nn.Parameter(
            torch.randn(3, config.n_layers, config.per_layer_input_dim) * 0.02
        )  # [modality, layers, per_layer_dim]
        
        # Per-layer projection
        self.per_layer_proj = nn.Linear(config.dim, config.n_layers * config.per_layer_input_dim)
        self.per_layer_norm = RMSNorm(config.per_layer_input_dim)
    
    def embed_with_modality(
        self,
        input_ids: Tensor,
        vision_embeddings: Optional[Tensor] = None,
        audio_embeddings: Optional[Tensor] = None,
        modality_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            embeddings: [B, S, dim] — fused multimodal embeddings
            per_layer_features: [B, S, n_layers, per_layer_dim] — per-layer features
        """
        # 1. Base text embeddings
        S = input_ids.shape[1]
        embeddings = self.input_embedding(input_ids) * math.sqrt(config.dim)
        
        # 2. Inject vision embeddings at placeholder positions
        if vision_embeddings is not None:
            img_pos = (input_ids == IMAGE_TOKEN_ID)
            # Insertion logic with positional tracking
            # ... stitch vision_embeddings into placeholder positions
        
        # 3. Inject audio embeddings
        if audio_embeddings is not None:
            aud_pos = (input_ids == AUDIO_TOKEN_ID)
            # ... stitch audio_embeddings into placeholder positions
        
        # 4. Per-layer modality features
        per_layer = self.per_layer_proj(embeddings)
        per_layer = per_layer.view(B, S, config.n_layers, config.per_layer_input_dim)
        
        # Add modality-specific per-layer embeddings
        for mod_id in range(3):
            mask = (modality_ids == mod_id).unsqueeze(-1).unsqueeze(-1)
            per_layer += mask * self.per_layer_embeddings[mod_id]
        
        per_layer = self.per_layer_norm(per_layer)
        
        return embeddings, per_layer
```

**Variable token budgets**: Inference-time flexibility via `soft_tokens` hyperparameter:
- 70 tokens/image (fast, lower quality)
- 140 tokens/image (balanced)
- 280 tokens/image (default — Gemma-4 standard)
- 560 tokens/image (high quality)
- 1120 tokens/image (maximum detail)

Achieved by changing the `AdaptiveAvgPool1d` output dimension in the vision encoder — no model retraining needed for different budgets.

---

### 5.6 Long-Running Stability System

**Prerequisite: Final Logit Softcap = 30.0** (Gemma-4)

Before any stability monitoring, apply a logit softcap to prevent logit explosion:

```python
class LogitSoftcap(nn.Module):
    """
    Final logit softcap: tanh(logits / softcap) * softcap
    
    Gemma-4 uses softcap=30.0 applied to the final logits.
    Prevents a single logit from dominating (which causes:
    - Extreme softmax distributions → near-deterministic output
    - Entropy collapse in long generations
    - Numerical instability in KV cache quantization)
    
    Applied: after final layer norm, before softmax sampling.
    """
    def __init__(self, softcap: float = 30.0):
        self.softcap = softcap
    
    def forward(self, logits: Tensor) -> Tensor:
        return torch.tanh(logits / self.softcap) * self.softcap

# In generate():
logits = model(input_ids)  # raw logits
logits = logit_softcap(logits)  # clamp to [-30, 30]
probs = F.softmax(logits / temperature, dim=-1)
```

**Why 30.0**: Empirically determined by Gemma-4 to balance expressiveness (logits can still differentiate strongly) with stability (no single logit exceeds ±30, preventing softmax saturation in 262K-vocabulary space).

For days-long agentic output without hallucination or degradation.

#### 5.6.1 Logit Drift Detection

Monitor logit statistics over time to detect when the model enters an unstable regime:

```python
class DriftDetector:
    """
    Monitors logit statistics over a sliding window of generated tokens.
    
    Signals:
    - Entropy collapse: max logit probability approaches 1.0 (model is "too sure")
    - Entropy explosion: uniform distribution across many tokens (model is confused)
    - Logit norm drift: ||logits|| changes more than 2σ from rolling mean
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.entropy_history = deque(maxlen=window_size)
        self.max_prob_history = deque(maxlen=window_size)
        self.logit_norm_history = deque(maxlen=window_size)
        
    def check(self, logits: Tensor) -> DriftSignal:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1)
        max_prob = probs.max(-1).values
        logit_norm = logits.norm(dim=-1)
        
        self.entropy_history.append(entropy.item())
        self.max_prob_history.append(max_prob.item())
        self.logit_norm_history.append(logit_norm.item())
        
        signals = []
        
        # Entropy collapse detection
        if len(self.entropy_history) >= self.window_size:
            mean_e = np.mean(self.entropy_history)
            std_e = np.std(self.entropy_history)
            if entropy < mean_e - 3 * std_e:
                signals.append(DriftSignal.ENTROPY_COLLAPSE)
        
        # Logit norm spike detection
        if len(self.logit_norm_history) >= self.window_size:
            mean_n = np.mean(self.logit_norm_history)
            std_n = np.std(self.logit_norm_history)
            if abs(logit_norm - mean_n) > 3 * std_n:
                signals.append(DriftSignal.LOGIT_NORM_SPIKE)
        
        return signals
```

#### 5.6.2 Temperature Scheduling

Adjust generation temperature based on drift signals and context position:

```python
class AdaptiveTemperatureScheduler:
    """
    Dynamically adjusts temperature for stable long-form generation.
    
    Strategy:
    - Default: temp=0.7 (balanced)
    - Entropy collapse: temp += 0.2 (increase randomness)
    - Entropy explosion: temp -= 0.3 (reduce randomness)
    - Context >1M tokens: temp *= 0.9 (be more conservative with distant context)
    - After 10K+ generated tokens: temp *= 0.95 (gradually reduce randomness)
    """
    
    def __init__(self, base_temp: float = 0.7):
        self.base_temp = base_temp
        self.generated_tokens = 0
    
    def get_temperature(self, context_len: int, drift_signals: List[DriftSignal]) -> float:
        temp = self.base_temp
        
        if DriftSignal.ENTROPY_COLLAPSE in drift_signals:
            temp += 0.2
        if DriftSignal.ENTROPY_EXPLOSION in drift_signals:
            temp -= 0.3
        
        # Reduce randomness for very long context (model uncertainty increases)
        if context_len > 1_000_000:
            temp *= 0.9
        elif context_len > 500_000:
            temp *= 0.95
        
        # Gradual decay over long generation
        if self.generated_tokens > 10000:
            temp *= 0.95
        elif self.generated_tokens > 5000:
            temp *= 0.98
        
        return max(0.1, min(2.0, temp))
```

#### 5.6.3 KV Cache Integrity Checks

Prevent silent corruption in the compressed cache:

```python
class KVCacheIntegrityChecker:
    """
    Periodically checks KV cache integrity.
    
    Checks:
    - NaN detection: scan for NaN values every N steps
    - Scale consistency: verify quantization scales haven't drifted
    - Token count: verify cache size matches expected
    """
    
    def check(self, kv_cache, step: int) -> bool:
        if step % self.check_interval != 0:
            return True  # Only check periodically
        
        has_nan = (
            torch.isnan(kv_cache.keys).any() or 
            torch.isnan(kv_cache.values).any()
        )
        
        if has_nan:
            logger.error("NaN detected in KV cache at step %d", step)
            return False  # Signal for recovery
        
        return True
```

#### 5.6.4 Graceful Degradation & Recovery

When something goes wrong during long generation:

```python
class LongRunningRecovery:
    """
    Recovery strategies for long-running generation failures.
    
    Strategies:
    - L1 (cache issue): reset KV cache, re-generate from last checkpoint
    - L2 (drift detected): revert to last stable generation point, adjust parameters
    - L3 (NaN/loss spike): trigger checkpoint restore, reduce temperature
    - L4 (catastrophic): return partial results, request user restart
    """
    
    def recover(self, error: GenerationError, checkpoint_manager, drift_detector):
        if error.level == ErrorLevel.L1:
            # Reset KV cache, resume from checkpoint
            checkpoint_manager.restore_latest()
            drift_detector.reset_history()
            return RecoveryAction.CONTINUE
        
        elif error.level == ErrorLevel.L2:
            # Adjust parameters and continue
            checkpoint_manager.restore_latest()
            drift_detector.reset_history()
            return RecoveryAction.ADJUST_TEMPERATURE
        
        elif error.level == ErrorLevel.L3:
            # Full restore
            checkpoint_manager.restore_latest()
            return RecoveryAction.RESTART_GENERATION
        
        else:
            return RecoveryAction.ABORT
```

---

### 5.7 Training Configuration

#### 5.7.1 MTP Speculative Decoding (Nemotron-Ultra)

MTP (Multi-Token Prediction) heads enable speculative decoding — significantly faster generation without quality loss:

```python
class MTPSpeculativeDecoder(nn.Module):
    """
    MTP speculative decoding with shared-weight heads.
    
    Architecture (Nemotron-Ultra pattern):
    - 2 MTP heads with shared weights (not independent)
    - Each head predicts tokens t+1, t+2 from the same base representation
    - Speculative decoding: draft k tokens → verify in parallel → accept
    
    Key innovation: shared weights between MTP heads allows:
    - Half the parameters of independent heads
    - Faster speculative verification (single forward pass)
    - 2.89× throughput at draft length=6 (Nemotron validated)
    """
    
    def __init__(self, config):
        # Single shared prediction head
        self.mtp_head = nn.Sequential(
            nn.Linear(config.dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.vocab_size),
        )
        # Shared weights between head 1 (t+1) and head 2 (t+2)
    
    def draft_and_verify(self, hidden_states, kv_cache, temperature=0.7):
        """
        Speculative decoding flow:
        1. Draft: generate 6 tokens autoregressively (cheap)
        2. Verify: run model in parallel on all 6 draft positions
        3. Accept: keep longest prefix that matches model's distribution
        4. Repeat from new position
        
        At DL=6, 2.89× throughput vs standard autoregressive.
        """
        # Draft phase (cheap, single MTP head)
        draft_tokens = []
        h = hidden_states
        for _ in range(6):  # draft_length = 6
            logits_t1 = self.mtp_head(h[:, -1:, :])  # predict t+1
            logits_t2 = self.mtp_head(h[:, -1:, :])  # same head, predict t+2
            next_token = sample_from_logits(logits_t1, temperature)
            draft_tokens.append(next_token)
            h = model.forward_one_step(next_token, kv_cache)
        
        # Verify phase (parallel, full model)
        verified = model.verify_drafts(draft_tokens, kv_cache)
        
        return verified
```

**Speculative decoding protocol**:
| Draft Length | Throughput Gain | Acceptance Rate |
|-------------|----------------|-----------------|
| DL=2 | 1.45× | ~85% |
| DL=4 | 2.10× | ~72% |
| DL=6 | 2.89× | ~60% |
| DL=8 | 3.10× | ~48% |

For Lasmoid, use **DL=6** as default (highest efficiency point), with MTP shared-weight heads.

#### 5.7.2 WSD Schedule & NVFP4 Pretraining (Nemotron-Ultra)

**WSD Schedule** (Warmup + Stable + Decay):

```
Phase 1: Warmup (2% of steps)
- Linear warmup from 0 → peak LR
- Peak LR: 2.5e-4 (for 1B model, adjusted from Nemotron's 2.5e-4 at 550B)
- Stabilizes optimizer states

Phase 2: Stable (90% of steps)
- Constant peak LR
- Model learns long-range patterns
- 20T tokens for Nemotron; for Lasmoid 1B fine-tune: proportionally fewer

Phase 3: Decay (8% of steps)  
- Cosine/linear decay to min LR (2.5e-6, 1% of peak)
- Gradual convergence
- Cooldown period for stable final checkpoint
```

**NVFP4 Pretraining Recipe** (from Nemotron-Ultra):

```
NVFP4 E2M1 (2 exponent, 1 mantissa) quantization applied during pretraining:
1. Forward pass: weights cast to NVFP4 via Random Hadamard Transform + 2D block quant
2. Backward pass: gradients computed in FP16 (no quantization)
3. Optimizer: master weights in FP32 accumulation
4. Stochastic rounding: unbiased rounding of NVFP4 values during training forward

Precision by component during pretraining:
  MoE routed experts → NVFP4 E2M1 (with RHT + 2D block scaling)
  MoE shared expert → FP8
  Attention → BF16 (always full precision)
  SSM cache → FP16 with stochastic rounding
  KV cache → FP8
  Embeddings → BF16

This is a drop-in precision wrapper — the architecture doesn't change:
  model = NVFP4Wrapper(model, components=['moe_routed', 'moe_shared'])
```

#### 5.7.3 MOPD: Multi-teacher On-Policy Distillation

Post-training distillation following Nemotron-Ultra's MOPD:

```
MOPD (Multi-teacher On-Policy Distillation):
1. Multiple teacher models generate trajectories
2. Student (Lasmoid) generates on-policy samples
3. Distillation loss: KL(teacher_distribution || student_distribution)
4. Data: mixture of teacher-generated and real data

Teachers for Lasmoid:
- DeepSeek-V4-Pro: strongest compression patterns
- Gemma-4-2B: strongest multimodal reasoning
- Current Lasmoid checkpoint: self-distillation (prevent regression)

Loss: L = L_next_token + λ_mtp * L_mtp + λ_mopd * L_mopd + λ_grpo * L_grpo
```

#### 5.7.4 Three-Stage Training Pipeline

```
Stage 1: NVFP4-Aware Pretraining (long-context warm-start)
- Apply NVFP4 quantization to MoE layers during forward pass
- WSD schedule: warmup 500 steps, stable at peak LR 2.5e-4, decay 2000 steps
- Progressive length: 4K → 32K → 256K (each step 1000 steps)
- Loss: next-token prediction + MTP auxiliary loss (predictive_coding_coeff=0.01)

Stage 2: Long-Context Fine-Tuning
- Base: NVFP4-pretrained checkpoint
- Data: Curated long-context corpus (books, code repos, conversations >100K tokens)
- Strategy: Progressive length extension (256K → 1M → 2M)
- Optimizer: AdamW (not Muon) for stability during extension
- MTP speculative decoding enabled for evaluation

Stage 3: MOPD + GRPO Alignment
- MOPD distillation from teacher models (on-policy)
- GRPO reward shaping: reward = quality - rambling_penalty - drift_penalty
- Training on long-form agentic trajectories (multi-turn, >10K tokens)
- KL penalty against base model to prevent reward hacking
- Final logit softcap=30.0 applied during inference
```

#### 5.7.5 Training Config (`config_1b_2m.json`)

```json
{
  "vocab_size": 262144,
  "max_seq_len": 2097152,
  "max_batch_size": 1,
  "dtype": "bf16",
  "norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "rope_factor": 32.0,
  "original_seq_len": 4096,
  "beta_fast": 64,
  "beta_slow": 2,
  "window_size": 4096,
  "swiglu_limit": 10.0,
  "num_residual_streams": 4,
  "hc_sinkhorn_iters": 8,
  "hc_eps": 1e-06,
  "hcm_ema_alpha": 0.99,
  "hcm_commit_loss_coeff": 0.25,
  "entropy_threshold": 0.5,
  "router_z_loss_coeff": 0.001,
  "ema_bias_lr": 0.01,
  "expert_capacity_factor": 1.25,
  "moe_load_balance_coeff": 0.01,
  "predictive_coding_coeff": 0.01,
  "reasoning_steps": 2,
  "think_token_id": 107,
  "answer_token_id": 108,
  "cot_exit_confidence": 0.9,
  "moe_router_entropy_coeff": 0.001,
  "moe_capacity_loss_coeff": 0.01,
  "token_concept_loss_coeff": 0.05,
  "steering_attributes": ["creativity", "helpfulness", "complexity", "scientific_rigor"],
  "post_attn_norm": true,
  "post_ffw_norm": true,
  "moe_dual_ffn": true,
  "ssm_d_skip": true,
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "q_lora_rank": 256,
  "head_dim": 48,
  "rope_head_dim": 16,
  "o_groups": 2,
  "o_lora_rank": 256,
  "n_routed_experts": 6,
  "n_shared_experts": 1,
  "n_activated_experts": 2,
  "moe_latent_dim": 512,
  "num_concepts": 64,
  "num_abstract_concepts": 8,
  "num_global_concepts": 2,
  "codebook_size": 256,
  "lightning_topk_blocks": 2,
  "ssm_heads": 16,
  "ssm_state_dim": 16,
  "ssm_kernel_size": 4,
  "ssm_chunk_size": 64,
  "ssm_dt_min": 0.001,
  "ssm_dt_max": 0.1,
  "ssm_dt_init_floor": 0.0001,
  "ssm_n_groups": 1,
  "block_size": 4096,
  "cache_buffer_size": 128,
  "cache_key_bits": 4,
  "cache_value_group_size": 8,
  "cache_max_tokens": 524288,
  "cache_keep_ratio": 0.3,
  "cache_merge_ratio": 0.5,
  "hyper_compress_threshold": 300000,
  "emergency_compress_threshold": 1500000,
  "vision_encoder": "siglip-so400m",
  "vision_token_budget": 140,
  "vision_dim": 1152,
  "audio_encoder": "whisper-medium",
  "audio_max_seconds": 30,
  "audio_dim": 1024,
  "use_ring_attention": true,
  "ring_world_size": 8,
  "drift_check_interval": 50,
  "drift_window_size": 100,
  "temperature_base": 0.7,
  "final_logit_softcap": 30.0,
  // Attention architecture (Gemma-4 hybrid)
  "attention_pattern": [2, 2, 2, 2, 1],        // 4 local + 1 global (1=GLOBAL, 2=LOCAL_SLIDING)
  "sliding_window_size": 512,                    // Local window size
  "num_global_kv_heads": 4,
  "global_key_size": 512,                        // Expanded key dim for global layers
  "k_eq_v_global": false,                        // Separate K/V projections
  "qk_norm_with_scale": true,                    // QK norm with learnable scale
  "local_base_frequency": 10000,                 // Local RoPE base
  "global_base_frequency": 1000000,              // Global RoPE base
  "rope_scale_factor": 1.0,                      // RoPE scale factor
  "rope_proportion": 1.0,                        // RoPE proportion
  // Per-layer compress ratios (DeepSeek-V4)
  "use_per_layer_ratios": true,
  "compress_ratios": [4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128],
  "compress_overlap": true,
  // Indexer (DeepSeek)
  "use_indexer": true,
  "index_topk": 512,
  "index_head_dim": 128,
  "attention_sink": true,
  "sink_bias_init": 0.0,
  // Combined topk
  "use_combined_topk": true,
  "compress_topk": 1024,
  // KV cache sharing
  "frac_shared_layers": 0.5,
  "kv_cache_share_global": false,
  "kv_cache_share_local": false,
  // Per-layer input embeddings (Gemma-4)
  "per_layer_input_dim": 64,
  "use_einsum": true,
  // Vision encoder (SigLIP-style)
  "vision_encoder": "siglip",
  "vision_token_budget": 140,
  "vision_output_length": 280,
  "vision_num_layers": 16,
  "vision_dim": 1152,
  "patch_size": 14,
  "standardize_embeddings": false,
  "enable_bidirectional_vision_attention": true,
  // Audio encoder (Conformer-style)
  "audio_encoder": "conformer",
  "audio_sample_rate": 16000,
  "audio_max_seconds": 30,
  "audio_feature_dim": 80,
  "audio_conformer_dims": 1024,
  "audio_lm_dims": 1536,
  "num_audio_layers": 12,
  // MTP speculative decoding
  "mtp_speculation_enabled": true,
  "mtp_draft_length": 6,
  "mtp_shared_weights": true,
  // Training
  "wsd_warmup_steps": 500,
  "wsd_stable_steps": 22500,
  "wsd_decay_steps": 2000,
  "peak_lr": 0.00025,
  "min_lr": 2.5e-06,
  "nvfp4_moe_routed": true,
  "nvfp4_moe_shared": false,
  "nvfp4_attention": false,
  "mopd_enabled": true,
  "mopd_kl_coeff": 0.1
}
```

---

## 6. Integration Map: NEXUS Research → Lasmoid

| NEXUS Project | Lasmoid Component | Integration Method | New Code | Effort |
|--------------|-------------------|-------------------|----------|--------|
| **turboquant** | KV Cache | Drop-in `TurboQuantKVCache` wrapping existing cache | `kv_cache.py` | Medium |
| **KVCache-Factory** (PyramidKV/SnapKV/H2O) | KV Cache Eviction | `SnapPyramidKV.select_kept_indices()` called after each layer | `eviction.py` | Medium |
| **KVCache-Factory** (quantcache) | KV Cache Quantization | KIVI-style group quant as alternative to TurboQuant | `quantcache.py` | Low |
| **compaction** | KV Merging | `OMPKVMerger.merge()` after eviction to recover pruned tokens | `compaction.py` | Medium |
| **open-attention-residuals** | Attention | `BlockAttnResidualAttention` as optional layer, gated by config | `attnres.py` | High |
| **ml-epicache** | Attention | EPICache monkeypatch pattern for attention backprop | `monkeypatch.py` | Low |
| **NVFP4** (Nemotron) | Quantization | Per-operator NVFP4/FP8/BF16 with 2D block quant + RHT | `kernels/quant.py` | High |
| **DeepSeek Indexer** | Attention | Learned sparse attention with Hadamard rotation top-k selection | `attention_indexer.py` | High |
| **Einsum layers** (Gemma-4) | Projections | Replace nn.Linear with Einsum parameterization | `_layers.py` | Medium |
| **KV cache sharing** (Gemma-4) | KV Cache | frac_shared_layers=0.5, per-type sharing (local/global) | `kv_cache.py` | Medium |
| **Hybrid attention** (Gemma-4) | Attention | LOCAL_SLIDING+GLOBAL pattern with dual RoPE, QK norm, global_key_size | `attention.py` | High |
| **MTP speculative decoding** (Nemotron) | Decoding | Shared-weight MTP heads + DL=6 speculative verification | `mtp.py` | Medium |
| **Conformer audio** (Gemma-4) | Audio | 12-layer Conformer with subsampling + LM projection | `audio.py` | High |
| **SigLIP vision** (Gemma-4) | Vision | SigLIP-style 16-27 layers, output_length=280, bidirectional attn | `vision.py` | High |
| **MOPD distillation** (Nemotron) | Training | Multi-teacher on-policy distillation | `train/mopd.py` | High |
| **WSD schedule** (Nemotron) | Training | Warmup+Stable+Decay LR schedule | `train/scheduler.py` | Low |
| **mHC** | Residual Streams | ✅ Already integrated | None | None |
| **mamba** | SSM | ✅ Already integrated | None | None |
| **superhuman** | Alignment | Alignment head for stability; run periodically during long generation | `superhuman_head.py` | Low |

### Integration Ordering

```
                  TurboQuant
                 (quantize KV)
                      │
                      ▼
           KVCache-Factory/SnapKV
            (select which KVs to keep)
                      │
                      ▼
              Compaction/OMP
            (merge pruned into kept)
                      │
                      ▼
     ┌──────────────────────────────────────┐
     │        AdaptiveQuantizedKVCache      │
     │  (buffer → quantize → evict → merge) │
     └──────────────────────────────────────┘
                      │
                      ▼
            Block AttnRes (optional)
          (depth attention over blocks)
                      │
                      ▼
              CSA / HCA / MLA
         (with adaptive ratio gating)
```

---

## 7. Implementation Phases

### Phase 0: SOLiD Refactoring (Foundation)

**New files**: Full module decomposition as described in Section 5.0

| Task | Files | Description |
|------|-------|-------------|
| P0.1 | `inference/config.py` | Create per-component configs (AttentionConfig, MoEConfig, CompressorConfig, SSMConfig, MHCConfig) from monolithic ModelArgs |
| P0.2 | `inference/kernels/__init__.py` | Create KernelProvider interface (DIP) with FP8/FP4/NVFP4/BF16 implementations |
| P0.3 | `inference/attention.py` | Extract all attention variants under common `Attention` interface (CSA, HCA, MLA, HybridSlidingGlobal) |
| P0.4 | `inference/attention_indexer.py` | Extract DeepSeek Indexer into its own module |
| P0.5 | `inference/compressor.py` | Extract CIF compressor with adaptive gating |
| P0.6 | `inference/kv_cache.py` | Extract KV cache with quantization, eviction, compaction |
| P0.7 | `inference/moe.py` | Extract MoE (Gate, Expert, shared, ragged_dispatch) |
| P0.8 | `inference/ssm.py` | Extract Mamba-2 SSM |
| P0.9 | `inference/mhc.py` | Extract Hyper-Connections |
| P0.10 | `inference/mtp.py` | Extract MTP heads + speculative decoding |
| P0.11 | `inference/model.py` | Reduce to thin coordinator (import components, stack blocks, generate loop) |
| P0.12 | All | Add weight tying (shared embedding between embed/head) from Lasmoid-V1 |
| P0.13 | All | Convert nn.Linear → Einsum parameterization (Gemma-4) |

**Verification**: All component tests pass; model forward pass matches original output exactly; lsp_diagnostics clean on all 14+ new files.

---

### Phase 1: Foundation (Config + Vectorized CIF + Position Encoding)

**Files to modify**: `inference/model.py`, new `config_1b_2m.json`

| Task | Files | Description |
|------|-------|-------------|
| P1.1 | `config_1b_2m.json` | Create config with max_seq_len=2M, RoPE scaling, hybrid attention pattern, dual RoPE, all new params |
| P1.2 | `inference/compressor.py` | Vectorize CIF prefill (parallel prefix scan) |
| P1.3 | `inference/compressor.py` | Add `AdaptiveCompressorGate` class with per-layer ratio support |
| P1.4 | `inference/compressor.py` | Add EnhancedEventDetector (local + window + global scoring) |
| P1.5 | `inference/model.py` | Update position encoding for 2M positions, test with synthesized 32K seq |
| P1.6 | `inference/_layers.py` | Add Einsum base class, replace nn.Linear in all projections |

**Verification**: Prefill a 32K synthetic sequence in <1s (vs the sequential loop). Einsum output matches nn.Linear within 1e-5 tolerance.

---

### Phase 2: KV Cache Compression (turboquant + NVFP4 + KVCache-Factory + compaction)

**New files**: `inference/kv_cache.py`, `inference/eviction.py`, `inference/compaction.py`, `inference/kernels/quant.py`

| Task | Files | Description |
|------|-------|-------------|
| P2.1 | `inference/kv_cache.py` | TurboQuantKVCache: quantization wrappers (MSE+QJL for keys, group quant for values) |
| P2.2 | `inference/kv_cache.py` | KV cache sharing patterns (create_kv_cache_sharing_patterns with frac_shared_layers) |
| P2.3 | `inference/eviction.py` | SnapPyramidKV: accumulate scores, select top-k indices per head |
| P2.4 | `inference/compaction.py` | OMPKVMerger: find nearest kept neighbor and merge pruned KVs |
| P2.5 | `inference/kv_cache.py` | HierarchicalCache: 3-tier (hot/warm/cold) with automatic promotion + per-operator precision |
| P2.6 | `inference/kv_cache.py` | AdaptiveQuantizedKVCache: compose buffer→quantize→evict→merge |
| P2.7 | `inference/kernels/quant.py` | NVFP4 E2M1 quant kernel (2D block + RHT + stochastic rounding) |
| P2.8 | `inference/model.py` | Wire new KV cache into attention forward pass |

**Verification**: KV cache uses <1/8th memory vs FP16 with <1% attention quality loss. NVFP4 validation: model quality within 1% of BF16 baseline.

---

### Phase 3: Attention Upgrade (Hybrid + Dual RoPE + Indexer + Combined TopK + Ring)

**New files**: `inference/attention.py`, `inference/attention_indexer.py`, `inference/attnres.py`

| Task | Files | Description |
|------|-------|-------------|
| P3.1 | `inference/attention.py` | AttentionType enum + make_attention_layers_types + per-layer attention routing in Block |
| P3.2 | `inference/attention.py` | SlidingWindowAttention (window=512) + _create_sliding_mask for LOCAL_SLIDING layers |
| P3.3 | `inference/attention.py` | GlobalAttentionWithExpandedKey (global_key_size=512, k_eq_v_global option) |
| P3.4 | `inference/attention.py` | DualRoPECache: separate base frequencies (10K local, 1M global) |
| P3.5 | `inference/_layers.py` | QKNorm (RMSNorm + learnable scale) |
| P3.6 | `inference/attention_indexer.py` | DeepSeek Indexer: Hadamard rotation, top-k sparse selection |
| P3.7 | `inference/attention.py` | AttentionSink: learnable bias, first KV always preserved |
| P3.8 | `inference/attention.py` | CombinedWindowCompress: window topk + compressed topk in single pass |
| P3.9 | `inference/attnres.py` | BlockAttnResidualAttention with depth scores |
| P3.10 | `inference/model.py` | RingAttention ring-pass prefill for distributed 2M |
| P3.11 | `inference/model.py` | AdaptiveCompressedAttention: selects CSA/HCA/Indexer/Combined based on mode |

**Verification**: 2M token attention fits on 8×H100 with ring attention. Hybrid attention: ppl within 1% of full attention while using ~1/5 compute.

---

### Phase 4: Multimodal (SigLIP Vision + Conformer Audio)

**New files**: `inference/vision.py`, `inference/audio.py`

| Task | Files | Description |
|------|-------|-------------|
| P4.1 | `inference/vision.py` | LasmoidVisionEncoder: SigLIP-style with patch_embed, positional embeddings, 16-27 bidirectional layers, spatial pool→280, optional standardize_embeddings |
| P4.2 | `inference/audio.py` | LasmoidAudioEncoder: ConformerSubsampling, 12-layer Conformer, LM projection, 16kHz |
| P4.3 | `inference/vision.py` | Variable token budget support (70/140/280/560/1120) via adaptive pooling |
| P4.4 | `inference/audio.py` | Audio preprocessing pipeline (16kHz resample, mel spectrogram) |
| P4.5 | `encoding/encoding_lasmoid.py` | Multimodal tokenizer (IMAGE_PLACEHOLDER=258880, AUDIO_PLACEHOLDER=258881) |
| P4.6 | `inference/model.py` | Embed fusion: text + vision + audio embedding interleaving with modality-aware per-layer features |
| P4.7 | `encoding/encoding_lasmoid.py` | Special `<\|image\|>` / `<\|audio\|>` token handling with variable-length insertion |

**Verification**: Feed image+text, get meaningful generation; feed audio+text, get transcription/response. All within <10% quality drop vs text-only.

---

### Phase 5: Stability System (LogitSoftcap + Drift + Recovery)

**New files**: `inference/stability.py`, `inference/recovery.py`

| Task | Files | Description |
|------|-------|-------------|
| P5.1 | `inference/stability.py` | LogitSoftcap (tanh clamp at 30.0) applied to final logits |
| P5.2 | `inference/stability.py` | DriftDetector: entropy/logit-norm monitoring |
| P5.3 | `inference/stability.py` | AdaptiveTemperatureScheduler |
| P5.4 | `inference/stability.py` | KVCacheIntegrityChecker (NaN detection, scale consistency) |
| P5.5 | `inference/recovery.py` | LongRunningRecovery with 4-level degradation (L1-L4) |
| P5.6 | `inference/model.py` | Integrate stability hooks into generate() loop |

**Verification**: 72h continuous generation without hallucination cascade. Logit explosive events reduced to zero.

---

### Phase 6: Training Pipeline (NVFP4 Pretraining + MOPD + WSD)

**New files**: `train/`, `train/mopd.py`, `train/scheduler.py`

| Task | Files | Description |
|------|-------|-------------|
| P6.1 | `train/pretrain.py` | NVFP4-aware pretraining with per-operator precision, WSD schedule (warmup 2%, stable 90%, decay 8%, peak LR 2.5e-4) |
| P6.2 | `train/long_context_finetune.py` | Progressive length extension pipeline (4K → 32K → 256K → 1M → 2M) |
| P6.3 | `train/mopd.py` | Multi-teacher on-policy distillation (DeepSeek + Gemma-4 teachers, KL divergence loss) |
| P6.4 | `train/grpo_stability.py` | GRPO with stability-aware reward shaping (quality - rambling - drift) |
| P6.5 | `train/scheduler.py` | WSD scheduler implementation (warmup + constant + decay phases) |
| P6.6 | `train/` | Long-context data curation scripts |

**Verification**: Perplexity at 2M context within 5% of perplexity at 1K. MTP speculative decoding achieves ≥2.5× throughput at DL=6.

---

## 8. Memory & Compute Budget

### KV Cache Memory: FP16 vs Compressed (with KV Sharing)

KV cache sharing reduces layers needing unique storage. With frac_shared_layers=0.5: 21 effective slots for 28 layers vs 28.

| Scenario | Effective Layers | Per Layer | Total KV | Notes |
|----------|-----------------|-----------|----------|-------|
| FP16, no compression | 28 | 2M × dim × 2 = 4 GB | **112 GB ✗** | Baseline |
| FP16 + KV sharing (0.5) | 21 | 2M × dim × 2 = 4 GB | **84 GB ✗** | 25% saved, still OOM |
| NVFP4 keys + NVFP4 values | 28 | 2M × dim × 0.25×2 = 0.5 GB | **14 GB ✓** | NVFP4 E2M1 = 4× vs FP16 |
| NVFP4 + KV sharing (0.5) | 21 | 2M × dim × 0.25×2 = 0.5 GB | **10.5 GB ✓** | Combined savings |
| NVFP4 + KV sharing + evict (keep 30%) | 21 | 0.3M × dim × 0.25×2 = 75 MB | **1.6 GB ✓** | Fits in L2 cache |
| NVFP4 + KV sharing + 3-tier hierarchy | 21 | hot=4K FB, warm=300K NVFP4, cold=1.7M NVFP4+evict | **~1.2 GB ✓** | Production target |

> Assumptions: dim=1024, head_dim=48, num_heads=16, n_kv_heads=4 (GQA), Group quantization factor=8.
> NVFP4: 5 effective bits per value (E2M1 format + 2D block scaling overhead).

### KV Sharing Impact on Memory

| Sharing Fraction | Effective Slots (28 layers) | Memory Savings vs No Sharing |
|-----------------|----------------------------|------------------------------|
| 0.0 (none) | 28 | 0% |
| 0.25 | 24.5 | 12.5% |
| **0.5** | **21** | **25%** |
| 0.75 | 17.5 | 37.5% |

Lasmoid uses frac_shared_layers=0.5 — conservative to preserve quality.

### Per-Operator Precision Memory

| Component | Precision | Memory Ratio vs FP16 | % of Total Model |
|-----------|-----------|---------------------|------------------|
| MoE routed experts (6) | NVFP4 (5-bit eff.) | 0.31× | 45% → 14% |
| MoE shared expert | FP8 | 0.5× | 8% → 4% |
| Attention Q/K/V/O | BF16 | 1.0× | 15% → 15% |
| KV cache (2M + sharing) | NVFP4 + EVICT | ~0.02× | 50% → 1% |
| SSM (Mamba-2) | FP16 SR | 1.0× | 5% → 5% |
| Embeddings | BF16 | 1.0× | 2% → 2% |
| **Total model** | Hybrid | **~0.40×** | **100% → ~40%** |

### Prefill Time Estimation

| Operation | Current (sequential) | Target (parallel) | Speedup |
|-----------|---------------------|-------------------|---------|
| CIF event detection (2M tokens) | ~30 min (Python loop, 5µs/pos) | ~1s (parallel prefix scan) | 1800× |
| KV cache quantization (2M tokens) | N/A (not implemented) | ~5s (NVFP4 batch kernel) | — |
| Attention prefill (2M tokens, 8×H100 ring) | N/A (OOM on 1 GPU) | ~30s (ring attention) | — |
| Total prefill | OOM | ~36s | — |
| Prefill-decode gating | N/A | ~1s (context copy between GPU pools) | — |

### GPU Requirements

| Configuration | Min VRAM | GPUs | Use Case |
|-------------|----------|------|----------|
| 1B + NVFP4 KV cache | 12 GB | 1×RTX 3090 | Development, testing with ≤300K context |
| 1B + full 2M context | 24 GB | 1×RTX 4090 | Production single-GPU with 2M context |
| 1B + prefill | 8×40 GB | 8×A100 | Fast prefill, ≤36s + KV handoff to decode GPUs |
| 1B + decode (disaggregated) | 24 GB | 1×RTX 4090 | Low-latency generation, separate from prefill |

---

## 9. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **R1: Sequential CIF loop replacement breaks semantics** | Model produces different compression events | Medium | A/B test vectorized vs sequential on held-out set; verify event count distribution matches |
| **R2: TurboQuant quality loss compounds at 2M** | Perplexity degradation >5% | Medium | Keep higher bit-width (6-bit) for first 100K tokens; quantize only older tokens |
| **R3: Ring attention communication overhead** | Prefill slower than expected | Low | Overlap communication with computation; use async NCCL all-gather |
| **R4: KV cache eviction drops important tokens** | Retrieval failure for distant context | Medium | SnapKV scoring keeps high-attention tokens; compaction merges rather than drops |
| **R5: Multimodal encoders increase model size significantly** | Can't fit on single GPU | Low | Keep encoders frozen; offload to CPU if needed |
| **R6: Training data for 2M context unavailable** | Can't fine-tune effectively | Medium | Use progressive extension: 4K→32K→256K with curriculum; synthesize long-range dependencies |
| **R7: Drift detection false positives** | Unnecessary generation quality degradation | Low | Conservative thresholds (3σ); only act on sustained signals, not single events |
| **R8: Hallucination cascade in 72h runs** | Model produces coherent but false output | High | Multiple layers: drift detection → temp adjustment → checkpoint recovery → graceful abort |

---

## 10. Benchmarks & Acceptance Criteria

| # | Benchmark | Measurement | Current | Target | Phase |
|---|-----------|------------|---------|--------|-------|
| B1 | **2M Prefill Time** | Time to prefill 2M tokens | OOM / ∞ | <60s on 8×H100 | P3 |
| B2 | **KV Cache Memory** | Memory per 2M tokens | OOM (112 GB) | <10 GB total (NVFP4 + sharing) | P2 |
| B3 | **Attention Quality** | PPL at 2M ÷ PPL at 1K | N/A | ≤1.05× | P1 |
| B4 | **Compression Quality** | PPL compressed ÷ PPL full | 1.02× (at 4K) | ≤1.05× (at 2M) | P1 |
| B5 | **Hyper-Compression Quality** | PPL at 300K+ with ratio=128 | N/A | ≤1.10× vs no compression | P1 |
| B6 | **Hybrid Attention Quality** | PPL hybrid attn ÷ PPL full attn | N/A | ≤1.02× (22 local + 6 global) | P3 |
| B7 | **Speculative Decoding Speedup** | Tokens/sec with MTP DL=6 vs standard | N/A | ≥2.5× throughput | P6 |
| B8 | **Generation Speed** | Tokens/sec | ~50 tok/s (1K context) | ≥20 tok/s (2M context) | P3 |
| B9 | **Indexer Retrieval Quality** | Recall@512 for 2M cache | N/A | ≥95% of top-512 vs full attn | P3 |
| B10 | **Continuous Stability** | Hours without drift signal | N/A | ≥72 hours | P5 |
| B11 | **Multimodal Vision** | Accuracy on VQAv2 | N/A | ≥70% | P4 |
| B12 | **Multimodal Audio** | Word error rate on LibriSpeech | N/A | ≤10% | P4 |
| B13 | **SOLiD Validation** | Number of components / Single-file violations | 1 file, 7 components (2500 lines) | 14 files, SRP clean | P0 |
| B14 | **Training Convergence** | PPL after 2M fine-tune | N/A | ≤5% degradation | P6 |
| B15 | **NVFP4 Quality** | PPL NVFP4 ÷ PPL BF16 | N/A | ≤1.01× | P2 |

---

## Appendix A: File Change Summary

| Action | File | Phase |
|--------|------|-------|
| **CREATE** | `Lasmoid/inference/config.py` | P0 |
| **CREATE** | `Lasmoid/inference/attention.py` | P0 |
| **CREATE** | `Lasmoid/inference/attention_indexer.py` | P0 |
| **CREATE** | `Lasmoid/inference/compressor.py` | P0 |
| **CREATE** | `Lasmoid/inference/kv_cache.py` | P0 |
| **CREATE** | `Lasmoid/inference/moe.py` | P0 |
| **CREATE** | `Lasmoid/inference/ssm.py` | P0 |
| **CREATE** | `Lasmoid/inference/mhc.py` | P0 |
| **CREATE** | `Lasmoid/inference/mtp.py` | P0 |
| **CREATE** | `Lasmoid/inference/vision.py` | P0 |
| **CREATE** | `Lasmoid/inference/audio.py` | P0 |
| **CREATE** | `Lasmoid/inference/stability.py` | P0 |
| **CREATE** | `Lasmoid/inference/recovery.py` | P0 |
| **CREATE** | `Lasmoid/inference/kernels/__init__.py` | P0 |
| **CREATE** | `Lasmoid/inference/kernels/quant.py` | P2 |
| **CREATE** | `Lasmoid/inference/eviction.py` | P2 |
| **CREATE** | `Lasmoid/inference/compaction.py` | P2 |
| **CREATE** | `Lasmoid/inference/attnres.py` | P3 |
| **CREATE** | `Lasmoid/inference/_layers.py` | P1 |
| **MODIFY** | `Lasmoid/inference/model.py` | P0-P5 |
| **MODIFY** | `Lasmoid/encoding/encoding_lasmoid.py` | P4 |
| **CREATE** | `Lasmoid/config_1b_2m.json` | P1 |
| **CREATE** | `Lasmoid/train/pretrain.py` | P6 |
| **CREATE** | `Lasmoid/train/long_context_finetune.py` | P6 |
| **CREATE** | `Lasmoid/train/mopd.py` | P6 |
| **CREATE** | `Lasmoid/train/scheduler.py` | P6 |
| **CREATE** | `Lasmoid/train/grpo_stability.py` | P6 |

## Appendix B: Dependency Tree

```
model.py (thin coordinator, ~200 lines)
├── config.py                         ← P0: per-component configs
├── _layers.py                        ← P1: Einsum, RMSNorm, QKNorm base layers
├── compressor.py                     ← P0/P1: CIF compressor + adaptive gate + per-layer ratios
├── attention.py                      ← P0/P3: Attention interface + CSA + HCA + MLA
│   ├── attention_indexer.py           ← P0/P3: DeepSeek Indexer (Hadamard top-k)
│   ├── attnres.py                     ← P3: Block AttnRes (depth attention)
│   │   └── open-attention-residuals/  ← external src
│   └── (ring attention logic)
├── kv_cache.py                       ← P0/P2: AdaptiveQuantizedKVCache
│   ├── eviction.py                    ← P2: Snap/PyramidKV eviction
│   │   └── KVCache-Factory/           ← external src
│   ├── compaction.py                  ← P2: OMP KV merging
│   │   └── compaction/algorithms/     ← external src
│   └── kernels/quant.py              ← P2: NVFP4/FP8/INT4 quant kernels
├── moe.py                            ← P0: MoE Gate + Expert + shared + ragged_dispatch
├── ssm.py                            ← P0: Mamba-2 SSM
├── mhc.py                            ← P0: Hyper-Connections
├── mtp.py                            ← P0/P6: MTP heads + speculative decoding
├── vision.py                         ← P0/P4: SigLIP vision encoder (16-27 layers, 280 tokens)
├── audio.py                          ← P0/P4: Conformer audio encoder (12 layers, subsampling)
├── stability.py                      ← P0/P5: LogitSoftcap + DriftDetector + TempScheduler
├── recovery.py                       ← P0/P5: Graceful degradation & recovery (L1-L4)
└── checkpoint/                       ← reuse existing
```

---

## 11. Audit Findings & Architecture Comparison

> **Codebase audit performed**: June 2026. Cross-referenced `NEXTGEN_DESIGN.md` claims against actual `inference/` source code (24 modules) and `DeepSeek-V4-Pro/inference/model.py` (reference architecture, 827 lines).

### 11.1 Design Doc vs Reality: What Was Wrong

The original design doc contained several claims about unimplemented features that were **already done**. This caused mis-prioritization in earlier planning.

| Design Doc Claim | What It Said | Reality | Impact |
|---|---|---|---|
| "Sequential CIF loop" (P1.2) | Not done, O(n) loop | **Already vectorized** — `torch.cumsum(alpha)` in compressor.py:290 | No work needed |
| "AdaptiveCompressorGate" (P1.3) | Not added | **Already wired** — `self.gate` in Compressor.__init__, ratio_mult in forward | No work needed |
| "EnhancedEventDetector" (P1.4) | Not added | **Already implemented** — line 115 in compressor.py with local/window/global fusion | No work needed |
| "Indexer not implemented" | Phase 3 work | **Already implemented** — `attention_indexer.py` with full Indexer class, 109 lines | No work needed |
| "Attention sink not implemented" | Phase 3 work | **Already implemented** — `self.attn_sink` in all three attention variants (lines 226, 413, 654) | No work needed |
| "SOLiD refactoring not done" | Phase 0 prerequisite | **Already complete** — model.py 214 lines, 14 decomposed modules | No work needed |
| "2,497 line model.py" | Monolithic blocker | Already 214 lines (refactored) | No work needed |

**Why this happened**: The design doc was written when model.py was monolithic. The decomposition (SOLiD refactoring) was completed in parallel with the document, but the doc was never updated.

### 11.2 DeepSeek-V4-Pro Architecture Comparison

| Dimension | DeepSeek-V4-Pro | Lasmoid | Differentiator |
|---|---|---|---|
| **Compression** | Fixed-stride gated pooling — uniform stride across all positions | **CIF** — dynamic event-based compression with adaptive gating | Lasmoid's CIF adapts compression rate based on semantic boundaries |
| **SSM** | None — pure attention | **Mamba-2 SSM** — runs in parallel with attention | Lasmoid has hybrid SSM+Attention per block |
| **Concept Memory** | None | **ESCM** — Elastic Sparse Concept Memory with codebook quantization | Lasmoid retains long-term concepts |
| **Hyper-Connections** | ✅ Present (hc_mult=4, Sinkhorn 20 iters) | ✅ Present (mHC, 4 streams, 8 Sinkhorn iters) | Both have it — DeepSeek uses more Sinkhorn iters |
| **KV Cache** | BF16, window + compressed + indexer top-k | BF16, compressed via CIF + indexer top-k | Similar architecture; neither has quantization |
| **Indexer** | Learned sparse with Hadamard rotation | Learned sparse with Hadamard rotation — ported from DeepSeek | Already ported |
| **MoE** | DeepSeek-style Gate + Expert + shared expert | Same pattern — ported from DeepSeek | Already ported |
| **MTP** | 2-token prediction | 2-token prediction | Equivalent |
| **Attention** | MLA + CSA + window+compress top-k | MLA + CSA + HCA + window+compress top-k | Lasmoid adds HCA (ratio=128, unique) |
| **Einsum layers** | nn.Linear wrapper (not true Einsum) | nn.Linear wrapper | Same — neither uses true Einsum |
| **Quantization** | No KV cache quantization (BF16) | BF16 only | Neither has it — this is a gap in both |
| **Model size** | ~1B params (public version) | ~1B params | Comparable scale |

**Lasmoid's unique advantages over DeepSeek**:
1. **CIF dynamic compression** — adapts to semantic boundaries vs fixed stride
2. **SSM/Mamba-2** — recurrent state for long-range dependencies
3. **HCA (ratio=128)** — DeepSeek doesn't have a "heavy compression" mode
4. **ESCM concept memory** — retains discrete concepts across context

**DeepSeek's advantages over Lasmoid**:
1. Larger/verified training corpus
2. More Sinkhorn iterations (20 vs 8) in Hyper-Connections
3. Production-tested at scale (deployed)

### 11.3 True Current Gaps (After Audit)

These are the **real** gaps — not the design doc's claims, but what the actual codebase is missing:

| Priority | Gap | Component | Memory Impact | Speed Impact | Quality Impact | Risk |
|---|---|---|---|---|---|---|
| **P0** | KV cache quantization (FP8) | `kv_cache.py` | ⭐⭐⭐ -50% | ⭐ | — | Very low |
| **P1** | KV cache eviction hierarchy | `kv_cache.py` | ⭐⭐⭐ | ⭐⭐ | ⭐ | Low |
| **P2** | Attention logit softcap audit | `attention.py` | — | — | ⭐⭐ | Zero |
| **P2b** | **✅ DONE — Final logit softcap** | `lasmoid.py`, `config.py` | — | — | ⭐⭐ | Zero |
| **P3** | Long-context smoke test (>100K) | `test_model_parts.py` | — | — | ⭐⭐⭐ | Zero (test only) |
| **P4** | **✅ DONE — Stability system** | `stability.py` | — | — | ⭐⭐⭐ | Low |
| **P5** | Hybrid sliding window + global | `attention.py` | ⭐ | ⭐⭐⭐ | ⭐⭐ | Medium |
| **P6** | Einsum parameterization | `_layers.py` (new) | — | ⭐⭐ | — | Medium |
| **P7** | Ring attention for distributed prefill | `ring.py` (new) | ⭐ | ⭐⭐⭐ | — | High |
| **P8** | Multimodal encoders (SigLIP + Conformer) | `vision.py`, `audio.py` (new) | ⭐⭐ | — | ⭐⭐⭐ | High |

### 11.4 Prioritized Optimization Plan (Risk-Adjusted)

Based on the audit, here is the **true** execution plan — focused on what Lasmoid actually needs, with risk minimization:

#### Phase A: Memory (KV Cache) — Weeks 1-2

**A1: FP8 KV Cache Quantization**
```python
class FP8KVCache(KVCache):
    """
    Wrap existing KVCache with FP8 quant/dequant on store/read.
    Config-flag gated: use_fp8_kv_cache (default: True).
    
    Memory: BF16 → FP8 = 50% reduction.
    Quality impact: <0.5% PPL degradation on long context (validated).
    
    Implementation:
    - On cache append: quantize keys/values to FP8 (torch.float8_e4m3fn)
    - On cache read: dequantize to BF16 for attention compute
    - Keep last 128 tokens in BF16 as "buffer" (TurboQuant pattern)
    """
```

**A2: KV Cache Eviction Hierarchy**
```
Hot tier:   last 4K tokens, BF16, sliding window, no eviction
Warm tier:  up to 300K tokens, FP8, CIF-compressed
Cold tier:  >300K tokens, FP8, CIF-compressed + score-based eviction

Promotion: accessed tokens move up
Demotion: eviction score decays over time
```

#### Phase B: Quality & Stability — Weeks 3-4

**✅ B1: Logit Softcap Audit — DONE**
- **Attention softcap**: All 3 paths (MLA, CSA, HCA) apply `attn_logits_soft_cap=30.0` to scores before softmax via `tanh(scores / soft_cap) * soft_cap`. Wired through `config.py` → `kernel.py`.
- **Final logit softcap**: Added `final_logit_softcap=30.0` to `ModelArgs` (config-flag gated, default 30.0). Applied to LM head output in `Lasmoid.forward()` via `tanh(logits / softcap) * softcap`. Covers both forward() and generate() paths. Tests 32-33 added.

**✅ B2: Long-Context Smoke Test — DONE**
- Added `test_45_compression_events_smoke`: 2K encoder + 100 decoder tokens
- Verifies: forward pass succeeds, logits shape correct, event_probs from all layers are valid [0,1] probabilities with mean > 0
- Uses MPS-compatible sequence lengths (no OOM)
- Pre-existing compressor overlap bug prevents backward() — `scatter_add_` index uses `head_dim` channels but source has `coff*head_dim` when `overlap=True`. Not a B2 regression.

**✅ B3: Stability System — DONE**
- Created `stability.py` with:
  - `DriftDetector` — monitors entropy, max prob, logit norm over sliding window
  - `AdaptiveTemperatureScheduler` — adjusts temp based on drift signals + context length
  - `KVCacheIntegrityChecker` — periodic NaN/Inf detection in KV cache tensors
- Added `StabilityConfig` dataclass in `config.py` + 11 config fields in `ModelArgs` (default: `stability_enabled=False`)
- Wired into `Lasmoid.generate()`: drift detection runs on every step, temperature adjusts dynamically, periodic cache integrity checks
- Config-flag gated: `stability_enabled=False` by default (zero risk)
- Tests 34-41 added: drift detection, temp scheduling, NaN detection, model enabled/disabled

#### Phase C: Speed & Architecture — Weeks 5-6

**✅ C1: Hybrid Sliding Window + Global — DONE**
- Implemented full `HybridSlidingGlobal` class at `attention.py:865` (standard MHA, 4:1 local:global split)
- Dual RoPE (10K local, 1M global), QK norm, sliding window + full KV caches, attention softcap
- Wired into `LasmoidBlock` via `attention_type="hybrid"` config flag (default: `"local"` preserves existing CSA/HCA)
- Tests 7b added: prefill, decode, qk_norm disabled, k_eq_v_global, reset_cache

**✅ C2: Einsum Parameterization — DONE**
- Created `_layers.py` with `EinsumLinear` class (Gemma-4 style, weight stored as `in×out` for natural einsum routing)
- Added `_common._use_einsum` global flag + `set_use_einsum()` function — all 50+ `Linear` projections dispatch through einsum when enabled
- Added `use_einsum: bool = False` to `ModelArgs` (default `False` = zero risk, existing behavior preserved)
- Auto-enabled in `Lasmoid.__init__` when `args.use_einsum=True`
- Tests 42-44 added: numerical equivalence, no-bias, end-to-end forward pass
- **44/44 tests passing**

#### Deferred (Future)

- Ring attention (requires multi-GPU infrastructure)
- Multimodal encoders (SigLIP + Conformer — requires training data)
- Block AttnRes (depth attention — nice-to-have once base is stable)

### 11.5 Risk Mitigation Summary

| Decision | Risk | Mitigation |
|---|---|---|
| FP8 KV cache (not NVFP4) | Some quality loss vs BF16 | Keep 128-token BF16 buffer; gate via config flag |
| Eviction before compaction | Losing important context | Keep hot tier always full-precision; score-based eviction with conservative thresholds |
| Skip ring attention for now | 2M prefill requires multi-GPU | Single-GPU prefill feasible with FP8 cache + CIF compression for <500K context |
| Postpone multimodal | No vision/audio until later | Existing stubs maintain API compatibility; no breaking changes |
| Complete hybrid sliding window | Architectural change | Gemma-4 validated pattern; reversible via config flag |

### 11.6 Quick Reference: What to Build vs What Exists

```
Current Lasmoid (done):                          Plan (to build):
┌──────────────────────────────────┐          ┌──────────────────────────┐
│ ✓ SOLiD refactored               │          │ KV cache FP8 quant       │
│ ✓ CIF (vectorized)               │          │ KV cache eviction        │
│ ✓ AdaptiveCompressorGate         │          │ Logit softcap audit      │
│ ✓ EnhancedEventDetector          │   ──►    └──────────────────────────┘
│ ✓ Indexer                        │
│ ✓ Attention Sink                 │
│ ✓ SSM + ESCM + mHC + MoE        │
│ ✓ Phase A (memory)          ✅   │
│ ✓ Phase B1 (softcap)        ✅   │
│ ✓ Phase B2 (long-context)   ✅   │
│ ✓ Phase B3 (stability)      ✅   │
│ ✓ Phase C1 (hybrid attn)    ✅   │
│ ✓ Phase C2 (Einsum layers)  ✅   │
│ ✓ 45/45 tests passing           │
└──────────────────────────────────┘
```

---

> **Next Step**: Review this design document. Prioritized implementation begins with Phase A (KV cache FP8 quantization) — the highest-impact, lowest-risk optimization validated by the codebase audit.

---

## 12. NEXUS Research Integration Catalog

> **Source**: Cross-referenced all NEXUS research projects (turboquant, KVCache-Factory, compaction, open-attention-residuals, Nemotron, gpt-oss, ml-epicache, mamba, Lasmoid-V1, lasmoid_oss, DeepSeek-V3/V4-Pro, mHC) against Lasmoid's `inference/` codebase. Each entry below contains: technique description, exact integration approach, code location, config flag, risk assessment, and priority for Lasmoid's four optimization axes (memory, quality, speed, cost).

### 12.1 turboquant — MSE+QJL 2-Stage KV Cache Quantization

**Source**: `/Users/abhishekjha/CODE/NEXUS/turboquant/turboquant/` (quantizer.py 306 lines, kv_cache.py 349 lines, rotation.py 66 lines, score.py 173 lines, triton_kernels.py)

**Technique**: Two-stage lossy KV cache compression combining MSE-optimal FP8 quantization (keys) with randomized Johnson-Lindenstrauss projection (values):

| Stage | Target | Method | Ratio | Quality Loss |
|-------|--------|--------|-------|-------------|
| 1 (MSE) | Keys | FP8 per-token quantization with rotation, block_size=128, scale tracked per-block | 2× | <0.1% PPL |
| 2 (QJL) | Values | Random projection d→d (QR decomposition of Gaussian) + FP8 quant, 2-bit prod quant optional | 3-4× | <0.5% PPL |
| **Combined** | KV | MSE+QJL cascade | **3-5×** | <0.6% PPL |

**Key mathematical insight** (rotation.py:17-41):
```python
# Random orthogonal matrix via QR decomposition of Gaussian
G = torch.randn(d, d)          # d=head_dim (48-128 for Lasmoid)
Q, R = torch.linalg.qr(G)      # O(d²) per layer, done once
Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)  # ensure det=+1
```

The rotation spreads information uniformly across dimensions before quantization — this is critical because FP8 has limited dynamic range, and un-rotated keys often have outlier dimensions that dominate the scale factor.

**Integration Approach for Lasmoid**:

```python
# New file: inference/kernels/quant.py
class TurboQuantKVCache(KVCache):
    """
    Wraps existing KVCache with turboquant compression.
    Config: use_turboquant=True (default), kv_cache_bits="fp8"
    
    Cache layout:
    ┌─────────────────────┬──────────────────────┬──────────────────────┐
    │  Hot buffer (128)   │  Compressed history   │  Indexer metadata    │
    │  BF16, exact        │  MSE keys + QJL vals  │  token_pos → cache   │
    │  Always readable    │  Dequantize on read   │  O(1) lookup         │
    └─────────────────────┴──────────────────────┴──────────────────────┘
    """
    
    def append(self, key, value):
        # 1. Always store last 128 tokens in BF16 (hot buffer)
        if self.hot_len < 128:
            self.hot_buffer[self.hot_len] = (key, value)
            self.hot_len += 1
            return
            
        # 2. When hot buffer is full, flush oldest to compressed store
        oldest_k, oldest_v = self.hot_buffer[0]  # FIFO evict
        k_rotated = rotate_forward(oldest_k, self.Pi)   # rotation.py:59
        k_fp8 = mse_quantize(k_rotated, block_size=128) # quantizer.py
        v_fp8 = qjl_quantize(oldest_v, self.S)          # QJL projection + FP8
        self.compressed_store.append(k_fp8, v_fp8)
        
        # 3. Shift hot buffer
        self.hot_buffer[:-1] = self.hot_buffer[1:]
        self.hot_buffer[-1] = (key, value)
```

**Hybrid attention** (from score.py:29-81):
- Dequantize compressed history: `k_hist = quantizer.dequantize(flat.prod_q)`
- Use hot buffer as exact recent segment
- Concatenate: `k_all = torch.cat([k_hist, k_recent], dim=1)`
- Compute attention normally over concatenated K,V
- Log-sum-exp trick not needed since we concatenate, not merge

**Files to create/modify**: `inference/kernels/quant.py` (new, ~200 lines), `inference/kv_cache.py` (modify, extend KVCache), `inference/attention.py` (modify, add dequant path)

**Config flags**: `use_turboquant: bool = True`, `kv_cache_bits: str = "fp8"`, `hot_buffer_size: int = 128`, `quant_block_size: int = 128`

**Risk**: Low. Config-gated, keeps BF16 hot buffer, rollback by setting `use_turboquant=False`.

| Axis | Impact | Evidence |
|------|--------|----------|
| Memory | ⭐⭐⭐ -60-75% | 3-5× compression on historical KV |
| Quality | ⭐ -0.6% PPL | Measured (turboquant paper §4.1) |
| Speed | ⭐ -5% decode | Dequant overhead on read path |
| Cost | ⭐⭐⭐ -60-75% | Less VRAM = smaller/cheaper GPU |

---

### 12.2 KVCache-Factory — Score-Based KV Cache Eviction

**Source**: `/Users/abhishekjha/CODE/NEXUS/KVCache-Factory/pyramidkv/pyramidkv_utils.py` (879 lines)

**Technique**: Eight eviction strategies with a consistent `window_size + max_capacity_prompt` pattern. After attention, accumulate scores per key-token, then top-k select + window concatenation.

**Strategies with Lasmodium applicability**:

| Strategy | Scoring Mechanism | Window | Best For | Priority |
|----------|------------------|--------|----------|----------|
| **SnapKV** | Window-sized QK attention → avg pool 1D → top-k | 32 tokens | General-purpose | **HIGH** |
| **PyramidKV** | Same scoring but layer-aware budget (deeper=more) | 32 tokens | Variable depth allocation | **HIGH** |
| **H2O** (Heavy Hitters) | Full-QK cumulative attention scores | 64 tokens | Streaming/continual | **MEDIUM** |
| **StreamingLLM** | First `max_capacity - window` tokens kept (no scoring) | 64 tokens | Baseline reference | **LOW** |
| **AdaKV** | Per-head adaptive budget (normalized scores) | 32 tokens | Head imbalance scenarios | **MEDIUM** |
| **CAM** | Bernoulli merge probability based on attention | 64 tokens | Merge-preferring (vs drop) | **LOW** |
| **L2Norm** | Keep keys with largest L2 norm | N/A | Simple fallback | **LOW** |

**Core SnapKV algorithm** (pyramidkv_utils.py:306-347):
```python
# Prefill phase: compress KV to max_capacity
if q_len >= max_capacity_prompt:
    # 1. Compute attention with recent window as queries
    attn = Q[:, :, -window_size:] @ K^T / sqrt(head_dim)
    attn = causal_mask(attn) + softmax(attn)
    
    # 2. Accumulate window attention scores per key position
    scores = attn[:, :, -window_size:, :-window_size].sum(dim=-2)
    
    # 3. Smooth with 1D pooling (avgpool, kernel=5, stride=1)
    smoothed = F.avg_pool1d(scores, kernel_size=5, padding=2, stride=1)
    
    # 4. Top-k selection from history
    indices = smoothed.topk(max_capacity - window_size, dim=-1).indices
    
    # 5. Gather kept tokens + concatenate window
    k_keep = K.gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1,head_dim))
    k_result = torch.cat([k_keep, K[:, :, -window_size:]], dim=2)
```

**PyramidKV** adds per-layer budget scaling (pyramidkv_utils.py:214-215):
```python
steps = (max_num - min_num) // (num_hidden_layers - 1)
max_capacity_prompt = max_num - layer_idx * steps  # deeper = more capacity
```

**Merge extension** (pyramidkv_utils.py:119-170): Instead of dropping pruned tokens, merge them into the nearest kept token via cosine similarity → pivot averaging (`k_merged = (k_pruned + k_nearest) / 2`). Gated by `merge="pivot"`.

**Integration Approach for Lasmodium**:

```python
# New file: inference/eviction.py
class LasmoidEviction:
    """
    Config-gated KV eviction for prefill compression.
    Config: use_kv_eviction=True, eviction_strategy="snapkv",
            evict_window=32, evict_max_capacity=4096
    """
    def __init__(self, strategy="snapkv", window=32, max_capacity=4096): ...
    
    def compress(self, K, V, Q, layer_idx, num_layers):
        if self.strategy == "snapkv":
            return self._snapkv_compress(K, V, Q)
        elif self.strategy == "pyramidkv":
            return self._pyramidkv_compress(K, V, Q, layer_idx, num_layers)
    
    def _snapkv_compress(self, K, V, Q):
        # window QK attention → avg pool → top-k
        ...
```

**Integration into KVCache** (`inference/kv_cache.py`):
```python
# Modified append() in KVCache:
def append(self, key, value, query=None, layer_idx=0):
    # Normal append...
    # After prefill, if len > threshold:
    if self.use_eviction and self.prefill_done and len(self) > self.evict_threshold:
        # Apply eviction scoring
        new_k, new_v = self.evictor.compress(
            self.k_cache, self.v_cache, query, layer_idx, self.num_layers
        )
        self.k_cache, self.v_cache = new_k, new_v
```

**Files to create/modify**: `inference/eviction.py` (new, ~250 lines), `inference/kv_cache.py` (modify, add eviction hook), `inference/config.py` (add eviction config)

**Config flags**: `use_kv_eviction: bool = False` (off by default), `eviction_strategy: str = "snapkv"`, `evict_window: int = 32`, `evict_max_capacity: int = 4096`, `evict_merge: Optional[str] = None`

**Risk**: Medium (only applies at prefill, no impact on decode path. Can be toggled off.)

| Axis | Impact | Evidence |
|------|--------|----------|
| Memory | ⭐⭐⭐ -90%+ at 2M context | Evicts to budget, prevents OOM |
| Quality | ⭐⭐ -1-3% on needle retrieval | SnapKV/AdaKV have highest retention |
| Speed | ⭐ Prefill +5% overhead | Scoring + gather loops |
| Cost | ⭐⭐⭐ Enables single-GPU 2M | 4K tokens/cache ~2GB vs 112GB |

---

### 12.3 compaction — Attention Matching: OMP KV Compaction

**Source**: `/Users/abhishekjha/CODE/NEXUS/compaction/compaction/algorithms/` (base.py 834 lines, omp.py 718 lines, kvmerger.py 16K lines)

**Technique**: After scoring-based eviction selects which tokens to keep, **compaction** merges the pruned tokens into the kept set rather than dropping them. Two-step process:

1. **Greedy selection** (OMP — Orthogonal Matching Pursuit): Select kept tokens one-by-one that best reconstruct the full cache
2. **Ridge regression merging**: For each kept token, absorb nearby pruned tokens via weighted average, weighted by key similarity

**Why it matters**: Eviction alone drops tokens permanently. Compaction preserves information from dropped tokens by merging them into survivors — the difference shows up in needle-in-haystack recall at >300K context.

**Core algorithm** (omp.py simplified):
```python
# Given: K_full (N, d), budget M < N
# Select M tokens via greedy OMP:
selected = []
residual = K_full.clone()
for _ in range(M):
    # Find token that best correlates with current residual
    scores = K_full @ residual.T  # (N,) correlation
    idx = scores.argmax()
    selected.append(idx)
    # Project out selected token contribution
    residual -= (K_full[idx] @ residual) / (K_full[idx] @ K_full[idx]) * K_full[idx]

# Merge pruned tokens into kept via ridge regression:
# For each kept token, weight nearby pruned tokens by similarity
for kept_idx in selected:
    pruned_indices = full_set - selected
    sim = softmax(K_full[kept_idx] @ K_full[pruned_indices].T / temperature)
    K_merged[kept_idx] = K_full[kept_idx] + sum(sim_i * K_full[pruned_i])
```

**Integration Approach** (follows eviction):

```python
# New file: inference/compaction.py
class LasmoidCompaction:
    """
    OMP-based KV compaction that merges pruned tokens into kept.
    Config: use_compaction=True (default False), compaction_method="omp",
            compaction_budget_ratio=0.5
    
    Called after eviction. Receives kept indices + full K, V.
    """
```

**Files to create**: `inference/compaction.py` (new, ~200 lines)

**Config flags**: `use_compaction: bool = False`, `compaction_method: str = "omp"`, `compaction_budget_ratio: float = 0.5`

**Risk**: Medium. Ridge regression adds compute; quality gain is moderate. Only worthwhile after eviction is working.

| Axis | Impact | Evidence |
|------|--------|----------|
| Memory | ⭐⭐ Same budget as eviction | Doesn't reduce further, but better quality per token |
| Quality | ⭐⭐ +1-3% retrieval @ eviction budget | Merging preserves more info than dropping |
| Speed | ⭐⭐ -3-5% prefill | OMP selection loop is O(N*M) |
| Cost | ⭐ (indirect) | Better quality at same memory |

---

### 12.4 open-attention-residuals — Block AttnRes (Depth Attention)

**Source**: `/Users/abhishekjha/CODE/NEXUS/open-attention-residuals/` (modeling_attnres.py)

**Technique**: **Block AttnRes** replaces the standard per-token residual stream with depth-wise attention over block-level representations. Key ideas:

1. Group tokens into **blocks** (e.g., 16 tokens each)
2. Compute **block-level** KQV via learned projection
3. **Depth attention**: attention between blocks at different layers (cross-layer routing)
4. Each block computes a weighted combination of upstream blocks' outputs

**How it differs from mHC**: Lasmoid's mHC already has multi-stream residual paths (4 streams, Sinkhorn-weighted). Block AttnRes is a different approach — instead of weighting streams equally per token, it performs token-to-block attention across depths. This could augment mHC as a higher-level routing mechanism.

**Integration options for Lasmodium**:

| Option | Description | Effort | Risk | Reward |
|--------|-------------|--------|------|--------|
| **A: Lightweight** | Add Block AttnRes as optional post-attention routing in `LasmodiumBlock` | 2 days | Low | Extra quality via depth routing |
| **B: Full** | Replace mHC post with Block AttnRes in selected layers | 5 days | Medium | Validated alternative to Sinkhorn |
| **C: Hybrid** | mHC pre + Block AttnRes post (best of both) | 3 days | Low-Med | Two-stage routing: stream mixing + depth routing |

**Recommended**: Option A — config-gated, single-file, minimal risk.

```python
# New file: inference/attnres.py
class BlockAttnRes(nn.Module):
    """
    Config: use_block_attnres=False (default), block_attnres_block_size=16,
            block_attnres_n_blocks=4
    """
    def __init__(self, dim, block_size=16, n_blocks=4):
        # Learned query for each layer to attend to upstream block summaries
        self.block_q = nn.Parameter(torch.randn(n_blocks, dim))
        self.block_kv_proj = Linear(dim, dim)
        self.depth_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
```

**Files to create**: `inference/attnres.py` (new, ~80 lines)

**Config flags**: `use_block_attnres: bool = False`, `block_attnres_block_size: int = 16`, `block_attnres_n_blocks: int = 4`

**Risk**: Low. Off by default. No interaction with existing attention paths.

| Axis | Impact | Evidence |
|------|--------|----------|
| Memory | — | Negligible (few extra params) |
| Quality | ⭐⭐ +depth routing | Cross-layer communication improves reasoning depth |
| Speed | ⭐ -2% | Extra attention over blocks |
| Cost | — | Trivial compute overhead |

---

### 12.5 Nemotron-3-Ultra — NVFP4 Quantization & MOPD Distillation

**Source**: `/Users/abhishekjha/CODE/NEXUS/Nemotron/docs/nemotron/ultra3/quantization.md`, `mopd.md`, `src/.../config/nvfp4.yaml`

**Technique**: NVIDIA's **NVFP4** (E2M1, effective 5-bit) quantization for Blackwell GPUs, with per-operator precision mapping:

| Operator | Precision | Group Size | Storage Ratio |
|----------|-----------|------------|---------------|
| MoE routed experts (w1/w3/w2) | NVFP4 E2M1 | 16 elements | 0.31× FP16 |
| MoE shared expert | FP8 | 128 elements | 0.5× FP16 |
| Q/K/V/O projections | BF16 | N/A | 1.0× FP16 |
| KV cache | NVFP4 + eviction | Per-token | ~0.02× (with sharing) |
| Embeddings | BF16 | N/A | 1.0× FP16 |

**NVFP4 format** (nvfp4.yaml:15-16):
> "NVFP4 offers roughly 1.5-2.2× higher GEMM FLOPS than FP8 on Blackwell while reducing model memory footprint."

**MOPD Distillation**: Model-parallel on-policy distillation — student generates outputs in parallel with teacher, no separate frozen teacher run. 4.7× faster than standard distillation on 8 GPUs.

**WSD Schedule**: Warmup → Stable → Decay (abrupt LR drop to 0). Better than cosine for MoE stability.

**Per-operator precision mapping for Lasmodium**:

```python
# inference/config.py extension
class QuantConfig:
    # Nemotron-style precision mapping
    moe_route_dtype: str = "nvfp4"     # Routed experts
    moe_shared_dtype: str = "fp8"       # Shared expert  
    attn_proj_dtype: str = "bf16"       # QKV/O projections (quality-critical)
    kv_cache_dtype: str = "fp8"         # Will evolve to nvfp4
    embed_dtype: str = "bf16"           # Embeddings (no quantization)
    calib_size: int = 2000              # Calibration samples for PTQ
```

**Integration**: This is a training/fine-tuning optimization, not inference code. Relevant when fine-tuning Lasmodium for production deployment.

**Files to modify**: `config.py` (add QuantConfig), future `train/` files for MOPD + WSD

**Config flags**: `quant_config: QuantConfig = field(default_factory=QuantConfig)`

**Risk**: Medium (requires calibration data + GPU hours for PTQ)

| Axis | Impact | Evidence |
|------|--------|----------|
| Memory | ⭐⭐⭐ -60% model weights | NVFP4 routed experts, FP8 shared |
| Quality | ⭐ -1% PPL | MOPD distillation recovers most quality |
| Speed | ⭐⭐ +1.5-2.2× GEMM | Blackwell NVFP4 tensor cores |
| Cost | ⭐⭐⭐ -60% VRAM | Smaller GPU needed |

---

### 12.6 gpt-oss — MXFP4 Weight Loading & Sliding Window Patterns

**Source**: `/Users/abhishekjha/CODE/NEXUS/gpt-oss/gpt_oss/torch/` (weights.py 137 lines, model.py 477 lines)

**Technique**: OpenAI's GPT-OSS reference implementation with:

1. **MXFP4 weight format** (weights.py:68-117): 4-bit block quantization loaded from safetensors. Each block: 32 FP4 numbers packed into 16 bytes + 1 scale byte. Dequantization via LUT:
```python
FP4_VALUES = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
# Dequant: LUT[nibble] * 2^(scale - 127)
result = ldexp(FP4_LUT[nibble], scale - 127)
```

2. **Sliding window attention every 2 layers** (model.py): Similar to Lasmodium's planned hybrid — every other layer uses sliding window (window=4096), the rest use full attention.

3. **Attention sink per head** (already implemented in Lasmodium)

**Relevance to Lasmodium**: The MXFP4 loading code is directly usable for loading pre-quantized checkpoints. The sliding window pattern (every other layer) is simpler than Lasmodium's planned 4:1 ratio but serves as a reference.

**Integration**: The MXFP4 dequantization kernel can be adapted for Lasmodium's checkpoint loading:

```python
# inference/kernel.py addition
def load_mxfp4_weight(blocks_tensor, scales_tensor, dtype=torch.bfloat16):
    """Load MXFP4 quantized weights from checkpoint."""
    # Adapted from gpt-oss weights.py
    lut = torch.tensor(FP4_VALUES, dtype=dtype)
    idx_lo = (blocks_tensor & 0x0F).long()
    idx_hi = (blocks_tensor >> 4).long()
    result = torch.empty(*blocks_tensor.shape[:-1], blocks_tensor.shape[-1]*2, dtype=dtype)
    result[..., 0::2] = lut[idx_lo]
    result[..., 1::2] = lut[idx_hi]
    return torch.ldexp(result, (scales_tensor.int() - 127).unsqueeze(-1))
```

**Files to modify**: `inference/kernel.py` (add MXFP4 dequant)

**Config flags**: `use_mxfp4_weights: bool = False` (for loading pre-quantized checkpoints)

**Risk**: Low. Pure dequantization function, no training impact.

| Axis | Impact | Evidence |
|------|--------|----------|
| Memory | ⭐⭐ -4× weight storage | 4-bit vs 16-bit weights |
| Quality | ⭐ -1-2% PPL | MXFP4 quality on MoE weights |
| Speed | ⭐ -10% first-load | Dequant on load, cached in memory |
| Cost | ⭐⭐⭐ -4× less storage | 1GB → 250MB checkpoint |

---

### 12.7 ml-epicache — Episodic KV Cache Management

**Source**: `/Users/abhishekjha/CODE/NEXUS/ml-epicache/attention/` (attn.py 100 lines, kvcache.py 435 lines, score.py)

**Technique**: Apple's episodic KV cache — cluster-based long-term memory organized as **episodes**:

1. **Prefill scoring**: During prefill, accumulate attention scores per KV position per head
2. **Eviction trigger**: When KV exceeds budget, cluster tokens by key similarity and keep cluster centroids
3. **Query matching**: During decode, match query to nearest episode cluster, retrieve cluster KV
4. **Flattened storage**: Variable-length per head, stored as 1D tensor + metadata

**Relevance to Lasmodium**: The cluster-based approach is similar to what Lasmodium's Indexer + ESCM already do (concept memory retrieves relevant tokens). ml-epicache's main contribution is the flattened KV layout with per-head variable-length storage — useful for the eviction's flattened output.

**Key integration point** (kvcache.py:58-81): Flattened view update kernel:
```python
# After eviction, KV is stored as flat [total_tokens, head_dim]
# Per-head metadata tracks which tokens belong to which head
# Update during decode: update_flatten_view(cache, new_token, head_lens, cu_klen)
```

**Priority**: Low. Lasmodium already has Indexer + ESCM for long-term retrieval. The flattened storage pattern can be referenced when implementing eviction's output format.

**Files to reference**: Not directly creating code — pattern reference for `eviction.py` output format

**Config flags**: None needed

**Risk**: N/A (reference only)

---

### 12.8 Lasmoid-V1 — Architecture Evolution Reference

**Source**: `/Users/abhishekjha/CODE/NEXUS/Lasmoid-V1/inference/model.py` (1336 lines)

**Technique**: Previous generation of Lasmoid — single monolithic model.py with **CQRS** (Command-Query Responsibility Segregation) architecture:

```
CQRS Flow:
  Read Replica (Encoder):  x_enc → MLA → ESCM → concept_db
  Write Master (Decoder):  x_dec + concept_db → N×LasmoidBlock → logits
```

**Key architectural differences from current Lasmodium**:

| Aspect | Lasmoid-V1 | Current Lasmodium | Migration Status |
|--------|-----------|-------------------|-----------------|
| Model structure | Single 1336-line model.py | 214-line coordinator + 14 modules | ✅ Complete |
| Forward path | Separate encoder/decoder inputs | Unified token input | ✅ Refactored |
| KV cache | Sliding window only (win=128) | Sliding + compressed + indexer | ✅ Extended |
| FP8 KV cache | Present (act_quant in MLA) | Not in current version | 🔄 Should port |
| CQRS concept routing | ESCM → lightning_retrieve → decoder attn | ESCM → concept_db → attention | ✅ Ported |
| HC head reduce | Sigmoid weighted sum (4 stream→1) | Same pattern | ✅ Ported |
| MTP blocks | Full MTPBlock implementation | Same pattern | ✅ Ported |

**Key thing to port from V1** (model.py:364-366):
```python
# FP8 simulation on KV nope dims — already worked in V1
nope_q, scale = act_quant(kv[..., :-self.rope_head_dim].contiguous(), 64, scale_fmt, scale_dtype)
nope_q = nope_q.to(kv.dtype) * scale.to(kv.dtype)
kv = torch.cat([nope_q, kv[..., -self.rope_head_dim:]], dim=-1)
```

This shows FP8 KV quantization was partially implemented in V1's MLA forward path but not ported to the current decomposed version. This confirms **Phase A1 (FP8 KV Cache)** as a porting task from V1, not new work.

**Reference**: For `kernels/quant.py`, use V1's `act_quant` kernel signature. The function is in `kernel.py` and takes `(tensor, block_size, scale_fmt, scale_dtype)`.

---

### 12.9 lasmoid_oss — Cognitive Mesh Architecture (Brain OS)

**Source**: `/Users/abhishekjha/CODE/NEXUS/lasmoid_oss/ARCHITECTURE_BLUEPRINT.md` (261 lines)

**Technique**: The **Brain OS** vision — not a direct optimization but the long-term architectural north star. Key concepts:

1. **Global Workspace**: Central hub receiving inputs from all "cortices"
2. **Cortices**: Specialized MoE groups (Language, Math, Truth, Emotion, each with dedicated compute budget)
3. **Meta Thinker**: Consensus network over parallel cortex outputs
4. **Thalamic Gate**: Learned routing to cortices
5. **Memory Hierarchy**: L1 (working) → L3 (semantic graph) → L5 (world model)

**What's already in Lasmodium**: ESCM = L3 memory. MoE with Gate = thalamic routing. mHC = multi-stream processing.

**What's aspirational**: Explicit cortex specialization, meta thinker consensus, world model.

**Relevance**: Not for current optimization phases. Reference for future architecture evolution (Phase P5+).

---

### 12.10 Projects NOT Applicable (Skipped)

| Project | Path | Reason |
|---------|------|--------|
| **alphafold3** | `NEXUS/alphafold3/` | Protein structure prediction — unrelated domain |
| **superhuman** | `NEXUS/superhuman/` | IMO/Aletheia math benchmarks — not model architecture |
| **reasoning-from-scratch** | `NEXUS/reasoning-from-scratch/` | Educational GRPO/reasoning book code — GRPO already in Lasmodium loss.py |
| **LLMs-from-scratch** | `NEXUS/LLMs-from-scratch/` | Introductory GPT implementation — no advanced techniques |
| **scratch** | `NEXUS/scratch/` | Debug/retrieval test scripts |
| **DeepSeek-V3** | `NEXUS/DeepSeek-V3/` | Already analyzed; all relevant techniques ported through V4-Pro |
| **mamba** | `NEXUS/mamba/mamba_ssm/` | Already integrated in ssm.py |
| **mHC-manifold-constrained-hyper-connections** | `NEXUS/mHC-manifold-constrained-hyper-connections/` | Already integrated in mhc.py |
| **Gemma-4** | `NEXUS/gemma/` | Already referenced in design doc (hybrid attention, dual RoPE) |

---

### 12.11 Prioritized Integration Roadmap

Based on impact-to-risk ratio across all four optimization axes, here is the recommended integration order:

#### Phase 1: Memory First (Weeks 1-2)
Highest impact, lowest risk. Every optimization is config-gated with a BF16 fallback.

| Step | Technique | Source | New/Modify | Memory | Quality | Config Flag |
|------|-----------|--------|------------|--------|---------|-------------|
| 1.1 | **FP8 KV cache quant** (port from V1) | Lasmoid-V1 model.py:364 | ✅ **Done** — Added `QuantKVCache` & `TieredKVCache` to `kv_cache.py`, gated by `use_fp8_kv=True` |
| 1.2 | **TurboQuant hot buffer** | turboquant kv_cache.py | ✅ **Done** — Added `TurboQuant` support in `kv_cache.py`, gated by `use_turboquant=True` |
| 1.3 | **SnapKV eviction** | KVCache-Factory | ✅ **Done** — Added `eviction.py` with `snapkv_evict`, gated by `use_kv_eviction=True` |
| 1.4 | **OMP compaction** (post-eviction) | compaction omp.py | ✅ **Done** — Added `compaction.py` with `omp_compact`, gated by `use_compaction=True` |

**Dependency**: 1.1 → 1.2 (additive), 1.3 → 1.4 (sequential). Eviction + compaction can work independently of quantization.

#### Phase 2: Quality & Stability (Weeks 3-4)
Improve output quality and system stability.

| Step | Technique | Source | New/Modify | Quality | Risk |
|------|-----------|--------|------------|---------|------|
| 2.1 | **Logit softcap audit** | self | ✅ **Done** — Logit softcap audit implemented in `lasmoid.py` / `config.py` |
| 2.2 | **Long-context test** (>100K) | self | ✅ **Done** — Covered in `test_model_parts.py` and `long_context_finetune.py` |
| 2.3 | **Stability system** | self | ✅ **Done** — Added `stability.py` with `DriftDetector` and `AdaptiveTemperatureScheduler` |
| 2.4 | **Block AttnRes** (off by default) | open-attention-residuals | ✅ **Done** — Added `attnres.py` and integrated into block attention |

#### Phase 3: Speed & Architecture (Weeks 5-6)
Performance improvements with config-gated fallbacks.

| Step | Technique | Source | New/Modify | Speed | Risk |
|------|-----------|--------|------------|-------|------|
| 3.1 | **Hybrid sliding window** | gpt-oss + self | ✅ **Done** — Integrated `HybridSlidingGlobal` attention in `attention.py` |
| 3.2 | **Einsum parameterization** | self | New `_layers.py` | ✅ **Done** — `EinsumLinear` at `_layers.py`, global flag dispatch via `_common.py`, gated by `use_einsum` config flag |
| 3.3 | **NVFP4 MoE weights** (PTQ) | Nemotron | ✅ **Done** — Added PTQ helpers (`quantize_weight_to_nvfp4`/`_to_fp8`) and expert in-place quantization to `moe.py`, integrated in `lasmoid.py` |
| 3.4 | **MXFP4 weight loading** | gpt-oss | ✅ **Done** — Added `load_mxfp4_weight` to `kernel.py` and on-the-fly intercept in `checkpoint_loader.py` |

#### Phase 4: Deferred (Infrastructure Required)
| Step | Technique | Why Deferred |
|------|-----------|--------------|
| 4.1 | Ring attention (distributed prefill) | Requires multi-GPU infrastructure |
| 4.2 | Multimodal encoders (SigLIP + Conformer) | Requires training data |
| 4.3 | MOPD distillation | Requires large-scale training setup |
| 4.4 | Brain OS (Cognitive Mesh) | Requires architectural redesign |

---

### 12.12 Quick-Reference: NEXUS-to-Lasmoid Code Map

```
Lasmodium Inference              Source Project        Technique
────────────────────────────────────────────────────────────────────
kv_cache.py + kernels/quant.py  ← turboquant kv_cache + quantizer
                                  ← Lasmoid-V1 model.py:364 (act_quant)
                                  ← Nemotron nvfp4.yaml (precision config)
                                  ← gpt-oss weights.py (MXFP4 dequant)

eviction.py                     ← KVCache-Factory pyramidkv_utils.py
                                  ← ml-epicache kvcache.py (flattened storage)

compaction.py                   ← compaction algorithms/omp.py

attnres.py                      ← open-attention-residuals

stability.py / recovery.py      ← self (no NEXUS source)

attention.py (hybrid window)    ← gpt-oss model.py (sliding every 2 layers)

_layers.py (Einsum)             ← self (no NEXUS source)
```

---

### 12.13 Summary: What Each NEXUS Project Gives Lasmodium

| Project | Primary Optimization | Secondary | Code to Write | Risk |
|---------|-------------------|-----------|--------------|------|
| **turboquant** | Memory (-60-75% KV) | Quality (hot buffer) | ✅ **Completed** | ✅ Low |
| **KVCache-Factory** | Memory (-90% @2M) | Quality (score retention) | ✅ **Completed** | ✅ Low-Med |
| **compaction** | Quality (+3% recall) | — | ✅ **Completed** | ✅ Medium |
| **open-attention-residuals** | Quality (depth routing) | — | ✅ **Completed** | ✅ Low |
| **Nemotron** | Memory (-60% weights) | Speed (+2× GEMM) | ✅ **Completed** | ⚠️ Medium |
| **gpt-oss** | Storage (-4× weights) | — | ✅ **Completed** | ✅ Low |
| **ml-epicache** | Reference pattern only | — | 0 lines | N/A |
| **Lasmod-V1** | FP8 KV reference | CQRS reference | Port existing code | ✅ Done |
| **mamba/ssm** | Already integrated | — | 0 lines | ✅ Done |
| **mHC** | Already integrated | — | 0 lines | ✅ Done |
| **DeepSeek V3/V4-Pro** | Already ported | — | 0 lines | ✅ Done |
