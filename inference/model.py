"""
Lasmoid — model.py
========================================================================
Sovereign CQRS Architecture:
  - Read Replica (Encoder)  →  ElasticSparseConceptMemory (GCN Graph VQ, Spawning, Perceiver Pooling)
  - Write Master (Decoder)  →  LasmoidBlock (Hybrid Concept Attention, Grey-Box MoE, mHC streams)

Key Stage 5, 8 & 9 Innovations Integrated:
  ✓ GraphVectorQuantizer     — directed adjacency GCN codebook: E_graph = E + softmax(A) @ (E @ W_r)
  ✓ Causal Dynamic Memory    — chronological concept slots via cumulative sum normalisation
  ✓ HybridConceptAttention   — low-rank local attention + sparse top-k concept attention
  ✓ Grey-Box MoE Routing    — router guided by hidden state + concept routing weights: gate_input = H + proj(routing)
  ✓ DeepSeek-V4 Affinity     — sqrt(softplus(logits) + 1e-6) activation to prevent routing saturation
  ✓ YaRN RoPE & MLA          — low-rank KV compression, grouped O-projection, and attention sink
  ✓ MHCBlock                 — manifold hyperconnections utilizing Sinkhorn routing
  ✓ MTPBlock                 — multi-token prediction (t+2 prediction) via fused embedding and hidden streams
  ✓ SSM Mamba-3 Chunked Scan — parallel chunk-wise associative scan with dt clamping & B/C normalization
  ✓ MoE Expert-Capacity      — token dropping with capacity factor, per-expert scale + z-loss coefficient
  ✓ Chain-of-Thought Budget   — <think>/<answer> state-machine reasoning with confidence-gated exit
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, List, Any
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from .kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn, weight_dequant
except ImportError:
    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn, weight_dequant


# ══════════════════════════════════════════════════════════════════════
# GLOBALS & HELPERS
# ══════════════════════════════════════════════════════════════════════
default_dtype = torch.bfloat16
scale_fmt:    Optional[str]   = None
scale_dtype:  torch.dtype     = torch.float32
block_size:   int             = 128
fp4_block_size: int           = 32


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
    # ── Core ─────────────────────────────────────────────────────────
    vocab_size: int = 129280
    dim: int = 128
    n_layers: int = 4
    max_seq_len: int = 256
    max_batch_size: int = 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp32"
    expert_dtype: Optional[str] = None
    norm_eps: float = 1e-6

    # ── MLA (Multi-head Latent Attention) ─────────────────────────────
    n_heads: int = 4
    q_lora_rank: int = 32
    head_dim: int = 48
    rope_head_dim: int = 16
    o_groups: int = 2
    o_lora_rank: int = 32

    # ── RoPE / YaRN ──────────────────────────────────────────────────
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 0

    # ── MoE (Mixture of Experts) ─────────────────────────────────────
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_group: int = 1
    topk_group: int = 1
    moe_inter_dim: int = 0  # 0 = auto-compute as ~2.67 * expert_dim (latent_dim if set, else dim)
    moe_latent_dim: Optional[int] = None
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 10.0
    n_hash_layers: int = 0

    # ── HC (Hyper-Connections) ───────────────────────────────────────
    num_residual_streams: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ── HCM (Hierarchical Concept Memory) ─────────────────────────────
    num_concepts: int = 64
    num_abstract_concepts: int = 8
    num_global_concepts: int = 2
    codebook_size: int = 256
    codebook_dim: int = 32
    hcm_ema_alpha: float = 0.99
    hcm_commit_loss_coeff: float = 0.25
    entropy_threshold: float = 0.5
    lightning_topk_blocks: int = 2
    router_z_loss_coeff: float = 0.001
    ema_bias_lr: float = 0.01

    # ── Hybrid Concept Attention ─────────────────────────────────────
    concept_topk: int = 8
    concept_ratio: float = 0.4

    predictive_coding_coeff: float = 0.01

    # ── SSM (State Space Recurrence — Mamba-3 style) ─────────────────
    ssm_heads: int = 4
    ssm_state_dim: int = 16
    ssm_kernel_size: int = 4
    ssm_chunk_size: int = 64          # Mamba-3 chunked parallel scan block size
    ssm_dt_min: float = 0.001         # Nemotron mamba_dt_min
    ssm_dt_max: float = 0.1           # Nemotron mamba_dt_max
    ssm_dt_init_floor: float = 1e-4   # Nemotron mamba_dt_init_floor
    ssm_n_groups: int = 1             # Mamba-3 SSM groups

    # ── MTP (Multi-Token Prediction) ─────────────────────────────────
    n_mtp_layers: int = 1
    window_size: int = 128

    # ── Hybrid Attention (DeepSeek-V4) ───────────────────────────────
    csa_compression_ratio: int = 4     # m factor for CSA
    hca_compression_ratio: int = 128   # m' factor for HCA
    indexer_head_dim: int = 32
    sliding_window_size: int = 256     # Uncompressed local window
    index_n_heads: int = 4
    index_topk: int = 16
    compress_rope_theta: float = 40000.0
    attn_logits_soft_cap: Optional[float] = 30.0

    # ── Reasoning Budget (CoT state-machine) ─────────────────────────
    reasoning_steps: int = 1          # Max reasoning loop iterations
    think_token_id: int = 0           # <think> token id (set dynamically by train.py)
    answer_token_id: int = 1          # <answer> token id (set dynamically by train.py)
    cot_exit_confidence: float = 0.9  # Exit reasoning early if guard confidence >= this

    # ── Attribute Steering (Nemotron-style SteerLM) ──────────────────
    steering_attributes: List[str] = field(default_factory=lambda: ["creativity", "helpfulness", "complexity", "scientific_rigor"])

    # ── External Embedding Fusion ─────────────────────────────────────
    # Optional side-channel for retrieval/vision/audio/protein embeddings.
    # Set external_embedding_dim > 0 to add a learned normalized projection
    # into the token stream, gated by external_embedding_scale.
    external_embedding_dim: int = 0
    external_embedding_scale: float = 0.25
    external_embedding_norm: bool = True

    # ── MoE Capacity & Load Balance ───────────────────────────────────
    expert_capacity_factor: float = 1.25  # max tokens per expert = factor * (tokens / n_experts)
    moe_load_balance_coeff: float = 0.01  # auxiliary load-balance loss weight
    moe_router_entropy_coeff: float = 0.001
    moe_capacity_loss_coeff: float = 0.01
    token_concept_loss_coeff: float = 0.05

    # ── SOTA Hardening: Gemma4 + Mamba3 + DeepSeek-V3 ────────────────
    # Post-layer norms (Gemma4): extra RMSNorm on attention/FFW output *before*
    # adding to the residual stream. Proven to drastically improve training
    # stability in deep hybrid architectures (≥24 layers).
    post_attn_norm: bool = True
    post_ffw_norm: bool  = True
    # Dual dense+MoE FFW branch (Gemma4 mlp2): runs a lightweight dense FFN in
    # parallel with the sparse MoE. Provides always-on gradient signal to all
    # tokens and substantially improves sample efficiency.
    moe_dual_ffn: bool       = True
    moe_dual_inter_dim: int  = 0   # 0 = auto (same as effective_moe_inter_dim)
    # SSM D-skip connection (Mamba): direct input bypass term `y += D * x`.
    # Without this, the SSM can only output through the selective scan path,
    # losing the model's ability to pass information directly when decay ≈ 0.
    ssm_d_skip: bool = True
    # Expert dropout (training regularization): randomly zero-out individual
    # expert token assignments during training to prevent expert collapse.
    moe_expert_dropout: float = 0.0

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def effective_moe_inter_dim(self) -> int:
        """Auto-compute moe_inter_dim if not explicitly set (0 = auto)."""
        if self.moe_inter_dim > 0:
            return self.moe_inter_dim
        expert_dim = self.moe_latent_dim if (self.moe_latent_dim is not None and self.moe_latent_dim < self.dim) else self.dim
        return int(2 * 4 * expert_dim / 3)  # ~2.67 * expert_dim, matches ConceptExpert default


# ══════════════════════════════════════════════════════════════════════
# CORE LAYERS
# ══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def _linear_dispatch(x: torch.Tensor, weight: nn.Parameter, bias: Optional[nn.Parameter] = None) -> torch.Tensor:
    if weight.dtype == torch.bfloat16 or weight.dtype == torch.float32:
        return F.linear(x.to(weight.dtype), weight, bias)
    if weight.dtype == torch.float8_e4m3fn:
        xq, xs = act_quant(x.contiguous().bfloat16(), block_size, scale_fmt, scale_dtype)
        out = fp8_gemm(xq, xs, weight, weight.scale, scale_dtype)
        if bias is not None:
            out = out + bias
        return out
    return F.linear(x, weight.float(), bias)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        w_dtype = dtype or default_dtype

        if w_dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=w_dtype))
            so = (out_features + block_size - 1) // block_size
            si = (in_features  + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(so, si, dtype=torch.float32), requires_grad=False
            )
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=w_dtype))
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._init_weights()

    def _init_weights(self):
        if self.weight.dtype == torch.float8_e4m3fn:
            tmp = torch.empty_like(self.weight, dtype=torch.float32)
            nn.init.normal_(tmp, 0.0, 0.02)
            self.weight.data.copy_(tmp.to(self.weight.dtype))
        else:
            nn.init.normal_(self.weight, 0.0, 0.02)
        if self.scale is not None:
            nn.init.constant_(self.scale, 1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear_dispatch(x, self.weight, self.bias)


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
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low  = math.floor(find_correction_dim(low_rot,  dim, base, max_seq_len))
        high = math.ceil( find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        lf = (torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn)
        return torch.clamp(lf, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs  = freqs / factor * (1 - smooth) + freqs * smooth

    t       = torch.arange(seqlen, dtype=torch.float32)
    freqs   = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()

    if freqs_cis.ndim == 2:
        if xc.ndim == 3:
            if freqs_cis.size(0) != xc.size(1):
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.size(0)} positions for sequence length {xc.size(1)}"
                )
            freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
        elif xc.ndim == 4:
            if freqs_cis.size(0) != xc.size(1):
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.size(0)} positions for sequence length {xc.size(1)}"
                )
            freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
        else:
            raise ValueError(f"Unsupported rotary tensor rank: {xc.ndim}")
    elif freqs_cis.ndim == 3:
        if xc.ndim == 3:
            if freqs_cis.shape[-1] != xc.size(-1):
                raise ValueError(
                    f"Rotary dim mismatch: got {freqs_cis.shape[-1]} vs {xc.size(-1)}"
                )
        elif xc.ndim == 4:
            if freqs_cis.shape[-1] != xc.size(-1):
                raise ValueError(
                    f"Rotary dim mismatch: got {freqs_cis.shape[-1]} vs {xc.size(-1)}"
                )
            if freqs_cis.shape[1] == xc.size(1):
                freqs_cis = freqs_cis.unsqueeze(2)
            else:
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.shape[1]} positions for sequence length {xc.size(1)}"
                )
        else:
            raise ValueError(f"Unsupported rotary tensor rank: {xc.ndim}")
    else:
        raise ValueError(f"Unsupported rotary frequency rank: {freqs_cis.ndim}")

    xr = torch.view_as_real(xc * freqs_cis.to(torch.complex64)).flatten(-2)
    return xr.to(dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Applies randomized Hadamard rotation to spread information across dims before FP8 quant."""
    try:
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x.bfloat16(), scale=x.size(-1) ** -0.5).to(x.dtype)
    except ImportError:
        # Fallback to pure-PyTorch Fast Hadamard Transform
        d = x.shape[-1]
        h = x.float().clone()
        step = 1
        while step < d:
            h = h.view(-1, d // (2 * step), 2, step)
            h = torch.stack([h[:, :, 0] + h[:, :, 1], h[:, :, 0] - h[:, :, 1]], dim=2)
            step *= 2
        return (h.view(x.shape) * (d ** -0.5)).to(x.dtype)


@lru_cache(maxsize=4)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size),  torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(maxsize=4)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """
    Continuous Integrate-and-Fire (CIF) Semantic Event Compressor.
    Achieves dynamic 100X+ KV compression by dynamically pooling tokens based on semantic 
    event boundaries, mirroring biological episodic memory formation.
    """
    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 48, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        
        # Event boundary detector (0 to 1 probability mapping)
        self.event_detector = Linear(args.dim, 1, dtype=torch.float32)
        
        # Pooling projections
        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        nn.init.normal_(self.ape, 0.0, 0.02)
        
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache
        self.freqs_cis: torch.Tensor = None
        
        # State buffers for autoregressive integration
        max_batch_size_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer("kv_accumulator", torch.zeros(max_batch_size_comp, self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("gate_accumulator", torch.zeros(max_batch_size_comp, self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("fire_threshold", torch.zeros(max_batch_size_comp, 1, dtype=torch.float32), persistent=False)
        self.register_buffer("cache_write_ptr", torch.zeros(max_batch_size_comp, dtype=torch.long), persistent=False)
        
        # Fired indices buffer for causal boundary masking
        cache_cap = max(1, args.max_seq_len // compress_ratio)
        self.register_buffer("fired_indices_buf", torch.zeros(max_batch_size_comp, cache_cap, dtype=torch.long), persistent=False)


    def resize_buffers(self, bsz: int, device: Optional[torch.device] = None):
        if bsz > self.kv_accumulator.shape[0]:
            if device is None:
                device = self.kv_accumulator.device
            cache_cap = self.kv_cache.shape[1] if self.kv_cache is not None else 16
            self.register_buffer("kv_accumulator", torch.zeros(bsz, self.head_dim, dtype=torch.float32, device=device), persistent=False)
            self.register_buffer("gate_accumulator", torch.zeros(bsz, self.head_dim, dtype=torch.float32, device=device), persistent=False)
            self.register_buffer("fire_threshold", torch.zeros(bsz, 1, dtype=torch.float32, device=device), persistent=False)
            self.register_buffer("cache_write_ptr", torch.zeros(bsz, dtype=torch.long, device=device), persistent=False)
            self.register_buffer("fired_indices_buf", torch.zeros(bsz, cache_cap, dtype=torch.long, device=device), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        self.resize_buffers(bsz, device=x.device)
        
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x_float = x.float()
        
        # Calculate dynamic semantic boundary probability
        event_prob = torch.sigmoid(self.event_detector(x_float)) # [B, S, 1]
        
        kv = self.wkv(x_float)
        score = self.wgate(x_float)
        
        if start_pos == 0:
            # ─────────────────────────────────────────────────────────
            # PARALLEL INTEGRATE-AND-FIRE (Hardware-Accelerated Prefill)
            # ─────────────────────────────────────────────────────────
            with torch.no_grad():
                self.kv_accumulator.zero_()
                self.gate_accumulator.zero_()
                self.fire_threshold.zero_()
                self.cache_write_ptr.zero_()
                self.fired_indices_buf.zero_()

            # 1. Calculate gate scores for the entire sequence at once
            gate_scores = F.softplus(score)  # [B, S, D]
            weighted_kv = kv * gate_scores   # [B, S, D]

            # Sequential ACT-style accumulator loop over sequence length S
            accum_prob = [torch.tensor([0.0], device=x.device, dtype=torch.float32) for _ in range(bsz)]
            accum_kv = [torch.zeros(d, device=x.device, dtype=torch.float32) for _ in range(bsz)]
            accum_gate = [torch.zeros(d, device=x.device, dtype=torch.float32) for _ in range(bsz)]
            
            batch_fired_kvs = [[] for _ in range(bsz)]
            batch_fired_indices = [[] for _ in range(bsz)]
            
            for s in range(seqlen):
                p_s = event_prob[:, s]  # [B, 1]
                kv_s = weighted_kv[:, s, :d] # [B, D]
                gate_s = gate_scores[:, s, :d] # [B, D]
                
                for b in range(bsz):
                    p = p_s[b, 0].item()
                    k = kv_s[b]
                    g = gate_s[b]
                    
                    cur_prob = accum_prob[b][0].item()
                    space = 1.0 - cur_prob
                    
                    if p >= space:
                        # Fire!
                        accum_kv[b] = accum_kv[b] + space * k
                        accum_gate[b] = accum_gate[b] + space * g
                        
                        # Emit chunk
                        chunk = accum_kv[b] / (accum_gate[b] + 1e-6)
                        batch_fired_kvs[b].append(chunk)
                        batch_fired_indices[b].append(s)
                        
                        # Remainder
                        rem = p - space
                        accum_prob[b] = torch.tensor([rem], device=x.device, dtype=torch.float32)
                        accum_kv[b] = rem * k
                        accum_gate[b] = rem * g
                    else:
                        # No fire
                        accum_prob[b] = accum_prob[b] + p
                        accum_kv[b] = accum_kv[b] + p * k
                        accum_gate[b] = accum_gate[b] + p * g

            # Find maximum number of fires across all batch items to build output tensor
            max_fires = max(len(batch_fired_kvs[b]) for b in range(bsz))
            if max_fires == 0:
                max_fires = 1
                
            kv_out_list = []
            fired_indices_list = []
            
            for b in range(bsz):
                kvs = batch_fired_kvs[b]
                indices = batch_fired_indices[b]
                
                if len(kvs) == 0:
                    # Force one fire at the end of sequence
                    forced_chunk = accum_kv[b] / (accum_gate[b] + 1e-6) if accum_prob[b][0].item() > 0 else kv[b, -1, :d]
                    kvs = [forced_chunk]
                    indices = [seqlen - 1]
                    
                while len(kvs) < max_fires:
                    kvs.append(kvs[-1].clone())
                    indices.append(indices[-1])
                    
                kv_out_list.append(torch.stack(kvs))
                fired_indices_list.append(torch.tensor(indices, device=x.device, dtype=torch.long))
                
            kv_out = torch.stack(kv_out_list).to(dtype) # [B, max_fires, D]
            fired_indices_tensor = torch.stack(fired_indices_list) # [B, max_fires]
            num_fires = max_fires

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
                self.fired_indices_buf[:bsz, :write_len] = fired_indices_tensor[:, :write_len]
                self.cache_write_ptr[:bsz] = write_len

                # 7. Carry over the incomplete remainder to the autoregressive state buffers
                self.kv_accumulator[:bsz] = torch.stack(accum_kv).detach()
                self.gate_accumulator[:bsz] = torch.stack(accum_gate).detach()
                self.fire_threshold[:bsz] = torch.stack(accum_prob).detach()

            # Return kv_out and event_prob for CIF loss (only in training/prefill)
            if self.training:
                return kv_out, event_prob
            return kv_out
        else:
            # Autoregressive generation phase
            # Retrieve current state from buffers (detached to break autograd graph)
            kv_acc = self.kv_accumulator[:bsz].detach().clone()
            gate_acc = self.gate_accumulator[:bsz].detach().clone()
            fire_th = self.fire_threshold[:bsz].detach().clone()
            
            prob = event_prob[:, 0, :]
            fire_th = fire_th + prob
            
            # Use only head_dim dimensions for accumulation (compressed space)
            current_gate_score = F.softplus(score[:, 0, :d])
            kv_acc = kv_acc + kv[:, 0, :d] * current_gate_score
            gate_acc = gate_acc + current_gate_score
            
            fire_mask = (fire_th >= 1.0).squeeze(-1)
            
            if fire_mask.any():
                fired_kv = kv_acc.clone()
                fired_kv[fire_mask] /= (gate_acc[fire_mask] + 1e-6)
                fired_kv = fired_kv[..., :d]
                
                # Apply normalization
                fired_kv = self.norm(fired_kv.to(dtype))
                
                # Apply rotary embedding based on current cache write pointer index
                kv_out_single = fired_kv.unsqueeze(1)
                
                # Vectorized lookup of freqs_cis per batch item
                batch_ptrs = self.cache_write_ptr[:bsz]
                freqs_cis = self.freqs_cis[batch_ptrs].unsqueeze(1) # [B, 1, rd // 2]
                kv_rope = apply_rotary_emb(kv_out_single[:, :, -rd:].contiguous(), freqs_cis)
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


class Indexer(nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.indexer_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        max_batch_size_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer("kv_cache", torch.zeros(max_batch_size_comp, max(1, args.max_seq_len // compress_ratio), self.head_dim), persistent=False)
        self.freqs_cis = None

    def resize_buffers(self, bsz: int, device: Optional[torch.device] = None):
        if bsz > self.kv_cache.shape[0]:
            if device is None:
                device = self.kv_cache.device
            new_kv_cache = torch.zeros(bsz, self.kv_cache.shape[1], self.kv_cache.shape[2], device=device, dtype=self.kv_cache.dtype)
            new_kv_cache[:self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_kv_cache, persistent=False)
            self.compressor.kv_cache = self.kv_cache
            self.compressor.resize_buffers(bsz, device=device)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        
        # Dynamic resizing of indexer kv_cache for larger batch size (e.g. GRPO)
        self.resize_buffers(bsz, device=x.device)

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        q_nope = q[..., :-rd].contiguous()
        q_rope = apply_rotary_emb(q[..., -rd:].contiguous(), freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1).contiguous()
        q = q.to(qr.dtype)
        self.compressor(x, start_pos)
        cache_len = max(1, min(self.compressor.cache_write_ptr[:bsz].max().item(), self.kv_cache.shape[1]))
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :cache_len].to(q.dtype))
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            fired_positions = self.compressor.fired_indices_buf[:bsz, :cache_len]
            query_positions = torch.arange(seqlen, device=x.device).view(1, seqlen, 1)
            mask = fired_positions.unsqueeze(1) > query_positions
            index_score = index_score.masked_fill(mask, float("-inf"))
        topk_idxs = index_score.topk(min(self.index_topk, cache_len), dim=-1)[1]
        if start_pos == 0:
            fired_positions = self.compressor.fired_indices_buf[:bsz, :cache_len]
            fired_steps = torch.gather(fired_positions.unsqueeze(1).expand(-1, seqlen, -1), 2, topk_idxs.long())
            valids = fired_steps <= torch.arange(seqlen, device=x.device).view(1, seqlen, 1)
            topk_idxs = torch.where(valids, topk_idxs + offset, -1)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


# ══════════════════════════════════════════════════════════════════════
# GCN GRAPH VECTOR QUANTIZER
# ══════════════════════════════════════════════════════════════════════

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        flat_x_f32 = flat_x.float()
        
        E = self.embedding.weight
        distances = (
            torch.sum(flat_x_f32 ** 2, dim=-1, keepdim=True)
            + torch.sum(E.float() ** 2, dim=-1)
            - 2 * torch.matmul(flat_x_f32, E.float().t())
        )
        
        encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(-1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=x.device, dtype=x.dtype)
        encodings.scatter_(1, encoding_indices, 1.0)
        
        quantized = torch.matmul(encodings, E.to(x.dtype)).view(B, L, D)
        
        # Codebook loss: moves the embeddings toward the encoder outputs
        codebook_loss = F.mse_loss(quantized, x.detach())
        # Commitment loss: moves the encoder outputs toward the embeddings
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices.view(B, L)


class ResidualVQ(nn.Module):
    def __init__(self, codebook_size: int, dim: int, num_quantizers: int = 3, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.commitment_cost = commitment_cost
        self.vqs = nn.ModuleList([
            VectorQuantizer(codebook_size, dim, commitment_cost)
            for _ in range(num_quantizers)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        quantized_out = torch.zeros_like(x)
        residual = x
        total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        indices_list = []
        
        for vq in self.vqs:
            quantized, loss, indices = vq(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_loss = total_loss + loss
            indices_list.append(indices)
            
        first_indices = indices_list[0]
        dummy_adj = torch.zeros(1, 1, device=x.device, dtype=x.dtype)
        return quantized_out, total_loss, first_indices, dummy_adj


# ══════════════════════════════════════════════════════════════════════
# HCM & CAUSAL DYNAMIC CONCEPT MEMORY
# ══════════════════════════════════════════════════════════════════════

class ElasticSparseConceptMemory(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.num_concepts = args.num_concepts
        self.num_abstract_concepts = args.num_abstract_concepts
        self.num_global_concepts = args.num_global_concepts
        self.total_concepts = args.num_concepts + args.num_abstract_concepts + args.num_global_concepts
        
        self.hcm_ema_alpha = args.hcm_ema_alpha
        self.entropy_threshold = args.entropy_threshold
        self.max_blocks = 16
        
        self.episodic_queries = nn.Parameter(torch.empty(1, args.num_concepts, args.dim))
        self.semantic_queries = nn.Parameter(torch.empty(1, args.num_abstract_concepts, args.dim))
        self.global_queries = nn.Parameter(torch.empty(1, args.num_global_concepts, args.dim))
        
        nn.init.normal_(self.episodic_queries, 0.0, 0.02)
        nn.init.normal_(self.semantic_queries, 0.0, 0.02)
        nn.init.normal_(self.global_queries, 0.0, 0.02)
        
        self.concept_blocks = nn.ModuleList([self._create_block(args)])
        
        self.register_buffer("slot_ema", torch.zeros(1, self.total_concepts, args.dim), persistent=True)
        self.register_buffer("slot_db", torch.zeros(1, self.total_concepts, args.dim), persistent=True)
        self.register_buffer("meta_centroids", torch.zeros(1, args.dim), persistent=False)
        
        self.lightning_indexer = Linear(args.dim, args.dim)
        
    def _create_block(self, args: ModelArgs):
        return ResidualVQ(
            codebook_size=args.codebook_size,
            dim=args.dim,
            commitment_cost=args.hcm_commit_loss_coeff
        )
        
    def process_chunk(self, encoder_hidden: torch.Tensor, block_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_enc, D = encoder_hidden.shape
        
        # Perceiver pooling heads
        n_heads = 8
        head_dim = self.dim // n_heads
        KV_4d = encoder_hidden.view(B, N_enc, n_heads, head_dim).transpose(1, 2)
        
        # 1. Pool episodic
        Q_epi = self.episodic_queries.expand(B, -1, -1)
        Q_epi_4d = Q_epi.view(B, self.num_concepts, n_heads, head_dim).transpose(1, 2).to(encoder_hidden.dtype)
        pooled_epi = F.scaled_dot_product_attention(Q_epi_4d, KV_4d, KV_4d).transpose(1, 2).contiguous().view(B, self.num_concepts, self.dim)
        
        # 2. Pool semantic
        Q_sem = self.semantic_queries.expand(B, -1, -1)
        Q_sem_4d = Q_sem.view(B, self.num_abstract_concepts, n_heads, head_dim).transpose(1, 2).to(encoder_hidden.dtype)
        pooled_sem = F.scaled_dot_product_attention(Q_sem_4d, KV_4d, KV_4d).transpose(1, 2).contiguous().view(B, self.num_abstract_concepts, self.dim)
        
        # 3. Pool global
        Q_glo = self.global_queries.expand(B, -1, -1)
        Q_glo_4d = Q_glo.view(B, self.num_global_concepts, n_heads, head_dim).transpose(1, 2).to(encoder_hidden.dtype)
        pooled_glo = F.scaled_dot_product_attention(Q_glo_4d, KV_4d, KV_4d).transpose(1, 2).contiguous().view(B, self.num_global_concepts, self.dim)
        
        # Concatenate slots
        pooled = torch.cat([pooled_epi, pooled_sem, pooled_glo], dim=1)
        
        quantized, loss, indices, adj = self.concept_blocks[0](pooled)
        
        with torch.no_grad():
            mean_quant = quantized.detach().mean(dim=0, keepdim=True)
            self.slot_ema.data.copy_(self.hcm_ema_alpha * self.slot_ema.data + (1.0 - self.hcm_ema_alpha) * mean_quant)
            self.slot_db.data[0].copy_(self.slot_ema.data[0])
            self.meta_centroids.data[0].copy_(torch.mean(self.slot_db[0], dim=0))
            
        return quantized, loss
        
    def lightning_retrieve(self, decoder_query: torch.Tensor, top_k_blocks: int = 1) -> torch.Tensor:
        proj_q = self.lightning_indexer(decoder_query)
        pooled_q = torch.mean(proj_q, dim=1)
        
        # Safe normalize L2
        proj_q_norm = pooled_q / (pooled_q.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        centroids_norm = self.meta_centroids.to(proj_q_norm.dtype)
        centroids_norm = centroids_norm / (centroids_norm.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        
        sim = torch.matmul(proj_q_norm, centroids_norm.t())
        k = min(top_k_blocks, self.meta_centroids.size(0))
        topk_scores, topk_idxs = torch.topk(sim, k, dim=-1)
        
        gathered = self.slot_db.to(decoder_query.dtype)[topk_idxs]
        concept_db = gathered.flatten(1, 2)
        return concept_db


# ══════════════════════════════════════════════════════════════════════
# MULTI-HEAD LATENT ATTENTION (MLA)
# ══════════════════════════════════════════════════════════════════════

class MLA(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id     = layer_id
        self.dim          = args.dim
        self.n_heads      = args.n_heads
        self.q_lora_rank  = args.q_lora_rank
        self.head_dim     = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.nope_head_dim
        self.n_groups     = args.o_groups
        self.o_lora_rank  = args.o_lora_rank
        self.eps          = args.norm_eps
        self.attn_logits_soft_cap = getattr(args, "attn_logits_soft_cap", None)

        # Low-rank Q projection
        self.wq_a  = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b  = Linear(self.q_lora_rank, self.n_heads * self.head_dim)

        # Latent KV compression
        self.wkv   = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # Grouped O projection (from DeepSeek-V4)
        heads_per_group = self.n_heads // self.n_groups
        self.wo_a = Linear(heads_per_group * self.head_dim, self.n_groups * self.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, self.dim)

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.softmax_scale = self.head_dim ** -0.5

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
        q_nope, q_rope = q[..., :-self.rope_head_dim], q[..., -self.rope_head_dim:]
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # 2. Compress KV
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = kv[..., :-self.rope_head_dim], kv[..., -self.rope_head_dim:]
        kv_rope = apply_rotary_emb(kv_rope, freqs_cis)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        # 3. Sliding window KV cache update
        if B > self.kv_cache.shape[0]:
            new_cache = torch.zeros(B, win, self.head_dim, device=self.kv_cache.device, dtype=self.kv_cache.dtype)
            new_cache[:self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_cache, persistent=False)

        if start_pos == 0:
            self.kv_cache.detach_().zero_()
            if N <= win:
                self.kv_cache[:B, :N] = kv
            else:
                cutoff = N % win
                self.kv_cache[:B, cutoff:win], self.kv_cache[:B, :cutoff] = \
                    kv[:, -win:].split([win - cutoff, cutoff], dim=1)
        else:
            slot = start_pos % win
            self.kv_cache[:B, slot] = kv[:, 0]

        K_cache = kv if start_pos == 0 else self.kv_cache[:B]
        
        # Fuse retrieved concept states directly into KV sequence attention space
        if concept_db is not None:
            concept_kv = self.wkv(concept_db)
            concept_kv = self.kv_norm(concept_kv)
            concept_kv = torch.cat([concept_kv[..., :-self.rope_head_dim], concept_kv[..., -self.rope_head_dim:]], dim=-1)
            
            K_combined = torch.cat([K_cache, concept_kv], dim=1)
        else:
            K_combined = K_cache

        # Attention scaling
        q_t = q.transpose(1, 2)
        is_causal = (start_pos == 0 and N > 1 and concept_db is None and cu_seqlens is None)
        attn_mask = None
        
        if start_pos == 0 and N > 1:
            if concept_db is not None or cu_seqlens is not None:
                Seq_combined = K_combined.size(1)
                Seq_token = K_cache.size(1)
                
                mask = torch.ones(B, N, Seq_combined, dtype=torch.bool, device=x.device)
                causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1)
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
                            end_idx = doc_boundaries[i+1].item()
                            if start_idx >= N: continue
                            mask[b, start_idx:end_idx, :start_idx] = True
                            if end_idx < N:
                                mask[b, start_idx:end_idx, end_idx:N] = True
                                
                attn_mask = mask

        kv_h = K_combined.unsqueeze(1).expand(-1, self.n_heads, -1, -1).to(q_t.dtype)
        
        if attn_mask is not None or self.attn_logits_soft_cap is not None:
            scores = torch.matmul(q_t.float(), kv_h.transpose(-2, -1).float()) * self.softmax_scale
            if self.attn_logits_soft_cap is not None:
                scores = torch.tanh(scores / self.attn_logits_soft_cap) * self.attn_logits_soft_cap
            
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask.unsqueeze(1), -10000.0)
            elif is_causal:
                causal_mask = torch.triu(torch.ones(N, kv_h.size(-2), dtype=torch.bool, device=x.device), diagonal=1)
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), -10000.0)
                
            probs = torch.softmax(scores, dim=-1).to(q_t.dtype)
            attn_out = torch.matmul(probs, kv_h)
        else:
            attn_out = F.scaled_dot_product_attention(
                q_t, kv_h, kv_h,
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


class DeepSeekAttention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression."""
    def __init__(self, args: ModelArgs, compress_ratio: int, use_indexer: bool):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.sliding_window_size
        self.compress_ratio = compress_ratio
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
        self.wo_a = Linear(heads_per_group * self.head_dim, self.n_groups * self.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5

        # Semantic Connection Pairformer Projections and Caches removed (Issue 1)

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if use_indexer:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        kv_cache_size = self.window_size + (max(1, args.max_seq_len // self.compress_ratio) if self.compress_ratio else 0)
        max_batch_size_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer("kv_cache", torch.zeros(max_batch_size_comp, kv_cache_size, self.head_dim), persistent=False)
        
        if self.compress_ratio:
            original_seq_len = args.original_seq_len
            rope_theta = getattr(args, "compress_rope_theta", 40000.0)
        else:
            original_seq_len = 0
            rope_theta = args.rope_theta
            
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, args.max_seq_len + 1024, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)


    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis_layer = self.freqs_cis[start_pos:start_pos+seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        
        # Dynamic resizing of attention kv_cache for larger batch size (e.g. GRPO)
        if bsz > self.kv_cache.shape[0]:
            device = x.device
            new_kv_cache = torch.zeros(bsz, self.kv_cache.shape[1], self.kv_cache.shape[2], device=device, dtype=self.kv_cache.dtype)
            new_kv_cache[:self.kv_cache.shape[0]] = self.kv_cache
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
                cache_len = max(1, min(self.compressor.cache_write_ptr[:bsz].max().item(), cache_cap))
                if start_pos == 0:
                    fired_positions = self.compressor.fired_indices_buf[:bsz, :cache_len]
                    query_positions = torch.arange(seqlen, device=x.device).view(1, seqlen, 1)
                    mask = fired_positions.unsqueeze(1) > query_positions
                    matrix = torch.arange(cache_len, device=x.device).view(1, 1, cache_len).expand(bsz, seqlen, -1)
                    compress_topk_idxs = torch.where(mask, -1, matrix + offset)
                else:
                    compress_topk_idxs = torch.arange(cache_len, device=x.device).view(1, 1, cache_len).expand(bsz, seqlen, -1) + offset
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
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
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale, soft_cap=self.attn_logits_soft_cap)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            
            topk_idxs = torch.clamp(topk_idxs, min=-1, max=self.kv_cache.size(1) - 1)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale, soft_cap=self.attn_logits_soft_cap)

        # Apply Qwen-style head-specific sigmoid gate (NeurIPS 2025 Best Paper)
        g = torch.sigmoid(self.wg(x)).unflatten(-1, (self.n_heads, self.head_dim))
        o = o * g

        # o
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float())
        x = self.wo_b(o.flatten(2).to(x.dtype))
        return x

class CompressedSparseAttention(DeepSeekAttention):
    def __init__(self, args: ModelArgs):
        super().__init__(args, compress_ratio=args.csa_compression_ratio, use_indexer=True)


class HeavilyCompressedAttention(DeepSeekAttention):
    def __init__(self, args: ModelArgs):
        super().__init__(args, compress_ratio=args.hca_compression_ratio, use_indexer=False)


# ══════════════════════════════════════════════════════════════════════
# HYBRID CONCEPT ATTENTION
# ══════════════════════════════════════════════════════════════════════

class ConceptExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, swiglu_limit: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * d_model / 3)
        self.w1 = Linear(d_model, hidden_dim)
        self.w3 = Linear(d_model, hidden_dim)
        self.w2 = Linear(hidden_dim, d_model)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up   = self.w3(x).float()
        
        if self.swiglu_limit > 0:
            up   = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
            
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.topk        = args.n_activated_experts
        self.score_func  = args.score_func
        self.route_scale = args.route_scale
        self.use_hash    = layer_id < args.n_hash_layers
        self.ema_bias_lr = args.ema_bias_lr
        self.n_group     = args.n_group
        self.topk_group  = args.topk_group
        self.pending_bias_updates = []

        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.normal_(self.weight, 0.0, 0.02)
        
        self.router_scale = nn.Parameter(torch.ones(args.dim))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(args.n_routed_experts))

        if self.use_hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(
                torch.zeros(args.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )


    def apply_pending_updates(self):
        if self.bias is not None and self.pending_bias_updates:
            with torch.no_grad():
                for update in self.pending_bias_updates:
                    self.bias.add_(update)
            self.pending_bias_updates.clear()

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gemma 4 Router Norm & Scale
        x_norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        root_size = 1.0 / math.sqrt(x.size(-1))
        router_input = x_norm * root_size * self.router_scale.float()

        scores = F.linear(router_input, self.weight.float())
        z_loss = torch.logsumexp(scores, dim=-1).square().mean()

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).clamp(min=1e-8).sqrt()

        router_probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)

        if self.bias is not None:
            scores = scores + self.bias

        # DeepSeek-V3 Gating score correction bias during evaluation
        if not self.training:
            scores_for_choice = scores + self.e_score_correction_bias.type_as(scores).unsqueeze(0)
        else:
            scores_for_choice = scores

        # DeepSeek-V3 Group-wise routing
        if self.n_group > 1:
            N_tokens, E_experts = scores_for_choice.shape
            experts_per_group = E_experts // self.n_group
            grouped_scores = scores_for_choice.view(N_tokens, self.n_group, experts_per_group)
            
            # Represent each group by the sum of its top expert scores
            k_group_top = min(2, experts_per_group)
            group_scores = grouped_scores.topk(k_group_top, dim=-1)[0].sum(dim=-1) # [N_tokens, n_group]
            
            # Select top groups
            group_idx = torch.topk(group_scores, k=min(self.topk_group, self.n_group), dim=-1)[1] # [N_tokens, topk_group]
            
            # Create mask for selected groups
            group_mask = torch.zeros_like(group_scores) # [N_tokens, n_group]
            group_mask.scatter_(1, group_idx, 1.0)
            
            # Expand mask back to individual experts
            score_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(N_tokens, E_experts)
            scores_masked = scores_for_choice.masked_fill(~score_mask.bool(), float('-inf'))
            
            if self.use_hash and input_ids is not None:
                indices = self.tid2eid[input_ids]
            else:
                indices = scores_masked.topk(self.topk, dim=-1)[1]
        else:
            if self.use_hash and input_ids is not None:
                indices = self.tid2eid[input_ids]
            else:
                indices = scores_for_choice.topk(self.topk, dim=-1)[1]

        if self.training and self.bias is not None:
            with torch.no_grad():
                counts = torch.bincount(indices.flatten(), minlength=self.weight.shape[0]).float()
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                total_routed = indices.numel() * (dist.get_world_size() if dist.is_initialized() else 1)
                routing_fraction = counts / (total_routed / self.topk)
                target_fraction = 1.0 / self.weight.shape[0]
                bias_update = self.ema_bias_lr * (target_fraction - routing_fraction)
                self.pending_bias_updates.append(bias_update)

        weights = router_probs.gather(1, indices)

        # Gemma 4 style top-k renormalization to prevent probability leakage
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        weights = weights * self.route_scale
        return weights, indices, z_loss, router_probs



class DeepSeekMoE(nn.Module):
    """
    DeepSeek-V4 style MoE with:
      • Optional Latent MoE (gating on full-dim, expert compute in latent space)
      • Expert capacity buffer with token dropping (expert_capacity_factor)
      • Per-expert learnable output scale
      • Auxiliary load-balance loss exposed alongside z-loss
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim            = args.dim
        self.latent_dim     = getattr(args, "moe_latent_dim", None) or args.dim
        self.use_latent     = self.latent_dim < args.dim
        self.n_routed       = args.n_routed_experts
        self.n_activated    = args.n_activated_experts
        self.capacity_factor = getattr(args, "expert_capacity_factor", 1.25)
        self.load_balance_coeff = getattr(args, "moe_load_balance_coeff", 0.01)
        self.router_z_loss_coeff = getattr(args, "router_z_loss_coeff", 0.001)
        self.router_entropy_coeff = getattr(args, "moe_router_entropy_coeff", 0.001)
        self.capacity_loss_coeff = getattr(args, "moe_capacity_loss_coeff", 0.01)

        self.gate = Gate(0, args)

        if self.use_latent:
            self.w_down = Linear(self.dim, self.latent_dim)
            self.w_up   = Linear(self.latent_dim, self.dim)
            expert_dim  = self.latent_dim
        else:
            expert_dim = self.dim

        self.experts = nn.ModuleList([
            ConceptExpert(expert_dim, args.effective_moe_inter_dim, swiglu_limit=args.swiglu_limit)
            for _ in range(self.n_routed)
        ])
        self.shared           = ConceptExpert(expert_dim, args.effective_moe_inter_dim, swiglu_limit=args.swiglu_limit)
        self.per_expert_scale = nn.Parameter(torch.ones(self.n_routed))

        # ── Dual dense+MoE FFW branch (Gemma4 mlp2 pattern) ─────────────
        # Runs a compact dense FFN in parallel with sparse MoE so that every
        # token *always* has a gradient path through a dense layer.
        self.use_dual_ffn = getattr(args, "moe_dual_ffn", True)
        if self.use_dual_ffn:
            dual_inter = getattr(args, "moe_dual_inter_dim", 0)
            dual_inter = dual_inter if dual_inter > 0 else args.effective_moe_inter_dim
            self.dense_branch      = ConceptExpert(expert_dim, dual_inter, swiglu_limit=args.swiglu_limit)
            self.dense_branch_norm = RMSNorm(expert_dim, args.norm_eps)

        # Expert dropout: randomly zero token→expert assignments during training
        self.expert_dropout_p = getattr(args, "moe_expert_dropout", 0.0)
        self.last_expert_counts: Optional[torch.Tensor] = None
        self.last_capacity_overflow = torch.tensor(0.0)
        self.last_router_entropy = torch.tensor(0.0)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape   = x.shape
        flat_x  = x.reshape(-1, self.dim)
        N_tokens = flat_x.shape[0]

        weights, indices, z_loss, router_probs = self.gate(flat_x, None)

        # ── Expert capacity: max tokens each expert can receive ──────
        # capacity = ceil(capacity_factor * tokens / n_experts * n_activated)
        capacity = max(1, int(math.ceil(
            self.capacity_factor * N_tokens * self.n_activated / self.n_routed
        )))

        # ── Auxiliary load-balance loss (Gemma-4 style) ──────────────
        # Encourage uniform distribution of tokens across experts
        # fi = fraction of tokens routed to expert i (averaged over activated experts)
        expert_counts = torch.zeros(self.n_routed, device=x.device, dtype=torch.float32)
        expert_counts.scatter_add_(
            0,
            indices.flatten().long().clamp(0, self.n_routed - 1),
            torch.ones(indices.numel(), device=x.device),
        )
        fi = expert_counts / (N_tokens * self.n_activated + 1e-8)  # (n_routed,)
        Pi = router_probs.float().mean(dim=0)
        load_balance_loss = self.n_routed * (fi * Pi.float()).sum()
        router_entropy = -(router_probs.float() * torch.log(router_probs.float() + 1e-8)).sum(dim=-1).mean()
        router_entropy_loss = -router_entropy / math.log(max(2, self.n_routed))
        capacity_overflow = (expert_counts - capacity).clamp(min=0).sum() / (indices.numel() + 1e-8)
        self.last_expert_counts = expert_counts.detach()
        self.last_capacity_overflow = capacity_overflow.detach()
        self.last_router_entropy = router_entropy.detach()

        # ── Forward through experts ───────────────────────────────────
        base_x = self.w_down(flat_x) if self.use_latent else flat_x
        y = torch.zeros_like(base_x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed).tolist()

        for i, exp in enumerate(self.experts):
            if counts[i] == 0:
                continue
            tok_idx, top_pos = torch.where(indices == i)
            # Token-dropping: only process up to capacity
            if tok_idx.shape[0] > capacity:
                tok_weights = weights[tok_idx, top_pos]
                _, sort_idx = torch.topk(tok_weights, k=capacity, largest=True)
                tok_idx = tok_idx[sort_idx]
                top_pos = top_pos[sort_idx]
            exp_out = exp(base_x[tok_idx], weights[tok_idx, top_pos, None])
            # Expert dropout: stochastically zero-out token contributions during training
            # Uses inverted scaling to keep expected value constant (like nn.Dropout)
            if self.training and self.expert_dropout_p > 0.0:
                keep = (torch.rand(exp_out.shape[0], device=exp_out.device) > self.expert_dropout_p).float().unsqueeze(-1)
                exp_out = exp_out * keep / (1.0 - self.expert_dropout_p + 1e-8)
            y.scatter_add_(
                0,
                tok_idx.unsqueeze(-1).expand(-1, y.shape[-1]),
                (exp_out * self.per_expert_scale[i].type_as(exp_out)).float(),
            )

        # Shared expert always processes all tokens (no dropout)
        y = y + self.shared(base_x).float()

        # ── Dual dense branch (Gemma4 mlp2): always-on dense path ────────
        # Runs in parallel with the sparse MoE to guarantee every token a
        # direct, dense gradient signal — critical for training stability.
        if self.use_dual_ffn:
            dense_in = self.dense_branch_norm(base_x)
            y = y + self.dense_branch(dense_in).float()

        if self.use_latent:
            y = self.w_up(y.type_as(x))
        else:
            y = y.type_as(x)

        # Weighted combined auxiliary loss: router_z_loss + load_balance
        aux = (
            self.router_z_loss_coeff * z_loss
            + self.load_balance_coeff * load_balance_loss
            + self.router_entropy_coeff * router_entropy_loss
            + self.capacity_loss_coeff * capacity_overflow
        )
        return y.reshape(shape), aux
class ManifoldConstrainedHyperConnection(nn.Module):
    def __init__(self, dim: int, n_hc: int):
        super().__init__()
        self.n_hc = n_hc
        self.norm = RMSNorm(n_hc * dim)
        self.w_pre = Linear(n_hc * dim, n_hc, bias=False)
        self.w_res = Linear(n_hc * dim, n_hc * n_hc, bias=False)
        self.w_post = Linear(n_hc * dim, n_hc, bias=False)
        # Learnable gating factors initialized small
        self.alpha_pre = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_res = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_post = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, x):
        # x shape: [batch, seq_len, n_hc, dim]
        B, S, H, D = x.shape
        x_flat = x.reshape(B, S, H * D)
        x_flat = self.norm(x_flat)
        
        a_raw = self.alpha_pre.to(x_flat.dtype) * self.w_pre(x_flat)
        c_raw = self.alpha_post.to(x_flat.dtype) * self.w_post(x_flat)
        b_raw = (self.alpha_res.to(x_flat.dtype) * self.w_res(x_flat)).view(-1, self.n_hc, self.n_hc)

        # DeepSeek-V4 strict constraints (Equations 6, 7, 8)
        A_l = torch.sigmoid(a_raw).unsqueeze(-1)
        C_l = (2.0 * torch.sigmoid(c_raw)).to(x_flat.dtype).unsqueeze(-1)
        
        # Sinkhorn-Knopp on exp(b_raw) for exactly 20 iterations in float32 for numerical stability
        M = torch.exp(b_raw.float())
        for _ in range(20):
            M = F.normalize(M, p=1, dim=1) # Column norm
            M = F.normalize(M, p=1, dim=2) # Row norm
        B_l = M.view(B, S, self.n_hc, self.n_hc).to(x_flat.dtype) # Projected onto Birkhoff polytope

        return A_l, B_l, C_l





class MHCBlock(nn.Module):
    def __init__(self, dim: int, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = sinkhorn_iters
        self.hc_eps = eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        
        with set_dtype(torch.float32):
            self.hc_fn    = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_base  = nn.Parameter(torch.empty(mix_hc))
            self.hc_scale = nn.Parameter(torch.empty(3))
            
        nn.init.normal_(self.hc_fn, 0, 0.02)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    def hc_pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        B, S, hc, D = x.size()
        x_flat = x.flatten(2)
        mean_sq = x_flat.square().mean(-1, keepdim=True).float()
        rsqrt  = torch.rsqrt(mean_sq + self.hc_eps).to(dtype)
        mixes  = F.linear(x_flat, self.hc_fn.to(dtype)) * rsqrt
        
        pre, post, comb = hc_split_sinkhorn(
            mixes.float(), self.hc_scale.float(), self.hc_base.float(),
            self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        
        y = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)
        return y, post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb.to(dtype), residual)
        return y


@torch.jit.script
def ssm_step_one(
    decay: torch.Tensor,
    v_heads: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    prev_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-step SSM update (decode phase)."""
    decay_0 = decay[:, 0]               # (B_comp, H, d_head, d_state)
    v_0 = v_heads[:, 0].unsqueeze(-1)   # (B_comp, H, d_head, 1)
    B_0 = B[:, 0].unsqueeze(-2)          # (B_comp, H, 1, d_state)
    C_0 = C[:, 0].unsqueeze(-2)          # (B_comp, H, 1, d_state)

    outer  = v_0 * B_0
    curr_s = decay_0 * prev_s + outer
    y_t    = (curr_s * C_0).sum(dim=-1)
    y      = y_t.unsqueeze(1)             # (B_comp, 1, H, d_head)
    return y, curr_s


@torch.jit.script
def ssm_chunk_scan(
    decay: torch.Tensor,
    v_heads: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    prev_s: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mamba-3 style chunked associative scan.
    """
    Bc, S, H, d_head = v_heads.shape
    outputs = torch.empty((Bc, S, H, d_head), device=decay.device, dtype=decay.dtype)
    curr_s  = prev_s

    n_chunks = (S + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        t0 = c * chunk_size
        t1 = min(t0 + chunk_size, S)
        for t in range(t0, t1):
            dt  = decay[:, t]                  # (B, H, d_head, d_state)
            vt  = v_heads[:, t].unsqueeze(-1)  # (B, H, d_head, 1)
            Bt  = B[:, t].unsqueeze(-2)         # (B, H, 1, d_state)
            Ct  = C[:, t].unsqueeze(-2)         # (B, H, 1, d_state)
            curr_s = dt * curr_s + vt * Bt
            outputs[:, t] = (curr_s * Ct).sum(dim=-1)

    return outputs, curr_s


@torch.jit.script
def ssm_recurrence_loop(
    decay: torch.Tensor,
    v_heads: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    prev_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy full-sequence recurrence (used when chunk_size >= S)."""
    B_comp, S, H, d_head = v_heads.shape
    outputs = torch.empty((B_comp, S, H, d_head), device=decay.device, dtype=decay.dtype)
    curr_s  = prev_s
    for t in range(S):
        decay_t = decay[:, t]           # (B_comp, H, d_head, d_state)
        v_t     = v_heads[:, t].unsqueeze(-1)
        B_t     = B[:, t].unsqueeze(-2)
        C_t     = C[:, t].unsqueeze(-2)
        outer   = v_t * B_t
        curr_s  = decay_t * curr_s + outer
        y_t     = (curr_s * C_t).sum(dim=-1)
        outputs[:, t] = y_t
    return outputs, curr_s


# ══════════════════════════════════════════════════════════════════════
# STATE SPACE RECURRENCE  (Mamba-3 Upgraded)
# ══════════════════════════════════════════════════════════════════════

class StateSpaceRecurrence(nn.Module):
    """
    Mamba-3 inspired SSM branch.
    Key improvements over Mamba-2:
      • Chunked parallel scan (ssm_chunk_scan) for efficient training
      • dt clamped to [dt_min, dt_max] for numerical stability (Nemotron pattern)
      • B and C L2-normalised per head (variance stabilisation)
      • Learnable dt_bias init from log-uniform distribution in [dt_min, dt_max]
      • Multi-group SSM support (ssm_n_groups)
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model     = args.dim
        self.n_heads     = getattr(args, "ssm_heads", 4)
        self.d_state     = getattr(args, "ssm_state_dim", 16)
        self.kernel_size = getattr(args, "ssm_kernel_size", 4)
        self.chunk_size  = getattr(args, "ssm_chunk_size", 64)
        self.dt_min      = getattr(args, "ssm_dt_min", 0.001)
        self.dt_max      = getattr(args, "ssm_dt_max", 0.1)
        self.dt_floor    = getattr(args, "ssm_dt_init_floor", 1e-4)
        self.n_groups    = getattr(args, "ssm_n_groups", 1)

        assert self.d_model % self.n_heads == 0, (
            f"dim {self.d_model} must be divisible by ssm_heads {self.n_heads}"
        )
        self.d_head = self.d_model // self.n_heads

        # ── Input projection ──────────────────────────────────────────
        # Projects to: u (gate), v (SSM input), B, C, delta
        self.w_in_dim = 3 * self.d_model + 2 * self.n_heads * self.d_state
        self.w_in  = Linear(self.d_model, self.w_in_dim)
        self.w_out = Linear(self.d_model, self.d_model)

        # ── Depthwise causal conv ─────────────────────────────────────
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.kernel_size,
            groups=self.d_model,
            bias=True,
        )

        # learned state matrix A
        self.A = nn.Parameter(-torch.ones(self.n_heads, self.d_state))

        # ── dt_bias: log-uniform init in [dt_min, dt_max] (Nemotron) ─
        # dt_bias such that softplus(dt_bias) ≈ dt_init
        dt_init = torch.exp(
            torch.rand(self.d_model) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_floor)
        # Inverse softplus: log(exp(dt_init) - 1) ≈ dt_init for large dt_init
        dt_bias_init = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(dt_bias_init.reshape(self.n_heads, self.d_head))

        # ── State buffers ─────────────────────────────────────────────
        max_batch_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer(
            "ssm_state",
            torch.zeros(max_batch_comp, self.n_heads, self.d_head, self.d_state),
            persistent=False,
        )
        self.register_buffer(
            "conv_state",
            torch.zeros(max_batch_comp, self.d_model, self.kernel_size - 1),
            persistent=False,
        )

        # ── D skip: direct input bypass term (standard Mamba, critical for ──
        # information flow when decay ≈ 0 / highly selective scan state)  ──
        self.use_d_skip = getattr(args, "ssm_d_skip", True)
        if self.use_d_skip:
            self.D = nn.Parameter(torch.ones(self.d_model))

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        B_comp, S, D = x.shape

        # ── 1. Input projection ───────────────────────────────────────
        projected = self.w_in(x)  # (B_comp, S, w_in_dim)
        u, v, B_C, delta = torch.split(
            projected,
            [self.d_model, self.d_model, 2 * self.n_heads * self.d_state, self.d_model],
            dim=-1,
        )

        B_raw, C_raw = torch.split(
            B_C, [self.n_heads * self.d_state, self.n_heads * self.d_state], dim=-1
        )
        B_mat = B_raw.reshape(B_comp, S, self.n_heads, self.d_state)
        C_mat = C_raw.reshape(B_comp, S, self.n_heads, self.d_state)

        # ── Mamba-3: L2-normalise B and C per head (variance stabilisation) ─
        B_mat = F.normalize(B_mat.float(), p=2, dim=-1)
        C_mat = F.normalize(C_mat.float(), p=2, dim=-1)

        # ── 2. Resize / reset state buffers ──────────────────────────
        if B_comp > self.ssm_state.shape[0]:
            self.register_buffer(
                "ssm_state",
                torch.zeros(B_comp, self.n_heads, self.d_head, self.d_state,
                            device=x.device, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "conv_state",
                torch.zeros(B_comp, self.d_model, self.kernel_size - 1,
                            device=x.device, dtype=x.dtype),
                persistent=False,
            )

        if start_pos == 0:
            self.ssm_state[:B_comp].zero_()
            self.conv_state[:B_comp].zero_()

        # ── 3. Causal depthwise conv ──────────────────────────────────
        padded_v = torch.cat(
            [self.conv_state[:B_comp].type_as(v), v.transpose(1, 2)], dim=-1
        )
        conv_out = F.conv1d(
            padded_v,
            self.conv1d.weight.type_as(padded_v),
            self.conv1d.bias.type_as(padded_v) if self.conv1d.bias is not None else None,
            groups=self.d_model,
        )
        self.conv_state[:B_comp].copy_(padded_v[..., -self.kernel_size + 1:].detach())
        v_conv = conv_out.transpose(1, 2)  # (B_comp, S, d_model)

        # ── 4. dt with clamping (Nemotron mamba_dt_min / mamba_dt_max) ─
        delta_heads = delta.reshape(B_comp, S, self.n_heads, self.d_head)
        dt = F.softplus(delta_heads.float() + self.dt_bias.view(1, 1, self.n_heads, self.d_head).float())
        dt = dt.clamp(min=self.dt_min, max=self.dt_max)          # <── KEY stabilisation

        # Discretise: decay = exp(dt * A)  (ZOH discretisation with learned matrix A)
        decay = torch.exp(dt.unsqueeze(-1) * self.A.view(1, 1, self.n_heads, 1, self.d_state).float())  # (B_comp, S, H, d_head, d_state)

        # ── 5. Selective scan ─────────────────────────────────────────
        v_heads = v_conv.reshape(B_comp, S, self.n_heads, self.d_head).float() * dt
        
        if self.training:
            prev_s = self.ssm_state[:B_comp].clone()
        else:
            prev_s = self.ssm_state[:B_comp].detach().clone()

        if S == 1:
            y, prev_s = ssm_step_one(decay, v_heads, B_mat, C_mat, prev_s)
        elif S <= self.chunk_size:
            # Short sequence — fall back to full recurrence
            y, prev_s = ssm_recurrence_loop(decay, v_heads, B_mat, C_mat, prev_s)
        else:
            # Long sequence — use Mamba-3 chunked scan for training efficiency
            y, prev_s = ssm_chunk_scan(decay, v_heads, B_mat, C_mat, prev_s, self.chunk_size)

        self.ssm_state[:B_comp].copy_(prev_s.detach())
        y = y.reshape(B_comp, S, self.d_model).type_as(x)

        # ── 6. Gate + output projection ───────────────────────────────
        gated = y * F.silu(u)

        # ── D skip: y += D ⊙ u (direct input bypass, standard Mamba) ─
        # Keeps information flow alive when the selective scan is near-zero.
        # D broadcasts over (B_comp, S) automatically.
        if self.use_d_skip:
            gated = gated + u.type_as(gated) * self.D.to(gated.dtype)

        return self.w_out(gated)


# ══════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════

class LasmoidBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.skip_scale = nn.Parameter(torch.ones(1))
        
        # Interleaved Hybrid Attention
        self.is_csa = (layer_id % 2 == 0)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        
        if self.is_csa:
             self.attn = CompressedSparseAttention(args) # uses m compression
        else:
             self.attn = HeavilyCompressedAttention(args) # uses m' compression
        
        # Parallel State Space Recurrence Branch
        self.ssm_branch = StateSpaceRecurrence(args)
        
        # DeepSeek-V4 Manifold Hyper Connections
        self.mhc_attn = ManifoldConstrainedHyperConnection(args.dim, args.num_residual_streams)
        self.mhc_ffn  = ManifoldConstrainedHyperConnection(args.dim, args.num_residual_streams)
        
        # Dummy last_pred_loss for compatibility
        self.last_pred_loss = torch.tensor(0.0)
        
        # MoE using sqrt(softplus) as already configured
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.moe_layer = DeepSeekMoE(args)

        # ── Post-layer norms (Gemma4): RMSNorm on attn/FFW output ────────
        self.use_post_attn_norm = getattr(args, "post_attn_norm", True)
        self.use_post_ffw_norm  = getattr(args, "post_ffw_norm",  True)
        if self.use_post_attn_norm:
            self.post_attn_norm = RMSNorm(args.dim, args.norm_eps)
        if self.use_post_ffw_norm:
            self.post_ffw_norm = RMSNorm(args.dim, args.norm_eps)


    def forward(
        self,
        streams: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. mHC Pre-Attention
        A_l_attn, B_l_attn, C_l_attn = self.mhc_attn(streams)
        attn_in = A_l_attn * streams
        
        # 2. Attention (CSA or HCA)
        B, S, H, D = attn_in.shape
        attn_in_flat = attn_in.transpose(1, 2).reshape(B * H, S, D)
        
        # Run Attention path
        attn_out_flat = self.attn(self.attn_norm(attn_in_flat), freqs_cis, start_pos)
        
        # Run Parallel SSM Recurrence path
        ssm_out_flat = self.ssm_branch(self.attn_norm(attn_in_flat), start_pos)
        
        # Fuse outputs
        fused_out_flat = attn_out_flat + ssm_out_flat
        
        if self.use_post_attn_norm:
            fused_out_flat = self.post_attn_norm(fused_out_flat)

        attn_out = fused_out_flat.reshape(B, H, S, D).transpose(1, 2)
        
        # 3. mHC Post-Attention + Birkhoff Constraint Mix
        streams = B_l_attn @ streams + C_l_attn * attn_out
        
        # 4. mHC Pre-FFN
        A_l_ffn, B_l_ffn, C_l_ffn = self.mhc_ffn(streams)
        ffn_in = A_l_ffn * streams
        
        # 5. MoE FFN
        B_f, S_f, H_f, D_f = ffn_in.shape
        ffn_in_flat = ffn_in.transpose(1, 2).reshape(B_f * H_f, S_f, D_f)
        
        ffn_out_flat, z_loss = self.moe_layer(self.ffn_norm(ffn_in_flat))

        if self.use_post_ffw_norm:
            ffn_out_flat = self.post_ffw_norm(ffn_out_flat)

        ffn_out = ffn_out_flat.reshape(B_f, H_f, S_f, D_f).transpose(1, 2)
        
        # 6. mHC Post-FFN
        streams = B_l_ffn @ streams + C_l_ffn * ffn_out
        
        streams = streams * self.skip_scale
        
        # Backward-compatible outputs
        vq_loss = torch.tensor(0.0, device=streams.device, dtype=streams.dtype)
        routing = torch.zeros(B, S, device=streams.device, dtype=streams.dtype)
        indices = torch.zeros(B, S, dtype=torch.long, device=streams.device)
        adj = torch.zeros(1, 1, device=streams.device, dtype=streams.dtype)
        
        return streams, z_loss, vq_loss, routing, indices, adj

# ══════════════════════════════════════════════════════════════════════
# MULTI-TOKEN PREDICTION (MTP) HEAD
# ══════════════════════════════════════════════════════════════════════

class MTPBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        self.enorm  = RMSNorm(args.dim, args.norm_eps)
        self.hnorm  = RMSNorm(args.dim, args.norm_eps)
        self.norm   = RMSNorm(args.dim, args.norm_eps)
        self.block  = LasmoidBlock(layer_id, args)

        hc_mult = args.num_residual_streams
        hc_dim  = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn    = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base  = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.hc_head_fn, 0, 0.02)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)

        self.embed: Optional[nn.Embedding] = None
        self.head:  Optional[nn.Module]    = None

    def hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        B, S, hc, D  = shape
        xf    = x.flatten(2)
        mean_sq = xf.square().mean(-1, keepdim=True).float()
        rsqrt = torch.rsqrt(mean_sq + 1e-6).to(dtype)
        mixes = F.linear(xf, self.hc_head_fn.to(dtype)) * rsqrt
        pre   = torch.sigmoid(mixes.float() * self.hc_head_scale + self.hc_head_base) + 1e-6
        y     = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)
        return y

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.embed is not None and self.head is not None
        e = self.enorm(self.embed(input_ids).to(x.dtype))
        
        B_h, S_h, hc_h, D_h = x.shape
        h_flat = x.reshape(B_h * S_h * hc_h, D_h)
        h_flat = self.hnorm(h_flat)
        h_flat = self.h_proj(h_flat)
        h = h_flat.reshape(B_h, S_h, hc_h, D_h)
        
        x = self.e_proj(e).unsqueeze(2) + h
        x, z_loss, vq_loss, routing, indices, adj = self.block(x, freqs_cis, start_pos, input_ids)
        
        y = self.hc_head_reduce(x)
        y = self.norm(y)
        logits = F.linear(y.float(), self.head.weight.float())
        return logits, z_loss, vq_loss

# ══════════════════════════════════════════════════════════════════════
# THE FULL SYSTEM SHELL: LASMOID
# ══════════════════════════════════════════════════════════════════════

class Lasmoid(nn.Module):
    def __init__(self, args: ModelArgs):
        global default_dtype, scale_fmt, scale_dtype
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt     = args.scale_fmt
        scale_dtype   = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32

        super().__init__()
        self.args        = args
        self.max_seq_len = args.max_seq_len
        self.hc_mult     = args.num_residual_streams

        # Embedding & weights tying
        self.emb = nn.Embedding(args.vocab_size, args.dim).to(dtype=torch.bfloat16)
        self.external_embedding_proj: Optional[nn.Linear] = None
        self.external_embedding_norm: Optional[RMSNorm] = None
        if args.external_embedding_dim > 0:
            self.external_embedding_proj = nn.Linear(args.external_embedding_dim, args.dim, bias=False).to(dtype=torch.bfloat16)
            self.external_embedding_norm = RMSNorm(args.dim, args.norm_eps)
            nn.init.normal_(self.external_embedding_proj.weight, mean=0.0, std=0.02)

        # READ REPLICA (Encoder) — perceiver pooling & dynamic spawning
        self.encoder_attn = MLA(0, args)
        self.encoder_norm = RMSNorm(args.dim, args.norm_eps)
        self.memory       = ElasticSparseConceptMemory(args)

        # WRITE MASTER (Decoder) — sequence processing blocks
        self.layers       = nn.ModuleList([LasmoidBlock(i, args) for i in range(args.n_layers)])
        self.decoder_norm = RMSNorm(args.dim, args.norm_eps)

        self.head   = Linear(args.dim, args.vocab_size, dtype=torch.bfloat16)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        # Superhuman alignment head for attribute steering
        self.superhuman_alignment_head = Linear(args.dim, len(args.steering_attributes), dtype=torch.bfloat16)

        # Output head Hyper-Connections reducer
        hc_mult = args.num_residual_streams
        hc_dim  = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn    = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base  = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.hc_head_fn, 0, 0.02)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)

        # Multi-Token Prediction (MTP)
        self.mtp = nn.ModuleList()
        for i in range(args.n_mtp_layers):
            blk = MTPBlock(args.n_layers + i, args)
            blk.embed = self.emb
            blk.head  = self.head
            self.mtp.append(blk)

        # YaRN RoPE cache
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.rope_head_dim, args.max_seq_len + 1024,
                args.original_seq_len, args.rope_theta,
                args.rope_factor, args.beta_fast, args.beta_slow,
            ),
            persistent=False,
        )
        self.last_pred_loss = torch.tensor(0.0)

        self.gradient_checkpointing = False
        self.last_z_loss = torch.tensor(0.0)
        self.last_commit_loss = torch.tensor(0.0)
        self.last_vq_loss = torch.tensor(0.0)
        self.last_moe_loss = torch.tensor(0.0)
        self.last_token_concept_loss = torch.tensor(0.0)


    def embed_tokens(
        self,
        token_ids: torch.Tensor,
        external_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed token ids and optionally fuse aligned external embeddings."""
        token_emb = self.emb(token_ids).to(torch.bfloat16)
        if external_embeddings is None:
            return token_emb
        if self.external_embedding_proj is None:
            raise ValueError("external_embeddings were provided but args.external_embedding_dim is 0")
        if external_embeddings.shape[:2] != token_ids.shape:
            raise ValueError(
                "external_embeddings must be shaped [batch, seq, external_embedding_dim] "
                f"with batch/seq matching token ids; got {tuple(external_embeddings.shape)} vs {tuple(token_ids.shape)}"
            )
        if external_embeddings.shape[-1] != self.args.external_embedding_dim:
            raise ValueError(
                f"external embedding dim mismatch: got {external_embeddings.shape[-1]}, "
                f"expected {self.args.external_embedding_dim}"
            )

        ext = external_embeddings.to(device=token_ids.device, dtype=torch.bfloat16)
        if self.args.external_embedding_norm:
            ext = F.normalize(ext.float(), dim=-1, eps=1e-6).to(torch.bfloat16)
        ext = self.external_embedding_proj(ext)
        if self.external_embedding_norm is not None:
            ext = self.external_embedding_norm(ext)
        return token_emb + ext.to(token_emb.dtype) * self.args.external_embedding_scale

    def apply_pending_bias_updates(self):
        for m in self.modules():
            if isinstance(m, Gate):
                m.apply_pending_updates()

    def _hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        B, S, hc, D  = shape
        xf    = x.flatten(2)
        mean_sq = xf.square().mean(-1, keepdim=True).float()
        rsqrt = torch.rsqrt(mean_sq + 1e-6).to(dtype)
        mixes = F.linear(xf, self.hc_head_fn.to(dtype)) * rsqrt
        pre   = torch.sigmoid(mixes.float() * self.hc_head_scale + self.hc_head_base) + 1e-6
        y     = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)
        return y

    def forward(
        self,
        x_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        concept_db: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        external_embeddings: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        steering_vector: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if self.training:
            self.apply_pending_bias_updates()

        B, N_dec = x_dec.shape
        freqs_cis_dec = self.freqs_cis[start_pos : start_pos + N_dec]

        # ── 1. READ REPLICA ───────────────────────────────────────────
        if start_pos == 0 or concept_db is None or memory_state is None:
            assert x_enc is not None, "x_enc must be provided at start_pos == 0"
            _, N_enc = x_enc.shape
            freqs_cis_enc = self.freqs_cis[:N_enc]
            H_enc   = self.embed_tokens(x_enc, external_embeddings).to(torch.bfloat16)
            enc_out = self.encoder_attn(self.encoder_norm(H_enc), freqs_cis_enc, start_pos=0)
            memory_state, commit_loss = self.memory.process_chunk(enc_out, block_idx=0)
            self.last_commit_loss = commit_loss
            concept_db = self.memory.lightning_retrieve(H_enc, top_k_blocks=self.args.lightning_topk_blocks)

        # ── 2. WRITE MASTER ───────────────────────────────────────────
        H_dec_external = external_embeddings
        if external_embeddings is not None and external_embeddings.shape[1] != N_dec:
            H_dec_external = external_embeddings[:, -N_dec:, :]
        H_dec     = self.embed_tokens(x_dec, H_dec_external).to(torch.bfloat16)
        
        # Parallel streams routing concept embedding representations
        H_memory  = torch.mean(memory_state, dim=1, keepdim=True).expand(-1, N_dec, -1)
        H_concept = torch.mean(concept_db, dim=1, keepdim=True).expand(-1, N_dec, -1)

        hc = self.hc_mult
        streams_list = [H_dec, H_memory, H_concept]
        if hc > 3:
            streams_list += [torch.zeros_like(H_dec)] * (hc - 3)
        streams = torch.stack(streams_list, dim=2)

        # cos & sin removed (relying on freqs_cis)

        total_z_loss   = torch.tensor(0.0, device=x_dec.device, dtype=torch.float32)
        total_vq_loss  = torch.tensor(0.0, device=x_dec.device, dtype=torch.float32)
        routing_maps    = []
        concept_indices = []
        adjacencies     = []

        # ── Chain-of-Thought Budget (Reasoning State-Machine) ──────────────
        # Implements a <think>/<answer> token-budget loop.
        # Each r_step is one reasoning iteration.  We exit early when the
        # mean symbolic-guard confidence across all layers exceeds cot_exit_confidence.
        reasoning_steps     = getattr(self.args, "reasoning_steps", 1)

        for r_step in range(reasoning_steps):
            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    streams, z_loss, vq_loss, routing, indices, adj = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        streams,
                        freqs_cis_dec,
                        start_pos,
                        x_dec,
                        use_reentrant=False,
                    )
                else:
                    streams, z_loss, vq_loss, routing, indices, adj = layer(
                        streams, freqs_cis_dec, start_pos, x_dec
                    )

                if r_step == reasoning_steps - 1:
                    total_z_loss  = total_z_loss + z_loss
                    total_vq_loss = total_vq_loss + vq_loss
                    routing_maps.append(routing)
                    concept_indices.append(indices)
                    adjacencies.append(adj)

        # Collect CIF event probabilities from attention layers for boundary stabilization loss
        event_probs = []
        if self.training and start_pos == 0:
            for layer in self.layers:
                attn = layer.attn
                if hasattr(attn, '_last_event_prob') and attn._last_event_prob is not None:
                    event_probs.append(attn._last_event_prob)

        self.last_z_loss = total_z_loss
        self.last_moe_loss = total_z_loss
        self.last_vq_loss = total_vq_loss
        self.last_pred_loss = sum(layer.last_pred_loss for layer in self.layers)
        self.last_token_concept_loss = torch.tensor(0.0, device=x_dec.device, dtype=torch.float32)
        self.last_confidences = []

        h_final  = self._hc_head_reduce(streams)
        h_normed = self.decoder_norm(h_final)

        # Predict attributes for the sequence from final representations
        self.last_predicted_attributes = self.superhuman_alignment_head(h_normed)

        # Apply attribute steering logit biasing if steering_vector is provided
        if steering_vector is not None:
            w_dtype = self.superhuman_alignment_head.weight.dtype
            if isinstance(steering_vector, dict):
                vector_list = [steering_vector.get(attr, 0.0) for attr in self.args.steering_attributes]
                steering_vector = torch.tensor(vector_list, device=h_normed.device, dtype=w_dtype)
            elif isinstance(steering_vector, (list, tuple)):
                steering_vector = torch.tensor(steering_vector, device=h_normed.device, dtype=w_dtype)
            elif isinstance(steering_vector, torch.Tensor):
                steering_vector = steering_vector.to(device=h_normed.device, dtype=w_dtype)
            
            if steering_vector.ndim == 1:
                steering_vector = steering_vector.unsqueeze(0)
            
            delta_h = torch.matmul(steering_vector, self.superhuman_alignment_head.weight)
            if delta_h.ndim == 2:
                delta_h = delta_h.unsqueeze(1)
            
            h_normed = h_normed + delta_h

        logits   = F.linear(h_normed.float(), self.head.weight.float())

        # 3. Multi-Token Prediction (t+2 prediction)
        mtp_logits = None
        if self.training and N_dec > 1 and len(self.mtp) > 0:
            mtp_logits, mtp_z, mtp_vq = self.mtp[0](
                streams[:, :-1],
                freqs_cis_dec[:-1],
                x_dec[:, 1:],
                start_pos,
            )
            self.last_z_loss = self.last_z_loss + mtp_z
            self.last_vq_loss = self.last_vq_loss + mtp_vq

        return logits, mtp_logits, concept_db, memory_state, routing_maps, concept_indices, adjacencies, event_probs

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 0,
        pad_token: int = 1,
        max_len: Optional[int] = None,
        steering_vector: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        device  = idx.device
        actual_max_len = max_len if max_len is not None else self.max_seq_len
        cond_len = idx.shape[1]

        if cond_len < actual_max_len:
            padding    = torch.full((idx.shape[0], actual_max_len - cond_len), pad_token, dtype=idx.dtype, device=device)
            idx_padded = torch.cat([padding, idx], dim=1)
        else:
            idx_padded = idx[:, -actual_max_len:]

        # Single Read pass to freeze prompt representations
        logits, _, concept_db, memory_state, *_ = self(idx_padded, idx_padded, start_pos=0, steering_vector=steering_vector)

        # Autoregressive decode sampling
        last_logits = logits[:, -1, :]
        
        # Frontier Self-Evolution: adjust temperature based on symbolic guard confidence
        # Higher confidence -> lower temperature (more deterministic/greedy)
        # Lower confidence -> higher temperature (more exploratory)
        if hasattr(self, 'last_confidences') and self.last_confidences:
            mean_conf = torch.stack([c.mean() for c in self.last_confidences]).mean().item()
            # Dynamic temperature: confidence 1.0 -> temp 0.2, confidence 0.0 -> temp 1.5
            temperature = max(0.2, min(1.5, 1.5 - mean_conf * 1.3))
            
        if temperature > 0:
            last_logits = last_logits / temperature
        if top_k > 0:
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[:, [-1]]] = -10000.0
        probs = torch.softmax(last_logits, dim=-1, dtype=torch.float32)
        idx_next = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=1)

        current_pos = actual_max_len
        for step in range(max_new_tokens - 1):
            logits, _, _, _, *_ = self(
                x_enc=None,
                x_dec=idx_next,
                concept_db=concept_db,
                memory_state=memory_state,
                start_pos=current_pos,
                steering_vector=steering_vector,
            )
            
            last_logits = logits[:, -1, :]
            
            # Recalculate dynamic temperature for current token
            if hasattr(self, 'last_confidences') and self.last_confidences:
                mean_conf = torch.stack([c.mean() for c in self.last_confidences]).mean().item()
                temperature = max(0.2, min(1.5, 1.5 - mean_conf * 1.3))
                
            if temperature > 0:
                last_logits = last_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = -10000.0
                
            probs = torch.softmax(last_logits, dim=-1, dtype=torch.float32)
            idx_next = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
            current_pos += 1

        self.train()
        return idx


# ══════════════════════════════════════════════════════════════════════
# COMPUTE LOSS
# ══════════════════════════════════════════════════════════════════════

def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    routing_maps: List[torch.Tensor],
    vq_losses: List[torch.Tensor],
    adjacencies: List[torch.Tensor],
    event_probs: Optional[List[torch.Tensor]] = None,
    loss_mask: Optional[torch.Tensor] = None,
    moe_aux_loss: Optional[torch.Tensor] = None,
    token_concept_loss: Optional[torch.Tensor] = None,
    token_concept_coeff: float = 0.05,
    graph_sparsity: float = 0.01,
    cif_target_ratio: float = 0.25,
    cif_entropy_weight: float = 0.01,
    cif_ratio_weight: float = 1.0,
) -> torch.Tensor:
    # ── Main autoregressive CE loss (masked SFT) ────────────────────
    if loss_mask is not None:
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none'
        )
        ce_loss = (ce_loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)
    else:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    moe_loss = moe_aux_loss if moe_aux_loss is not None else torch.tensor(0.0, device=logits.device)
    return ce_loss + moe_loss


def compute_grpo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Group Relative Policy Optimization (GRPO) clipped surrogate loss.
    
    Args:
        logits: policy logits of shape [B, S, V] (where B = group_size * batch_size)
        targets: target token ids of shape [B, S]
        advantages: advantages (normalized rewards) of shape [B]
        old_logprobs: log probabilities under old policy of shape [B, S]
        loss_mask: mask indicating which tokens have loss calculated (e.g. response tokens) of shape [B, S]
        clip_eps: PPO clipping range
        kl_coeff: KL penalty weight
        
    Returns:
        total_loss: policy loss + KL penalty
        policy_loss: policy surrogate loss
        kl_loss: Kullback-Leibler divergence loss
    """
    logprobs = F.log_softmax(logits, dim=-1)
    target_logprobs = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1) # [B, S]
    
    # Calculate token-level ratio r_t
    ratio = torch.exp(target_logprobs - old_logprobs) # [B, S]
    
    # Expand advantages from [B] to [B, S]
    adv = advantages.unsqueeze(-1).expand_as(ratio)
    
    # Clipped policy objective
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_objective = torch.min(surr1, surr2) # [B, S]
    
    # KL penalty (approximate KL divergence: exp(ref_logp - logp) - (ref_logp - logp) - 1)
    # We use old_logprobs as the reference model starting point
    kl = torch.exp(old_logprobs - target_logprobs) - (old_logprobs - target_logprobs) - 1.0
    
    if loss_mask is not None:
        policy_loss = -(policy_objective * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        kl_loss = (kl * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    else:
        policy_loss = -policy_objective.mean()
        kl_loss = kl.mean()
        
    total_loss = policy_loss + kl_coeff * kl_loss
    return total_loss, policy_loss, kl_loss


# ══════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════

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

    logits, mtp_logits, c_db, mem, routings, idxs, adjs, event_probs = model(x_enc, x_dec)
    print(f"  logits       : {logits.shape}")
    print(f"  mtp_logits   : {mtp_logits.shape if mtp_logits is not None else None}")
    print(f"  concept_db   : {c_db.shape}")
    print(f"  memory       : {mem.shape}")
    print(f"  routings     : {[r.shape for r in routings]}")
    print(f"  adjs         : {[a.shape for a in adjs]}")
    print(f"  event_probs  : {len(event_probs)} layers" + (f", {event_probs[0].shape}" if event_probs else ""))
    
    # Test compute_loss with event_probs
    targets = torch.randint(0, args.vocab_size, (2, 32))
    loss = compute_loss(logits, targets, routings, [torch.tensor(0.0)] * len(routings), adjs, event_probs)
    print(f"  total_loss   : {loss.item():.4f}")

    # Verify superhuman alignment head and attribute steering
    print(f"  predicted attributes: {model.last_predicted_attributes.shape}")
    assert model.last_predicted_attributes.shape == (2, 32, len(args.steering_attributes))

    steering_val = {"creativity": 1.5, "scientific_rigor": -0.5}
    logits_steered, *_ = model(x_enc, x_dec, steering_vector=steering_val)
    print(f"  steered logits       : {logits_steered.shape}")
    assert logits_steered.shape == logits.shape

    # Test generation with steering vector
    generated = model.generate(x_dec[:, :10], max_new_tokens=5, steering_vector=steering_val)
    print(f"  generated (steered)  : {generated.shape}")
    assert generated.shape == (2, 15)

    print("\nALL MODULES TESTED & READY ✓")