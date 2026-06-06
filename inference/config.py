"""
Lasmoid — Per-Component Configs (Phase 0 SOLiD Refactoring)
=================================================================
Splits monolithic ModelArgs (90+ fields) into focused per-component configs.
ModelArgs remains backward-compatible by inheriting from a BaseConfig mixin.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, List, Any


# ══════════════════════════════════════════════════════════════════════
# PER-COMPONENT CONFIGS  (ISP: each component sees only its own params)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class AttentionConfig:
    """Multi-head Latent Attention (MLA) + sliding window + optional KV compression."""

    n_heads: int = 16
    head_dim: int = 64
    rope_head_dim: int = 16
    q_lora_rank: int = 256
    o_lora_rank: int = 256
    o_groups: int = 2
    eps: float = 1e-6
    attn_logits_soft_cap: Optional[float] = 30.0
    window_size: int = 512

    # Indexer (DeepSeek-V4 learned sparse)
    use_indexer: bool = False
    index_n_heads: int = 4
    indexer_head_dim: int = 32
    index_topk: int = 16
    compress_rope_theta: float = 40000.0

    # Gemma-4 hybrid attention
    attention_type: Literal["local", "global", "hybrid"] = "local"
    global_heads: int = 4
    global_key_size: int = 512
    k_eq_v_global: bool = False
    qk_norm_with_scale: bool = True
    local_base_frequency: int = 10_000
    global_base_frequency: int = 1_000_000


@dataclass
class MoEConfig:
    """Mixture of Experts config — supports Latent MoE, dual-branch, expert dropout."""

    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_group: int = 1
    topk_group: int = 1
    moe_inter_dim: int = 0  # 0 = auto
    moe_latent_dim: Optional[int] = None
    moe_dual_inter_dim: int = 0
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    expert_capacity_factor: float = 1.25
    load_balance_coeff: float = 0.01
    router_z_loss_coeff: float = 0.001
    router_entropy_coeff: float = 0.001
    capacity_loss_coeff: float = 0.01
    moe_expert_dropout: float = 0.0
    n_hash_layers: int = 0
    # Dual dense+MoE FFW (Gemma-4 mlp2)
    moe_dual_ffn: bool = True
    expert_dtype: Literal["bf16", "fp8", "nvfp4"] = "bf16"


@dataclass
class CompressorConfig:
    """CIF (Continuous Integrate-and-Fire) semantic event compressor."""

    ratio_normal: int = 4
    ratio_hyper: int = 128
    ratio_emergency: int = 512
    hyper_threshold: int = 300_000
    emergency_threshold: int = 1_500_000
    overlap: bool = True
    rotate: bool = False
    per_layer_ratios: Tuple[int, ...] = ()
    # Indexer compression
    index_compression_ratio: int = 4
    # Original naming compat
    csa_compression_ratio: int = 4
    hca_compression_ratio: int = 128


@dataclass
class SSMConfig:
    """Mamba-3 style State Space Model."""

    heads: int = 8
    state_dim: int = 16
    kernel_size: int = 4
    chunk_size: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 0.0001
    n_groups: int = 1
    d_skip: bool = True


@dataclass
class MHCConfig:
    """Manifold Hyper-Connections."""

    num_residual_streams: int = 4
    sinkhorn_iters: int = 8
    eps: float = 1e-6


@dataclass
class ConceptMemoryConfig:
    """Elastic Sparse Concept Memory."""

    num_concepts: int = 128
    num_abstract_concepts: int = 16
    num_global_concepts: int = 4
    codebook_size: int = 512
    codebook_dim: int = 32
    hcm_ema_alpha: float = 0.99
    hcm_commit_loss_coeff: float = 0.25
    entropy_threshold: float = 0.5
    lightning_topk_blocks: int = 4
    concept_topk: int = 8
    concept_ratio: float = 0.4
    predictive_coding_coeff: float = 0.01


@dataclass
class MTPConfig:
    """Multi-Token Prediction."""

    n_mtp_layers: int = 1
    n_mtp_heads: int = 1  # for speculative decoding
    draft_length: int = 6


@dataclass
class RoPEConfig:
    """Rotary Position Embedding."""

    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 0


@dataclass
class StabilityConfig:
    """Stability system: drift detection, adaptive temp, KV cache integrity checks."""

    enabled: bool = False
    drift_window_size: int = 100
    drift_check_interval: int = 1
    entropy_collapse_threshold: float = 3.0
    logit_norm_spike_threshold: float = 3.0
    base_temperature: float = 0.7
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    kmin_cache_check_interval: int = 64
    long_context_threshold_1: int = 500_000
    long_context_threshold_2: int = 1_000_000
    long_gen_decay_1: int = 5000
    long_gen_decay_2: int = 10000


@dataclass
class RegularizationConfig:
    """Loss coefficients and regularization."""

    ema_bias_lr: float = 0.01
    router_z_loss_coeff: float = 0.001
    moe_load_balance_coeff: float = 0.01
    moe_router_entropy_coeff: float = 0.001
    moe_capacity_loss_coeff: float = 0.01
    token_concept_loss_coeff: float = 0.05
    post_attn_norm: bool = True
    post_ffw_norm: bool = True


@dataclass
class QuantConfig:
    """Precision mapping configuration for post-training quantization."""

    moe_route_dtype: Literal["bf16", "fp8", "nvfp4"] = "bf16"
    moe_shared_dtype: Literal["bf16", "fp8"] = "bf16"
    attn_proj_dtype: Literal["bf16", "fp8"] = "bf16"
    kv_cache_dtype: Literal["bf16", "fp8"] = "bf16"
    embed_dtype: Literal["bf16"] = "bf16"
    calib_size: int = 2000


@dataclass

class ModelArgs:
    """Monolithic config for backward compatibility — delegates to sub-configs."""

    # ── Core ──
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

    # ── Attention ──
    n_heads: int = 16
    q_lora_rank: int = 256
    head_dim: int = 64
    rope_head_dim: int = 16
    o_groups: int = 2
    o_lora_rank: int = 256
    attn_logits_soft_cap: Optional[float] = 30.0
    final_logit_softcap: Optional[float] = 30.0
    sliding_window_size: int = 512
    window_size: int = 512  # alias for sliding_window_size

    # ── Gemma-4 Hybrid Attention ──
    attention_type: Literal["local", "global", "hybrid"] = "local"
    global_heads: int = 4
    global_key_size: int = 512
    k_eq_v_global: bool = False
    qk_norm_with_scale: bool = True
    local_base_frequency: int = 10_000
    global_base_frequency: int = 1_000_000

    # ── Indexer ──
    index_n_heads: int = 4
    indexer_head_dim: int = 32
    index_topk: int = 16
    compress_rope_theta: float = 40000.0

    # ── Compression ──
    csa_compression_ratio: int = 4
    hca_compression_ratio: int = 128

    # ── MoE ──
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_group: int = 1
    topk_group: int = 1
    moe_inter_dim: int = 0
    moe_latent_dim: Optional[int] = None
    moe_dual_inter_dim: int = 0
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    expert_capacity_factor: float = 1.25
    moe_load_balance_coeff: float = 0.01
    router_z_loss_coeff: float = 0.001
    moe_router_entropy_coeff: float = 0.001
    moe_capacity_loss_coeff: float = 0.01
    moe_expert_dropout: float = 0.0
    n_hash_layers: int = 0
    moe_dual_ffn: bool = True
    swiglu_limit: float = 10.0

    # ── Quantization ──
    quant_config: QuantConfig = field(default_factory=QuantConfig)
    use_mxfp4_weights: bool = False


    # ── SSM ──
    ssm_heads: int = 8
    ssm_state_dim: int = 16
    ssm_kernel_size: int = 4
    ssm_chunk_size: int = 64
    ssm_dt_min: float = 0.001
    ssm_dt_max: float = 0.1
    ssm_dt_init_floor: float = 0.0001
    ssm_n_groups: int = 1
    ssm_d_skip: bool = True

    # ── HC ──
    num_residual_streams: int = 4
    hc_sinkhorn_iters: int = 8
    hc_eps: float = 1e-6

    # ── Concept Memory ──
    num_concepts: int = 128
    num_abstract_concepts: int = 16
    num_global_concepts: int = 4
    codebook_size: int = 512
    codebook_dim: int = 32
    hcm_ema_alpha: float = 0.99
    hcm_commit_loss_coeff: float = 0.25
    entropy_threshold: float = 0.5
    lightning_topk_blocks: int = 4
    ema_bias_lr: float = 0.01

    # ── Hybrid Concept Attention ──
    concept_topk: int = 8
    concept_ratio: float = 0.4
    predictive_coding_coeff: float = 0.01

    # ── MTP ──
    n_mtp_layers: int = 1

    # ── Einsum Parameterization (Gemma-4) ──
    use_einsum: bool = False

    # ── RoPE / YaRN ──
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 0

    # ── Vision / Audio ──
    vision_dim: int = 0
    audio_dim: int = 0
    n_vision_layers: int = 16
    n_audio_layers: int = 12
    audio_conformer_dims: int = 1024
    audio_lm_dims: int = 1536
    audio_feature_dim: int = 80
    per_layer_input_dim: int = 64

    # ── Reasoning ──
    reasoning_steps: int = 1
    think_token_id: int = 0
    answer_token_id: int = 1
    cot_exit_confidence: float = 0.9

    # ── Steering ──
    steering_attributes: List[str] = field(
        default_factory=lambda: [
            "creativity",
            "helpfulness",
            "complexity",
            "scientific_rigor",
        ]
    )

    # ── External Embedding Fusion ──
    external_embedding_dim: int = 0
    external_embedding_scale: float = 0.25
    external_embedding_norm: bool = True

    # ── Regularization / Norms ──
    post_attn_norm: bool = True
    post_ffw_norm: bool = True
    token_concept_loss_coeff: float = 0.05

    # ── KV Cache ──
    frac_shared_layers: float = 0.5
    use_fp8_kv: bool = False
    use_turboquant: bool = False
    use_kv_eviction: bool = False
    use_compaction: bool = False

    # ── Block AttnRes ──
    use_block_attnres: bool = False
    block_attnres_block_size: int = 16
    block_attnres_n_blocks: int = 4

    # ── Ring Attention ──
    use_ring_attention: bool = False

    # ── Stability ──
    stability_enabled: bool = False
    stability_drift_window_size: int = 100
    stability_entropy_collapse_sigma: float = 3.0
    stability_logit_norm_spike_sigma: float = 3.0
    stability_base_temperature: float = 0.7
    stability_min_temperature: float = 0.1
    stability_max_temperature: float = 2.0
    stability_cache_check_interval: int = 64
    stability_long_context_soft: int = 500_000
    stability_long_context_hard: int = 1_000_000
    stability_long_gen_soft: int = 5_000
    stability_long_gen_hard: int = 10_000

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def effective_moe_inter_dim(self) -> int:
        if self.moe_inter_dim > 0:
            return self.moe_inter_dim
        expert_dim = (
            self.moe_latent_dim
            if (self.moe_latent_dim is not None and self.moe_latent_dim < self.dim)
            else self.dim
        )
        return int(2 * 4 * expert_dim / 3)

    @property
    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            o_groups=self.o_groups,
            eps=self.norm_eps,
            attn_logits_soft_cap=self.attn_logits_soft_cap,
            window_size=self.sliding_window_size,
            index_n_heads=self.index_n_heads,
            indexer_head_dim=self.indexer_head_dim,
            index_topk=self.index_topk,
            compress_rope_theta=self.compress_rope_theta,
            attention_type=self.attention_type,
            global_heads=self.global_heads,
            global_key_size=self.global_key_size,
            k_eq_v_global=self.k_eq_v_global,
            qk_norm_with_scale=self.qk_norm_with_scale,
            local_base_frequency=self.local_base_frequency,
            global_base_frequency=self.global_base_frequency,
        )

    @property
    def moe_config(self) -> MoEConfig:
        return MoEConfig(
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            n_activated_experts=self.n_activated_experts,
            n_group=self.n_group,
            topk_group=self.topk_group,
            moe_inter_dim=self.moe_inter_dim,
            moe_latent_dim=self.moe_latent_dim,
            moe_dual_inter_dim=self.moe_dual_inter_dim,
            score_func=self.score_func,
            route_scale=self.route_scale,
            expert_capacity_factor=self.expert_capacity_factor,
            load_balance_coeff=self.moe_load_balance_coeff,
            router_z_loss_coeff=self.router_z_loss_coeff,
            router_entropy_coeff=self.moe_router_entropy_coeff,
            capacity_loss_coeff=self.moe_capacity_loss_coeff,
            moe_expert_dropout=self.moe_expert_dropout,
            n_hash_layers=self.n_hash_layers,
            moe_dual_ffn=self.moe_dual_ffn,
        )

    @property
    def compressor_config(self) -> CompressorConfig:
        return CompressorConfig(
            csa_compression_ratio=self.csa_compression_ratio,
            hca_compression_ratio=self.hca_compression_ratio,
        )

    @property
    def ssm_config(self) -> SSMConfig:
        return SSMConfig(
            heads=self.ssm_heads,
            state_dim=self.ssm_state_dim,
            kernel_size=self.ssm_kernel_size,
            chunk_size=self.ssm_chunk_size,
            dt_min=self.ssm_dt_min,
            dt_max=self.ssm_dt_max,
            dt_init_floor=self.ssm_dt_init_floor,
            n_groups=self.ssm_n_groups,
            d_skip=self.ssm_d_skip,
        )

    @property
    def mhc_config(self) -> MHCConfig:
        return MHCConfig(
            num_residual_streams=self.num_residual_streams,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )

    @property
    def stability_config(self) -> StabilityConfig:
        return StabilityConfig(
            enabled=self.stability_enabled,
            drift_window_size=self.stability_drift_window_size,
            entropy_collapse_threshold=self.stability_entropy_collapse_sigma,
            logit_norm_spike_threshold=self.stability_logit_norm_spike_sigma,
            base_temperature=self.stability_base_temperature,
            min_temperature=self.stability_min_temperature,
            max_temperature=self.stability_max_temperature,
            kmin_cache_check_interval=self.stability_cache_check_interval,
            long_context_threshold_1=self.stability_long_context_soft,
            long_context_threshold_2=self.stability_long_context_hard,
            long_gen_decay_1=self.stability_long_gen_soft,
            long_gen_decay_2=self.stability_long_gen_hard,
        )
