# Requirements Document

## Introduction

This document specifies the requirements for the Lasmoid Next-Gen model build: transforming the existing 1B-parameter hybrid transformer-SSM architecture into a production-grade 2M-context, hyper-compressed, multimodal, days-stable agentic model. Requirements are derived from the approved design covering 7 priority gaps: NVFP4 KV cache quantization, layer-aware eviction, OMP compaction, MoE weight quantization, multimodal encoders, ring attention distributed prefill, and the training pipeline.

## Glossary

- **Lasmoid_Core**: The 28-block hybrid transformer-SSM model backbone (4:1 local:global attention pattern)
- **KV_Cache_System**: The tiered key-value cache comprising hot (BF16), warm (FP8), and cold (NVFP4) storage tiers
- **NVFP4_Quantizer**: The module performing E2M1 (2 exponent + 1 mantissa) 4-bit quantization with 2D block scaling
- **RHT**: Random Hadamard Transform — an orthogonal rotation applied before quantization to distribute outlier magnitudes
- **PyramidKV_Evictor**: The layer-aware KV eviction module implementing SnapKV scoring with pyramid budget allocation
- **OMP_Compactor**: The Orthogonal Matching Pursuit module that merges pruned tokens into their nearest kept neighbors
- **Ring_Attention_Prefill**: The distributed prefill module splitting sequences across GPUs with ring-pass KV communication
- **WSD_Scheduler**: Warmup-Stable-Decay learning rate schedule used during training
- **MOPD_Distiller**: Multi-teacher On-Policy Distillation module for knowledge transfer from larger models
- **MoE_Quantizer**: The per-operator precision module applying NVFP4 to routed experts, FP8 to shared expert, BF16 to attention
- **SigLIP_Encoder**: The 16-layer vision encoder producing 280 tokens per image
- **Conformer_Encoder**: The 12-layer audio encoder with 4× subsampling
- **Hot_Tier**: The BF16 full-precision buffer holding the most recent 512 tokens per head
- **Cold_Tier**: The NVFP4-quantized long-term KV store
- **Drift_Detector**: The stability monitor tracking entropy, repetition, and confidence degradation signals
- **Adaptive_Temperature_Scheduler**: The module adjusting sampling temperature in response to drift signals

## Requirements

### Requirement 1: NVFP4 KV Cache Quantization

**User Story:** As a model operator, I want KV cache entries quantized to NVFP4 E2M1 format with 2D block scaling, so that 2M-context inference fits within GPU memory (112 GB → ~1.2 GB).

#### Acceptance Criteria

1. WHEN a KV tensor is written to the Cold_Tier, THE NVFP4_Quantizer SHALL apply Random Hadamard Transform followed by 2D block quantization to E2M1 format with per-block scales
2. WHEN a quantized KV tensor is read from the Cold_Tier, THE NVFP4_Quantizer SHALL dequantize the E2M1 packed data back to BF16 using stored per-block scales and inverse RHT
3. THE NVFP4_Quantizer SHALL produce quantized output whose dequantized reconstruction error is bounded by the E2M1 maximum representable step size per block
4. WHEN the head_dim is not divisible by the configured block_size, THE NVFP4_Quantizer SHALL raise a configuration error before quantization begins
5. THE NVFP4_Quantizer SHALL store quantized data at 4 bits per element plus per-block FP32 scales, achieving at least 4× memory reduction versus BF16 storage

### Requirement 2: Tiered KV Cache Storage

**User Story:** As a model operator, I want a tiered hot/warm/cold KV cache, so that recent tokens remain at full precision while older tokens are progressively compressed.

#### Acceptance Criteria

1. THE KV_Cache_System SHALL maintain a Hot_Tier buffer of the most recent 512 tokens per head stored at BF16 precision with zero quantization error
2. WHEN the Hot_Tier buffer reaches capacity, THE KV_Cache_System SHALL flush the oldest entries to the warm tier at FP8 precision
3. WHEN warm tier entries exceed the configured budget, THE KV_Cache_System SHALL promote them to Cold_Tier at NVFP4 precision
4. WHILE a token resides in the Hot_Tier, THE KV_Cache_System SHALL return that token at BF16 precision without any lossy transformation
5. WHEN a read request spans multiple tiers, THE KV_Cache_System SHALL dequantize each tier to a common BF16 format before returning concatenated results

### Requirement 3: PyramidKV Layer-Aware Eviction

**User Story:** As a model operator, I want layer-aware KV eviction with pyramid budget allocation, so that deeper layers retain more KV entries while the total cache stays within budget.

#### Acceptance Criteria

1. THE PyramidKV_Evictor SHALL compute per-layer budgets using the formula: `budget = max_budget - layer_idx × ((max_budget - min_budget) / (n_layers - 1))` where deeper layers receive higher budgets
2. FOR ALL layer indices, THE PyramidKV_Evictor SHALL guarantee that the computed budget is at least `sink_size + observation_window` tokens
3. WHEN the current sequence length exceeds the computed layer budget, THE PyramidKV_Evictor SHALL score KV positions using attention weights from the most recent `observation_window` queries
4. WHEN scoring KV positions, THE PyramidKV_Evictor SHALL apply 1D average-pool smoothing with the configured kernel size before selecting top-k positions
5. WHEN eviction is triggered, THE PyramidKV_Evictor SHALL always preserve the first `sink_size` tokens and the last `observation_window` tokens regardless of score
6. WHEN eviction completes, THE PyramidKV_Evictor SHALL produce an output of exactly `computed_budget` tokens — no more, no less
7. WHEN the current sequence length is at or below the computed layer budget, THE PyramidKV_Evictor SHALL return the cache unchanged without performing eviction

### Requirement 4: OMP Compaction Integration

**User Story:** As a model operator, I want pruned KV tokens merged into their nearest kept neighbors via OMP, so that information from evicted tokens is preserved rather than discarded.

#### Acceptance Criteria

1. WHEN eviction produces a set of pruned tokens, THE OMP_Compactor SHALL assign each pruned token to its nearest kept token by key-space cosine similarity
2. THE OMP_Compactor SHALL merge pruned token values into their assigned kept token using softmax-normalized similarity weights
3. FOR ALL compaction runs, THE OMP_Compactor SHALL ensure every pruned token contributes to exactly one kept token
4. FOR ALL compaction runs, THE OMP_Compactor SHALL ensure that merge weights for each pruned token sum to 1.0
5. WHEN compaction completes, THE OMP_Compactor SHALL return tensors of shape (batch, n_kept, head_dim) matching the kept set dimensions
6. THE OMP_Compactor SHALL leave kept keys unchanged — only kept values receive merged contributions
7. IF compaction produces NaN or Inf values, THEN THE OMP_Compactor SHALL log the anomaly and return unmerged kept values as fallback

### Requirement 5: NVFP4 MoE Weight Quantization

**User Story:** As a model operator, I want MoE routed expert weights quantized to NVFP4, so that MoE memory drops from 400 MB to 100 MB while maintaining inference quality.

#### Acceptance Criteria

1. THE MoE_Quantizer SHALL quantize all routed expert linear weights (w1, w2, w3) to NVFP4 E2M1 format with 2D block scaling (128×128 tiles)
2. THE MoE_Quantizer SHALL keep shared expert weights at FP8 precision
3. THE MoE_Quantizer SHALL keep attention layer weights at BF16 precision
4. THE MoE_Quantizer SHALL keep SSM recurrence weights at BF16 precision to avoid error amplification through sequential state updates
5. WHEN a quantized expert weight is used in a forward pass, THE MoE_Quantizer SHALL dequantize to BF16 before the GEMM operation
6. THE MoE_Quantizer SHALL apply Random Hadamard Transform before quantization to distribute outlier magnitudes across the weight matrix
7. WHEN weight dimensions are not divisible by the 128×128 block size, THE MoE_Quantizer SHALL pad the weight matrix to the nearest block boundary before quantization

### Requirement 6: Multimodal Encoder Integration

**User Story:** As a model developer, I want SigLIP vision and Conformer audio encoders integrated into the training pipeline, so that Lasmoid processes image and audio inputs alongside text.

#### Acceptance Criteria

1. WHEN an image input is provided, THE SigLIP_Encoder SHALL encode it into exactly 280 tokens using 16 transformer layers
2. WHEN an audio waveform is provided, THE Conformer_Encoder SHALL encode it with 4× temporal subsampling across 12 conformer layers
3. WHEN multimodal inputs are present, THE Lasmoid_Core SHALL fuse encoder output tokens with text embeddings at the embedding layer before the first block
4. WHEN only text input is provided, THE Lasmoid_Core SHALL bypass encoder paths with zero additional computation or latency
5. THE SigLIP_Encoder SHALL produce output tokens of the same embedding dimension as the Lasmoid_Core text embeddings
6. THE Conformer_Encoder SHALL produce output tokens of the same embedding dimension as the Lasmoid_Core text embeddings

### Requirement 7: Ring Attention Distributed Prefill

**User Story:** As a model operator, I want distributed ring attention prefill across multiple GPUs, so that 2M-token sequences can be processed without OOM on a single device.

#### Acceptance Criteria

1. WHEN a sequence exceeds single-GPU memory capacity, THE Ring_Attention_Prefill SHALL split the sequence into equal segments across `world_size` GPUs
2. THE Ring_Attention_Prefill SHALL produce attention output equivalent to single-GPU full-sequence attention within floating-point accumulation tolerance (absolute difference less than 1e-5)
3. WHEN computing partial attention from rotated KV segments, THE Ring_Attention_Prefill SHALL accumulate outputs using log-sum-exp correction for numerical stability
4. THE Ring_Attention_Prefill SHALL enforce causal masking such that no token attends to future positions, regardless of which GPU holds those positions
5. WHEN the sequence length is not evenly divisible by world_size, THE Ring_Attention_Prefill SHALL pad the sequence to the nearest multiple before splitting
6. THE Ring_Attention_Prefill SHALL overlap ring-pass NCCL communication with attention computation where hardware supports it
7. IF an NCCL communication timeout occurs during ring-pass, THEN THE Ring_Attention_Prefill SHALL fall back to sequential single-GPU processing and log the failure
8. WHEN prefill completes on 8×H100 GPUs, THE Ring_Attention_Prefill SHALL complete a 2M-token prefill in less than 60 seconds

### Requirement 8: WSD Training Schedule

**User Story:** As a model trainer, I want a Warmup-Stable-Decay learning rate schedule, so that training converges reliably with predictable LR behavior at each phase.

#### Acceptance Criteria

1. WHILE the training step is in the warmup phase (step less than warmup_steps), THE WSD_Scheduler SHALL increase the learning rate linearly from min_lr to peak_lr
2. WHILE the training step is in the stable phase (between warmup_steps and warmup_steps + stable_steps), THE WSD_Scheduler SHALL hold the learning rate constant at peak_lr
3. WHILE the training step is in the decay phase (beyond warmup_steps + stable_steps), THE WSD_Scheduler SHALL decrease the learning rate via cosine decay from peak_lr to min_lr
4. FOR ALL training steps, THE WSD_Scheduler SHALL guarantee the learning rate stays within the bounds [min_lr, peak_lr]
5. THE WSD_Scheduler SHALL be monotonically non-decreasing during warmup, constant during stable, and monotonically non-increasing during decay

### Requirement 9: MOPD Distillation

**User Story:** As a model trainer, I want multi-teacher on-policy distillation, so that the 1B student model absorbs capabilities from larger teacher models during training.

#### Acceptance Criteria

1. WHEN distillation is enabled, THE MOPD_Distiller SHALL compute KL divergence between the student distribution and each teacher distribution on the student's own generated samples
2. THE MOPD_Distiller SHALL weight the KL loss term by the configured `kl_coeff` (default 0.1) before adding it to the primary training loss
3. WHEN multiple teachers are configured, THE MOPD_Distiller SHALL average their KL contributions equally
4. THE MOPD_Distiller SHALL use on-policy sampling (student-generated tokens) rather than teacher-generated tokens for distillation targets
5. IF a teacher model fails to produce logits for a batch, THEN THE MOPD_Distiller SHALL skip that teacher for that batch and log a warning

### Requirement 10: Progressive Length Extension

**User Story:** As a model trainer, I want progressive context length extension during training, so that the model gradually learns to handle 2M-token sequences without instability.

#### Acceptance Criteria

1. THE training pipeline SHALL progress through length stages [4096, 32768, 262144, 1048576, 2097152] in order, spending `steps_per_stage` training steps at each stage
2. WHEN transitioning between length stages, THE training pipeline SHALL increase the maximum sequence length to the next stage value without resetting optimizer state
3. WHEN the final stage (2097152) is reached, THE training pipeline SHALL continue training at that length for the remaining configured steps
4. THE training pipeline SHALL adjust batch size inversely with sequence length to maintain constant token throughput per step

### Requirement 11: Stability and Drift Detection

**User Story:** As a model operator, I want continuous drift detection during long generation sessions, so that the model remains stable over 72+ hours of continuous operation.

#### Acceptance Criteria

1. WHILE generation is active, THE Drift_Detector SHALL monitor entropy, repetition rate, and confidence metrics at each generated token
2. WHEN entropy collapse is detected (entropy below configured threshold for consecutive tokens), THE Drift_Detector SHALL signal the Adaptive_Temperature_Scheduler
3. WHEN a drift signal is received, THE Adaptive_Temperature_Scheduler SHALL increase the sampling temperature by 0.2
4. FOR ALL generation steps with stability enabled, THE Adaptive_Temperature_Scheduler SHALL keep the adjusted temperature within the bounds [0.1, 2.0]
5. IF drift signals persist for more than 10 consecutive tokens, THEN THE Drift_Detector SHALL trigger L2 recovery: restore from last checkpoint and increase base temperature
6. THE KV_Cache_System SHALL perform integrity checks (NaN/Inf detection) on cached values at configurable intervals during long-running generation

### Requirement 12: Error Recovery

**User Story:** As a model operator, I want robust error recovery for quantization failures, OOM events, and communication errors, so that inference degrades gracefully rather than crashing.

#### Acceptance Criteria

1. IF the NVFP4_Quantizer produces NaN values in a cache segment, THEN THE KV_Cache_System SHALL reset that segment, fall back to FP8 quantization for the affected layer, and log the incident
2. IF GPU memory is exhausted during prefill, THEN THE KV_Cache_System SHALL switch to emergency eviction mode (compression ratio 512×) and continue with reduced context
3. IF ring-pass communication fails, THEN THE Ring_Attention_Prefill SHALL fall back to sequential single-GPU processing with a performance degradation warning
4. IF all KV positions receive near-identical eviction scores (score variance below threshold), THEN THE PyramidKV_Evictor SHALL fall back to recency-based eviction and log the anomaly
