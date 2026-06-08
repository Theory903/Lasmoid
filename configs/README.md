# Lasmoid Config Reference

All model configuration files live in `configs/model/`. Each JSON file is a full `ModelArgs` definition.

---

## Model Size Quick Reference

| File | Params | dim | Layers | VRAM | Use Case |
|------|--------|-----|--------|------|----------|
| `config_10m.json`  | ~10M  | 256 | 6  | 0.2 GB | Fast experiments, CPU |
| `config_100m.json` | ~100M | 384 | 12 | 1.0 GB | **Kaggle T4 default** |
| `config_300m.json` | ~300M | 640 | 18 | 2.5 GB | Better quality, T4/A10 |
| `config_500m.json` | ~500M | 768 | 24 | 4.0 GB | Near-GPT2-large quality |
| `config_1b.json`   | ~1B   | 1024| 28 | 8.0 GB | A100 pretraining |
| `config_1b_2m.json`| ~1B   | 1024| 28 | 8.0 GB | 1B variant with 2M ctx |
| `config_gemma4_100m.json` | ~100M | 384 | 12 | 1.0 GB | Optimized for Gemma-4 distillation |
| `config_gemma4_1b.json`   | ~1B  | 1024 | 28 | 8.0 GB | Large-scale Gemma-4 distillation |

---

## Field Reference (ModelArgs)

### Core Dimensions

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `dim` | int | 384 | Model hidden dimension |
| `n_layers` | int | 12 | Number of transformer blocks |
| `n_heads` | int | 6 | Number of attention heads (Q) |
| `head_dim` | int | 48 | Dimension per attention head |
| `vocab_size` | int | 129280 | Vocabulary size (BPE tokenizer) |
| `max_seq_len` | int | 512 | Maximum sequence length |
| `max_batch_size` | int | 4 | Maximum batch size for KV cache |
| `dtype` | str | "bf16" | Compute dtype (`bf16` or `fp32`) |

### Attention (CSA — Compressed Sparse Attention)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `q_lora_rank` | int | 96 | LoRA rank for Q projection (MLA) |
| `o_lora_rank` | int | 96 | LoRA rank for O projection |
| `rope_head_dim` | int | 16 | RoPE-active head dimensions (rest are NoPE) |
| `o_groups` | int | 2 | GQA grouping for output projection |
| `rope_theta` | float | 10000.0 | RoPE base frequency |
| `rope_factor` | float | 1.0 | YaRN rope extension factor (1.0 = no extension) |
| `beta_fast` | int | 32 | YaRN fast dim boundary |
| `beta_slow` | int | 1 | YaRN slow dim boundary |
| `original_seq_len` | int | 0 | Original pretraining length for YaRN (0 = auto) |

### SSM (Mamba-2 / SSD)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `ssm_heads` | int | 6 | Number of SSM heads |
| `ssm_state_dim` | int | 16 | SSM hidden state dimension per head |
| `ssm_kernel_size` | int | 4 | 1D conv kernel size before SSM |
| `ssm_chunk_size` | int | 64 | Chunk size for parallel SSD scan |
| `ssm_dt_min` | float | 0.001 | Min time step (dt) after softplus |
| `ssm_dt_max` | float | 0.1 | Max time step (dt) after softplus |
| `ssm_dt_init_floor` | float | 0.0001 | Floor for dt initialization |
| `ssm_n_groups` | int | 1 | SSM grouping (>1 = grouped scan) |
| `ssm_d_skip` | bool | true | Enable skip connection in SSM output |

### MoE (Grey-Box Mixture of Experts)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `n_routed_experts` | int | 6 | Total routed expert count |
| `n_shared_experts` | int | 1 | Always-active shared experts |
| `n_activated_experts` | int | 2 | Top-K experts selected per token |
| `moe_latent_dim` | int | 192 | Expert hidden dimension (< dim for efficiency) |
| `expert_capacity_factor` | float | 1.25 | Buffer factor to avoid token dropping |
| `moe_dual_ffn` | bool | true | Add dense FFN branch alongside MoE |
| `moe_load_balance_coeff` | float | 0.01 | Load balance regularization weight |
| `moe_capacity_loss_coeff` | float | 0.01 | Capacity overflow penalty |
| `moe_router_entropy_coeff` | float | 0.001 | Router entropy regularization |
| `router_z_loss_coeff` | float | 0.001 | Z-loss to prevent router collapse |
| `ema_bias_lr` | float | 0.01 | EMA learning rate for routing bias update |

### mHC (Manifold-Constrained Hyper-Connections)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `num_residual_streams` | int | 4 | Parallel residual stream count (M) |
| `hc_sinkhorn_iters` | int | 8 | Sinkhorn normalization iterations |
| `hc_eps` | float | 1e-6 | Sinkhorn numerical stability epsilon |
| `hcm_ema_alpha` | float | 0.99 | EMA decay for stream statistics |
| `hcm_commit_loss_coeff` | float | 0.25 | Stream commitment loss weight |

### Concept Memory (ESCM)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `num_concepts` | int | 64 | Episodic concept slot count |
| `num_abstract_concepts` | int | 8 | Semantic (abstract) concept slots |
| `num_global_concepts` | int | 2 | Global concept slots |
| `codebook_size` | int | 256 | VQ codebook entry count |
| `token_concept_loss_coeff` | float | 0.05 | Token↔concept alignment loss |

### Reasoning & MTP

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `reasoning_steps` | int | 2 | Internal reasoning loop iterations |
| `think_token_id` | int | 128821 | Token ID for `<think>` |
| `answer_token_id` | int | 129285 | Token ID for answer boundary |
| `cot_exit_confidence` | float | 0.9 | CoT early-exit confidence threshold |
| `lightning_topk_blocks` | int | 2 | Lightning attention top-K block count |

### Normalization

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `norm_eps` | float | 1e-6 | RMSNorm epsilon |
| `swiglu_limit` | float | 10.0 | SwiGLU gate clamp limit |
| `post_attn_norm` | bool | true | Post-attention RMSNorm |
| `post_ffw_norm` | bool | true | Post-feedforward RMSNorm |

### Steering

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `steering_attributes` | list | `["creativity","helpfulness",...]` | Named activation steering dimensions |
| `predictive_coding_coeff` | float | 0.01 | Predictive coding auxiliary loss weight |
| `entropy_threshold` | float | 0.5 | Entropy gate threshold for sparse routing |

---

## Example: 100M Config (`config_100m.json`)

```json
{
  "dim": 384,
  "n_layers": 12,
  "n_heads": 6,
  "head_dim": 48,
  "vocab_size": 129280,
  "max_seq_len": 512,
  "n_routed_experts": 6,
  "n_activated_experts": 2,
  "n_shared_experts": 1,
  "ssm_heads": 6,
  "ssm_state_dim": 16,
  "num_residual_streams": 4,
  "codebook_size": 256,
  "num_concepts": 64
  // ... (see file for full spec)
}
```

---

## Creating a Custom Config

```python
from inference.config import ModelArgs
import json

# Start from 100M and scale up
with open("configs/model/config_100m.json") as f:
    cfg = json.load(f)

# Scale to ~200M
cfg["dim"] = 512
cfg["n_layers"] = 16
cfg["n_heads"] = 8
cfg["head_dim"] = 64
cfg["moe_latent_dim"] = 256
cfg["codebook_size"] = 384

# Validate
args = ModelArgs(**cfg)
print(f"Config valid: dim={args.dim}, layers={args.n_layers}")

with open("configs/model/config_200m_custom.json", "w") as f:
    json.dump(cfg, f, indent=2)
```
