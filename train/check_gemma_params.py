import sys
import os
import json

# Add root folder to sys.path
lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lasmoid_dir)

import torch
from inference.model import Lasmoid, ModelArgs

def get_params_for_args(args):
    with torch.device("meta"):
        model = Lasmoid(args)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main():
    base_config = {
        "vocab_size": 262144,
        "max_seq_len": 512,
        "max_batch_size": 4,
        "dtype": "bf16",
        "norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "rope_factor": 1.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_seq_len": 0,
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
        "think_token_id": 107,  # Will verify with tokenizer later
        "answer_token_id": 108,
        "cot_exit_confidence": 0.9,
        "moe_router_entropy_coeff": 0.001,
        "moe_capacity_loss_coeff": 0.01,
        "token_concept_loss_coeff": 0.05,
        "steering_attributes": ["creativity", "helpfulness", "complexity", "scientific_rigor"],
        "post_attn_norm": True,
        "post_ffw_norm": True,
        "moe_dual_ffn": True,
        "ssm_d_skip": True,
    }

    # 1. Optimize 100M parameter candidates
    print("--- 100M Candidates search (vocab_size = 262,144) ---")
    candidates_100m = []
    # Try different dims and layer counts
    for d in [128, 192, 256]:
        for l in [4, 6, 8, 10, 12, 14, 16]:
            for re in [4, 6, 8]:
                cfg = base_config.copy()
                cfg.update({
                    "dim": d,
                    "n_layers": l,
                    "n_heads": max(4, d // 64),
                    "q_lora_rank": d // 4,
                    "head_dim": 48,
                    "rope_head_dim": 16,
                    "o_groups": 2,
                    "o_lora_rank": d // 4,
                    "n_routed_experts": re,
                    "n_shared_experts": 1,
                    "n_activated_experts": 2,
                    "moe_latent_dim": d // 2,
                    "num_concepts": 64,
                    "num_abstract_concepts": 8,
                    "num_global_concepts": 2,
                    "codebook_size": 256,
                    "lightning_topk_blocks": 2,
                    "ssm_heads": max(2, d // 64),
                    "ssm_state_dim": 16,
                    "ssm_kernel_size": 4,
                    "ssm_chunk_size": 64,
                    "ssm_dt_min": 0.001,
                    "ssm_dt_max": 0.1,
                    "ssm_dt_init_floor": 0.0001,
                    "ssm_n_groups": 1,
                })
                from dataclasses import fields
                valid_fields = {f.name for f in fields(ModelArgs)}
                filtered_config = {k: v for k, v in cfg.items() if k in valid_fields}
                args = ModelArgs(**filtered_config)
                try:
                    params = get_params_for_args(args)
                    if params < 100_000_000:
                        candidates_100m.append((params, cfg))
                except Exception as e:
                    continue

    candidates_100m.sort(key=lambda x: x[0], reverse=True)
    for i, (p, cfg) in enumerate(candidates_100m[:5]):
        print(f"{i+1}. Params: {p:,} | dim={cfg['dim']}, layers={cfg['n_layers']}, experts={cfg['n_routed_experts']}")

    # 2. Optimize 1B parameter candidates
    print("\n--- 1B Candidates search (vocab_size = 262,144) ---")
    candidates_1b = []
    # Try different dims and layer counts
    for d in [512, 768, 1024, 1152]:
        for l in [12, 16, 20, 24, 28, 32]:
            for re in [4, 6, 8]:
                cfg = base_config.copy()
                cfg.update({
                    "dim": d,
                    "n_layers": l,
                    "n_heads": max(4, d // 64),
                    "q_lora_rank": d // 4,
                    "head_dim": 48,
                    "rope_head_dim": 16,
                    "o_groups": 2,
                    "o_lora_rank": d // 4,
                    "n_routed_experts": re,
                    "n_shared_experts": 1,
                    "n_activated_experts": 2,
                    "moe_latent_dim": d // 2,
                    "num_concepts": 64,
                    "num_abstract_concepts": 8,
                    "num_global_concepts": 2,
                    "codebook_size": 256,
                    "lightning_topk_blocks": 2,
                    "ssm_heads": max(2, d // 64),
                    "ssm_state_dim": 16,
                    "ssm_kernel_size": 4,
                    "ssm_chunk_size": 64,
                    "ssm_dt_min": 0.001,
                    "ssm_dt_max": 0.1,
                    "ssm_dt_init_floor": 0.0001,
                    "ssm_n_groups": 1,
                })
                from dataclasses import fields
                valid_fields = {f.name for f in fields(ModelArgs)}
                filtered_config = {k: v for k, v in cfg.items() if k in valid_fields}
                args = ModelArgs(**filtered_config)
                try:
                    params = get_params_for_args(args)
                    if params < 1_000_000_000:
                        candidates_1b.append((params, cfg))
                except Exception as e:
                    continue

    candidates_1b.sort(key=lambda x: x[0], reverse=True)
    for i, (p, cfg) in enumerate(candidates_1b[:5]):
        print(f"{i+1}. Params: {p:,} | dim={cfg['dim']}, layers={cfg['n_layers']}, experts={cfg['n_routed_experts']}")

if __name__ == "__main__":
    main()
