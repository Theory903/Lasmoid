import sys
import os
import json

# Add root folder to sys.path
lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lasmoid_dir)

import torch
from inference.model import Lasmoid, ModelArgs

def get_params_for_args(args):
    # Initialize model without loading weights to count parameters
    with torch.device("meta"):
        model = Lasmoid(args)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def optimize():
    # Base configuration template matching our config.json structure
    base_config = {
        "vocab_size": 129286,
        "max_seq_len": 512,  # Expand sequence length for longer context reasoning
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
        "reasoning_steps": 2,  # Increased reasoning budget for deep thinking
        "think_token_id": 128821,
        "answer_token_id": 129285,
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
    
    # We will search the design space to maximize representation capacity (dim, n_layers, MoE, SSM)
    # under the 100M parameter limit.
    candidates = []
    
    # Option Grid
    dims = [192, 256, 320, 384]
    layers = [4, 6, 8, 10, 12]
    routed_experts = [4, 6, 8]
    activated_experts = [2]
    
    for d in dims:
        for l in layers:
            for re in routed_experts:
                # Adjust attention and SSM proportions to match dimension d
                config = base_config.copy()
                config.update({
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
                
                # Build arguments dataclass
                from dataclasses import fields
                valid_fields = {f.name for f in fields(ModelArgs)}
                filtered_config = {k: v for k, v in config.items() if k in valid_fields}
                args = ModelArgs(**filtered_config)
                
                try:
                    params = get_params_for_args(args)
                    if params < 100_000_000:
                        candidates.append((params, config))
                except Exception as e:
                    # Skip invalid shape configurations
                    continue
                    
    # Sort candidates by parameter count (descending, to get closest to 100M)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Top 5 configuration candidates under 100M limit:")
    for i, (p, cfg) in enumerate(candidates[:5]):
        print(f"\n{i+1}. Configuration (Parameters: {p:,})")
        print(f"   - Model Dimension (dim)  : {cfg['dim']}")
        print(f"   - Number of Layers (n_layers)  : {cfg['n_layers']}")
        # effective experts = routed + shared
        print(f"   - Routed Experts: {cfg['n_routed_experts']} | Shared Experts: {cfg['n_shared_experts']} | Activated: {cfg['n_activated_experts']}")
        print(f"   - Attention Heads: {cfg['n_heads']} | SSM Heads: {cfg['ssm_heads']}")
        
    if candidates:
        best_cfg = candidates[0][1]
        best_params = candidates[0][0]
        print(f"\nWriting best configuration (Parameters: {best_params:,}) to config_100m.json")
        with open(os.path.join(lasmoid_dir, "config_100m.json"), "w") as f:
            json.dump(best_cfg, f, indent=2)
            
if __name__ == "__main__":
    optimize()
