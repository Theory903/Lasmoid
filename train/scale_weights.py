"""
Lasmoid — scale_weights.py
========================================================================
Scales and expands trained weights from a smaller model (e.g., 100M)
to warm-start a larger model (e.g., 1B) via progressive layer stacking
and dimension zero-padding.
"""

import os
import sys
import argparse
import torch
import json

# Add root folder to sys.path
lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lasmoid_dir)

from inference.model import Lasmoid, ModelArgs


def scale_tensor(old_tensor: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    new_tensor = torch.zeros(
        new_shape, dtype=old_tensor.dtype, device=old_tensor.device
    )
    slices = tuple(
        slice(0, min(old_dim, new_dim))
        for old_dim, new_dim in zip(old_tensor.shape, new_shape)
    )
    new_tensor[slices] = old_tensor[slices]
    return new_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Scale Lasmoid model weights from 100M to 1B."
    )
    parser.add_argument(
        "--src-ckpt",
        type=str,
        required=True,
        help="Path to smaller source checkpoint (e.g. lasmoid_step_1000.pt or lasmoid_final.pt)",
    )
    parser.add_argument(
        "--src-config",
        type=str,
        required=True,
        help="Path to source config.json (100M)",
    )
    parser.add_argument(
        "--target-config",
        type=str,
        required=True,
        help="Path to target config_1b.json (1B)",
    )
    parser.add_argument(
        "--out-ckpt",
        type=str,
        default="checkpoints/current/lasmoid_1b_warmstart.pt",
        help="Path to output 1B checkpoint",
    )
    args_cli = parser.parse_args()

    print("=== Model Weight Scaling (Progressive Growing) ===")

    # 1. Load Configurations
    with open(args_cli.src_config) as f:
        src_cfg_dict = json.load(f)
    with open(args_cli.target_config) as f:
        tgt_cfg_dict = json.load(f)

    src_args = ModelArgs(**src_cfg_dict)
    tgt_args = ModelArgs(**tgt_cfg_dict)

    print(f"Source Model Config: dim={src_args.dim}, layers={src_args.n_layers}")
    print(f"Target Model Config: dim={tgt_args.dim}, layers={tgt_args.n_layers}")

    # 2. Load Source Weights
    print(f"Loading source weights from {args_cli.src_ckpt}...")
    src_state = torch.load(args_cli.src_ckpt, map_location="cpu")

    # Check if checkpoint is wrapped in dict
    if "model_state_dict" in src_state:
        src_state = src_state["model_state_dict"]

    # 3. Initialize target 1B model structure on CPU
    print("Initializing target 1B model structure...")
    target_model = Lasmoid(tgt_args)
    tgt_state = target_model.state_dict()

    # 4. Map and expand weights
    print("Scaling and mapping weights...")
    scaled_state = {}

    # Helper to map source layers to target layers (progressive stacking)
    # Target has 28 layers, Source has 12 layers.
    # Map index: target_layer_idx * (src_layers / tgt_layers)
    def get_src_layer_idx(tgt_layer_idx):
        ratio = src_args.n_layers / tgt_args.n_layers
        return int(tgt_layer_idx * ratio)

    for name, tgt_param in tgt_state.items():
        # A. If it's a layer-specific weight (e.g., layers.N.attention.wq.weight)
        layer_match = name.split(".")
        if len(layer_match) > 1 and layer_match[0] == "layers":
            tgt_layer_idx = int(layer_match[1])
            src_layer_idx = get_src_layer_idx(tgt_layer_idx)

            # Construct equivalent source weight name
            src_name_list = layer_match.copy()
            src_name_list[1] = str(src_layer_idx)
            src_name = ".".join(src_name_list)

            if src_name in src_state:
                old_param = src_state[src_name]
                scaled_state[name] = scale_tensor(old_param, tgt_param.shape)
            else:
                print(
                    f"Warning: Layer weight {src_name} not found in source state. Using default target initialization."
                )
                scaled_state[name] = tgt_param

        # B. Global weights (embeddings, output projection, concept memory)
        else:
            if name in src_state:
                old_param = src_state[name]
                scaled_state[name] = scale_tensor(old_param, tgt_param.shape)
            else:
                print(
                    f"Warning: Global weight {name} not found in source state. Using default target initialization."
                )
                scaled_state[name] = tgt_param

    # 5. Load scaled state dict into target model structure
    target_model.load_state_dict(scaled_state)
    print("Weights scaled and verified successfully.")

    # 6. Save target checkpoint
    os.makedirs(os.path.dirname(args_cli.out_ckpt), exist_ok=True)
    torch.save(target_model.state_dict(), args_cli.out_ckpt)
    print(f"Warm-started 1B model checkpoint saved to: {args_cli.out_ckpt}")
    print("You can now train this model further using config_1b.json!")


if __name__ == "__main__":
    main()
