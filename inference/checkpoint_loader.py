"""
Lasmod — checkpoint_loader.py
========================================
Checkpoint discovery, validation, and model initialization.
Supports partial loading for debugging and cross-architecture warm-start.
"""

import glob
import json
import os
import re
from dataclasses import fields
from typing import Optional

import torch

try:
    from .model import Lasmoid, ModelArgs
except ImportError:
    from model import Lasmoid, ModelArgs


def select_checkpoint_file(ckpt_path: str) -> Optional[str]:
    """Find the best checkpoint in a directory (latest > step > final)."""
    latest_pt = os.path.join(ckpt_path, "lasmoid_latest.pt")
    if os.path.exists(latest_pt):
        return latest_pt

    step_files = glob.glob(os.path.join(ckpt_path, "lasmoid_step_*.pt"))
    if step_files:

        def step_num(path: str) -> int:
            stem = os.path.basename(path)
            match = re.search(r"lasmoid_step_(\d+)\.pt$", stem)
            return int(match.group(1)) if match else -1

        return max(step_files, key=step_num)

    final = os.path.join(ckpt_path, "lasmoid_final.pt")
    if os.path.exists(final):
        return final

    return None


def _prepare_state_dict_for_model(sd: dict, target_model: Lasmoid) -> None:
    """Pad tokenizer tensors and drop stale architecture tensors before loading."""
    model_sd = target_model.state_dict()
    target_rows = target_model.args.vocab_size
    vocab_keys = [
        key
        for key, target in model_sd.items()
        if target.ndim == 2 and target.shape[0] == target_rows and key in sd
    ]

    for key in vocab_keys:
        tensor = sd.get(key)
        if tensor is None or tensor.ndim != 2 or tensor.shape[0] == target_rows:
            continue
        if tensor.shape[0] > target_rows:
            sd[key] = tensor[:target_rows].contiguous()
            print(
                f"[checkpoint] Trimmed {key} from {tensor.shape[0]} to {target_rows} rows."
            )
            continue

        target_tensor = model_sd[key].detach().to(tensor.device, dtype=tensor.dtype)
        target_tensor[: tensor.shape[0]].copy_(tensor)
        sd[key] = target_tensor
        print(
            f"[checkpoint] Expanded {key} from {tensor.shape[0]} to {target_rows} rows."
        )

    dropped = []
    for key in list(sd.keys()):
        target = model_sd.get(key)
        if target is None:
            dropped.append(key)
            sd.pop(key)
            continue
        if sd[key].shape != target.shape:
            dropped.append(key)
            sd.pop(key)
    if dropped:
        print(
            f"[checkpoint] Skipped {len(dropped)} stale checkpoint tensors with incompatible shapes/names."
        )


def _validate_state_dict_compatibility(sd: dict, target_model: Lasmoid) -> list[str]:
    """Check all tensors in a state dict match the target model architecture."""
    model_sd = target_model.state_dict()
    param_keys = {name for name, _ in target_model.named_parameters()}
    buffer_keys = {name for name, _ in target_model.named_buffers()}
    issues = []

    for key, tensor in sd.items():
        target = model_sd.get(key)
        if target is None:
            if key not in buffer_keys:
                issues.append(f"unexpected tensor `{key}`")
            continue
        if key in param_keys and tensor.shape != target.shape:
            issues.append(
                f"shape mismatch for `{key}`: checkpoint {tuple(tensor.shape)} vs model {tuple(target.shape)}"
            )

    for key in param_keys:
        if key not in sd:
            issues.append(f"missing parameter `{key}`")

    return issues


def load_checkpoint_and_model(
    ckpt_path: str, config_path: str, device: str, allow_partial_load: bool = False
) -> tuple:
    """Load a checkpoint (or config-only init) and return (model, model_args)."""
    ckpt_file = select_checkpoint_file(ckpt_path)

    model_args = None
    state_dict = None
    if ckpt_file:
        try:
            sd = torch.load(ckpt_file, map_location=device, weights_only=False)
            state_dict = sd.get("model_state_dict", sd)
            model_args = sd.get("model_args")
            print(f"[checkpoint] Found checkpoint: {ckpt_file}")
        except Exception as e:
            print(f"[checkpoint] Failed to load checkpoint file directly: {e}")

    config_dict = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)

    if model_args is None:
        valid_fields = {f.name for f in fields(ModelArgs)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        model_args = ModelArgs(**filtered_config)
        print(f"[checkpoint] Loaded config from {config_path}")
    else:
        for key in (
            "vocab_size",
            "think_token_id",
            "answer_token_id",
            "cot_exit_confidence",
            "expert_capacity_factor",
            "moe_load_balance_coeff",
            "ssm_chunk_size",
            "ssm_dt_min",
            "ssm_dt_max",
            "ssm_dt_init_floor",
            "ssm_n_groups",
        ):
            if key in config_dict:
                setattr(model_args, key, config_dict[key])

    model = Lasmoid(model_args).to(device)

    if state_dict is not None:
        if getattr(model_args, "use_mxfp4_weights", False):
            try:
                from .kernel import load_mxfp4_weight
            except ImportError:
                from kernel import load_mxfp4_weight

            keys_to_dequant = []
            for key in list(state_dict.keys()):
                if key.endswith(".blocks"):
                    base_key = key[:-7]
                    scales_key = base_key + ".scales"
                    if scales_key in state_dict:
                        keys_to_dequant.append((base_key, key, scales_key))

            for base_key, blocks_key, scales_key in keys_to_dequant:
                blocks_tensor = state_dict.pop(blocks_key)
                scales_tensor = state_dict.pop(scales_key)
                dequantized = load_mxfp4_weight(blocks_tensor, scales_tensor, dtype=torch.bfloat16)
                state_dict[base_key] = dequantized
                print(f"[checkpoint] Dequantized MXFP4 weight for {base_key}")

        if allow_partial_load:
            print(
                "[checkpoint] WARNING: partial checkpoint load enabled; "
                "outputs are not expected to be coherent until retrained/fine-tuned."
            )
            _prepare_state_dict_for_model(state_dict, model)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(
                    f"[checkpoint] Initialised {len(missing)} new tensors from current model defaults."
                )
            if unexpected:
                print(f"[checkpoint] Ignored {len(unexpected)} unexpected tensors.")
            print("[checkpoint] Successfully loaded weights from checkpoint.")
        else:
            issues = _validate_state_dict_compatibility(state_dict, model)
            if issues:
                issue_text = "\n  - " + "\n  - ".join(issues[:20])
                raise RuntimeError(
                    "Checkpoint is incompatible with the current Lasmoid architecture.\n"
                    f"First issues:{issue_text}\n"
                    "Re-run with `--allow-partial-load` only for debugging "
                    "or use a compatible fresh checkpoint."
                )
            model.load_state_dict(state_dict, strict=True)
            print("[checkpoint] Successfully loaded weights from checkpoint.")
    else:
        print("[checkpoint] WARNING: Running with random initialisation.")

    return model, model_args
