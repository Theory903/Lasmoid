"""
Lasmoid — convert.py
====================
Weight converter and dtype conversion utility.
Casts weights between fp32, bf16, fp16, and FP8 formats.
Supports .pt (torch.save) and .safetensors output formats.
"""

import os
import sys
from argparse import ArgumentParser
from typing import Dict, Any, Optional

import torch

# ── Dtype mapping ────────────────────────────────────────────────────

DTYPE_MAP: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.bfloat16,
}

FP8_AVAILABLE = hasattr(torch, "float8_e4m3fn")


def _resolve_dtype(target_dtype: str) -> torch.dtype:
    if target_dtype == "fp8" and not FP8_AVAILABLE:
        print(
            "WARNING: torch.float8_e4m3fn not available (PyTorch ≥ 2.1 required). "
            "Falling back to bfloat16."
        )
        return torch.bfloat16
    if target_dtype not in DTYPE_MAP:
        raise ValueError(
            f"Unknown target dtype: {target_dtype!r}. "
            f"Choose from: {', '.join(DTYPE_MAP)}"
        )
    return DTYPE_MAP[target_dtype]


def _load_input(input_path: str) -> tuple[Dict[str, Any], bool]:
    if not os.path.exists(input_path):
        print(f"Error: input path {input_path} does not exist.")
        sys.exit(1)

    sd = torch.load(input_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        return sd["model_state_dict"], True
    return sd, False


def _save_output(
    state_dict: Dict[str, Any],
    output_path: str,
    nested_container: Optional[Dict[str, Any]],
    target_dtype: str,
) -> None:
    if output_path.endswith(".safetensors"):
        from safetensors.torch import save_file as st_save

        tensors_only = {
            k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)
        }
        st_save(tensors_only, output_path)
        print(
            f"Successfully converted checkpoint saved to: {output_path} (safetensors)"
        )
    elif nested_container is not None:
        nested_container["model_state_dict"] = state_dict
        if "model_args" in nested_container:
            nested_container["model_args"].dtype = target_dtype
        torch.save(nested_container, output_path)
        print(f"Successfully converted checkpoint saved to: {output_path}")
    else:
        torch.save(state_dict, output_path)
        print(f"Successfully converted checkpoint saved to: {output_path}")


def convert_checkpoint(
    input_path: str,
    output_path: str,
    target_dtype: str,
) -> None:
    """Load a checkpoint, cast all floating-point tensors, and save."""
    print(f"Loading checkpoint from: {input_path}")
    state_dict, is_nested = _load_input(input_path)
    torch_dtype = _resolve_dtype(target_dtype)

    print(f"Casting weights to dtype: {target_dtype} ({torch_dtype})")
    new_state_dict: Dict[str, Any] = {}
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor) and param.is_floating_point():
            new_state_dict[key] = param.to(torch_dtype)
        else:
            new_state_dict[key] = param

    nested_container = state_dict if is_nested else None
    _save_output(new_state_dict, output_path, nested_container, target_dtype)


if __name__ == "__main__":
    parser = ArgumentParser(description="Lasmoid Weight Converter")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input checkpoint (.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted checkpoint (.pt or .safetensors)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(DTYPE_MAP),
        default="bf16",
        help="Target data type",
    )
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output, args.dtype)
