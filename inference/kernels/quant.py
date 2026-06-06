"""
Lasmoid — kernels/quant.py
==========================
FP8 block-wise KV cache quantisation helpers.

Wraps `act_quant` / `weight_dequant` from `kernel.py` into ergonomic
quantise-store / dequantise-load primitives for KV cache tensors.

Memory saving: 2× on KV data (BF16→FP8), ~0.8 % overhead for per-block
scales (1 × fp32 per 128 elements per row).
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from ..kernel import act_quant, weight_dequant
except ImportError:
    from kernel import act_quant, weight_dequant


DEFAULT_BLOCK_SIZE: int = 128


def quantize_kv(
    kv: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    scale_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-wise FP8 quantisation of a KV cache tensor.

    Args:
        kv:          (…, head_dim) BF16/FP32 tensor.
        block_size:  Elements per quantisation block (default 128).
                     Automatically reduced if head_dim < block_size.

    Returns:
        (fp8_data, scales) where:
            fp8_data  – (…, head_dim)  torch.float8_e4m3fn
            scales    – (…, n_blocks)  scale_dtype
    """
    shape = kv.shape
    head_dim = shape[-1]

    if head_dim < block_size:
        block_size = head_dim
    else:
        while head_dim % block_size != 0 and block_size > 1:
            block_size //= 2

    flat = kv.reshape(-1, head_dim).contiguous()
    fp8_flat, scale_flat = act_quant(flat.bfloat16(), block_size, "none", scale_dtype)

    fp8_data = fp8_flat.reshape(shape)
    scale_shape = (*shape[:-1], head_dim // block_size)
    scales = scale_flat.reshape(scale_shape)
    return fp8_data, scales


def dequantize_kv(
    fp8_data: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Reconstruct BF16 KV from FP8 quantised representation.

    Args:
        fp8_data:  (…, head_dim)  torch.float8_e4m3fn
        scales:    (…, n_blocks)  per-block scales
        block_size: Block size used during quantisation.

    Returns:
        (…, head_dim) BF16 tensor.
    """
    head_dim = fp8_data.shape[-1]
    if head_dim < block_size:
        block_size = head_dim
    else:
        while head_dim % block_size != 0 and block_size > 1:
            block_size //= 2

    flat = fp8_data.reshape(-1, head_dim).contiguous()
    scale_flat = scales.reshape(-1, head_dim // block_size).contiguous()
    bf16 = weight_dequant(flat, scale_flat, block_size)
    return bf16.reshape(fp8_data.shape).to(dtype=out_dtype)


def maybe_quantize_kv(
    kv: torch.Tensor,
    enabled: bool,
    block_size: int = DEFAULT_BLOCK_SIZE,
    scale_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Conditionally quantise KV cache — identity when *enabled* is ``False``.

    Returns:
        (kv_data, scales):
            enabled=True  → (fp8_data, scales)
            enabled=False → (bf16_data, None)
    """
    if not enabled:
        return kv, None
    return quantize_kv(kv, block_size, scale_dtype)


def maybe_dequantize_kv(
    kv_data: torch.Tensor,
    scales: Optional[torch.Tensor],
    enabled: bool,
    block_size: int = DEFAULT_BLOCK_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Conditionally dequantise KV cache — identity when *enabled* is ``False``.

    Returns:
        BF16 tensor (always).
    """
    if not enabled or scales is None:
        return kv_data.to(dtype=out_dtype)
    return dequantize_kv(kv_data, scales, block_size, out_dtype)


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

from dataclasses import dataclass


@dataclass
class KVQuantConfig:
    """Configuration for FP8 KV cache quantisation."""

    enabled: bool = False
    block_size: int = DEFAULT_BLOCK_SIZE
    scale_dtype: str = "fp32"  # "fp32" | "fp8"

    @property
    def torch_scale_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32, "fp8": torch.float8_e8m0fnu}.get(
            self.scale_dtype, torch.float32
        )
