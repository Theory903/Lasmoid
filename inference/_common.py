"""
Lasmoid — _common.py
=====================================================================
Shared base classes extracted from model.py: Linear, RMSNorm, apply_rotary_emb,
set_dtype, and floating-point globals.

Zero dependencies on model.py or any SOLiD module — breaks the circular
import chain between model.py ↔ extracted modules.
"""

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .kernel import (
        act_quant,
        fp4_act_quant,
        fp4_gemm,
        fp8_gemm,
        weight_dequant,
    )
except ImportError:
    from kernel import (
        act_quant,
        fp4_act_quant,
        fp4_gemm,
        fp8_gemm,
        weight_dequant,
    )


# ══════════════════════════════════════════════════════════════════════
# GLOBALS
# ══════════════════════════════════════════════════════════════════════
default_dtype = torch.bfloat16
scale_fmt: Optional[str] = None
scale_dtype: torch.dtype = torch.float32
block_size: int = 128
fp4_block_size: int = 32

# Einsum parameterization global flag (Gemma-4)
_use_einsum: bool = False


def set_use_einsum(val: bool) -> None:
    """Toggle einsum parameterization for Linear layers globally."""
    global _use_einsum
    _use_einsum = val


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# ══════════════════════════════════════════════════════════════════════
# NORMALISATION
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


# ══════════════════════════════════════════════════════════════════════
# CUSTOM LINEAR LAYER (fp8-aware dispatch)
# ══════════════════════════════════════════════════════════════════════
def _linear_dispatch(
    x: torch.Tensor, weight: nn.Parameter, bias: Optional[nn.Parameter] = None
) -> torch.Tensor:
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        if getattr(weight, "use_fp4_weights", False):
            out = fp4_gemm(x, None, weight, getattr(weight, "scale", None)).to(x.dtype)
            if bias is not None:
                out = out + bias
            return out
        return F.linear(x.to(weight.dtype), weight, bias)
    if weight.dtype == torch.float8_e4m3fn:
        xq, xs = act_quant(
            x.contiguous().bfloat16(), block_size, scale_fmt, scale_dtype
        )
        out = fp8_gemm(xq, xs, weight, weight.scale, scale_dtype)
        if bias is not None:
            out = out + bias
        return out
    return F.linear(x, weight.float(), bias)


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w_dtype = dtype or default_dtype

        if w_dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=w_dtype)
            )
            so = (out_features + block_size - 1) // block_size
            si = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(so, si, dtype=torch.float32), requires_grad=False
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=w_dtype)
            )
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
        if (
            _use_einsum
            and self.weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and not getattr(self.weight, "use_fp4_weights", False)
        ):
            # Einsum path (Gemma-4): weight is (out, in), use '...d,od->...o'
            # which is mathematically equivalent to F.linear(x, weight).
            x = x.to(self.weight.dtype)
            out = torch.einsum("...d,od->...o", x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            return out
        return _linear_dispatch(x, self.weight, self.bias)


# ══════════════════════════════════════════════════════════════════════
# ROTARY EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════
def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply rotary positional embeddings via complex multiplication.

    Handles both 3D (B, S, D) and 4D (B, S, H, D) input tensors, and both
    2D (S, D//2) and 3D (B, S, D//2) frequency tensors. Uses conjugate for
    inverse (de-rotation during decoding).

    Args:
        x: Input tensor, shape (B, S, D) or (B, S, H, D) where D is even.
        freqs_cis: Complex frequency tensor, shape (S, D//2) or (B, S, D//2).
        inverse: If True, conjugate freqs_cis to reverse the rotation.

    Returns:
        Rotated tensor with the same shape and dtype as x.

    Raises:
        ValueError: On rank mismatch or dimension incompatibility.
    """
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()

    if freqs_cis.ndim == 2:
        # freqs_cis shape: (S, D//2)
        if xc.ndim == 3:
            # xc shape: (B, S, D//2)
            if freqs_cis.size(0) != xc.size(1):
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.size(0)} positions "
                    f"for sequence length {xc.size(1)}"
                )
            if freqs_cis.size(1) != xc.size(-1):
                raise ValueError(
                    f"Rotary freq dim mismatch: got {freqs_cis.size(1)} freq dims "
                    f"for head half-dim {xc.size(-1)}"
                )
            freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
        elif xc.ndim == 4:
            # xc shape: (B, S, H, D//2)
            if freqs_cis.size(0) != xc.size(1):
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.size(0)} positions "
                    f"for sequence length {xc.size(1)}"
                )
            if freqs_cis.size(1) != xc.size(-1):
                raise ValueError(
                    f"Rotary freq dim mismatch: got {freqs_cis.size(1)} freq dims "
                    f"for head half-dim {xc.size(-1)}"
                )
            freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
        else:
            raise ValueError(
                f"Unsupported rotary input rank: expected 3D (B,S,D) or 4D (B,S,H,D) "
                f"but got {xc.ndim + 1}D input (complex view is {xc.ndim}D)"
            )
    elif freqs_cis.ndim == 3:
        # freqs_cis shape: (B, S, D//2) or (1, S, D//2)
        if xc.ndim == 3:
            # xc shape: (B, S, D//2) — direct broadcast
            if freqs_cis.shape[-1] != xc.size(-1):
                raise ValueError(
                    f"Rotary freq dim mismatch: got {freqs_cis.shape[-1]} freq dims "
                    f"for head half-dim {xc.size(-1)}"
                )
            if freqs_cis.shape[1] != xc.size(1) and freqs_cis.shape[1] != 1:
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.shape[1]} positions "
                    f"for sequence length {xc.size(1)} (not broadcastable)"
                )
        elif xc.ndim == 4:
            # xc shape: (B, S, H, D//2) — need unsqueeze for head dim
            if freqs_cis.shape[-1] != xc.size(-1):
                raise ValueError(
                    f"Rotary freq dim mismatch: got {freqs_cis.shape[-1]} freq dims "
                    f"for head half-dim {xc.size(-1)}"
                )
            if freqs_cis.shape[1] == xc.size(1) or freqs_cis.shape[1] == 1:
                freqs_cis = freqs_cis.unsqueeze(2)
            else:
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.shape[1]} positions "
                    f"for sequence length {xc.size(1)} (not broadcastable)"
                )
        else:
            raise ValueError(
                f"Unsupported rotary input rank: expected 3D (B,S,D) or 4D (B,S,H,D) "
                f"but got {xc.ndim + 1}D input (complex view is {xc.ndim}D)"
            )
    else:
        raise ValueError(
            f"Unsupported rotary frequency rank: expected 2D (S, D//2) or "
            f"3D (B, S, D//2) but got {freqs_cis.ndim}D"
        )

    xr = torch.view_as_real(xc * freqs_cis.to(torch.complex64)).flatten(-2)
    return xr.to(dtype)
