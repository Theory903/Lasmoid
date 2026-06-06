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
    if weight.dtype == torch.bfloat16 or weight.dtype == torch.float32:
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
        return _linear_dispatch(x, self.weight, self.bias)


# ══════════════════════════════════════════════════════════════════════
# ROTARY EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════
def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()

    if freqs_cis.ndim == 2:
        if xc.ndim == 3:
            if freqs_cis.size(0) != xc.size(1):
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.size(0)} positions for sequence length {xc.size(1)}"
                )
            freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
        elif xc.ndim == 4:
            if freqs_cis.size(0) != xc.size(1):
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.size(0)} positions for sequence length {xc.size(1)}"
                )
            freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
        else:
            raise ValueError(f"Unsupported rotary tensor rank: {xc.ndim}")
    elif freqs_cis.ndim == 3:
        if xc.ndim == 3:
            if freqs_cis.shape[-1] != xc.size(-1):
                raise ValueError(
                    f"Rotary dim mismatch: got {freqs_cis.shape[-1]} vs {xc.size(-1)}"
                )
        elif xc.ndim == 4:
            if freqs_cis.shape[-1] != xc.size(-1):
                raise ValueError(
                    f"Rotary dim mismatch: got {freqs_cis.shape[-1]} vs {xc.size(-1)}"
                )
            if freqs_cis.shape[1] == xc.size(1):
                freqs_cis = freqs_cis.unsqueeze(2)
            else:
                raise ValueError(
                    f"Rotary freq length mismatch: got {freqs_cis.shape[1]} positions for sequence length {xc.size(1)}"
                )
        else:
            raise ValueError(f"Unsupported rotary tensor rank: {xc.ndim}")
    else:
        raise ValueError(f"Unsupported rotary frequency rank: {freqs_cis.ndim}")

    xr = torch.view_as_real(xc * freqs_cis.to(torch.complex64)).flatten(-2)
    return xr.to(dtype)
