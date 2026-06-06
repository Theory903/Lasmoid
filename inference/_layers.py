"""
Lasmoid — _layers.py
=====================================================================
Einsum-based parameter layers (Gemma-4 style), QKNorm, and future
base layer primitives.

Key differences from nn.Linear / _common.Linear:
- Weight stored as (in_features, out_features) — natural for einsum
- Forward uses torch.einsum for explicit dimension routing
- More TorchDynamo/XLA-friendly
- Supports multi-dimensional weight tensors for future use
"""

from typing import Optional

import torch
import torch.nn as nn


class EinsumLinear(nn.Module):
    """
    Einsum-parameterized linear layer (Gemma-4 style).

    Stores weight as (in_features, out_features) for natural einsum
    dimension routing:  einsum('...d,do->...o', x, weight)

    Unlike nn.Linear (weight shape = out_features, in_features),
    this layout avoids implicit transposes and enables:
      - Clean dimension routing for fused/custom equations
      - Future multi-dimensional weight tensors
      - Better TorchDynamo / XLA compilation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        w_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_scale = w_scale

        w_dtype = dtype or torch.bfloat16
        # Weight stored as (in_features, out_features) for natural einsum routing
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, dtype=w_dtype)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=w_dtype))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, 0.0, 0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: einsum('...d,do->...o', x, weight) + bias

        Args:
            x: Input tensor, shape (..., in_features)

        Returns:
            Output tensor, shape (..., out_features)
        """
        x = x.to(self.weight.dtype)
        out = torch.einsum("...d,do->...o", x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        s = f"in={self.in_features}, out={self.out_features}"
        if self.bias is not None:
            s += ", bias=True"
        if self.w_scale != 1.0:
            s += f", w_scale={self.w_scale}"
        return s


class QKNorm(nn.Module):
    """
    QKNorm: RMSNorm + learnable scale parameter (Gemma-4 style).

    Applies RMSNorm to input, then multiplies by a learnable parameter `scale`.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        try:
            from ._common import RMSNorm
        except ImportError:
            from _common import RMSNorm
        self.norm = RMSNorm(dim, eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * self.scale
