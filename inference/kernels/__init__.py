"""Kernel provider interface (DIP) — wraps low-level kernel functions."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from ..kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, weight_dequant
except ImportError:
    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, weight_dequant

__all__ = [
    "KernelProvider",
    "BF16KernelProvider",
    "FP8KernelProvider",
    "FP4KernelProvider",
    "NVFP4KernelProvider",
    "get_kernel_provider",
    "kernel_provider",
]


class KernelProvider(ABC):
    """Abstract interface for quantised matmul kernels, quantize, and dequantize."""

    @abstractmethod
    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def dequantize(
        self, x: torch.Tensor, scale: torch.Tensor, block_size: int = 128
    ) -> torch.Tensor: ...


class BF16KernelProvider(KernelProvider):
    """BF16 / FP32 — plain F.linear."""

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x.to(weight.dtype), weight, bias)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.tensor([])

    def dequantize(
        self, x: torch.Tensor, scale: torch.Tensor, block_size: int = 128
    ) -> torch.Tensor:
        return x


class FP8KernelProvider(KernelProvider):
    """FP8 — act_quant + fp8_gemm + weight_dequant."""

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xq, xs = self.quantize(x)
        s = scale if scale is not None else weight.scale
        out = fp8_gemm(xq, xs, weight, s, torch.float32)
        return out + bias if bias is not None else out

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return act_quant(x.contiguous().bfloat16(), 128, "none", torch.float32)

    def dequantize(
        self, x: torch.Tensor, scale: torch.Tensor, block_size: int = 128
    ) -> torch.Tensor:
        return weight_dequant(x, scale, block_size)


class FP4KernelProvider(KernelProvider):
    """FP4 — fp4_act_quant + fp4_gemm."""

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xq, xs = self.quantize(x)
        s = scale if scale is not None else weight.scale
        out = fp4_gemm(xq, xs, weight, s)
        return out + bias if bias is not None else out

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return fp4_act_quant(x.contiguous(), block_size=32)

    def dequantize(
        self, x: torch.Tensor, scale: torch.Tensor, block_size: int = 32
    ) -> torch.Tensor:
        return weight_dequant(x.float(), scale, block_size)


class NVFP4KernelProvider(KernelProvider):
    """NVFP4 — fp4_act_quant + fp4_gemm (E2M1 simulated format)."""

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xq, xs = self.quantize(x)
        s = scale if scale is not None else getattr(weight, "scale", None)
        out = fp4_gemm(xq, xs, weight, s)
        return out + bias if bias is not None else out

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return fp4_act_quant(x.contiguous(), block_size=32)

    def dequantize(
        self, x: torch.Tensor, scale: torch.Tensor, block_size: int = 32
    ) -> torch.Tensor:
        return weight_dequant(x.float(), scale, block_size)


_PROVIDER_MAP: dict[str, type[KernelProvider]] = {
    "bf16": BF16KernelProvider,
    "fp8": FP8KernelProvider,
    "fp4": FP4KernelProvider,
    "nvfp4": NVFP4KernelProvider,
}


def get_kernel_provider(dtype: str) -> KernelProvider:
    """Factory — returns the correct KernelProvider for *dtype*."""
    cls = _PROVIDER_MAP.get(dtype.lower())
    if cls is None:
        msg = f"Unknown dtype '{dtype}'. Choose from: {list(_PROVIDER_MAP)}"
        raise ValueError(msg)
    return cls()


kernel_provider: KernelProvider = BF16KernelProvider()
