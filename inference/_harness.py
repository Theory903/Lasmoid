"""
Lasmoid — Shared CPU Test Harness
===================================
Reusable fixtures and assertion helpers for the hardening test suite.

All fixtures run on **CPU in float32** regardless of config dtype, using
``config_100m.json`` (Tiny_Config). This isolates algorithmic error from
precision-format artifacts.

Run on macOS / Python 3.14:

    DYLD_LIBRARY_PATH=$(brew --prefix expat)/lib python3 -m pytest inference/ -v

Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 21.1, 21.2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import numpy as np

if TYPE_CHECKING:
    from config import ModelArgs

# ---------------------------------------------------------------------------
# Path setup — ensure `inference/` and project root are importable
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

for _p in (_THIS_DIR, _PROJECT_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ---------------------------------------------------------------------------
# Constants (single source of truth)
# ---------------------------------------------------------------------------

TOLERANCES: Dict[str, float] = {
    "pure_math": 1e-5,   # RMSNorm, RoPE, ZOH decay, MoE renorm
    "multi_step": 1e-4,  # SSM scan, KV-cache vs full, CIF AR vs parallel
    "sinkhorn": 1e-3,    # doubly-stochastic row/col sums
    "refactor": 1e-5,    # output-equality guard for redundant-work removal
}

SEED: int = 1234

# ---------------------------------------------------------------------------
# Structured error for NaN/Inf detection
# ---------------------------------------------------------------------------


class NonFiniteError(Exception):
    """Raised when a tensor contains NaN or Inf values.

    Attributes:
        module_name: The module that produced the non-finite tensor.
        tensor_name: The name/label of the offending tensor.
        nan_count: Number of NaN elements.
        inf_count: Number of Inf elements.
        chunk_index: Optional chunk index (for SSM scan reporting).
    """

    def __init__(
        self,
        module_name: str,
        tensor_name: str,
        *,
        nan_count: int = 0,
        inf_count: int = 0,
        chunk_index: Optional[int] = None,
    ):
        self.module_name = module_name
        self.tensor_name = tensor_name
        self.nan_count = nan_count
        self.inf_count = inf_count
        self.chunk_index = chunk_index
        parts = [
            f"NonFiniteError in {module_name}.{tensor_name}: "
            f"NaN={nan_count}, Inf={inf_count}"
        ]
        if chunk_index is not None:
            parts.append(f" (chunk_index={chunk_index})")
        super().__init__("".join(parts))


# ---------------------------------------------------------------------------
# Reusable assertion helpers
# ---------------------------------------------------------------------------


def nan_inf_guard(
    tensor: torch.Tensor,
    module_name: str,
    tensor_name: str,
    *,
    chunk_index: Optional[int] = None,
) -> None:
    """Assert that *tensor* contains no NaN or Inf values.

    Raises ``NonFiniteError`` with structured metadata on failure.
    """
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    if nan_count > 0 or inf_count > 0:
        raise NonFiniteError(
            module_name,
            tensor_name,
            nan_count=nan_count,
            inf_count=inf_count,
            chunk_index=chunk_index,
        )


def assert_shape(
    tensor: torch.Tensor,
    expected: Tuple[int, ...],
    *,
    label: str = "tensor",
) -> None:
    """Assert that *tensor* has exactly *expected* shape."""
    if tensor.shape != torch.Size(expected):
        raise ValueError(
            f"Shape mismatch for {label}: "
            f"got {tuple(tensor.shape)}, expected {expected}"
        )


def grad_flow(
    module: nn.Module,
    inputs: Any,
    *,
    forward_fn: Optional[Callable[..., torch.Tensor]] = None,
) -> None:
    """Run a forward+backward pass and assert finite nonzero grads.

    Parameters:
        module: The module under test (must have parameters).
        inputs: A tensor or tuple of tensors passed to forward_fn or module().
        forward_fn: Optional callable; if None, calls module(*inputs) and
                    sums the output for backward.
    """
    module.train()
    module.zero_grad()

    if forward_fn is not None:
        out = forward_fn(module, inputs)
    else:
        if isinstance(inputs, (tuple, list)):
            out = module(*inputs)
        else:
            out = module(inputs)

    # Handle tuple/list outputs (take first tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]

    out.sum().backward()

    for name, p in module.named_parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise AssertionError(
                    f"grad_flow: non-finite gradient for parameter '{name}'"
                )
            if p.grad.abs().sum() == 0:
                raise AssertionError(
                    f"grad_flow: zero gradient for parameter '{name}'"
                )


def determinism(
    fn: Callable[..., torch.Tensor],
    *args: Any,
    seed: int = SEED,
    label: str = "output",
) -> None:
    """Run *fn* twice under the same seed and assert bitwise equality.

    Parameters:
        fn: Callable that returns a tensor. May accept positional *args*.
        *args: Positional arguments forwarded to *fn*.
        seed: RNG seed applied before each invocation.
        label: Descriptive label for error messages.
    """
    torch.manual_seed(seed)
    out1 = fn(*args)

    torch.manual_seed(seed)
    out2 = fn(*args)

    if not torch.equal(out1, out2):
        max_diff = (out1 - out2).abs().max().item()
        raise AssertionError(
            f"determinism: {label} differs across two seeded runs "
            f"(max |Δ| = {max_diff})"
        )


def expert_coverage(
    gate_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    input_tensor: torch.Tensor,
    n_experts: int,
    *,
    label: str = "MoE",
) -> None:
    """Assert that every expert is selected at least once.

    Parameters:
        gate_fn: Callable(input_flat) -> (weights, indices) where indices has
                 shape (tokens, top_k) with expert indices.
        input_tensor: A (B, S, D) tensor — flattened to (B*S, D) internally.
        n_experts: Total number of routed experts.
    """
    flat = input_tensor.reshape(-1, input_tensor.shape[-1])
    _, indices = gate_fn(flat)
    activated = set(indices.unique().tolist())
    expected = set(range(n_experts))
    missing = expected - activated
    if missing:
        raise AssertionError(
            f"expert_coverage ({label}): experts {sorted(missing)} "
            f"never activated over {flat.shape[0]} tokens"
        )


def tool_dispatch(
    registry: Any,
    *,
    label: str = "ToolRegistry",
) -> None:
    """Assert every registered tool dispatches without error.

    Parameters:
        registry: A ToolRegistry instance with .names() and .dispatch(name, args).
    """
    try:
        from .tools import ToolResult
    except ImportError:
        from tools import ToolResult

    for name in registry.names():
        # Build minimal valid arguments from the schema
        spec = registry._specs.get(name)
        if spec is None:
            continue
        args: Dict[str, Any] = {}
        for pname, pschema in (spec.parameters or {}).items():
            ptype = pschema.get("type", "string")
            if ptype == "number":
                args[pname] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            elif ptype == "integer":
                args[pname] = 2
            elif ptype == "string":
                args[pname] = "linear"
            elif ptype == "array":
                args[pname] = [[1.0, 2.0], [3.0, 4.0]]
            else:
                args[pname] = [[1.0, 2.0], [3.0, 4.0]]

        result = registry.dispatch(name, args)
        if not isinstance(result, ToolResult):
            raise AssertionError(
                f"tool_dispatch ({label}): {name} returned "
                f"{type(result).__name__}, expected ToolResult"
            )
        if result.error is not None:
            raise AssertionError(
                f"tool_dispatch ({label}): {name} returned error: {result.error}"
            )


def equiv_guard(
    out_old: torch.Tensor,
    out_new: torch.Tensor,
    *,
    atol: float = TOLERANCES["refactor"],
    label: str = "equiv_guard",
) -> None:
    """Assert two outputs are equal within *atol* (refactor safety net).

    Reports ``max |Δ|`` on failure.
    """
    max_diff = (out_old - out_new).abs().max().item()
    if max_diff > atol:
        raise AssertionError(
            f"{label}: max |Δ| = {max_diff:.2e} exceeds tolerance {atol:.2e}"
        )


def ref_parity(
    lasmoid_out: torch.Tensor,
    reference_out: torch.Tensor,
    *,
    atol: float = TOLERANCES["pure_math"],
    label: str = "ref_parity",
) -> float:
    """Compare Lasmoid output against a reference implementation.

    Returns the ``max |Δ|`` and raises AssertionError if it exceeds *atol*.
    """
    max_diff = (lasmoid_out - reference_out).abs().max().item()
    if max_diff > atol:
        raise AssertionError(
            f"{label}: max |Δ| = {max_diff:.2e} exceeds tolerance {atol:.2e}"
        )
    return max_diff


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIG_PATH = _PROJECT_ROOT / "config_100m.json"


def tiny_config() -> "ModelArgs":
    """Load ``config_100m.json`` as a ``ModelArgs`` instance on CPU/float32.

    Forces dtype to float32 regardless of what the JSON specifies, ensuring
    all test computations are precision-stable on CPU.
    """
    from config import ModelArgs

    with open(_CONFIG_PATH) as f:
        raw = json.load(f)

    # Override dtype to float32 for CPU testing
    raw["dtype"] = "bf16"  # keep field valid; we force tensors to float32 below

    args = ModelArgs(**{k: v for k, v in raw.items() if hasattr(ModelArgs, k)})
    return args


def tiny_model() -> "nn.Module":
    """Build the full ``Lasmoid`` model from ``Tiny_Config`` on CPU in float32.

    The model is returned in eval mode with all parameters cast to float32.
    """
    from lasmoid import Lasmoid

    args = tiny_config()
    torch.manual_seed(SEED)
    model = Lasmoid(args)
    model = model.float().cpu().eval()
    return model


def random_hidden(B: int, S: int) -> torch.Tensor:
    """Return a ``(B, S, dim)`` normal tensor in float32 on CPU.

    Uses ``Tiny_Config.dim`` (384) as the feature dimension.
    """
    args = tiny_config()
    torch.manual_seed(SEED)
    return torch.randn(B, S, args.dim, dtype=torch.float32)


def mixed_multimodal_batch() -> Dict[str, Any]:
    """Return a synthetic mixed text + image + audio batch for fusion tests.

    Contents:
        - text_ids: (B=2, S=16) integer token ids
        - image: (B=2, C=3, H=32, W=32) float32 pixel tensor (small patch grid)
        - audio_waveform: (B=2, T=4000) float32 waveform (short clip)
    """
    args = tiny_config()
    torch.manual_seed(SEED)
    return {
        "text_ids": torch.randint(0, args.vocab_size, (2, 16)),
        "image": torch.randn(2, 3, 32, 32, dtype=torch.float32),
        "audio_waveform": torch.randn(2, 4000, dtype=torch.float32),
    }


def reward_group() -> Dict[str, torch.Tensor]:
    """Return synthetic reward vectors for GRPO advantage tests.

    Contents:
        - rewards: (G=4,) float32 reward scalars per group member
        - advantages: (G=4,) float32 expected normalized advantages
          (computed as (r - mean) / (std + eps))
    """
    torch.manual_seed(SEED)
    rewards = torch.randn(4, dtype=torch.float32)
    eps = 1e-8
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantages = (rewards - mean) / (std + eps)
    return {
        "rewards": rewards,
        "advantages": advantages,
    }


def eda_dataset() -> Dict[str, Any]:
    """Return small numeric arrays with known closed-form answers for tool parity.

    Contents:
        - data_2d: (10, 3) numpy array for describe/correlate/reduce_dim/cluster
        - x_1d / y_1d: (10,) numpy arrays with a known linear relationship
          y = 2*x + 1 + small noise (for fit_model parity)
        - sample_a / sample_b: (20,) numpy arrays from different distributions
          (for hypothesis_test)
        - expected_slope: ~2.0 (ground truth for linear fit)
        - expected_intercept: ~1.0 (ground truth for linear fit)
    """
    rng = np.random.default_rng(SEED)
    data_2d = rng.standard_normal((10, 3))
    x_1d = np.linspace(0, 5, 10)
    noise = rng.standard_normal(10) * 0.01
    y_1d = 2.0 * x_1d + 1.0 + noise
    sample_a = rng.normal(loc=0.0, scale=1.0, size=20)
    sample_b = rng.normal(loc=2.0, scale=1.0, size=20)
    return {
        "data_2d": data_2d,
        "x_1d": x_1d,
        "y_1d": y_1d,
        "sample_a": sample_a,
        "sample_b": sample_b,
        "expected_slope": 2.0,
        "expected_intercept": 1.0,
    }
