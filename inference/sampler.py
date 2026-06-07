"""
Lasmod — sampler.py
============================
2026 SOTA sampling pipeline primitives:
  logits -> NaN/Inf guard -> final softcap -> temperature -> DRY repetition penalty
  -> XTC creative exclusion -> Min-P/Top-P/Top-K -> sample

Determinism: When a torch.Generator is supplied (seeded externally), all stochastic
operations use that generator, guaranteeing identical outputs for identical inputs+seed.
"""

from typing import List, Optional

import torch

try:
    from ._harness import nan_inf_guard, NonFiniteError
except ImportError:
    try:
        from _harness import nan_inf_guard, NonFiniteError
    except ImportError:
        # Minimal fallback if _harness is unavailable (e.g. standalone use)
        class NonFiniteError(Exception):
            """Raised when a tensor contains NaN or Inf values."""

            def __init__(self, module_name: str, tensor_name: str, **kwargs):
                self.module_name = module_name
                self.tensor_name = tensor_name
                super().__init__(
                    f"NonFiniteError in {module_name}.{tensor_name}"
                )

        def nan_inf_guard(
            tensor: torch.Tensor,
            module_name: str,
            tensor_name: str,
            *,
            chunk_index=None,
        ) -> None:
            nan_count = int(torch.isnan(tensor).sum().item())
            inf_count = int(torch.isinf(tensor).sum().item())
            if nan_count > 0 or inf_count > 0:
                raise NonFiniteError(module_name, tensor_name)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature (zero = argmax mode)."""
    if temperature == 0.0:
        return logits
    return logits / max(temperature, 1e-8)


def apply_final_softcap(
    logits: torch.Tensor, cap: Optional[float] = None
) -> torch.Tensor:
    """Apply tanh-based soft cap to bound logits within [-cap, +cap].

    Uses the standard soft-capping formula: cap * tanh(logits / cap).
    This smoothly limits logit magnitudes without hard clipping.
    """
    if cap is None or cap <= 0.0:
        return logits
    return cap * torch.tanh(logits / cap)


def apply_dry(
    logits: torch.Tensor,
    generated: List[int],
    dry_multiplier: float = 0.8,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
) -> torch.Tensor:
    """DRY repetition control via history n-gram tracking."""
    if dry_multiplier == 0.0 or len(generated) < dry_allowed_length:
        return logits

    logits = logits.clone()
    last_token = generated[-1]
    match_indices = [i for i, t in enumerate(generated[:-1]) if t == last_token]

    for idx in match_indices:
        match_len = 1
        while (
            match_len <= dry_allowed_length
            and idx - match_len >= 0
            and len(generated) - 1 - match_len >= 0
            and generated[idx - match_len] == generated[-1 - match_len]
        ):
            match_len += 1

        if match_len < dry_allowed_length:
            continue

        if idx + 1 < len(generated):
            penalised_token = generated[idx + 1]
            penalty = dry_multiplier * (dry_base ** (match_len - dry_allowed_length))
            logits[0, penalised_token] -= penalty

    return logits


def apply_xtc(
    logits: torch.Tensor,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """XTC (eXclude Top Choices) for open-ended creative tasks."""
    if xtc_probability == 0.0:
        return logits
    if torch.rand(1, generator=generator).item() > xtc_probability:
        return logits

    logits = logits.clone()
    probs = torch.softmax(logits, dim=-1)

    mask = probs > xtc_threshold
    if mask.sum() < probs.numel():
        logits[mask] = float("-inf")
    return logits


def apply_min_p(logits: torch.Tensor, min_p: float = 0.05) -> torch.Tensor:
    """Min-P sampling — scale threshold by leading choice probability."""
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    p_max = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * p_max
    logits = logits.clone()
    logits[probs < threshold] = float("-inf")
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    """Top-P (nucleus) sampling."""
    if top_p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = (cumulative - sorted_probs) > top_p
    remove_original = torch.zeros_like(logits, dtype=torch.bool)
    remove_original.scatter_(-1, sorted_idx, remove_mask)
    logits = logits.clone()
    logits[remove_original] = float("-inf")
    return logits


def apply_top_k(logits: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Hard Top-K filter."""
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = values[..., -1, None]
    logits = logits.clone()
    logits[logits < threshold] = float("-inf")
    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Argmax (temp=0) or multinomial sampling from logits."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def full_sample(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
    final_logit_softcap: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Full 2026 SOTA sampling pipeline.

    Pipeline order:
      1. NaN/Inf guard (raises NonFiniteError if any non-finite logit)
      2. Final logit soft-cap (bounds logits before any further processing)
      3. Temperature scaling
      4. DRY repetition penalty
      5. XTC creative exclusion
      6. Min-P or Top-P filtering
      7. Top-K filtering
      8. Multinomial / argmax sampling

    Args:
        logits: Raw logit tensor from the model, shape (B, V).
        generated_ids: Previously generated token ids (for DRY penalty).
        temperature: Sampling temperature. 0.0 = argmax.
        top_k: Hard top-k filter (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        min_p: Min-P threshold (0.0 = disabled). Takes priority over top_p.
        xtc_probability: Per-step probability of applying XTC (0.0 = disabled).
        xtc_threshold: XTC probability threshold for masking top choices.
        dry_multiplier: DRY repetition penalty multiplier (0.0 = disabled).
        dry_base: DRY exponential base.
        dry_allowed_length: DRY minimum match length.
        final_logit_softcap: If set, apply tanh soft-cap bounding logits to
            [-cap, +cap] before all other processing.
        generator: Optional torch.Generator for deterministic sampling.
            When provided, all stochastic ops (XTC coin flip, multinomial)
            use this generator, ensuring reproducibility under a fixed seed.

    Returns:
        Sampled token id tensor, shape (B, 1).

    Raises:
        NonFiniteError: If the input logit tensor contains NaN or Inf values.
    """
    # Step 1: Guard against non-finite logits
    nan_inf_guard(logits, "Sampler", "input_logits")

    # Step 2: Final logit soft-cap (bounds logits before sampling)
    logits = apply_final_softcap(logits, final_logit_softcap)

    # Step 3: Temperature scaling
    logits = apply_temperature(logits, temperature)

    # Step 4: DRY repetition penalty
    logits = apply_dry(
        logits, generated_ids, dry_multiplier, dry_base, dry_allowed_length
    )

    # Step 5: XTC creative exclusion
    logits = apply_xtc(logits, xtc_probability, xtc_threshold, generator=generator)

    # Step 6: Min-P or Top-P filtering
    if min_p > 0.0:
        logits = apply_min_p(logits, min_p)
    else:
        logits = apply_top_p(logits, top_p)

    # Step 7: Top-K filtering
    logits = apply_top_k(logits, top_k)

    # Step 8: Sample
    return sample_token(logits, temperature, generator=generator)
