"""
Lasmod — sampler.py
============================
2026 SOTA sampling pipeline primitives:
  logits -> temperature -> DRY repetition penalty
  -> XTC creative exclusion -> Min-P/Top-P/Top-K -> sample
"""

from typing import List

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature (zero = argmax mode)."""
    if temperature == 0.0:
        return logits
    return logits / max(temperature, 1e-8)


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
) -> torch.Tensor:
    """XTC (eXclude Top Choices) for open-ended creative tasks."""
    if xtc_probability == 0.0:
        return logits
    if torch.rand(1).item() > xtc_probability:
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


def sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Argmax (temp=0) or multinomial sampling from logits."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


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
) -> torch.Tensor:
    """Full 2026 SOTA sampling pipeline."""
    logits = apply_temperature(logits, temperature)
    logits = apply_dry(
        logits, generated_ids, dry_multiplier, dry_base, dry_allowed_length
    )
    logits = apply_xtc(logits, xtc_probability, xtc_threshold)

    if min_p > 0.0:
        logits = apply_min_p(logits, min_p)
    else:
        logits = apply_top_p(logits, top_p)

    logits = apply_top_k(logits, top_k)
    return sample_token(logits, temperature)
