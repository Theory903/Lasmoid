"""
Lasmoid — grpo_stability.py
===========================
Stability-aware reward shaping for Group Relative Policy Optimization (GRPO).
Combines reasoning quality, rambling penalties, and logit drift detection.

Provides:
- ``compute_group_advantages``: normalize rewards by group mean/std with eps.
- ``compute_rambling_penalty``: penalise repetitive/malformed responses.
- ``compute_drift_penalty``: penalise logit entropy collapse/explosion/norm spike.
- ``stability_aware_reward``: composite reward (quality - rambling - drift).
- ``safe_reward``: wraps any reward function to catch malformed completions and
  return a defined penalty value rather than raising.
"""

import re
from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:
    from .reward import reasoning_self_evolution_reward
except ImportError:
    from reward import reasoning_self_evolution_reward

# ── Constants ────────────────────────────────────────────────────────────────
# Default penalty returned when a completion is malformed (None, non-string,
# causes an exception in the reward function, etc.).
MALFORMED_COMPLETION_PENALTY: float = -5.0

# Epsilon used when normalizing group advantages by the group standard deviation
# to prevent division by zero on constant-reward groups.
GROUP_ADVANTAGE_EPS: float = 1e-8


# ── Group Advantage Normalization ────────────────────────────────────────────


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = GROUP_ADVANTAGE_EPS,
) -> torch.Tensor:
    """Normalize rewards by group mean and standard deviation with eps.

    Implements the GRPO group-relative advantage:

        advantages = (rewards - group_mean) / (group_std + eps)

    Parameters
    ----------
    rewards : torch.Tensor
        Flat tensor of shape ``[B * G]`` containing per-completion reward scores.
    group_size : int
        Number of completions per prompt group (``G``).
    eps : float
        Small constant added to the group standard deviation to avoid division
        by zero when all completions in a group receive the same reward.

    Returns
    -------
    torch.Tensor
        Flat tensor of shape ``[B * G]`` of group-normalized advantages.
    """
    batch_size = rewards.numel() // group_size
    rewards_grouped = rewards.view(batch_size, group_size)
    mean = rewards_grouped.mean(dim=-1, keepdim=True)
    std = rewards_grouped.std(dim=-1, keepdim=True)
    advantages = (rewards_grouped - mean) / (std + eps)
    return advantages.view(-1)


# ── Safe Reward Wrapper ──────────────────────────────────────────────────────


def safe_reward(
    response: object,
    reward_fn: Callable[[str, Optional[str]], float],
    gt_answer: Optional[str] = None,
    malformed_penalty: float = MALFORMED_COMPLETION_PENALTY,
) -> float:
    """Evaluate a reward function, returning a defined penalty for malformed input.

    A completion is considered malformed if:
    - It is ``None`` or not a string.
    - It is an empty / whitespace-only string.
    - The reward function raises any exception when processing it.

    In all malformed cases the function returns ``malformed_penalty`` instead of
    propagating an unhandled exception (Req 16.5).

    Parameters
    ----------
    response : object
        The completion to score — may be None or non-string (treated as malformed).
    reward_fn : callable
        A reward function with signature ``(response: str, gt_answer: str | None) -> float``.
    gt_answer : str | None
        Optional ground-truth answer forwarded to ``reward_fn``.
    malformed_penalty : float
        Penalty value returned for malformed completions.

    Returns
    -------
    float
        The reward score, or ``malformed_penalty`` if the completion is malformed.
    """
    # Guard: None or non-string
    if not isinstance(response, str):
        return malformed_penalty
    # Guard: empty / whitespace-only
    if not response.strip():
        return malformed_penalty
    # Guard: exceptions during reward evaluation
    try:
        return float(reward_fn(response, gt_answer))
    except Exception:
        return malformed_penalty


def compute_rambling_penalty(response: str, max_chars: int = 4000) -> float:
    """
    Penalises repetitive, excessively long, or structurally malformed responses.
    """
    penalty = 0.0

    # 1. Structure check: unclosed or duplicated tags
    think_open = response.count("<think>")
    think_close = response.count("</think>")
    if think_open != think_close or think_open > 1:
        penalty += 1.5

    # 2. Length penalty for rambling
    if len(response) > max_chars:
        # Scale penalty with excess length
        penalty += 0.0005 * (len(response) - max_chars)

    # 3. Word-level repetition penalty (n-grams)
    words = re.findall(r"\b\w+\b", response.lower())
    if len(words) > 20:
        # Check 3-gram repetitions
        tri_grams = [
            (words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)
        ]
        unique_trigrams = set(tri_grams)
        if len(tri_grams) > 0:
            repetition_ratio = 1.0 - (len(unique_trigrams) / len(tri_grams))
            if repetition_ratio > 0.15:
                penalty += (repetition_ratio - 0.15) * 4.0

    return float(penalty)


def compute_drift_penalty(logits: torch.Tensor) -> float:
    """
    Computes a penalty if the model's output logits exhibit signs of drift or collapse:
      - Entropy collapse (too deterministic/rambling)
      - Entropy explosion (noise/confusion)
      - Logit norm spike (explosive magnitude)
    """
    # logits shape: [seq_len, vocab_size] or [batch, seq_len, vocab_size]
    if logits.ndim == 3:
        logits = logits.view(-1, logits.size(-1))

    probs = F.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    mean_entropy = entropy.mean().item()

    norm = logits.float().norm(dim=-1)
    mean_norm = norm.mean().item()

    penalty = 0.0

    # 1. Entropy Collapse Check: Model is too overconfident (repetitive/rambling loop)
    if mean_entropy < 1.2:
        penalty += (1.2 - mean_entropy) * 2.0

    # 2. Entropy Explosion Check: Output is near-uniform noise
    if mean_entropy > 7.5:
        penalty += (mean_entropy - 7.5) * 1.5

    # 3. Logit Norm Spike Check: Massive logit magnitude (instability risk)
    if mean_norm > 80.0:
        penalty += (mean_norm - 80.0) * 0.15

    return float(penalty)


def stability_aware_reward(
    response: str,
    logits: torch.Tensor,
    gt_answer: Optional[str] = None,
    max_chars: int = 4000,
    malformed_penalty: float = MALFORMED_COMPLETION_PENALTY,
) -> float:
    """
    Composite GRPO Reward:
      Reward = Quality_Reward - Rambling_Penalty - Drift_Penalty

    If the response is malformed (None, non-string, empty, or causes an
    exception during quality scoring), returns ``malformed_penalty`` without
    raising (Req 16.5).

    Note: this function computes a *per-completion reward*.  Group advantage
    normalization and the clipped-ratio/KL objective are applied externally via
    ``compute_group_advantages`` and ``compute_grpo_loss`` (inference/loss.py).
    """
    # Guard: malformed completion → defined penalty, no exception
    if not isinstance(response, str) or not response.strip():
        return malformed_penalty

    try:
        # 1. Quality Reward (via safe_reward to catch internal errors)
        quality = safe_reward(response, reasoning_self_evolution_reward, gt_answer, malformed_penalty)
        if quality == malformed_penalty:
            return malformed_penalty

        # 2. Rambling Penalty
        rambling = compute_rambling_penalty(response, max_chars)

        # 3. Logit Drift Penalty
        drift = compute_drift_penalty(logits)

        # Return composite score
        return float(quality - rambling - drift)
    except Exception:
        return malformed_penalty
