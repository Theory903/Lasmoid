"""
Lasmoid — grpo_stability.py
===========================
Stability-aware reward shaping for Group Relative Policy Optimization (GRPO).
Combines reasoning quality, rambling penalties, and logit drift detection.
"""

import re
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from .reward import reasoning_self_evolution_reward
except ImportError:
    from reward import reasoning_self_evolution_reward


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
) -> float:
    """
    Composite GRPO Reward:
      Reward = Quality_Reward - Rambling_Penalty - Drift_Penalty
    """
    # 1. Quality Reward
    quality = reasoning_self_evolution_reward(response, gt_answer)

    # 2. Rambling Penalty
    rambling = compute_rambling_penalty(response, max_chars)

    # 3. Logit Drift Penalty
    drift = compute_drift_penalty(logits)

    # Return composite score
    return float(quality - rambling - drift)
