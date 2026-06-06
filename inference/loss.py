"""
Loss Functions — compute_loss / compute_grpo_loss
==================================================
Extracted from monolithic model.py during Phase 0 SOLiD refactoring.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    routing_maps: List[torch.Tensor],
    vq_losses: List[torch.Tensor],
    adjacencies: List[torch.Tensor],
    event_probs: Optional[List[torch.Tensor]] = None,
    loss_mask: Optional[torch.Tensor] = None,
    moe_aux_loss: Optional[torch.Tensor] = None,
    token_concept_loss: Optional[torch.Tensor] = None,
    token_concept_coeff: float = 0.05,
    graph_sparsity: float = 0.01,
    cif_target_ratio: float = 0.25,
    cif_entropy_weight: float = 0.01,
    cif_ratio_weight: float = 1.0,
) -> torch.Tensor:
    # ── Main autoregressive CE loss (masked SFT) ────────────────────
    if loss_mask is not None:
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
        )
        ce_loss = (ce_loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)
    else:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    moe_loss = (
        moe_aux_loss
        if moe_aux_loss is not None
        else torch.tensor(0.0, device=logits.device)
    )
    return ce_loss + moe_loss


def compute_grpo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Group Relative Policy Optimization (GRPO) clipped surrogate loss.

    Args:
        logits: policy logits of shape [B, S, V] (where B = group_size * batch_size)
        targets: target token ids of shape [B, S]
        advantages: advantages (normalized rewards) of shape [B]
        old_logprobs: log probabilities under old policy of shape [B, S]
        loss_mask: mask indicating which tokens have loss calculated (e.g. response tokens) of shape [B, S]
        clip_eps: PPO clipping range
        kl_coeff: KL penalty weight

    Returns:
        total_loss: policy loss + KL penalty
        policy_loss: policy surrogate loss
        kl_loss: Kullback-Leibler divergence loss
    """
    logprobs = F.log_softmax(logits, dim=-1)
    target_logprobs = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, S]

    # Calculate token-level ratio r_t
    ratio = torch.exp(target_logprobs - old_logprobs)  # [B, S]

    # Expand advantages from [B] to [B, S]
    adv = advantages.unsqueeze(-1).expand_as(ratio)

    # Clipped policy objective
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_objective = torch.min(surr1, surr2)  # [B, S]

    # KL penalty (approximate KL divergence: exp(ref_logp - logp) - (ref_logp - logp) - 1)
    # We use old_logprobs as the reference model starting point
    kl = (
        torch.exp(old_logprobs - target_logprobs)
        - (old_logprobs - target_logprobs)
        - 1.0
    )

    if loss_mask is not None:
        policy_loss = -(policy_objective * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        kl_loss = (kl * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    else:
        policy_loss = -policy_objective.mean()
        kl_loss = kl.mean()

    total_loss = policy_loss + kl_coeff * kl_loss
    return total_loss, policy_loss, kl_loss
