"""
Loss Functions — compute_loss / compute_grpo_loss
==================================================
Training objective for Lasmoid. Combines the autoregressive cross-entropy with
the auxiliary signals the model actually produces, so every trainable component
receives gradient:

    L = CE(ignore_index, label_smoothed)
      + mtp_coeff * MTP_CE (multi-token prediction cross-entropy)
      + moe_aux_coeff * moe_aux  (router z-loss + load-balance + cortex balance)
      + vq_coeff * vq      (concept-memory residual-VQ codebook loss)
      + commit_coeff * commit (concept-memory commitment loss)
      + curiosity_coeff * curiosity (intrinsic forward-model prediction loss)
      + token_concept_coeff * token_concept
      + graph_sparsity·||A||₁
      + CIF boundary       (firing-rate target + crisp-boundary entropy)

The primary CE uses ``ignore_index`` to properly skip padded positions (Req 14.1).
The aggregate combines primary + MTP + VQ-commitment + MoE-balance losses by
configured weights (Req 14.2).
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _sum_losses(losses: Optional[List[torch.Tensor]], device, dtype=torch.float32) -> torch.Tensor:
    acc = torch.zeros((), device=device, dtype=dtype)
    if not losses:
        return acc
    for x in losses:
        if x is not None:
            acc = acc + x.float()
    return acc


def cif_boundary_loss(
    event_probs: List[torch.Tensor],
    target_ratio: float,
    ratio_weight: float,
    entropy_weight: float,
) -> torch.Tensor:
    """Continuous-Integrate-and-Fire boundary regulariser.

    Pushes the mean firing rate toward ``target_ratio`` (controls compression
    rate) and uses binary entropy to make firing decisions *crisp* (probabilities
    toward 0/1) for stable, well-placed semantic boundaries.
    """
    if not event_probs:
        return torch.zeros(())
    ratio_term = torch.zeros(())
    entropy_term = torch.zeros(())
    n = 0
    for ep in event_probs:
        if ep is None:
            continue
        p = ep.float().clamp(1e-6, 1.0 - 1e-6)
        rate = p.mean()
        ratio_term = ratio_term.to(p.device) + (rate - target_ratio) ** 2
        entropy_term = entropy_term.to(p.device) + (
            -(p * p.log() + (1 - p) * (1 - p).log())
        ).mean()
        n += 1
    if n == 0:
        return torch.zeros(())
    return (ratio_weight * ratio_term + entropy_weight * entropy_term) / n


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    routing_maps: List[torch.Tensor],
    vq_losses: List[torch.Tensor],
    adjacencies: List[torch.Tensor],
    event_probs: Optional[List[torch.Tensor]] = None,
    loss_mask: Optional[torch.Tensor] = None,
    moe_aux_loss: Optional[torch.Tensor] = None,
    moe_aux_coeff: float = 1.0,
    mtp_loss: Optional[torch.Tensor] = None,
    mtp_coeff: float = 0.3,
    token_concept_loss: Optional[torch.Tensor] = None,
    token_concept_coeff: float = 0.05,
    graph_sparsity: float = 0.01,
    cif_target_ratio: float = 0.25,
    cif_entropy_weight: float = 0.01,
    cif_ratio_weight: float = 1.0,
    commit_loss: Optional[torch.Tensor] = None,
    commit_coeff: float = 0.25,
    curiosity_loss: Optional[torch.Tensor] = None,
    curiosity_coeff: float = 0.01,
    vq_coeff: float = 1.0,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    device = logits.device
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)

    # ── Main autoregressive CE loss (label-smoothed, with ignore_index for padding) ────
    if loss_mask is not None:
        # When an explicit loss_mask is provided, use it for fine-grained masking
        # but also pass ignore_index so invalid target indices don't blow up.
        ce = F.cross_entropy(
            flat_logits, flat_targets, reduction="none",
            label_smoothing=label_smoothing, ignore_index=ignore_index,
        )
        m = loss_mask.reshape(-1).to(ce.dtype)
        ce_loss = (ce * m).sum() / (m.sum() + 1e-8)
    else:
        # Without a mask, ignore_index handles padded positions directly.
        ce_loss = F.cross_entropy(
            flat_logits, flat_targets,
            label_smoothing=label_smoothing, ignore_index=ignore_index,
        )

    total = ce_loss

    # ── MTP (multi-token prediction) loss ──
    if mtp_loss is not None:
        total = total + mtp_coeff * mtp_loss.float()

    # ── MoE auxiliary (router z-loss + load balance + cortex balance) ──
    if moe_aux_loss is not None:
        total = total + moe_aux_coeff * moe_aux_loss.float()

    # ── Concept-memory VQ + commitment ──
    total = total + vq_coeff * _sum_losses(vq_losses, device)
    if commit_loss is not None:
        total = total + commit_coeff * commit_loss.float()

    # ── Intrinsic curiosity (forward-model prediction) ──
    if curiosity_loss is not None:
        total = total + curiosity_coeff * curiosity_loss.float()

    # ── Token→concept assignment loss ──
    if token_concept_loss is not None:
        total = total + token_concept_coeff * token_concept_loss.float()

    # ── Concept-graph sparsity (L1 on adjacencies) ──
    if adjacencies and graph_sparsity > 0.0:
        a = _sum_losses([adj.abs().mean() for adj in adjacencies if adj is not None], device)
        total = total + graph_sparsity * a

    # ── CIF semantic-boundary regulariser ──
    if event_probs:
        total = total + cif_boundary_loss(
            event_probs, cif_target_ratio, cif_ratio_weight, cif_entropy_weight
        ).to(device)

    return total


def compute_grpo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group Relative Policy Optimization (GRPO) clipped surrogate loss.

    logits [B,S,V], targets [B,S], advantages [B], old_logprobs [B,S].
    Returns (total_loss, policy_loss, kl_loss).
    """
    logprobs = F.log_softmax(logits.float(), dim=-1)
    target_logprobs = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, S]

    ratio = torch.exp(target_logprobs - old_logprobs)  # token-level r_t
    adv = advantages.unsqueeze(-1).expand_as(ratio)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_objective = torch.min(surr1, surr2)

    # Schulman's low-variance unbiased KL estimator: exp(Δ) - Δ - 1 ≥ 0.
    delta = old_logprobs - target_logprobs
    kl = torch.exp(delta) - delta - 1.0

    if loss_mask is not None:
        m = loss_mask.to(policy_objective.dtype)
        denom = m.sum() + 1e-8
        policy_loss = -(policy_objective * m).sum() / denom
        kl_loss = (kl * m).sum() / denom
    else:
        policy_loss = -policy_objective.mean()
        kl_loss = kl.mean()

    total_loss = policy_loss + kl_coeff * kl_loss
    return total_loss, policy_loss, kl_loss
