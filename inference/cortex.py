"""
Lasmoid — cortex.py
======================================================================
DomainCortexRouter: brain-like sparse activation over scientific domains.

Biological motivation
----------------------
The neocortex is organised into *cortical columns* — local populations of
neurons specialised for a modality/function — and only a sparse subset fire for
any given stimulus.  We mirror this by partitioning the MoE's routed experts
into ``n_domains`` contiguous *domain columns* (e.g. mathematics, physics,
chemistry, biology/medical, astronomy, computer-science, data-analysis,
general).  A dedicated, semantically-meaningful router selects the top-k
relevant columns per token, so only the experts inside active columns can fire.

This is a hierarchical (two-level) sparse router that sits *on top of* the
existing expert ``Gate``:

    token  ──►  DomainCortexRouter  ──►  active domain columns (top-k)
                                         │
                                         ▼  (expert eligibility mask)
                            existing expert top-k selection within columns

Reused techniques
-----------------
* Doubly-stochastic (Sinkhorn) cross-domain affinity — adapted from the
  manifold-constrained hyper-connections repo — lets related columns
  (e.g. physics ↔ mathematics) softly co-activate in a norm-preserving way.
* A domain *steering* vector allows explicit, external control of which
  scientific domain is engaged (e.g. a tool/agent forcing "chemistry").

The whole module is config-gated by ``use_domain_cortex`` (default False) and is
a no-op when disabled, preserving existing behaviour exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

try:
    from ._common import Linear  # noqa: F401 — used by subclasses
except ImportError:
    from _common import Linear  # noqa: F401


def sinkhorn_log(logits: torch.Tensor, num_iters: int = 8, tau: float = 0.05) -> torch.Tensor:
    """Log-domain Sinkhorn normalisation → doubly-stochastic matrix.

    Adapted from mHC (manifold-constrained hyper-connections).  Produces a
    norm-preserving square mixing matrix from arbitrary logits.
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=Z.dtype)
    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)
    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class DomainCortexRouter(nn.Module):
    """Sparse cortical-column router over scientific domains.

    Args:
        dim: model hidden dim (router input dim).
        n_routed_experts: total routed experts (must be divisible by n_domains).
        n_domains: number of domain columns.
        domain_topk: number of columns activated per token (sparse activation).
        eps: numerical epsilon.
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_domains: int = 8,
        domain_topk: int = 2,
        route_noise: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert n_routed_experts % n_domains == 0, (
            f"n_routed_experts ({n_routed_experts}) must be divisible by "
            f"n_domains ({n_domains}) for contiguous cortical columns"
        )
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_domains = n_domains
        self.domain_topk = min(domain_topk, n_domains)
        self.experts_per_domain = n_routed_experts // n_domains
        self.route_noise = route_noise
        self.eps = eps

        # Dedicated, semantically-meaningful domain router (separate from expert gate).
        self.domain_weight = nn.Parameter(torch.empty(n_domains, dim))
        self.domain_scale = nn.Parameter(torch.ones(dim))
        nn.init.normal_(self.domain_weight, 0.0, 0.02)

        # Noisy top-k gating (Shazeer 2017): a learned, per-token, per-column noise
        # scale lets columns *explore* and self-specialise during training instead
        # of freezing on whichever column won at init (emergent, MoE-like routing).
        if route_noise > 0:
            self.noise_weight = nn.Parameter(torch.zeros(n_domains, dim))
        else:
            self.noise_weight = None

        # Cross-domain affinity logits (→ doubly-stochastic mix via Sinkhorn).
        # Initialised near identity so columns start independent.
        init_mix = torch.full((n_domains, n_domains), -8.0)
        init_mix.fill_diagonal_(0.0)
        self.mix_logits = nn.Parameter(init_mix)

        # Telemetry (populated each forward).
        self.last_domain_probs: Optional[torch.Tensor] = None
        self.last_domain_indices: Optional[torch.Tensor] = None
        self.last_aux_loss = torch.tensor(0.0)

    def domain_affinity(self) -> torch.Tensor:
        """Doubly-stochastic [n_domains, n_domains] cross-domain mixing matrix."""
        return sinkhorn_log(self.mix_logits.float())

    def forward(
        self,
        router_input: torch.Tensor,
        domain_steer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select active domain columns and build an expert eligibility mask.

        Args:
            router_input: [N_tokens, dim] normalised router input.
            domain_steer: optional [n_domains] additive bias on domain logits for
                explicit external domain control (e.g. agent forcing "chemistry").

        Returns:
            expert_mask:    [N_tokens, n_routed_experts] bool — True = eligible.
            domain_probs:   [N_tokens, n_domains] soft (affinity-smoothed) probs.
            domain_indices: [N_tokens, domain_topk] selected column indices.
            aux_loss:       scalar load-balance loss over domain usage.
        """
        N = router_input.shape[0]  # noqa: F841 — retained for documentation
        x = router_input.float() * self.domain_scale.float()
        logits = F.linear(x, self.domain_weight.float())  # [N, n_domains]

        if domain_steer is not None:
            logits = logits + domain_steer.to(device=logits.device, dtype=logits.dtype).view(1, -1)

        raw_probs = logits.softmax(dim=-1)  # [N, n_domains] (clean, for load balance)

        # Noisy top-k gating: inject learned exploration noise during training so
        # columns self-specialise (emergent routing). Deterministic at inference.
        if self.training and self.noise_weight is not None and self.route_noise > 0:
            noise_std = F.softplus(F.linear(x, self.noise_weight.float()))  # [N, D]
            select_logits = logits + torch.randn_like(logits) * noise_std * self.route_noise
            select_probs = select_logits.softmax(dim=-1)  # different from raw_probs due to noise
        else:
            # Reuse raw_probs — select_logits == logits so softmax is identical.
            # (Req 18.3: remove redundant softmax pass over same tokens)
            select_probs = raw_probs

        # Soft cross-domain smoothing (related columns co-activate, norm-preserving).
        affinity = self.domain_affinity().to(select_probs.dtype)  # [D, D]
        domain_probs = select_probs @ affinity  # [N, n_domains]

        # Sparse activation: top-k columns per token.
        k = self.domain_topk
        topk_idx = domain_probs.topk(k, dim=-1).indices  # [N, k]
        domain_mask = torch.zeros_like(domain_probs, dtype=torch.bool)
        domain_mask.scatter_(1, topk_idx, True)  # [N, n_domains]

        # Expand column mask to per-expert eligibility mask.
        expert_mask = domain_mask.repeat_interleave(self.experts_per_domain, dim=1)
        # [N, n_domains * experts_per_domain] == [N, n_routed_experts]

        # Load-balance aux: encourage uniform column usage (Switch-style fi*Pi).
        fi = domain_mask.float().mean(dim=0)  # fraction of tokens selecting each column
        Pi = raw_probs.mean(dim=0)            # mean router prob per column
        aux_loss = self.n_domains * (fi * Pi).sum()

        self.last_domain_probs = domain_probs.detach()
        self.last_domain_indices = topk_idx.detach()
        self.last_aux_loss = aux_loss.detach()

        return expert_mask, domain_probs, topk_idx, aux_loss


__all__ = ["DomainCortexRouter", "sinkhorn_log"]
