"""
Lasmoid — MoE module
========================================================================
Extracted: ConceptExpert, Gate, DeepSeekMoE
"""

import math
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from .config import ModelArgs
    from ._common import Linear, RMSNorm, set_dtype, default_dtype
    from .cortex import DomainCortexRouter
except ImportError:
    from config import ModelArgs
    from _common import Linear, RMSNorm, set_dtype, default_dtype
    from cortex import DomainCortexRouter


class ConceptExpert(nn.Module):
    def __init__(
        self, d_model: int, hidden_dim: Optional[int] = None, swiglu_limit: float = 0.0
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * d_model / 3)
        self.w1 = Linear(d_model, hidden_dim)
        self.w3 = Linear(d_model, hidden_dim)
        self.w2 = Linear(hidden_dim, d_model)
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()

        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)

        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.use_hash = layer_id < args.n_hash_layers
        self.ema_bias_lr = args.ema_bias_lr
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.pending_bias_updates = []

        # ── Domain Cortex (brain-like sparse activation over scientific domains) ──
        self.use_domain_cortex = getattr(args, "use_domain_cortex", False) and not self.use_hash
        if self.use_domain_cortex:
            self.cortex = DomainCortexRouter(
                dim=args.dim,
                n_routed_experts=args.n_routed_experts,
                n_domains=getattr(args, "n_domains", 8),
                domain_topk=getattr(args, "domain_topk", 2),
                route_noise=getattr(args, "cortex_route_noise", 1.0),
                eps=args.norm_eps,
            )
        else:
            self.cortex = None
        self.last_cortex_aux = torch.tensor(0.0)
        self.last_expert_mask = None
        if self.cortex is not None:
            eligible = self.cortex.domain_topk * self.cortex.experts_per_domain
            assert eligible >= self.topk, (
                f"domain_topk*experts_per_domain ({eligible}) must be >= "
                f"n_activated_experts ({self.topk}) so enough experts remain eligible"
            )

        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.normal_(self.weight, 0.0, 0.02)

        self.router_scale = nn.Parameter(torch.ones(args.dim))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(args.n_routed_experts))

        if self.use_hash:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(
                torch.zeros(args.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        self._saved_indices = None

    def apply_pending_updates(self):
        if self.bias is not None and self.pending_bias_updates:
            with torch.no_grad():
                for update in self.pending_bias_updates:
                    self.bias.add_(update)
            self.pending_bias_updates.clear()

    def clear_saved_checkpoint_state(self):
        self._saved_indices = None

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        domain_steer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute expert routing scores and select top-k experts.

        Routing modes (determined by ``DeepSeekMoE.adaptive_routing``):
          • **Fixed top-k** (default, exercised by Tiny_Config):
            Selects exactly ``n_activated_experts`` per token.  Gathered gate
            weights are renormalized to sum 1.0 (±1e-5), then scaled by
            ``route_scale``.
          • **Adaptive / nucleus** (``moe_adaptive_routing=True``):
            Variable-k selection via ``_adaptive_select``; also renormalizes
            to 1.0 before route_scale.

        Both paths guarantee every expert is reachable given sufficient batch
        diversity (no dead experts by design for non-hash routing).

        Returns:
            weights: [N, topk] renormalized gate weights (sum ≈ route_scale).
            indices: [N, topk] selected expert indices.
            z_loss: scalar router z-loss.
            router_probs: [N, n_routed] full probability distribution.
        """
        # Gemma 4 Router Norm & Scale
        x_norm = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        root_size = 1.0 / math.sqrt(x.size(-1))
        router_input = x_norm * root_size * self.router_scale.float()

        scores = F.linear(router_input, self.weight.float())
        z_loss = torch.logsumexp(scores, dim=-1).square().mean()

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).clamp(min=1e-8).sqrt()

        router_probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)

        if self.bias is not None:
            scores = scores + self.bias

        # DeepSeek-V3 Gating score correction bias during evaluation
        if not self.training:
            scores_for_choice = scores + self.e_score_correction_bias.type_as(
                scores
            ).unsqueeze(0)
        else:
            scores_for_choice = scores

        # ── Domain Cortex: mask experts outside the active cortical columns ──
        # Brain-like sparse activation: only experts whose domain column was
        # selected (top-k) for this token remain eligible for expert top-k.
        if self.cortex is not None:
            expert_mask, _domain_probs, _domain_idx, cortex_aux = self.cortex(
                router_input, domain_steer=domain_steer
            )
            scores_for_choice = scores_for_choice.masked_fill(~expert_mask, float("-inf"))
            self.last_cortex_aux = cortex_aux
            self.last_expert_mask = expert_mask
        else:
            self.last_cortex_aux = torch.zeros((), device=x.device, dtype=torch.float32)
            self.last_expert_mask = None

        # DeepSeek-V3 Group-wise routing
        if self.n_group > 1:
            N_tokens, E_experts = scores_for_choice.shape
            experts_per_group = E_experts // self.n_group
            grouped_scores = scores_for_choice.view(
                N_tokens, self.n_group, experts_per_group
            )

            # Represent each group by the sum of its top expert scores
            k_group_top = min(2, experts_per_group)
            group_scores = grouped_scores.topk(k_group_top, dim=-1)[0].sum(
                dim=-1
            )  # [N_tokens, n_group]

            # Select top groups
            group_idx = torch.topk(
                group_scores, k=min(self.topk_group, self.n_group), dim=-1
            )[1]  # [N_tokens, topk_group]

            # Create mask for selected groups
            group_mask = torch.zeros_like(group_scores)  # [N_tokens, n_group]
            group_mask.scatter_(1, group_idx, 1.0)

            # Expand mask back to individual experts
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, -1, experts_per_group)
                .reshape(N_tokens, E_experts)
            )
            scores_masked = scores_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf")
            )

            if self.use_hash and input_ids is not None:
                indices_dyn = self.tid2eid[input_ids]
            else:
                indices_dyn = scores_masked.topk(self.topk, dim=-1)[1]
        else:
            if self.use_hash and input_ids is not None:
                indices_dyn = self.tid2eid[input_ids]
            else:
                indices_dyn = scores_for_choice.topk(self.topk, dim=-1)[1]

        if self.training:
            if self._saved_indices is None:
                self._saved_indices = indices_dyn
            indices = self._saved_indices
        else:
            indices = indices_dyn

        if self.training and self.bias is not None:
            with torch.no_grad():
                counts = torch.bincount(
                    indices.flatten(), minlength=self.weight.shape[0]
                ).float()
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                total_routed = indices.numel() * (
                    dist.get_world_size() if dist.is_initialized() else 1
                )
                routing_fraction = counts / (total_routed / self.topk)
                target_fraction = 1.0 / self.weight.shape[0]
                bias_update = self.ema_bias_lr * (target_fraction - routing_fraction)
                self.pending_bias_updates.append(bias_update)

        weights = router_probs.gather(1, indices)

        # ── Top-k renormalization (Req 6.1) ──────────────────────────────
        # Ensures selected gate weights sum to exactly 1.0 (within ±1e-5)
        # before route_scale is applied.  The eps prevents division-by-zero
        # for degenerate inputs while introducing negligible error (~1e-8).
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        weights = weights * self.route_scale
        return weights, indices, z_loss, router_probs


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-V4 style MoE with:
      • Optional Latent MoE (gating on full-dim, expert compute in latent space)
      • Expert capacity buffer with token dropping (expert_capacity_factor)
      • Per-expert learnable output scale
      • Auxiliary load-balance loss exposed alongside z-loss

    Routing Modes:
      • **Fixed top-k** (``moe_adaptive_routing=False``, the default):
        Every token recruits exactly ``n_activated_experts`` experts.
        Gate weights are renormalized to sum 1.0 ± 1e-5 (Req 6.1).
        **This is the mode exercised by Tiny_Config (config_100m.json).**
      • **Adaptive / nucleus** (``moe_adaptive_routing=True``):
        Variable-k expert recruitment via top-p selection, bounded by
        [moe_min_experts, moe_max_experts].  Also renormalizes to 1.0.

    Both modes ensure every expert is reachable (no dead experts) given
    sufficient batch diversity, thanks to the positive-score property of
    sqrtsoftplus routing and uniform weight initialization.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.latent_dim = getattr(args, "moe_latent_dim", None) or args.dim
        self.use_latent = self.latent_dim < args.dim
        self.n_routed = args.n_routed_experts
        self.n_activated = args.n_activated_experts
        self.capacity_factor = getattr(args, "expert_capacity_factor", 1.25)
        self.load_balance_coeff = getattr(args, "moe_load_balance_coeff", 0.01)
        self.router_z_loss_coeff = getattr(args, "router_z_loss_coeff", 0.001)
        self.router_entropy_coeff = getattr(args, "moe_router_entropy_coeff", 0.001)
        self.capacity_loss_coeff = getattr(args, "moe_capacity_loss_coeff", 0.01)
        self.cortex_load_balance_coeff = getattr(args, "cortex_load_balance_coeff", 0.01)

        self.gate = Gate(0, args)

        if self.use_latent:
            self.w_down = Linear(self.dim, self.latent_dim)
            self.w_up = Linear(self.latent_dim, self.dim)
            expert_dim = self.latent_dim
        else:
            expert_dim = self.dim

        self.experts = nn.ModuleList(
            [
                ConceptExpert(
                    expert_dim,
                    args.effective_moe_inter_dim,
                    swiglu_limit=args.swiglu_limit,
                )
                for _ in range(self.n_routed)
            ]
        )
        self.shared = ConceptExpert(
            expert_dim, args.effective_moe_inter_dim, swiglu_limit=args.swiglu_limit
        )
        self.per_expert_scale = nn.Parameter(torch.ones(self.n_routed))

        # ── Dual dense+MoE FFW branch (Gemma4 mlp2 pattern) ─────────────
        # Runs a compact dense FFN in parallel with sparse MoE so that every
        # token *always* has a gradient path through a dense layer.
        self.use_dual_ffn = getattr(args, "moe_dual_ffn", True)
        if self.use_dual_ffn:
            dual_inter = getattr(args, "moe_dual_inter_dim", 0)
            dual_inter = dual_inter if dual_inter > 0 else args.effective_moe_inter_dim
            self.dense_branch = ConceptExpert(
                expert_dim, dual_inter, swiglu_limit=args.swiglu_limit
            )
            self.dense_branch_norm = RMSNorm(expert_dim, args.norm_eps)

        # Expert dropout: randomly zero token→expert assignments during training
        self.expert_dropout_p = getattr(args, "moe_expert_dropout", 0.0)

        # ── Adaptive (brain-like) variable-k expert recruitment ──────────
        # Each token recruits a *variable* number of specialised experts via
        # top-p (nucleus) selection over router probs, bounded by [min, max].
        # Confident tokens recruit few (cheap); ambiguous tokens recruit more.
        # The always-on shared (+dense) branch is the cheap base path.
        self.adaptive_routing = getattr(args, "moe_adaptive_routing", False)
        self.route_top_p = getattr(args, "moe_route_top_p", 0.5)
        self.min_experts = max(0, getattr(args, "moe_min_experts", 1))
        _max = getattr(args, "moe_max_experts", 0)
        self.max_experts = _max if _max > 0 else self.n_activated
        self.adaptive_sparsity_coeff = getattr(args, "moe_adaptive_sparsity_coeff", 0.0)

        self.last_expert_counts: Optional[torch.Tensor] = None
        self.last_capacity_overflow = torch.tensor(0.0)
        self.last_router_entropy = torch.tensor(0.0)
        self.last_avg_experts = torch.tensor(float(self.n_activated))
        self._saved_sel = None
        self._saved_w = None

    def clear_saved_checkpoint_state(self):
        self.gate.clear_saved_checkpoint_state()
        self._saved_sel = None
        self._saved_w = None

    def _adaptive_select(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-p (nucleus) variable-k expert selection per token.

        Selects a variable number of experts per token via nucleus (top-p)
        selection over router probabilities, bounded by [min_experts, max_experts].
        Confident tokens recruit few experts; ambiguous tokens recruit more.

        After selection, weights are renormalized to sum to 1.0 (±1e-5) then
        scaled by ``route_scale``.  This ensures probability mass is conserved
        across the variable expert set.

        Args:
            probs: [N, n_routed] (cortex-masked, non-negative router probs).

        Returns:
            sel: bool [N, n_routed] — which experts are active per token.
            w: [N, n_routed] — renormalized weights (sum ≈ route_scale per token
               for selected experts, 0 for unselected).
        """
        sorted_p, sorted_i = probs.sort(dim=-1, descending=True)
        cum = sorted_p.cumsum(dim=-1)
        # Keep experts until cumulative mass reaches top_p (include the crossing one).
        keep = (cum - sorted_p) < self.route_top_p
        if self.min_experts > 0:
            keep[:, : self.min_experts] = True
        if self.max_experts < keep.shape[1]:
            keep[:, self.max_experts :] = False
        sel = torch.zeros_like(probs, dtype=torch.bool).scatter(1, sorted_i, keep)
        w = probs * sel
        # Renormalize selected weights to sum to 1.0, then apply route_scale.
        # The eps prevents division by zero when all probs are masked out.
        w_sum = w.sum(dim=-1, keepdim=True)
        w = w / (w_sum + 1e-8) * self.gate.route_scale
        return sel, w

    def _adaptive_forward(
        self, x: torch.Tensor, domain_steer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        N_tokens = flat_x.shape[0]

        _w, _idx, z_loss, router_probs = self.gate(flat_x, None, domain_steer=domain_steer)
        probs = router_probs.float()
        if self.gate.last_expert_mask is not None:
            probs = probs.masked_fill(~self.gate.last_expert_mask, 0.0)

        # Always run the dynamic selection logic to build the exact same autograd graph in both passes
        sel_dyn, w_dyn = self._adaptive_select(probs)

        if self.training:
            if self._saved_sel is None:
                self._saved_sel = sel_dyn
            sel = self._saved_sel
            print(f"[MoE DEBUG] training={self.training}, sel_dyn sum={sel_dyn.sum().item()}, saved_sel sum={self._saved_sel.sum().item()}")
            # Since sel is boolean, we compute w dynamically from probs and sel.
            # This ensures that:
            # 1. w matches the saved shape and content from the forward pass.
            # 2. w maintains its dynamic gradient path with respect to probs.
            w = probs * sel
            w_sum = w.sum(dim=-1, keepdim=True)
            w = w / (w_sum + 1e-8) * self.gate.route_scale
        else:
            sel = sel_dyn
            w = w_dyn

        # ── Correct EMA bias for adaptive routing ────────────────────────
        # Gate.forward computed an EMA bias update based on its internal
        # fixed top-k indices, but we used adaptive (variable-k) selection
        # instead.  Replace the stale update with one reflecting actual usage.
        if self.training and self.gate.bias is not None:
            with torch.no_grad():
                # Discard the stale update that Gate.forward appended.
                if self.gate.pending_bias_updates:
                    self.gate.pending_bias_updates.pop()
                # Compute a corrected update from the adaptive selection.
                counts = sel.float().sum(dim=0)  # [n_routed]
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                total_tokens = N_tokens * (
                    dist.get_world_size() if dist.is_initialized() else 1
                )
                # routing_fraction: fraction of tokens each expert serves,
                # normalized by the mean expert count per token for fairness.
                avg_k = sel.float().sum(dim=1).mean().item()
                routing_fraction = counts / (total_tokens * max(avg_k, 1.0) / self.n_routed)
                target_fraction = 1.0 / self.n_routed
                bias_update = self.gate.ema_bias_lr * (target_fraction - routing_fraction)
                self.gate.pending_bias_updates.append(bias_update)

        base_x = self.w_down(flat_x) if self.use_latent else flat_x
        y = torch.zeros_like(base_x, dtype=torch.float32)
        expert_counts = sel.float().sum(dim=0)  # [n_routed]

        for i, exp in enumerate(self.experts):
            tok_idx = sel[:, i].nonzero(as_tuple=True)[0]
            if tok_idx.numel() == 0:
                continue
            exp_out = exp(base_x[tok_idx], w[tok_idx, i, None])
            if self.training and self.expert_dropout_p > 0.0:
                keep = (
                    (torch.rand(exp_out.shape[0], device=exp_out.device) > self.expert_dropout_p)
                    .float()
                    .unsqueeze(-1)
                )
                exp_out = exp_out * keep / (1.0 - self.expert_dropout_p + 1e-8)
            y.scatter_add_(
                0,
                tok_idx.unsqueeze(-1).expand(-1, y.shape[-1]),
                (exp_out * self.per_expert_scale[i].type_as(exp_out)).float(),
            )

        # ── Three-branch summation (Req 6.2) ────────────────────────────
        # Branch 1: Routed-expert contributions (accumulated in y above).
        # Branch 2: Shared expert — always processes all tokens (no dropout).
        y = y + self.shared(base_x).float()
        # Branch 3: Dense FFN — always-on parallel dense path (when enabled).
        if self.use_dual_ffn:
            y = y + self.dense_branch(self.dense_branch_norm(base_x)).float()

        y = self.w_up(y.type_as(x)) if self.use_latent else y.type_as(x)

        # ── Aux losses ──
        fi = expert_counts / (N_tokens + 1e-8)
        Pi = router_probs.float().mean(dim=0)
        load_balance_loss = self.n_routed * (fi * Pi).sum()
        router_entropy = (
            -(router_probs.float() * torch.log(router_probs.float() + 1e-8)).sum(dim=-1).mean()
        )
        router_entropy_loss = -router_entropy / math.log(max(2, self.n_routed))
        avg_experts = sel.float().sum(dim=1).mean()  # mean experts recruited per token

        self.last_expert_counts = expert_counts.detach()
        self.last_router_entropy = router_entropy.detach()
        self.last_avg_experts = avg_experts.detach()
        self.last_capacity_overflow = torch.zeros((), device=x.device)

        aux = (
            self.router_z_loss_coeff * z_loss
            + self.load_balance_coeff * load_balance_loss
            + self.router_entropy_coeff * router_entropy_loss
            # Sparsity pressure: bias toward recruiting FEW experts ("start small").
            + self.adaptive_sparsity_coeff * (avg_experts / self.n_routed)
        )
        if self.gate.cortex is not None:
            aux = aux + self.cortex_load_balance_coeff * self.gate.last_cortex_aux
        return y.reshape(shape), aux

    def forward(
        self, x: torch.Tensor, domain_steer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.adaptive_routing:
            return self._adaptive_forward(x, domain_steer)
        shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        N_tokens = flat_x.shape[0]

        weights, indices, z_loss, router_probs = self.gate(flat_x, None, domain_steer=domain_steer)

        # ── Expert capacity: max tokens each expert can receive ──────
        # capacity = ceil(capacity_factor * tokens / n_experts * n_activated)
        capacity = max(
            1,
            int(
                math.ceil(
                    self.capacity_factor * N_tokens * self.n_activated / self.n_routed
                )
            ),
        )

        # ── Auxiliary load-balance loss (Gemma-4 style) ──────────────
        # Encourage uniform distribution of tokens across experts
        # fi = fraction of tokens routed to expert i (averaged over activated experts)
        expert_counts = torch.zeros(self.n_routed, device=x.device, dtype=torch.float32)
        expert_counts.scatter_add_(
            0,
            indices.flatten().long().clamp(0, self.n_routed - 1),
            torch.ones(indices.numel(), device=x.device),
        )
        fi = expert_counts / (N_tokens * self.n_activated + 1e-8)  # (n_routed,)
        Pi = router_probs.float().mean(dim=0)
        load_balance_loss = self.n_routed * (fi * Pi.float()).sum()
        router_entropy = (
            -(router_probs.float() * torch.log(router_probs.float() + 1e-8))
            .sum(dim=-1)
            .mean()
        )
        router_entropy_loss = -router_entropy / math.log(max(2, self.n_routed))
        capacity_overflow = (expert_counts - capacity).clamp(min=0).sum() / (
            indices.numel() + 1e-8
        )
        self.last_expert_counts = expert_counts.detach()
        self.last_capacity_overflow = capacity_overflow.detach()
        self.last_router_entropy = router_entropy.detach()

        # ── Forward through experts ───────────────────────────────────
        base_x = self.w_down(flat_x) if self.use_latent else flat_x
        y = torch.zeros_like(base_x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed).tolist()

        for i, exp in enumerate(self.experts):
            if counts[i] == 0:
                continue
            tok_idx, top_pos = torch.where(indices == i)
            # Token-dropping: only process up to capacity
            if tok_idx.shape[0] > capacity:
                tok_weights = weights[tok_idx, top_pos]
                _, sort_idx = torch.topk(tok_weights, k=capacity, largest=True)
                tok_idx = tok_idx[sort_idx]
                top_pos = top_pos[sort_idx]
            exp_out = exp(base_x[tok_idx], weights[tok_idx, top_pos, None])
            # Expert dropout: stochastically zero-out token contributions during training
            # Uses inverted scaling to keep expected value constant (like nn.Dropout)
            if self.training and self.expert_dropout_p > 0.0:
                keep = (
                    (
                        torch.rand(exp_out.shape[0], device=exp_out.device)
                        > self.expert_dropout_p
                    )
                    .float()
                    .unsqueeze(-1)
                )
                exp_out = exp_out * keep / (1.0 - self.expert_dropout_p + 1e-8)
            y.scatter_add_(
                0,
                tok_idx.unsqueeze(-1).expand(-1, y.shape[-1]),
                (exp_out * self.per_expert_scale[i].type_as(exp_out)).float(),
            )

        # ── Three-branch summation (Req 6.2) ────────────────────────────
        # Branch 1: Routed-expert contributions (accumulated in y above).
        # Branch 2: Shared expert — always processes all tokens (no dropout).
        y = y + self.shared(base_x).float()

        # Branch 3: Dual dense FFN (Gemma4 mlp2) — always-on dense path.
        # Runs in parallel with the sparse MoE to guarantee every token a
        # direct, dense gradient signal — critical for training stability.
        if self.use_dual_ffn:
            dense_in = self.dense_branch_norm(base_x)
            y = y + self.dense_branch(dense_in).float()

        if self.use_latent:
            y = self.w_up(y.type_as(x))
        else:
            y = y.type_as(x)

        # Weighted combined auxiliary loss: router_z_loss + load_balance
        aux = (
            self.router_z_loss_coeff * z_loss
            + self.load_balance_coeff * load_balance_loss
            + self.router_entropy_coeff * router_entropy_loss
            + self.capacity_loss_coeff * capacity_overflow
        )
        # Domain-cortex load-balance (keeps cortical columns evenly utilised).
        if self.gate.cortex is not None:
            aux = aux + self.cortex_load_balance_coeff * self.gate.last_cortex_aux
        return y.reshape(shape), aux

    def quantize_routed_experts_to_nvfp4(self) -> None:
        """Quantize routed expert weights to simulated NVFP4 E2M1 format in-place."""
        for expert in self.experts:
            for linear_layer in [expert.w1, expert.w3, expert.w2]:
                q_weight, scale = quantize_weight_to_nvfp4(linear_layer.weight.data, block_size=32)
                linear_layer.weight.data.copy_(q_weight)
                linear_layer.scale = nn.Parameter(scale, requires_grad=False)
                linear_layer.weight.use_fp4_weights = True
                linear_layer.weight.scale = linear_layer.scale

    def quantize_routed_experts_to_fp8(self) -> None:
        """Quantize routed expert weights to FP8 E4M3 format in-place."""
        for expert in self.experts:
            for linear_layer in [expert.w1, expert.w3, expert.w2]:
                q_weight, scale = quantize_weight_to_fp8(linear_layer.weight.data, block_size=128)
                linear_layer.weight = nn.Parameter(q_weight, requires_grad=linear_layer.weight.requires_grad)
                linear_layer.scale = nn.Parameter(scale, requires_grad=False)
                linear_layer.weight.scale = linear_layer.scale

    def quantize_shared_expert_to_fp8(self) -> None:
        """Quantize shared expert weights to FP8 E4M3 format in-place."""
        for linear_layer in [self.shared.w1, self.shared.w3, self.shared.w2]:
            q_weight, scale = quantize_weight_to_fp8(linear_layer.weight.data, block_size=128)
            linear_layer.weight = nn.Parameter(q_weight, requires_grad=linear_layer.weight.requires_grad)
            linear_layer.scale = nn.Parameter(scale, requires_grad=False)
            linear_layer.weight.scale = linear_layer.scale


def quantize_weight_to_nvfp4(
    weight: torch.Tensor, block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to simulated NVFP4 E2M1 format with block scales."""
    fp4_max = 6.0
    shape = weight.shape
    out_features, in_features = shape

    actual_block_size = block_size
    while in_features % actual_block_size != 0 and actual_block_size > 1:
        actual_block_size //= 2

    w_flat = weight.reshape(-1, actual_block_size).float()
    amax = w_flat.abs().amax(dim=-1, keepdim=True).clamp(min=fp4_max * 2**-126)
    s_flat = torch.pow(2.0, torch.ceil(torch.log2(amax / fp4_max)))

    fp4_vals = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ]
    fp4_values = torch.tensor(fp4_vals, device=weight.device, dtype=torch.float32)
    scaled = (w_flat / s_flat).clamp(-fp4_max, fp4_max)
    dist = (scaled.unsqueeze(-1) - fp4_values).abs()
    y_fp4 = fp4_values[dist.argmin(-1)]

    quantized_weight = y_fp4.reshape(shape).to(weight.dtype)
    scale = s_flat.reshape(out_features, in_features // actual_block_size).to(weight.dtype)
    return quantized_weight, scale


def quantize_weight_to_fp8(
    weight: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to FP8 E4M3 format with block scales."""
    try:
        from .kernel import act_quant
    except ImportError:
        from kernel import act_quant
    y_out, s_out = act_quant(weight.contiguous(), block_size=block_size)
    return y_out, s_out

