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
except ImportError:
    from config import ModelArgs
    from _common import Linear, RMSNorm, set_dtype, default_dtype


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

    def apply_pending_updates(self):
        if self.bias is not None and self.pending_bias_updates:
            with torch.no_grad():
                for update in self.pending_bias_updates:
                    self.bias.add_(update)
            self.pending_bias_updates.clear()

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                indices = self.tid2eid[input_ids]
            else:
                indices = scores_masked.topk(self.topk, dim=-1)[1]
        else:
            if self.use_hash and input_ids is not None:
                indices = self.tid2eid[input_ids]
            else:
                indices = scores_for_choice.topk(self.topk, dim=-1)[1]

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

        # Gemma 4 style top-k renormalization to prevent probability leakage
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
        self.last_expert_counts: Optional[torch.Tensor] = None
        self.last_capacity_overflow = torch.tensor(0.0)
        self.last_router_entropy = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        N_tokens = flat_x.shape[0]

        weights, indices, z_loss, router_probs = self.gate(flat_x, None)

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

        # Shared expert always processes all tokens (no dropout)
        y = y + self.shared(base_x).float()

        # ── Dual dense branch (Gemma4 mlp2): always-on dense path ────────
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

