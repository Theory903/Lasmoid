"""
LasmoidBlock — Hybrid Concept Attention Transformer Block
==========================================================
Extracted from monolithic model.py during Phase 0 SOLiD refactoring.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from ._common import RMSNorm, Linear
    from .attention import CSAAttention, HCAAttention, HybridSlidingGlobal
    from .mhc import ManifoldConstrainedHyperConnection
    from .moe import DeepSeekMoE
    from .ssm import StateSpaceRecurrence
    from .config import ModelArgs
    from .attnres import BlockAttnRes
except ImportError:
    from _common import RMSNorm, Linear
    from attention import CSAAttention, HCAAttention, HybridSlidingGlobal
    from mhc import ManifoldConstrainedHyperConnection
    from moe import DeepSeekMoE
    from ssm import StateSpaceRecurrence
    from config import ModelArgs
    from attnres import BlockAttnRes


class LasmoidBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.skip_scale = nn.Parameter(torch.ones(1))

        # Interleaved Hybrid Attention (Gemma-4 style)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        attn_type = getattr(args, "attention_type", "local")
        is_hybrid_layer = attn_type == "hybrid" and (layer_id % 5 == 4)

        if is_hybrid_layer:
            self.attn = HybridSlidingGlobal(args, layer_id)
        elif layer_id % 2 == 0:
            self.attn = CSAAttention(args)  # uses m compression
        else:
            self.attn = HCAAttention(args)  # uses m' compression

        # Parallel State Space Recurrence Branch
        self.ssm_branch = StateSpaceRecurrence(args)

        # DeepSeek-V4 Manifold Hyper Connections
        self.mhc_attn = ManifoldConstrainedHyperConnection(
            args.dim, args.num_residual_streams, args.hc_sinkhorn_iters
        )
        self.mhc_ffn = ManifoldConstrainedHyperConnection(
            args.dim, args.num_residual_streams, args.hc_sinkhorn_iters
        )

        # Dummy last_pred_loss for compatibility
        self.last_pred_loss = torch.tensor(0.0)

        # MoE using sqrt(softplus) as already configured
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.moe_layer = DeepSeekMoE(args)

        # ── Post-layer norms (Gemma4): RMSNorm on attn/FFW output ────────
        self.use_post_attn_norm = getattr(args, "post_attn_norm", True)
        self.use_post_ffw_norm = getattr(args, "post_ffw_norm", True)
        if self.use_post_attn_norm:
            self.post_attn_norm = RMSNorm(args.dim, args.norm_eps)
        if self.use_post_ffw_norm:
            self.post_ffw_norm = RMSNorm(args.dim, args.norm_eps)

        # Block AttnRes (gated)
        self.use_block_attnres = getattr(args, "use_block_attnres", False)
        if self.use_block_attnres:
            self.block_attnres = BlockAttnRes(
                args.dim,
                block_size=getattr(args, "block_attnres_block_size", 16),
                n_blocks=getattr(args, "block_attnres_n_blocks", 4),
                gate_type=getattr(args, "attnres_gate_type", "alpha"),
            )

        # Per-layer modality feature projection
        per_layer_dim = getattr(args, "per_layer_input_dim", 64)
        self.layer_feats_proj = Linear(per_layer_dim, args.dim, dtype=torch.bfloat16)

    def forward(
        self,
        streams: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        input_ids: Optional[torch.Tensor] = None,
        layer_feats: Optional[torch.Tensor] = None,
        domain_steer: Optional[torch.Tensor] = None,
        r_step: int = 0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        print(f"--- BLOCK {self.layer_id} FORWARD ---")
        if layer_feats is not None:
            proj_feats = self.layer_feats_proj(layer_feats.to(streams.dtype))
            # Inject modality features into the primary stream (stream 0)
            streams = streams.clone()
            streams[:, :, 0] = streams[:, :, 0] + proj_feats
        # 1. mHC Pre-Attention
        A_l_attn, B_l_attn, C_l_attn = self.mhc_attn(streams)
        attn_in = A_l_attn * streams

        # 2. Attention (CSA or HCA)
        B, S, H, D = attn_in.shape
        attn_in_flat = attn_in.transpose(1, 2).reshape(B * H, S, D)

        # Compute normalized input once (shared by attention and SSM branches)
        normed_in = self.attn_norm(attn_in_flat)

        # Run Attention path
        attn_out_flat = self.attn(normed_in, freqs_cis, start_pos, r_step=r_step)

        # Run Parallel SSM Recurrence path
        ssm_out_flat = self.ssm_branch(normed_in, start_pos)

        # Fuse outputs
        fused_out_flat = attn_out_flat + ssm_out_flat

        if self.use_post_attn_norm:
            fused_out_flat = self.post_attn_norm(fused_out_flat)

        attn_out = fused_out_flat.reshape(B, H, S, D).transpose(1, 2)

        # 3. mHC Post-Attention + Birkhoff Constraint Mix
        streams = B_l_attn @ streams + C_l_attn * attn_out

        # 4. mHC Pre-FFN
        A_l_ffn, B_l_ffn, C_l_ffn = self.mhc_ffn(streams)
        ffn_in = A_l_ffn * streams

        # 5. MoE FFN
        B_f, S_f, H_f, D_f = ffn_in.shape
        ffn_in_flat = ffn_in.transpose(1, 2).reshape(B_f * H_f, S_f, D_f)

        ffn_out_flat, z_loss = self.moe_layer(self.ffn_norm(ffn_in_flat), domain_steer=domain_steer, r_step=r_step)

        if self.use_post_ffw_norm:
            ffn_out_flat = self.post_ffw_norm(ffn_out_flat)

        ffn_out = ffn_out_flat.reshape(B_f, H_f, S_f, D_f).transpose(1, 2)

        # 6. mHC Post-FFN
        streams = B_l_ffn @ streams + C_l_ffn * ffn_out

        if self.use_block_attnres:
            streams = self.block_attnres(streams)

        streams = streams * self.skip_scale

        # Backward-compatible outputs
        vq_loss = torch.tensor(0.0, device=streams.device, dtype=streams.dtype)
        routing = torch.zeros(B, S, device=streams.device, dtype=streams.dtype)
        indices = torch.zeros(B, S, dtype=torch.long, device=streams.device)
        adj = torch.zeros(1, 1, device=streams.device, dtype=streams.dtype)

        event_prob = getattr(self.attn, "_last_event_prob", None)

        return streams, z_loss, vq_loss, routing, indices, adj, event_prob
