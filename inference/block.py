"""
LasmoidBlock — Hybrid Concept Attention Transformer Block
==========================================================
Extracted from monolithic model.py during Phase 0 SOLiD refactoring.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from ._common import RMSNorm
    from .attention import CSAAttention, HCAAttention
    from .mhc import ManifoldConstrainedHyperConnection
    from .moe import DeepSeekMoE
    from .ssm import StateSpaceRecurrence
    from .config import ModelArgs
except ImportError:
    from _common import RMSNorm
    from attention import CSAAttention, HCAAttention
    from mhc import ManifoldConstrainedHyperConnection
    from moe import DeepSeekMoE
    from ssm import StateSpaceRecurrence
    from config import ModelArgs


class LasmoidBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.skip_scale = nn.Parameter(torch.ones(1))

        # Interleaved Hybrid Attention
        self.is_csa = layer_id % 2 == 0
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)

        if self.is_csa:
            self.attn = CSAAttention(args)  # uses m compression
        else:
            self.attn = HCAAttention(args)  # uses m' compression

        # Parallel State Space Recurrence Branch
        self.ssm_branch = StateSpaceRecurrence(args)

        # DeepSeek-V4 Manifold Hyper Connections
        self.mhc_attn = ManifoldConstrainedHyperConnection(
            args.dim, args.num_residual_streams
        )
        self.mhc_ffn = ManifoldConstrainedHyperConnection(
            args.dim, args.num_residual_streams
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

    def forward(
        self,
        streams: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # 1. mHC Pre-Attention
        A_l_attn, B_l_attn, C_l_attn = self.mhc_attn(streams)
        attn_in = A_l_attn * streams

        # 2. Attention (CSA or HCA)
        B, S, H, D = attn_in.shape
        attn_in_flat = attn_in.transpose(1, 2).reshape(B * H, S, D)

        # Run Attention path
        attn_out_flat = self.attn(self.attn_norm(attn_in_flat), freqs_cis, start_pos)

        # Run Parallel SSM Recurrence path
        ssm_out_flat = self.ssm_branch(self.attn_norm(attn_in_flat), start_pos)

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

        ffn_out_flat, z_loss = self.moe_layer(self.ffn_norm(ffn_in_flat))

        if self.use_post_ffw_norm:
            ffn_out_flat = self.post_ffw_norm(ffn_out_flat)

        ffn_out = ffn_out_flat.reshape(B_f, H_f, S_f, D_f).transpose(1, 2)

        # 6. mHC Post-FFN
        streams = B_l_ffn @ streams + C_l_ffn * ffn_out

        streams = streams * self.skip_scale

        # Backward-compatible outputs
        vq_loss = torch.tensor(0.0, device=streams.device, dtype=streams.dtype)
        routing = torch.zeros(B, S, device=streams.device, dtype=streams.dtype)
        indices = torch.zeros(B, S, dtype=torch.long, device=streams.device)
        adj = torch.zeros(1, 1, device=streams.device, dtype=streams.dtype)

        return streams, z_loss, vq_loss, routing, indices, adj
