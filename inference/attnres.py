"""
Lasmoid — attnres.py
======================
Implements Block AttnRes (block-level depth attention routing) for cross-layer reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._common import Linear
except ImportError:
    from _common import Linear


class BlockAttnRes(nn.Module):
    """
    Block AttnRes: Pools activation streams into sequence blocks,
    and applies a small depth-attention pass over blocks using learned query tokens.

    Adds (a) a learned *recency bias* that, at initialisation, makes the depth
    attention strongly prefer the most-recent block so the module behaves like a
    plain residual connection, and (b) a learnable *gate* on the residual merge
    (scalar / per-channel sigmoid gate, or an additive ``alpha`` gate) so depth
    routing can be turned on gradually and safely during training.
    """

    def __init__(
        self,
        dim: int,
        block_size: int = 16,
        n_blocks: int = 4,
        gate_type: str = "alpha",
    ):
        super().__init__()
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.dim = dim
        self.gate_type = gate_type

        self.block_q = nn.Parameter(torch.randn(n_blocks, dim))
        self.block_kv_proj = Linear(dim, dim)
        self.num_heads = 4
        self.depth_attn = nn.MultiheadAttention(dim, num_heads=self.num_heads, batch_first=True)

        nn.init.normal_(self.block_q, mean=0.0, std=0.02)

        # Recency bias: large positive value so softmax over blocks is dominated
        # by the most-recent block at init → output ≈ residual (current block).
        self.recency_bias = nn.Parameter(torch.tensor(10.0))

        # Residual merge gates.
        #   "alpha"  : streams + alpha * summary          (alpha init 0 → exact residual)
        #   "scalar" : sigmoid(gate)*summary + ... merge  (scalar init 0)
        #   "vector" : per-channel sigmoid gate           (vector init 0)
        #   "none"   : streams + summary                  (legacy behaviour)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gate = nn.Parameter(torch.zeros(1))
        self.gate_vec = nn.Parameter(torch.zeros(dim))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        # streams shape: [B, S, HC, D]
        B, S, HC, D = streams.shape
        dtype = streams.dtype

        # 1. Pad and pool streams into blocks along sequence dimension
        pad_len = (self.block_size - (S % self.block_size)) % self.block_size
        if pad_len > 0:
            padded = F.pad(streams, (0, 0, 0, 0, 0, pad_len))
        else:
            padded = streams

        num_blocks = padded.shape[1] // self.block_size
        blocked = padded.view(B, num_blocks, self.block_size, HC, D)
        # Pool over (seq_within_block, streams) → one repr per block (Req 18.3)
        block_reprs = blocked.mean(dim=(2, 3))  # Shape: [B, num_blocks, D]

        # 2. Project block representations to KV space — compute once, reuse below
        kv = self.block_kv_proj(block_reprs)  # Shape: [B, num_blocks, D]

        # 3. Depth attention: learned queries attend to block summaries.
        # Expand queries once; cast to float32 for stable attention
        q = self.block_q.unsqueeze(0).expand(B, -1, -1).to(device=streams.device, dtype=torch.float32)
        kv_f32 = kv.to(dtype=torch.float32)

        # Additive recency bias on the most-recent block (last key position),
        # broadcast across all query rows. Shape: (n_blocks, num_blocks).
        attn_bias = torch.zeros(
            self.n_blocks, num_blocks, device=streams.device, dtype=torch.float32
        )
        attn_bias[:, -1] = self.recency_bias.to(torch.float32)

        attn_out, _ = self.depth_attn(q, kv_f32, kv_f32, attn_mask=attn_bias)  # [B, n_blocks, D]
        attn_out = attn_out.to(dtype)

        # 4. Route depth attention output back to streams via the chosen gate
        # Compute summary once (mean over block queries) and broadcast (Req 18.3)
        summary = attn_out.mean(dim=1, keepdim=True).unsqueeze(2)  # [B, 1, 1, D]

        if self.gate_type == "alpha":
            return streams + self.alpha.to(dtype) * summary
        elif self.gate_type == "scalar":
            g = torch.sigmoid(self.gate).to(dtype)
            return (1.0 - g) * streams + g * summary
        elif self.gate_type == "vector":
            g = torch.sigmoid(self.gate_vec).to(dtype).view(1, 1, 1, D)
            return (1.0 - g) * streams + g * summary
        else:  # "none" — legacy additive residual
            return streams + summary
