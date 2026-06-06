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
    """

    def __init__(self, dim: int, block_size: int = 16, n_blocks: int = 4):
        super().__init__()
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.dim = dim

        self.block_q = nn.Parameter(torch.randn(n_blocks, dim))
        self.block_kv_proj = Linear(dim, dim)
        self.depth_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        nn.init.normal_(self.block_q, mean=0.0, std=0.02)

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
        block_reprs = blocked.mean(dim=(2, 3))  # Shape: [B, num_blocks, D]

        # 2. Project block representations to KV space
        kv = self.block_kv_proj(block_reprs)  # Shape: [B, num_blocks, D]

        # 3. Depth attention: learned queries attend to block summaries
        # Cast inputs to float32 to match block_q and depth_attn default dtype
        q = self.block_q.unsqueeze(0).expand(B, -1, -1).to(device=streams.device, dtype=torch.float32)
        kv_f32 = kv.to(device=streams.device, dtype=torch.float32)

        # Ensure depth_attn is on the correct device and in float32
        self.depth_attn = self.depth_attn.to(device=streams.device).float()

        attn_out, _ = self.depth_attn(q, kv_f32, kv_f32)  # Shape: [B, n_blocks, D]
        attn_out = attn_out.to(dtype)

        # 4. Route depth attention output back to streams
        summary = attn_out.mean(dim=1).unsqueeze(1).unsqueeze(2)  # Shape: [B, 1, 1, D]
        return streams + summary
