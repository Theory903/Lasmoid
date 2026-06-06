import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from ._common import Linear, RMSNorm, set_dtype, default_dtype
    from .block import LasmoidBlock
    from .config import ModelArgs
except ImportError:
    from _common import Linear, RMSNorm, set_dtype, default_dtype
    from block import LasmoidBlock
    from config import ModelArgs


class MTPBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.block = LasmoidBlock(layer_id, args)

        hc_mult = args.num_residual_streams
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.hc_head_fn, 0, 0.02)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)

        self.embed: Optional[nn.Embedding] = None
        self.head: Optional[nn.Module] = None

    def hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        B, S, hc, D = shape
        xf = x.flatten(2)
        mean_sq = xf.square().mean(-1, keepdim=True).float()
        rsqrt = torch.rsqrt(mean_sq + 1e-6).to(dtype)
        mixes = F.linear(xf, self.hc_head_fn.to(dtype)) * rsqrt
        pre = (
            torch.sigmoid(mixes.float() * self.hc_head_scale + self.hc_head_base) + 1e-6
        )
        y = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)
        return y

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.embed is not None and self.head is not None
        e = self.enorm(self.embed(input_ids).to(x.dtype))

        B_h, S_h, hc_h, D_h = x.shape
        h_flat = x.reshape(B_h * S_h * hc_h, D_h)
        h_flat = self.hnorm(h_flat)
        h_flat = self.h_proj(h_flat)
        h = h_flat.reshape(B_h, S_h, hc_h, D_h)

        x = self.e_proj(e).unsqueeze(2) + h
        x, z_loss, vq_loss, routing, indices, adj = self.block(
            x, freqs_cis, start_pos, input_ids
        )

        y = self.hc_head_reduce(x)
        y = self.norm(y)
        logits = F.linear(y.float(), self.head.weight.float())
        return logits, z_loss, vq_loss
