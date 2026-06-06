import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._common import Linear, RMSNorm, set_dtype, default_dtype
    from .kernel import hc_split_sinkhorn
    from .config import ModelArgs
except ImportError:
    from _common import Linear, RMSNorm, set_dtype, default_dtype
    from kernel import hc_split_sinkhorn
    from config import ModelArgs


class ManifoldConstrainedHyperConnection(nn.Module):
    def __init__(self, dim: int, n_hc: int):
        super().__init__()
        self.n_hc = n_hc
        self.norm = RMSNorm(n_hc * dim)
        self.w_pre = Linear(n_hc * dim, n_hc, bias=False)
        self.w_res = Linear(n_hc * dim, n_hc * n_hc, bias=False)
        self.w_post = Linear(n_hc * dim, n_hc, bias=False)
        # Learnable gating factors initialized small
        self.alpha_pre = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_res = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_post = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, x):
        # x shape: [batch, seq_len, n_hc, dim]
        B, S, H, D = x.shape
        x_flat = x.reshape(B, S, H * D)
        x_flat = self.norm(x_flat)

        a_raw = self.alpha_pre.to(x_flat.dtype) * self.w_pre(x_flat)
        c_raw = self.alpha_post.to(x_flat.dtype) * self.w_post(x_flat)
        b_raw = (self.alpha_res.to(x_flat.dtype) * self.w_res(x_flat)).view(
            -1, self.n_hc, self.n_hc
        )

        # DeepSeek-V4 strict constraints (Equations 6, 7, 8)
        A_l = torch.sigmoid(a_raw).unsqueeze(-1)
        C_l = (2.0 * torch.sigmoid(c_raw)).to(x_flat.dtype).unsqueeze(-1)

        # Sinkhorn-Knopp on exp(b_raw) for exactly 20 iterations in float32 for numerical stability
        M = torch.exp(b_raw.float())
        for _ in range(20):
            M = F.normalize(M, p=1, dim=1)  # Column norm
            M = F.normalize(M, p=1, dim=2)  # Row norm
        B_l = M.view(B, S, self.n_hc, self.n_hc).to(
            x_flat.dtype
        )  # Projected onto Birkhoff polytope

        return A_l, B_l, C_l


class MHCBlock(nn.Module):
    def __init__(
        self, dim: int, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = sinkhorn_iters
        self.hc_eps = eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim

        with set_dtype(torch.float32):
            self.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_scale = nn.Parameter(torch.empty(3))

        nn.init.normal_(self.hc_fn, 0, 0.02)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    def hc_pre(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        B, S, hc, D = x.size()
        x_flat = x.flatten(2)
        mean_sq = x_flat.square().mean(-1, keepdim=True).float()
        rsqrt = torch.rsqrt(mean_sq + self.hc_eps).to(dtype)
        mixes = F.linear(x_flat, self.hc_fn.to(dtype)) * rsqrt

        pre, post, comb = hc_split_sinkhorn(
            mixes.float(),
            self.hc_scale.float(),
            self.hc_base.float(),
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )

        y = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)
        return y, post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        dtype = x.dtype
        y = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), residual
        )
        return y
