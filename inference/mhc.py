"""
Manifold-Constrained Hyper-Connections (mHC)
============================================
Implements DeepSeek-V4 style Sinkhorn-projected residual stream mixing.

The canonical implementation is `ManifoldConstrainedHyperConnection`, used by
`block.py` to mix parallel residual streams via a doubly stochastic (Birkhoff
polytope) matrix. When `n_hc == 1` (single stream), the module reduces to a
plain residual add with no Sinkhorn overhead.

Historical note: A second implementation (`MHCBlock`) existed that used
`hc_split_sinkhorn` from `kernel.py` with a different parametrisation.
It was never wired into the active `LasmoidBlock` path and was removed
during the hardening effort (2025-07-13, task 5.2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._common import Linear, RMSNorm
except ImportError:
    from _common import Linear, RMSNorm


class ManifoldConstrainedHyperConnection(nn.Module):
    """Sinkhorn-projected doubly stochastic mixing of parallel residual streams.

    Args:
        dim: hidden dimension per stream.
        n_hc: number of parallel residual streams (hyper-connection multiplicity).
        sinkhorn_iters: number of row/column normalisation iterations for the
            Sinkhorn-Knopp projection (read from config via ``hc_sinkhorn_iters``).

    When ``n_hc == 1`` the module becomes a no-op identity (plain residual add):
    A_l = 1, B_l = [[1]], C_l = 1, so the caller's ``B_l @ streams + C_l * output``
    reduces to ``streams + output``.
    """

    def __init__(self, dim: int, n_hc: int, sinkhorn_iters: int = 20):
        super().__init__()
        self.n_hc = n_hc
        self.dim = dim
        self.sinkhorn_iters = sinkhorn_iters

        # Single-stream shortcut: no learnable parameters needed — acts as
        # identity gating (plain residual add).
        if n_hc == 1:
            # Register dummy buffers so forward() can return correct shapes
            # without any learnable overhead.
            return

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

        # ── Single-stream fast path: plain residual identity ────────────
        if self.n_hc == 1:
            # A_l = 1 (pre-gate), B_l = identity 1x1, C_l = 1 (post-gate)
            ones_gate = torch.ones(B, S, 1, 1, device=x.device, dtype=x.dtype)
            eye_mix = torch.ones(B, S, 1, 1, device=x.device, dtype=x.dtype)
            return ones_gate, eye_mix, ones_gate

        # ── Multi-stream path: full Sinkhorn mixing ─────────────────────
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

        # Sinkhorn-Knopp on exp(b_raw) for configured iterations in float32
        # for numerical stability
        M = torch.exp(b_raw.float())
        for _ in range(self.sinkhorn_iters):
            M = F.normalize(M, p=1, dim=1)  # Column norm
            M = F.normalize(M, p=1, dim=2)  # Row norm
        B_l = M.view(B, S, self.n_hc, self.n_hc).to(
            x_flat.dtype
        )  # Projected onto Birkhoff polytope

        return A_l, B_l, C_l
