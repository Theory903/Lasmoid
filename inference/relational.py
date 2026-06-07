"""
Lasmoid — relational.py
======================================================================
RelationalCortex: AlphaFold-3 Evoformer-inspired multi-track relational
reasoning over a *small* set of concept anchors.

Motivation
----------
Scientific reasoning is relational: entities (variables, molecules, bodies,
symptoms) matter through their *pairwise* interactions and multi-hop
consistency. AlphaFold-3's Evoformer/Pairformer captures this with two
co-evolving tracks — a per-entity "single" track and a pairwise "pair" track —
coupled by:
  • OuterProductMean: single → pair (build relations from entity features)
  • TriangleMultiplication: pair → pair (enforce multi-hop / triangle consistency)
  • pair-biased single update: pair → single (read relations back into entities)

We adapt this to refine Lasmoid's concept-memory slots (a small set of K
anchors, so the O(K²) pair track and O(K³) triangle ops are cheap). The module
is config-gated (`use_relational_cortex`) and **identity at initialisation**
(final single-update projection zero-initialised), so enabling it never
destabilises a trained model.

Reference: alphafold3/src/alphafold3/model/network/modules.py
(OuterProductMean, TriangleMultiplication, EvoformerIteration).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ._common import Linear, RMSNorm
except ImportError:
    from _common import Linear, RMSNorm


class RelationalCortex(nn.Module):
    """Co-evolving single/pair tracks over K concept anchors.

    forward(single: [B, K, dim]) -> refined single [B, K, dim]
    """

    def __init__(
        self,
        dim: int,
        pair_dim: int = 16,
        opm_chan: int = 4,
        n_iters: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.pair_dim = pair_dim
        self.opm_chan = opm_chan
        self.n_iters = n_iters
        self.eps = eps

        # ── OuterProductMean: single → pair ──
        self.single_norm_opm = RMSNorm(dim, eps)
        self.opm_left = Linear(dim, opm_chan)
        self.opm_right = Linear(dim, opm_chan)
        self.opm_out = Linear(opm_chan * opm_chan, pair_dim)

        # ── TriangleMultiplication (outgoing + incoming) ──
        self.tri_norm = nn.ModuleList([RMSNorm(pair_dim, eps) for _ in range(2 * n_iters)])
        self.tri_a = nn.ModuleList([Linear(pair_dim, pair_dim) for _ in range(2 * n_iters)])
        self.tri_b = nn.ModuleList([Linear(pair_dim, pair_dim) for _ in range(2 * n_iters)])
        self.tri_ga = nn.ModuleList([Linear(pair_dim, pair_dim) for _ in range(2 * n_iters)])
        self.tri_gb = nn.ModuleList([Linear(pair_dim, pair_dim) for _ in range(2 * n_iters)])
        self.tri_out = nn.ModuleList([Linear(pair_dim, pair_dim) for _ in range(2 * n_iters)])
        self.tri_gate = nn.ModuleList([Linear(pair_dim, pair_dim) for _ in range(2 * n_iters)])

        # ── Pair transition (gated MLP) ──
        self.pair_trans_norm = nn.ModuleList([RMSNorm(pair_dim, eps) for _ in range(n_iters)])
        self.pair_trans = nn.ModuleList(
            [
                nn.Sequential(Linear(pair_dim, pair_dim * 2), nn.GELU(), Linear(pair_dim * 2, pair_dim))
                for _ in range(n_iters)
            ]
        )

        # ── pair → single read-back (zero-init → identity at start) ──
        self.pair_to_single_norm = RMSNorm(pair_dim, eps)
        self.pair_to_single = Linear(pair_dim, dim)
        nn.init.zeros_(self.pair_to_single.weight)
        if self.pair_to_single.bias is not None:
            nn.init.zeros_(self.pair_to_single.bias)

    def _outer_product_mean(self, single: torch.Tensor) -> torch.Tensor:
        """OuterProductMean: single → pair.

        Shape contract:
          single: [B, K, dim] → pair: [B, K, K, pair_dim]
        """
        s = self.single_norm_opm(single)
        left = self.opm_left(s)   # [B, K, c]
        right = self.opm_right(s)  # [B, K, c]
        outer = torch.einsum("bic,bjd->bijcd", left.float(), right.float())
        outer = outer.flatten(-2)  # [B, K, K, c*c]
        return self.opm_out(outer.to(single.dtype))

    def _triangle(self, pair: torch.Tensor, k: int, incoming: bool) -> torch.Tensor:
        """TriangleMultiplication: enforce multi-hop consistency in the pair track.

        Shape contract:
          pair: [B, K, K, pair_dim] → output: [B, K, K, pair_dim]
        """
        p = self.tri_norm[k](pair)
        a = torch.sigmoid(self.tri_ga[k](p)) * self.tri_a[k](p)
        b = torch.sigmoid(self.tri_gb[k](p)) * self.tri_b[k](p)
        if incoming:
            # consistency through incoming edges: sum over source index
            out = torch.einsum("bkic,bkjc->bijc", a.float(), b.float())
        else:
            # outgoing edges
            out = torch.einsum("bikc,bjkc->bijc", a.float(), b.float())
        out = out.to(pair.dtype)
        gate = torch.sigmoid(self.tri_gate[k](pair))
        return gate * self.tri_out[k](out)

    def forward(self, single: torch.Tensor) -> torch.Tensor:
        """Refine concept anchors through co-evolving single/pair tracks.

        Shape contract:
          single: [B, K, dim] → output: [B, K, dim]
        """
        if single.dim() != 3:
            raise ValueError("RelationalCortex expects single of shape [B, K, dim]")
        pair = self._outer_product_mean(single)  # [B, K, K, p]

        for it in range(self.n_iters):
            ko = 2 * it
            ki = 2 * it + 1
            pair = pair + self._triangle(pair, ko, incoming=False)
            pair = pair + self._triangle(pair, ki, incoming=True)
            pair = pair + self.pair_trans[it](self.pair_trans_norm[it](pair))

        # pair → single: aggregate each anchor's relations and read back.
        relational = self.pair_to_single_norm(pair.mean(dim=2))  # [B, K, p]
        single = single + self.pair_to_single(relational)        # zero-init → identity at start
        return single


__all__ = ["RelationalCortex"]
