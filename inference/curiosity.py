"""
Lasmoid — curiosity.py
======================================================================
CuriosityExpert: a special "questioning" expert that learns the way a human
does — by noticing when new information does NOT follow from what it already
knows, and asking bridging questions ("I understand classical physics; why is
quantum needed? oh — because measurement is probabilistic ...").

Concept & math (Intrinsic Curiosity Module, Pathak et al. 2017):
  • A small *forward model* g predicts the next concept from the current one:
        pred_t = g(h_t)
  • The prediction error is the *curiosity / surprise* signal:
        curiosity_t = || pred_t - sg(h_{t+1}) ||²        (sg = stop-grad)
    High where new content is NOT predictable from prior knowledge — exactly
    the points worth questioning.
  • The forward-model loss  L_fwd = mean(curiosity)  trains g to predict
    (i.e. to *learn the concept*); the curiosity magnitude is exposed as an
    intrinsic signal usable for reweighting / RL reward / data selection.

Questioning mechanism:
  • From the (surprising) state we form K "question" queries and attend them to
    the concept-memory bank to retrieve the most relevant prior knowledge to
    bridge to. The retrieved bridge is added back to the representation.

Output projection is zero-initialised → identity at init (backward compatible).
Config-gated by ``use_curiosity_expert``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._common import Linear, RMSNorm
except ImportError:
    from _common import Linear, RMSNorm


class CuriosityExpert(nn.Module):
    """Forward-model curiosity + concept-memory bridging questions.

    forward(h, concept_db=None) -> (refined_h, curiosity, fwd_loss)
        h:          [B, S, dim] current hidden states
        concept_db: [B, M, dim] prior-knowledge slots to bridge to (optional)
        curiosity:  [B, S] per-token surprise (detached)
        fwd_loss:   scalar forward-model prediction loss (trainable signal)
    """

    def __init__(self, dim: int, n_questions: int = 4, n_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.n_questions = n_questions
        self.n_heads = n_heads
        self.eps = eps

        # Forward model g: predicts next concept from current state.
        self.fwd_norm = RMSNorm(dim, eps)
        self.forward_model = nn.Sequential(Linear(dim, dim * 2), nn.GELU(), Linear(dim * 2, dim))

        # Question generator: K query vectors per token.
        self.q_norm = RMSNorm(dim, eps)
        self.q_proj = Linear(dim, n_questions * dim)

        # Bridge read-back (zero-init → identity at start).
        self.out = Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

        self.last_curiosity = torch.tensor(0.0)
        self.last_fwd_loss = torch.tensor(0.0)

    def forward(
        self, h: torch.Tensor, concept_db: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = h.shape

        # ── Curiosity = forward-model prediction error (ICM) ──
        pred = self.forward_model(self.fwd_norm(h))  # [B, S, D]
        if S > 1:
            target = h[:, 1:, :].detach()
            err = ((pred[:, :-1, :] - target) ** 2).mean(dim=-1)  # [B, S-1]
            curiosity = F.pad(err, (0, 1), value=0.0)  # align to [B, S]
        else:
            curiosity = torch.zeros(B, S, device=h.device, dtype=h.dtype)
        fwd_loss = curiosity.mean()

        # ── Questioning: form K queries, bridge to prior knowledge ──
        q = self.q_proj(self.q_norm(h)).view(B, S, self.n_questions, D)
        if concept_db is not None and concept_db.shape[1] > 0:
            # Multi-head attention of questions over concept memory.
            M = concept_db.shape[1]
            hd = D // self.n_heads
            cdb = concept_db.to(q.dtype)
            qh = q.reshape(B, S * self.n_questions, self.n_heads, hd).transpose(1, 2)
            kv = cdb.reshape(B, M, self.n_heads, hd).transpose(1, 2)
            ans = F.scaled_dot_product_attention(qh, kv, kv)  # [B, heads, S*K, hd]
            ans = ans.transpose(1, 2).reshape(B, S, self.n_questions, D)
            bridge = ans.mean(dim=2)  # [B, S, D]
        else:
            bridge = q.mean(dim=2)

        refined = h + self.out(bridge)  # zero-init → == h at start

        self.last_curiosity = curiosity.mean().detach()
        self.last_fwd_loss = fwd_loss.detach()
        return refined, curiosity.detach(), fwd_loss


__all__ = ["CuriosityExpert"]
