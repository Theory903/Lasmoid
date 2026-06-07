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
        self.mtp_fused_norm = getattr(args, "mtp_fused_norm", False)
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        if not self.mtp_fused_norm:
            # Legacy: separate pre-norms on each stream before projection
            self.enorm = RMSNorm(args.dim, args.norm_eps)
            self.hnorm = RMSNorm(args.dim, args.norm_eps)
        # Fusion norm: applied to projected sum per ARCHITECTURE.md §H
        self.fusion_norm = RMSNorm(args.dim, args.norm_eps)
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
        emb = self.embed(input_ids).to(x.dtype)

        B_h, S_h, hc_h, D_h = x.shape

        if self.mtp_fused_norm:
            # Spec-correct: project first, sum, then RMSNorm on the fused result
            # h_fused = RMSNorm(W_e * e_{t+1} + W_h * h_t)  [ARCHITECTURE.md §H]
            e_proj = self.e_proj(emb)  # [B, S, D]
            h_flat = x.reshape(B_h * S_h * hc_h, D_h)
            h_proj = self.h_proj(h_flat).reshape(B_h, S_h, hc_h, D_h)
            fused = e_proj.unsqueeze(2) + h_proj  # [B, S, HC, D]
            # Apply fusion norm per-element (flatten → norm → reshape)
            fused_flat = fused.reshape(-1, D_h)
            fused_flat = self.fusion_norm(fused_flat)
            x_out = fused_flat.reshape(B_h, S_h, hc_h, D_h)
        else:
            # Legacy: pre-norm each stream separately before projection
            e = self.enorm(emb)
            h_flat = x.reshape(B_h * S_h * hc_h, D_h)
            h_flat = self.hnorm(h_flat)
            h_flat = self.h_proj(h_flat)
            h = h_flat.reshape(B_h, S_h, hc_h, D_h)
            x_out = self.e_proj(e).unsqueeze(2) + h

        x_out, z_loss, vq_loss, routing, indices, adj, _ = self.block(
            x_out, freqs_cis, start_pos, input_ids
        )

        y = self.hc_head_reduce(x_out)
        y = self.norm(y)
        logits = F.linear(y.float(), self.head.weight.float())
        return logits, z_loss, vq_loss


# ══════════════════════════════════════════════════════════════════════
# SPECULATIVE DECODING ENGINE  (draft → verify → accept, DL=6)
# ══════════════════════════════════════════════════════════════════════


class SpeculativeDecoder:
    """Multi-Token-Prediction speculative decoding.

    Uses the lightweight MTP head as a *draft* model to propose ``draft_length``
    future tokens cheaply (a single block per step), then *verifies* them with a
    single batched pass of the full model.  For greedy decoding this yields output
    that is **identical** to standard greedy decoding (only faster), per the
    standard speculative-sampling acceptance rule:

        accept draft d_i  iff  argmax(target_logits_{i-1}) == d_i

    On the first rejected position the target model's own argmax token is emitted
    as a correction; if every draft is accepted a bonus token is sampled from the
    final target distribution.

    Acceptance rate is tracked; if it stays below ``accept_floor`` for
    ``fallback_patience`` consecutive rounds, callers may fall back to standard
    decoding.
    """

    def __init__(
        self,
        model: nn.Module,
        draft_length: int = 6,
        accept_floor: float = 0.3,
        fallback_patience: int = 10,
    ):
        self.model = model
        args = model.args
        self.draft_length = int(getattr(args, "mtp_draft_length", draft_length))
        self.accept_floor = accept_floor
        self.fallback_patience = fallback_patience
        self.mtp_block: Optional[MTPBlock] = (
            model.mtp[0] if getattr(model, "mtp", None) and len(model.mtp) > 0 else None
        )
        # Draft head shares weights with the model's tied output head.
        self.head_weight = model.head.weight
        self._low_accept_streak = 0

    # ── draft ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def draft_tokens(
        self,
        streams: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_ids: torch.Tensor,
        start_pos: int,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedily propose ``k`` draft token ids using the cheap MTP head.

        ``streams`` is the current residual-stream tensor [B, S, HC, D]; only the
        last position is used as the drafting anchor.  Returns LongTensor [B, k].
        """
        k = k or self.draft_length
        assert self.mtp_block is not None, "SpeculativeDecoder requires at least one MTP block"
        B = streams.shape[0]
        device = streams.device

        anchor = streams[:, -1:].contiguous()  # [B, 1, HC, D]
        cur_ids = input_ids[:, -1:].contiguous()  # [B, 1]
        drafts = []
        for i in range(k):
            f = freqs_cis[i : i + 1] if freqs_cis.shape[0] > i else freqs_cis[-1:]
            logits, _, _ = self.mtp_block(anchor, f, cur_ids, start_pos + i)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            drafts.append(next_id)
            cur_ids = next_id
        return torch.cat(drafts, dim=1)  # [B, k]

    # ── verify ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def verify_drafts(
        self, draft_ids: torch.Tensor, target_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Accept the longest correct greedy prefix of the drafted tokens.

        Args:
            draft_ids:    [B, k] proposed token ids.
            target_logits:[B, k+1, vocab] full-model logits where position ``i``
                          predicts the token that should follow ``draft_ids[:, i-1]``
                          (position 0 predicts the first draft token).
        Returns:
            (accepted_ids [B, n_accepted+1], n_accepted)
            The returned sequence always includes one extra (corrected/bonus) token.
        """
        B, k = draft_ids.shape
        target_greedy = target_logits.argmax(dim=-1)  # [B, k+1]

        # Operate batch-wise on the common accepted prefix length.
        match = (target_greedy[:, :k] == draft_ids)  # [B, k]
        # First mismatch position per batch row (k if all match).
        n_accepted = k
        for i in range(k):
            if not bool(match[:, i].all()):
                n_accepted = i
                break

        accepted = draft_ids[:, :n_accepted]
        # Correction / bonus token comes from the target distribution at the
        # first non-accepted slot.
        bonus = target_greedy[:, n_accepted : n_accepted + 1]
        out = torch.cat([accepted, bonus], dim=1)
        return out, n_accepted

    @property
    def low_acceptance(self) -> bool:
        return self._low_accept_streak >= self.fallback_patience

    def _update_acceptance(self, n_accepted: int, k: int) -> None:
        if k > 0 and (n_accepted / k) < self.accept_floor:
            self._low_accept_streak += 1
        else:
            self._low_accept_streak = 0
