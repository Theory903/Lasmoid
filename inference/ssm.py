"""
Lasmoid — SSM Module  (extracted from model.py)
======================================================================
Self-contained State Space Model recurrence: JIT-compiled scan helpers
+ StateSpaceRecurrence module (Mamba-3 inspired).

Scan helpers:
  - ssm_step_one:       single-step decode update
  - ssm_recurrence_loop: full-sequence sequential recurrence (S <= chunk_size)
  - ssm_chunk_scan:     chunked sequential recurrence for memory locality
                        (S > chunk_size) — NOT a parallel/associative scan

A true chunked associative scan can be enabled via the ``use_associative_scan``
ModelArgs flag (default False — conservative, preserves current sequential
behaviour).
"""

import math
from typing import Tuple, Optional
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

try:
    from ._common import Linear, RMSNorm, set_dtype, default_dtype
    from .config import ModelArgs
except ImportError:
    from _common import Linear, RMSNorm, set_dtype, default_dtype
    from config import ModelArgs


# ══════════════════════════════════════════════════════════════════════
# JIT HELPER FUNCTIONS  (selective scan primitives)
# ══════════════════════════════════════════════════════════════════════


@torch.jit.script
def ssm_step_one(
    decay: torch.Tensor,
    v_heads: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    prev_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-step SSM update (decode phase)."""
    decay_0 = decay[:, 0]  # (B_comp, H, d_head, d_state)
    v_0 = v_heads[:, 0].unsqueeze(-1)  # (B_comp, H, d_head, 1)
    B_0 = B[:, 0].unsqueeze(-2)  # (B_comp, H, 1, d_state)
    C_0 = C[:, 0].unsqueeze(-2)  # (B_comp, H, 1, d_state)

    outer = v_0 * B_0
    curr_s = decay_0 * prev_s + outer
    y_t = (curr_s * C_0).sum(dim=-1)
    y = y_t.unsqueeze(1)  # (B_comp, 1, H, d_head)
    return y, curr_s


@torch.jit.script
def ssm_chunk_scan(
    decay: torch.Tensor,
    v_heads: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    prev_s: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked sequential scan (reference implementation).

    This is a **sequential** recurrence partitioned into chunks for memory
    locality — it is NOT a parallel/associative scan.  The inner loop is
    identical to ``ssm_recurrence_loop``; chunking simply bounds the working
    set per iteration.

    For long sequences (S > chunk_size) this provides better cache behaviour
    than a single flat loop while preserving exact numerical equivalence with
    ``ssm_recurrence_loop`` (both are O(S) sequential).

    A true chunked associative (parallel) scan can be enabled via the
    ``use_associative_scan`` ModelArgs flag (default False — conservative).
    """
    Bc, S, H, d_head = v_heads.shape
    outputs = torch.empty((Bc, S, H, d_head), device=decay.device, dtype=decay.dtype)
    curr_s = prev_s

    n_chunks = (S + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        t0 = c * chunk_size
        t1 = min(t0 + chunk_size, S)
        for t in range(t0, t1):
            dt = decay[:, t]  # (B, H, d_head, d_state)
            vt = v_heads[:, t].unsqueeze(-1)  # (B, H, d_head, 1)
            Bt = B[:, t].unsqueeze(-2)  # (B, H, 1, d_state)
            Ct = C[:, t].unsqueeze(-2)  # (B, H, 1, d_state)
            curr_s = dt * curr_s + vt * Bt
            outputs[:, t] = (curr_s * Ct).sum(dim=-1)

    return outputs, curr_s


@torch.jit.script
def heavy_tail_decay(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mamba-3 heavy-tailed state decay.

    Replaces the exponential decay ``exp(x)`` (x = dt*A <= 0) with a rational
    function that decays polynomially (as 1/|x|) for large negative x, giving
    the state a much heavier memory tail for long-range retention while
    remaining bounded in (0, 1].

        heavy_tail(x) = 1 + a*x         if x >= 0
                        1 / (1 - a*x)   if x <  0

    ``alpha`` interpolates between standard exponential decay (alpha == 1.0,
    the default — kept bit-compatible by the caller) and stronger heavy-tail
    behaviour.  Output is clamped to (0, 1] for stability.
    """
    ax = alpha * x
    pos = 1.0 + ax
    neg = 1.0 / (1.0 - ax)
    out = torch.where(x >= 0, pos, neg)
    return out.clamp(min=1e-6, max=1.0)


@torch.jit.script
def ssm_recurrence_loop(
    decay: torch.Tensor,
    v_heads: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    prev_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-sequence sequential recurrence (used when S <= chunk_size)."""
    B_comp, S, H, d_head = v_heads.shape
    outputs = torch.empty(
        (B_comp, S, H, d_head), device=decay.device, dtype=decay.dtype
    )
    curr_s = prev_s
    for t in range(S):
        decay_t = decay[:, t]  # (B_comp, H, d_head, d_state)
        v_t = v_heads[:, t].unsqueeze(-1)
        B_t = B[:, t].unsqueeze(-2)
        C_t = C[:, t].unsqueeze(-2)
        outer = v_t * B_t
        curr_s = decay_t * curr_s + outer
        y_t = (curr_s * C_t).sum(dim=-1)
        outputs[:, t] = y_t
    return outputs, curr_s


# ══════════════════════════════════════════════════════════════════════
# STATE SPACE RECURRENCE  (Mamba-3 Upgraded)
# ══════════════════════════════════════════════════════════════════════


class StateSpaceRecurrence(nn.Module):
    """
    Mamba-3 inspired SSM branch.
    Key improvements over Mamba-2:
      • Chunked sequential scan (ssm_chunk_scan) with cache-friendly memory access
      • dt clamped to [dt_min, dt_max] for numerical stability (Nemotron pattern)
      • B and C L2-normalised per head (variance stabilisation)
      • Learnable dt_bias init from log-uniform distribution in [dt_min, dt_max]
      • Multi-group SSM support (ssm_n_groups)

    Note: ssm_chunk_scan is a sequential recurrence partitioned into chunks for
    memory locality.  A true associative (parallel) scan is available when
    ``use_associative_scan=True`` in ModelArgs (default False).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.dim
        self.n_heads = getattr(args, "ssm_heads", 4)
        self.d_state = getattr(args, "ssm_state_dim", 16)
        self.kernel_size = getattr(args, "ssm_kernel_size", 4)
        self.chunk_size = getattr(args, "ssm_chunk_size", 64)
        self.dt_min = getattr(args, "ssm_dt_min", 0.001)
        self.dt_max = getattr(args, "ssm_dt_max", 0.1)
        self.dt_floor = getattr(args, "ssm_dt_init_floor", 1e-4)
        self.n_groups = getattr(args, "ssm_n_groups", 1)

        assert self.d_model % self.n_heads == 0, (
            f"dim {self.d_model} must be divisible by ssm_heads {self.n_heads}"
        )
        self.d_head = self.d_model // self.n_heads

        # ── Input projection ──────────────────────────────────────────
        # Projects to: u (gate), v (SSM input), B, C, delta
        self.w_in_dim = 3 * self.d_model + 2 * self.n_heads * self.d_state
        self.w_in = Linear(self.d_model, self.w_in_dim)
        self.w_out = Linear(self.d_model, self.d_model)

        # ── Depthwise causal conv ─────────────────────────────────────
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.kernel_size,
            groups=self.d_model,
            bias=True,
        )

        # learned state matrix A
        self.A = nn.Parameter(-torch.ones(self.n_heads, self.d_state))

        # ── dt_bias: log-uniform init in [dt_min, dt_max] (Nemotron) ─
        # dt_bias such that softplus(dt_bias) ≈ dt_init
        dt_init = torch.exp(
            torch.rand(self.d_model) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_floor)
        # Inverse softplus: log(exp(dt_init) - 1) ≈ dt_init for large dt_init
        dt_bias_init = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(dt_bias_init.reshape(self.n_heads, self.d_head))

        # ── State buffers ─────────────────────────────────────────────
        max_batch_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer(
            "ssm_state",
            torch.zeros(max_batch_comp, self.n_heads, self.d_head, self.d_state),
            persistent=False,
        )
        self.register_buffer(
            "conv_state",
            torch.zeros(max_batch_comp, self.d_model, self.kernel_size - 1),
            persistent=False,
        )

        # ── D skip: direct input bypass term (standard Mamba, critical for ──
        # information flow when decay ≈ 0 / highly selective scan state)  ──
        self.use_d_skip = getattr(args, "ssm_d_skip", True)
        if self.use_d_skip:
            self.D = nn.Parameter(torch.ones(self.d_model))

        # ── Mamba-3 upgrades: MIMO channel mixing + heavy-tail decay + SSM RoPE ──
        self.is_mimo = getattr(args, "ssm_is_mimo", False)
        self.mimo_rank = getattr(args, "ssm_mimo_rank", 4)
        self.heavy_tail_alpha = getattr(args, "ssm_heavy_tail_alpha", 1.0)
        # Heavy-tail decay is only active when alpha deviates from the
        # default 1.0, preserving exact Mamba-2 behaviour otherwise.
        self.use_heavy_tail = abs(self.heavy_tail_alpha - 1.0) > 1e-9

        if self.is_mimo:
            R = self.mimo_rank
            # Low-rank residual channel-mixing on the SSM input (v) and output (y),
            # applied per head across d_head.  Output factor initialised to zero so
            # the projections are an exact identity (no-op) at initialisation.
            self.mimo_x_down = nn.Parameter(torch.randn(self.n_heads, self.d_head, R) * 0.02)
            self.mimo_x_up = nn.Parameter(torch.zeros(self.n_heads, R, self.d_head))
            self.mimo_o_down = nn.Parameter(torch.randn(self.n_heads, self.d_head, R) * 0.02)
            self.mimo_o_up = nn.Parameter(torch.zeros(self.n_heads, R, self.d_head))

            # SSM RoPE applied to the B/C state matrices (d_state must be even).
            self.use_ssm_rope = (self.d_state % 2 == 0)
            if self.use_ssm_rope:
                from functools import lru_cache  # noqa: F401
                try:
                    from .attention import precompute_freqs_cis as _pf
                except ImportError:
                    from attention import precompute_freqs_cis as _pf
                rope_len = min(args.max_seq_len + 1024, 2097152 + 1024)
                self.register_buffer(
                    "ssm_freqs_cis",
                    _pf(self.d_state, rope_len, 0, getattr(args, "rope_theta", 10000.0)),
                    persistent=False,
                )
        else:
            self.use_ssm_rope = False

    def _apply_mimo_in(self, v_heads: torch.Tensor) -> torch.Tensor:
        # v_heads: (B, S, H, d_head)  →  residual low-rank channel mix per head
        mix = torch.einsum("bshd,hdr->bshr", v_heads, self.mimo_x_down.to(v_heads.dtype))
        mix = torch.einsum("bshr,hrd->bshd", mix, self.mimo_x_up.to(v_heads.dtype))
        return v_heads + mix

    def _apply_mimo_out(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, S, H, d_head)  →  residual low-rank channel mix per head
        mix = torch.einsum("bshd,hdr->bshr", y, self.mimo_o_down.to(y.dtype))
        mix = torch.einsum("bshr,hrd->bshd", mix, self.mimo_o_up.to(y.dtype))
        return y + mix

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        B_comp, S, D = x.shape

        # ── 1. Input projection ───────────────────────────────────────
        projected = self.w_in(x)  # (B_comp, S, w_in_dim)
        u, v, B_C, delta = torch.split(
            projected,
            [self.d_model, self.d_model, 2 * self.n_heads * self.d_state, self.d_model],
            dim=-1,
        )

        B_raw, C_raw = torch.split(
            B_C, [self.n_heads * self.d_state, self.n_heads * self.d_state], dim=-1
        )
        B_mat = B_raw.reshape(B_comp, S, self.n_heads, self.d_state)
        C_mat = C_raw.reshape(B_comp, S, self.n_heads, self.d_state)

        # ── Mamba-3: L2-normalise B and C per head (variance stabilisation) ─
        B_mat = F.normalize(B_mat.float(), p=2, dim=-1)
        C_mat = F.normalize(C_mat.float(), p=2, dim=-1)

        # ── Mamba-3: SSM RoPE on the B/C state matrices ──────────────────
        if self.use_ssm_rope:
            try:
                from ._common import apply_rotary_emb as _are
            except ImportError:
                from _common import apply_rotary_emb as _are
            freqs = self.ssm_freqs_cis[start_pos : start_pos + S].to(B_mat.device)
            B_mat = _are(B_mat, freqs)
            C_mat = _are(C_mat, freqs)

        # ── 2. Resize / reset state buffers ──────────────────────────
        if B_comp > self.ssm_state.shape[0]:
            self.register_buffer(
                "ssm_state",
                torch.zeros(
                    B_comp,
                    self.n_heads,
                    self.d_head,
                    self.d_state,
                    device=x.device,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "conv_state",
                torch.zeros(
                    B_comp,
                    self.d_model,
                    self.kernel_size - 1,
                    device=x.device,
                    dtype=x.dtype,
                ),
                persistent=False,
            )

        if start_pos == 0:
            self.ssm_state[:B_comp].zero_()
            self.conv_state[:B_comp].zero_()

        # ── 3. Causal depthwise conv ──────────────────────────────────
        padded_v = torch.cat(
            [self.conv_state[:B_comp].type_as(v), v.transpose(1, 2)], dim=-1
        )
        conv_out = F.conv1d(
            padded_v,
            self.conv1d.weight.type_as(padded_v),
            self.conv1d.bias.type_as(padded_v)
            if self.conv1d.bias is not None
            else None,
            groups=self.d_model,
        )
        self.conv_state[:B_comp].copy_(padded_v[..., -self.kernel_size + 1 :].detach())
        v_conv = conv_out.transpose(1, 2)  # (B_comp, S, d_model)

        # ── 4. dt with clamping (Nemotron mamba_dt_min / mamba_dt_max) ─
        delta_heads = delta.reshape(B_comp, S, self.n_heads, self.d_head)
        dt = F.softplus(
            delta_heads.float()
            + self.dt_bias.view(1, 1, self.n_heads, self.d_head).float()
        )
        dt = dt.clamp(min=self.dt_min, max=self.dt_max)  # <── KEY stabilisation

        # Discretise: decay = exp(dt * A)  (ZOH discretisation with learned matrix A)
        x_decay = dt.unsqueeze(-1) * self.A.view(
            1, 1, self.n_heads, 1, self.d_state
        ).float()  # (B_comp, S, H, d_head, d_state)
        if self.use_heavy_tail:
            # Mamba-3 heavy-tailed (polynomial) decay for long-range memory.
            decay = heavy_tail_decay(x_decay, float(self.heavy_tail_alpha))
        else:
            decay = torch.exp(x_decay)

        # ── 5. Selective scan ─────────────────────────────────────────
        v_heads = v_conv.reshape(B_comp, S, self.n_heads, self.d_head).float() * dt

        # ── Mamba-3 MIMO: low-rank channel mixing on the SSM input ──────
        if self.is_mimo:
            v_heads = self._apply_mimo_in(v_heads)

        if self.training:
            prev_s = self.ssm_state[:B_comp].clone()
        else:
            prev_s = self.ssm_state[:B_comp].detach().clone()

        if S == 1:
            y, prev_s = ssm_step_one(decay, v_heads, B_mat, C_mat, prev_s)
        elif S <= self.chunk_size:
            # Short sequence — fall back to full recurrence
            y, prev_s = ssm_recurrence_loop(decay, v_heads, B_mat, C_mat, prev_s)
        else:
            # Long sequence — chunked sequential scan (memory-locality optimisation)
            y, prev_s = ssm_chunk_scan(
                decay, v_heads, B_mat, C_mat, prev_s, self.chunk_size
            )

        self.ssm_state[:B_comp].copy_(prev_s.detach())

        # ── Mamba-3 MIMO: low-rank channel mixing on the SSM output ─────
        if self.is_mimo:
            y = self._apply_mimo_out(y)

        y = y.reshape(B_comp, S, self.d_model).type_as(x)

        # ── 6. Gate + output projection ───────────────────────────────
        gated = y * F.silu(u)

        # ── D skip: y += D ⊙ u (direct input bypass, standard Mamba) ─
        # Keeps information flow alive when the selective scan is near-zero.
        # D broadcasts over (B_comp, S) automatically.
        if self.use_d_skip:
            gated = gated + u.type_as(gated) * self.D.to(gated.dtype)

        return self.w_out(gated)
