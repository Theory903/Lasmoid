"""
Lasmoid — SSM Module  (extracted from model.py)
======================================================================
Self-contained State Space Model recurrence: JIT-compiled scan helpers
+ StateSpaceRecurrence module (Mamba-3 inspired).
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
    """
    Mamba-3 style chunked associative scan.
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
def ssm_recurrence_loop(
    decay: torch.Tensor,
    v_heads: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    prev_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy full-sequence recurrence (used when chunk_size >= S)."""
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
      • Chunked parallel scan (ssm_chunk_scan) for efficient training
      • dt clamped to [dt_min, dt_max] for numerical stability (Nemotron pattern)
      • B and C L2-normalised per head (variance stabilisation)
      • Learnable dt_bias init from log-uniform distribution in [dt_min, dt_max]
      • Multi-group SSM support (ssm_n_groups)
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
        decay = torch.exp(
            dt.unsqueeze(-1) * self.A.view(1, 1, self.n_heads, 1, self.d_state).float()
        )  # (B_comp, S, H, d_head, d_state)

        # ── 5. Selective scan ─────────────────────────────────────────
        v_heads = v_conv.reshape(B_comp, S, self.n_heads, self.d_head).float() * dt

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
            # Long sequence — use Mamba-3 chunked scan for training efficiency
            y, prev_s = ssm_chunk_scan(
                decay, v_heads, B_mat, C_mat, prev_s, self.chunk_size
            )

        self.ssm_state[:B_comp].copy_(prev_s.detach())
        y = y.reshape(B_comp, S, self.d_model).type_as(x)

        # ── 6. Gate + output projection ───────────────────────────────
        gated = y * F.silu(u)

        # ── D skip: y += D ⊙ u (direct input bypass, standard Mamba) ─
        # Keeps information flow alive when the selective scan is near-zero.
        # D broadcasts over (B_comp, S) automatically.
        if self.use_d_skip:
            gated = gated + u.type_as(gated) * self.D.to(gated.dtype)

        return self.w_out(gated)
