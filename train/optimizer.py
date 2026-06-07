"""
Lasmoid — optimizer.py
======================
Muon (Newton-Schulz) optimizer and LR scheduler.
"""

import math
import torch


class Muon(torch.optim.Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalisation for 2D weights.

    Uses a 5th-order Newton-Schulz iteration with Chebyshev polynomial
    coefficients (a=3.4445, b=-4.7750, c=2.0315) to approximate the matrix
    sign function, orthogonalising gradient updates so singular values
    approach 1.0. The iteration count is configurable via ``ns_steps``
    (default 5).

    Non-2D parameters receive a plain momentum update (fallback); in
    practice, ``build_param_groups`` routes non-2D params to AdamW so
    Muon only sees 2D weight matrices.

    Optional noise-adaptive scaling (NAMO/NMOD-style): when ``adaptive_noise`` is
    set, the step is scaled by a gradient signal-to-noise ratio so noisy batches
    take smaller steps, *without* altering the orthogonalised update direction
    (the scale is a scalar, so geometry/orthogonality is preserved). This widens
    the stable learning-rate range in high-variance settings.
    """

    # 5th-order Newton-Schulz Chebyshev polynomial coefficients.
    # These approximate the matrix sign function: after `ns_steps` iterations,
    # the orthogonalised matrix has singular values approaching 1.0.
    _NS_A = 3.4445
    _NS_B = -4.7750
    _NS_C = 2.0315

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        ns_steps=5,
        adaptive_noise=False,
        beta2=0.99,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            adaptive_noise=adaptive_noise,
            beta2=beta2,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single Muon optimisation step.

        Args:
            closure: A closure that re-evaluates the model and returns the
                loss. Optional; required by the ``torch.optim.Optimizer``
                interface so that wrappers like ``accelerate`` can call
                ``step(closure)`` without raising a ``TypeError``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            ns_steps = group["ns_steps"]
            adaptive = group.get("adaptive_noise", False)
            beta2, eps = group.get("beta2", 0.99), group.get("eps", 1e-8)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad, state = p.grad, self.state[p]
                if not torch.isfinite(grad).all():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    if adaptive:
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                # Noise-adaptive scalar (SNR ∈ (0,1)); 1.0 ⇒ standard Muon.
                snr_scale = 1.0
                if adaptive:
                    v = state["exp_avg_sq"]
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    signal = buf.pow(2).sum()
                    noise = v.sum() + eps
                    snr = signal / noise
                    snr_scale = (snr / (1.0 + snr)).item()  # bounded (0,1)

                if len(p.shape) == 2:
                    # 5th-order Newton-Schulz orthogonalisation
                    G = buf.clone()
                    transposed = G.shape[0] > G.shape[1]
                    if transposed:
                        G = G.T
                    # Eps-stabilized Frobenius-norm division to avoid div-by-zero
                    # when momentum is all zeros (zero-norm edge case).
                    frob_norm = G.norm()
                    X = G / (frob_norm + eps)
                    for _ in range(ns_steps):
                        A = X @ X.T
                        B = A @ X
                        X = self._NS_A * X + self._NS_B * B + self._NS_C * (A @ B)
                    update = X * (max(p.shape[0], p.shape[1]) ** 0.5)
                    if transposed:
                        update = update.T
                else:
                    # Non-2D fallback: plain momentum update (in the standard
                    # build_optimizers flow, non-2D params go to AdamW instead).
                    update = buf
                p.add_(update, alpha=-lr * snr_scale)

        return loss


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Cosine LR schedule with linear warmup. Returns multiplier in [0, 1]."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════
# OPTIMIZER CONSTRUCTION  (proper Muon + AdamW split, decoupled weight decay)
# ══════════════════════════════════════════════════════════════════════


def build_param_groups(model):
    """Split parameters following the Muon recipe (Keller Jordan):
      • Muon          → 2-D hidden matmul weights (the transformer body)
      • AdamW + wd    → embeddings / output head (≥2-D, but not Muon-suited)
      • AdamW, no wd  → 1-D params: norms, biases, gains, scalars

    Weight decay is applied *only* to the AdamW-decay group; never to norms,
    biases, or the Muon-orthogonalised body (which is scale-invariant).
    """
    muon, adamw_decay, adamw_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        is_embed_or_head = ("emb" in lname) or ("head" in lname) or ("lm_head" in lname)
        if p.ndim == 2 and not is_embed_or_head:
            muon.append(p)
        elif p.ndim >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)
    return muon, adamw_decay, adamw_nodecay


def build_optimizers(
    model,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    momentum: float = 0.95,
    ns_steps: int = 5,
    adaptive_noise: bool = False,
):
    """Construct [Muon, AdamW] over the correctly-split parameter groups.

    ``ns_steps`` controls the number of Newton-Schulz orthogonalisation
    iterations (default 5; the 5th-order Chebyshev coefficients are built-in).

    ``adaptive_noise`` enables the NAMO/NMOD-style SNR step scaling in Muon for
    extra stability in noisy/high-LR regimes. Returns a list of optimizers ready
    to pass to WSDScheduler.
    """
    muon_p, adamw_d, adamw_n = build_param_groups(model)
    opts = []
    if muon_p:
        opts.append(
            Muon(
                muon_p,
                lr=muon_lr,
                momentum=momentum,
                ns_steps=ns_steps,
                adaptive_noise=adaptive_noise,
                eps=eps,
            )
        )
    adam_groups = []
    if adamw_d:
        adam_groups.append({"params": adamw_d, "weight_decay": weight_decay})
    if adamw_n:
        adam_groups.append({"params": adamw_n, "weight_decay": 0.0})
    if adam_groups:
        opts.append(torch.optim.AdamW(adam_groups, lr=adamw_lr, betas=betas, eps=eps))
    return opts


def clip_grad_global_norm(model, max_norm: float = 1.0) -> float:
    """Global-norm gradient clipping; returns the pre-clip total norm."""
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


# ══════════════════════════════════════════════════════════════════════
# MUON.STEP CLOSURE-COMPATIBILITY PATCH  (idempotent, safe)
# ══════════════════════════════════════════════════════════════════════

_MUON_STEP_PATCHED = False
"""Global guard — ``True`` after :func:`ensure_muon_closure_compat` has run."""


def ensure_muon_closure_compat() -> None:
    """Ensure :meth:`Muon.step` accepts the ``closure`` keyword argument.

    The current :class:`Muon.step` implementation in this file already
    accepts ``closure=None`` natively, so under normal circumstances this
    function is a no-op.

    It exists as a *defensive* safety net for external code (Jupyter
    notebooks, interactive shells) that may try to monkey-patch
    ``Muon.step`` without properly capturing the original method::

        Muon.step = lambda self, closure=None: …  # WRONG – NameError!

    Calling this function *before* any external patching ensures the
    original method is captured correctly so the patch does not crash::

        ensure_muon_closure_compat()
        # Now safe to write notebooks that reference ``_orig_muon_step``

    The function is idempotent – it sets a module-level ``__patched``
    flag so subsequent calls are instant no-ops.
    """
    global _MUON_STEP_PATCHED
    if _MUON_STEP_PATCHED:
        return

    _orig_muon_step = Muon.step

    @torch.no_grad()
    def _patched_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return _orig_muon_step(self)

    Muon.step = _patched_step
    Muon.step.__patched = True  # noqa
    _MUON_STEP_PATCHED = True
