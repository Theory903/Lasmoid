"""
Lasmoid — scheduler.py
======================
Warmup-Stable-Decay (WSD) learning rate scheduler.
Outputs a multiplier in [0.0, 1.0] to scale the base learning rate.
"""

import math


def get_wsd_lr_multiplier(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    """
    Computes a learning rate multiplier using the WSD schedule:
      1. Warmup: Linearly increases from 0.0 to 1.0 (2% default)
      2. Stable: Constant at 1.0 (90% default)
      3. Decay: Cosine decay from 1.0 to min_lr_ratio (8% default)
    """
    total_steps = warmup_steps + stable_steps + decay_steps
    if step >= total_steps:
        return min_lr_ratio

    if step < warmup_steps:
        # Linear Warmup
        return float(step) / float(max(1, warmup_steps))

    if step < warmup_steps + stable_steps:
        # Stable phase
        return 1.0

    # Decay phase: Cosine decay
    decay_step = step - (warmup_steps + stable_steps)
    progress = float(decay_step) / float(max(1, decay_steps))
    cosine_out = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_out


class WSDScheduler:
    """
    Stateful wrapper for WSD learning rate schedule.
    Adjusts parameter group learning rates in one or more optimizers.
    """

    def __init__(
        self,
        optimizers,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        base_lrs,
        min_lr_ratio: float = 0.0,
    ):
        """
        Args:
            optimizers: Single optimizer or list of optimizers (e.g. [Muon, AdamW]).
            warmup_steps: Number of steps for linear warmup.
            stable_steps: Number of steps for constant peak LR.
            decay_steps: Number of steps for cosine decay.
            base_lrs: Dict or list of base learning rates corresponding to parameter groups.
            min_lr_ratio: Minimum learning rate as a fraction of peak learning rate.
        """
        self.optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio

        # Store base learning rates for each parameter group in each optimizer
        self.base_lrs = []
        if isinstance(base_lrs, list):
            self.base_lrs = base_lrs
        else:
            # If a single float or dict/list is passed, extract it from optimizers' initial param_groups
            for opt in self.optimizers:
                opt_lrs = [group["lr"] for group in opt.param_groups]
                self.base_lrs.append(opt_lrs)

    def step(self, step: int) -> float:
        """Update optimizer learning rates and return current multiplier."""
        mult = get_wsd_lr_multiplier(
            step,
            self.warmup_steps,
            self.stable_steps,
            self.decay_steps,
            self.min_lr_ratio,
        )

        for i, opt in enumerate(self.optimizers):
            opt_base = self.base_lrs[i]
            for j, group in enumerate(opt.param_groups):
                group["lr"] = opt_base[j] * mult

        return mult
