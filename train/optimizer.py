"""
Lasmoid — optimizer.py
======================
Muon (Newton-Schulz) optimizer and LR scheduler.
"""

import math
import torch


class Muon(torch.optim.Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalisation for 2D weights."""

    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad, state = p.grad, self.state[p]
                if not torch.isfinite(grad).all():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if len(p.shape) == 2:
                    G = buf.clone()
                    transposed = G.shape[0] > G.shape[1]
                    if transposed:
                        G = G.T
                    a, b, c = 3.4445, -4.7750, 2.0315
                    X = G / (G.norm() + 1e-8)
                    for _ in range(5):
                        A = X @ X.T
                        B = A @ X
                        X = a * X + b * B + c * A @ B
                    update = X * (max(p.shape[0], p.shape[1]) ** 0.5)
                    if transposed:
                        update = update.T
                else:
                    update = buf
                p.add_(update, alpha=-lr)


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Cosine LR schedule with linear warmup. Returns multiplier in [0, 1]."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
