"""
Stability system for long-running generation:
  - Drift detection (entropy, logit norm monitoring)
  - Adaptive temperature scheduling
  - KV cache integrity checks
Config-flag gated via StabilityConfig / ModelArgs.stability_enabled.
"""

from collections import deque
from enum import Enum
from typing import List, Optional

import torch
import torch.nn.functional as F

try:
    from .config import StabilityConfig
except ImportError:
    from config import StabilityConfig  # type: ignore[import]


class DriftSignal(Enum):
    ENTROPY_COLLAPSE = "entropy_collapse"
    ENTROPY_EXPLOSION = "entropy_explosion"
    LOGIT_NORM_SPIKE = "logit_norm_spike"


class DriftDetector:
    """Monitors logit statistics over a sliding window of generated tokens.

    Signals:
      - ENTROPY_COLLAPSE: max logit prob near 1.0 (model is too sure)
      - ENTROPY_EXPLOSION: near-uniform distribution (model confused)
      - LOGIT_NORM_SPIKE: ||logits|| deviates > N sigma from rolling mean
    """

    def __init__(self, config: StabilityConfig):
        self.window_size = config.drift_window_size
        self.entropy_collapse_sigma = config.entropy_collapse_threshold
        self.logit_norm_spike_sigma = config.logit_norm_spike_threshold
        self.entropy_history: deque = deque(maxlen=self.window_size)
        self.max_prob_history: deque = deque(maxlen=self.window_size)
        self.logit_norm_history: deque = deque(maxlen=self.window_size)
        self.generated_tokens = 0

    @torch.no_grad()
    def check(self, logits: torch.Tensor) -> List[DriftSignal]:
        """Analyse logits and return active drift signals."""
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        logit_norm = logits.norm(dim=-1)

        self.entropy_history.append(entropy.mean().item())
        self.max_prob_history.append(max_prob.mean().item())
        self.logit_norm_history.append(logit_norm.mean().item())
        self.generated_tokens += 1

        signals: List[DriftSignal] = []

        if len(self.entropy_history) < self.window_size:
            return signals

        mean_e = sum(self.entropy_history) / self.window_size
        var_e = sum((x - mean_e) ** 2 for x in self.entropy_history) / self.window_size
        std_e = var_e**0.5
        current_e = entropy.mean().item()
        if current_e < mean_e - self.entropy_collapse_sigma * std_e:
            signals.append(DriftSignal.ENTROPY_COLLAPSE)

        mean_n = sum(self.logit_norm_history) / self.window_size
        var_n = (
            sum((x - mean_n) ** 2 for x in self.logit_norm_history) / self.window_size
        )
        std_n = var_n**0.5
        current_n = logit_norm.mean().item()
        if abs(current_n - mean_n) > self.logit_norm_spike_sigma * std_n:
            signals.append(DriftSignal.LOGIT_NORM_SPIKE)

        return signals


class AdaptiveTemperatureScheduler:
    """Dynamically adjusts temperature for stable long-form generation."""

    def __init__(self, config: StabilityConfig):
        self.base_temp = config.base_temperature
        self.min_temp = config.min_temperature
        self.max_temp = config.max_temperature
        self.long_ctx_soft = config.long_context_threshold_1
        self.long_ctx_hard = config.long_context_threshold_2
        self.long_gen_soft = config.long_gen_decay_1
        self.long_gen_hard = config.long_gen_decay_2
        self.generated_tokens = 0

    def get_temperature(
        self, context_len: int, drift_signals: List[DriftSignal]
    ) -> float:
        temp = self.base_temp

        if DriftSignal.ENTROPY_COLLAPSE in drift_signals:
            temp += 0.2
        if DriftSignal.ENTROPY_EXPLOSION in drift_signals:
            temp -= 0.3

        if context_len > self.long_ctx_hard:
            temp *= 0.85
        elif context_len > self.long_ctx_soft:
            temp *= 0.93

        if self.generated_tokens > self.long_gen_hard:
            temp *= 0.95
        elif self.generated_tokens > self.long_gen_soft:
            temp *= 0.98

        self.generated_tokens += 1
        return max(self.min_temp, min(self.max_temp, temp))


class KVCacheIntegrityChecker:
    """Periodically validates KV cache entries for NaN/Inf, scale consistency, and token count."""

    def __init__(self, config: StabilityConfig):
        self.check_interval = config.kmin_cache_check_interval

    @torch.no_grad()
    def check(self, attn, step: int, expected_tokens: Optional[int] = None) -> bool:
        """Return False if corruption or scale/token drift detected."""
        if step % self.check_interval != 0:
            return True

        # 1. NaN/Inf and Scale Consistency checks on attributes
        for name in dir(attn):
            if name.startswith("__") or name in ("_buffers", "_modules", "_parameters"):
                continue
            try:
                val = getattr(attn, name)
            except AttributeError:
                continue
            if isinstance(val, torch.Tensor):
                if torch.isnan(val).any() or torch.isinf(val).any():
                    return False
                if "scale" in name.lower():
                    if (val == 0).any():
                        return False
                    if val.abs().max().item() > 1e4 or val.abs().min().item() < 1e-4:
                        return False

        # Checks on registered buffers
        for name, val in getattr(attn, "_buffers", {}).items():
            if isinstance(val, torch.Tensor):
                if torch.isnan(val).any() or torch.isinf(val).any():
                    return False
                if "scale" in name.lower():
                    if (val == 0).any():
                        return False
                    if val.abs().max().item() > 1e4 or val.abs().min().item() < 1e-4:
                        return False

        # 2. Token count check / Pointer check
        write_ptr = getattr(attn, "global_write_ptr", None)
        if write_ptr is not None:
            if isinstance(write_ptr, torch.Tensor):
                ptr_val = write_ptr.item()
            else:
                ptr_val = write_ptr
            
            if ptr_val < 0:
                return False
            
            if expected_tokens is not None and ptr_val > expected_tokens + 1024:
                return False

        return True
