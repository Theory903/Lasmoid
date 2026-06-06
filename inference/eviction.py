"""
Lasmod — SnapKV-style eviction for KV cache (Phase A3).

Selectively retains important KV positions using observation-window
attention score heuristic. Keeps: sink tokens + top-k scored tokens +
recent window tokens.

Reference: SnapKV (Xiao et al., 2024)
Adapted for Lasmod per-head flat cache shape (seq, head_dim).
"""

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class SnapKVConfig:
    enabled: bool = False
    sink_size: int = 4
    window_size: int = 64
    max_keep_size: int = 512
    observation_length: int = 32
    topk_ratio: float = 0.5


def snapkv_select_indices(k: torch.Tensor, config: SnapKVConfig) -> torch.Tensor:
    """Select KV positions to retain using SnapKV heuristic.

    Args:
        k: Key tensor (seq_len, head_dim) — single head.
        config: SnapKV eviction configuration.

    Returns:
        1D LongTensor of indices to keep, sorted.
    """
    seq_len = k.shape[0]
    if seq_len <= config.sink_size + config.window_size:
        return torch.arange(seq_len, device=k.device)

    sink_end = config.sink_size
    window_start = seq_len - config.window_size
    middle_len = window_start - sink_end

    if middle_len <= 0:
        return torch.arange(seq_len, device=k.device)

    obs_start = max(0, seq_len - config.observation_length)
    obs = k[obs_start:]

    scores = obs @ k[sink_end:window_start].T
    pooled = scores.max(dim=0)[0]

    keep_middle = max(1, int(middle_len * config.topk_ratio))
    keep_middle = min(keep_middle, middle_len)
    _, mid = torch.topk(pooled, keep_middle)
    mid = mid.sort()[0] + sink_end

    return torch.cat(
        [
            torch.arange(0, sink_end, device=k.device),
            mid,
            torch.arange(window_start, seq_len, device=k.device),
        ]
    )


def snapkv_evict(
    k: torch.Tensor, v: torch.Tensor, config: SnapKVConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evict KV cache using SnapKV selection.

    Returns (k_evicted, v_evicted, indices) each with shape
    (batch, keep_len, head_dim) / (batch, keep_len).
    """
    B, S, D = k.shape
    cfg = SnapKVConfig(
        enabled=config.enabled,
        sink_size=min(config.sink_size, S),
        window_size=min(config.window_size, S - config.sink_size),
        max_keep_size=config.max_keep_size,
        observation_length=config.observation_length,
        topk_ratio=config.topk_ratio,
    )

    all_idx = []
    for b in range(B):
        sel = snapkv_select_indices(k[b], cfg)
        all_idx.append(sel.unsqueeze(0))

    indices = torch.cat(all_idx, dim=0)
    B2, keep = indices.shape
    bidx = torch.arange(B2, device=k.device).unsqueeze(1).expand(-1, keep)
    return k[bidx, indices], v[bidx, indices], indices
