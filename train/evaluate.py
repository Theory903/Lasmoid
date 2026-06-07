"""
Lasmoid — train/evaluate.py
======================================================================
Held-out evaluation suite for the ablation harness. Every metric is computed
automatically so feature value can be measured rather than asserted.

Metrics:
  • perplexity        — language-modelling quality on a token stream
  • toolcall_success  — deterministic ML/EDA tool dispatch success rate
  • avg_experts       — adaptive MoE: mean experts recruited per token
  • domain_entropy    — cortex: entropy of domain-column usage (specialisation)
  • curiosity         — mean intrinsic-curiosity (forward-model error) signal

Data format (prepare_data.py): a flat uint32 .bin of token ids; we chunk it
into fixed-length sequences for evaluation.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_INF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference")
sys.path.insert(0, _INF_DIR)


def load_tokens(bin_path: str, seq_len: int, max_seqs: int = 64) -> Optional[torch.Tensor]:
    """Load a uint32 .bin token file and chunk into [N, seq_len] sequences."""
    if not os.path.exists(bin_path) or os.path.getsize(bin_path) == 0:
        return None
    arr = np.fromfile(bin_path, dtype=np.uint32)
    n = (len(arr) // seq_len) * seq_len
    if n == 0:
        return None
    arr = arr[:n].reshape(-1, seq_len)[:max_seqs]
    return torch.from_numpy(arr.astype(np.int64))


@torch.no_grad()
def perplexity(model, sequences: torch.Tensor, device: str = "cpu") -> float:
    """Token-level perplexity over [N, L] sequences."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(sequences.shape[0]):
        x = sequences[i : i + 1].to(device)
        out = model(x, x, start_pos=0)
        logits = out[0]  # [1, L, V]
        tgt = x[:, 1:]
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()
        total_tok += tgt.numel()
    if total_tok == 0:
        return float("nan")
    return math.exp(total_nll / total_tok)


def toolcall_success(registry=None) -> float:
    """Deterministic tool-dispatch success rate over a fixed battery of calls."""
    try:
        from tools import make_eda_registry
    except ImportError:
        return float("nan")
    reg = registry or make_eda_registry()
    battery = [
        {"function": {"name": "describe", "arguments": json.dumps({"data": [[1.0], [2.0], [3.0]]})}},
        {"function": {"name": "correlate", "arguments": json.dumps({"data": [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]})}},
        {"function": {"name": "fit_model", "arguments": json.dumps({"x": [0, 1, 2, 3], "y": [1, 3, 5, 7]})}},
        {"function": {"name": "cluster", "arguments": json.dumps({"data": [[0.0], [1.0], [9.0], [10.0]], "k": 2})}},
        {"function": {"name": "reduce_dim", "arguments": json.dumps({"data": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], "n_components": 1})}},
    ]
    results = reg.dispatch_many(battery)
    ok = sum(1 for r in results if r.status == "success")
    return ok / len(battery)


@torch.no_grad()
def model_diagnostics(model, sequences: torch.Tensor, device: str = "cpu") -> Dict[str, float]:
    """Cortex / adaptive / curiosity telemetry from a single forward pass."""
    diag: Dict[str, float] = {}
    if sequences is None or sequences.shape[0] == 0:
        return diag
    model.eval()
    x = sequences[:1].to(device)
    model(x, x, start_pos=0)
    layer0 = model.layers[0]
    moe = getattr(layer0, "moe_layer", None)
    if moe is not None:
        if getattr(moe, "last_avg_experts", None) is not None:
            diag["avg_experts"] = float(moe.last_avg_experts.item())
        gate = getattr(moe, "gate", None)
        if gate is not None and getattr(gate, "cortex", None) is not None:
            probs = gate.cortex.last_domain_probs
            if probs is not None:
                p = probs.float().mean(dim=0)
                p = p / (p.sum() + 1e-8)
                ent = -(p * (p + 1e-8).log()).sum().item()
                diag["domain_entropy"] = ent
    if getattr(model, "curiosity_expert", None) is not None:
        diag["curiosity"] = float(model.curiosity_expert.last_curiosity.item())
    return diag


def evaluate_model(model, args, val_tokens: Optional[torch.Tensor], device: str = "cpu") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if val_tokens is not None:
        metrics["perplexity"] = perplexity(model, val_tokens, device)
        metrics.update(model_diagnostics(model, val_tokens, device))
    metrics["toolcall_success"] = toolcall_success()
    return metrics


if __name__ == "__main__":
    # Standalone smoke: evaluate tool layer only (no model needed).
    print(json.dumps({"toolcall_success": toolcall_success()}, indent=2))
