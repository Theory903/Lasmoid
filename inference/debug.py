"""
Lasmoid — debug.py
==================
Structured debug-visualisation system for compression and attention diagnostics.

Provides:
  • Collector buffers (deque) keyed by step/block/layer
  • Per-step compact formatted tables (human-readable at a glance)
  • JSONL event log for offline analysis
  • On-demand matplotlib heatmap + bar chart plots (optional, no-op if missing)

Usage
-----
    from inference.debug import (
        DEBUG_ENABLED,
        push_compressor_debug, push_attention_debug,
        dump_compressor_summary, dump_attention_summary,
        write_debug_jsonl, visualize_compression_patterns,
        set_debug_step, reset_debug_buffer,
    )

Master toggle::

    from inference import debug
    debug.DEBUG_ENABLED = False   # silence everything (zero overhead thereafter)

Switches
--------
DEBUG_ENABLED         – master toggle (default True)
BLOCK_TRACE_ENABLED   – per-block ``--- BLOCK X FORWARD ---`` trace (default False)
LOG_DIR               – directory for JSONL + plot output (default ``logs/``)
"""

from __future__ import annotations

import os
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import torch

# ── Switches ──────────────────────────────────────────────────────────────────
DEBUG_ENABLED: bool = True
"""Master toggle – set ``False`` to disable all debug tracing."""

BLOCK_TRACE_ENABLED: bool = False
"""Per-block forward trace (``--- BLOCK X FORWARD ---``)."""

LOG_DIR: str = "logs"
"""Directory for ``debug.jsonl`` and plot images."""

# ── Global state ──────────────────────────────────────────────────────────────
_DEBUG_STEP: int = 0

_COMPRESSOR_BUF: deque = deque(maxlen=5000)
"""List[CompressorEntry]"""

_ATTENTION_BUF: deque = deque(maxlen=5000)
"""List[AttentionEntry]"""

# ── Data entries ──────────────────────────────────────────────────────────────


@dataclass
class CompressorEntry:
    """Snapshot of a single Compressor forward-pass."""

    step: int
    layer_id: int
    r_step: int
    training: bool
    saved_num_fires: dict[int, int] = field(default_factory=dict)
    num_complete_fires_dyn: int = 0
    num_complete_fires: int = 0


@dataclass
class AttentionEntry:
    """Snapshot of a single Attention cache utilisation event."""

    step: int
    layer_id: int
    bsz: int
    cache_write_ptr: list[float] = field(default_factory=list)
    ptr_val: float = 0.0
    cache_cap: float = 0.0


# ── Push functions (called from model code) ──────────────────────────────────


def push_compressor_debug(
    layer_id: int,
    r_step: int,
    training: bool,
    saved_num_fires: dict[int, int] | None = None,
    num_complete_fires_dyn: int = 0,
    num_complete_fires: int = 0,
) -> None:
    """Record a compressor debug snapshot.

    Call from :meth:`Compressor.forward` after the fire-count decision.
    """
    if not DEBUG_ENABLED:
        return
    _COMPRESSOR_BUF.append(
        CompressorEntry(
            step=_DEBUG_STEP,
            layer_id=layer_id,
            r_step=r_step,
            training=training,
            saved_num_fires=(saved_num_fires or {}).copy(),
            num_complete_fires_dyn=num_complete_fires_dyn,
            num_complete_fires=num_complete_fires,
        )
    )


def push_attention_debug(
    layer_id: int,
    bsz: int,
    cache_write_ptr: list[float] | None = None,
    ptr_val: float = 0.0,
    cache_cap: float = 0.0,
) -> None:
    """Record an attention cache-utilisation snapshot.

    Call from attention's debug site (after the Compressor has run).
    """
    if not DEBUG_ENABLED:
        return
    _ATTENTION_BUF.append(
        AttentionEntry(
            step=_DEBUG_STEP,
            layer_id=layer_id,
            bsz=bsz,
            cache_write_ptr=(cache_write_ptr or []),
            ptr_val=ptr_val,
            cache_cap=cache_cap,
        )
    )


# ── Summary dumps ─────────────────────────────────────────────────────────────


def _fmt(n: float, width: int = 5) -> str:
    """Format a number for table cells (int if whole, else compact float)."""
    if abs(n - round(n)) < 1e-6:
        return str(int(round(n))).rjust(width)
    return f"{n:{width}.2f}"


def dump_compressor_summary(*, force: bool = False) -> None:
    """Print a compact table of fires **per layer × r_step** from the buffer.

    Table layout::

        Layer  │  R0   R1   R2   │  Dyn   Sel
        ───────┼─────────────────┼────────────
          0    │  3    5    5    │   5     5
          1    │  4    6    6    │   6     6
    """
    if not DEBUG_ENABLED and not force:
        return
    if not _COMPRESSOR_BUF:
        return

    # Group by layer_id, then r_step → take LAST entry per (layer, r_step)
    by_layer: dict[int, dict[int, CompressorEntry]] = defaultdict(dict)
    step_latest = 0
    for e in _COMPRESSOR_BUF:
        by_layer[e.layer_id][e.r_step] = e
        if e.step > step_latest:
            step_latest = e.step

    all_r_steps: set[int] = set()
    for rd in by_layer.values():
        all_r_steps.update(rd.keys())
    sorted_r = sorted(all_r_steps)

    # Build rows
    rows: list[list[str]] = []
    for layer_id in sorted(by_layer):
        rd = by_layer[layer_id]
        cells: list[str] = [str(layer_id).rjust(2)]
        for r in sorted_r:
            e = rd.get(r)
            cells.append(_fmt(e.num_complete_fires if e else 0))
        # dynamic max + selected (all r_steps share same dyn/selected)
        first_e = next(iter(rd.values()))
        cells.append(_fmt(first_e.num_complete_fires_dyn))
        cells.append(_fmt(first_e.num_complete_fires))
        rows.append(cells)

    if not rows:
        return

    header_r = "  ".join(f"R{r}" for r in sorted_r)
    sep = "─" * (4 + len(header_r) + 12)
    print(f"\n──[ Compressor Fires  (step {step_latest}) ]{'─' * 40}")
    print(f"     Layer  │  {header_r}  │  Dyn   Sel")
    print(f"     ───────┼{'─' * (len(header_r) + 2)}┼──────────")
    for cells in rows:
        vals = "  ".join(cells[1 : len(sorted_r) + 1])
        print(f"       {cells[0]:>2}   │  {vals}  │  {cells[-2]:>3}  {cells[-1]:>3}")
    print(f"     ─{sep} ")
    print(f"     Buffer: {len(_COMPRESSOR_BUF)} entries  |  {len(by_layer)} layers\n")


def dump_attention_summary(*, force: bool = False) -> None:
    """Print a compact cache-utilisation table per layer.

    Table layout::

        Layer  │  Bsz  │  Ptr  │  Cap  │  Util %
        ───────┼───────┼───────┼───────┼────────
          0    │   4   │  23   │  64   │  35.9
          1    │   4   │  18   │  64   │  28.1
    """
    if not DEBUG_ENABLED and not force:
        return
    if not _ATTENTION_BUF:
        return

    # Last entry per layer
    last_by_layer: dict[int, AttentionEntry] = {}
    for e in _ATTENTION_BUF:
        last_by_layer[e.layer_id] = e

    step_latest = max(e.step for e in _ATTENTION_BUF)
    print(f"\n──[ Attention Cache Utilisation  (step {step_latest}) ]{'─' * 35}")
    print(f"     Layer  │  Bsz  │  Ptr  │  Cap  │  Util %")
    print(f"     ───────┼───────┼───────┼───────┼─────────")
    for layer_id in sorted(last_by_layer):
        e = last_by_layer[layer_id]
        util = 100.0 * e.ptr_val / max(e.cache_cap, 1.0)
        print(
            f"       {layer_id:>2}   │  {e.bsz:>3}  │  {e.ptr_val:>4.0f}  │  {e.cache_cap:>4.0f}  │  {util:>5.1f}"
        )
    print(f"     ─{'─' * 43}")
    print()


# ── JSONL logging ─────────────────────────────────────────────────────────────


def write_debug_jsonl(filepath: str | None = None, *, force: bool = False) -> str:
    """Append buffer contents to a JSONL file (one JSON object per line).

    Returns the file path used.
    """
    if not DEBUG_ENABLED and not force:
        return ""
    if filepath is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        filepath = os.path.join(LOG_DIR, "debug.jsonl")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    entries: list[dict[str, Any]] = []
    for e in _COMPRESSOR_BUF:
        d = asdict(e)
        d["_type"] = "compressor"
        entries.append(d)
    for e in _ATTENTION_BUF:
        d = asdict(e)
        d["_type"] = "attention"
        entries.append(d)

    if not entries:
        return filepath

    with open(filepath, "a") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, default=str) + "\n")

    print(f"  [debug] wrote {len(entries)} entries → {filepath}")
    return filepath


# ── Plotting (optional, requires matplotlib) ──────────────────────────────────


def visualize_compression_patterns(
    filepath: str | None = None,
    *,
    force: bool = False,
) -> bool:
    """Generate a matplotlib heatmap + bar chart from the compressor buffer.

    * Heatmap: ``(layer_id, r_step) → num_complete_fires``
    * Bar:     per-layer cache utilisation (ptr / cap)

    Returns ``True`` if a plot was saved, ``False`` if matplotlib was unavailable.
    """
    if not DEBUG_ENABLED and not force:
        return False
    if not _COMPRESSOR_BUF or not _ATTENTION_BUF:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if filepath is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        filepath = os.path.join(LOG_DIR, f"debug_step_{_DEBUG_STEP}.png")

    # ── Compressor heatmap ──────────────────────────────────────────────
    by_layer: dict[int, dict[int, int]] = defaultdict(dict)
    for e in _COMPRESSOR_BUF:
        by_layer[e.layer_id][e.r_step] = e.num_complete_fires
    all_rs: set[int] = set()
    for rd in by_layer.values():
        all_rs.update(rd.keys())
    sorted_rs = sorted(all_rs)
    sorted_layers = sorted(by_layer)

    heat = [[by_layer[l].get(r, 0) for r in sorted_rs] for l in sorted_layers]

    # ── Attention utilisation bar ───────────────────────────────────────
    last_attn: dict[int, AttentionEntry] = {}
    for e in _ATTENTION_BUF:
        last_attn[e.layer_id] = e

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    im = ax1.imshow(heat, aspect="auto", cmap="YlOrRd")
    ax1.set_xticks(range(len(sorted_rs)))
    ax1.set_xticklabels([f"R{r}" for r in sorted_rs])
    ax1.set_yticks(range(len(sorted_layers)))
    ax1.set_yticklabels([str(l) for l in sorted_layers])
    ax1.set_xlabel("Residual Step")
    ax1.set_ylabel("Layer")
    ax1.set_title(f"Compressor Fires (step {_DEBUG_STEP})")
    for i in range(len(sorted_layers)):
        for j in range(len(sorted_rs)):
            ax1.text(
                j,
                i,
                str(heat[i][j]),
                ha="center",
                va="center",
                fontsize=9,
                color="black"
                if heat[i][j] < max(1, max(max(r) for r in heat)) // 2
                else "white",
            )
    plt.colorbar(im, ax=ax1)

    # Bar chart
    layers_b = sorted(last_attn)
    ptrs = [last_attn[l].ptr_val for l in layers_b]
    caps = [last_attn[l].cache_cap for l in layers_b]
    utils = [100.0 * p / max(c, 1.0) for p, c in zip(ptrs, caps)]
    x = range(len(layers_b))
    ax2.bar(x, utils, color="steelblue")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(l) for l in layers_b])
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cache Utilisation (%)")
    ax2.set_title(f"Attention Cache Utilisation (step {_DEBUG_STEP})")
    ax2.axhline(y=100.0, color="red", linestyle="--", linewidth=0.8, label="Capacity")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  [debug] plot saved → {filepath}")
    return True


# ── Step tracking ─────────────────────────────────────────────────────────────


def set_debug_step(step: int) -> None:
    """Set the current training step for debug entry timestamps."""
    global _DEBUG_STEP
    _DEBUG_STEP = step


def reset_debug_buffer() -> None:
    """Clear all debug buffers (call at the start of each training step)."""
    _COMPRESSOR_BUF.clear()
    _ATTENTION_BUF.clear()
