#!/usr/bin/env python3
"""
Lasmoid — scratch_pretrain.py
==============================
Non-distill scratch pretraining for Lasmoid.
Dual-GPU (DDP via Accelerate), streaming FineWeb-Edu + Cosmopedia.
Pure CE loss — no teacher, no KL distillation.

Usage:
  # Single GPU (testing):
  python train/scratch_pretrain.py --config configs/model/config_gemma4_10m.json --max_steps 100

  # Dual GPU (Kaggle T4 x2):
  accelerate launch --num_processes 2 train/scratch_pretrain.py \
      --config configs/model/config_gemma4_10m.json --max_steps 15000

  # Resume from checkpoint:
  accelerate launch --num_processes 2 train/scratch_pretrain.py \
      --config configs/model/config_gemma4_10m.json --resume

Key optimizations applied:
  - PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
  - unwrap_model cached outside micro-batch loop
  - Gradient checkpointing enabled
  - NaN/Inf gradient guard with per-param fallback
  - WSD scheduler with cosine decay
"""

import os

# ── Memory optimisation (must be set before torch import) ─────────────────────
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

import sys
import json
import time
import math
import random
import gc
import re
import signal
import shutil
from pathlib import Path
from typing import Iterator, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Imports with fallbacks
# ═════════════════════════════════════════════════════════════════════════════

# Determine project root (works both as script and module)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_INFERENCE_DIR = _PROJECT_ROOT / "inference"

for d in [_PROJECT_ROOT, _INFERENCE_DIR, _SCRIPT_DIR]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

# Model components
from inference.config import ModelArgs
from inference.lasmoid import Lasmoid
from inference.loss import compute_loss

# Training components
try:
    from optimizer import (
        build_optimizers,
        clip_grad_global_norm,
        ensure_muon_closure_compat,
    )
    from scheduler import WSDScheduler
except ImportError:
    from train.optimizer import (
        build_optimizers,
        clip_grad_global_norm,
        ensure_muon_closure_compat,
    )
    from train.scheduler import WSDScheduler

# Accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs

# Data
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Config & CLI
# ═════════════════════════════════════════════════════════════════════════════

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Lasmoid Scratch Pretraining")
    p.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "configs/model/config_gemma4_10m.json"),
        help="Path to model config JSON",
    )
    p.add_argument("--max_steps", type=int, default=15000, help="Total training steps")
    p.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    p.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    p.add_argument("--muon_lr", type=float, default=2e-3, help="Muon learning rate")
    p.add_argument("--adamw_lr", type=float, default=3e-4, help="AdamW learning rate")
    p.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    p.add_argument(
        "--warmup_frac", type=float, default=0.02, help="Warmup fraction of max_steps"
    )
    p.add_argument("--stable_frac", type=float, default=0.80, help="Stable LR fraction")
    p.add_argument(
        "--min_lr_ratio", type=float, default=0.1, help="Min LR as fraction of peak"
    )
    p.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    p.add_argument(
        "--ckpt_every", type=int, default=500, help="Checkpoint interval (steps)"
    )
    p.add_argument("--log_every", type=int, default=10, help="Logging interval (steps)")
    p.add_argument(
        "--val_every", type=int, default=500, help="Validation interval (steps)"
    )
    p.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--data_dir", type=str, default=None, help="Local data cache dir")
    # Data composition
    p.add_argument(
        "--fineweb_p", type=float, default=0.70, help="FineWeb-Edu sampling probability"
    )
    p.add_argument(
        "--cosmo_p", type=float, default=0.30, help="Cosmopedia sampling probability"
    )
    # Teacher logit cache (optimization D)
    p.add_argument(
        "--teacher_logit_path",
        type=str,
        default=None,
        help="Path to precomputed teacher logits (.pt). If set, uses offline teacher.",
    )

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Dataset & Sequence Packer
# ═════════════════════════════════════════════════════════════════════════════

MIN_CHARS = 200
MAX_CHARS = 8000

_BOILERPLATE_PATTERNS = [
    "Click here",
    "Subscribe to",
    "Cookie Policy",
    "Terms of Service",
    "©",
    "All rights reserved",
    "Skip to content",
]


def clean_text(text: str) -> str:
    """Basic quality filter for web text."""
    text = re.sub(r"\s+", " ", text).strip()
    for pat in _BOILERPLATE_PATTERNS:
        if pat.lower() in text.lower():
            return ""
    return text


def stream_packed(
    dataset, tokenizer, seq_len: int, max_batches: int = None
) -> Iterator:
    """
    Yields (input_ids [seq_len], labels [seq_len]) from streaming dataset.
    Packs multiple docs into one sequence with EOS separator.
    """
    buf: list = []
    count = 0
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 1  # fallback

    for example in dataset:
        text = clean_text(example.get("text", ""))
        if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        if not isinstance(ids, list):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        ids.append(eos_id)
        buf.extend(ids)

        while len(buf) >= seq_len + 1:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y
            count += 1
            if max_batches and count >= max_batches:
                return


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Monitor Classes (Expert + Concept)
# ═════════════════════════════════════════════════════════════════════════════


class ExpertMonitor:
    """Track MoE expert utilization, detect dead experts and routing collapse."""

    def __init__(self, n_experts: int, n_layers: int):
        self.n_experts = n_experts
        self.n_layers = n_layers
        self.reset()

    def reset(self):
        self.counts = torch.zeros(self.n_layers, self.n_experts)
        self.steps = 0

    def update(self, routing_maps: list):
        for layer_idx, routing in enumerate(routing_maps):
            if routing is None or layer_idx >= self.n_layers:
                continue
            try:
                r = routing.detach().cpu().view(-1)
                for idx in r.tolist():
                    if 0 <= int(idx) < self.n_experts:
                        self.counts[layer_idx, int(idx)] += 1
            except Exception:
                pass
        self.steps += 1

    def stats(self) -> dict:
        total = self.counts.sum(-1, keepdim=True).clamp(min=1)
        freq = self.counts / total
        dead_mask = freq < 0.01
        dead_count = dead_mask.sum().item()
        ent = -(freq.clamp(min=1e-8) * freq.clamp(min=1e-8).log()).sum(-1)
        max_ent = math.log(self.n_experts)
        return {
            "dead_experts": int(dead_count),
            "dead_pct": 100 * dead_count / max(1, self.n_layers * self.n_experts),
            "mean_entropy": ent.mean().item(),
            "max_entropy": max_ent,
            "entropy_ratio": (ent.mean() / max_ent).item() if max_ent > 0 else 0.0,
        }


class ConceptMonitor:
    """Track ESCM concept usage, collapse, and entropy."""

    def __init__(self, num_concepts: int):
        self.num_concepts = num_concepts
        self.reset()

    def reset(self):
        self.usage = torch.zeros(self.num_concepts)
        self.steps = 0

    def update(self, concept_indices: list):
        for ci in concept_indices:
            if ci is None:
                continue
            try:
                idx = ci.detach().cpu().view(-1)
                for i in idx.tolist():
                    if 0 <= int(i) < self.num_concepts:
                        self.usage[int(i)] += 1
            except Exception:
                pass
        self.steps += 1

    def stats(self) -> dict:
        total = self.usage.sum().clamp(min=1)
        freq = self.usage / total
        ent = -(freq.clamp(min=1e-8) * freq.clamp(min=1e-8).log()).sum().item()
        max_ent = math.log(self.num_concepts)
        top5_frac = freq.topk(min(5, self.num_concepts)).values.sum().item()
        return {
            "entropy": ent,
            "max_entropy": max_ent,
            "entropy_ratio": ent / max_ent if max_ent > 0 else 0.0,
            "collapsed": top5_frac > 0.80,
            "top5_coverage": top5_frac,
            "dead_concepts": int((freq < 0.001).sum().item()),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Checkpoint System
# ═════════════════════════════════════════════════════════════════════════════

CURSOR_FILE = None  # set during init


def save_cursor(step: int, dataset_idx: int):
    cursor = {
        "step": step,
        "dataset_idx": dataset_idx,
        "timestamp": time.time(),
    }
    with open(CURSOR_FILE, "w") as f:
        json.dump(cursor, f, indent=2)


def load_cursor() -> dict:
    if CURSOR_FILE and CURSOR_FILE.exists():
        with open(CURSOR_FILE) as f:
            return json.load(f)
    return {"step": 0, "dataset_idx": 0}


def save_checkpoint(
    step: int,
    model,
    optimizers: list,
    scheduler,
    loss_history: list,
    expert_stats: dict,
    concept_stats: dict,
    dataset_idx: int = 0,
    ckpt_dir: Path = None,
    is_main: bool = True,
    unwrapped=None,
):
    if not is_main:
        return
    from safetensors.torch import save_file as sf_save

    ckpt_path = ckpt_dir / f"step_{step:06d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 1. Model weights (safetensors)
    sd = unwrapped.state_dict() if unwrapped else model.state_dict()
    sf_save({k: v.cpu() for k, v in sd.items()}, str(ckpt_path / "model.safetensors"))

    # 2. Optimizer states
    opt_states = [opt.state_dict() for opt in optimizers]
    torch.save(opt_states, ckpt_path / "optimizer.pt")

    # 3. Scheduler state
    if hasattr(scheduler, "warmup_steps"):
        sched_state = {
            "warmup_steps": scheduler.warmup_steps,
            "stable_steps": scheduler.stable_steps,
            "decay_steps": scheduler.decay_steps,
            "base_lrs": scheduler.base_lrs,
            "min_lr_ratio": scheduler.min_lr_ratio,
        }
    else:
        sched_state = {}
    with open(ckpt_path / "scheduler.json", "w") as f:
        json.dump(sched_state, f)

    # 4. RNG state
    rng_state = {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
    }
    torch.save(rng_state, ckpt_path / "rng.pt")

    # 5. Meta
    meta = {
        "step": step,
        "dataset_idx": dataset_idx,
        "loss_history": loss_history[-500:],
        "expert_stats": expert_stats,
        "concept_stats": concept_stats,
        "timestamp": time.time(),
    }
    with open(ckpt_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 6. Cursor
    save_cursor(step, dataset_idx)

    # 7. Latest symlink
    latest = ckpt_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink(missing_ok=True)
    try:
        latest.symlink_to(ckpt_path.name)
    except (OSError, AttributeError):
        pass

    print(f"  💾 Checkpoint saved: step {step} → {ckpt_path.name}")


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    latest = ckpt_dir / "latest"
    if (latest.is_symlink() or latest.exists()) and latest.resolve().exists():
        return latest.resolve()
    candidates = sorted(ckpt_dir.glob("step_*"))
    return candidates[-1] if candidates else None


def load_checkpoint(model, optimizers, scheduler, ckpt_path: Path):
    from safetensors.torch import load_file as sf_load

    print(f"  📂 Resuming from: {ckpt_path}")

    # Model
    state = sf_load(str(ckpt_path / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)

    # Optimizers
    if (ckpt_path / "optimizer.pt").exists():
        opt_states = torch.load(ckpt_path / "optimizer.pt", map_location="cpu")
        for opt, st in zip(optimizers, opt_states):
            try:
                opt.load_state_dict(st)
            except Exception as e:
                print(f"  ⚠️  Optimizer load error (continuing): {e}")

    # RNG
    if (ckpt_path / "rng.pt").exists():
        rng = torch.load(ckpt_path / "rng.pt", map_location="cpu")
        torch.set_rng_state(rng["cpu"])
        if rng.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("python"):
            random.setstate(rng["python"])

    # Meta
    if (ckpt_path / "meta.json").exists():
        with open(ckpt_path / "meta.json") as f:
            meta = json.load(f)
        return (
            meta.get("step", 0),
            meta.get("loss_history", []),
            meta.get("expert_stats", {}),
            meta.get("concept_stats", {}),
        )
    return 0, [], {}, {}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Graceful Shutdown
# ═════════════════════════════════════════════════════════════════════════════

_CHECKPOINT_ON_KILL = {"step": 0, "dataset_idx": 0}


def _shutdown_handler(
    signum,
    frame,
    accelerator=None,
    ckpt_dir=None,
    model=None,
    optimizers=None,
    scheduler=None,
    loss_history=None,
    expert_stats=None,
    concept_stats=None,
    dataset_idx=0,
    is_main=True,
    unwrapped=None,
):
    sig_name = signal.Signals(signum).name
    print(f"\n⚠️  Received {sig_name} — saving emergency checkpoint...")
    step = _CHECKPOINT_ON_KILL.get("step", 0)
    ds_idx = _CHECKPOINT_ON_KILL.get("dataset_idx", 0)
    if step > 0 and is_main:
        save_checkpoint(
            step=step,
            model=model,
            optimizers=optimizers,
            scheduler=scheduler,
            loss_history=loss_history,
            expert_stats=expert_stats or {},
            concept_stats=concept_stats or {},
            dataset_idx=ds_idx,
            ckpt_dir=ckpt_dir,
            is_main=True,
            unwrapped=unwrapped,
        )
        print(f"  💾 Emergency checkpoint saved at step {step}")
    print("Exiting gracefully.")
    exit(0)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Main Training Function
# ═════════════════════════════════════════════════════════════════════════════


def train():
    args = parse_args()

    # ── Accelerate ────────────────────────────────────────────────────────────
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    is_main = accelerator.is_main_process
    n_proc = accelerator.num_processes

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints/scratch_pretrain")
    log_dir = Path("logs/scratch_pretrain")
    export_dir = Path("export/scratch_pretrain")
    for d in [ckpt_dir, log_dir, export_dir]:
        d.mkdir(parents=True, exist_ok=True)

    global CURSOR_FILE
    CURSOR_FILE = ckpt_dir / "cursor.json"

    # ── GPU info ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available() and is_main:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU{i}: {name} ({vram:.1f} GB)")

    if is_main:
        print(f"Processes      : {n_proc}")
        print(f"Grad accum     : {args.grad_accum}")
        print(f"Max steps      : {args.max_steps}")
        print(f"Config         : {args.config}")

    # ── Load model config ─────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = json.load(f)
    cfg["max_seq_len"] = args.seq_len
    cfg["max_batch_size"] = args.batch_size
    model_args = ModelArgs(**cfg)
    VOCAB_SIZE = model_args.vocab_size

    if is_main:
        print(
            f"Model: dim={model_args.dim}, layers={model_args.n_layers}, "
            f"heads={model_args.n_heads}, vocab={VOCAB_SIZE:,}"
        )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    TOK_PATH = str(_PROJECT_ROOT)
    tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    assert len(tokenizer) == VOCAB_SIZE, (
        f"Tokenizer vocab ({len(tokenizer)}) != config vocab ({VOCAB_SIZE})"
    )
    if is_main:
        print(f"Tokenizer vocab: {VOCAB_SIZE:,}")

    # ── Datasets (streaming) ──────────────────────────────────────────────────
    if is_main:
        print("Loading datasets (streaming)...")

    ds_fineweb = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        streaming=True,
        trust_remote_code=True,
    ).select_columns(["text"])

    ds_cosmo = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
        trust_remote_code=True,
    ).select_columns(["text"])

    ds_train = interleave_datasets(
        [ds_fineweb, ds_cosmo],
        probabilities=[args.fineweb_p, args.cosmo_p],
        seed=args.seed,
    )
    if is_main:
        print("✅ Datasets ready")

    # ── Build model ───────────────────────────────────────────────────────────
    model = Lasmoid(model_args).to(device)
    model.gradient_checkpointing = True

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed_params = sum(
        p.numel() for n, p in model.named_parameters() if "emb" in n or "head" in n
    )
    if is_main:
        print(f"\n{'=' * 45}")
        print(
            f"Model: {total_params / 1e6:.2f}M total, {train_params / 1e6:.2f}M trainable"
        )
        print(
            f"Embed/head: {embed_params / 1e6:.2f}M ({100 * embed_params / train_params:.0f}%)"
        )
        print(f"Body: {(train_params - embed_params) / 1e6:.2f}M")
        print(f"{'=' * 45}\n")

    # ── Optimizers ────────────────────────────────────────────────────────────
    ensure_muon_closure_compat()
    optimizers = build_optimizers(
        model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        momentum=0.95,
        ns_steps=5,
        adaptive_noise=False,
    )
    if is_main:
        print(f"Optimizers: {[type(o).__name__ for o in optimizers]}")

    # ── WSD scheduler ─────────────────────────────────────────────────────────
    warmup_steps = max(1, int(args.warmup_frac * args.max_steps))
    stable_steps = int(args.stable_frac * args.max_steps)
    decay_steps = args.max_steps - warmup_steps - stable_steps

    base_lrs = []
    for opt in optimizers:
        base_lrs.append([g["lr"] for g in opt.param_groups])

    scheduler = WSDScheduler(
        optimizers=optimizers,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        base_lrs=base_lrs,
        min_lr_ratio=args.min_lr_ratio,
    )
    if is_main:
        print(f"WSD: warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps}")

    # ── Monitors ──────────────────────────────────────────────────────────────
    expert_monitor = ExpertMonitor(
        n_experts=model_args.n_routed_experts, n_layers=model_args.n_layers
    )
    concept_monitor = ConceptMonitor(num_concepts=model_args.num_concepts)

    # ── Accelerate prepare ────────────────────────────────────────────────────
    model, *optimizers = accelerator.prepare(model, *optimizers)

    # ── OOM safety ────────────────────────────────────────────────────────────
    def safe_forward(m, x_enc, x_dec):
        try:
            return m(x_enc, x_dec)
        except torch.cuda.OutOfMemoryError:
            print("  ⚠️  OOM on forward — skipping batch")
            torch.cuda.empty_cache()
            gc.collect()
            return None
        except ValueError as _e:
            _msg = str(_e)
            if any(kw in _msg for kw in ("stoi", "storage", "symbolize")):
                print(f"  ⚠️  Symbolizer ValueError (skipping): {_msg}")
                return None
            raise

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def _make_handler(**ctx):
        def _handler(signum, frame):
            _shutdown_handler(signum, frame, **ctx)

        return _handler

    if is_main:
        shutdown_ctx = dict(
            accelerator=accelerator,
            ckpt_dir=ckpt_dir,
            model=model,
            optimizers=optimizers,
            scheduler=scheduler,
            loss_history=[],
            expert_stats={},
            concept_stats={},
            dataset_idx=0,
            is_main=True,
            unwrapped=accelerator.unwrap_model(model),
        )
        signal.signal(signal.SIGTERM, _make_handler(**shutdown_ctx))
        signal.signal(signal.SIGINT, _make_handler(**shutdown_ctx))

    # ── Checkpoint / Resume ──────────────────────────────────────────────────
    loss_history = []
    expert_stats = {}
    concept_stats = {}
    dataset_cursor = load_cursor()
    start_step = 0

    if args.resume:
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None and is_main:
            start_step, loss_history, expert_stats, concept_stats = load_checkpoint(
                accelerator.unwrap_model(model),
                optimizers,
                scheduler,
                latest_ckpt,
            )
            print(f"✅ Resumed from step {start_step}")
        elif is_main:
            print("No checkpoint found — starting fresh")

    # Broadcast start_step
    if n_proc > 1:
        t = torch.tensor([start_step], dtype=torch.long, device=device)
        torch.distributed.broadcast(t, src=0)
        start_step = t.item()

    # ── Data iterator ─────────────────────────────────────────────────────────
    data_iter = stream_packed(ds_train, tokenizer, args.seq_len)
    dataset_idx = dataset_cursor.get("dataset_idx", 0)

    # Fast-forward resume
    if dataset_idx > 0 and is_main:
        print(f"Fast-forwarding dataset to position {dataset_idx}...")
        for _ in range(dataset_idx):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = stream_packed(ds_train, tokenizer, args.seq_len)

    # ── Logging ───────────────────────────────────────────────────────────────
    loss_log_path = log_dir / "loss.jsonl"

    def log_step(d: dict):
        with open(loss_log_path, "a") as f:
            f.write(json.dumps(d) + "\n")

    # ── Training Loop ─────────────────────────────────────────────────────────
    model.train()
    from tqdm.auto import tqdm

    pbar = tqdm(
        range(start_step, args.max_steps),
        initial=start_step,
        total=args.max_steps,
        desc="Pretraining",
        disable=not is_main,
    )

    accum_steps = accelerator.gradient_accumulation_steps
    tokens_per_step = args.batch_size * args.seq_len * accum_steps * n_proc
    global_step = start_step

    # ══ OPTIMIZATION B: Cache unwrap_model outside loop ══
    _unwrapped = accelerator.unwrap_model(model)

    for step in pbar:
        t0 = time.time()
        global_step = step
        _CHECKPOINT_ON_KILL.update({"step": step, "dataset_idx": dataset_idx})

        # LR schedule
        lr_mult = scheduler.step(step)

        # Gradient accumulation
        total_loss_accum = 0.0
        for micro in range(accum_steps):
            try:
                x, y = next(data_iter)
                dataset_idx += 1
            except StopIteration:
                data_iter = stream_packed(ds_train, tokenizer, args.seq_len)
                x, y = next(data_iter)

            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            with accelerator.accumulate(model):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = safe_forward(model, x, x)
                    if out is None:
                        continue

                    (
                        logits,
                        mtp_logits,
                        concept_db,
                        memory_state,
                        routing_maps,
                        concept_indices,
                        adjacencies,
                        event_probs,
                    ) = out

                    # ── OPTIMIZATION B: use cached unwrapped model ──
                    total_loss = compute_loss(
                        logits=logits,
                        targets=y,
                        routing_maps=routing_maps,
                        vq_losses=[_unwrapped.last_vq_loss],
                        adjacencies=adjacencies,
                        event_probs=event_probs,
                        moe_aux_loss=_unwrapped.last_moe_loss,
                        moe_aux_coeff=getattr(model_args, "moe_aux_coeff", 1.0),
                        mtp_loss=(
                            F.cross_entropy(
                                mtp_logits[:, : y.shape[1] - 1].reshape(
                                    -1, mtp_logits.shape[-1]
                                ),
                                y[:, 1:].reshape(-1),
                                ignore_index=-100,
                            )
                            if mtp_logits is not None and x.shape[1] > 1
                            else None
                        ),
                        mtp_coeff=getattr(model_args, "mtp_loss_coeff", 0.3),
                        token_concept_loss=_unwrapped.last_token_concept_loss,
                        token_concept_coeff=model_args.token_concept_loss_coeff,
                        commit_loss=_unwrapped.last_commit_loss,
                        commit_coeff=model_args.hcm_commit_loss_coeff,
                        curiosity_loss=(
                            _unwrapped.last_curiosity_loss
                            if getattr(model_args, "use_curiosity_expert", False)
                            else None
                        ),
                        graph_sparsity=0.01,
                        label_smoothing=0.0,
                        ignore_index=-100,
                    )

                accelerator.backward(total_loss / accum_steps)
                total_loss_accum += total_loss.item() / accum_steps

            # Update monitors
            if is_main:
                expert_monitor.update(routing_maps)
                concept_monitor.update(concept_indices)

        # ── Gradient clipping & optimizer step ──────────────────────────────
        if accelerator.sync_gradients:
            grad_norm = clip_grad_global_norm(_unwrapped, max_norm=args.max_grad_norm)

            # NaN/Inf gradient guard
            _skip_step = False
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                for _p in _unwrapped.parameters():
                    if _p.grad is not None and (
                        torch.isnan(_p.grad).any() or torch.isinf(_p.grad).any()
                    ):
                        _skip_step = True
                        break

            if _skip_step:
                print(f"  ⚠️  NaN/Inf gradient at step {step} — skipping")
                log_step(
                    {"step": step, "nan_gradient": True, "grad_norm": float("nan")}
                )
            else:
                for opt in optimizers:
                    opt.step()
                _unwrapped.apply_pending_bias_updates()

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

        # ── Logging ──────────────────────────────────────────────────────────
        loss_history.append(total_loss_accum)
        step_time = time.time() - t0
        total_tokens_seen = tokens_per_step * (step + 1)

        if is_main:
            exp_s = expert_monitor.stats()
            con_s = concept_monitor.stats()
            tokens_per_sec = tokens_per_step / max(step_time, 1e-6)

            # GPU memory
            if torch.cuda.is_available() and step % 10 == 0:
                _alloc = torch.cuda.memory_allocated() / 1024**3
                _total = torch.cuda.get_device_properties(0).total_memory / 1e9
                _free = _total - _alloc
                _mem_str = f"{_alloc:.1f}G/{_free:.1f}G free"
            else:
                _mem_str = ""

            pbar.set_postfix(
                {
                    "loss": f"{total_loss_accum:.3f}",
                    "tok/s": f"{tokens_per_sec:,.0f}",
                    "mem": _mem_str or "",
                    "ent": f"{exp_s['entropy_ratio']:.2f}",
                    "dead_e": exp_s["dead_experts"],
                }
            )

            if step % args.log_every == 0:
                log_step(
                    {
                        "step": step,
                        "loss": total_loss_accum,
                        "step_time_s": step_time,
                        "tokens_per_sec": tokens_per_sec,
                        "lr_mult": lr_mult,
                        "grad_norm": grad_norm if accelerator.sync_gradients else 0.0,
                        "gpu_mem_gb": _alloc if torch.cuda.is_available() else 0,
                        "expert_stats": exp_s,
                        "concept_stats": con_s,
                    }
                )

            # ── Checkpoint ────────────────────────────────────────────────
            if step > 0 and step % args.ckpt_every == 0:
                save_checkpoint(
                    step=step,
                    model=model,
                    optimizers=optimizers,
                    scheduler=scheduler,
                    loss_history=loss_history,
                    expert_stats=exp_s,
                    concept_stats=con_s,
                    dataset_idx=dataset_idx,
                    ckpt_dir=ckpt_dir,
                    is_main=True,
                    unwrapped=_unwrapped,
                )
                expert_monitor.reset()
                concept_monitor.reset()

        # ── Validation ─────────────────────────────────────────────────────
        if step > 0 and step % args.val_every == 0 and is_main:
            _run_validation(model, tokenizer, args, device, _unwrapped)

    # ── Training complete ─────────────────────────────────────────────────────
    if is_main:
        print("\n✅ Training complete!")
        save_checkpoint(
            step=args.max_steps,
            model=model,
            optimizers=optimizers,
            scheduler=scheduler,
            loss_history=loss_history,
            expert_stats=exp_s,
            concept_stats=con_s,
            dataset_idx=dataset_idx,
            ckpt_dir=ckpt_dir,
            is_main=True,
            unwrapped=_unwrapped,
        )
        _run_validation(model, tokenizer, args, device, _unwrapped)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: Validation
# ═════════════════════════════════════════════════════════════════════════════


def _run_validation(model, tokenizer, args, device, unwrapped):
    """Compute perplexity on a small val set (TinyStories or random sample)."""
    model.eval()
    try:
        # Try TinyStories val binary
        val_path = _PROJECT_ROOT / "datasets" / "tinystories_val.bin"
        if val_path.exists():
            val_data = np.frombuffer(val_path.read_bytes(), dtype=np.uint16).astype(
                np.int64
            )
            val_data = torch.from_numpy(val_data)

            n_val = min(512, len(val_data) // args.seq_len)
            total_nll, total_tokens = 0.0, 0

            with torch.no_grad():
                for i in range(n_val):
                    chunk = val_data[i * args.seq_len : (i + 1) * args.seq_len + 1]
                    if len(chunk) < args.seq_len + 1:
                        continue
                    x = chunk[: args.seq_len].unsqueeze(0).to(device)
                    y = chunk[1 : args.seq_len + 1].unsqueeze(0).to(device)
                    out = unwrapped(x, x)
                    logits = out[0]
                    nll = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        y.reshape(-1),
                        reduction="sum",
                    )
                    total_nll += nll.item()
                    total_tokens += y.numel()

            ppl = math.exp(min(total_nll / max(total_tokens, 1), 20))
            print(f"  📊 Validation perplexity: {ppl:.2f} (n={n_val})")
            return ppl
        else:
            print("  📊 No validation set found — skipping perplexity")
    except Exception as e:
        print(f"  ⚠️  Validation error: {e}")
    finally:
        model.train()
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: Entry Point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
