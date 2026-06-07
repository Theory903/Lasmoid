#!/usr/bin/env python3
"""
scripts/gpu_train.py — Full-scale GPU pretraining for Lasmoid
=============================================================

Usage
-----
    python scripts/gpu_train.py --config path/to/config.json --device cuda:0

Required Arguments
------------------
    --config <path>     Path to a Lasmoid model configuration JSON file
                        (e.g. config.json, config_1b.json). The file must
                        conform to the ModelArgs schema in inference/config.py.
    --device <cuda:N>   CUDA device identifier (e.g. cuda:0, cuda:1).
                        This script is authored exclusively for NVIDIA GPUs
                        and does NOT support CPU or Apple MPS backends.

Optional Arguments
------------------
    --max_iters <int>       Total training iterations (default: 5000).
    --batch_size <int>      Per-device micro-batch size (default: 8).
    --learning_rate <float> Peak learning rate for AdamW params (default: 2.5e-4).
    --muon_lr <float>       Peak learning rate for Muon (2D) params (default: 2e-3).
    --grad_accum <int>      Gradient accumulation steps (default: 4).
    --warmup_steps <int>    WSD warmup phase steps (default: 100).
    --stable_steps <int>    WSD stable phase steps (default: 4500).
    --decay_steps <int>     WSD decay phase steps (default: 400).
    --save_interval <int>   Checkpoint save interval in steps (default: 500).
    --checkpoint_dir <str>  Directory for checkpoint output (default: checkpoints/pretrain).
    --expert_dtype <str>    QAT precision for MoE experts: bf16|fp8|nvfp4 (default: nvfp4).
    --dataset <path>        Path to a UTF-8 text file for training data. Falls back to
                            input.txt in the project root if not provided.
    --max_grad_norm <float> Gradient clipping max norm (default: 1.0).
    --seed <int>            Random seed for reproducibility (default: 1234).
    --compile              Enable torch.compile for the model (requires PyTorch 2.0+).

Hardware Assumptions
--------------------
    • NVIDIA GPU with CUDA support (compute capability >= 7.0 recommended).
    • Sufficient VRAM for the chosen config (config_1b.json requires ~8–12 GB;
      config.json / config_100m.json fit in ~2–4 GB).
    • PyTorch >= 2.0.0 with CUDA toolkit.
    • If using --compile, PyTorch >= 2.1 with triton is recommended for best perf.
    • BF16 training is used by default where supported (Ampere+); falls back to FP16
      AMP on older architectures.

This script wraps the training loop from train/pretrain.py and adapts it for full-scale
GPU training with larger batch sizes, gradient accumulation, and longer schedules. It
reuses the same Muon + AdamW optimizer split and WSD scheduler.

DO NOT run this script on CPU or MPS — it will raise an error if the specified device
is not a CUDA device.
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — ensure inference/ and train/ are importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "inference"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "train"))

from inference.model import Lasmoid, ModelArgs, compute_loss
from train.optimizer import Muon
from train.scheduler import WSDScheduler
from train.pretrain import apply_per_operator_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lasmoid full-scale GPU pretraining script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config JSON (ModelArgs schema).")
    parser.add_argument("--device", type=str, required=True,
                        help="CUDA device, e.g. cuda:0.")
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--muon_lr", type=float, default=2e-3)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--stable_steps", type=int, default=4500)
    parser.add_argument("--decay_steps", type=int, default=400)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--expert_dtype", type=str, default="nvfp4",
                        choices=["bf16", "fp8", "nvfp4"])
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to UTF-8 text dataset. Defaults to input.txt in project root.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile on the model.")
    return parser.parse_args()


def validate_device(device_str: str) -> torch.device:
    """Validate that the requested device is a CUDA device and is available."""
    if not device_str.startswith("cuda"):
        raise RuntimeError(
            f"This script requires a CUDA device. Got '{device_str}'. "
            "Do NOT run on CPU or MPS — use train/pretrain.py for non-GPU training."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this PyTorch installation. "
            "Please install PyTorch with CUDA support."
        )
    device = torch.device(device_str)
    # Validate device index
    idx = device.index if device.index is not None else 0
    if idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested device {device_str} but only {torch.cuda.device_count()} "
            "CUDA device(s) available."
        )
    return device


def load_model_args(config_path: str) -> ModelArgs:
    """Load ModelArgs from a JSON config file, filtering to valid fields."""
    from dataclasses import fields as dc_fields

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    valid_fields = {f.name for f in dc_fields(ModelArgs)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return ModelArgs(**filtered)


def load_dataset(dataset_path: str, model_args: ModelArgs, device: torch.device):
    """Load and tokenize the training dataset. Returns (train_data, val_data) on device."""
    import transformers

    enc = transformers.PreTrainedTokenizerFast.from_pretrained(
        _PROJECT_ROOT, fix_mistral_regex=True
    )
    model_args.vocab_size = len(enc)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "Provide a valid --dataset path or place input.txt in the project root."
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    data = torch.tensor(enc.encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"[GPU Train] Dataset: {len(data):,} tokens "
          f"(train: {len(train_data):,}, val: {len(val_data):,})")
    return train_data, val_data


def build_optimizers(model: torch.nn.Module, muon_lr: float, adamw_lr: float):
    """Partition model parameters into Muon (2D weights) and AdamW (everything else)."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2D weight matrices (excluding embeddings, output heads, adjacency, gates, hc)
        # go through the Muon optimizer for Newton-Schulz orthogonalization.
        if (len(p.shape) == 2
                and "emb" not in name
                and "head" not in name
                and "adj" not in name
                and "gate" not in name
                and "hc" not in name):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon = Muon(muon_params, lr=muon_lr)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr)
    return opt_muon, opt_adamw


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    """Sample a random batch of (input, target, loss_mask) from the data tensor."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i: i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1: i + seq_len + 1] for i in ix]).to(device)
    loss_mask = torch.ones_like(x, dtype=torch.float32)
    return x, y, loss_mask


def main():
    args = parse_args()

    # --- Device validation ---
    device = validate_device(args.device)
    torch.cuda.set_device(device)

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[GPU Train] Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"[GPU Train] Config: {args.config}")

    # --- Load config and build model ---
    model_args = load_model_args(args.config)

    # --- Load dataset ---
    dataset_path = args.dataset or os.path.join(_PROJECT_ROOT, "input.txt")
    train_data, val_data = load_dataset(dataset_path, model_args, device)

    # --- Build model ---
    print(f"[GPU Train] Building model (dim={model_args.dim}, "
          f"layers={model_args.n_layers}, heads={model_args.n_heads})...")
    model = Lasmoid(model_args).to(device)

    # Apply per-operator QAT precision simulation
    if args.expert_dtype != "bf16":
        print(f"[GPU Train] QAT: MoE experts -> {args.expert_dtype.upper()}, shared -> FP8")
        apply_per_operator_precision(model, args.expert_dtype)

    # Optional torch.compile
    if args.compile:
        print("[GPU Train] Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GPU Train] Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # --- Optimizers ---
    opt_muon, opt_adamw = build_optimizers(model, args.muon_lr, args.learning_rate)

    # --- WSD Scheduler ---
    scheduler = WSDScheduler(
        optimizers=[opt_muon, opt_adamw],
        warmup_steps=args.warmup_steps,
        stable_steps=args.stable_steps,
        decay_steps=args.decay_steps,
        base_lrs=[[args.muon_lr], [args.learning_rate]],
        min_lr_ratio=0.1,
    )

    # --- Checkpoint directory ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    print(f"[GPU Train] Starting training: {args.max_iters} iters, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum}, "
          f"effective_batch={args.batch_size * args.grad_accum}")
    print(f"[GPU Train] Schedule: warmup={args.warmup_steps}, "
          f"stable={args.stable_steps}, decay={args.decay_steps}")

    t_start = time.perf_counter()

    for step in range(args.max_iters):
        # Update LR via WSD scheduler
        lr_mult = scheduler.step(step)

        opt_muon.zero_grad()
        opt_adamw.zero_grad()

        total_step_loss = 0.0
        for _micro in range(args.grad_accum):
            xb, yb, loss_mask = get_batch(
                train_data, args.batch_size, model_args.max_seq_len, device
            )

            logits, mtp_logits, _, _, routing_maps, _, adjs, event_probs = model(xb, xb)

            main_loss = compute_loss(
                logits,
                yb,
                routing_maps,
                [model.last_vq_loss],
                adjs,
                event_probs,
                loss_mask=loss_mask,
                moe_aux_loss=model.last_moe_loss,
                moe_aux_coeff=getattr(model_args, "moe_aux_coeff", 1.0),
                token_concept_loss=model.last_token_concept_loss,
                token_concept_coeff=getattr(model_args, "token_concept_loss_coeff", 0.05),
                ignore_index=getattr(model_args, "loss_ignore_index", -100),
            )

            # MTP auxiliary loss
            if mtp_logits is not None:
                ce_loss_mtp = F.cross_entropy(
                    mtp_logits.view(-1, model_args.vocab_size),
                    yb[:, 1:].contiguous().view(-1),
                    ignore_index=getattr(model_args, "loss_ignore_index", -100),
                )
                main_loss = main_loss + getattr(model_args, "mtp_loss_coeff", 0.3) * ce_loss_mtp

            # Predictive coding loss
            pred_coeff = getattr(model_args, "predictive_coding_coeff", 0.01)
            loss = main_loss + pred_coeff * model.last_pred_loss
            loss = loss / args.grad_accum
            loss.backward()
            total_step_loss += loss.item() * args.grad_accum

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

        opt_muon.step()
        opt_adamw.step()

        # Logging
        if step % 10 == 0:
            elapsed = time.perf_counter() - t_start
            tokens_seen = (step + 1) * args.batch_size * args.grad_accum * model_args.max_seq_len
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f"Step {step:5d}/{args.max_iters} | "
                f"Loss: {total_step_loss:.4f} | "
                f"LR mult: {lr_mult:.4f} | "
                f"Tok/s: {tps:.0f} | "
                f"Peak mem: {mem_gb:.2f} GB"
            )

        # Checkpoint saving
        if step > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"lasmoid_gpu_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "muon_state_dict": opt_muon.state_dict(),
                "adamw_state_dict": opt_adamw.state_dict(),
                "config": args.config,
                "loss": total_step_loss,
            }, ckpt_path)
            print(f"[GPU Train] Checkpoint saved: {ckpt_path}")

    # --- Final checkpoint ---
    final_path = os.path.join(args.checkpoint_dir, "lasmoid_gpu_final.pt")
    torch.save({
        "step": args.max_iters,
        "model_state_dict": model.state_dict(),
        "muon_state_dict": opt_muon.state_dict(),
        "adamw_state_dict": opt_adamw.state_dict(),
        "config": args.config,
        "loss": total_step_loss,
    }, final_path)

    elapsed = time.perf_counter() - t_start
    print(f"[GPU Train] Training complete. Final loss: {total_step_loss:.4f}")
    print(f"[GPU Train] Total time: {elapsed:.1f}s")
    print(f"[GPU Train] Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
