#!/usr/bin/env python3
"""
scripts/gpu_generate.py — GPU text generation for Lasmoid
=========================================================

Usage
-----
    python scripts/gpu_generate.py \\
        --config path/to/config.json \\
        --checkpoint path/to/checkpoint_dir \\
        --device cuda:0 \\
        --prompt "Once upon a time"

Required Arguments
------------------
    --config <path>         Path to a Lasmoid model configuration JSON file
                            (e.g. config.json, config_1b.json). The file must
                            conform to the ModelArgs schema in inference/config.py.
    --checkpoint <path>     Path to a checkpoint directory containing a
                            lasmoid_latest.pt, lasmoid_step_*.pt, or lasmoid_final.pt
                            file. The checkpoint_loader will select the best available.
    --device <cuda:N>       CUDA device identifier (e.g. cuda:0, cuda:1).
                            This script is authored for NVIDIA GPUs; CPU/MPS will work
                            but are not the intended target.
    --prompt <text>         The text prompt to generate from.

Optional Arguments
------------------
    --max_new_tokens <int>  Maximum number of tokens to generate (default: 256).
    --temperature <float>   Sampling temperature (default: 0.8). 0.0 = argmax.
    --top_k <int>           Hard top-k filter (default: 0 = disabled).
    --top_p <float>         Nucleus sampling threshold (default: 1.0 = disabled).
    --min_p <float>         Min-P threshold (default: 0.05). 0.0 = disabled.
    --xtc_probability <float>  Per-step XTC probability (default: 0.0 = disabled).
    --xtc_threshold <float>    XTC prob threshold (default: 0.1).
    --dry_multiplier <float>   DRY repetition penalty strength (default: 0.0 = off).
    --dry_base <float>         DRY exponential base (default: 1.75).
    --dry_allowed_length <int> DRY minimum match length (default: 2).
    --seed <int>            Random seed for reproducible generation (default: None).
    --allow_partial_load    Allow incompatible checkpoint tensors to be expanded/skipped.
    --stream                Print tokens as they are generated (streaming mode).

Hardware Assumptions
--------------------
    • NVIDIA GPU with CUDA support (compute capability >= 7.0 recommended).
    • Sufficient VRAM for the chosen config (config_1b.json ~4–8 GB inference;
      config_100m.json fits in ~1–2 GB).
    • PyTorch >= 2.0.0 with CUDA toolkit.
    • BF16 inference is used where supported (Ampere+).
    • The script will also work on CPU or MPS for debugging, but GPU is the
      intended deployment target for acceptable throughput.

This script loads a Lasmoid checkpoint and produces sampled output using the
generation engine from inference/generate.py and the full 2026 SOTA sampling
pipeline from inference/sampler.py (Min-P, Top-P, Top-K, XTC, DRY, softcap).

Example
-------
    # Generate from a trained 1B checkpoint:
    python scripts/gpu_generate.py \\
        --config config_1b.json \\
        --checkpoint checkpoints/pretrain \\
        --device cuda:0 \\
        --prompt "The future of artificial intelligence" \\
        --max_new_tokens 512 \\
        --temperature 0.7 \\
        --min_p 0.05

    # Deterministic generation with seed:
    python scripts/gpu_generate.py \\
        --config config_100m.json \\
        --checkpoint checkpoints/current \\
        --device cuda:0 \\
        --prompt "Hello world" \\
        --seed 42 \\
        --temperature 0.0
"""

import os
import sys
import argparse
import time

import torch
import transformers

# ---------------------------------------------------------------------------
# Path setup — ensure inference/ is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "inference"))

from inference.checkpoint_loader import load_checkpoint_and_model
from inference.generate import generate, generate_stream
from inference._common import Linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lasmoid GPU text generation script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config JSON (ModelArgs schema).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory.")
    parser.add_argument("--device", type=str, required=True,
                        help="Device to use, e.g. cuda:0, cuda:1, cpu, mps.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to generate from.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate (default: 256).")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8). 0.0 = argmax.")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k filter (default: 0 = disabled).")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling threshold (default: 1.0 = disabled).")
    parser.add_argument("--min_p", type=float, default=0.05,
                        help="Min-P threshold (default: 0.05). 0.0 = disabled.")
    parser.add_argument("--xtc_probability", type=float, default=0.0,
                        help="XTC per-step probability (default: 0.0 = disabled).")
    parser.add_argument("--xtc_threshold", type=float, default=0.1,
                        help="XTC prob threshold (default: 0.1).")
    parser.add_argument("--dry_multiplier", type=float, default=0.0,
                        help="DRY repetition penalty strength (default: 0.0 = off).")
    parser.add_argument("--dry_base", type=float, default=1.75,
                        help="DRY exponential base (default: 1.75).")
    parser.add_argument("--dry_allowed_length", type=int, default=2,
                        help="DRY minimum match length (default: 2).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation.")
    parser.add_argument("--allow_partial_load", action="store_true",
                        help="Allow incompatible checkpoint tensors to be expanded/skipped.")
    parser.add_argument("--stream", action="store_true",
                        help="Print tokens as they are generated (streaming mode).")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Device setup ---
    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[gpu_generate] WARNING: CUDA not available, falling back to CPU.")
            device = "cpu"
        else:
            idx = int(device.split(":")[-1]) if ":" in device else 0
            if idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {device} but only {torch.cuda.device_count()} "
                    "CUDA device(s) available."
                )
            torch.cuda.set_device(torch.device(device))

    print(f"[gpu_generate] Device: {device.upper()}")
    print(f"[gpu_generate] Config: {args.config}")
    print(f"[gpu_generate] Checkpoint: {args.checkpoint}")

    # --- Reproducibility ---
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)

    # --- Load model from checkpoint ---
    model, model_args = load_checkpoint_and_model(
        args.checkpoint, args.config, device, allow_partial_load=args.allow_partial_load
    )
    model.eval()

    # Set linear precision based on model config
    Linear.dtype = torch.float8_e4m3fn if model_args.dtype == "fp8" else torch.bfloat16
    Linear.scale_fmt = getattr(model_args, "scale_fmt", None)

    # --- Load tokenizer ---
    enc = transformers.PreTrainedTokenizerFast.from_pretrained(
        _PROJECT_ROOT, fix_mistral_regex=True
    )
    eos_token_id = enc.eos_token_id if enc.eos_token_id is not None else 1

    # --- Tokenize prompt ---
    prompt_tokens = enc.encode(args.prompt)
    print(f"[gpu_generate] Prompt ({len(prompt_tokens)} tokens): {args.prompt!r}")
    print(f"[gpu_generate] Generating up to {args.max_new_tokens} tokens "
          f"(temp={args.temperature}, min_p={args.min_p}, top_p={args.top_p}, "
          f"top_k={args.top_k})")
    print("─" * 60)

    # --- Sampler configuration ---
    sampler_kwargs = dict(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        dry_multiplier=args.dry_multiplier,
        dry_base=args.dry_base,
        dry_allowed_length=args.dry_allowed_length,
        seed=args.seed,
    )

    # --- Generate ---
    t_start = time.perf_counter()

    if args.stream:
        # Streaming mode: print tokens as generated
        completion_tokens = []
        decoded_text = ""
        for token_id in generate_stream(
            model, prompt_tokens, args.max_new_tokens, eos_token_id, **sampler_kwargs
        ):
            if token_id == eos_token_id:
                break
            completion_tokens.append(token_id)
            new_decoded = enc.decode(completion_tokens)
            # Print only the newly decoded portion
            print(new_decoded[len(decoded_text):], end="", flush=True)
            decoded_text = new_decoded
        print()  # Final newline
    else:
        # Batch mode: generate all tokens then decode
        results = generate(
            model, [prompt_tokens], args.max_new_tokens, eos_token_id, **sampler_kwargs
        )
        completion_tokens = results[0]
        decoded_text = enc.decode(completion_tokens)
        print(decoded_text)

    # --- Stats ---
    elapsed = time.perf_counter() - t_start
    n_tokens = len(completion_tokens)
    tps = n_tokens / max(elapsed, 1e-6)

    print("─" * 60)
    print(f"[gpu_generate] Generated {n_tokens} tokens in {elapsed:.2f}s "
          f"({tps:.1f} tokens/sec)")

    if device.startswith("cuda"):
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[gpu_generate] Peak GPU memory: {peak_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
