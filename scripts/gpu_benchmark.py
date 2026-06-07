#!/usr/bin/env python3
"""
scripts/gpu_benchmark.py — GPU throughput and memory benchmark for Lasmoid
==========================================================================

Usage
-----
    python scripts/gpu_benchmark.py --config path/to/config.json --device cuda:0

    # With custom benchmark parameters:
    python scripts/gpu_benchmark.py \\
        --config config_1b.json \\
        --device cuda:0 \\
        --batch_size 4 \\
        --seq_len 512 \\
        --n_iters 50

Required Arguments
------------------
    --config <path>     Path to a Lasmoid model configuration JSON file
                        (e.g. config.json, config_1b.json). The file must
                        conform to the ModelArgs schema in inference/config.py.
    --device <cuda:N>   CUDA device identifier (e.g. cuda:0, cuda:1).
                        This script requires an NVIDIA GPU with CUDA support.

Optional Arguments
------------------
    --batch_size <int>  Batch size for benchmark forward passes (default: 1).
    --seq_len <int>     Sequence length in tokens (default: uses config max_seq_len).
    --n_iters <int>     Number of timed forward-pass iterations (default: 30).
    --warmup <int>      Number of warmup iterations before timing (default: 5).
    --seed <int>        Random seed for reproducibility (default: 1234).
    --compile           Enable torch.compile before benchmarking (requires PyTorch 2.1+).
    --mode <str>        Benchmark mode: "forward" (encoder+decoder forward pass) or
                        "generate" (autoregressive token generation). Default: "forward".
    --dtype <str>       Override compute dtype: bf16, fp16, fp32. Default: bf16.

Hardware Assumptions
--------------------
    • NVIDIA GPU with CUDA support (compute capability >= 7.0 recommended).
    • Sufficient VRAM for the chosen config and batch/seq dimensions:
      - config_100m.json: ~1–2 GB at batch=1, seq=512
      - config_1b.json:   ~4–8 GB at batch=1, seq=512
      - Larger batch sizes and sequence lengths scale memory linearly.
    • PyTorch >= 2.0.0 with CUDA toolkit.
    • BF16 is used by default (Ampere+ for best throughput); fp16 for older GPUs.
    • torch.compile requires PyTorch >= 2.1 with triton for optimal kernel fusion.

Metrics Reported
----------------
    • Tokens/sec throughput (total tokens processed / wall-clock time)
    • Latency per iteration (mean, median, min, max in milliseconds)
    • Peak GPU memory allocated (in GB)
    • Peak GPU memory reserved (in GB)
    • Model parameter count
    • Effective TFLOPS estimate (approximate, based on 2*params*tokens formula)

Example
-------
    # Quick benchmark of 100M model:
    python scripts/gpu_benchmark.py \\
        --config config_100m.json \\
        --device cuda:0 \\
        --batch_size 4 \\
        --seq_len 256 \\
        --n_iters 50

    # Benchmark with torch.compile:
    python scripts/gpu_benchmark.py \\
        --config config_1b.json \\
        --device cuda:0 \\
        --compile \\
        --n_iters 100

    # Autoregressive generation benchmark:
    python scripts/gpu_benchmark.py \\
        --config config_1b.json \\
        --device cuda:0 \\
        --mode generate \\
        --seq_len 128 \\
        --n_iters 20

DO NOT run this script on CPU or MPS — it requires CUDA for accurate memory and
throughput measurements.
"""

import os
import sys
import json
import argparse
import time
import statistics

import torch

# ---------------------------------------------------------------------------
# Path setup — ensure inference/ is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "inference"))

from inference.model import Lasmoid, ModelArgs
from inference._common import Linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lasmoid GPU throughput and memory benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config JSON (ModelArgs schema).")
    parser.add_argument("--device", type=str, required=True,
                        help="CUDA device, e.g. cuda:0.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for forward passes (default: 1).")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Sequence length in tokens (default: config max_seq_len).")
    parser.add_argument("--n_iters", type=int, default=30,
                        help="Number of timed iterations (default: 30).")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations before timing (default: 5).")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed (default: 1234).")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile on the model.")
    parser.add_argument("--mode", type=str, default="forward",
                        choices=["forward", "generate"],
                        help="Benchmark mode: forward pass or autoregressive (default: forward).")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Compute dtype (default: bf16).")
    return parser.parse_args()


def validate_device(device_str: str) -> torch.device:
    """Validate that the requested device is a CUDA device and is available."""
    if not device_str.startswith("cuda"):
        raise RuntimeError(
            f"This benchmark requires a CUDA device. Got '{device_str}'. "
            "GPU memory and throughput measurements require CUDA."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this PyTorch installation. "
            "Please install PyTorch with CUDA support."
        )
    device = torch.device(device_str)
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


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]


def benchmark_forward(
    model: torch.nn.Module,
    model_args: ModelArgs,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    n_iters: int,
    warmup: int,
    compute_dtype: torch.dtype,
) -> dict:
    """
    Benchmark encoder+decoder forward pass throughput.

    Returns a dict with timing and throughput metrics.
    """
    model.eval()

    # Generate fixed random input tokens
    x_enc = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)
    x_dec = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)

    # Warmup — let CUDA kernels compile and caches fill
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=compute_dtype):
        for _ in range(warmup):
            _ = model(x_enc, x_dec)
    torch.cuda.synchronize(device)

    # Reset peak memory stats after warmup
    torch.cuda.reset_peak_memory_stats(device)

    # Timed iterations
    latencies_ms = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=compute_dtype):
        for _ in range(n_iters):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            _ = model(x_enc, x_dec)

            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    # Compute metrics
    total_tokens = batch_size * seq_len * n_iters
    total_time_s = sum(latencies_ms) / 1000.0
    tokens_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0.0

    peak_mem_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_mem_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    return {
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "latency_mean_ms": statistics.mean(latencies_ms),
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "latency_stdev_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "peak_mem_allocated_gb": peak_mem_allocated_gb,
        "peak_mem_reserved_gb": peak_mem_reserved_gb,
    }


def benchmark_generate(
    model: torch.nn.Module,
    model_args: ModelArgs,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    n_iters: int,
    warmup: int,
    compute_dtype: torch.dtype,
) -> dict:
    """
    Benchmark autoregressive token generation throughput.

    Generates `seq_len` tokens starting from a short prompt for each iteration.
    Returns a dict with timing and throughput metrics.
    """
    model.eval()

    prompt_len = min(16, seq_len // 4)
    gen_len = seq_len - prompt_len

    # Fixed prompt tokens
    prompt = torch.randint(0, model_args.vocab_size, (batch_size, prompt_len), device=device)

    # Warmup
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=compute_dtype):
        for _ in range(warmup):
            _ = model.generate(prompt, max_new_tokens=min(gen_len, 16))
    torch.cuda.synchronize(device)

    # Reset peak memory stats after warmup
    torch.cuda.reset_peak_memory_stats(device)

    # Timed iterations
    latencies_ms = []
    total_generated_tokens = 0

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=compute_dtype):
        for _ in range(n_iters):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            output = model.generate(prompt, max_new_tokens=gen_len)

            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            # output shape: (batch_size, prompt_len + generated)
            total_generated_tokens += (output.shape[1] - prompt_len) * batch_size

    # Compute metrics
    total_time_s = sum(latencies_ms) / 1000.0
    tokens_per_sec = total_generated_tokens / total_time_s if total_time_s > 0 else 0.0

    peak_mem_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_mem_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    return {
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": total_generated_tokens,
        "total_time_s": total_time_s,
        "latency_mean_ms": statistics.mean(latencies_ms),
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "latency_stdev_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "peak_mem_allocated_gb": peak_mem_allocated_gb,
        "peak_mem_reserved_gb": peak_mem_reserved_gb,
    }


def print_results(results: dict, model_params: int, mode: str, args) -> None:
    """Print formatted benchmark results to stdout."""
    print()
    print("═" * 64)
    print(f"  LASMOID GPU BENCHMARK RESULTS — {mode.upper()} MODE")
    print("═" * 64)
    print()
    print(f"  Config:         {args.config}")
    print(f"  Device:         {args.device} ({torch.cuda.get_device_name(args.device)})")
    print(f"  Dtype:          {args.dtype}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Seq length:     {args.seq_len}")
    print(f"  Iterations:     {args.n_iters} (+ {args.warmup} warmup)")
    print(f"  torch.compile:  {'yes' if args.compile else 'no'}")
    print(f"  Parameters:     {model_params:,}")
    print()
    print("─" * 64)
    print("  THROUGHPUT")
    print("─" * 64)
    print(f"  Tokens/sec:     {results['tokens_per_sec']:,.1f}")
    print(f"  Total tokens:   {results['total_tokens']:,}")
    print(f"  Total time:     {results['total_time_s']:.3f} s")
    print()

    # Approximate TFLOPS: ~2 * params * tokens_per_sec / 1e12
    approx_tflops = 2 * model_params * results["tokens_per_sec"] / 1e12
    print(f"  Approx TFLOPS:  {approx_tflops:.2f} (2 × params × tok/s)")
    print()
    print("─" * 64)
    print("  LATENCY (per iteration)")
    print("─" * 64)
    print(f"  Mean:           {results['latency_mean_ms']:.2f} ms")
    print(f"  Median:         {results['latency_median_ms']:.2f} ms")
    print(f"  Min:            {results['latency_min_ms']:.2f} ms")
    print(f"  Max:            {results['latency_max_ms']:.2f} ms")
    print(f"  Stdev:          {results['latency_stdev_ms']:.2f} ms")
    print()
    print("─" * 64)
    print("  MEMORY")
    print("─" * 64)
    print(f"  Peak allocated: {results['peak_mem_allocated_gb']:.3f} GB")
    print(f"  Peak reserved:  {results['peak_mem_reserved_gb']:.3f} GB")
    print()
    print("═" * 64)


def main():
    args = parse_args()

    # --- Device validation ---
    device = validate_device(args.device)
    torch.cuda.set_device(device)

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Load config ---
    model_args = load_model_args(args.config)

    # Resolve sequence length
    if args.seq_len is None:
        args.seq_len = model_args.max_seq_len
    model_args.max_seq_len = max(model_args.max_seq_len, args.seq_len)
    model_args.max_batch_size = max(model_args.max_batch_size, args.batch_size)

    # --- Determine compute dtype ---
    compute_dtype = get_dtype(args.dtype)

    # --- Build model ---
    print(f"[gpu_benchmark] Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"[gpu_benchmark] Config: {args.config}")
    print(f"[gpu_benchmark] Building model (dim={model_args.dim}, "
          f"layers={model_args.n_layers}, heads={model_args.n_heads})...")

    # Set Linear layer precision
    Linear.dtype = compute_dtype

    model = Lasmoid(model_args).to(device)

    if args.compile:
        print("[gpu_benchmark] Compiling model with torch.compile...")
        model = torch.compile(model)

    model_params = sum(p.numel() for p in model.parameters())
    print(f"[gpu_benchmark] Parameters: {model_params:,}")
    print(f"[gpu_benchmark] Mode: {args.mode} | Batch: {args.batch_size} | "
          f"Seq: {args.seq_len} | Iters: {args.n_iters} (+{args.warmup} warmup)")
    print(f"[gpu_benchmark] Dtype: {args.dtype}")
    print()

    # --- Run benchmark ---
    if args.mode == "forward":
        results = benchmark_forward(
            model, model_args, device,
            args.batch_size, args.seq_len, args.n_iters, args.warmup,
            compute_dtype,
        )
    else:
        results = benchmark_generate(
            model, model_args, device,
            args.batch_size, args.seq_len, args.n_iters, args.warmup,
            compute_dtype,
        )

    # --- Print results ---
    print_results(results, model_params, args.mode, args)


if __name__ == "__main__":
    main()
