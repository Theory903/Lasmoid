"""
Lasmoid — deploy_verify.py
==========================
Deployment verification and gating script for production environments.
Checks CUDA status, verifies Triton/TileLang kernel status, runs VRAM memory audits,
and validates quantization precision roundtrips.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure relative imports resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelArgs, QuantConfig, StabilityConfig
from lasmoid import Lasmoid
from kv_cache import AdaptiveQuantizedKVCache
from kernel import HAS_TRITON, SUPPORT_FP8_TRITON


def run_gpu_check() -> dict:
    print("=== G1: Hardware & CUDA Verification ===")
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Fallback"
    capability = torch.cuda.get_device_capability(0) if cuda_avail else (0, 0)
    
    print(f"  CUDA Available: {cuda_avail}")
    print(f"  Target Device:  {device_name}")
    print(f"  CUDA Capability: {capability}")
    print(f"  Triton Installed: {HAS_TRITON}")
    print(f"  FP8 Native Support: {SUPPORT_FP8_TRITON}\n")
    
    return {
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "cuda_capability": capability,
        "has_triton": HAS_TRITON,
        "fp8_native": SUPPORT_FP8_TRITON,
    }


def verify_triton_kernels(device: str) -> dict:
    print("=== G2: Triton Kernel Compilation & Execution ===")
    if not HAS_TRITON or device == "cpu":
        print("  [SKIP] Triton verification skipped (no CUDA device or triton package).\n")
        return {"compiled": False, "error": "Triton not active"}
        
    try:
        from kernel import act_quant, dequantize_kv
        # Test act_quant compilation
        x = torch.randn(4, 512, dtype=torch.bfloat16, device=device)
        y, s = act_quant(x, block_size=128)
        print("  [SUCCESS] Triton act_quant compiled and executed successfully.")
        return {"compiled": True, "error": None}
    except Exception as e:
        print(f"  [ERROR] Triton kernel compilation failed: {e}\n")
        return {"compiled": False, "error": str(e)}


def run_vram_audit(device: str) -> dict:
    print("=== G3: VRAM Memory Savings Audit ===")
    if device == "cpu":
        print("  [SKIP] VRAM Audit skipped on CPU device.\n")
        return {}
        
    # Measure VRAM allocated for standard vs quantized cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    args_bf16 = ModelArgs(use_fp8_kv=False, use_turboquant=False)
    mem_start = torch.cuda.memory_allocated()
    cache_bf16 = AdaptiveQuantizedKVCache(
        max_batch=8, max_seq=2048, head_dim=128, args=args_bf16, dtype=torch.bfloat16
    ).to(device)
    mem_bf16 = torch.cuda.memory_allocated() - mem_start
    
    # FP8 Quantized Cache
    torch.cuda.empty_cache()
    args_fp8 = ModelArgs(use_fp8_kv=True, use_turboquant=False)
    mem_start = torch.cuda.memory_allocated()
    cache_fp8 = AdaptiveQuantizedKVCache(
        max_batch=8, max_seq=2048, head_dim=128, args=args_fp8, dtype=torch.bfloat16
    ).to(device)
    mem_fp8 = torch.cuda.memory_allocated() - mem_start
    
    savings = (1.0 - (mem_fp8 / max(1, mem_bf16))) * 100
    
    print(f"  BF16 KV Cache VRAM: {mem_bf16 / 1024 / 1024:.2f} MB")
    print(f"  FP8 KV Cache VRAM:  {mem_fp8 / 1024 / 1024:.2f} MB")
    print(f"  KV Data Plane Savings: {savings:.1f}%\n")
    
    return {
        "bf16_mem_bytes": mem_bf16,
        "fp8_mem_bytes": mem_fp8,
        "savings_percentage": savings
    }


def verify_quantization_roundtrip(device: str) -> dict:
    print("=== G4: Quantization Precision Verification ===")
    from moe import quantize_weight_to_nvfp4
    
    w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
    w_q, scale = quantize_weight_to_nvfp4(w, block_size=32)
    
    # Reconstruct quantized weights using scale
    block_size = 32
    N, K = w_q.shape
    s_exp = scale.float().repeat_interleave(block_size, dim=1)[:, :K]
    w_recon = (w_q * s_exp).to(w.dtype)
    
    # Calculate quantization error
    diff = (w - w_recon).abs()
    mean_err = diff.mean().item()
    max_err = diff.max().item()
    
    print(f"  NVFP4 Mean Roundtrip Error: {mean_err:.5f}")
    print(f"  NVFP4 Max Roundtrip Error:  {max_err:.5f}")
    
    # Gating threshold
    passed = mean_err < 0.25
    print(f"  Gating Threshold Passed: {passed}\n")
    
    return {
        "mean_error": mean_err,
        "max_error": max_err,
        "passed": passed
    }


def verify_stability_softcap(device: str) -> dict:
    print("=== G5: Stability softcap & Integrity Checks ===")
    from stability import DriftDetector, AdaptiveTemperatureScheduler, DriftSignal
    
    cfg = StabilityConfig(
        drift_window_size=5,
        entropy_collapse_threshold=1.5,
        base_temperature=0.7
    )
    detector = DriftDetector(cfg)
    scheduler = AdaptiveTemperatureScheduler(cfg)
    
    # Populate history with normal high-entropy logits to fill sliding window
    for _ in range(4):
        detector.check(torch.randn(1, 1000, device=device) * 2.0)
    
    # Simulate low entropy (collapse)
    logits_collapsed = torch.zeros(1, 1000, device=device)
    logits_collapsed[0, 0] = 50.0  # extreme logit spike
    
    drift_signals = detector.check(logits_collapsed)
    new_temp = scheduler.get_temperature(context_len=100, drift_signals=drift_signals)
    
    print(f"  Spike Event Drift Signals: {drift_signals}")
    print(f"  Temperature Scheduled: {new_temp:.2f} (Base: 0.70)")
    passed = DriftSignal.ENTROPY_COLLAPSE in drift_signals and new_temp > 0.70
    print(f"  Stability Gating Passed: {passed}\n")
    
    return {
        "drift_signals": [s.value for s in drift_signals],
        "adjusted_temperature": new_temp,
        "passed": passed
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    results["gpu_info"] = run_gpu_check()
    results["triton"] = verify_triton_kernels(device)
    results["vram"] = run_vram_audit(device)
    results["quantization"] = verify_quantization_roundtrip(device)
    results["stability"] = verify_stability_softcap(device)
    
    # Save Report
    report_path = "deployment_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"=== Verification Complete. Report saved to: {report_path} ===")


if __name__ == "__main__":
    main()
