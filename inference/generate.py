"""
Lasmod — generate.py
============================
Generation engine using the Lasmoid CQRS Hybrid Concept Transformer model.
Delegates sampling to sampler.py and checkpoint loading to checkpoint_loader.py.
"""

import os
import sys
from argparse import ArgumentParser
from typing import List, Optional, Generator

import time
import torch
import transformers

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .model import Lasmoid, ModelArgs
    from ._common import Linear
    from .sampler import full_sample
    from .checkpoint_loader import load_checkpoint_and_model
except ImportError:
    from model import Lasmoid, ModelArgs
    from _common import Linear
    from sampler import full_sample
    from checkpoint_loader import load_checkpoint_and_model


def _stability_temperature(model, logits, context_len, base_temperature):
    """Compute a drift-aware temperature when the model's stability system is on.

    Returns ``base_temperature`` unchanged if stability is disabled.
    """
    if not getattr(model, "stability_enabled", False):
        return base_temperature
    signals = model.drift_detector.check(logits)
    return model.temp_scheduler.get_temperature(context_len, signals)


def _stability_cache_check(model, step):
    """Periodic KV-cache integrity check across attention layers (no-op if disabled)."""
    if not getattr(model, "stability_enabled", False):
        return
    checker = getattr(model, "cache_checker", None)
    if checker is None:
        return
    for layer in model.layers:
        attn = getattr(layer, "attn", None)
        if attn is not None:
            checker.check(attn, step)


# ══════════════════════════════════════════════════════════════════════
# GENERATION ENGINE
# ══════════════════════════════════════════════════════════════════════


@torch.inference_mode()
def generate(
    model: Lasmoid,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
    seed: Optional[int] = None,
) -> List[List[int]]:
    model.eval()
    device = next(model.parameters()).device
    max_len = model.args.max_seq_len

    # Final logit soft-cap from model config (Req 13.3)
    final_logit_softcap = getattr(model.args, "final_logit_softcap", None)

    # Determinism: create a dedicated Generator seeded for reproducibility (Req 13.1)
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    results = []

    for tokens_list in prompt_tokens:
        idx = torch.tensor([tokens_list], dtype=torch.long, device=device)
        generated_ids: List[int] = list(tokens_list)

        # Prefill: run encoder + HCM once, freeze memory
        cond_len = idx.shape[1]
        if cond_len < max_len:
            padding = torch.full(
                (1, max_len - cond_len), eos_id, dtype=idx.dtype, device=device
            )
            idx_padded = torch.cat([padding, idx], dim=1)
        else:
            idx_padded = idx[:, -max_len:]

        # Run prefill forward
        logits, _, concept_db, memory_state, *_ = model(
            idx_padded, idx_padded, start_pos=0
        )

        # Sample first token
        last_logits = logits[:, -1, :]
        step_temp = _stability_temperature(model, logits, max_len, temperature)
        idx_next = full_sample(
            last_logits,
            generated_ids,
            temperature=step_temp,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            final_logit_softcap=final_logit_softcap,
            generator=generator,
        )

        token_id = idx_next.item()
        if token_id != eos_id:
            new_tokens = [token_id]
            generated_ids.append(token_id)
            idx = torch.cat([idx, idx_next], dim=1)

            # Decode loop
            current_pos = max_len
            for step in range(max_new_tokens - 1):
                logits, _, _, _, *_ = model(
                    x_enc=None,
                    x_dec=idx_next,
                    concept_db=concept_db,
                    memory_state=memory_state,
                    start_pos=current_pos,
                )

                step_temp = _stability_temperature(
                    model, logits, current_pos + 1, temperature
                )
                _stability_cache_check(model, step)
                idx_next = full_sample(
                    logits[:, -1, :],
                    generated_ids,
                    temperature=step_temp,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    dry_multiplier=dry_multiplier,
                    dry_base=dry_base,
                    dry_allowed_length=dry_allowed_length,
                    final_logit_softcap=final_logit_softcap,
                    generator=generator,
                )

                token_id = idx_next.item()
                if token_id == eos_id:
                    break
                new_tokens.append(token_id)
                generated_ids.append(token_id)
                idx = torch.cat([idx, idx_next], dim=1)
                current_pos += 1
        else:
            new_tokens = []

        results.append(new_tokens)

    return results


@torch.inference_mode()
def generate_stream(
    model: Lasmoid,
    prompt_tokens: List[int],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
    seed: Optional[int] = None,
) -> Generator[int, None, None]:
    model.eval()
    device = next(model.parameters()).device
    max_len = model.args.max_seq_len

    # Final logit soft-cap from model config (Req 13.3)
    final_logit_softcap = getattr(model.args, "final_logit_softcap", None)

    # Determinism: create a dedicated Generator seeded for reproducibility (Req 13.1)
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated_ids: List[int] = list(prompt_tokens)

    # Prefill: run encoder + HCM once, freeze memory
    cond_len = idx.shape[1]
    if cond_len < max_len:
        padding = torch.full(
            (1, max_len - cond_len), eos_id, dtype=idx.dtype, device=device
        )
        idx_padded = torch.cat([padding, idx], dim=1)
    else:
        idx_padded = idx[:, -max_len:]

    # Run prefill forward
    logits, _, concept_db, memory_state, *_ = model(idx_padded, idx_padded, start_pos=0)

    # Sample first token
    last_logits = logits[:, -1, :]
    step_temp = _stability_temperature(model, logits, max_len, temperature)
    idx_next = full_sample(
        last_logits,
        generated_ids,
        temperature=step_temp,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        dry_multiplier=dry_multiplier,
        dry_base=dry_base,
        dry_allowed_length=dry_allowed_length,
        final_logit_softcap=final_logit_softcap,
        generator=generator,
    )

    token_id = idx_next.item()
    yield token_id
    if token_id == eos_id:
        return

    generated_ids.append(token_id)
    idx = torch.cat([idx, idx_next], dim=1)

    # Decode loop
    current_pos = max_len
    for step in range(max_new_tokens - 1):
        logits, _, _, _, *_ = model(
            x_enc=None,
            x_dec=idx_next,
            concept_db=concept_db,
            memory_state=memory_state,
            start_pos=current_pos,
        )

        step_temp = _stability_temperature(model, logits, current_pos + 1, temperature)
        _stability_cache_check(model, step)
        idx_next = full_sample(
            logits[:, -1, :],
            generated_ids,
            temperature=step_temp,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            final_logit_softcap=final_logit_softcap,
            generator=generator,
        )

        token_id = idx_next.item()
        yield token_id
        if token_id == eos_id:
            break
        generated_ids.append(token_id)
        idx = torch.cat([idx, idx_next], dim=1)
        current_pos += 1


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dry_multiplier: float = 0.0,
    allow_partial_load: bool = False,
    device: Optional[str] = None,
) -> None:
    if device is not None:
        pass
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[generate] Device: {device.upper()}")

    torch.manual_seed(42)

    model, args = load_checkpoint_and_model(
        ckpt_path, config, device, allow_partial_load=allow_partial_load
    )
    model.eval()

    Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    Linear.scale_fmt = getattr(args, "scale_fmt", None)

    lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        enc = transformers.PreTrainedTokenizerFast.from_pretrained(
            lasmoid_dir, fix_mistral_regex=True
        )
    except Exception:
        enc = transformers.PreTrainedTokenizerFast.from_pretrained(
            lasmoid_dir
        )
    eos_token_id = enc.eos_token_id if enc.eos_token_id is not None else 1

    sampler_cfg = dict(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        dry_multiplier=dry_multiplier,
    )

    sampler_summary = (
        f"temperature={temperature}  min_p={min_p}  top_p={top_p}  top_k={top_k}\n"
        f"XTC prob={xtc_probability}  threshold={xtc_threshold}  "
        f"DRY mult={dry_multiplier}"
    )

    if interactive:
        print(f"\n{'─' * 60}")
        print(" Lasmoid — Interactive Shell (2026 Samplers)")
        print(f"{'─' * 60}")
        print(f" Samplers: {sampler_summary}")
        print(" Commands: /exit  /clear  /sampler")
        print(f"{'─' * 60}\n")

        history: List[str] = []
        while True:
            try:
                prompt = input(">>> ").strip()
            except EOFError:
                break
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                history.clear()
                print("[History cleared]")
                continue
            elif prompt == "/sampler":
                print(f"[Samplers] {sampler_summary}")
                continue
            elif not prompt:
                continue

            history.append(prompt)
            context = " ".join(history) + " "
            prompt_tokens = [enc.encode(context)]

            completion_tokens = []
            decoded_text = ""
            start_time = time.time()
            for token_id in generate_stream(
                model, prompt_tokens[0], max_new_tokens, eos_token_id, **sampler_cfg
            ):
                completion_tokens.append(token_id)
                new_decoded = enc.decode(completion_tokens)
                print(new_decoded[len(decoded_text) :], end="", flush=True)
                decoded_text = new_decoded
            print()
            elapsed_time = time.time() - start_time
            tps = len(completion_tokens) / max(elapsed_time, 1e-6)
            print(
                f"\033[90m[{len(completion_tokens)} tokens generated in {elapsed_time:.2f}s | {tps:.1f} TPS]\033[0m"
            )
            history.append(decoded_text.strip())

    else:
        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}")
            return
        with open(input_file) as f:
            prompts = [l.strip() for l in f if l.strip()]

        print(f"[generate] Batch mode: {len(prompts)} prompts")
        prompt_tokens = [enc.encode(p) for p in prompts]
        completion_tokens = generate(
            model, prompt_tokens, max_new_tokens, eos_token_id, **sampler_cfg
        )

        for p, ct in zip(prompts, completion_tokens):
            print(f"Prompt:     {p}")
            print(f"Completion: {enc.decode(ct)}")
            print("─" * 50)


if __name__ == "__main__":
    parser = ArgumentParser(description="Lasmoid inference — 2026 samplers")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    parser.add_argument("--top-p", type=float, default=1.0, help="1.0 = disabled")
    parser.add_argument(
        "--min-p", type=float, default=0.05, help="2026 default; 0 = disabled"
    )
    parser.add_argument(
        "--xtc-probability", type=float, default=0.0, help="XTC per-step probability"
    )
    parser.add_argument(
        "--xtc-threshold", type=float, default=0.1, help="XTC prob threshold"
    )
    parser.add_argument(
        "--dry-multiplier", type=float, default=0.0, help="DRY strength; 0 = disabled"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use: cpu, mps, or cuda"
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow incompatible checkpoint tensors to be expanded/skipped for debugging",
    )
    a = parser.parse_args()

    assert a.input_file or a.interactive, "Specify --input-file or --interactive"
    main(
        a.ckpt_path,
        a.config,
        a.input_file,
        a.interactive,
        a.max_new_tokens,
        a.temperature,
        a.top_k,
        a.top_p,
        a.min_p,
        a.xtc_probability,
        a.xtc_threshold,
        a.dry_multiplier,
        allow_partial_load=a.allow_partial_load,
        device=a.device,
    )
