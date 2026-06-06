"""
Lasmoid — generate.py
=====================
Generation engine using the Lasmoid CQRS Hybrid Concept Transformer model.
Implements the 2026 SOTA sampling pipeline:
  logits → temperature → DRY repetition penalty → XTC creative exclusion → Min-P/Top-P/Top-K → sample
"""

import os
import sys
import json
import glob
import re
from argparse import ArgumentParser
from typing import List, Optional, Generator

import time
import torch
import torch.nn.functional as F
import transformers

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .model import Lasmoid, ModelArgs, Linear
except ImportError:
    from model import Lasmoid, ModelArgs, Linear


# ══════════════════════════════════════════════════════════════════════
# SAMPLER PRIMITIVES (2026 SOTA)
# ══════════════════════════════════════════════════════════════════════

def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    return logits / max(temperature, 1e-8)


def _apply_dry(
    logits: torch.Tensor,
    generated: List[int],
    dry_multiplier: float = 0.8,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
) -> torch.Tensor:
    """DRY repetition control tracked via history n-grams."""
    if dry_multiplier == 0.0 or len(generated) < dry_allowed_length:
        return logits

    logits = logits.clone()
    last_token = generated[-1]
    match_indices = [i for i, t in enumerate(generated[:-1]) if t == last_token]

    for idx in match_indices:
        match_len = 1
        while (match_len <= dry_allowed_length and
               idx - match_len >= 0 and
               len(generated) - 1 - match_len >= 0 and
               generated[idx - match_len] == generated[-1 - match_len]):
            match_len += 1

        if match_len < dry_allowed_length:
            continue

        if idx + 1 < len(generated):
            penalised_token = generated[idx + 1]
            penalty = dry_multiplier * (dry_base ** (match_len - dry_allowed_length))
            logits[0, penalised_token] -= penalty

    return logits


def _apply_xtc(
    logits: torch.Tensor,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
) -> torch.Tensor:
    """XTC (eXclude Top Choices) for open-ended creative tasks."""
    if xtc_probability == 0.0:
        return logits
    if torch.rand(1).item() > xtc_probability:
        return logits

    logits = logits.clone()
    probs  = torch.softmax(logits, dim=-1)

    mask = probs > xtc_threshold
    if mask.sum() < probs.numel():
        logits[mask] = float("-inf")
    return logits


def _apply_min_p(logits: torch.Tensor, min_p: float = 0.05) -> torch.Tensor:
    """Min-P sampling scaling threshold by the leading choice probability."""
    if min_p <= 0.0:
        return logits
    probs     = torch.softmax(logits, dim=-1)
    p_max     = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * p_max
    logits = logits.clone()
    logits[probs < threshold] = float("-inf")
    return logits


def _apply_top_p(logits: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    """Top-P (nucleus) sampling."""
    if top_p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = (cumulative - sorted_probs) > top_p
    remove_original = torch.zeros_like(logits, dtype=torch.bool)
    remove_original.scatter_(-1, sorted_idx, remove_mask)
    logits = logits.clone()
    logits[remove_original] = float("-inf")
    return logits


def _apply_top_k(logits: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Hard Top-K filter."""
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold  = values[..., -1, None]
    logits     = logits.clone()
    logits[logits < threshold] = float("-inf")
    return logits


def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _full_sample(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
) -> torch.Tensor:
    logits = _apply_temperature(logits, temperature)
    logits = _apply_dry(logits, generated_ids, dry_multiplier, dry_base, dry_allowed_length)
    logits = _apply_xtc(logits, xtc_probability, xtc_threshold)

    if min_p > 0.0:
        logits = _apply_min_p(logits, min_p)
    else:
        logits = _apply_top_p(logits, top_p)

    logits = _apply_top_k(logits, top_k)
    return _sample_token(logits, temperature)


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
) -> List[List[int]]:
    model.eval()
    device  = next(model.parameters()).device
    max_len = model.args.max_seq_len
    results = []

    for tokens_list in prompt_tokens:
        idx = torch.tensor([tokens_list], dtype=torch.long, device=device)
        generated_ids: List[int] = list(tokens_list)

        # Prefill: run encoder + HCM once, freeze memory
        cond_len = idx.shape[1]
        if cond_len < max_len:
            padding   = torch.full((1, max_len - cond_len), eos_id, dtype=idx.dtype, device=device)
            idx_padded = torch.cat([padding, idx], dim=1)
        else:
            idx_padded = idx[:, -max_len:]

        # Run prefill forward
        logits, _, concept_db, memory_state, *_ = model(idx_padded, idx_padded, start_pos=0)

        # Sample first token
        last_logits = logits[:, -1, :]
        idx_next = _full_sample(
            last_logits,
            generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
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
                
                idx_next = _full_sample(
                    logits[:, -1, :],
                    generated_ids,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    dry_multiplier=dry_multiplier,
                    dry_base=dry_base,
                    dry_allowed_length=dry_allowed_length,
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
) -> Generator[int, None, None]:
    model.eval()
    device  = next(model.parameters()).device
    max_len = model.args.max_seq_len

    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated_ids: List[int] = list(prompt_tokens)

    # Prefill: run encoder + HCM once, freeze memory
    cond_len = idx.shape[1]
    if cond_len < max_len:
        padding   = torch.full((1, max_len - cond_len), eos_id, dtype=idx.dtype, device=device)
        idx_padded = torch.cat([padding, idx], dim=1)
    else:
        idx_padded = idx[:, -max_len:]

    # Run prefill forward
    logits, _, concept_db, memory_state, *_ = model(idx_padded, idx_padded, start_pos=0)

    # Sample first token
    last_logits = logits[:, -1, :]
    idx_next = _full_sample(
        last_logits,
        generated_ids,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        dry_multiplier=dry_multiplier,
        dry_base=dry_base,
        dry_allowed_length=dry_allowed_length,
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
        
        idx_next = _full_sample(
            logits[:, -1, :],
            generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
        )
        
        token_id = idx_next.item()
        yield token_id
        if token_id == eos_id:
            break
        generated_ids.append(token_id)
        idx = torch.cat([idx, idx_next], dim=1)
        current_pos += 1



# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADING
# ══════════════════════════════════════════════════════════════════════

def _select_checkpoint_file(ckpt_path: str) -> Optional[str]:
    latest_pt = os.path.join(ckpt_path, "lasmoid_latest.pt")
    if os.path.exists(latest_pt):
        return latest_pt

    step_files = glob.glob(os.path.join(ckpt_path, "lasmoid_step_*.pt"))
    if step_files:
        def step_num(path: str) -> int:
            stem = os.path.basename(path)
            match = re.search(r"lasmoid_step_(\d+)\.pt$", stem)
            return int(match.group(1)) if match else -1

        return max(step_files, key=step_num)

    final = os.path.join(ckpt_path, "lasmoid_final.pt")
    if os.path.exists(final):
        return final

    return None


def load_checkpoint_and_model(ckpt_path: str, config_path: str, device: str, allow_partial_load: bool = False) -> tuple:
    ckpt_file = _select_checkpoint_file(ckpt_path)
            
    model_args = None
    state_dict = None
    if ckpt_file:
        try:
            sd = torch.load(ckpt_file, map_location=device, weights_only=False)
            state_dict = sd.get("model_state_dict", sd)
            model_args = sd.get("model_args")
            print(f"[generate] Found checkpoint: {ckpt_file}")
        except Exception as e:
            print(f"[generate] Failed to load checkpoint file directly: {e}")
            
    config_dict = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)

    if model_args is None:
        from dataclasses import fields
        valid_fields = {f.name for f in fields(ModelArgs)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        model_args = ModelArgs(**filtered_config)
        print(f"[generate] Loaded config from {config_path}")
    else:
        for key in (
            "vocab_size",
            "think_token_id",
            "answer_token_id",
            "cot_exit_confidence",
            "expert_capacity_factor",
            "moe_load_balance_coeff",
            "ssm_chunk_size",
            "ssm_dt_min",
            "ssm_dt_max",
            "ssm_dt_init_floor",
            "ssm_n_groups",
        ):
            if key in config_dict:
                setattr(model_args, key, config_dict[key])
        
    model = Lasmoid(model_args).to(device)

    def _prepare_state_dict_for_model(sd: dict, target_model: Lasmoid) -> None:
        """Pad tokenizer tensors and drop stale architecture tensors before loading."""
        model_sd = target_model.state_dict()
        target_rows = target_model.args.vocab_size
        vocab_keys = [
            key for key, target in model_sd.items()
            if target.ndim == 2 and target.shape[0] == target_rows and key in sd
        ]

        for key in vocab_keys:
            tensor = sd.get(key)
            if tensor is None or tensor.ndim != 2 or tensor.shape[0] == target_rows:
                continue
            if tensor.shape[0] > target_rows:
                sd[key] = tensor[:target_rows].contiguous()
                print(f"[generate] Trimmed {key} from {tensor.shape[0]} to {target_rows} rows.")
                continue

            target_tensor = model_sd[key].detach().to(tensor.device, dtype=tensor.dtype)
            target_tensor[:tensor.shape[0]].copy_(tensor)
            sd[key] = target_tensor
            print(f"[generate] Expanded {key} from {tensor.shape[0]} to {target_rows} rows.")

        dropped = []
        for key in list(sd.keys()):
            target = model_sd.get(key)
            if target is None:
                dropped.append(key)
                sd.pop(key)
                continue
            if sd[key].shape != target.shape:
                dropped.append(key)
                sd.pop(key)
        if dropped:
            print(f"[generate] Skipped {len(dropped)} stale checkpoint tensors with incompatible shapes/names.")

    def _validate_state_dict_compatibility(sd: dict, target_model: Lasmoid) -> list[str]:
        model_sd = target_model.state_dict()
        param_keys = {name for name, _ in target_model.named_parameters()}
        buffer_keys = {name for name, _ in target_model.named_buffers()}
        issues = []

        for key, tensor in sd.items():
            target = model_sd.get(key)
            if target is None:
                if key not in buffer_keys:
                    issues.append(f"unexpected tensor `{key}`")
                continue
            if key in param_keys and tensor.shape != target.shape:
                issues.append(f"shape mismatch for `{key}`: checkpoint {tuple(tensor.shape)} vs model {tuple(target.shape)}")

        for key in param_keys:
            if key not in sd:
                issues.append(f"missing parameter `{key}`")

        return issues
    
    if state_dict is not None:
        if allow_partial_load:
            print("[generate] WARNING: partial checkpoint load enabled; outputs are not expected to be coherent until retrained/fine-tuned.")
            _prepare_state_dict_for_model(state_dict, model)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[generate] Initialised {len(missing)} new tensors from current model defaults.")
            if unexpected:
                print(f"[generate] Ignored {len(unexpected)} unexpected tensors.")
            print(f"[generate] Successfully loaded weights from checkpoint.")
        else:
            issues = _validate_state_dict_compatibility(state_dict, model)
            if issues:
                issue_text = "\n  - " + "\n  - ".join(issues[:20])
                raise RuntimeError(
                    "Checkpoint is incompatible with the current Lasmoid architecture.\n"
                    f"First issues:{issue_text}\n"
                    "Re-run with `--allow-partial-load` only for debugging or use a compatible fresh checkpoint."
                )
            model.load_state_dict(state_dict, strict=True)
            print(f"[generate] Successfully loaded weights from checkpoint.")
    else:
        print("[generate] WARNING: Running with random initialisation.")
        
    return model, model_args


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

    model, args = load_checkpoint_and_model(ckpt_path, config, device, allow_partial_load=allow_partial_load)
    model.eval()

    Linear.dtype     = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    Linear.scale_fmt = getattr(args, "scale_fmt", None)

    lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enc = transformers.PreTrainedTokenizerFast.from_pretrained(lasmoid_dir, fix_mistral_regex=True)
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
        print(f"\n{'─'*60}")
        print(" Lasmoid — Interactive Shell (2026 Samplers)")
        print(f"{'─'*60}")
        print(f" Samplers: {sampler_summary}")
        print(" Commands: /exit  /clear  /sampler")
        print(f"{'─'*60}\n")

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
            context       = " ".join(history) + " "
            prompt_tokens = [enc.encode(context)]

            completion_tokens = []
            decoded_text = ""
            start_time = time.time()
            for token_id in generate_stream(model, prompt_tokens[0], max_new_tokens, eos_token_id, **sampler_cfg):
                completion_tokens.append(token_id)
                new_decoded = enc.decode(completion_tokens)
                print(new_decoded[len(decoded_text):], end="", flush=True)
                decoded_text = new_decoded
            print()
            elapsed_time = time.time() - start_time
            tps = len(completion_tokens) / max(elapsed_time, 1e-6)
            print(f"\033[90m[{len(completion_tokens)} tokens generated in {elapsed_time:.2f}s | {tps:.1f} TPS]\033[0m")
            history.append(decoded_text.strip())

    else:
        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}")
            return
        with open(input_file) as f:
            prompts = [l.strip() for l in f if l.strip()]

        print(f"[generate] Batch mode: {len(prompts)} prompts")
        prompt_tokens     = [enc.encode(p) for p in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, eos_token_id, **sampler_cfg)

        for p, ct in zip(prompts, completion_tokens):
            print(f"Prompt:     {p}")
            print(f"Completion: {enc.decode(ct)}")
            print("─" * 50)


if __name__ == "__main__":
    parser = ArgumentParser(description="Lasmoid inference — 2026 samplers")
    parser.add_argument("--ckpt-path",      type=str, required=True)
    parser.add_argument("--config",         type=str, required=True)
    parser.add_argument("--input-file",     type=str, default="")
    parser.add_argument("--interactive",    action="store_true")
    parser.add_argument("--max-new-tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top-k",          type=int,   default=0,    help="0 = disabled")
    parser.add_argument("--top-p",          type=float, default=1.0,  help="1.0 = disabled")
    parser.add_argument("--min-p",          type=float, default=0.05, help="2026 default; 0 = disabled")
    parser.add_argument("--xtc-probability",type=float, default=0.0,  help="XTC per-step probability")
    parser.add_argument("--xtc-threshold",  type=float, default=0.1,  help="XTC prob threshold")
    parser.add_argument("--dry-multiplier", type=float, default=0.0,  help="DRY strength; 0 = disabled")
    parser.add_argument("--device",         type=str,   default=None, help="Device to use: cpu, mps, or cuda")
    parser.add_argument("--allow-partial-load", action="store_true", help="Allow incompatible checkpoint tensors to be expanded/skipped for debugging")
    a = parser.parse_args()

    assert a.input_file or a.interactive, "Specify --input-file or --interactive"
    main(
        a.ckpt_path, a.config, a.input_file, a.interactive,
        a.max_new_tokens, a.temperature,
        a.top_k, a.top_p, a.min_p,
        a.xtc_probability, a.xtc_threshold, a.dry_multiplier,
        allow_partial_load=a.allow_partial_load,
        device=a.device,
    )
