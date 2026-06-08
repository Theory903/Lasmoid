"""
Lasmoid — train.py
==================
Training script for the Lasmoid CQRS Hybrid Concept Transformer model.
Trains on Tiny Shakespeare using transformers PreTrainedTokenizerFast and the Muon/AdamW optimizer suite.
Supports DDP distributed training, mixed-precision (AMP), and gradient clipping.

Delegates Muon optimiser to optimizer.py and reward functions to reward.py.
"""

import os
import sys

# Optimize PyTorch memory allocations to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import urllib.request
import json
import torch
if torch.cuda.is_available():
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except Exception:
        pass
import warnings

# Monkey-patch torch.bf16 for compatibility with older/custom PyTorch versions in transformers
if not hasattr(torch, "bf16"):
    torch.bf16 = torch.bfloat16

# Suppress incorrect regex warnings from tokenizers at runtime
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

import torch.nn as nn
import torch.nn.functional as F
import transformers
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import Lasmoid, ModelArgs, Linear, compute_loss, compute_grpo_loss
from inference.debug import (
    set_debug_step,
    reset_debug_buffer,
    dump_compressor_summary,
    dump_attention_summary,
    write_debug_jsonl,
    visualize_compression_patterns,
)
from optimizer import Muon, get_lr_multiplier
from reward import reasoning_self_evolution_reward
from grpo_stability import compute_group_advantages, safe_reward


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/current")
    parser.add_argument("--save_interval", type=int, default=100)
    # GRPO RL options
    parser.add_argument(
        "--rl_grpo", action="store_true", help="Enable GRPO RL Alignment"
    )
    parser.add_argument("--group_size", type=int, default=4, help="GRPO group size")
    parser.add_argument(
        "--clip_eps", type=float, default=0.2, help="GRPO PPO clipping parameter"
    )
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.01,
        help="GRPO KL regularisation coefficient",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Force device (e.g. cpu, mps, cuda)"
    )
    # Binary dataset options
    parser.add_argument(
        "--train_bin", type=str, default=None, help="Path to pre-tokenized train.bin"
    )
    parser.add_argument(
        "--val_bin", type=str, default=None, help="Path to pre-tokenized val.bin"
    )
    parser.add_argument(
        "--train_meta",
        type=str,
        default=None,
        help="Path to metadata json for train split",
    )
    parser.add_argument(
        "--val_meta", type=str, default=None, help="Path to metadata json for val split"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path or HuggingFace identifier for the tokenizer",
    )
    args_cli = parser.parse_args()

    # DDP Distributed Bootstrapping
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = (
            f"cuda:{ddp_local_rank}"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        if "cuda" in device:
            torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        # Seed differently per rank for sharded batch generation
        torch.manual_seed(42 + ddp_rank)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        torch.manual_seed(42)

        if args_cli.device is not None:
            device = args_cli.device
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if master_process:
        print(f"Using device: {device.upper()} (DDP: {ddp})")

    # Setup Checkpoint Directory
    if master_process:
        os.makedirs(args_cli.checkpoint_dir, exist_ok=True)

    # Load Tokenizer (Hugging Face or Local Fast tokenizer)
    lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = args_cli.tokenizer_path if args_cli.tokenizer_path else lasmoid_dir
    if master_process:
        print(f"Loading tokenizer from: {tokenizer_path}")
    try:
        try:
            enc = transformers.AutoTokenizer.from_pretrained(
                tokenizer_path, fix_mistral_regex=True
            )
        except Exception:
            enc = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        if master_process:
            print(
                f"Standard AutoTokenizer failed to load ({e}). Falling back to PreTrainedTokenizerFast..."
            )
        try:
            enc = transformers.PreTrainedTokenizerFast.from_pretrained(
                tokenizer_path, fix_mistral_regex=True
            )
        except Exception as e_fast:
            try:
                enc = transformers.PreTrainedTokenizerFast.from_pretrained(
                    tokenizer_path
                )
            except Exception as e_fast2:
                if master_process:
                    print(f"Error: Failed to load tokenizer from '{tokenizer_path}'.")
                    print(f"Detailed error: {e_fast2}")
                    print(
                        "\nIf you are loading a gated Hugging Face model (such as Gemma), make sure:"
                    )
                print(
                    "1. You have accepted the license terms on Hugging Face model page."
                )
                print(
                    "2. You are logged in using 'huggingface-cli login' or have set 'HF_TOKEN' environment variable."
                )
            if ddp:
                dist.destroy_process_group()
            sys.exit(1)

    eos_token_id = enc.eos_token_id if enc.eos_token_id is not None else 1

    # Load Dataset
    train_metadata = None
    val_metadata = None

    if args_cli.train_bin is not None:
        if master_process:
            print(f"Loading binary train dataset from: {args_cli.train_bin}")
        import numpy as np

        train_tokens = np.fromfile(args_cli.train_bin, dtype=np.uint32)
        train_data = torch.from_numpy(train_tokens.astype(np.int64))

        if args_cli.val_bin is not None:
            if master_process:
                print(f"Loading binary val dataset from: {args_cli.val_bin}")
            val_tokens = np.fromfile(args_cli.val_bin, dtype=np.uint32)
            val_data = torch.from_numpy(val_tokens.astype(np.int64))
        else:
            val_data = train_data

        if args_cli.train_meta and os.path.exists(args_cli.train_meta):
            with open(args_cli.train_meta, "r") as f:
                train_metadata = json.load(f)
        if args_cli.val_meta and os.path.exists(args_cli.val_meta):
            with open(args_cli.val_meta, "r") as f:
                val_metadata = json.load(f)
    else:
        # Default fallback to Tiny Shakespeare text file
        dataset_path = "input.txt"
        if not os.path.exists(dataset_path):
            parent_dataset = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input.txt"
            )
            if os.path.exists(parent_dataset):
                dataset_path = parent_dataset
            else:
                if master_process:
                    print("Downloading Tiny Shakespeare dataset...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                    dataset_path,
                )

        if master_process:
            print(f"Loading text dataset from: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()

        data = torch.tensor(enc.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

    # Load ModelArgs
    config_path = os.path.join(lasmoid_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    from dataclasses import fields

    valid_fields = {f.name for f in fields(ModelArgs)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    model_args = ModelArgs(**filtered_config)
    tokenizer_vocab_size = len(enc)
    if model_args.vocab_size < tokenizer_vocab_size:
        if master_process:
            print(
                f"Expanding vocab_size from {model_args.vocab_size} to tokenizer size {tokenizer_vocab_size}"
            )
        model_args.vocab_size = tokenizer_vocab_size
    # Ensure think/answer special tokens exist in vocabulary, dynamically adding them if not
    think_token_id = enc.convert_tokens_to_ids("<think>")
    is_unk = False
    if think_token_id is None:
        is_unk = True
    elif hasattr(enc, "unk_token_id") and think_token_id == enc.unk_token_id:
        is_unk = True
    elif hasattr(enc, "unk_token") and enc.unk_token is not None:
        try:
            unk_id = enc.convert_tokens_to_ids(enc.unk_token)
            if think_token_id == unk_id:
                is_unk = True
        except:
            pass

    if is_unk:
        if master_process:
            print(
                "Special tokens '<think>' / '<answer>' not found in tokenizer. Adding them..."
            )
        enc.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<think>",
                    "</think>",
                    "<answer>",
                    "</answer>",
                ]
            }
        )
        think_token_id = enc.convert_tokens_to_ids("<think>")

    if isinstance(think_token_id, int) and think_token_id >= 0:
        model_args.think_token_id = think_token_id

    answer_token_id = enc.convert_tokens_to_ids("<answer>")
    model_args.answer_token_id = (
        answer_token_id
        if isinstance(answer_token_id, int) and 0 <= answer_token_id < len(enc)
        else eos_token_id
    )
    max_seq_len = model_args.max_seq_len

    def get_batch(split):
        if args_cli.train_bin is not None:
            d = train_data if split == "train" else val_data
            meta = train_metadata if split == "train" else val_metadata

            num_seqs = len(d) // max_seq_len
            ix = torch.randint(0, num_seqs, (args_cli.batch_size,))

            x_list, y_list, mask_list = [], [], []
            for idx in ix.tolist():
                seq_start = idx * max_seq_len
                seq_tokens = d[seq_start : seq_start + max_seq_len]
                x_list.append(seq_tokens)

                # Shift left to predict next token, padding the last one with eos
                y_tokens = torch.cat(
                    [seq_tokens[1:], torch.tensor([eos_token_id], dtype=torch.long)]
                )
                y_list.append(y_tokens)

                loss_mask = torch.ones(max_seq_len, dtype=torch.float32)
                if meta is not None and idx < len(meta):
                    split_idx = meta[idx].get("split_idx", 0)
                    loss_mask[:split_idx] = 0.0
                else:
                    loss_mask[: max_seq_len // 2] = 0.0
                mask_list.append(loss_mask)

            x = torch.stack(x_list).to(device)
            y = torch.stack(y_list).to(device)
            loss_mask = torch.stack(mask_list).to(device)
            return x, y, loss_mask
        else:
            d = train_data if split == "train" else val_data
            ix = torch.randint(len(d) - max_seq_len - 1, (args_cli.batch_size,))
            x = torch.stack([d[i : i + max_seq_len] for i in ix]).to(device)
            y = torch.stack([d[i + 1 : i + max_seq_len + 1] for i in ix]).to(device)

            loss_mask = torch.ones_like(x, dtype=torch.float32)
            for b in range(args_cli.batch_size):
                tokens = x[b].tolist()
                try:
                    text_seq = enc.decode(tokens)
                    if ":" in text_seq:
                        colon_idx = text_seq.find(":")
                        prompt_text = text_seq[:colon_idx]
                        split_idx = len(enc.encode(prompt_text))
                        split_idx = min(max(split_idx, 1), max_seq_len - 1)
                        loss_mask[b, :split_idx] = 0.0
                    else:
                        loss_mask[b, : max_seq_len // 2] = 0.0
                except:
                    loss_mask[b, : max_seq_len // 2] = 0.0
            return x, y, loss_mask

    # 2. Model Initialization
    model = Lasmoid(model_args).to(device)
    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank] if "cuda" in device else None,
            find_unused_parameters=True,
        )
        raw_model = model.module
    else:
        raw_model = model

    # Enable gradient checkpointing to save memory on Kaggle GPUs
    raw_model.gradient_checkpointing = True

    if master_process:
        print(
            f"Model initialized with dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}, vocab_size={model_args.vocab_size}"
        )

    # Partition Parameters
    muon_params, adamw_params = [], []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        # 2D weights updated by Muon (excluding embedding layers, heads, gate routers, and hyper-connections)
        if (
            len(p.shape) == 2
            and "emb" not in name
            and "head" not in name
            and "adj" not in name
            and "gate" not in name
            and "hc" not in name
        ):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon = Muon(muon_params, lr=2e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=args_cli.learning_rate)

    if master_process:
        print("Training initiated...")
    model.train()

    import contextlib

    device_type = (
        "cuda" if "cuda" in str(device) else ("cpu" if "cpu" in str(device) else None)
    )
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type
        else contextlib.nullcontext()
    )

    try:
        for step in range(args_cli.max_iters):
            # LR Scheduling (Warmup + Cosine Decay)
            lr_mult = get_lr_multiplier(step, args_cli.max_iters, args_cli.warmup_steps)
            for g in opt_muon.param_groups:
                g["lr"] = 2e-3 * lr_mult
            for g in opt_adamw.param_groups:
                g["lr"] = args_cli.learning_rate * lr_mult

            # ── Debug-visualisation hooks ──────────────────────────────────
            set_debug_step(step)
            reset_debug_buffer()

            if args_cli.rl_grpo:
                # ─── GRPO Reinforcement Learning Step ───
                # 1. Get batch of sequences, extract prompt sections
                xb, yb, _ = get_batch("train")
                prompt_len = max_seq_len // 2
                prompts = xb[:, :prompt_len]  # [B, prompt_len]

                # Repeat prompts to group size G to execute in a single forward pass
                G = args_cli.group_size
                prompts_expanded = prompts.repeat_interleave(
                    G, dim=0
                )  # [B * G, prompt_len]

                # Generate completions autoregressively
                raw_model.eval()
                with torch.no_grad():
                    completion_len = max_seq_len - prompt_len
                    # pad_token = 1 for the new tokenizer
                    generated_seqs = raw_model.generate(
                        prompts_expanded,
                        completion_len,
                        temperature=1.0,
                        pad_token=eos_token_id,
                    )
                raw_model.train()

                # Extract completions
                completions = generated_seqs[:, prompt_len:]

                # 2. Score completions via reasoning/self-evolution reward alignment
                rewards = []
                for b_g in range(args_cli.batch_size * G):
                    comp_tokens = completions[b_g].tolist()
                    text = enc.decode(comp_tokens)
                    rewards.append(safe_reward(text, reasoning_self_evolution_reward))

                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

                # 3. Compute relative advantages within each prompt group
                advantages = compute_group_advantages(rewards, G)

                # 4. Old policy logprob calculation
                raw_model.eval()
                with torch.no_grad():
                    with autocast_ctx:
                        logits_old, _, _, _, _, _, _, _ = model(
                            generated_seqs, generated_seqs
                        )
                        logits_old_shifted = logits_old[:, :-1, :]
                        targets_shifted = generated_seqs[:, 1:]

                        logprobs_old = F.log_softmax(logits_old_shifted, dim=-1)
                        old_logprobs = logprobs_old.gather(
                            2, targets_shifted.unsqueeze(-1)
                        ).squeeze(-1)
                raw_model.train()

                # 5. Policy optimization step
                opt_muon.zero_grad()
                opt_adamw.zero_grad()

                with autocast_ctx:
                    logits, _, _, _, routing_maps, _, adjs, event_probs = model(
                        generated_seqs, generated_seqs
                    )
                    logits_shifted = logits[:, :-1, :]
                    targets_shifted = generated_seqs[:, 1:]

                    # Loss mask: only calculate loss on the generated completion part
                    grpo_loss_mask = torch.zeros_like(
                        targets_shifted, dtype=torch.float32
                    )
                    grpo_loss_mask[:, prompt_len - 1 :] = 1.0

                    total_grpo, policy_loss, kl_loss = compute_grpo_loss(
                        logits_shifted,
                        targets_shifted,
                        advantages,
                        old_logprobs,
                        loss_mask=grpo_loss_mask,
                        clip_eps=args_cli.clip_eps,
                        kl_coeff=args_cli.kl_coeff,
                    )

                    # Concept, routing, and self-modeling auxiliary losses
                    vq_losses = [raw_model.last_vq_loss]
                    concept_aux_loss = 0.0
                    for r in routing_maps:
                        mean_routing = r.mean(dim=(0, 1))
                        concept_aux_loss += mean_routing.var()
                    total_vq_loss = sum(vq_losses)
                    graph_loss = (
                        sum(torch.mean(torch.abs(a)) for a in adjs)
                        if adjs
                        else torch.tensor(0.0, device=device)
                    )

                    pred_coeff = getattr(model_args, "predictive_coding_coeff", 0.01)
                    token_concept_coeff = getattr(
                        model_args, "token_concept_loss_coeff", 0.05
                    )
                    loss = (
                        total_grpo
                        + raw_model.last_moe_loss
                        + (0.5 * concept_aux_loss)
                        + total_vq_loss
                        + (0.01 * graph_loss)
                        + pred_coeff * raw_model.last_pred_loss
                        + token_concept_coeff * raw_model.last_token_concept_loss
                    )

                if not torch.isfinite(loss):
                    if master_process:
                        print(
                            f"Step {step:4d} | non-finite GRPO loss, skipping optimizer step",
                            flush=True,
                        )
                    opt_muon.zero_grad(set_to_none=True)
                    opt_adamw.zero_grad(set_to_none=True)
                    continue

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                opt_muon.step()
                opt_adamw.step()

                total_step_loss = loss.item()
                total_step_ce = policy_loss.item()
                total_step_mtp = kl_loss.item()
                total_step_vq = (
                    rewards.mean().item()
                )  # Display average reward for RL step display

            else:
                # ─── Standard SFT step with loss masking ───
                opt_muon.zero_grad()
                opt_adamw.zero_grad()

                total_step_loss = 0.0
                total_step_ce = 0.0
                total_step_mtp = 0.0
                total_step_vq = 0.0
                total_step_pred = 0.0

                # Gradient Accumulation Loop
                for micro in range(args_cli.grad_accum):
                    xb, yb, loss_mask = get_batch("train")

                    # Forward pass under mixed precision
                    with autocast_ctx:
                        logits, mtp_logits, _, _, routing_maps, _, adjs, event_probs = (
                            model(xb, xb)
                        )

                        # Autoregressive loss computation with mask
                        main_loss = compute_loss(
                            logits,
                            yb,
                            routing_maps,
                            [raw_model.last_vq_loss],
                            adjs,
                            event_probs,
                            loss_mask=loss_mask,
                            moe_aux_loss=raw_model.last_moe_loss,
                            moe_aux_coeff=getattr(model_args, "moe_aux_coeff", 1.0),
                            token_concept_loss=raw_model.last_token_concept_loss,
                            token_concept_coeff=getattr(
                                model_args, "token_concept_loss_coeff", 0.05
                            ),
                            ignore_index=getattr(model_args, "loss_ignore_index", -100),
                        )

                        # Next-token prediction loss for display
                        ce_loss_next = F.cross_entropy(
                            logits.view(-1, model_args.vocab_size),
                            yb.view(-1),
                            ignore_index=getattr(model_args, "loss_ignore_index", -100),
                        )

                        # MTP (next-next token) loss
                        ce_loss_mtp = torch.tensor(0.0, device=device)
                        if mtp_logits is not None:
                            ce_loss_mtp = F.cross_entropy(
                                mtp_logits.view(-1, model_args.vocab_size),
                                yb[:, 1:].contiguous().view(-1),
                                ignore_index=getattr(
                                    model_args, "loss_ignore_index", -100
                                ),
                            )

                        pred_coeff = getattr(
                            model_args, "predictive_coding_coeff", 0.01
                        )
                        mtp_coeff = getattr(model_args, "mtp_loss_coeff", 0.3)
                        loss = (
                            main_loss
                            + mtp_coeff * ce_loss_mtp
                            + pred_coeff * raw_model.last_pred_loss
                        )
                        loss = loss / args_cli.grad_accum

                    if not torch.isfinite(loss):
                        if master_process:
                            print(
                                f"Step {step:4d} micro {micro} | non-finite SFT loss, skipping microbatch",
                                flush=True,
                            )
                        opt_muon.zero_grad(set_to_none=True)
                        opt_adamw.zero_grad(set_to_none=True)
                        continue

                    loss.backward()

                    total_step_loss += loss.item() * args_cli.grad_accum
                    total_step_ce += ce_loss_next.item()
                    total_step_mtp += ce_loss_mtp.item()
                    total_step_vq += raw_model.last_vq_loss.item()
                    total_step_pred += raw_model.last_pred_loss.item()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Step optimizers
                opt_muon.step()
                opt_adamw.step()

            if step % 20 == 0 and master_process:
                if args_cli.rl_grpo:
                    print(
                        f"Step {step:4d} | Total Loss: {total_step_loss:.4f} | Policy Loss: {total_step_ce:.4f} | KL Loss: {total_step_mtp:.4f} | Avg Reward: {total_step_vq:.4f} | LR Scale: {lr_mult:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"Step {step:4d} | Total Loss: {total_step_loss:.4f} | CE (t+1): {total_step_ce:.4f} | MTP (t+2): {total_step_mtp:.4f} | VQ Loss: {total_step_vq:.4f} | Pred Loss: {total_step_pred:.4f} | LR Scale: {lr_mult:.4f}",
                        flush=True,
                    )

                # ── Debug visualisation output ──────────────────────────
                dump_compressor_summary()
                dump_attention_summary()

            # Checkpoint Saving
            if step > 0 and step % args_cli.save_interval == 0 and master_process:
                ckpt_path = os.path.join(
                    args_cli.checkpoint_dir, f"lasmoid_step_{step}.pt"
                )
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": raw_model.state_dict(),
                        "opt_muon_state": opt_muon.state_dict(),
                        "opt_adamw_state": opt_adamw.state_dict(),
                        "args": model_args,
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved to {ckpt_path}")
                # ── Debug persistence ────────────────────────────────
                write_debug_jsonl()
                visualize_compression_patterns()
    except KeyboardInterrupt:
        if master_process:
            print(
                "\nTraining interrupted by user. Saving latest parameters before exit..."
            )
            interrupted_path = os.path.join(
                args_cli.checkpoint_dir, "lasmoid_interrupted.pt"
            )
            torch.save(raw_model.state_dict(), interrupted_path)
            print(f"Interrupted weights saved to {interrupted_path}")
        if ddp:
            dist.destroy_process_group()
        sys.exit(0)

    # Save final model
    if master_process:
        final_path = os.path.join(args_cli.checkpoint_dir, "lasmoid_final.pt")
        torch.save(raw_model.state_dict(), final_path)
        print(f"Final model parameters saved to {final_path}")

    # Distributed Clean up
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
