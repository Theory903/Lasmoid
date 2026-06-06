"""
Lasmoid — train.py
==================
Training script for the Lasmoid CQRS Hybrid Concept Transformer model.
Trains on Tiny Shakespeare using transformers PreTrainedTokenizerFast and the Muon/AdamW optimizer suite.
Supports DDP distributed training, mixed-precision (AMP), and gradient clipping.
"""

import os
import sys
import argparse
import urllib.request
import math
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional

# Add root folder to sys.path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import Lasmoid, ModelArgs, Linear, compute_loss, compute_grpo_loss

# ══════════════════════════════════════════════════════════════════════
# MUON OPTIMIZER
# ══════════════════════════════════════════════════════════════════════

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            for p in group['params']:
                if p.grad is None: 
                    continue
                grad, state = p.grad, self.state[p]
                if not torch.isfinite(grad).all():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                if len(p.shape) == 2:  # Newton-Schulz for 2D matrices
                    G = buf.clone()
                    transposed = G.shape[0] > G.shape[1]
                    if transposed:
                        G = G.T
                    a, b, c = 3.4445, -4.7750, 2.0315
                    X = G / (G.norm() + 1e-8)
                    for _ in range(5):
                        A = X @ X.T
                        B = A @ X
                        X = a * X + b * B + c * A @ B
                    update = X * (max(p.shape[0], p.shape[1]) ** 0.5)
                    if transposed:
                        update = update.T
                else:
                    update = buf
                p.add_(update, alpha=-lr)


def get_lr_multiplier(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _extract_answer_text(response: str) -> str:
    if "</think>" in response:
        return response.split("</think>", 1)[-1].strip()
    if "</Summary>" in response:
        return response.split("</Summary>", 1)[-1].strip()
    return response.strip()


def _normalise_answer(text: str) -> str:
    text = _extract_answer_text(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.+\\-/% ]", "", text)
    return text


def reasoning_self_evolution_reward(response: str, gt_answer: Optional[str] = None) -> float:
    """
    Advanced Reward Function for Lasmoid Self-Evolution (AGI-Frontier Alignment).
    Integrates:
      - Chain-of-Thought (DeepSeek-R1 style)
      - Scientific Reasoning (AlphaFold3/Nemotron style)
      - Information Density (Gemma/Superhuman style)
      - Structural Complexity (GPT-5.5/Opus style)
    """
    reward = 0.0
    
    # 1. Structural reasoning markers
    has_think = "<think>" in response and "</think>" in response
    has_parallel = "<Parallel>" in response and "</Parallel>" in response
    answer = _extract_answer_text(response)

    if has_think:
        reward += 1.0  # Increased base for CoT
        trace = response.split("<think>", 1)[-1].split("</think>", 1)[0]
    elif has_parallel:
        reward += 0.8  # Parallel reasoning branch
        trace = response.split("<Parallel>", 1)[-1].split("</Parallel>", 1)[0]
    else:
        trace = response

    # 2. Information Density & Reasoning Quality
    trace_len = len(trace.strip())
    # Reward sweet spot for reasoning depth (neither too brief nor rambling)
    if 150 <= trace_len <= 2000:
        reward += 0.8
    elif trace_len > 0:
        reward += 0.3

    # 3. Scientific & Logical Markers (Nemotron/AlphaFold/Superhuman alignment)
    lower_trace = trace.lower()
    logic_markers = ["verify", "hypothesis", "empirical", "deduction", "axiom", "topology", "manifold", "synthesis"]
    scientific_markers = ["sequence", "fold", "energy", "minimum", "gradient", "stochastic", "converge", "optimization"]
    
    found_logic = sum(1 for m in logic_markers if m in lower_trace)
    found_sci = sum(1 for m in scientific_markers if m in lower_trace)
    
    reward += min(0.6, found_logic * 0.15)
    reward += min(0.6, found_sci * 0.15)

    # 4. Multi-agentic / Collaborative patterns (gpt-oss style)
    if any(x in lower_trace for x in ["collaborate", "consensus", "delegate", "orchestrate"]):
        reward += 0.4

    # 5. Accuracy Alignment
    if answer:
        reward += 0.5
    if gt_answer:
        gt = _normalise_answer(gt_answer)
        # Check for numeric or exact string matches in the answer segment
        pred = _normalise_answer(answer if answer else response)
        if gt and (gt in pred or pred in gt):
            reward += 3.0

    # 6. Negative Constraints (Repetition & Rambling)
    words = re.findall(r"\b\w+\b", response.lower())
    if len(words) > 30:
        # N-gram repetition penalty (trigram)
        repeats = sum(1 for a, b, c in zip(words, words[1:], words[2:]) if a == b == c)
        reward -= min(2.0, repeats * 0.4)
    
    if len(response) > 8000:
        reward -= 1.0  # Heavy penalty for extreme rambling

    return float(reward)


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
    parser.add_argument("--rl_grpo", action="store_true", help="Enable GRPO RL Alignment")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO group size")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="GRPO PPO clipping parameter")
    parser.add_argument("--kl_coeff", type=float, default=0.01, help="GRPO KL regularisation coefficient")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g. cpu, mps, cuda)")
    # Binary dataset options
    parser.add_argument("--train_bin", type=str, default=None, help="Path to pre-tokenized train.bin")
    parser.add_argument("--val_bin", type=str, default=None, help="Path to pre-tokenized val.bin")
    parser.add_argument("--train_meta", type=str, default=None, help="Path to metadata json for train split")
    parser.add_argument("--val_meta", type=str, default=None, help="Path to metadata json for val split")
    args_cli = parser.parse_args()

    # DDP Distributed Bootstrapping
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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

    # Load PreTrainedTokenizerFast
    lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enc = transformers.PreTrainedTokenizerFast.from_pretrained(lasmoid_dir, fix_mistral_regex=True)
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
            with open(args_cli.train_meta, 'r') as f:
                train_metadata = json.load(f)
        if args_cli.val_meta and os.path.exists(args_cli.val_meta):
            with open(args_cli.val_meta, 'r') as f:
                val_metadata = json.load(f)
    else:
        # Default fallback to Tiny Shakespeare text file
        dataset_path = "input.txt"
        if not os.path.exists(dataset_path):
            parent_dataset = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input.txt")
            if os.path.exists(parent_dataset):
                dataset_path = parent_dataset
            else:
                if master_process:
                    print("Downloading Tiny Shakespeare dataset...")
                urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", dataset_path)
        
        if master_process:
            print(f"Loading text dataset from: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
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
            print(f"Expanding vocab_size from {model_args.vocab_size} to tokenizer size {tokenizer_vocab_size}")
        model_args.vocab_size = tokenizer_vocab_size
    think_token_id = enc.convert_tokens_to_ids("<think>")
    if isinstance(think_token_id, int) and think_token_id >= 0:
        model_args.think_token_id = think_token_id
    answer_token_id = enc.convert_tokens_to_ids("<answer>")
    model_args.answer_token_id = (
        answer_token_id
        if isinstance(answer_token_id, int) and 0 <= answer_token_id < model_args.vocab_size
        else eos_token_id
    )
    max_seq_len = model_args.max_seq_len

    def get_batch(split):
        if args_cli.train_bin is not None:
            d = train_data if split == 'train' else val_data
            meta = train_metadata if split == 'train' else val_metadata
            
            num_seqs = len(d) // max_seq_len
            ix = torch.randint(0, num_seqs, (args_cli.batch_size,))
            
            x_list, y_list, mask_list = [], [], []
            for idx in ix.tolist():
                seq_start = idx * max_seq_len
                seq_tokens = d[seq_start : seq_start + max_seq_len]
                x_list.append(seq_tokens)
                
                # Shift left to predict next token, padding the last one with eos
                y_tokens = torch.cat([seq_tokens[1:], torch.tensor([eos_token_id], dtype=torch.long)])
                y_list.append(y_tokens)
                
                loss_mask = torch.ones(max_seq_len, dtype=torch.float32)
                if meta is not None and idx < len(meta):
                    split_idx = meta[idx].get("split_idx", 0)
                    loss_mask[:split_idx] = 0.0
                else:
                    loss_mask[:max_seq_len // 2] = 0.0
                mask_list.append(loss_mask)
                
            x = torch.stack(x_list).to(device)
            y = torch.stack(y_list).to(device)
            loss_mask = torch.stack(mask_list).to(device)
            return x, y, loss_mask
        else:
            d = train_data if split == 'train' else val_data
            ix = torch.randint(len(d) - max_seq_len - 1, (args_cli.batch_size,))
            x = torch.stack([d[i:i+max_seq_len] for i in ix]).to(device)
            y = torch.stack([d[i+1:i+max_seq_len+1] for i in ix]).to(device)
            
            loss_mask = torch.ones_like(x, dtype=torch.float32)
            for b in range(args_cli.batch_size):
                tokens = x[b].tolist()
                try:
                    text_seq = enc.decode(tokens)
                    if ':' in text_seq:
                        colon_idx = text_seq.find(':')
                        prompt_text = text_seq[:colon_idx]
                        split_idx = len(enc.encode(prompt_text))
                        split_idx = min(max(split_idx, 1), max_seq_len - 1)
                        loss_mask[b, :split_idx] = 0.0
                    else:
                        loss_mask[b, :max_seq_len // 2] = 0.0
                except:
                    loss_mask[b, :max_seq_len // 2] = 0.0
            return x, y, loss_mask

    # 2. Model Initialization
    model = Lasmoid(model_args).to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank] if "cuda" in device else None, find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    if master_process:
        print(f"Model initialized with dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}, vocab_size={model_args.vocab_size}")

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
    device_type = "cuda" if "cuda" in str(device) else ("cpu" if "cpu" in str(device) else None)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type else contextlib.nullcontext()
    
    for step in range(args_cli.max_iters):
        # LR Scheduling (Warmup + Cosine Decay)
        lr_mult = get_lr_multiplier(step, args_cli.max_iters, args_cli.warmup_steps)
        for g in opt_muon.param_groups:
            g['lr'] = 2e-3 * lr_mult
        for g in opt_adamw.param_groups:
            g['lr'] = args_cli.learning_rate * lr_mult

        if args_cli.rl_grpo:
            # ─── GRPO Reinforcement Learning Step ───
            # 1. Get batch of sequences, extract prompt sections
            xb, yb, _ = get_batch('train')
            prompt_len = max_seq_len // 2
            prompts = xb[:, :prompt_len] # [B, prompt_len]
            
            # Repeat prompts to group size G to execute in a single forward pass
            G = args_cli.group_size
            prompts_expanded = prompts.repeat_interleave(G, dim=0) # [B * G, prompt_len]
            
            # Generate completions autoregressively
            raw_model.eval()
            with torch.no_grad():
                completion_len = max_seq_len - prompt_len
                # pad_token = 1 for the new tokenizer
                generated_seqs = raw_model.generate(prompts_expanded, completion_len, temperature=1.0, pad_token=eos_token_id)
            raw_model.train()
            
            # Extract completions
            completions = generated_seqs[:, prompt_len:]
            
            # 2. Score completions via reasoning/self-evolution reward alignment
            rewards = []
            for b_g in range(args_cli.batch_size * G):
                comp_tokens = completions[b_g].tolist()
                text = enc.decode(comp_tokens)
                rewards.append(reasoning_self_evolution_reward(text))
                
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # 3. Compute relative advantages within each prompt group
            rewards_grouped = rewards.view(args_cli.batch_size, G)
            mean_rewards = rewards_grouped.mean(dim=-1, keepdim=True)
            std_rewards = rewards_grouped.std(dim=-1, keepdim=True) + 1e-8
            advantages = ((rewards_grouped - mean_rewards) / std_rewards).view(-1) # [B * G]
            
            # 4. Old policy logprob calculation
            raw_model.eval()
            with torch.no_grad():
                with autocast_ctx:
                    logits_old, _, _, _, _, _, _, _ = model(generated_seqs, generated_seqs)
                    logits_old_shifted = logits_old[:, :-1, :]
                    targets_shifted = generated_seqs[:, 1:]
                    
                    logprobs_old = F.log_softmax(logits_old_shifted, dim=-1)
                    old_logprobs = logprobs_old.gather(2, targets_shifted.unsqueeze(-1)).squeeze(-1)
            raw_model.train()
            
            # 5. Policy optimization step
            opt_muon.zero_grad()
            opt_adamw.zero_grad()
            
            with autocast_ctx:
                logits, _, _, _, routing_maps, _, adjs, event_probs = model(generated_seqs, generated_seqs)
                logits_shifted = logits[:, :-1, :]
                targets_shifted = generated_seqs[:, 1:]
                
                # Loss mask: only calculate loss on the generated completion part
                grpo_loss_mask = torch.zeros_like(targets_shifted, dtype=torch.float32)
                grpo_loss_mask[:, prompt_len - 1:] = 1.0
                
                total_grpo, policy_loss, kl_loss = compute_grpo_loss(
                    logits_shifted, targets_shifted, advantages, old_logprobs,
                    loss_mask=grpo_loss_mask, clip_eps=args_cli.clip_eps, kl_coeff=args_cli.kl_coeff
                )
                
                # Concept, routing, and self-modeling auxiliary losses
                vq_losses = [raw_model.last_vq_loss]
                concept_aux_loss = 0.0
                for r in routing_maps:
                    mean_routing = r.mean(dim=(0, 1))
                    concept_aux_loss += mean_routing.var()
                total_vq_loss = sum(vq_losses)
                graph_loss = sum(torch.mean(torch.abs(a)) for a in adjs) if adjs else torch.tensor(0.0, device=device)
                
                pred_coeff = getattr(model_args, 'predictive_coding_coeff', 0.01)
                token_concept_coeff = getattr(model_args, 'token_concept_loss_coeff', 0.05)
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
                    print(f"Step {step:4d} | non-finite GRPO loss, skipping optimizer step", flush=True)
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
            total_step_vq = rewards.mean().item()  # Display average reward for RL step display

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
                xb, yb, loss_mask = get_batch('train')
                
                # Forward pass under mixed precision
                with autocast_ctx:
                    logits, mtp_logits, _, _, routing_maps, _, adjs, event_probs = model(xb, xb)
                    
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
                        token_concept_loss=raw_model.last_token_concept_loss,
                        token_concept_coeff=getattr(model_args, 'token_concept_loss_coeff', 0.05),
                    )
                    
                    # Next-token prediction loss for display
                    ce_loss_next = F.cross_entropy(logits.view(-1, model_args.vocab_size), yb.view(-1))
                    
                    # MTP (next-next token) loss
                    ce_loss_mtp = torch.tensor(0.0, device=device)
                    if mtp_logits is not None:
                        ce_loss_mtp = F.cross_entropy(
                            mtp_logits.view(-1, model_args.vocab_size),
                            yb[:, 1:].contiguous().view(-1)
                        )
                    
                    pred_coeff = getattr(model_args, 'predictive_coding_coeff', 0.01)
                    loss = main_loss + 0.3 * ce_loss_mtp + pred_coeff * raw_model.last_pred_loss
                    loss = loss / args_cli.grad_accum
                
                if not torch.isfinite(loss):
                    if master_process:
                        print(f"Step {step:4d} micro {micro} | non-finite SFT loss, skipping microbatch", flush=True)
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
                print(f"Step {step:4d} | Total Loss: {total_step_loss:.4f} | Policy Loss: {total_step_ce:.4f} | KL Loss: {total_step_mtp:.4f} | Avg Reward: {total_step_vq:.4f} | LR Scale: {lr_mult:.4f}", flush=True)
            else:
                print(f"Step {step:4d} | Total Loss: {total_step_loss:.4f} | CE (t+1): {total_step_ce:.4f} | MTP (t+2): {total_step_mtp:.4f} | VQ Loss: {total_step_vq:.4f} | Pred Loss: {total_step_pred:.4f} | LR Scale: {lr_mult:.4f}", flush=True)

        # Checkpoint Saving
        if step > 0 and step % args_cli.save_interval == 0 and master_process:
            ckpt_path = os.path.join(args_cli.checkpoint_dir, f"lasmoid_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': raw_model.state_dict(),
                'opt_muon_state': opt_muon.state_dict(),
                'opt_adamw_state': opt_adamw.state_dict(),
                'args': model_args
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

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
