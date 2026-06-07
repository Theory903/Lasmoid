#!/usr/bin/env python3
"""
Lasmoid — Gemma-4-26B-A4B → Lasmoid-500M Knowledge Distillation
================================================================

Distills knowledge from Google's Gemma-4-26B-A4B (teacher) into
Lasmoid-500M (student) using online logit distillation with progressive
alpha scheduling.

Budget: $200 on Azure H100 NVL ($2.19/hr) = ~90 hours available
Plan:   ~50 hours distillation + 5h SFT + 3h GRPO = $127 total

Usage:
    python3 scripts/distill_gemma4.py

Requirements:
    - NVIDIA H100 NVL (94GB VRAM)
    - HuggingFace account with Gemma-4 access accepted
    - huggingface-cli login (done in setup.sh)

Memory Budget (H100 94GB):
    Teacher (Gemma-4-26B-A4B, bf16): ~52GB
    Student (Lasmoid-500M, bf16):     ~1GB
    Activations + optimizer:          ~30GB
    Headroom:                         ~11GB
"""

import os
import sys
import json
import time
import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import fields as dc_fields
from tqdm import tqdm

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'inference'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'train'))
sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TEACHER_MODEL = "google/gemma-4-26B-A4B"
STUDENT_CONFIG = os.path.join(PROJECT_ROOT, "config_500m.json")
HF_REPO = "Theory903/lasmoid-500m-distilled"

# Training
DISTILL_STEPS = 15000       # ~45h on H100
SFT_STEPS = 3000            # ~5h
GRPO_STEPS = 1500           # ~3h
BATCH_SIZE = 8              # per step (H100 handles this)
SEQ_LEN = 512
GRAD_ACCUM = 4              # effective batch = 32

# Distillation
ALPHA_START = 0.8           # Start heavily weighted on teacher
ALPHA_END = 0.3             # End more on hard labels
TEMPERATURE = 2.0           # Softening temperature
MUON_LR = 2e-3
ADAMW_LR = 3e-4

# Checkpointing
SAVE_EVERY = 1000
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "distill")
LOG_EVERY = 50

DEVICE = "cuda"


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_alpha(step, total_steps):
    """Progressive alpha: decays from ALPHA_START to ALPHA_END via cosine."""
    progress = step / max(total_steps, 1)
    return ALPHA_END + (ALPHA_START - ALPHA_END) * (1 + math.cos(math.pi * progress)) / 2


def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature, ignore_index=-100):
    """
    Combined distillation loss:
    L = (1-α) * CE(student, labels) + α * T² * KL(teacher_soft || student_soft)
    """
    # Hard label loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )

    # Soft label loss (KL divergence on softened distributions)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # Only compute KL on valid positions
    mask = (labels != ignore_index).unsqueeze(-1).float()
    kl = F.kl_div(student_soft, teacher_soft, reduction='none') * mask
    kl_loss = kl.sum() / mask.sum().clamp(min=1)

    total = (1 - alpha) * ce_loss + alpha * (temperature ** 2) * kl_loss
    return total, ce_loss, kl_loss


def free_mem():
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  Gemma-4 → Lasmoid-500M Distillation (Azure H100)       ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    assert torch.cuda.is_available(), "No GPU!"
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n  GPU: {gpu} ({vram:.0f}GB)")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Load Teacher (Gemma-4-26B-A4B) ──────────────────────────
    print(f"\n  📥 Loading teacher: {TEACHER_MODEL}...")
    print("     (This takes ~5 min to download 50GB of weights)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",       # Auto-shard across available memory
        low_cpu_mem_usage=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_vocab = teacher.config.vocab_size
    print(f"  ✅ Teacher loaded: {sum(p.numel() for p in teacher.parameters())/1e9:.1f}B params")
    print(f"     Teacher vocab: {teacher_vocab}")
    print(f"     GPU mem after teacher: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Load Student (Lasmoid-500M) ─────────────────────────────
    print(f"\n  🏗️  Building student: Lasmoid-500M...")

    from model import Lasmoid, ModelArgs, compute_loss
    from optimizer import build_optimizers
    from scheduler import WSDScheduler

    with open(STUDENT_CONFIG) as f:
        cfg = json.load(f)
    valid = {f.name for f in dc_fields(ModelArgs)}
    args = ModelArgs(**{k: v for k, v in cfg.items() if k in valid})

    # Use student's own tokenizer
    import transformers
    student_enc = transformers.PreTrainedTokenizerFast.from_pretrained(PROJECT_ROOT)
    args.vocab_size = max(args.vocab_size, len(student_enc))
    student_vocab = args.vocab_size
    EOS_ID = student_enc.eos_token_id or 1

    torch.manual_seed(42)
    student = Lasmoid(args).to(DEVICE)
    student.train()

    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"  ✅ Student built: {n_params:.0f}M params")
    print(f"     Student vocab: {student_vocab}")
    print(f"     GPU mem total: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Vocab projection layer (teacher 262K → student 129K) ────
    # Teacher has larger vocab. We project teacher logits to student vocab space.
    # Simple approach: only use the first 129K logits (shared BPE tokens).
    # Both use similar BPE tokenization — the first 129K tokens largely overlap.
    print(f"\n  📊 Vocab mapping: teacher {teacher_vocab} → student {student_vocab}")
    print(f"     Using first {student_vocab} logits from teacher (shared BPE subspace)")

    # ── Load Data ───────────────────────────────────────────────
    print(f"\n  📥 Loading FineWeb-Edu (streaming)...")
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu-score-2",
        split="train",
        streaming=True,
    )

    # Tokenize with STUDENT tokenizer (we train the student)
    def tokenize_batch(examples, max_tokens=500_000):
        """Tokenize a batch of text and pack into SEQ_LEN chunks."""
        tokens = []
        for doc in examples:
            text = doc.get("text", "")
            if text and len(text) > 50:
                ids = student_enc.encode(text) + [EOS_ID]
                tokens.extend(ids)
                if len(tokens) >= max_tokens:
                    break
        # Pack into seq_len chunks
        tokens = tokens[:len(tokens) - len(tokens) % SEQ_LEN]
        return torch.tensor(tokens, dtype=torch.long).reshape(-1, SEQ_LEN)

    # Pre-tokenize a buffer
    print("  Tokenizing initial buffer...")
    buffer_docs = []
    for i, doc in enumerate(dataset):
        buffer_docs.append(doc)
        if i >= 5000:
            break
    data_buffer = tokenize_batch(buffer_docs)
    print(f"  ✅ Buffer: {data_buffer.shape[0]} sequences × {SEQ_LEN} tokens")

    def get_batch(batch_size=BATCH_SIZE):
        ix = torch.randint(data_buffer.shape[0], (batch_size,))
        x = data_buffer[ix].to(DEVICE)
        y = torch.cat([x[:, 1:], torch.full((x.shape[0], 1), EOS_ID, dtype=torch.long, device=DEVICE)], dim=1)
        return x, y

    # ── Optimizers ──────────────────────────────────────────────
    opts = build_optimizers(student, muon_lr=MUON_LR, adamw_lr=ADAMW_LR, weight_decay=0.1)
    warmup = int(0.02 * DISTILL_STEPS)
    stable = int(0.88 * DISTILL_STEPS)
    decay = DISTILL_STEPS - warmup - stable
    scheduler = WSDScheduler(
        opts, warmup_steps=warmup, stable_steps=stable, decay_steps=decay,
        base_lrs=[[g['lr'] for g in o.param_groups] for o in opts],
        min_lr_ratio=0.1,
    )

    # ═══════════════════════════════════════════════════════════
    # DISTILLATION LOOP
    # ═══════════════════════════════════════════════════════════
    print(f"\n  🚀 Starting distillation: {DISTILL_STEPS} steps")
    print(f"     Effective batch: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"     Tokens: ~{DISTILL_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / 1e9:.1f}B")
    print(f"     ETA: ~{DISTILL_STEPS * BATCH_SIZE * GRAD_ACCUM * 0.5 / 3600:.0f}h")
    print("─" * 60)

    losses = []
    t0 = time.time()

    for step in range(DISTILL_STEPS):
        scheduler.step(step)
        alpha = get_alpha(step, DISTILL_STEPS)

        for o in opts:
            o.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = get_batch()

            # Teacher forward (no grad, bf16)
            with torch.no_grad():
                teacher_out = teacher(x)
                # Take only the first student_vocab logits
                teacher_logits = teacher_out.logits[:, :, :student_vocab].float()

            # Student forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                student_logits, mtp, _, _, rmaps, _, adjs, eprobs = student(x, x)

                # Distillation loss
                loss, ce, kl = distillation_loss(
                    student_logits, teacher_logits, y,
                    alpha=alpha, temperature=TEMPERATURE,
                )

                # Auxiliary losses from student
                from model import compute_loss as _cl
                aux = _cl(
                    student_logits, y, rmaps, [student.last_vq_loss], adjs, eprobs,
                    loss_mask=torch.ones_like(x, dtype=torch.float32),
                    moe_aux_loss=student.last_moe_loss,
                    ignore_index=-100,
                )
                # Blend: 90% distillation + 10% student-internal aux
                loss = 0.9 * loss + 0.1 * aux
                loss = loss / GRAD_ACCUM

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        for o in opts:
            o.step()
        losses.append(total_loss)

        # Logging
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            tps = (step + 1) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / elapsed
            avg = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
            eta_h = (DISTILL_STEPS - step) / max(step, 1) * elapsed / 3600
            cost = elapsed / 3600 * 2.19
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {step:5d}/{DISTILL_STEPS} | loss {avg:.3f} | α={alpha:.2f} | "
                  f"{tps:.0f} tok/s | {mem:.0f}GB | ETA {eta_h:.1f}h | ${cost:.0f} spent",
                  flush=True)

        # Checkpoint
        if step > 0 and step % SAVE_EVERY == 0:
            ckpt = os.path.join(CKPT_DIR, f"distill_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": student.state_dict(),
                "losses": losses[-SAVE_EVERY:],
                "alpha": alpha,
            }, ckpt)
            print(f"  💾 Checkpoint: {ckpt}")

        # Refill data buffer periodically
        if step > 0 and step % 2000 == 0:
            print("  🔄 Refilling data buffer...")
            new_docs = []
            for i, doc in enumerate(dataset):
                new_docs.append(doc)
                if i >= 5000:
                    break
            data_buffer = tokenize_batch(new_docs)

    # ── Save final distilled model ──────────────────────────────
    final_path = os.path.join(CKPT_DIR, "distill_final.pt")
    torch.save(student.state_dict(), final_path)
    elapsed_h = (time.time() - t0) / 3600
    print(f"\n  ✅ Distillation done in {elapsed_h:.1f}h (${elapsed_h*2.19:.0f})")
    print(f"     Loss: {losses[0]:.2f} → {sum(losses[-100:])/100:.2f}")
    print(f"     Saved: {final_path}")

    # ── Upload to HuggingFace ───────────────────────────────────
    print(f"\n  ☁️  Uploading to {HF_REPO}...")
    try:
        from safetensors.torch import save_file
        from huggingface_hub import HfApi, create_repo

        save_dir = os.path.join(CKPT_DIR, "hf_upload")
        os.makedirs(save_dir, exist_ok=True)
        save_file(student.state_dict(), os.path.join(save_dir, "model.safetensors"))
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(save_dir, "README.md"), "w") as f:
            f.write(f"""---
license: apache-2.0
tags: [lasmoid, distillation, gemma4, moe, mamba]
---
# Lasmoid-500M (Distilled from Gemma-4-26B-A4B)

500M parameter hybrid concept transformer distilled from Google's Gemma-4-26B-A4B.

**Architecture**: CSA/MLA attention + Mamba-2 SSM + Manifold-Constrained Hyper-Connections + Grey-Box MoE + Concept Memory + MTP

**Training**: {DISTILL_STEPS} steps of knowledge distillation on FineWeb-Edu, teacher=Gemma-4-26B-A4B.
**Hardware**: Azure H100 NVL (94GB)
**Cost**: ~${elapsed_h*2.19:.0f}
""")

        api = HfApi()
        create_repo(HF_REPO, exist_ok=True)
        api.upload_folder(folder_path=save_dir, repo_id=HF_REPO,
                          commit_message=f"Distilled from Gemma-4 ({DISTILL_STEPS} steps)")
        print(f"  ✅ Uploaded: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"  ⚠️  Upload failed: {e}")
        print(f"      Weights saved locally: {final_path}")

    print("\n  🏁 DONE. Total cost: ${:.0f}".format(elapsed_h * 2.19))


if __name__ == "__main__":
    main()
