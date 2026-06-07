"""
generate_lasmoid_12m_notebook.py
=================================
Generates notebooks/distillation/lasmoid_12m_frontier.ipynb

Lasmoid ~12M Kaggle training notebook with:
  · 3-phase curriculum (35K steps)
  · Qwen3.5-0.8B distillation (4-bit NF4)
  · Concept-structured training format
  · Frontier Evaluation Harness (arithmetic / reasoning / coding / concept)
  · Checkpoint arena + best-model tracking
  · Generation scorecard every 1000 steps
  · Crash-safe checkpointing every 200 steps
  · Routing entropy + concept collapse monitoring
  · SIGTERM / SIGINT graceful shutdown

Run:  python scripts/generate_lasmoid_12m_notebook.py
"""

import json, textwrap
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def cell(source: str, cell_type: str = "code", metadata: dict = None) -> dict:
    src = textwrap.dedent(source).lstrip("\n")
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": metadata or {}, "source": src}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata or {},
        "outputs": [],
        "source": src,
    }


cells = []

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(
    cell(
        """
# Lasmoid 12M Frontier — Kaggle Training
### 3-Phase Curriculum · Qwen3.5-0.8B Distillation · Frontier Eval Harness
*35K Steps · Seq Len 1024 · WSD Schedule · Dual T4*
""",
        "markdown",
    )
)

# ═════════════════════════════════════════════════════════════════════════════
# S1 — ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 1 — Environment Setup", "markdown"))
cells.append(
    cell("""
!pip install -q uv
!uv pip install --system -q \\
    git+https://github.com/huggingface/transformers.git \\
    bitsandbytes>=0.46.0 accelerate>=1.6.0 datasets>=3.6.0 \\
    safetensors>=0.5.3 sentencepiece einops tqdm matplotlib psutil

import os, sys, json, time, math, random, gc, shutil, re
from pathlib import Path
from typing import Iterator, Optional
from collections import defaultdict

os.environ["PYTORCH_ALLOC_CONF"] = (
    "expandable_segments:True,garbage_collection_threshold:0.8"
)

import torch
import numpy as np
import transformers
print(f"Transformers : {transformers.__version__}")
print(f"PyTorch      : {torch.__version__}")
print("✅ Environment ready")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S2 — GPU DETECTION
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 2 — GPU Detection", "markdown"))
cells.append(
    cell("""
def gpu_info():
    if not torch.cuda.is_available():
        return {"n_gpus": 0, "total_vram_gb": 0, "names": [], "vram_per": []}
    n     = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    vram  = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(n)]
    return {"n_gpus": n, "total_vram_gb": sum(vram), "names": names, "vram_per": vram}

GPU = gpu_info()
print(f"GPUs detected: {GPU['n_gpus']}")
for i, (n, v) in enumerate(zip(GPU["names"], GPU["vram_per"])):
    print(f"  GPU{i}: {n}  ({v:.1f} GB)")

if GPU["n_gpus"] == 0:
    raise RuntimeError("No GPU — enable T4 x2 in Kaggle Settings → Accelerator.")

DTYPE = torch.bfloat16
print(f"Compute dtype : {DTYPE}")
print(f"Total VRAM    : {GPU['total_vram_gb']:.1f} GB")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S3 — ACCELERATE
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 3 — Accelerate Multi-GPU Setup (DDP)", "markdown"))
cells.append(
    cell("""
from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=8,   # effective batch = 2 × 8 × 2 GPUs = 32
    kwargs_handlers=[ddp_kwargs],
)

DEVICE  = accelerator.device
IS_MAIN = accelerator.is_main_process
N_PROC  = accelerator.num_processes

if IS_MAIN:
    print(f"Distributed : {accelerator.distributed_type}")
    print(f"Processes   : {N_PROC}")
    print(f"Device      : {DEVICE}")
    print(f"Mixed prec. : {accelerator.mixed_precision}")

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S4a — PATHS & HF TOKEN
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 4a — Paths, HF Token, Config", "markdown"))
cells.append(
    cell("""
import os
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        raise RuntimeError("Add HF_TOKEN to Kaggle → Add-ons → Secrets.")
os.environ["HF_TOKEN"] = HF_TOKEN

import huggingface_hub
huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
print("✅ HuggingFace logged in")

REPO_URL   = "https://github.com/Theory903/Lasmoid.git"
REPO_DIR   = Path("/kaggle/working/Lasmoid")
CKPT_DIR   = Path("/kaggle/working/checkpoints/lasmoid_12m_frontier")
LOG_DIR    = Path("/kaggle/working/logs/lasmoid_12m_frontier")
EXPORT_DIR = Path("/kaggle/working/export/lasmoid_12m_frontier")
EVAL_DIR   = LOG_DIR / "evals"

for d in [CKPT_DIR, LOG_DIR, EXPORT_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if not (REPO_DIR / "inference").exists():
    os.system(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} pull --rebase --autostash")

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "inference"))
sys.path.insert(0, str(REPO_DIR / "train"))

print(f"Repo : {REPO_DIR}")
print(f"CKPTs: {CKPT_DIR}")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S4b — GRACEFUL SHUTDOWN
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 4b — Graceful Shutdown Handler", "markdown"))
cells.append(
    cell("""
import signal

_CHECKPOINT_ON_KILL = {"step": 0, "dataset_idx": 0, "phase": "phase1"}

def _shutdown_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    print(f"\\n⚠️  Received {sig_name} — saving emergency checkpoint...")
    step      = _CHECKPOINT_ON_KILL.get("step", 0)
    ds_idx    = _CHECKPOINT_ON_KILL.get("dataset_idx", 0)
    phase_str = _CHECKPOINT_ON_KILL.get("phase", "phase1")
    if step > 0:
        ckpt_dir = CKPT_DIR / "emergency"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file as sf_save
        sf_save(
            {k: v.cpu() for k, v in accelerator.unwrap_model(model).state_dict().items()},
            str(ckpt_dir / "model.safetensors"),
        )
        save_cursor(step, ds_idx, phase_str)
        latest = CKPT_DIR / "latest"
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(ckpt_dir.name)
        print(f"  💾 Emergency checkpoint saved at step {step}")
    else:
        print("  No progress yet — skipping emergency save.")
    exit(0)

signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT,  _shutdown_handler)
print("✅ Graceful shutdown handler registered (SIGTERM / SIGINT)")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S4c — TOKENIZER
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 4c — Tokenizer (Lasmodium BPE, 32768 vocab)", "markdown"))
cells.append(
    cell("""
from transformers import AutoTokenizer

TOK_PATH  = str(REPO_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 0
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = tokenizer.pad_token_id

VOCAB_SIZE = len(tokenizer)
print(f"Tokenizer vocab size: {VOCAB_SIZE:,}")

CONCEPT_SPECIAL_TOKENS = [
    "<|question|>", "<|concepts|>", "<|constraints|>", "<|answer|>",
    "<|think|>", "</think>", "<|code|>", "<|plan|>",
]
existing_tokens = set(tokenizer.get_vocab().keys())
missing = [t for t in CONCEPT_SPECIAL_TOKENS if t not in existing_tokens]
if missing:
    print(f"Adding missing special tokens: {missing}")
    tokenizer.add_special_tokens({"additional_special_tokens": missing})
    VOCAB_SIZE = len(tokenizer)
    print(f"Updated vocab size: {VOCAB_SIZE:,}")

for tok in CONCEPT_SPECIAL_TOKENS:
    print(f"  {tok:>15} → id={tokenizer.convert_tokens_to_ids(tok)}")
print("✅ Tokenizer ready")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S5 — STREAMING DATASETS + 3-PHASE CURRICULUM
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 5 — Streaming Datasets + 3-Phase Curriculum", "markdown"))
cells.append(
    cell("""
from datasets import load_dataset, interleave_datasets

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEQ_LEN       = 1024
BATCH_SIZE    = 2          # per GPU; effective = 2 × 2 GPUs × 8 accum = 32
MAX_STEPS     = 35000
CKPT_EVERY    = 200
MONITOR_EVERY = 500
EVAL_EVERY    = 1000       # full frontier eval
SEED          = 42

# Phase boundaries (exclusive upper bounds)
PHASE_BOUNDARIES = [10000, 25000, 35000]

def get_phase(step: int) -> int:
    for i, b in enumerate(PHASE_BOUNDARIES):
        if step < b:
            return i
    return 2

# ── Text cleaning ─────────────────────────────────────────────────────────────
MIN_CHARS = 200
MAX_CHARS = 8000
_BOILERPLATE = [
    "Click here", "Subscribe to", "Cookie Policy",
    "Terms of Service", "\\u00a9", "All rights reserved", "Skip to content",
]

def clean_text(text: str) -> str:
    text = re.sub(r"\\s+", " ", text).strip()
    for pat in _BOILERPLATE:
        if pat.lower() in text.lower():
            return ""
    return text

# ── Format helpers ────────────────────────────────────────────────────────────
def format_plain(text: str) -> str:
    return text

def format_concept_question(text: str) -> str:
    return (
        f"<|question|> {text}\\n"
        f"<|concepts|> general, language\\n"
        f"<|constraints|> none\\n"
        f"<|answer|> {text}"
    )

# ── Phase dataset loaders ─────────────────────────────────────────────────────
def load_phase_datasets(phase: int):
    if phase == 0:
        # Phase 1: Cosmopedia 40% + FineWeb-Edu 30% + TinyStories 20% + Code 10%
        ds1 = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                           split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        ds2 = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
                           split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        ds3 = load_dataset("roneneldan/TinyStories", "default",
                           split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        try:
            ds4 = (load_dataset("bigcode/the-stack-march-sample", "default",
                                split="train", streaming=True, trust_remote_code=True)
                   .select_columns(["content"]).rename_column("content", "text"))
        except Exception:
            ds4 = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                               split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        combined = interleave_datasets([ds1, ds2, ds3, ds4],
                                       probabilities=[0.40, 0.30, 0.20, 0.10], seed=SEED)
        return combined, format_plain

    elif phase == 1:
        # Phase 2: distill placeholders with concept formatting
        ds_a = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                            split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        ds_b = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
                            split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        ds_c = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                            split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        ds_d = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
                            split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        combined = interleave_datasets([ds_a, ds_b, ds_c, ds_d],
                                       probabilities=[0.40, 0.20, 0.20, 0.20], seed=SEED)
        return combined, format_concept_question

    else:
        # Phase 3
        ds_a = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                            split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        ds_b = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
                            split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        ds_c = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                            split="train", streaming=True, trust_remote_code=True).select_columns(["text"])
        combined = interleave_datasets([ds_a, ds_b, ds_c],
                                       probabilities=[0.50, 0.30, 0.20], seed=SEED)
        return combined, format_concept_question

# ── Sequence packer ───────────────────────────────────────────────────────────
def stream_packed(dataset, tokenizer, seq_len: int,
                  max_batches: int = None, format_fn=None) -> Iterator:
    if format_fn is None:
        format_fn = format_plain
    buf    = []
    count  = 0
    eos_id = (tokenizer.eos_token_id or tokenizer.pad_token_id or 0)

    for example in dataset:
        text = clean_text(example.get("text", "") or example.get("content", ""))
        if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
            continue
        formatted = format_fn(text)
        ids = tokenizer.encode(formatted, add_special_tokens=True)
        if not isinstance(ids, list):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        ids = [x for x in ids if x is not None]
        ids.append(eos_id)
        buf.extend(ids)
        while len(buf) >= seq_len + 1:
            chunk = buf[:seq_len + 1]
            buf   = buf[seq_len + 1:]
            yield (torch.tensor(chunk[:-1], dtype=torch.long),
                   torch.tensor(chunk[1:],  dtype=torch.long))
            count += 1
            if max_batches and count >= max_batches:
                return

print("Loading Phase 1 datasets (streaming)...")
phase0_ds, phase0_format = load_phase_datasets(0)
ds_train   = phase0_ds
format_fn  = phase0_format
current_phase = 0

print(f"✅ Datasets ready  SEQ_LEN={SEQ_LEN}  BATCH={BATCH_SIZE}  "
      f"MAX_STEPS={MAX_STEPS}  CKPT_EVERY={CKPT_EVERY}")
print(f"   Phase boundaries: {PHASE_BOUNDARIES}")
print(f"   Frontier eval   : every {EVAL_EVERY} steps")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S6 — MODEL CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 6 — Lasmoid Model Construction (~12M)", "markdown"))
cells.append(
    cell("""
import json, sys, functools

sys.path.insert(0, str(REPO_DIR / "inference"))

from config import ModelArgs
from lasmoid import Lasmoid
from loss import compute_loss

# ── Gradient checkpoint crash guards ────────────────────────────────────────
import torch.utils.checkpoint as _cp
_cp._checkpoint_debug_enabled = False
_cp.set_checkpoint_debug_enabled = lambda enabled=None: None
try:
    import torch.testing._internal.logging_tensor as _lt
    _orig_sym = _lt.symbolize_tracebacks
    def _safe_sym(tb_list):
        try:    return _orig_sym(tb_list)
        except: return [[] for _ in tb_list]
    _lt.symbolize_tracebacks = _safe_sym
except Exception:
    pass
del _cp

import torch.utils.checkpoint as _cp2
_orig_ckpt = _cp2.checkpoint
if not getattr(_cp2.checkpoint, "__reentrant_patched", False):
    @functools.wraps(_orig_ckpt)
    def _safe_ckpt(function, *args, **kwargs):
        kwargs["use_reentrant"] = False
        return _orig_ckpt(function, *args, **kwargs)
    _cp2.checkpoint = _safe_ckpt
    _cp2.checkpoint.__reentrant_patched = True
print("✅ Gradient checkpoint patched: use_reentrant=False")
del _cp2

# ── Load 12M config ──────────────────────────────────────────────────────────
# Point to your 12M config. If you only have the 10.5M config, copy and edit
# dim/n_layers upward — e.g. dim=384, n_layers=10 ≈ 12M with the same MoE/SSM.
cfg_path = REPO_DIR / "configs" / "model" / "config_gemma4_12m.json"
if not cfg_path.exists():
    # Fallback: load tiny-frontier and scale up
    cfg_path = REPO_DIR / "configs" / "model" / "config_gemma4_tiny_frontier.json"
    print(f"⚠️  12M config not found — loading tiny-frontier as base. "
          f"Edit dim/n_layers to reach 12M.")

with open(cfg_path) as f:
    cfg = json.load(f)

# Override vocab + sequence length
cfg["vocab_size"]    = VOCAB_SIZE
cfg["max_seq_len"]   = SEQ_LEN
cfg["max_batch_size"]= BATCH_SIZE

args  = ModelArgs(**cfg)
model = Lasmoid(args).to(DTYPE)
model.gradient_checkpointing = True

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
embed_params = sum(p.numel() for n, p in model.named_parameters()
                   if "emb" in n or "head" in n)

print("=" * 60)
print(f"  Lasmoid 12M Frontier")
print("=" * 60)
print(f"  Total params    : {total_params/1e6:.2f}M")
print(f"  Trainable       : {train_params/1e6:.2f}M")
print(f"  Embedding/head  : {embed_params/1e6:.2f}M")
print(f"  Model body      : {(train_params-embed_params)/1e6:.2f}M")
print(f"  Vocab size      : {VOCAB_SIZE:,}")
print(f"  Dim             : {cfg.get('dim')}")
print(f"  Layers          : {cfg.get('n_layers')}")
print(f"  MoE experts     : {cfg.get('n_routed_experts')} routed / "
      f"{cfg.get('n_activated_experts')} active")
print(f"  SSM             : {cfg.get('ssm_heads')} heads / "
      f"state={cfg.get('ssm_state_dim')}")
print(f"  Concepts        : {cfg.get('num_concepts')} "
      f"(codebook={cfg.get('codebook_size')})")
print(f"  VRAM estimate   : ~{train_params*4/1e9:.1f} GB (params+grads, bf16)")
print("=" * 60)
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S7 — MUON + ADAMW + WSD
# ═════════════════════════════════════════════════════════════════════════════
cells.append(
    cell("## Section 7 — Muon + AdamW Dual Optimizer & WSD Scheduler", "markdown")
)
cells.append(
    cell("""
sys.path.insert(0, str(REPO_DIR / "train"))

from optimizer import build_optimizers, clip_grad_global_norm, ensure_muon_closure_compat, Muon
from scheduler import WSDScheduler

ensure_muon_closure_compat()

MUON_LR  = 2e-3
ADAMW_LR = 3e-4

optimizers = build_optimizers(
    model,
    muon_lr       = MUON_LR,
    adamw_lr      = ADAMW_LR,
    weight_decay  = 0.1,
    betas         = (0.9, 0.95),
    eps           = 1e-8,
    momentum      = 0.95,
    ns_steps      = 5,
    adaptive_noise= False,
)
print(f"Optimizers: {[type(o).__name__ for o in optimizers]}")

# WSD: 2% warmup · 80% stable · 18% decay
warmup_steps = max(1, int(0.02 * MAX_STEPS))    # ~700
stable_steps = int(0.80 * MAX_STEPS)             # ~28000
decay_steps  = MAX_STEPS - warmup_steps - stable_steps

base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

scheduler = WSDScheduler(
    optimizers   = optimizers,
    warmup_steps = warmup_steps,
    stable_steps = stable_steps,
    decay_steps  = decay_steps,
    base_lrs     = base_lrs,
    min_lr_ratio = 0.1,
)

print(f"WSD: warmup={warmup_steps}  stable={stable_steps}  decay={decay_steps}")
print(f"Muon LR: {MUON_LR}    AdamW LR: {ADAMW_LR}")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S8 — TEACHER MODEL
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 8 — Teacher: Qwen3.5-0.8B (4-bit NF4)", "markdown"))
cells.append(
    cell("""
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer as HFTok

TEACHER_MODEL = "Qwen/Qwen3.5-0.8B"
TEACHER_EVERY = 32  # Only run teacher forward every 32 steps

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

# Put teacher on GPU 1 when running single-process to save GPU 0 for student
if N_PROC == 1 and torch.cuda.device_count() > 1:
    TEACHER_DEVICE = torch.device("cuda:1")
else:
    TEACHER_DEVICE = DEVICE

if IS_MAIN:
    print(f"Loading teacher : {TEACHER_MODEL}")
    print(f"Teacher device  : {TEACHER_DEVICE}")
    print(f"Student device  : {DEVICE}")

if "teacher" in globals():
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    quantization_config = bnb_cfg,
    device_map          = {"": TEACHER_DEVICE},
    torch_dtype         = torch.bfloat16,
    token               = HF_TOKEN,
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

teacher_tok  = HFTok.from_pretrained(TEACHER_MODEL, token=HF_TOKEN)
TEACHER_VOCAB = teacher_tok.vocab_size

print(f"✅ Teacher loaded: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params")
print(f"   Teacher vocab: {TEACHER_VOCAB:,}    Lasmoid vocab: {VOCAB_SIZE:,}")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S9 — DISTILLATION LOSS
# ═════════════════════════════════════════════════════════════════════════════
cells.append(
    cell(
        "## Section 9 — Distillation Loss (Top-K Sparse KL + Temperature Annealing)",
        "markdown",
    )
)
cells.append(
    cell("""
import torch.nn.functional as F

DISTILL_TOP_K = 4096
T_START       = 4.0
T_END         = 1.0
ALPHA_WARMUP  = 0.50
ALPHA_PEAK    = 0.90
ALPHA_FINAL   = 0.15

def get_distill_alpha(step: int, total: int) -> float:
    peak_step = int(0.05 * total)
    if step <= peak_step:
        t = step / max(1, peak_step)
        return ALPHA_WARMUP + (ALPHA_PEAK - ALPHA_WARMUP) * t
    t = (step - peak_step) / max(1, total - peak_step)
    return ALPHA_FINAL + (ALPHA_PEAK - ALPHA_FINAL) * 0.5 * (1 + math.cos(math.pi * t))

def get_distill_temperature(step: int, total: int) -> float:
    t = step / max(1, total)
    return T_END + (T_START - T_END) * 0.5 * (1 + math.cos(math.pi * t))

def sparse_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    top_k: int = 4096,
) -> torch.Tensor:
    B, S, V_t = teacher_logits.shape
    V_s = student_logits.shape[-1]
    V   = min(V_t, V_s)
    k   = min(top_k, V)
    t_slice          = teacher_logits[..., :V]
    tk_vals, tk_idx  = t_slice.topk(k, dim=-1)
    p_teacher        = F.softmax(tk_vals.float() / temperature, dim=-1)
    s_topk           = student_logits[..., :V].gather(-1, tk_idx)
    log_q            = F.log_softmax(s_topk.float() / temperature, dim=-1)
    kl = (p_teacher * (p_teacher.clamp(min=1e-8).log() - log_q)).sum(-1).mean()
    return kl * (temperature ** 2)

def compute_teacher_logits(teacher, input_ids: torch.Tensor, vocab_limit: int):
    with torch.no_grad():
        out    = teacher(input_ids=input_ids.to(TEACHER_DEVICE), use_cache=False)
        logits = out.logits[..., :vocab_limit].to(DEVICE)
    return logits

print("✅ Distillation loss ready")
print(f"   Top-K sparse KL  : K={DISTILL_TOP_K}")
print(f"   Temperature       : {T_START} → {T_END} (cosine)")
print(f"   Alpha schedule    : {ALPHA_WARMUP} → {ALPHA_PEAK} → {ALPHA_FINAL}")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S10 — EXPERT & CONCEPT MONITORING (with routing entropy)
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 10 — Expert & Concept Monitoring", "markdown"))
cells.append(
    cell("""
class ExpertMonitor:
    \"\"\"MoE utilization + routing entropy + Gini coefficient.\"\"\"

    def __init__(self, n_experts: int, n_layers: int):
        self.n_experts = n_experts
        self.n_layers  = n_layers
        self.reset()

    def reset(self):
        self.counts = torch.zeros(self.n_layers, self.n_experts)
        self.steps  = 0

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
        total     = self.counts.sum(-1, keepdim=True).clamp(min=1)
        freq      = self.counts / total
        dead_mask = freq < 0.01
        dead_count = dead_mask.sum().item()

        # Per-layer entropy → ratio vs max
        ent     = -(freq.clamp(min=1e-8) * freq.clamp(min=1e-8).log()).sum(-1)
        max_ent = math.log(self.n_experts) if self.n_experts > 1 else 1.0

        # Routing entropy (averaged across layers, ratio in [0,1])
        routing_entropy_ratio = (ent.mean() / max_ent).item() if max_ent > 0 else 1.0

        # Gini (0 = balanced, 1 = collapsed)
        freq_avg    = freq.mean(0)
        sorted_freq = freq_avg.sort()[0]
        n           = self.n_experts
        cumsum      = sorted_freq.cumsum(0)
        gini        = ((2 * cumsum.sum() - (n + 1)) / max(n, 1)).item()

        return {
            "dead_experts"        : int(dead_count),
            "dead_pct"            : 100.0 * dead_count / max(1, self.n_layers * self.n_experts),
            "entropy_ratio"       : routing_entropy_ratio,
            "routing_entropy"     : ent.mean().item(),
            "utilization_gini"    : gini,
            "expert_freq"         : freq.mean(0).tolist(),
        }


class ConceptMonitor:
    \"\"\"ESCM concept usage, entropy, and collapse detection.\"\"\"

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
                idxs = ci.detach().cpu().view(-1)
                for i in idxs.tolist():
                    if 0 <= int(i) < self.num_concepts:
                        self.usage[int(i)] += 1
            except Exception:
                pass
        self.steps += 1

    def stats(self) -> dict:
        total    = self.usage.sum().clamp(min=1)
        freq     = self.usage / total
        ent      = -(freq.clamp(min=1e-8) * freq.clamp(min=1e-8).log()).sum().item()
        max_ent  = math.log(self.num_concepts) if self.num_concepts > 1 else 1.0
        top5     = freq.topk(min(5, self.num_concepts)).values.sum().item()
        top1     = freq.max().item()
        collapsed = top5 > 0.80

        return {
            "entropy"        : ent,
            "entropy_ratio"  : ent / max_ent if max_ent > 0 else 1.0,
            "collapsed"      : collapsed,
            "top1_pct"       : 100.0 * top1,
            "top5_coverage"  : top5,
            "dead_concepts"  : int((freq < 0.001).sum().item()),
            "concept_freq"   : freq.tolist(),
        }


expert_monitor  = ExpertMonitor(n_experts=args.n_routed_experts, n_layers=args.n_layers)
concept_monitor = ConceptMonitor(num_concepts=args.num_concepts)

print(f"✅ Expert monitor  : {args.n_layers} layers × {args.n_routed_experts} experts")
print(f"✅ Concept monitor : {args.num_concepts} concepts")
print(f"   Routing entropy tracked  : yes")
print(f"   Concept collapse guard   : top5 > 80%")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S11 — CHECKPOINT SYSTEM
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 11 — Crash-Safe Checkpoint System", "markdown"))
cells.append(
    cell("""
from safetensors.torch import save_file as safetensors_save

CURSOR_FILE     = CKPT_DIR / "cursor.json"
BEST_SCORE_FILE = CKPT_DIR / "best_score.json"


def save_cursor(step: int, dataset_idx: int, stage: str = "phase1"):
    with open(CURSOR_FILE, "w") as f:
        json.dump({"step": step, "dataset_idx": dataset_idx,
                   "stage": stage, "timestamp": time.time()}, f, indent=2)


def load_cursor() -> dict:
    if CURSOR_FILE.exists():
        with open(CURSOR_FILE) as f:
            return json.load(f)
    return {"step": 0, "dataset_idx": 0, "stage": "phase1"}


def save_checkpoint(step, model, optimizers, scheduler, loss_history,
                    expert_stats, concept_stats, dataset_idx=0, stage="phase1",
                    tag=None):
    \"\"\"Full crash-safe checkpoint.\"\"\"
    folder_name = f"step_{step:06d}" if tag is None else f"step_{step:06d}_{tag}"
    ckpt_path   = CKPT_DIR / folder_name
    ckpt_path.mkdir(parents=True, exist_ok=True)

    safetensors_save(
        {k: v.cpu() for k, v in model.state_dict().items()},
        str(ckpt_path / "model.safetensors")
    )
    torch.save([opt.state_dict() for opt in optimizers], ckpt_path / "optimizer.pt")

    with open(ckpt_path / "scheduler.json", "w") as f:
        json.dump({
            "warmup_steps": scheduler.warmup_steps,
            "stable_steps": scheduler.stable_steps,
            "decay_steps" : scheduler.decay_steps,
            "base_lrs"    : scheduler.base_lrs,
            "min_lr_ratio": scheduler.min_lr_ratio,
        }, f)

    rng_state = {
        "cpu"   : torch.get_rng_state(),
        "cuda"  : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
    }
    torch.save(rng_state, ckpt_path / "rng.pt")

    with open(ckpt_path / "meta.json", "w") as f:
        json.dump({
            "step"         : step,
            "dataset_idx"  : dataset_idx,
            "stage"        : stage,
            "loss_history" : loss_history[-500:],
            "expert_stats" : expert_stats,
            "concept_stats": concept_stats,
            "timestamp"    : time.time(),
            "config"       : cfg,
        }, f, indent=2, default=str)

    save_cursor(step, dataset_idx, stage)

    latest = CKPT_DIR / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(folder_name)

    print(f"  💾 Checkpoint: step {step} → {folder_name} (stage={stage})")


def maybe_save_best(step, score, model, optimizers, scheduler,
                    loss_history, expert_stats, concept_stats,
                    dataset_idx, stage):
    \"\"\"Save checkpoint and update best_score.json if this is a new high score.\"\"\"
    current_best = -1.0
    if BEST_SCORE_FILE.exists():
        with open(BEST_SCORE_FILE) as f:
            current_best = json.load(f).get("score", -1.0)
    if score > current_best:
        with open(BEST_SCORE_FILE, "w") as f:
            json.dump({"step": step, "score": float(score)}, f, indent=2)
        save_checkpoint(
            step          = step,
            model         = model,
            optimizers    = optimizers,
            scheduler     = scheduler,
            loss_history  = loss_history,
            expert_stats  = expert_stats,
            concept_stats = concept_stats,
            dataset_idx   = dataset_idx,
            stage         = stage,
            tag           = "best",
        )
        print(f"  🏆 NEW BEST MODEL  step={step}  score={score:.4f}  "
              f"(prev={current_best:.4f})")
    return score > current_best


def find_latest_checkpoint() -> Path:
    latest = CKPT_DIR / "latest"
    if latest.is_symlink() and latest.exists():
        return latest.resolve()
    candidates = sorted(CKPT_DIR.glob("step_*"))
    return candidates[-1] if candidates else None


def load_checkpoint(model, optimizers, scheduler, ckpt_path: Path):
    from safetensors.torch import load_file as safetensors_load
    print(f"  📂 Resuming from: {ckpt_path}")
    state = safetensors_load(str(ckpt_path / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    opt_states = torch.load(ckpt_path / "optimizer.pt", map_location="cpu")
    for opt, st in zip(optimizers, opt_states):
        opt.load_state_dict(st)
    rng = torch.load(ckpt_path / "rng.pt", map_location="cpu")
    torch.set_rng_state(rng["cpu"])
    if rng.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if rng.get("python"):
        random.setstate(rng["python"])
    with open(ckpt_path / "meta.json") as f:
        meta = json.load(f)
    return (meta["step"], meta.get("loss_history", []),
            meta.get("expert_stats", {}), meta.get("concept_stats", {}),
            meta.get("stage", "phase1"))


print(f"✅ Checkpoint system ready  (every {CKPT_EVERY} steps)")
print(f"   Best-model tracking: {BEST_SCORE_FILE.name}")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S12 — OOM RECOVERY + ACCELERATE WRAP
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 12 — OOM Recovery + Accelerate Wrap", "markdown"))
cells.append(
    cell("""
import gc

model, *optimizers = accelerator.prepare(model, *optimizers)

def oom_step():
    torch.cuda.empty_cache()
    gc.collect()

def safe_forward(model, x_enc, x_dec):
    try:
        return model(x_enc, x_dec)
    except torch.cuda.OutOfMemoryError:
        print("  ⚠️  OOM on forward — skipping batch")
        oom_step()
        return None
    except ValueError as e:
        if any(kw in str(e) for kw in ("stoi", "storage", "symbolize")):
            print(f"  ⚠️  PyTorch symbolizer error (skipping): {e}")
            return None
        raise

print("✅ Model wrapped with Accelerate (DDP + bf16)")
print(f"   Model on: {next(model.parameters()).device}")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S13 — AUTO-RESUME
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 13 — Auto-Resume", "markdown"))
cells.append(
    cell("""
loss_history  = []
expert_stats  = {}
concept_stats = {}
dataset_cursor = load_cursor()

latest_ckpt = find_latest_checkpoint()
START_STEP  = 0

if latest_ckpt is not None and IS_MAIN:
    START_STEP, loss_history, expert_stats, concept_stats, resumed_stage = load_checkpoint(
        accelerator.unwrap_model(model),
        [accelerator.unwrap_model(o) if hasattr(o, "param_groups") else o
         for o in optimizers],
        scheduler, latest_ckpt,
    )
    print(f"✅ Resumed from step {START_STEP} "
          f"(phase={get_phase(START_STEP)+1}, stage={resumed_stage})")
    print(f"   Dataset position: {dataset_cursor.get('dataset_idx', 0)}")
else:
    print("Starting fresh (no checkpoint found)")

if N_PROC > 1:
    t = torch.tensor([START_STEP], dtype=torch.long, device=DEVICE)
    torch.distributed.broadcast(t, src=0)
    START_STEP = t.item()
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S14 — FRONTIER EVALUATION HARNESS
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 14 — Frontier Evaluation Harness", "markdown"))
cells.append(
    cell("""
# ─────────────────────────────────────────────────────────────────────────────
# Fixed prompts covering: arithmetic, reasoning, language, coding, concept-mem
# ─────────────────────────────────────────────────────────────────────────────
EVAL_PROMPTS = [
    # Arithmetic
    ("arith_1", "2 + 3 ="),
    ("arith_2", "15 + 28 ="),
    ("arith_3", "125 - 49 ="),
    # Reasoning
    ("reason_1", "A farmer has 17 sheep. All but 9 die. How many remain?"),
    ("reason_2", "Tom has 2 apples and buys 3 more. How many apples does he have?"),
    ("reason_3", "Mary has 3 cats. Each cat has 4 kittens. How many kittens total?"),
    # Language
    ("lang_1", "Write one sentence about a dog."),
    ("lang_2", "Explain gravity to a child."),
    # Coding
    ("code_1", "Write a Python function that adds two numbers."),
    ("code_2", "Write Python code that reverses a string."),
    ("code_3", "Write a Python function to check if a number is prime."),
    # Concept memory
    ("concept_1", "<|question|> What is photosynthesis?\\n<|concepts|>"),
    ("concept_2", "<|question|> How does gravity work?\\n<|concepts|>"),
]

# ── Generation helpers ────────────────────────────────────────────────────────
def generate_text(model, prompt: str, max_new_tokens: int = 80) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = accelerator.unwrap_model(model).generate(
            ids, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=False,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def teacher_generate(prompt: str, max_new_tokens: int = 80) -> str:
    ids = teacher_tok(prompt, return_tensors="pt").input_ids.to(TEACHER_DEVICE)
    with torch.no_grad():
        out = teacher.generate(
            ids, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=False,
        )
    return teacher_tok.decode(out[0], skip_special_tokens=True)


# ── Scoring functions ─────────────────────────────────────────────────────────
def score_arithmetic(text: str) -> float:
    digits = sum(c.isdigit() for c in text)
    return min(1.0, digits / 2)


def score_reasoning(text: str) -> float:
    # Length heuristic: at least 5 words → likely a coherent answer
    words = text.split()
    if len(words) >= 8:
        return 1.0
    if len(words) >= 4:
        return 0.5
    return 0.0


def score_code(text: str) -> float:
    score = 0.0
    if "def "    in text: score += 0.40
    if "return " in text: score += 0.40
    if len(text) > 30:    score += 0.20
    return score


def score_concept(text: str) -> float:
    # Concept output should NOT contain "general, general" collapse
    words = [w.strip().lower() for w in text.replace(",", " ").split()]
    unique_ratio = len(set(words)) / max(len(words), 1)
    collapse_penalty = 1.0 if words.count("general") > 2 else 0.0
    return max(0.0, min(1.0, unique_ratio - collapse_penalty))


def score_language(text: str) -> float:
    words = text.split()
    if len(words) >= 6 and len(set(words)) > 4:
        return 1.0
    if len(words) >= 3:
        return 0.5
    return 0.0


# ── Full evaluation at a checkpoint ──────────────────────────────────────────
def evaluate_checkpoint(step: int) -> dict:
    results = {"step": step, "timestamp": time.time(), "samples": []}
    scores_by_category = defaultdict(list)

    print()
    print("=" * 80)
    print(f"  FRONTIER EVALUATION  @  STEP {step}")
    print("=" * 80)

    for pid, prompt in EVAL_PROMPTS:
        student_out = generate_text(model, prompt)

        try:
            teacher_out = teacher_generate(prompt)
        except Exception:
            teacher_out = "(teacher unavailable)"

        # Score by category
        if pid.startswith("arith"):
            s = score_arithmetic(student_out)
            scores_by_category["arithmetic"].append(s)
        elif pid.startswith("reason"):
            s = score_reasoning(student_out)
            scores_by_category["reasoning"].append(s)
        elif pid.startswith("code"):
            s = score_code(student_out)
            scores_by_category["coding"].append(s)
        elif pid.startswith("concept"):
            s = score_concept(student_out)
            scores_by_category["concept"].append(s)
        elif pid.startswith("lang"):
            s = score_language(student_out)
            scores_by_category["language"].append(s)
        else:
            s = 0.5

        results["samples"].append({
            "id"      : pid,
            "prompt"  : prompt,
            "student" : student_out,
            "teacher" : teacher_out,
            "score"   : s,
        })

        print(f"\\n[{pid}]")
        print(f"  PROMPT  : {prompt[:80]}")
        print(f"  STUDENT : {student_out[:200]}")
        print(f"  TEACHER : {teacher_out[:200]}")
        print(f"  SCORE   : {s:.3f}")

    # Category averages
    for cat, vals in scores_by_category.items():
        results[f"{cat}_score"] = sum(vals) / max(len(vals), 1)

    # Overall = mean of category averages
    all_cat_scores = [results[f"{cat}_score"] for cat in scores_by_category]
    results["overall"] = sum(all_cat_scores) / max(len(all_cat_scores), 1)

    # Persist
    out_file = EVAL_DIR / f"step_{step:06d}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print(f"  SCORES @ step {step}")
    print("=" * 80)
    for cat in scores_by_category:
        print(f"  {cat.capitalize():<12} : {results[f'{cat}_score']:.3f}")
    print(f"  {'Overall':<12} : {results['overall']:.3f}")
    print("=" * 80)

    return results


# ── Generation scorecard (lightweight, printed every EVAL_EVERY steps) ────────
def print_scorecard(step, loss, ppl, exp_s, con_s, eval_results=None):
    print()
    print(f"{'─'*55}")
    print(f"  GENERATION SCORECARD — step {step}")
    print(f"{'─'*55}")
    print(f"  {'Perplexity':<22}: {ppl:.2f}")
    print(f"  {'Expert entropy':<22}: {exp_s['entropy_ratio']:.3f}  "
          f"(dead={exp_s['dead_experts']})")
    print(f"  {'Concept entropy':<22}: {con_s['entropy_ratio']:.3f}  "
          f"(collapse={'YES' if con_s['collapsed'] else 'no'})  "
          f"top5={con_s['top5_coverage']*100:.0f}%")
    print(f"  {'Routing Gini':<22}: {exp_s['utilization_gini']:.3f}")
    if eval_results:
        for cat in ("arithmetic", "reasoning", "coding", "concept", "language"):
            key = f"{cat}_score"
            if key in eval_results:
                print(f"  {cat.capitalize():<22}: {eval_results[key]:.3f}")
        print(f"  {'Overall':<22}: {eval_results.get('overall', 0.0):.3f}")
    print(f"{'─'*55}")


# ── Checkpoint arena: compare latest 4 checkpoints on shared prompts ──────────
def run_checkpoint_arena(current_step: int):
    \"\"\"Compare the last 4 step checkpoints on EVAL_PROMPTS[:5] (quick test).\"\"\"
    ARENA_PROMPTS = EVAL_PROMPTS[:5]
    candidates = sorted(CKPT_DIR.glob("step_??????"))[-4:]
    if len(candidates) < 2:
        return   # not enough history yet

    print()
    print("=" * 70)
    print(f"  CHECKPOINT ARENA @ step {current_step}  ({len(candidates)} candidates)")
    print("=" * 70)

    from safetensors.torch import load_file as sf_load
    from config import ModelArgs
    from lasmoid import Lasmoid

    arena_results = {}

    for ckpt_path in candidates:
        label = ckpt_path.name
        try:
            state = sf_load(str(ckpt_path / "model.safetensors"), device="cpu")
            tmp   = Lasmoid(args).to(DTYPE).to(DEVICE)
            tmp.load_state_dict(state, strict=False)
            tmp.eval()

            scores = []
            for pid, prompt in ARENA_PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
                with torch.no_grad():
                    out = tmp.generate(ids, max_new_tokens=50, do_sample=False)
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                if pid.startswith("arith"):
                    scores.append(score_arithmetic(text))
                elif pid.startswith("code"):
                    scores.append(score_code(text))
                elif pid.startswith("reason"):
                    scores.append(score_reasoning(text))
                else:
                    scores.append(score_language(text))

            arena_results[label] = sum(scores) / max(len(scores), 1)

            del tmp, state
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            arena_results[label] = -1.0
            print(f"  ⚠️  Arena error for {label}: {e}")

    print()
    for label, score in sorted(arena_results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if score == max(arena_results.values()) else ""
        print(f"  {label}  →  {score:.4f}{marker}")
    print("=" * 70)

    # Persist arena results
    arena_file = EVAL_DIR / f"arena_step_{current_step:06d}.json"
    with open(arena_file, "w") as f:
        json.dump({"step": current_step, "results": arena_results}, f, indent=2)


print("✅ Frontier Evaluation Harness ready")
print(f"   Eval prompts   : {len(EVAL_PROMPTS)}")
print(f"   Eval frequency : every {EVAL_EVERY} steps")
print(f"   Arena runs     : every {EVAL_EVERY * 5} steps")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S15 — TRAINING LOOP (35K steps, phase-aware, with eval + arena)
# ═════════════════════════════════════════════════════════════════════════════
cells.append(cell("## Section 15 — Training Loop (35K Steps, Phase-Aware)", "markdown"))
cells.append(
    cell("""
from tqdm.auto import tqdm

loss_log_path = LOG_DIR / "loss.jsonl"

def log_step(d: dict):
    with open(loss_log_path, "a") as f:
        f.write(json.dumps(d) + "\\n")

# ── Build initial data iterator ───────────────────────────────────────────────
data_iter     = stream_packed(ds_train, tokenizer, SEQ_LEN, format_fn=format_fn)
dataset_idx   = dataset_cursor.get("dataset_idx", 0)
current_phase = get_phase(START_STEP)

if dataset_idx > 0 and IS_MAIN:
    print(f"Fast-forwarding dataset to position {dataset_idx}...")
    for _ in range(dataset_idx):
        try:
            next(data_iter)
        except StopIteration:
            ds_train, format_fn = load_phase_datasets(current_phase)
            data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN, format_fn=format_fn)

# ── Training state ────────────────────────────────────────────────────────────
model.train()
pbar = tqdm(range(START_STEP, MAX_STEPS), initial=START_STEP, total=MAX_STEPS,
            desc="Lasmoid 12M", disable=not IS_MAIN)

accum_steps       = accelerator.gradient_accumulation_steps
tokens_per_step   = BATCH_SIZE * SEQ_LEN * accum_steps * N_PROC
total_tokens_seen = 0

_last_eval_results = None    # carry latest eval into scorecard
_arena_every       = EVAL_EVERY * 5

def bounded_ppl(loss_val: float) -> float:
    return math.exp(min(max(loss_val, 0.0), 20.0))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════
for step in pbar:
    _CHECKPOINT_ON_KILL.update({
        "step": step, "dataset_idx": dataset_idx,
        "phase": f"phase{current_phase + 1}",
    })

    # ── Phase transition ──────────────────────────────────────────────────────
    new_phase = get_phase(step)
    if new_phase != current_phase:
        if IS_MAIN:
            print(f"\\n{'='*60}")
            print(f"PHASE TRANSITION: {current_phase+1} → {new_phase+1}  (step {step})")
            print(f"{'='*60}")
        current_phase = new_phase
        ds_train, format_fn = load_phase_datasets(current_phase)
        data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN, format_fn=format_fn)
        expert_monitor.reset()
        concept_monitor.reset()

    # ── LR schedule ───────────────────────────────────────────────────────────
    lr_mult = scheduler.step(step)

    # ── Gradient accumulation ─────────────────────────────────────────────────
    total_loss_accum = 0.0
    t0 = time.time()

    for micro in range(accum_steps):
        try:
            x, y = next(data_iter)
            dataset_idx += 1
        except StopIteration:
            ds_train, format_fn = load_phase_datasets(current_phase)
            data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN, format_fn=format_fn)
            x, y = next(data_iter)

        x = x.unsqueeze(0).to(DEVICE)
        y = y.unsqueeze(0).to(DEVICE)

        with accelerator.accumulate(model):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = safe_forward(model, x, x)
                if out is None:
                    continue

                (logits, mtp_logits, concept_db, memory_state,
                 routing_maps, concept_indices, adjacencies, event_probs) = out

                T     = get_distill_temperature(step, MAX_STEPS)
                alpha = get_distill_alpha(step, MAX_STEPS)

                kl_loss = torch.zeros((), device=DEVICE)
                run_teacher = alpha > 0.01 and step % TEACHER_EVERY == 0
                if run_teacher:
                    teacher_logits = compute_teacher_logits(teacher, x, VOCAB_SIZE)
                    kl_loss = sparse_kl_distillation_loss(
                        logits, teacher_logits, temperature=T, top_k=DISTILL_TOP_K
                    )
                    del teacher_logits

                mtp_loss = None
                if mtp_logits is not None and x.shape[1] > 1:
                    mtp_targets        = y[:, 1:]
                    mtp_logits_trimmed = mtp_logits[:, :mtp_targets.shape[1], :]
                    mtp_loss = F.cross_entropy(
                        mtp_logits_trimmed.reshape(-1, mtp_logits_trimmed.shape[-1]),
                        mtp_targets.reshape(-1), ignore_index=-100,
                    )

                unwrapped  = accelerator.unwrap_model(model)
                total_loss = compute_loss(
                    logits              = logits,
                    targets             = y,
                    routing_maps        = routing_maps,
                    vq_losses           = [unwrapped.last_vq_loss],
                    adjacencies         = adjacencies,
                    event_probs         = event_probs,
                    moe_aux_loss        = unwrapped.last_moe_loss,
                    moe_aux_coeff       = args.moe_aux_coeff,
                    mtp_loss            = mtp_loss,
                    mtp_coeff           = args.mtp_loss_coeff,
                    token_concept_loss  = unwrapped.last_token_concept_loss,
                    token_concept_coeff = args.token_concept_loss_coeff,
                    commit_loss         = unwrapped.last_commit_loss,
                    commit_coeff        = args.hcm_commit_loss_coeff,
                    curiosity_loss      = None,
                    graph_sparsity      = 0.01,
                    label_smoothing     = 0.0,
                    ignore_index        = -100,
                )

                blended_loss = (1.0 - alpha) * total_loss + alpha * (T ** 2) * kl_loss

            accelerator.backward(blended_loss / accum_steps)
            total_loss_accum += blended_loss.item() / accum_steps

        if IS_MAIN:
            expert_monitor.update(routing_maps)
            concept_monitor.update(concept_indices)

    # ── Gradient clipping & optimizer step ────────────────────────────────────
    grad_norm  = 0.0
    _skip_step = False

    if accelerator.sync_gradients:
        grad_norm = clip_grad_global_norm(accelerator.unwrap_model(model), max_norm=1.0)
        for _p in accelerator.unwrap_model(model).parameters():
            if _p.grad is not None and (
                torch.isnan(_p.grad).any() or torch.isinf(_p.grad).any()
            ):
                _skip_step = True
                break

        if not _skip_step:
            for opt in optimizers:
                opt.step()
            accelerator.unwrap_model(model).apply_pending_bias_updates()
        else:
            log_step({"step": step, "nan_gradient": True})

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    loss_history.append(total_loss_accum)
    step_time = time.time() - t0
    total_tokens_seen += tokens_per_step

    if IS_MAIN:
        perplexity     = bounded_ppl(total_loss_accum)
        exp_s          = expert_monitor.stats()
        con_s          = concept_monitor.stats()
        tokens_per_sec = tokens_per_step / max(step_time, 1e-6)

        _alloc, _free = 0.0, 0.0
        if torch.cuda.is_available() and step % 10 == 0:
            _alloc = torch.cuda.memory_allocated()  / 1024**3
            _free  = (torch.cuda.get_device_properties(0).total_memory / 1e9) - _alloc

        pbar.set_postfix({
            "loss"   : f"{total_loss_accum:.3f}",
            "ppl"    : f"{perplexity:.1f}",
            "tok/s"  : f"{tokens_per_sec:,.0f}",
            "ent"    : f"{exp_s['entropy_ratio']:.2f}",
            "dead_e" : exp_s["dead_experts"],
            "phase"  : f"{current_phase+1}",
        })

        # ── Detailed log every MONITOR_EVERY ─────────────────────────────────
        if step % MONITOR_EVERY == 0:
            log_step({
                "step"               : step,
                "phase"              : current_phase + 1,
                "loss"               : total_loss_accum,
                "perplexity"         : perplexity,
                "kl_loss"            : kl_loss.item(),
                "mtp_loss"           : mtp_loss.item() if mtp_loss else 0.0,
                "moe_aux"            : unwrapped.last_moe_loss.item(),
                "vq_loss"            : unwrapped.last_vq_loss.item(),
                "commit_loss"        : unwrapped.last_commit_loss.item(),
                "token_concept_loss" : (unwrapped.last_token_concept_loss.item()
                                        if getattr(unwrapped, "last_token_concept_loss", None) is not None
                                        else 0.0),
                "alpha"              : alpha,
                "temperature"        : T,
                "lr_mult"            : lr_mult,
                "grad_norm"          : grad_norm,
                "step_time_s"        : step_time,
                "tokens_per_sec"     : tokens_per_sec,
                "gpu_mem_gb"         : _alloc,
                "expert_entropy"     : exp_s["entropy_ratio"],
                "expert_routing_ent" : exp_s["routing_entropy"],
                "expert_gini"        : exp_s["utilization_gini"],
                "dead_experts"       : exp_s["dead_experts"],
                "concept_entropy"    : con_s["entropy_ratio"],
                "concept_collapsed"  : con_s["collapsed"],
                "concept_top5"       : con_s["top5_coverage"],
                "eval_overall"       : (_last_eval_results or {}).get("overall", 0.0),
            })
        elif step % 10 == 0:
            log_step({
                "step"          : step,
                "phase"         : current_phase + 1,
                "loss"          : total_loss_accum,
                "perplexity"    : perplexity,
                "kl_loss"       : kl_loss.item(),
                "lr_mult"       : lr_mult,
                "grad_norm"     : grad_norm,
                "step_time_s"   : step_time,
                "tokens_per_sec": tokens_per_sec,
            })

        # ── Frontier evaluation + best-model tracking every EVAL_EVERY ───────
        if step > 0 and step % EVAL_EVERY == 0:
            model.eval()
            try:
                _last_eval_results = evaluate_checkpoint(step)
                log_step({
                    "step"              : step,
                    "eval_arithmetic"   : _last_eval_results.get("arithmetic_score", 0),
                    "eval_reasoning"    : _last_eval_results.get("reasoning_score", 0),
                    "eval_coding"       : _last_eval_results.get("coding_score", 0),
                    "eval_concept"      : _last_eval_results.get("concept_score", 0),
                    "eval_language"     : _last_eval_results.get("language_score", 0),
                    "eval_overall"      : _last_eval_results.get("overall", 0),
                })
                maybe_save_best(
                    step          = step,
                    score         = _last_eval_results["overall"],
                    model         = accelerator.unwrap_model(model),
                    optimizers    = optimizers,
                    scheduler     = scheduler,
                    loss_history  = loss_history,
                    expert_stats  = exp_s,
                    concept_stats = con_s,
                    dataset_idx   = dataset_idx,
                    stage         = f"phase{current_phase + 1}",
                )
            finally:
                model.train()

            print_scorecard(step, total_loss_accum, perplexity,
                            exp_s, con_s, _last_eval_results)

        # ── Checkpoint arena every EVAL_EVERY * 5 ────────────────────────────
        if step > 0 and step % _arena_every == 0:
            model.eval()
            try:
                run_checkpoint_arena(step)
            finally:
                model.train()

        # ── Periodic crash-safe checkpoint ────────────────────────────────────
        if step > 0 and step % CKPT_EVERY == 0:
            save_checkpoint(
                step          = step,
                model         = accelerator.unwrap_model(model),
                optimizers    = optimizers,
                scheduler     = scheduler,
                loss_history  = loss_history,
                expert_stats  = exp_s,
                concept_stats = con_s,
                dataset_idx   = dataset_idx,
                stage         = f"phase{current_phase + 1}",
            )
            expert_monitor.reset()
            concept_monitor.reset()

print("\\n✅ Training complete!")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S16 — VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
cells.append(
    cell("## Section 16 — Validation (Perplexity + Reasoning Probe)", "markdown")
)
cells.append(
    cell("""
model.eval()

val_path = REPO_DIR / "datasets" / "tinystories_val.bin"
if val_path.exists():
    import numpy as np
    val_data = torch.from_numpy(
        np.frombuffer(val_path.read_bytes(), dtype=np.uint16).astype(np.int64)
    )
    N_VAL  = min(512, len(val_data) // SEQ_LEN)
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(N_VAL):
            chunk = val_data[i * SEQ_LEN : (i + 1) * SEQ_LEN + 1]
            if len(chunk) < SEQ_LEN + 1:
                continue
            x   = chunk[:SEQ_LEN].unsqueeze(0).to(DEVICE)
            y   = chunk[1:SEQ_LEN+1].unsqueeze(0).to(DEVICE)
            out = accelerator.unwrap_model(model)(x, x)
            nll = F.cross_entropy(
                out[0].reshape(-1, out[0].shape[-1]), y.reshape(-1), reduction="sum"
            )
            total_nll += nll.item()
            total_tok += y.numel()
    ppl_val = math.exp(min(total_nll / total_tok, 20))
    print(f"Validation perplexity: {ppl_val:.2f}")
else:
    print("No val binary found — skipping perplexity.")
    ppl_val = None

# ── Final frontier evaluation ─────────────────────────────────────────────────
print("\\nRunning final frontier evaluation...")
final_eval = evaluate_checkpoint(MAX_STEPS)
print_scorecard(MAX_STEPS, loss_history[-1] if loss_history else 0.0,
                bounded_ppl(loss_history[-1]) if loss_history else 0.0,
                expert_monitor.stats(), concept_monitor.stats(), final_eval)

model.train()
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S17 — TRAINING REPORT
# ═════════════════════════════════════════════════════════════════════════════
cells.append(
    cell("## Section 17 — Training Report (Loss Curves + Eval Scores)", "markdown")
)
cells.append(
    cell("""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_lines = []
if loss_log_path.exists():
    with open(loss_log_path) as f:
        for line in f:
            try:    log_lines.append(json.loads(line.strip()))
            except: pass

steps        = [l["step"]               for l in log_lines]
losses       = [l["loss"]               for l in log_lines]
kl_losses    = [l.get("kl_loss", 0)    for l in log_lines]
dead_exps    = [l.get("dead_experts", 0)    for l in log_lines]
concept_ent  = [l.get("concept_entropy", 0) for l in log_lines]
expert_ents  = [l.get("expert_entropy", 0)  for l in log_lines]
eval_steps   = [l["step"]    for l in log_lines if "eval_overall" in l and l.get("eval_overall", 0) > 0]
eval_overall = [l["eval_overall"] for l in log_lines if "eval_overall" in l and l.get("eval_overall", 0) > 0]
eval_arith   = [l.get("eval_arithmetic", 0) for l in log_lines if "eval_arithmetic" in l]
eval_code    = [l.get("eval_coding", 0)     for l in log_lines if "eval_coding"     in l]
eval_reason  = [l.get("eval_reasoning", 0)  for l in log_lines if "eval_reasoning"  in l]

fig, axes = plt.subplots(3, 2, figsize=(14, 13))
fig.suptitle("Lasmoid 12M Frontier — Training Report", fontsize=15, fontweight="bold")

# Loss curves
ax = axes[0, 0]
ax.plot(steps, losses,    color="#6366f1", lw=1.5, label="Total Loss")
ax.plot(steps, kl_losses, color="#f59e0b", lw=1.0, alpha=0.7, label="KL Distill")
ax.set_title("Loss Curves"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# Frontier eval scores
ax = axes[0, 1]
if eval_steps:
    ax.plot(eval_steps, eval_overall, color="#10b981", lw=2, marker="o", label="Overall")
    ax.plot(eval_steps[:len(eval_arith)],  eval_arith,  color="#3b82f6", lw=1.2, label="Arithmetic")
    ax.plot(eval_steps[:len(eval_code)],   eval_code,   color="#8b5cf6", lw=1.2, label="Coding")
    ax.plot(eval_steps[:len(eval_reason)], eval_reason, color="#ef4444", lw=1.2, label="Reasoning")
ax.set_title("Frontier Eval Scores"); ax.set_xlabel("Step"); ax.set_ylabel("Score")
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)

# Dead expert count
ax = axes[1, 0]
ax.plot(steps, dead_exps, color="#dc2626", lw=1.5)
ax.set_title("Dead Experts (< 1% utilization)")
ax.set_xlabel("Step"); ax.set_ylabel("Count"); ax.grid(alpha=0.3)

# Concept entropy
ax = axes[1, 1]
ax.plot(steps, concept_ent, color="#7c3aed", lw=1.5)
ax.axhline(0.5, color="#94a3b8", ls="--", alpha=0.7, label="Collapse threshold")
ax.set_title("Concept Memory Entropy Ratio")
ax.set_xlabel("Step"); ax.set_ylabel("Ratio (1.0 = uniform)")
ax.legend(); ax.grid(alpha=0.3)

# Expert routing entropy
ax = axes[2, 0]
ax.plot(steps, expert_ents, color="#0ea5e9", lw=1.5)
ax.axhline(0.6, color="#94a3b8", ls="--", alpha=0.7, label="Min healthy")
ax.axhline(0.9, color="#10b981", ls="--", alpha=0.7, label="Max balanced")
ax.set_title("Expert Routing Entropy Ratio")
ax.set_xlabel("Step"); ax.set_ylabel("Ratio"); ax.legend(); ax.grid(alpha=0.3)

# Loss histogram (last 500 steps)
ax = axes[2, 1]
if losses:
    ax.hist(losses[-500:], bins=30, color="#6366f1", alpha=0.7, edgecolor="white")
    ax.set_title("Loss Distribution (last 500 steps)")
    ax.set_xlabel("Loss"); ax.set_ylabel("Frequency"); ax.grid(alpha=0.3)

plt.tight_layout()
report_path = LOG_DIR / "training_report.png"
plt.savefig(str(report_path), dpi=120, bbox_inches="tight")
plt.show()
print(f"✅ Report saved: {report_path}")

# Summary
if losses:
    best_eval = max(eval_overall) if eval_overall else 0.0
    best_step = eval_steps[eval_overall.index(best_eval)] if eval_overall else 0
    print(f"\\n{'='*60}")
    print(f"  Training Summary — Lasmoid 12M Frontier")
    print(f"{'='*60}")
    print(f"  Steps completed  : {steps[-1] if steps else 0}")
    print(f"  Final loss       : {losses[-1]:.4f}")
    print(f"  Best loss        : {min(losses):.4f}")
    print(f"  Best eval score  : {best_eval:.4f}  (step {best_step})")
    print(f"  Dead experts     : {dead_exps[-1] if dead_exps else '?'}")
    print(f"  Concept entropy  : {concept_ent[-1]:.3f}" if concept_ent else "")
    print(f"  Total tokens     : {total_tokens_seen:,}")
    print(f"{'='*60}")
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S18 — KAGGLE EXPORT
# ═════════════════════════════════════════════════════════════════════════════
cells.append(
    cell(
        "## Section 18 — Kaggle Dataset Export (Multi-Session Persistence)", "markdown"
    )
)
cells.append(
    cell("""
import shutil

def export_for_kaggle(step: int, model, export_dir: Path):
    from safetensors.torch import save_file as sf_save
    pkg = export_dir / f"lasmoid_12m_step{step}"
    pkg.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    sf_save({k: v.cpu() for k, v in unwrapped.state_dict().items()},
            str(pkg / "model.safetensors"))

    # Config + tokenizer
    shutil.copy(str(cfg_path), str(pkg / "config.json"))
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = REPO_DIR / fname
        if src.exists():
            shutil.copy(str(src), str(pkg / fname))

    # Eval history
    evals = sorted(EVAL_DIR.glob("step_*.json"))
    all_evals = []
    for ef in evals[-20:]:
        try:
            with open(ef) as f:
                all_evals.append(json.load(f))
        except Exception:
            pass

    with open(pkg / "training_meta.json", "w") as f:
        json.dump({
            "step"         : step,
            "teacher"      : TEACHER_MODEL,
            "loss_history" : loss_history[-50:],
            "eval_history" : all_evals,
            "architecture" : "Lasmoid-12M-Frontier",
            "vocab_size"   : VOCAB_SIZE,
        }, f, indent=2)

    if CURSOR_FILE.exists():
        shutil.copy(str(CURSOR_FILE), str(pkg / "cursor.json"))

    with open(export_dir / "dataset-metadata.json", "w") as f:
        json.dump({
            "title"   : f"Lasmoid-12M-Frontier-step{step}",
            "id"      : f"lasmoid-12m-frontier-step{step}",
            "licenses": [{"name": "apache-2.0"}],
        }, f, indent=2)

    print(f"✅ Export ready: {pkg}")
    print(f"   Upload: kaggle datasets create -p {export_dir}")
    return pkg

if IS_MAIN and loss_history:
    final_step = len(loss_history) + START_STEP
    export_for_kaggle(final_step, model, EXPORT_DIR)
""")
)

# ═════════════════════════════════════════════════════════════════════════════
# S19 — HUGGINGFACE EXPORT
# ═════════════════════════════════════════════════════════════════════════════
cells.append(
    cell("## Section 19 — Final Export (SafeTensors + HuggingFace Hub)", "markdown")
)
cells.append(
    cell("""
import shutil
from safetensors.torch import save_file as sf_save

if IS_MAIN:
    hf_dir = Path("/kaggle/working/hf_model")
    hf_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    sf_save({k: v.cpu() for k, v in unwrapped.state_dict().items()},
            str(hf_dir / "model.safetensors"))
    print(f"✅ Weights saved: {hf_dir / 'model.safetensors'}")

    shutil.copy(str(cfg_path), str(hf_dir / "config.json"))
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = REPO_DIR / fname
        if src.exists():
            shutil.copy(str(src), str(hf_dir / fname))

    # Read best eval for model card
    best_score, best_step_val = 0.0, 0
    if BEST_SCORE_FILE.exists():
        with open(BEST_SCORE_FILE) as f:
            bd = json.load(f)
            best_score, best_step_val = bd.get("score", 0), bd.get("step", 0)

    model_card = f\"\"\"---
language: en
license: apache-2.0
tags:
- lasmoid
- hybrid-transformer
- moe
- mamba
- knowledge-distillation
base_model: Qwen/Qwen3.5-0.8B
---

# Lasmoid 12M Frontier

**Architecture**: Hybrid Concept Transformer (Lasmodium)
**Parameters**: ~12M  |  **Teacher**: Qwen3.5-0.8B (4-bit NF4)
**Training**: 3-phase curriculum, 35K steps, Top-K Sparse KL distillation
**Vocab**: BPE 32,768  |  **Best eval score**: {best_score:.4f} @ step {best_step_val}

## Architecture Components
- Compressed Sparse Attention (CSA)
- Mamba-2 SSD State Space Recurrence
- Grey-Box MoE (routed + shared experts)
- ElasticSparseConceptMemory (ESCM)
- Manifold-Constrained Hyper-Connections (mHC)
- Multi-Token Prediction (MTP t+1)

## Training Phases
| Phase | Steps     | Datasets                                       | Format    |
|-------|-----------|------------------------------------------------|-----------|
| 1     | 0–9999    | Cosmopedia 40% + FineWeb-Edu 30% + TS 20% + Code 10% | Plain |
| 2     | 10K–24999 | Qwen + Claude distills + Code + FineWeb-Edu    | Structured |
| 3     | 25K–34999 | Qwen 50% + Claude 30% + Opus 20%               | Structured |

## Evaluation (Frontier Harness)
Arithmetic · Reasoning · Coding · Concept-memory · Language — scored every 1000 steps.
Best-model checkpoint saved automatically based on overall eval score.
\"\"\"
    (hf_dir / "README.md").write_text(model_card)

    # ── Optional push to Hub ───────────────────────────────────────────────
    # from huggingface_hub import HfApi
    # HfApi().upload_folder(
    #     folder_path   = str(hf_dir),
    #     repo_id       = "Theory903/lasmoid-12m-frontier",
    #     commit_message= f"step {MAX_STEPS}, best_score={best_score:.4f}",
    # )

    print(f"✅ Model ready at: {hf_dir}")
    print(f"   Best eval score : {best_score:.4f}  (step {best_step_val})")
""")
)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD NOTEBOOK JSON
# ─────────────────────────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
        },
    },
    "cells": cells,
}

OUT_DIR = Path(__file__).parent.parent / "notebooks" / "distillation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "lasmoid_12m_frontier.ipynb"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

size_kb = out_path.stat().st_size / 1024
print(f"✅ Notebook written : {out_path}")
print(f"   Size             : {size_kb:.1f} KB")
print(f"   Cells            : {len(cells)}")
print()
print("Key upgrades over the 10.5M notebook:")
print("  [+] Frontier Evaluation Harness (Section 14)")
print("  [+]   arithmetic / reasoning / coding / concept / language scoring")
print("  [+]   teacher comparison on every eval prompt")
print("  [+]   eval runs every 1000 steps, integrated into training loop")
print("  [+] Best-model tracking via maybe_save_best()")
print("  [+] Checkpoint arena (last 4 checkpoints vs fixed prompts)")
print("  [+] Generation scorecard printed every 1000 steps")
print("  [+] Routing entropy added to ExpertMonitor")
print("  [+] Concept collapse guard: top5 > 80% flagged")
print("  [+] EVAL_DIR created at startup for eval JSON persistence")
print("  [+] Arena results persisted to EVAL_DIR")
print("  [+] Best eval score shown in HF model card")
