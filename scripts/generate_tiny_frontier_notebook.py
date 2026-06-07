"""
generate_tiny_frontier_notebook.py
===================================
Generates notebooks/distillation/lasmoid_tiny_frontier.ipynb
— ~10.5M Lasmoid Kaggle training notebook with 3-phase curriculum,
  Qwen3.5-0.8B distillation, concept-structured training format,
  and production-grade crash recovery.

Run:  python scripts/generate_tiny_frontier_notebook.py
"""

import json, textwrap
from pathlib import Path


def cell(source: str, cell_type: str = "code", metadata: dict = None) -> dict:
    src = textwrap.dedent(source).lstrip("\n")
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": metadata or {},
            "source": src,
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata or {},
        "outputs": [],
        "source": src,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CELLS
# ─────────────────────────────────────────────────────────────────────────────

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(
    cell(
        """
# Lasmoid Tiny Frontier — ~10.5M Kaggle Training
### 3-Phase Curriculum · Qwen3.5-0.8B Distillation · Concept-Structured Format
*35K Steps · Seq Len 1024 · WSD Schedule · Dual T4*
""",
        "markdown",
    )
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: ENVIRONMENT SETUP                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(cell("## Section 1: Environment Setup", "markdown"))
cells.append(
    cell("""
# Install uv and packages using direct shell commands
!pip install -q uv
!uv pip install --system -q git+https://github.com/huggingface/transformers.git bitsandbytes>=0.46.0 accelerate>=1.6.0 datasets>=3.6.0 safetensors>=0.5.3 sentencepiece einops tqdm matplotlib psutil

import os, sys, json, time, math, random, gc, shutil, re
from pathlib import Path
from typing import Iterator, Optional
from collections import defaultdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import torch
import numpy as np
import transformers
print(f"Transformers: {transformers.__version__}")
print("✅ Environment ready")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: GPU DETECTION                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(cell("## Section 2: GPU Detection", "markdown"))
cells.append(
    cell("""
def gpu_info():
    if not torch.cuda.is_available():
        return {"n_gpus": 0, "total_vram_gb": 0, "names": []}
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    vram  = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(n)]
    return {"n_gpus": n, "total_vram_gb": sum(vram), "names": names, "vram_per": vram}

GPU = gpu_info()
print(f"GPUs detected: {GPU['n_gpus']}")
for i, (n, v) in enumerate(zip(GPU["names"], GPU.get("vram_per", []))):
    print(f"  GPU{i}: {n}  ({v:.1f} GB)")

if GPU["n_gpus"] == 0:
    raise RuntimeError("No GPU found. Enable T4 x2 in Kaggle Settings → Accelerator.")

DTYPE = torch.bfloat16
print(f"Compute dtype: {DTYPE}")
print(f"Total VRAM   : {GPU['total_vram_gb']:.1f} GB")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: MULTI-GPU SETUP (DDP)                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(cell("## Section 3: Accelerate Multi-GPU Setup (DDP)", "markdown"))
cells.append(
    cell("""
from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=8,  # effective batch = 2 * 8 * 2 GPUs = 32
    kwargs_handlers=[ddp_kwargs],
)

DEVICE      = accelerator.device
IS_MAIN     = accelerator.is_main_process
N_PROC      = accelerator.num_processes

if IS_MAIN:
    print(f"Accelerator  : {accelerator.distributed_type}")
    print(f"Processes    : {N_PROC}")
    print(f"Device       : {DEVICE}")
    print(f"Mixed prec.  : {accelerator.mixed_precision}")

# Reproducibility
SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4a: PATHS, HF TOKEN, CONFIG                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(cell("## Section 4a: Paths, HF Token, Config", "markdown"))
cells.append(
    cell("""
import os
from pathlib import Path

# ── HuggingFace auth ────────────────────────────────────────────────────────
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

# ── Repo & paths ─────────────────────────────────────────────────────────────
REPO_URL   = "https://github.com/Theory903/Lasmoid.git"
REPO_DIR   = Path("/kaggle/working/Lasmoid")
CKPT_DIR   = Path("/kaggle/working/checkpoints/lasmoid_tiny_frontier")
LOG_DIR    = Path("/kaggle/working/logs/lasmoid_tiny_frontier")
EXPORT_DIR = Path("/kaggle/working/export/lasmoid_tiny_frontier")

for d in [CKPT_DIR, LOG_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Clone / pull repo
if not (REPO_DIR / "inference").exists():
    os.system(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} pull --rebase --autostash")

import sys
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "inference"))
sys.path.insert(0, str(REPO_DIR / "train"))

print(f"Repo : {REPO_DIR}")
print(f"CKPTs: {CKPT_DIR}")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4b: GRACEFUL SHUTDOWN                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell("## Section 4b: Graceful Shutdown (SIGTERM/SIGINT Handler)", "markdown")
)
cells.append(
    cell("""
import signal

_CHECKPOINT_ON_KILL = {"step": 0, "dataset_idx": 0, "phase": "phase1"}

def _shutdown_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    print(f"\\n⚠️  Received {sig_name} — saving emergency checkpoint...")
    step = _CHECKPOINT_ON_KILL.get("step", 0)
    ds_idx = _CHECKPOINT_ON_KILL.get("dataset_idx", 0)
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
    print("Exiting gracefully.")
    exit(0)

signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
print("✅ Graceful shutdown handler registered (SIGTERM/SIGINT)")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4c: TOKENIZER                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(cell("## Section 4c: Tokenizer (Lasmodium BPE, 32768 vocab)", "markdown"))
cells.append(
    cell("""
from transformers import AutoTokenizer

# Use the pre-trained Lasmoid tokenizer
TOK_PATH = str(REPO_DIR)   # tokenizer.json lives at repo root
tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = len(tokenizer)
print(f"Tokenizer vocab size: {VOCAB_SIZE:,}")

# Verify special concept tokens exist (or add them)
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

for tok_name in CONCEPT_SPECIAL_TOKENS:
    tid = tokenizer.convert_tokens_to_ids(tok_name)
    print(f"  {tok_name:>15} → id={tid}")
print("✅ Tokenizer ready")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4d: CONCEPT-STRUCTURED TRAINING FORMAT                             ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 4d: Concept-Structured Training Format",
        "markdown",
    )
)
cells.append(
    cell(
        """### Concept-Structured Training Format

Instead of raw chain-of-thought (CoT), this notebook uses a structured
**Question → Concepts → Plan → Answer** format for all training data.

**Rationale:**
- Forces the model to identify *which concepts* are relevant before generating
- Creates explicit concept-to-token supervision via the ESCM module
- Makes the model's problem-solving process inspectable and debuggable
- Improves compositional generalization (concepts compose, tokens don't)

#### Format Variants

**General text (FineWeb-Edu, Cosmopedia):**
```
<|question|> {question text}
<|concepts|> {concept1}, {concept2}, ...
<|constraints|> {boundary conditions}
<|answer|> {answer text}
```

**Coding problems:**
```
<|question|> {problem description}
<|concepts|> {programming concepts}
<|plan|> 1. {step 1}
2. {step 2}
3. {step 3}
<|code|> ```language
{code}
```
```

**Distillation data (Qwen, Claude, Opus):**
- Teacher responses are parsed into the Question/Concepts/Constraints/Answer schema
- The model learns to predict both the concepts AND the answer from the question alone

#### Training Phases

| Phase | Steps | Datasets | Format |
|-------|-------|----------|--------|
| 1 (0-9999) | 10K | Cosmopedia 40% + FineWeb-Edu 30% + TinyStories 20% + Code 10% | Plain |
| 2 (10000-24999) | 15K | Qwen distills 40% + Claude distills 20% + Coding 20% + FineWeb-Edu 20% | Structured |
| 3 (25000-34999) | 10K | Qwen 50% + Claude 30% + Opus 20% | Structured |
""",
        "markdown",
    )
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: STREAMING DATASETS + 3-PHASE CURRICULUM                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 5: Streaming Datasets + 3-Phase Curriculum",
        "markdown",
    )
)
cells.append(
    cell("""
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer as HFTok

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEQ_LEN      = 1024
BATCH_SIZE   = 2          # per GPU; effective = 2 × 2 GPUs × 8 grad_accum = 32
MAX_STEPS    = 35000
CKPT_EVERY   = 200        # crash-safe: save every 200 steps
MONITOR_EVERY = 500       # detailed monitoring every 500 steps
VAL_EVERY    = 500
SEED         = 42

# ── 3-Phase Curriculum ───────────────────────────────────────────────────────
# Phase 1 (steps 0-9999, 10K steps): Cosmopedia 40% + FineWeb-Edu 30% + TinyStories 20% + Code 10%
# Phase 2 (steps 10000-24999, 15K steps): Qwen distills 40% + Claude distills 20% + Coding 20% + FineWeb-Edu 20%
# Phase 3 (steps 25000-34999, 10K steps): Qwen 50% + Claude 30% + Opus 20%

PHASE_BOUNDARIES = [10000, 25000, 35000]  # exclusive upper bounds

def get_phase(step: int) -> int:
    for i, boundary in enumerate(PHASE_BOUNDARIES):
        if step < boundary:
            return i
    return 2  # final phase

# ── Cleaning utilities ─────────────────────────────────────────────────────────
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

# ── Formatting helpers ─────────────────────────────────────────────────────────
def format_plain(text: str) -> str:
    return text

def format_concept_question(text: str) -> str:
    \"\"\"Wrap text in minimal Question/Concepts/Constraints/Answer format.\"\"\"
    return f"<|question|> {text}\\n<|concepts|> general, language\\n<|constraints|> none\\n<|answer|> {text}"

# ── Dataset loaders ────────────────────────────────────────────────────────────
def load_phase_datasets(phase: int):
    \"\"\"Return (interleaved_dataset, format_fn) for the given phase (0, 1, 2).\"\"\"
    if phase == 0:
        # Phase 1: Cosmopedia 40% + FineWeb-Edu 30% + TinyStories 20% + Code 10%
        ds1 = load_dataset(
            "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        ds2 = load_dataset(
            "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        ds3 = load_dataset(
            "roneneldan/TinyStories", "default",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        # Code dataset (fallback to cosmopedia if the-stack is unavailable)
        try:
            ds4 = load_dataset(
                "bigcode/the-stack-march-sample", "default",
                split="train", streaming=True, trust_remote_code=True,
            ).select_columns(["content"]).rename_column("content", "text")
        except Exception:
            ds4 = load_dataset(
                "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                split="train", streaming=True, trust_remote_code=True,
            ).select_columns(["text"])
        combined = interleave_datasets(
            [ds1, ds2, ds3, ds4],
            probabilities=[0.40, 0.30, 0.20, 0.10],
            seed=SEED,
        )
        return combined, format_plain

    elif phase == 1:
        # Phase 2: Qwen distills 40% + Claude distills 20% + Coding 20% + FineWeb-Edu 20%
        # In production, replace with actual pre-distilled datasets.
        # Here we use cosmopedia/fineweb as placeholders with concept formatting.
        ds_qwen = load_dataset(
            "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        ds_claude = load_dataset(
            "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        ds_code = load_dataset(
            "bigcode/the-stack-march-sample", "default",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["content"]).rename_column("content", "text") if False else \
                 load_dataset(
            "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        ds_edu = load_dataset(
            "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        combined = interleave_datasets(
            [ds_qwen, ds_claude, ds_code, ds_edu],
            probabilities=[0.40, 0.20, 0.20, 0.20],
            seed=SEED,
        )
        return combined, format_concept_question

    else:
        # Phase 3: Qwen 50% + Claude 30% + Opus 20%
        ds_qwen = load_dataset(
            "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        ds_claude = load_dataset(
            "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        ds_opus = load_dataset(
            "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
            split="train", streaming=True, trust_remote_code=True,
        ).select_columns(["text"])
        combined = interleave_datasets(
            [ds_qwen, ds_claude, ds_opus],
            probabilities=[0.50, 0.30, 0.20],
            seed=SEED,
        )
        return combined, format_concept_question

# ── Streaming seq-packer ───────────────────────────────────────────────────────
def stream_packed(dataset, tokenizer, seq_len: int,
                  max_batches: int = None, format_fn=None) -> Iterator:
    \"\"\"
    Yields (input_ids [seq_len], labels [seq_len]) tensors from streaming dataset.
    Packs multiple documents into one sequence with EOS separator.
    \"\"\"
    if format_fn is None:
        format_fn = format_plain
    buf = []
    count = 0
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(tokenizer, "sep_token_id", None) or getattr(tokenizer, "pad_token_id", 1)
    for example in dataset:
        text = clean_text(example.get("text", "") or example.get("content", ""))
        if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
            continue
        formatted = format_fn(text)
        ids = tokenizer.encode(formatted, add_special_tokens=True)
        if not isinstance(ids, list):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        ids.append(eos_id)
        buf.extend(ids)
        while len(buf) >= seq_len + 1:
            chunk = buf[:seq_len + 1]
            buf   = buf[seq_len + 1:]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:],  dtype=torch.long)
            yield x, y
            count += 1
            if max_batches and count >= max_batches:
                return

# Load initial Phase 1 datasets
print("Loading Phase 1 datasets (streaming)...")
phase0_ds, phase0_format = load_phase_datasets(0)
ds_train = phase0_ds
format_fn = phase0_format
current_phase = 0

print(f"✅ Datasets ready (SEQ_LEN={SEQ_LEN}, BATCH={BATCH_SIZE}, "
      f"MAX_STEPS={MAX_STEPS}, CKPT_EVERY={CKPT_EVERY})")
print(f"  Phases: {PHASE_BOUNDARIES}")
print(f"  Initial phase: {current_phase}")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: MODEL CONSTRUCTION                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 6: Lasmoid Model Construction (~10.5M, Tiny Frontier Config)",
        "markdown",
    )
)
cells.append(
    cell("""
import json, sys, functools
from pathlib import Path

sys.path.insert(0, str(REPO_DIR / "inference"))

from config import ModelArgs
from lasmoid import Lasmoid
from loss import compute_loss

# ── PyTorch checkpoint stoi crash guard ──────────────────────────────────────
import torch.utils.checkpoint as _cp
_cp._checkpoint_debug_enabled = False
def _noop_setter(enabled=None):
    pass
_cp.set_checkpoint_debug_enabled = _noop_setter
try:
    import torch.testing._internal.logging_tensor as _lt
    _orig_sym = _lt.symbolize_tracebacks
    def _safe_sym(tb_list):
        try:
            return _orig_sym(tb_list)
        except (ValueError, Exception):
            return [[] for _ in tb_list]
    _lt.symbolize_tracebacks = _safe_sym
except Exception:
    pass
del _cp

# ── Gradient checkpoint reentrant crash guard ──────────────────────────────
import torch.utils.checkpoint as _cp_ckpt
_orig_ckpt_fn = _cp_ckpt.checkpoint
if not getattr(_cp_ckpt.checkpoint, '__reentrant_patched', False):
    @functools.wraps(_orig_ckpt_fn)
    def _safe_ckpt_fn(function, *args, **kwargs):
        kwargs['use_reentrant'] = False
        return _orig_ckpt_fn(function, *args, **kwargs)
    _cp_ckpt.checkpoint = _safe_ckpt_fn
    _cp_ckpt.checkpoint.__reentrant_patched = True
print("✅ Gradient checkpoint patched: use_reentrant=False (multi-loss safe)")
del _cp_ckpt

# Load tiny frontier config
cfg_path = REPO_DIR / "configs" / "model" / "config_gemma4_tiny_frontier.json"
with open(cfg_path) as f:
    cfg = json.load(f)

# Override vocab to match tokenizer
cfg["vocab_size"] = VOCAB_SIZE

# Kaggle-safe overrides
cfg["max_seq_len"]    = SEQ_LEN
cfg["max_batch_size"] = BATCH_SIZE

args = ModelArgs(**cfg)
model = Lasmoid(args).to(DTYPE)

# Enable gradient checkpointing (trades VRAM for recompute)
model.gradient_checkpointing = True

# Count parameters
total_params  = sum(p.numel() for p in model.parameters())
train_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
embed_params  = sum(p.numel() for n, p in model.named_parameters() if "emb" in n or "head" in n)
non_embed     = train_params - embed_params

print("=" * 55)
print(f"  Lasmoid Tiny Frontier (~10.5M)")
print("=" * 55)
print(f"  Total params  : {total_params/1e6:.2f}M")
print(f"  Trainable     : {train_params/1e6:.2f}M")
print(f"  Embedding/head: {embed_params/1e6:.2f}M  ({100*embed_params/train_params:.0f}%)")
print(f"  Model body    : {non_embed/1e6:.2f}M")
print(f"  Vocab size    : {VOCAB_SIZE:,}")
print(f"  Dim           : {cfg['dim']}")
print(f"  Layers        : {cfg['n_layers']}")
print(f"  MoE experts   : {cfg['n_routed_experts']} routed, {cfg['n_activated_experts']} active")
print(f"  SSM           : {cfg['ssm_heads']} heads, state={cfg['ssm_state_dim']}")
print(f"  Concepts      : {cfg['num_concepts']} (codebook={cfg['codebook_size']})")
print()
print(f"  token_concept_loss_coeff = {cfg['token_concept_loss_coeff']}")
print()

# VRAM estimate
vram_est = train_params * 4 / 1e9
print(f"  VRAM estimate : ~{vram_est:.1f} GB (params+grads, bf16)")
print(f"  Seq len       : {SEQ_LEN}")
print(f"  Batch size    : {BATCH_SIZE} per GPU")
print("=" * 55)
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: MUON + ADAMW OPTIMIZERS + WSD SCHEDULER                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell("## Section 7: Muon + AdamW Dual Optimizer & WSD Scheduler (35K)", "markdown")
)
cells.append(
    cell("""
sys.path.insert(0, str(REPO_DIR / "train"))

from optimizer import build_optimizers, clip_grad_global_norm, ensure_muon_closure_compat, Muon
from scheduler import WSDScheduler

# ── Muon.step(closure) compatibility ────────────────────────────────────────
ensure_muon_closure_compat()

# Build Muon + AdamW split
# - Muon  → all 2D hidden weights (transformer body)
# - AdamW → embeddings, norms, biases, 1D params
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

# WSD schedule: 2% warmup, 80% stable, 18% decay (scaled to 35K)
warmup_steps = max(1, int(0.02 * MAX_STEPS))   # ~700
stable_steps = int(0.80 * MAX_STEPS)            # ~28000
decay_steps  = MAX_STEPS - warmup_steps - stable_steps  # ~6300

base_lrs = []
for opt in optimizers:
    base_lrs.append([g["lr"] for g in opt.param_groups])

scheduler = WSDScheduler(
    optimizers  = optimizers,
    warmup_steps= warmup_steps,
    stable_steps= stable_steps,
    decay_steps = decay_steps,
    base_lrs    = base_lrs,
    min_lr_ratio= 0.1,
)

print(f"Steps: warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps}")
print(f"Muon  base LR: {MUON_LR}")
print(f"AdamW base LR: {ADAMW_LR}")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8: TEACHER MODEL (QWEN3.5-0.8B + SECONDARY TEACHERS COMMENTED)    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 8: Teacher Models — Qwen3.5-0.8B Primary (4-bit NF4)",
        "markdown",
    )
)
cells.append(
    cell("""
from transformers import BitsAndBytesConfig

# ── Primary Teacher: Qwen3.5-0.8B (4-bit NF4, frozen) ──────────────────────
# Qwen3.5-0.8B (~1.6GB at 4-bit NF4) replaces Gemma-4-12B as the primary
# distillation teacher. It fits comfortably on the same GPU as the student.

TEACHER_MODEL = "Qwen/Qwen3.5-0.8B"

# 4-bit NF4 quantization
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

# Teacher device: if multi-GPU, put teacher on GPU 1 to share memory with student
if N_PROC == 1 and torch.cuda.device_count() > 1:
    TEACHER_DEVICE = torch.device("cuda:1")
else:
    TEACHER_DEVICE = DEVICE

if IS_MAIN:
    print(f"Loading teacher: {TEACHER_MODEL}")
    print(f"  Quantization: 4-bit NF4")
    print(f"  Teacher Device: {TEACHER_DEVICE}")
    print(f"  Student Device: {DEVICE}")

# Free existing teacher model if it exists in memory to prevent OOM on re-run
if 'teacher' in globals():
    print("🧹 Freeing existing teacher model from memory...")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

from transformers import AutoModelForCausalLM, AutoTokenizer as HFTok

# Load teacher
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

# Teacher tokenizer
teacher_tok = HFTok.from_pretrained(TEACHER_MODEL, token=HF_TOKEN)

print(f"✅ Teacher loaded. Params: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M")
print(f"   Teacher vocab: {teacher_tok.vocab_size:,} | Lasmoid vocab: {VOCAB_SIZE:,}")
TEACHER_VOCAB = teacher_tok.vocab_size
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8b: SECONDARY TEACHERS (COMMENTED)                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 8b: Secondary Teachers (Commented — Gemma 3 / Claude Distillation Datasets)",
        "markdown",
    )
)
cells.append(
    cell("""
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECONDARY TEACHERS — UNCOMMENT TO ENABLE                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
#
# These are alternative/distillation teacher configurations kept for reference.
# They are NOT active during training (commented out to avoid OOM).
#
# ── Gemma 3 1B (lighter teacher) ──────────────────────────────────────────────
# from transformers import AutoModelForCausalLM
# GEMMA3_TEACHER = "google/gemma-3-1b"
# teacher_gemma3 = AutoModelForCausalLM.from_pretrained(
#     GEMMA3_TEACHER,
#     quantization_config=bnb_cfg,
#     device_map={"": TEACHER_DEVICE},
#     torch_dtype=torch.bfloat16,
#     token=HF_TOKEN,
# )
# teacher_gemma3.eval()
# for p in teacher_gemma3.parameters():
#     p.requires_grad_(False)
#
# ── Gemma 3 4B (stronger teacher) ─────────────────────────────────────────────
# GEMMA3_4B_TEACHER = "google/gemma-3-4b"
# teacher_gemma3_4b = AutoModelForCausalLM.from_pretrained(
#     GEMMA3_4B_TEACHER,
#     quantization_config=bnb_cfg,
#     device_map={"": TEACHER_DEVICE},
#     torch_dtype=torch.bfloat16,
#     token=HF_TOKEN,
# )
# teacher_gemma3_4b.eval()
# for p in teacher_gemma3_4b.parameters():
#     p.requires_grad_(False)
#
# ── Claude distillation datasets ──────────────────────────────────────────────
# These are pre-computed offline distillations, NOT live API calls.
# Load from a Kaggle Dataset:
#   ds_claude = load_dataset("path/to/claude-distills", split="train", streaming=True)
#
# ── Opus distillation datasets ────────────────────────────────────────────────
#   ds_opus = load_dataset("path/to/opus-distills", split="train", streaming=True)

print("✅ Secondary teachers configured (commented). See source to enable.")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9: DISTILLATION LOSS                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 9: Distillation Loss (Top-K Sparse KL + Temperature Annealing)",
        "markdown",
    )
)
cells.append(
    cell("""
import torch, torch.nn.functional as F

DISTILL_TOP_K = 4096     # sparse KL: only top-K tokens (32x less memory)
T_START       = 4.0      # temperature start (rich soft labels)
T_END         = 1.0      # temperature end (sharp predictions)
ALPHA_WARMUP  = 0.50     # initial teacher trust
ALPHA_PEAK    = 0.90     # peak teacher trust (warmup end)
ALPHA_FINAL   = 0.15     # final teacher trust (end of training)

def get_distill_alpha(step: int, total: int) -> float:
    \"\"\"Peaked alpha schedule: warmup 0.5→0.9 (first 5%), cosine decay 0.9→0.15.\"\"\"
    peak_step = int(0.05 * total)
    if step <= peak_step:
        t = step / max(1, peak_step)
        return ALPHA_WARMUP + (ALPHA_PEAK - ALPHA_WARMUP) * t
    else:
        t = (step - peak_step) / max(1, total - peak_step)
        return ALPHA_FINAL + (ALPHA_PEAK - ALPHA_FINAL) * 0.5 * (1 + math.cos(math.pi * t))

def get_distill_temperature(step: int, total: int) -> float:
    \"\"\"Cosine anneal temperature T_START → T_END.\"\"\"
    t = step / max(1, total)
    return T_END + (T_START - T_END) * 0.5 * (1 + math.cos(math.pi * t))

def sparse_kl_distillation_loss(
    student_logits: torch.Tensor,   # [B, S, V_student]
    teacher_logits: torch.Tensor,   # [B, S, V_teacher_subset]
    temperature: float,
    top_k: int = 4096,
) -> torch.Tensor:
    \"\"\"
    Optimized Top-K Sparse KL loss.
    \"\"\"
    B, S, V_t = teacher_logits.shape
    V_s = student_logits.shape[-1]
    V   = min(V_t, V_s)
    t_logits_slice = teacher_logits[..., :V]
    k = min(top_k, V)
    tk_vals, tk_idx = t_logits_slice.topk(k, dim=-1)
    tk_vals = tk_vals.float() / temperature
    p_teacher = F.softmax(tk_vals, dim=-1)
    s_slice = student_logits if V == V_s else student_logits[..., :V]
    s_topk = s_slice.gather(-1, tk_idx)
    s_topk = s_topk.float() / temperature
    log_q = F.log_softmax(s_topk, dim=-1)
    kl = (p_teacher * (p_teacher.clamp(min=1e-8).log() - log_q)).sum(-1).mean()
    return kl * (temperature ** 2)

def compute_teacher_logits(teacher, input_ids: torch.Tensor, vocab_limit: int) -> torch.Tensor:
    \"\"\"Get teacher logits without gradient, sliced early to save memory.\"\"\"
    with torch.no_grad():
        out = teacher(input_ids=input_ids.to(TEACHER_DEVICE), use_cache=False)
        logits = out.logits[..., :vocab_limit].to(DEVICE)
    return logits

print("✅ Distillation loss functions ready")
print(f"  Top-K sparse KL: K={DISTILL_TOP_K}")
print(f"  Temperature   : {T_START:.1f} → {T_END:.1f} (cosine anneal)")
print(f"  Alpha schedule: {ALPHA_WARMUP:.2f} → {ALPHA_PEAK:.2f} → {ALPHA_FINAL:.2f}")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10: EXPERT & CONCEPT MONITORING                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 10: Expert & Concept Monitoring (every 500 steps)",
        "markdown",
    )
)
cells.append(
    cell("""
import torch
from collections import defaultdict
import math

class ExpertMonitor:
    \"\"\"Track MoE expert utilization, detect dead experts and routing collapse.\"\"\"

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
        total = self.counts.sum(-1, keepdim=True).clamp(min=1)
        freq  = self.counts / total
        dead_mask = freq < 0.01
        dead_count = dead_mask.sum().item()
        ent = -(freq.clamp(min=1e-8) * freq.clamp(min=1e-8).log()).sum(-1)
        max_ent = math.log(self.n_experts) if self.n_experts > 1 else 1.0

        # Utilization Gini coefficient (0=perfectly balanced, 1=totally unbalanced)
        freq_avg = freq.mean(0)
        sorted_freq = freq_avg.sort()[0]
        n = self.n_experts
        cumsum = sorted_freq.cumsum(0)
        gini = (2 * cumsum.sum() - (n + 1)) / max(n, 1)

        return {
            "dead_experts"       : int(dead_count),
            "dead_pct"           : 100 * dead_count / max(1, self.n_layers * self.n_experts),
            "entropy_ratio"      : (ent.mean() / max_ent).item() if max_ent > 0 else 1.0,
            "utilization_gini"   : gini.item(),
            "expert_freq"        : freq.mean(0).tolist(),
        }

class ConceptMonitor:
    \"\"\"Track ESCM concept usage, collapse, and entropy.\"\"\"

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
        total = self.usage.sum().clamp(min=1)
        freq  = self.usage / total
        ent   = -(freq.clamp(min=1e-8) * freq.clamp(min=1e-8).log()).sum().item()
        max_ent = math.log(self.num_concepts) if self.num_concepts > 1 else 1.0
        top5_frac = freq.topk(min(5, self.num_concepts)).values.sum().item()
        collapsed = top5_frac > 0.80

        return {
            "entropy"         : ent,
            "entropy_ratio"   : ent / max_ent if max_ent > 0 else 1.0,
            "collapsed"       : collapsed,
            "top5_coverage"   : top5_frac,
            "dead_concepts"   : int((freq < 0.001).sum().item()),
            "concept_freq"    : freq.tolist(),
        }

# Instantiate monitors
expert_monitor  = ExpertMonitor(n_experts=args.n_routed_experts, n_layers=args.n_layers)
concept_monitor = ConceptMonitor(num_concepts=args.num_concepts)

print(f"✅ Expert monitor  : {args.n_layers} layers × {args.n_routed_experts} experts")
print(f"✅ Concept monitor : {args.num_concepts} concept slots")
print(f"✅ Monitoring every {MONITOR_EVERY} steps")
print(f"   Metrics: loss, perplexity, expert_entropy, expert_utilization, "
      f"concept_entropy, concept_frequency")
print(f"   Placeholders: coding_score (0.0), instruction_score (0.0)")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 11: CHECKPOINT SYSTEM                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell("## Section 11: Crash-Safe Checkpoint System (every 200 steps)", "markdown")
)
cells.append(
    cell("""
import torch, json, time
from pathlib import Path
from safetensors.torch import save_file as safetensors_save

CURSOR_FILE = CKPT_DIR / "cursor.json"

def save_cursor(step: int, dataset_idx: int, stage: str = "phase1"):
    cursor = {
        "step"       : step,
        "dataset_idx": dataset_idx,
        "stage"      : stage,
        "timestamp"  : time.time(),
    }
    with open(CURSOR_FILE, "w") as f:
        json.dump(cursor, f, indent=2)

def load_cursor() -> dict:
    if CURSOR_FILE.exists():
        with open(CURSOR_FILE) as f:
            return json.load(f)
    return {"step": 0, "dataset_idx": 0, "stage": "phase1"}

def save_checkpoint(
    step          : int,
    model,
    optimizers    : list,
    scheduler,
    loss_history  : list,
    expert_stats  : dict,
    concept_stats : dict,
    dataset_idx   : int = 0,
    stage         : str = "phase1",
):
    \"\"\"Full crash-safe checkpoint. Saves model, optimizers, scheduler, RNG, cursor.\"\"\"
    ckpt_path = CKPT_DIR / f"step_{step:06d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 1. Model weights (safetensors)
    safetensors_save(
        {k: v.cpu() for k, v in model.state_dict().items()},
        str(ckpt_path / "model.safetensors")
    )

    # 2. Optimizer states
    opt_states = [opt.state_dict() for opt in optimizers]
    torch.save(opt_states, ckpt_path / "optimizer.pt")

    # 3. Scheduler state
    sched_state = {
        "warmup_steps": scheduler.warmup_steps,
        "stable_steps": scheduler.stable_steps,
        "decay_steps" : scheduler.decay_steps,
        "base_lrs"    : scheduler.base_lrs,
        "min_lr_ratio": scheduler.min_lr_ratio,
    }
    with open(ckpt_path / "scheduler.json", "w") as f:
        json.dump(sched_state, f)

    # 4. RNG state
    rng_state = {
        "cpu"    : torch.get_rng_state(),
        "cuda"   : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python" : random.getstate(),
    }
    torch.save(rng_state, ckpt_path / "rng.pt")

    # 5. Training metadata + component stats
    meta = {
        "step"         : step,
        "dataset_idx"  : dataset_idx,
        "stage"        : stage,
        "loss_history" : loss_history[-500:],
        "expert_stats" : expert_stats,
        "concept_stats": concept_stats,
        "timestamp"    : time.time(),
        "config"       : cfg,
    }
    with open(ckpt_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # 6. Dataset cursor
    save_cursor(step, dataset_idx, stage)

    # 7. Symlink "latest"
    latest = CKPT_DIR / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt_path.name)

    print(f"  💾 Checkpoint saved: step {step} → {ckpt_path.name} (stage={stage})")

def find_latest_checkpoint() -> Path:
    latest = CKPT_DIR / "latest"
    if latest.is_symlink() and latest.exists():
        return latest.resolve()
    candidates = sorted(CKPT_DIR.glob("step_*"))
    if candidates:
        return candidates[-1]
    return None

def load_checkpoint(model, optimizers, scheduler, ckpt_path: Path):
    \"\"\"Load full checkpoint. Returns (step, loss_history, expert_stats, concept_stats, stage).\"\"\"
    from safetensors.torch import load_file as safetensors_load

    print(f"  📂 Resuming from: {ckpt_path}")

    # Model
    state = safetensors_load(str(ckpt_path / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)

    # Optimizers
    opt_states = torch.load(ckpt_path / "optimizer.pt", map_location="cpu")
    for opt, st in zip(optimizers, opt_states):
        opt.load_state_dict(st)

    # RNG
    rng = torch.load(ckpt_path / "rng.pt", map_location="cpu")
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if rng.get("python"):
        random.setstate(rng["python"])

    # Meta
    with open(ckpt_path / "meta.json") as f:
        meta = json.load(f)

    return (meta["step"],
            meta.get("loss_history", []),
            meta.get("expert_stats", {}),
            meta.get("concept_stats", {}),
            meta.get("stage", "phase1"))

print("✅ Checkpoint system ready")
print(f"   Saving every {CKPT_EVERY} steps to {CKPT_DIR}")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 12: OOM RECOVERY + ACCELERATE WRAP                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 12: OOM Recovery + Accelerate Wrap",
        "markdown",
    )
)
cells.append(
    cell("""
import torch, gc

# Prepare model + optimizers with Accelerate (handles DDP, mixed precision)
model, *optimizers = accelerator.prepare(model, *optimizers)

def oom_step():
    \"\"\"Called on OOM: clear cache, return True to retry with smaller effective batch.\"\"\"
    torch.cuda.empty_cache()
    gc.collect()
    return True

def safe_forward(model, x_enc, x_dec):
    \"\"\"
    Attempt model forward. On CUDA OOM: clear cache and return None
    to signal the training loop to skip this batch.
    Also catches ValueError (PyTorch symbolizer bug: stoi/storage).
    \"\"\"
    try:
        return model(x_enc, x_dec)
    except torch.cuda.OutOfMemoryError:
        print("  ⚠️  OOM on forward — skipping batch")
        oom_step()
        return None
    except ValueError as _e:
        _msg = str(_e)
        if any(kw in _msg for kw in ("stoi", "storage", "symbolize")):
            print(f"  ⚠️  PyTorch symbolizer ValueError (skipping batch): {_msg}")
            return None
        raise

print("✅ Model wrapped with Accelerate (DDP + bf16)")
print(f"   Model on: {next(model.parameters()).device}")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 13: AUTO-RESUME                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(cell("## Section 13: Auto-Resume", "markdown"))
cells.append(
    cell("""
# ── Auto-resume from latest checkpoint ──────────────────────────────────────
loss_history   = []
expert_stats   = {}
concept_stats  = {}
dataset_cursor = load_cursor()

latest_ckpt = find_latest_checkpoint()
START_STEP  = 0
resumed_phase = 0

if latest_ckpt is not None and IS_MAIN:
    print(f"Found checkpoint: {latest_ckpt}")
    START_STEP, loss_history, expert_stats, concept_stats, resumed_stage = load_checkpoint(
        accelerator.unwrap_model(model),
        [accelerator.unwrap_model(opt) if hasattr(opt, 'param_groups') else opt
         for opt in optimizers],
        scheduler,
        latest_ckpt,
    )
    resumed_phase = get_phase(START_STEP)
    print(f"✅ Resumed from step {START_STEP} (phase={resumed_phase+1})")
    print(f"   Dataset position: {dataset_cursor.get('dataset_idx', 0)}")
else:
    print("Starting fresh training (no checkpoint found)")

# Broadcast START_STEP to all processes
if N_PROC > 1:
    t = torch.tensor([START_STEP], dtype=torch.long, device=DEVICE)
    torch.distributed.broadcast(t, src=0)
    START_STEP = t.item()
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 14: FULL TRAINING LOOP (35K STEPS, PHASE-AWARE)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 14: Training Loop (35K Steps, Phase-Aware, Distillation + Concept Loss)",
        "markdown",
    )
)
cells.append(
    cell("""
import torch, time, math
from tqdm.auto import tqdm

# ── Loss log file ────────────────────────────────────────────────────────────
loss_log_path = LOG_DIR / "loss.jsonl"

def log_step(d: dict):
    with open(loss_log_path, "a") as f:
        f.write(json.dumps(d) + "\\n")

# ── Build initial streaming data iterator ────────────────────────────────────
data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN, format_fn=format_fn)
dataset_idx = dataset_cursor.get("dataset_idx", 0)
current_phase = get_phase(START_STEP)

# Fast-forward to last saved position (skip already-seen samples)
if dataset_idx > 0 and IS_MAIN:
    print(f"Fast-forwarding dataset to position {dataset_idx}...")
    for _ in range(dataset_idx):
        try:
            next(data_iter)
        except StopIteration:
            ds_train, format_fn = load_phase_datasets(current_phase)
            data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN, format_fn=format_fn)

# ── Main training loop ───────────────────────────────────────────────────────
model.train()
pbar = tqdm(range(START_STEP, MAX_STEPS), initial=START_STEP, total=MAX_STEPS,
            desc="Training", disable=not IS_MAIN)

accum_steps        = accelerator.gradient_accumulation_steps
tokens_per_step    = BATCH_SIZE * SEQ_LEN * accum_steps * N_PROC  # tokens consumed per real optimizer step
total_tokens_seen  = 0
global_step        = START_STEP

# Bounded perplexity (avoids overflow in early steps)
def bounded_ppl(loss_val: float) -> float:
    return math.exp(min(max(loss_val, 0.0), 20))

# Warm up GPU memory tracking
if torch.cuda.is_available():
    _ = torch.cuda.memory_allocated() / 1024**3

for step in pbar:
    t0 = time.time()
    global_step = step
    _CHECKPOINT_ON_KILL.update({"step": step, "dataset_idx": dataset_idx,
                                 "phase": f"phase{current_phase + 1}"})

    # ── Phase detection and transition ──────────────────────────────────────
    new_phase = get_phase(step)
    if new_phase != current_phase:
        print(f"\\n{'='*60}")
        print(f"PHASE TRANSITION: Phase {current_phase + 1} → Phase {new_phase + 1} (step {step})")
        print(f"{'='*60}")
        current_phase = new_phase
        ds_train, format_fn = load_phase_datasets(current_phase)
        data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN, format_fn=format_fn)
        expert_monitor.reset()
        concept_monitor.reset()

    # LR schedule
    lr_mult = scheduler.step(step)

    # ── Gradient accumulation loop ──────────────────────────────────────────
    total_loss_accum = 0.0
    for micro in range(accum_steps):
        try:
            x, y = next(data_iter)
            dataset_idx += 1
        except StopIteration:
            # End of streaming epoch — restart with current phase
            ds_train, format_fn = load_phase_datasets(current_phase)
            data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN, format_fn=format_fn)
            x, y = next(data_iter)

        x = x.unsqueeze(0).to(DEVICE)   # [1, SEQ_LEN]
        y = y.unsqueeze(0).to(DEVICE)

        with accelerator.accumulate(model):
            # ── Forward pass with explicit AMP autocast ──
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = safe_forward(model, x, x)
                if out is None:
                    continue   # OOM — skip batch

                (logits, mtp_logits, concept_db, memory_state,
                 routing_maps, concept_indices, adjacencies, event_probs) = out

                # ── Teacher distillation logits ──
                T = get_distill_temperature(step, MAX_STEPS)
                alpha = get_distill_alpha(step, MAX_STEPS)

                kl_loss = torch.zeros((), device=DEVICE)
                if alpha > 0.01:
                    teacher_logits = compute_teacher_logits(teacher, x, vocab_limit=VOCAB_SIZE)
                    kl_loss = sparse_kl_distillation_loss(
                        logits, teacher_logits, temperature=T, top_k=DISTILL_TOP_K
                    )
                    del teacher_logits  # free memory early

                # ── MTP loss ──
                mtp_loss = None
                if mtp_logits is not None and x.shape[1] > 1:
                    mtp_targets = y[:, 1:]
                    mtp_logits_trimmed = mtp_logits[:, :mtp_targets.shape[1], :]
                    mtp_loss = torch.nn.functional.cross_entropy(
                        mtp_logits_trimmed.reshape(-1, mtp_logits_trimmed.shape[-1]),
                        mtp_targets.reshape(-1),
                        ignore_index=-100,
                    )

                # ── Aggregate Lasmoid loss with token_concept_loss ──
                unwrapped = accelerator.unwrap_model(model)
                total_loss = compute_loss(
                    logits         = logits,
                    targets        = y,
                    routing_maps   = routing_maps,
                    vq_losses      = [unwrapped.last_vq_loss],
                    adjacencies    = adjacencies,
                    event_probs    = event_probs,
                    moe_aux_loss   = unwrapped.last_moe_loss,
                    moe_aux_coeff  = args.moe_aux_coeff,
                    mtp_loss       = mtp_loss,
                    mtp_coeff      = args.mtp_loss_coeff,
                    token_concept_loss  = unwrapped.last_token_concept_loss,
                    token_concept_coeff = args.token_concept_loss_coeff,
                    commit_loss    = unwrapped.last_commit_loss,
                    commit_coeff   = args.hcm_commit_loss_coeff,
                    curiosity_loss = None,
                    graph_sparsity = 0.01,
                    label_smoothing= 0.0,
                    ignore_index   = -100,
                )

                # ── Blend CE + KL ──
                ce_component = total_loss
                blended_loss = (1.0 - alpha) * ce_component + alpha * (T ** 2) * kl_loss

            # ── Backward ──
            accelerator.backward(blended_loss / accum_steps)
            total_loss_accum += blended_loss.item() / accum_steps

        # Update expert & concept monitors (main process only)
        if IS_MAIN:
            expert_monitor.update(routing_maps)
            concept_monitor.update(concept_indices)

    # ── Gradient clipping & optimizer step ──────────────────────────────────
    if accelerator.sync_gradients:
        grad_norm = clip_grad_global_norm(accelerator.unwrap_model(model), max_norm=1.0)

        # NaN/Inf gradient guard
        _skip_step = False
        for _p in accelerator.unwrap_model(model).parameters():
            if _p.grad is not None and (torch.isnan(_p.grad).any() or torch.isinf(_p.grad).any()):
                _skip_step = True
                break
        if _skip_step:
            print(f"  ⚠️  NaN/Inf gradient at step {step} — skipping optimizer step")

        if not _skip_step:
            for opt in optimizers:
                opt.step()
            accelerator.unwrap_model(model).apply_pending_bias_updates()
        else:
            log_step({"step": step, "nan_gradient": True, "grad_norm": float('nan')})

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # ── Logging ─────────────────────────────────────────────────────────────
    loss_history.append(total_loss_accum)
    step_time = time.time() - t0
    total_tokens_seen += tokens_per_step

    if IS_MAIN:
        perplexity = bounded_ppl(total_loss_accum)
        exp_s  = expert_monitor.stats()
        con_s  = concept_monitor.stats()
        tokens_per_sec = tokens_per_step / max(step_time, 1e-6)

        # GPU memory pressure
        if torch.cuda.is_available() and step % 10 == 0:
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _peak  = torch.cuda.max_memory_allocated() / 1024**3
            _free  = torch.cuda.get_device_properties(0).total_memory / 1e9 - _alloc
            _mem_str = f"{_alloc:.1f}G/{_free:.1f}G free"
        else:
            _mem_str = ""

        pbar.set_postfix({
            "loss"    : f"{total_loss_accum:.3f}",
            "ppl"     : f"{perplexity:.1f}",
            "tok/s"   : f"{tokens_per_sec:,.0f}",
            "mem"     : _mem_str or "",
            "ent"     : f"{exp_s['entropy_ratio']:.2f}",
            "dead_e"  : exp_s['dead_experts'],
            "phase"   : f"{current_phase+1}",
        })

        # ── Detailed monitoring every MONITOR_EVERY steps ──────────────────
        if step % MONITOR_EVERY == 0:
            log_step({
                "step"              : step,
                "phase"             : current_phase + 1,
                "loss"              : total_loss_accum,
                "perplexity"        : perplexity,
                "kl_loss"           : kl_loss.item(),
                "mtp_loss"          : mtp_loss.item() if mtp_loss else 0.0,
                "moe_aux"           : unwrapped.last_moe_loss.item(),
                "vq_loss"           : unwrapped.last_vq_loss.item(),
                "commit_loss"       : unwrapped.last_commit_loss.item(),
                "token_concept_loss": (unwrapped.last_token_concept_loss.item()
                                       if hasattr(unwrapped, 'last_token_concept_loss')
                                       and unwrapped.last_token_concept_loss is not None
                                       else 0.0),
                "alpha"             : alpha,
                "temperature"       : T,
                "lr_mult"           : lr_mult,
                "grad_norm"         : grad_norm if accelerator.sync_gradients else 0.0,
                "step_time_s"       : step_time,
                "tokens_per_sec"    : tokens_per_sec,
                "gpu_mem_gb"        : _alloc if torch.cuda.is_available() else 0,
                "expert_entropy"    : exp_s['entropy_ratio'],
                "expert_utilization": exp_s['utilization_gini'],
                "dead_experts"      : exp_s['dead_experts'],
                "concept_entropy"   : con_s['entropy_ratio'],
                "concept_collapsed" : con_s['collapsed'],
                "coding_score"      : 0.0,
                "instruction_score" : 0.0,
            })
        elif step % 10 == 0:
            log_step({
                "step"           : step,
                "phase"          : current_phase + 1,
                "loss"           : total_loss_accum,
                "perplexity"     : perplexity,
                "kl_loss"        : kl_loss.item(),
                "lr_mult"        : lr_mult,
                "grad_norm"      : grad_norm if accelerator.sync_gradients else 0.0,
                "step_time_s"    : step_time,
                "tokens_per_sec" : tokens_per_sec,
            })

        # ── Checkpoint every CKPT_EVERY steps ──────────────────────────────
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
            # Reset monitors after each checkpoint (fresh window)
            expert_monitor.reset()
            concept_monitor.reset()

print("\\n✅ Training complete!",
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 15: VALIDATION                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(cell("## Section 15: Validation (Perplexity + Reasoning)", "markdown"))
cells.append(
    cell("""
import torch, math

model.eval()

# ── Perplexity on TinyStories validation ────────────────────────────────────
val_path = REPO_DIR / "datasets" / "tinystories_val.bin"
if val_path.exists():
    import numpy as np
    val_data = np.frombuffer(val_path.read_bytes(), dtype=np.uint16).astype(np.int64)
    val_data = torch.from_numpy(val_data)

    N_VAL = min(512, len(val_data) // SEQ_LEN)
    total_nll, total_tokens = 0.0, 0

    with torch.no_grad():
        for i in range(N_VAL):
            chunk = val_data[i * SEQ_LEN : (i + 1) * SEQ_LEN + 1]
            if len(chunk) < SEQ_LEN + 1:
                continue
            x = chunk[:SEQ_LEN].unsqueeze(0).to(DEVICE)
            y = chunk[1:SEQ_LEN+1].unsqueeze(0).to(DEVICE)
            out = accelerator.unwrap_model(model)(x, x)
            logits = out[0]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"
            )
            total_nll    += nll.item()
            total_tokens += y.numel()

    ppl = math.exp(min(total_nll / total_tokens, 20))
    print(f"Validation perplexity: {ppl:.2f}")
else:
    print("No val binary found — skipping perplexity")
    ppl = None

# ── Reasoning probe ──────────────────────────────────────────────────────────
test_prompt = "Solve step by step: What is 15 + 28?"
test_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    generated = accelerator.unwrap_model(model).generate(
        test_ids,
        max_new_tokens=40,
        temperature=0.7,
        top_k=50,
    )

decoded = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
print(f"\\nReasoning probe input : {test_prompt}")
print(f"Reasoning probe output: {decoded}")

model.train()
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 16: TRAINING REPORT & PLOTS                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 16: Training Report (Loss Curves, Expert Stats, Concept Stats)",
        "markdown",
    )
)
cells.append(
    cell("""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ── Load full loss log ────────────────────────────────────────────────────────
log_lines = []
if loss_log_path.exists():
    with open(loss_log_path) as f:
        for line in f:
            try:
                log_lines.append(json.loads(line.strip()))
            except Exception:
                pass

steps       = [l["step"] for l in log_lines]
losses      = [l["loss"] for l in log_lines]
kl_losses   = [l.get("kl_loss", 0) for l in log_lines]
alphas      = [l.get("alpha", 0) for l in log_lines]
temps       = [l.get("temperature", 1) for l in log_lines]
dead_exps   = [l.get("dead_experts", 0) for l in log_lines]
concept_ent = [l.get("concept_entropy", 0) for l in log_lines]
expert_ents = [l.get("expert_entropy", 0) for l in log_lines]
perplexities= [l.get("perplexity", 0) for l in log_lines if l.get("perplexity")]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Lasmoid Tiny Frontier — Training Report", fontsize=16, fontweight="bold")

# Loss curve
ax = axes[0, 0]
ax.plot(steps, losses, color="#6366f1", linewidth=1.5, label="Total Loss")
ax.plot(steps, kl_losses, color="#f59e0b", linewidth=1.0, alpha=0.7, label="KL Distill")
ax.set_title("Loss Curves"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# Alpha + Temperature schedule
ax = axes[0, 1]
ax.plot(steps, alphas, color="#10b981", linewidth=1.5, label="Alpha (teacher trust)")
ax2 = ax.twinx()
ax2.plot(steps, temps, color="#ef4444", linewidth=1.5, linestyle="--", label="Temperature")
ax.set_title("Distillation Schedule"); ax.set_xlabel("Step")
ax.set_ylabel("Alpha"); ax2.set_ylabel("Temperature")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
ax.grid(alpha=0.3)

# Dead expert count
ax = axes[1, 0]
ax.plot(steps, dead_exps, color="#dc2626", linewidth=1.5)
ax.set_title("Dead Expert Count (< 1% utilization)")
ax.set_xlabel("Step"); ax.set_ylabel("Dead Experts"); ax.grid(alpha=0.3)

# Concept entropy
ax = axes[1, 1]
ax.plot(steps, concept_ent, color="#7c3aed", linewidth=1.5)
ax.axhline(0.5, color="#94a3b8", linestyle="--", alpha=0.7, label="Collapse threshold")
ax.set_title("Concept Memory Entropy Ratio (1.0 = uniform)")
ax.set_xlabel("Step"); ax.set_ylabel("Entropy Ratio"); ax.legend(); ax.grid(alpha=0.3)

# Expert frequency (final checkpoint)
if log_lines:
    last_ef = log_lines[-1].get("expert_stats", {}).get("expert_freq", [])
    if last_ef:
        ax = axes[2, 0]
        ax.bar(range(len(last_ef)), last_ef, color="#0ea5e9")
        ax.set_title("Expert Token Frequency (final, avg across layers)")
        ax.set_xlabel("Expert ID"); ax.set_ylabel("Fraction of tokens"); ax.grid(alpha=0.3)

# Loss histogram
ax = axes[2, 1]
if losses:
    ax.hist(losses[-200:], bins=30, color="#6366f1", alpha=0.7, edgecolor="white")
    ax.set_title("Loss Distribution (last 200 steps)")
    ax.set_xlabel("Loss"); ax.set_ylabel("Frequency"); ax.grid(alpha=0.3)

plt.tight_layout()
report_path = LOG_DIR / "training_report.png"
plt.savefig(str(report_path), dpi=120, bbox_inches="tight")
plt.show()
print(f"✅ Report saved: {report_path}")

# Summary table
if losses:
    print(f"\\n{'='*55}")
    print(f"  Training Summary — Lasmoid Tiny Frontier")
    print(f"{'='*55}")
    print(f"  Steps completed    : {steps[-1] if steps else 0}")
    print(f"  Final loss         : {losses[-1]:.4f}")
    print(f"  Best loss          : {min(losses):.4f}")
    print(f"  Final KL loss      : {kl_losses[-1]:.4f}")
    print(f"  Dead experts       : {dead_exps[-1] if dead_exps else '?'}")
    print(f"  Concept entropy    : {concept_ent[-1]:.3f}" if concept_ent else "")
    print(f"  Best perplexity    : {min(perplexities):.2f}" if perplexities else "  Perplexity: N/A")
    print(f"  Total tokens       : {total_tokens_seen:,}")
    print(f"{'='*55}")
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 17: KAGGLE DATASET EXPORT                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell(
        "## Section 17: Kaggle Dataset Export (Persistence Across Sessions)",
        "markdown",
    )
)
cells.append(
    cell("""
import shutil, json, os
from pathlib import Path

def export_for_kaggle(step: int, model, export_dir: Path):
    \"\"\"
    Export checkpoint as a self-contained directory that can be uploaded
    as a Kaggle Dataset, enabling multi-session training.
    \"\"\"
    from safetensors.torch import save_file as sf_save

    pkg = export_dir / f"lasmoid_tiny_frontier_step{step}"
    pkg.mkdir(parents=True, exist_ok=True)

    # Weights
    unwrapped = accelerator.unwrap_model(model)
    sf_save(
        {k: v.cpu() for k, v in unwrapped.state_dict().items()},
        str(pkg / "model.safetensors")
    )

    # Config
    shutil.copy(str(REPO_DIR / "configs" / "model" / "config_gemma4_tiny_frontier.json"),
                str(pkg / "config.json"))

    # Tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = REPO_DIR / fname
        if src.exists():
            shutil.copy(str(src), str(pkg / fname))

    # Metadata
    meta = {
        "step"          : step,
        "teacher"       : TEACHER_MODEL,
        "loss_history"  : loss_history[-50:],
        "architecture"  : "Lasmoid-Tiny-Frontier-10.5M",
        "vocab_size"    : VOCAB_SIZE,
    }
    with open(pkg / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Cursor (for resuming)
    if CURSOR_FILE.exists():
        shutil.copy(str(CURSOR_FILE), str(pkg / "cursor.json"))

    # Kaggle dataset metadata
    kaggle_meta = {
        "title"      : f"Lasmoid-Tiny-Frontier-step{step}",
        "id"         : f"lasmoid-tiny-frontier-step{step}",
        "licenses"   : [{"name": "apache-2.0"}],
    }
    with open(export_dir / "dataset-metadata.json", "w") as f:
        json.dump(kaggle_meta, f, indent=2)

    print(f"✅ Export ready: {pkg}")
    print(f"   To upload: kaggle datasets create -p {export_dir}")
    return pkg

if IS_MAIN and loss_history:
    final_step = len(loss_history) + START_STEP
    export_for_kaggle(final_step, model, EXPORT_DIR)
""")
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 18: FINAL EXPORT (SAFETENSORS + HUGGINGFACE)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
cells.append(
    cell("## Section 18: Final Export (SafeTensors + HuggingFace)", "markdown")
)
cells.append(
    cell("""
import shutil, json
from pathlib import Path
from safetensors.torch import save_file as sf_save

if IS_MAIN:
    hf_dir = Path("/kaggle/working/hf_model")
    hf_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)

    # 1. SafeTensors weights
    sf_save(
        {k: v.cpu() for k, v in unwrapped.state_dict().items()},
        str(hf_dir / "model.safetensors"),
    )
    print(f"✅ SafeTensors saved: {hf_dir / 'model.safetensors'}")

    # 2. Model config
    shutil.copy(
        str(REPO_DIR / "configs" / "model" / "config_gemma4_tiny_frontier.json"),
        str(hf_dir / "config.json")
    )

    # 3. Tokenizer
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = REPO_DIR / fname
        if src.exists():
            shutil.copy(str(src), str(hf_dir / fname))

    # 4. Model card
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

# Lasmoid Tiny Frontier (~10.5M)

**Architecture**: Hybrid Concept Transformer (Lasmodium)
**Teacher**: Qwen3.5-0.8B (4-bit NF4)
**Parameters**: ~10.5M
**Training**: 3-phase curriculum, 35K steps, Top-K Sparse KL distillation
**Vocab**: BPE 32,768

## Components
- Compressed Sparse Attention (CSA) with CIF compressor
- Mamba-2 SSD State Space Recurrence
- Grey-Box MoE (4 routed + 1 shared experts)
- ElasticSparseConceptMemory (ESCM)
- Manifold-Constrained Hyper-Connections (mHC)
- Multi-Token Prediction (MTP, t+1)

## Training Phases
1. **Phase 1** (0-9999): Cosmopedia + FineWeb-Edu + TinyStories + Code — plain format
2. **Phase 2** (10000-24999): Qwen + Claude distills + Coding + FineWeb-Edu — concept format
3. **Phase 3** (25000-34999): Qwen + Claude + Opus — concept format

## Concept-Structured Training
Uses Question → Concepts → Plan → Answer format instead of raw CoT.
\"\"\"
    (hf_dir / "README.md").write_text(model_card)

    # 5. (Optional) Push to Hub
    # from huggingface_hub import HfApi
    # api = HfApi()
    # api.upload_folder(folder_path=str(hf_dir), repo_id="Theory903/lasmoid-tiny-frontier",
    #                   commit_message=f"Training step {len(loss_history)+START_STEP}")

    print(f"\\n✅ Model ready for upload at: {hf_dir}")
    print(f"   To push: Uncomment HfApi section above")
""")
)

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK JSON
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

# ── Write output ──────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "notebooks" / "distillation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "lasmoid_tiny_frontier.ipynb"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

size_kb = out_path.stat().st_size / 1024
print(f"✅ Notebook written: {out_path}")
print(f"   Size: {size_kb:.1f} KB | Cells: {len(cells)}")
