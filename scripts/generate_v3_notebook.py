"""
generate_v3_notebook.py
=======================
Generates notebooks/distillation/lasmoid_kaggle_v3.ipynb
— a production-grade Lasmoid-specific training notebook.
Run:  python scripts/generate_v3_notebook.py
"""

import json, textwrap, os
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
# Lasmoid V3 — Production Kaggle Training Notebook
### Hybrid Concept Transformer | Gemma-4-12B Distillation
*20 Sections · Lasmoid-specific APIs · Production-grade crash recovery*
""",
        "markdown",
    )
)

# ── SECTION 1: Environment Setup ─────────────────────────────────────────────
cells.append(cell("## Section 1: Environment Setup", "markdown"))
cells.append(
    cell("""
# Install uv and packages using direct shell commands
!pip install -q uv
!uv pip install --system -q git+https://github.com/huggingface/transformers.git bitsandbytes>=0.46.0 accelerate>=1.6.0 datasets>=3.6.0 safetensors>=0.5.3 sentencepiece einops tqdm matplotlib psutil

import os, sys, json, time, math, random, gc, shutil
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Auto-detect if new transformers is imported in this session. If not, restart kernel.
try:
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    has_gemma4 = "gemma4_unified" in CONFIG_MAPPING
except Exception:
    has_gemma4 = False

if not has_gemma4:
    print("🔄 Upgraded transformers to main branch, but the running Python kernel is still using the old version.")
    print("🔄 Restarting kernel automatically to load the new transformers code...")
    os.kill(os.getpid(), 9)
else:
    print(f"✅ Transformers has Gemma-4 support ready! (version: {transformers.__version__})")

# Skip flash-attn compilation to avoid long installation times.
# Native PyTorch SDPA (scaled_dot_product_attention) is already optimized and uses FlashAttention under the hood when available.
HAS_FLASH = False
print("Using native PyTorch SDPA (scaled_dot_product_attention)")

print("✅ Environment ready")
""")
)

# ── SECTION 2: GPU Detection ─────────────────────────────────────────────────
cells.append(cell("## Section 2: GPU Detection", "markdown"))
cells.append(
    cell("""
import torch

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

# Compute dtype
DTYPE = torch.bfloat16  # T4 supports bf16 via software emulation; works for training
print(f"Compute dtype: {DTYPE}")
print(f"Total VRAM   : {GPU['total_vram_gb']:.1f} GB")
""")
)

# ── SECTION 3: Multi-GPU Setup ───────────────────────────────────────────────
cells.append(cell("## Section 3: Accelerate Multi-GPU Setup (DDP)", "markdown"))
cells.append(
    cell("""
from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=16,  # effective batch = BATCH * 16 per GPU
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

# ── SECTION 4: Paths & Config ────────────────────────────────────────────────
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
CKPT_DIR   = Path("/kaggle/working/checkpoints")
LOG_DIR    = Path("/kaggle/working/logs")
EXPORT_DIR = Path("/kaggle/working/export")

for d in [REPO_DIR, CKPT_DIR, LOG_DIR, EXPORT_DIR]:
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

# ── SECTION 4b: Graceful Shutdown ────────────────────────────────────────────
cells.append(
    cell("## Section 4b: Graceful Shutdown (SIGTERM/SIGINT Handler)", "markdown")
)
cells.append(
    cell("""
import signal

# Register a handler that saves checkpoint on Kaggle session timeout (SIGTERM)
# or manual interrupt (SIGINT).  Without this, an abrupt kill loses all progress
# since the last checkpoint save.
_CHECKPOINT_ON_KILL = {"step": 0, "dataset_idx": 0}

def _shutdown_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    print(f"\\n⚠️  Received {sig_name} — saving emergency checkpoint...")
    step = _CHECKPOINT_ON_KILL.get("step", 0)
    ds_idx = _CHECKPOINT_ON_KILL.get("dataset_idx", 0)
    if step > 0:
        ckpt_dir = CKPT_DIR / "emergency"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file as sf_save
        sf_save(
            {k: v.cpu() for k, v in accelerator.unwrap_model(model).state_dict().items()},
            str(ckpt_dir / "model.safetensors"),
        )
        save_cursor(step, ds_idx)
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

# ── SECTION 4c: Dataset Streaming ────────────────────────────────────────────
cells.append(
    cell(
        "## Section 4c: Dataset Streaming (FineWeb-Edu + Cosmopedia + Distillation)",
        "markdown",
    )
)
cells.append(
    cell("""
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import torch

# ── Tokenizer (Lasmoid BPE, 129,286 vocab) ──────────────────────────────────
TOK_PATH = str(REPO_DIR)   # tokenizer.json lives at repo root
tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = len(tokenizer)
print(f"Tokenizer vocab size: {VOCAB_SIZE:,}")

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEQ_LEN      = 512       # max_seq_len for 100M config
BATCH_SIZE   = 1         # per GPU; effective = 1 × 2 GPUs × 16 grad_accum = 32 (optimized for T4 VRAM)
MAX_STEPS    = 5000
CKPT_EVERY   = 50        # crash-safe: save every 50 steps
TEACHER_MODEL= "google/gemma-4-12B"

# ── Lasmoid ablation flags (from ablation_results.jsonl) ────────────────────
# domain_cortex: KEEP  (perplexity 54366 vs baseline 54840)
# adaptive_k:    REMOVE (perplexity 54893 — WORSE)
# relational:    REMOVE (perplexity 60458 — MUCH WORSE)
# curiosity:     REMOVE (perplexity 55520 — worse)
ABLATION_FLAGS = {
    "use_domain_cortex"    : True,
    "moe_adaptive_routing" : False,   # adaptive_k = OFF
    "use_relational_cortex": False,   # relational  = OFF
    "use_curiosity_expert" : False,   # curiosity   = OFF
}

# ── Streaming datasets ───────────────────────────────────────────────────────
print("Loading datasets (streaming)...")

ds_fineweb = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "fineweb-edu-dedup",
    split="train", streaming=True,
    trust_remote_code=True,
).select_columns(["text"])

ds_cosmo = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "cosmopedia-v2",
    split="train", streaming=True,
    trust_remote_code=True,
).select_columns(["text"])

# Interleave with 70/30 split (FineWeb-Edu richer for language, Cosmo for reasoning)
ds_train = interleave_datasets(
    [ds_fineweb, ds_cosmo],
    probabilities=[0.70, 0.30],
    seed=SEED,
)
print("✅ Datasets ready (streaming)")
""")
)

# ── SECTION 5: Dataset Cleaning ──────────────────────────────────────────────
cells.append(cell("## Section 5: Dataset Cleaning & Sequence Packing", "markdown"))
cells.append(
    cell("""
import re
from typing import Iterator

MIN_CHARS = 200
MAX_CHARS = 8000

def clean_text(text: str) -> str:
    \"\"\"Basic quality filter for web text.\"\"\"
    text = re.sub(r"\\s+", " ", text).strip()
    # Remove nav/boilerplate markers
    for pat in ["Click here", "Subscribe to", "Cookie Policy", "Terms of Service",
                "©", "All rights reserved", "Skip to content"]:
        if pat.lower() in text.lower():
            return ""
    return text

def stream_packed(dataset, tokenizer, seq_len: int, max_batches: int = None) -> Iterator:
    \"\"\"
    Yields (input_ids [seq_len], labels [seq_len]) tensors from streaming dataset.
    Packs multiple documents into one sequence with EOS separator.
    Tracks position for crash-resume (via global cursor state).
    \"\"\"
    buf = []
    count = 0
    for example in dataset:
        text = clean_text(example.get("text", ""))
        if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.vocab.get("<｜end▁of▁sentence｜>", 1)
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

print("✅ Sequence packer ready")
""")
)

# ── SECTION 6: Tokenizer Analysis ────────────────────────────────────────────
cells.append(cell("## Section 6: Tokenizer Efficiency Analysis", "markdown"))
cells.append(
    cell("""
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def tokenizer_analysis(tokenizer, sample_texts: list, top_n: int = 50) -> dict:
    \"\"\"
    Compute:
    - Vocab utilization: how many unique tokens appear in sample
    - Compression ratio: chars / tokens
    - Dead token fraction: tokens never used in sample
    \"\"\"
    all_ids, all_chars = [], 0
    for text in sample_texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        all_chars += len(text)

    counter   = Counter(all_ids)
    n_unique  = len(counter)
    n_total   = len(all_ids)
    vocab_sz  = tokenizer.vocab_size
    compress  = all_chars / max(n_total, 1)
    dead_frac = 1.0 - n_unique / vocab_sz

    print(f"  Vocab size       : {vocab_sz:,}")
    print(f"  Unique tokens    : {n_unique:,}  ({100*n_unique/vocab_sz:.1f}% utilized)")
    print(f"  Dead tokens      : {vocab_sz - n_unique:,}  ({100*dead_frac:.1f}%)")
    print(f"  Chars/token (↑=better compression): {compress:.2f}")

    # ⚠️ For 100M model: embedding table = vocab_size × dim = 129286 × 384 = 49.6M params
    # = ~50% of model capacity just in embeddings!
    embed_params = vocab_sz * 384  # 100M config dim
    print(f"\\n  ⚠️  Embedding table: {embed_params/1e6:.1f}M params")
    print(f"     = {100*embed_params/(100e6):.0f}% of a 100M model's capacity")
    print(f"     Consider vocab pruning if dead_frac > 40%")

    # Plot token frequency distribution
    freqs = sorted(counter.values(), reverse=True)[:top_n]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(freqs)), freqs, color="#6366f1")
    ax.set_title(f"Top-{top_n} Token Frequencies (vocab util={100*n_unique/vocab_sz:.1f}%)")
    ax.set_xlabel("Token rank"); ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(str(LOG_DIR / "tokenizer_analysis.png"), dpi=100)
    plt.close()

    return {"vocab_util": n_unique/vocab_sz, "compress_ratio": compress, "dead_frac": dead_frac}

# Sample 200 texts for analysis
sample_gen = stream_packed(ds_train, tokenizer, SEQ_LEN, max_batches=1)
sample_texts = []
for ex in ds_train:
    sample_texts.append(ex.get("text","")[:2000])
    if len(sample_texts) >= 200:
        break

TOK_STATS = tokenizer_analysis(tokenizer, sample_texts)
""")
)

# ── SECTION 7: Model Construction ────────────────────────────────────────────
cells.append(
    cell(
        "## Section 7: Lasmoid Model Construction (100M, Ablation-Corrected)",
        "markdown",
    )
)
cells.append(
    cell("""
import json, sys
from pathlib import Path

sys.path.insert(0, str(REPO_DIR / "inference"))

from config import ModelArgs
from lasmoid import Lasmoid
from loss import compute_loss

# Load base 100M config
cfg_path = REPO_DIR / "configs" / "model" / "config_gemma4_100m.json"
with open(cfg_path) as f:
    cfg = json.load(f)

# Apply ablation flags: disable failed modules, enable winners
cfg.update(ABLATION_FLAGS)

# Kaggle-safe overrides
cfg["max_seq_len"]   = SEQ_LEN
cfg["max_batch_size"]= BATCH_SIZE

args = ModelArgs(**cfg)
model = Lasmoid(args).to(DEVICE)

# Enable gradient checkpointing (trades VRAM for recompute)
model.gradient_checkpointing = True

# Count parameters
total_params  = sum(p.numel() for p in model.parameters())
train_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
embed_params  = sum(p.numel() for n, p in model.named_parameters() if "emb" in n or "head" in n)
non_embed     = train_params - embed_params

print("=" * 55)
print(f"  Lasmoid 100M (Gemma-4 distillation target)")
print("=" * 55)
print(f"  Total params  : {total_params/1e6:.2f}M")
print(f"  Trainable     : {train_params/1e6:.2f}M")
print(f"  Embedding/head: {embed_params/1e6:.2f}M  ({100*embed_params/train_params:.0f}%)")
print(f"  Model body    : {non_embed/1e6:.2f}M")
print()
print(f"  Ablation config:")
for k, v in ABLATION_FLAGS.items():
    status = "✅ ON " if v else "❌ OFF"
    print(f"    {status}  {k}")
print()

# Estimate VRAM: params × 4 bytes (bf16 = 2 bytes × 2 for grad)
vram_est = train_params * 4 / 1e9
print(f"  VRAM estimate : ~{vram_est:.1f} GB (params+grads, bf16)")
print(f"  Seq len       : {SEQ_LEN}")
print(f"  Batch size    : {BATCH_SIZE} per GPU")
print("=" * 55)
""")
)

# ── SECTION 8: Optimizers ────────────────────────────────────────────────────
cells.append(
    cell("## Section 8: Muon + AdamW Dual Optimizer & WSD Scheduler", "markdown")
)
cells.append(
    cell("""
sys.path.insert(0, str(REPO_DIR / "train"))

from optimizer import build_optimizers, clip_grad_global_norm, Muon
from scheduler import WSDScheduler

# ── Muon.step(closure) compatibility patch ──────────────────────────────────
# Guard against a stale remote-repo clone where Muon.step() may not accept
# `closure=None`.  Accelerate's optimizer wrapper calls `step(closure)`, so
# the method *must* accept it; otherwise every step raises TypeError.
#
# The patch is idempotent (guarded by __patched flag) and safe even if the
# local file already has closure support — extra protection costs nothing.
if not getattr(Muon.step, '__patched', False):
    _orig_muon_step = Muon.step
    @torch.no_grad()
    def _patched_muon_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return _orig_muon_step(self)
    Muon.step = _patched_muon_step
    Muon.step.__patched = True
    del _orig_muon_step

# Build Muon + AdamW split (uses build_param_groups internally)
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

# WSD schedule: 2% warmup, 80% stable, 18% decay
warmup_steps = max(1, int(0.02 * MAX_STEPS))
stable_steps = int(0.80 * MAX_STEPS)
decay_steps  = MAX_STEPS - warmup_steps - stable_steps

# base_lrs: list[list[float]] — one list per optimizer, one float per param_group
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

# ── SECTION 9: Teacher Loading ───────────────────────────────────────────────
cells.append(cell("## Section 9: Gemma-4-12B Teacher (4-bit NF4, frozen)", "markdown"))
cells.append(
    cell("""
from transformers import AutoTokenizer as HFTok, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText as TeacherCls
except ImportError:
    from transformers import AutoModelForCausalLM as TeacherCls

import torch

# 4-bit NF4 quantization: Gemma-4-12B (≈24GB fp16) → ~6.5GB NF4
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

# Determine teacher device: if running single-process with multiple GPUs, offload teacher to GPU 1
if N_PROC == 1 and torch.cuda.device_count() > 1:
    TEACHER_DEVICE = torch.device("cuda:1")
else:
    TEACHER_DEVICE = DEVICE

if IS_MAIN:
    print(f"Loading teacher: {TEACHER_MODEL}")
    print("  Quantization: 4-bit NF4 (≈6.5 GB)")
    print(f"  Teacher Device: {TEACHER_DEVICE}")
    print(f"  Student Device: {DEVICE}")

# Free existing teacher model if it exists in memory to prevent OOM on re-run
if 'teacher' in globals():
    print("🧹 Freeing existing teacher model from memory...")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

from transformers import AutoConfig
config = AutoConfig.from_pretrained(TEACHER_MODEL, trust_remote_code=True, token=HF_TOKEN)

teacher = TeacherCls.from_pretrained(
    TEACHER_MODEL,
    config              = config,
    quantization_config = bnb_cfg,
    device_map          = {"": TEACHER_DEVICE},   # map teacher to designated device
    torch_dtype         = torch.bfloat16,
    trust_remote_code   = True,
    token               = HF_TOKEN,
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

# Teacher tokenizer (may differ from Lasmoid tokenizer — use for teacher logits only)
teacher_tok = HFTok.from_pretrained(TEACHER_MODEL, token=HF_TOKEN)

print(f"✅ Teacher loaded. Params: {sum(p.numel() for p in teacher.parameters())/1e9:.1f}B")
print(f"   Teacher vocab: {teacher_tok.vocab_size:,} | Lasmoid vocab: {VOCAB_SIZE:,}")
TEACHER_VOCAB = teacher_tok.vocab_size
""")
)

# ── SECTION 10: Distillation Loss ────────────────────────────────────────────
cells.append(
    cell(
        "## Section 10: SOTA Distillation Loss (Top-K Sparse KL + Temperature Annealing)",
        "markdown",
    )
)
cells.append(
    cell("""
import torch, torch.nn.functional as F

DISTILL_TOP_K = 4096     # sparse KL: only top-4096 tokens (32x less memory than full vocab)
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
    teacher_logits: torch.Tensor,   # [B, S, V_teacher_subset] — already on GPU/sliced
    temperature: float,
    top_k: int = 4096,
) -> torch.Tensor:
    \"\"\"
    Optimized Top-K Sparse KL loss.
    1. Select top-K teacher tokens by logit value in bfloat16.
    2. Softmax over top-K in float32.
    3. Gather student logits at same positions *before* scaling.
    4. Compute KL divergence.
    \"\"\"
    B, S, V_t = teacher_logits.shape
    V_s = student_logits.shape[-1]
    V   = min(V_t, V_s)

    # Slice teacher logits to match student vocab (just in case)
    t_logits_slice = teacher_logits[..., :V]

    # Find top-K indices using original dtype (bfloat16) to save memory/time
    k = min(top_k, V)
    tk_vals, tk_idx = t_logits_slice.topk(k, dim=-1)

    # Cast to float32 and scale for softmax
    tk_vals = tk_vals.float() / temperature
    p_teacher = F.softmax(tk_vals, dim=-1)

    # Gather student logits at teacher's top-K positions directly (before scaling)
    s_slice = student_logits if V == V_s else student_logits[..., :V]
    s_topk = s_slice.gather(-1, tk_idx)

    # Cast and scale student gathered logits
    s_topk = s_topk.float() / temperature
    log_q = F.log_softmax(s_topk, dim=-1)

    # KL divergence (sum over K, mean over B×S)
    kl = (p_teacher * (p_teacher.clamp(min=1e-8).log() - log_q)).sum(-1).mean()
    return kl * (temperature ** 2)

def compute_teacher_logits(teacher, input_ids: torch.Tensor, vocab_limit: int) -> torch.Tensor:
    \"\"\"Get Gemma-4 logits without gradient, sliced early to save memory.\"\"\"
    with torch.no_grad():
        out = teacher(input_ids=input_ids.to(TEACHER_DEVICE), use_cache=False)
        # Slice on-device to avoid large VRAM allocation/transfer
        logits = out.logits[..., :vocab_limit].to(DEVICE)
    return logits

print("✅ Distillation loss functions ready")
print(f"  Top-K sparse KL: K={DISTILL_TOP_K}")
print(f"  Temperature   : {T_START:.1f} → {T_END:.1f} (cosine anneal)")
print(f"  Alpha schedule: {ALPHA_WARMUP:.2f} → {ALPHA_PEAK:.2f} → {ALPHA_FINAL:.2f}")
""")
)

# ── SECTION 11: Checkpoint System ────────────────────────────────────────────
cells.append(
    cell("## Section 11: Crash-Safe Checkpoint System (every 50 steps)", "markdown")
)
cells.append(
    cell("""
import torch, json, time
from pathlib import Path
from safetensors.torch import save_file as safetensors_save

CURSOR_FILE = CKPT_DIR / "cursor.json"

def save_cursor(step: int, dataset_idx: int, epoch: int = 0):
    \"\"\"Persist dataset position so streaming resumes correctly after crash.\"\"\"
    cursor = {
        "step"       : step,
        "dataset_idx": dataset_idx,
        "epoch"      : epoch,
        "timestamp"  : time.time(),
        "dataset"    : "FineWeb-Edu+Cosmopedia",
    }
    with open(CURSOR_FILE, "w") as f:
        json.dump(cursor, f, indent=2)

def load_cursor() -> dict:
    if CURSOR_FILE.exists():
        with open(CURSOR_FILE) as f:
            return json.load(f)
    return {"step": 0, "dataset_idx": 0, "epoch": 0}

def save_checkpoint(
    step          : int,
    model,
    optimizers    : list,
    scheduler,
    loss_history  : list,
    expert_stats  : dict,
    concept_stats : dict,
    dataset_idx   : int = 0,
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

    # 4. RNG state (for reproducibility across restarts)
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
        "loss_history" : loss_history[-200:],
        "expert_stats" : expert_stats,
        "concept_stats": concept_stats,
        "timestamp"    : time.time(),
        "config"       : cfg,
    }
    with open(ckpt_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # 6. Dataset cursor
    save_cursor(step, dataset_idx)

    # 7. Symlink "latest"
    latest = CKPT_DIR / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt_path.name)

    print(f"  💾 Checkpoint saved: step {step} → {ckpt_path.name}")

def find_latest_checkpoint() -> Path:
    latest = CKPT_DIR / "latest"
    if latest.is_symlink() and latest.exists():
        return latest.resolve()
    # Fallback: scan for highest step_XXXXXX directory
    candidates = sorted(CKPT_DIR.glob("step_*"))
    if candidates:
        return candidates[-1]
    return None

def load_checkpoint(model, optimizers, scheduler, ckpt_path: Path):
    \"\"\"Load full checkpoint. Returns (step, loss_history, expert_stats, concept_stats).\"\"\"
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

    return meta["step"], meta.get("loss_history", []), meta.get("expert_stats", {}), meta.get("concept_stats", {})

print("✅ Checkpoint system ready")
""")
)

# ── SECTION 12: Expert Monitoring ────────────────────────────────────────────
cells.append(cell("## Section 12: Expert & Concept Monitoring", "markdown"))
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
        # [n_layers, n_experts] cumulative token counts
        self.counts = torch.zeros(self.n_layers, self.n_experts)
        self.steps  = 0

    def update(self, routing_maps: list):
        \"\"\"routing_maps: list[Tensor] from model forward — one per layer.\"\"\"
        for layer_idx, routing in enumerate(routing_maps):
            if routing is None or layer_idx >= self.n_layers:
                continue
            # routing shape: [B, S, n_activated] — expert indices selected
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
        freq  = self.counts / total            # [n_layers, n_experts]

        # Dead expert: < 1% of tokens ever
        dead_mask = freq < 0.01
        dead_count = dead_mask.sum().item()

        # Routing entropy per layer (higher = more balanced)
        ent = -(freq.clamp(min=1e-8) * freq.clamp(min=1e-8).log()).sum(-1)  # [n_layers]
        max_ent = math.log(self.n_experts)

        return {
            "dead_experts"    : int(dead_count),
            "dead_pct"        : 100 * dead_count / (self.n_layers * self.n_experts),
            "mean_entropy"    : ent.mean().item(),
            "max_entropy"     : max_ent,
            "entropy_ratio"   : (ent.mean() / max_ent).item(),
            "expert_freq"     : freq.mean(0).tolist(),    # mean across layers
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
        \"\"\"concept_indices: list[Tensor] from model forward — codebook assignments.\"\"\"
        for ci in concept_indices:
            if ci is None:
                continue
            try:
                idx = ci.detach().cpu().view(-1)
                for i in idx.tolist():
                    if 0 <= int(i) < self.num_concepts:
                        self.usage[int(i)] += 1
            except Exception:
                pass
        self.steps += 1

    def stats(self) -> dict:
        total = self.usage.sum().clamp(min=1)
        freq  = self.usage / total
        ent   = -(freq.clamp(min=1e-8) * freq.clamp(min=1e-8).log()).sum().item()
        max_ent = math.log(self.num_concepts)

        # Concept collapse: top-5 concepts take > 80% of usage
        top5_frac = freq.topk(min(5, self.num_concepts)).values.sum().item()
        collapsed = top5_frac > 0.80

        return {
            "entropy"         : ent,
            "max_entropy"     : max_ent,
            "entropy_ratio"   : ent / max_ent,
            "collapsed"       : collapsed,
            "top5_coverage"   : top5_frac,
            "dead_concepts"   : int((freq < 0.001).sum().item()),
        }

# Instantiate monitors
expert_monitor  = ExpertMonitor(n_experts=args.n_routed_experts, n_layers=args.n_layers)
concept_monitor = ConceptMonitor(num_concepts=args.num_concepts)

print(f"✅ Expert monitor  : {args.n_layers} layers × {args.n_routed_experts} experts")
print(f"✅ Concept monitor : {args.num_concepts} concept slots")
""")
)

# ── SECTION 13: OOM Recovery ─────────────────────────────────────────────────
cells.append(cell("## Section 13: OOM Recovery + Accelerate Wrap", "markdown"))
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
    Also catches ValueError (PyTorch symbolizer bug: stoi/storage)
    which can occur under gradient checkpointing in Kaggle/Colab.
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

# ── SECTION 14: Auto Resume ───────────────────────────────────────────────────
cells.append(cell("## Section 14: Auto Resume", "markdown"))
cells.append(
    cell("""
# ── Auto-resume from latest checkpoint ──────────────────────────────────────
loss_history   = []
expert_stats   = {}
concept_stats  = {}
dataset_cursor = load_cursor()

latest_ckpt = find_latest_checkpoint()
START_STEP  = 0

if latest_ckpt is not None and IS_MAIN:
    print(f"Found checkpoint: {latest_ckpt}")
    START_STEP, loss_history, expert_stats, concept_stats = load_checkpoint(
        accelerator.unwrap_model(model),
        [accelerator.unwrap_model(opt) if hasattr(opt, 'param_groups') else opt
         for opt in optimizers],
        scheduler,
        latest_ckpt,
    )
    print(f"✅ Resumed from step {START_STEP}")
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

# ── SECTION 15: Full Training Loop ───────────────────────────────────────────
cells.append(
    cell(
        "## Section 15: Training Loop (Multi-Loss, Distillation, Monitoring)",
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

# ── Build streaming data iterator ────────────────────────────────────────────
data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN)
dataset_idx = dataset_cursor.get("dataset_idx", 0)

# Fast-forward to last saved position (skip already-seen samples)
if dataset_idx > 0 and IS_MAIN:
    print(f"Fast-forwarding dataset to position {dataset_idx}...")
    for _ in range(dataset_idx):
        try:
            next(data_iter)
        except StopIteration:
            data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN)

# ── Main training loop ───────────────────────────────────────────────────────
model.train()
pbar = tqdm(range(START_STEP, MAX_STEPS), initial=START_STEP, total=MAX_STEPS,
            desc="Training", disable=not IS_MAIN)

accum_steps        = accelerator.gradient_accumulation_steps
tokens_per_step    = BATCH_SIZE * SEQ_LEN * accum_steps * N_PROC  # tokens consumed per real optimizer step
steps_since_ckpt   = 0
total_tokens_seen  = 0
global_step        = START_STEP

# Warm up GPU memory tracking by touching a small tensor (avoids cold-read bias)
if torch.cuda.is_available():
    _ = torch.cuda.memory_allocated() / 1024**3

for step in pbar:
    t0 = time.time()
    global_step = step
    _CHECKPOINT_ON_KILL.update({"step": step, "dataset_idx": dataset_idx})

    # LR schedule
    lr_mult = scheduler.step(step)

    # Gradient accumulation loop
    total_loss_accum = 0.0
    for micro in range(accum_steps):
        # Get next batch
        try:
            x, y = next(data_iter)
            dataset_idx += 1
        except StopIteration:
            # End of streaming epoch — restart
            data_iter = stream_packed(ds_train, tokenizer, SEQ_LEN)
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

                # ── Aggregate Lasmoid loss ──
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
                    curiosity_loss = unwrapped.last_curiosity_loss if args.use_curiosity_expert else None,
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

        # NaN/Inf gradient guard — prevents silent loss divergence.
        # If any param has NaN/Inf grad, zero everything and skip step.
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
            # Apply EMA bias updates for MoE routing (after each real step)
            accelerator.unwrap_model(model).apply_pending_bias_updates()
        else:
            # Log NaN for post-hoc analysis
            log_step({"step": step, "nan_gradient": True, "grad_norm": float('nan')})

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # ── Logging ─────────────────────────────────────────────────────────────
    loss_history.append(total_loss_accum)
    step_time = time.time() - t0
    total_tokens_seen += tokens_per_step

    if IS_MAIN:
        exp_s  = expert_monitor.stats()
        con_s  = concept_monitor.stats()
        tokens_per_sec = tokens_per_step / max(step_time, 1e-6)

        # GPU memory pressure (query once)
        if torch.cuda.is_available() and step % 10 == 0:
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _peak  = torch.cuda.max_memory_allocated() / 1024**3
            _free  = torch.cuda.get_device_properties(0).total_memory / 1e9 - _alloc
            _mem_str = f"{_alloc:.1f}G/{_free:.1f}G free"
        else:
            _mem_str = ""

        pbar.set_postfix({
            "loss"   : f"{total_loss_accum:.3f}",
            "kl"     : f"{kl_loss.item():.3f}",
            "tok/s"  : f"{tokens_per_sec:,.0f}",
            "mem"    : _mem_str or "",
            "ent"    : f"{exp_s['entropy_ratio']:.2f}",
            "dead_e" : exp_s['dead_experts'],
        })

        if step % 10 == 0:
            log_step({
                "step"           : step,
                "loss"           : total_loss_accum,
                "kl_loss"        : kl_loss.item(),
                "mtp_loss"       : mtp_loss.item() if mtp_loss else 0.0,
                "moe_aux"        : unwrapped.last_moe_loss.item(),
                "vq_loss"        : unwrapped.last_vq_loss.item(),
                "commit_loss"    : unwrapped.last_commit_loss.item(),
                "alpha"          : alpha,
                "temperature"    : T,
                "lr_mult"        : lr_mult,
                "grad_norm"      : grad_norm if accelerator.sync_gradients else 0.0,
                "step_time_s"    : step_time,
                "tokens_per_sec" : tokens_per_sec,
                "gpu_mem_gb"     : _alloc if torch.cuda.is_available() else 0,
                "expert_stats"   : exp_s,
                "concept_stats"  : con_s,
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
            )
            # Reset monitors after each checkpoint (fresh window)
            expert_monitor.reset()
            concept_monitor.reset()

print("\\n✅ Training complete!")
""")
)

# ── SECTION 16: Validation ───────────────────────────────────────────────────
cells.append(cell("## Section 16: Validation (Perplexity + Reasoning)", "markdown"))
cells.append(
    cell("""
import torch, math

model.eval()

# ── Perplexity on TinyStories val ────────────────────────────────────────────
val_path = REPO_DIR / "datasets" / "tinystories_val.bin"
if val_path.exists():
    import numpy as np
    val_data = np.frombuffer(val_path.read_bytes(), dtype=np.uint16).astype(np.int64)
    val_data = torch.from_numpy(val_data)

    N_VAL = min(1024, len(val_data) // SEQ_LEN)
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
# Test that <think> token triggers reasoning_steps loop
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

# ── SECTION 17: Training Report & Plots ──────────────────────────────────────
cells.append(
    cell(
        "## Section 17: Training Report (Loss Curves, Expert Stats, Concept Stats)",
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
dead_exps   = [l.get("expert_stats", {}).get("dead_experts", 0) for l in log_lines]
concept_ent = [l.get("concept_stats", {}).get("entropy_ratio", 0) for l in log_lines]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Lasmoid Training Report", fontsize=16, fontweight="bold")

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

# Concept memory entropy
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
    print(f"\\n{'='*45}")
    print(f"  Training Summary")
    print(f"{'='*45}")
    print(f"  Steps completed  : {steps[-1] if steps else 0}")
    print(f"  Final loss       : {losses[-1]:.4f}")
    print(f"  Best loss        : {min(losses):.4f}")
    print(f"  Final KL loss    : {kl_losses[-1]:.4f}")
    print(f"  Dead experts     : {dead_exps[-1] if dead_exps else '?'}")
    print(f"  Concept entropy  : {concept_ent[-1]:.3f}" if concept_ent else "")
    print(f"  Perplexity       : {ppl:.2f}" if ppl else "  Perplexity: N/A")
    print(f"{'='*45}")
""")
)

# ── SECTION 18: Kaggle Persistence ──────────────────────────────────────────
cells.append(
    cell(
        "## Section 18: Kaggle Dataset Export (Persistence Across Sessions)", "markdown"
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

    Upload with: kaggle datasets create -p {export_dir}
    Next session: Load from /kaggle/input/<dataset-slug>/latest/
    \"\"\"
    from safetensors.torch import save_file as sf_save

    pkg = export_dir / f"lasmoid_100m_step{step}"
    pkg.mkdir(parents=True, exist_ok=True)

    # Weights
    unwrapped = accelerator.unwrap_model(model)
    sf_save(
        {k: v.cpu() for k, v in unwrapped.state_dict().items()},
        str(pkg / "model.safetensors")
    )

    # Config
    shutil.copy(str(REPO_DIR / "configs" / "model" / "config_gemma4_100m.json"),
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
        "ablation_flags": ABLATION_FLAGS,
        "architecture"  : "Lasmoid-100M-Gemma4-Distilled",
        "vocab_size"    : VOCAB_SIZE,
    }
    with open(pkg / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Cursor (for resuming dataset position)
    if CURSOR_FILE.exists():
        shutil.copy(str(CURSOR_FILE), str(pkg / "cursor.json"))

    # Generate dataset-metadata.json for Kaggle upload
    kaggle_meta = {
        "title"      : f"Lasmoid-100M step{step}",
        "id"         : f"lasmoid-100m-step{step}",
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

# ── SECTION 19: Final Export ─────────────────────────────────────────────────
cells.append(
    cell("## Section 19: Final Export (SafeTensors + HuggingFace)", "markdown")
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
        str(REPO_DIR / "configs" / "model" / "config_gemma4_100m.json"),
        str(hf_dir / "config.json")
    )

    # 3. Tokenizer
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = REPO_DIR / fname
        if src.exists():
            shutil.copy(str(src), str(hf_dir / fname))
    shutil.copy(
        str(REPO_DIR / "configs" / "generation_config.json"),
        str(hf_dir / "generation_config.json")
    )

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
base_model: google/gemma-4-12B
---

# Lasmoid-100M (Gemma-4-12B Distilled)

**Architecture**: Hybrid Concept Transformer  
**Teacher**: Gemma-4-12B (4-bit NF4)  
**Parameters**: ~100M  
**Training**: {len(loss_history)} steps, Top-K Sparse KL distillation  

## Components
- Compressed Sparse Attention (CSA) with CIF compressor
- Mamba-2 SSD State Space Recurrence  
- Grey-Box MoE (6 routed + 1 shared experts)
- ElasticSparseConceptMemory (ESCM) with GVQ
- Manifold-Constrained Hyper-Connections (mHC)
- Multi-Token Prediction (MTP, t+1 and t+2)

## Ablation Flags Applied
{json.dumps(ABLATION_FLAGS, indent=2)}
\"\"\"
    (hf_dir / "README.md").write_text(model_card)

    # 5. (Optional) Push to Hub
    # from huggingface_hub import HfApi
    # api = HfApi()
    # api.upload_folder(folder_path=str(hf_dir), repo_id="Theory903/lasmoid-100m",
    #                   commit_message=f"Distillation step {len(loss_history)+START_STEP}")

    print(f"\\n✅ Model ready for upload at: {hf_dir}")
    print(f"   To push: Uncomment HfApi section above")
""")
)

# ── SECTION 20: Ablation Runner ──────────────────────────────────────────────
cells.append(
    cell("## Section 20: Ablation Runner (Auto-configure from results)", "markdown")
)
cells.append(
    cell("""
import json

# ── Ablation results from project (ablation_results.jsonl) ───────────────────
ABLATION_RESULTS = [
    {"variant": "baseline",     "perplexity": 54840.62, "verdict": "baseline"},
    {"variant": "domain_cortex","perplexity": 54366.49, "verdict": "KEEP ✅"},
    {"variant": "adaptive_k",   "perplexity": 54893.48, "verdict": "REMOVE ❌"},
    {"variant": "relational",   "perplexity": 60458.66, "verdict": "REMOVE ❌ (WORST)"},
    {"variant": "curiosity",    "perplexity": 55520.33, "verdict": "REMOVE ❌"},
    {"variant": "all_on",       "perplexity": 55372.53, "verdict": "REMOVE ❌ (worse than baseline)"},
]

print("Ablation Summary (from ablation_results.jsonl):")
print(f"{'Variant':<20} {'Perplexity':>12}  Verdict")
print("-" * 50)
for r in sorted(ABLATION_RESULTS, key=lambda x: x["perplexity"]):
    marker = "★" if r["variant"] == "domain_cortex" else " "
    print(f"{marker} {r['variant']:<19} {r['perplexity']:>12.0f}  {r['verdict']}")

print()
print("Applied flags for this run:")
for k, v in ABLATION_FLAGS.items():
    status = "✅ ENABLED" if v else "❌ DISABLED"
    print(f"  {k:<35}: {status}")

print()
print("⚡ Key insight: domain_cortex reduces perplexity by 474 points.")
print("   Relational adds 5618 points of perplexity — never enable during distillation.")
print()

# ── To run a fresh ablation, uncomment: ─────────────────────────────────────
# for variant_flags in [
#     {"use_domain_cortex": False},          # pure baseline
#     {"use_domain_cortex": True},           # our winner
# ]:
#     combined = {**ABLATION_FLAGS, **variant_flags}
#     # re-build model with combined flags and train for N_ABLATION_STEPS steps
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
out_path = OUT_DIR / "lasmoid_kaggle_v3.ipynb"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

size_kb = out_path.stat().st_size / 1024
print(f"✅ Notebook written: {out_path}")
print(f"   Size: {size_kb:.1f} KB | Cells: {len(cells)}")
