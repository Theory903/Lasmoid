# Lasmoid Training Guide

Complete reference for pretraining, fine-tuning, and distillation.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Pretraining from Scratch](#pretraining-from-scratch)
4. [Knowledge Distillation (Gemma-4-12B)](#knowledge-distillation)
5. [Fine-tuning & GRPO](#fine-tuning--grpo)
6. [Hyperparameter Reference](#hyperparameter-reference)
7. [Checkpointing](#checkpointing)
8. [Monitoring](#monitoring)

---

## Prerequisites

```bash
pip install torch>=2.5.0 transformers>=4.51.0 datasets bitsandbytes accelerate safetensors tqdm
```

GPU requirements by model size:

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| 10M   | 2 GB     | CPU / T4    |
| 100M  | 4 GB     | T4 (16GB)   |
| 300M  | 8 GB     | A10 (24GB)  |
| 500M  | 16 GB    | A100 (40GB) |
| 1B    | 32 GB    | A100 (80GB) |

---

## Data Preparation

```bash
# Tokenize raw text into packed binary
python train/prepare_data.py \
    --input  datasets/input.txt \
    --output datasets/ \
    --seq-len 1024

# For streaming datasets (FineWeb-Edu)
# → handled automatically by the distillation notebook
```

**Supported dataset formats:**
- Raw `.txt` files → binary `.bin` packs
- HuggingFace streaming (`HuggingFaceFW/fineweb-edu-score-2`)
- GSM8K (math reasoning): pre-packed in `datasets/gsm8k_*.bin`
- TinyStories: pre-packed in `datasets/tinystories_*.bin`

---

## Pretraining from Scratch

### Kaggle (Free T4 x2)
Open `notebooks/training/lasmoid_kaggle_train_100m.ipynb`

### Local

```bash
python train/train.py \
    --config    configs/model/config_100m.json \
    --data      datasets/input.txt \
    --out       checkpoints/ \
    --steps     10000 \
    --batch     4 \
    --seq-len   512 \
    --lr-muon   2e-3 \
    --lr-adamw  3e-4
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | Path to model config JSON |
| `--data` | required | Training data path |
| `--steps` | 10000 | Total training steps |
| `--batch` | 4 | Micro-batch size |
| `--grad-accum` | 8 | Gradient accumulation steps |
| `--seq-len` | 512 | Sequence length |
| `--lr-muon` | 2e-3 | Muon LR (2D weights) |
| `--lr-adamw` | 3e-4 | AdamW LR (embeddings, norms) |
| `--warmup` | 2% | Warmup fraction |
| `--decay` | 18% | LR decay fraction |

---

## Knowledge Distillation

**Teacher**: Gemma-4-12B (12B params, loaded in 4-bit NF4 ≈ 6.5 GB)  
**Student**: Lasmoid-100M (100M params, bf16 ≈ 1 GB)

### 2025 SOTA Techniques Used

| Technique | Setting | Why |
|-----------|---------|-----|
| **Top-K Sparse KL** | K=4096 | 32x less logit memory vs full vocab KL |
| **Temperature Annealing** | T: 4.0→1.0 | Rich soft labels early, sharp predictions late |
| **Peaked Alpha Schedule** | 0.5→0.9→0.15 | Trust teacher more as training stabilizes |

### Loss Function

```
L = (1-α)·CE(student, hard_labels) + α·T²·SparseKL(teacher_topK ‖ student_topK)

α schedule: linear warmup 0.50→0.90 (0-5% of steps)
            cosine decay  0.90→0.15 (5-100% of steps)

T schedule: cosine anneal 4.0→1.0 (captures more signal early)
```

### Run on Kaggle (Recommended — Free T4 x2)

1. Go to `notebooks/distillation/lasmoid_kaggle_v3.ipynb`
2. Import to Kaggle, set **T4 x2** accelerator
3. Add `HF_TOKEN` secret
4. Run All

### Run Locally (Azure A100)

```bash
export HF_TOKEN=hf_...
python scripts/distill_gemma4.py \
    --student-config configs/model/config_100m.json \
    --steps          5000 \
    --batch          4 \
    --top-k          4096 \
    --t-start        4.0 \
    --t-end          1.0
```

---

## Fine-tuning & GRPO

### Supervised Fine-tuning

```bash
python train/train.py \
    --config     configs/model/config_100m.json \
    --checkpoint checkpoints/current/ \
    --data       datasets/gsm8k_train.bin \
    --steps      2000 \
    --lr-muon    5e-4 \
    --lr-adamw   1e-4
```

### GRPO (Reinforcement Learning)

```bash
python train/grpo_stability.py \
    --checkpoint checkpoints/current/ \
    --config     configs/model/config_100m.json \
    --reward-fn  math  \
    --steps      1000
```

GRPO reward signals supported:
- `math` — GSM8K format (checks numeric answer)
- `format` — `<think>...</think>` structure
- `length` — penalizes responses > max_len

---

## Hyperparameter Reference

### Optimizer (Dual: Muon + AdamW)

The Muon optimizer applies to all 2D weight matrices (linear layers).
AdamW applies to embeddings, norms, biases, and 1D parameters.

```
Muon (2D weights):
  - lr: 2e-3
  - momentum: 0.95
  - weight_decay: 0.1
  - ns_steps: 5  (Newton-Schulz iterations)

AdamW (1D + embeddings):
  - lr: 3e-4
  - betas: (0.9, 0.95)
  - weight_decay: 0.1
  - eps: 1e-8
```

### WSD Learning Rate Schedule

```
Step 0 → warmup_steps:  linear ramp 0 → peak_lr
Step warmup → stable:   constant peak_lr
Step stable → end:      cosine decay peak_lr → peak_lr × min_lr_ratio

Default splits: warmup=2%, stable=80%, decay=18%
min_lr_ratio: 0.1
```

### Loss Coefficients (from config files)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `router_z_loss_coeff` | 0.001 | MoE router z-loss (prevents collapse) |
| `moe_load_balance_coeff` | 0.01 | Load balance across experts |
| `hcm_commit_loss_coeff` | 0.25 | VQ commitment loss weight |
| `token_concept_loss_coeff` | 0.05 | Concept-token alignment |
| `predictive_coding_coeff` | 0.01 | SSM predictive coding |
| `moe_router_entropy_coeff` | 0.001 | Router entropy regularization |

---

## Checkpointing

Checkpoints save to `checkpoints/` by default:

```
checkpoints/
├── current/              ← Latest checkpoint (auto-updated)
│   └── lasmoid_final.pt
├── pretrain/             ← Pretraining snapshots
│   └── step_05000.pt
└── extension/            ← Context extension checkpoints
    └── lasmoid_stage_64.pt
```

**Checkpoint format:**
```python
{
    "step":             int,
    "model_state_dict": OrderedDict,
    "optimizer_states": [muon_state, adamw_state],
    "loss_history":     list[float],
    "config":           dict,          # full ModelArgs dict
    "teacher":          str,           # teacher model name (if distilled)
}
```

**Loading a checkpoint:**
```python
from inference.model import Lasmoid, ModelArgs
import torch, json

cfg  = json.load(open("configs/model/config_100m.json"))
args = ModelArgs(**cfg)
model = Lasmoid(args)

ckpt = torch.load("checkpoints/current/lasmoid_final.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
```

---

## Monitoring

During training, the loop prints:

```
Step  1000/5000 | loss=5.891 | ce=4.947 | kl=0.812 | α=0.74 | T=2.91 | GPU0=7.1GB | GPU1=1.8GB | ETA=5.5h
```

| Metric | Meaning | Healthy range |
|--------|---------|---------------|
| `loss` | Total training loss | decreasing |
| `ce` | Cross-entropy (hard labels) | 10→3 over training |
| `kl` | KL divergence from teacher | 0.3–1.5 |
| `α` | Teacher trust weight | 0.9→0.15 |
| `T` | Distillation temperature | 4.0→1.0 |

**Warning signs:**
- `kl = 0.000` → teacher logits not reaching student
- `loss > 13 after step 100` → tokenizer mismatch
- `GPU OOM` → reduce `BATCH_SIZE` or `SEQ_LEN`
