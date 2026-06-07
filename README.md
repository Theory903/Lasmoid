# Lasmoid — Hybrid Concept Transformer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org)

Lasmoid is a **research-grade, sovereign AI architecture** for low-VRAM cognitive operations. It combines Compressed Sparse Attention, Mamba-2 State Space recurrence, and Graph-based Concept Memory into a single unified model family ranging from **10M to 1B+ parameters**.

---

## 🏗️ Architecture at a Glance

```
Input Tokens
    │
    ▼
Token Embedding (vocab=129,286)
    │
    ├──▶ Read Replica (Encoder)
    │       └── ElasticSparseConceptMemory (ESCM)
    │               ├── Perceiver Pooling
    │               ├── Residual Vector Quantization (RVQ)
    │               └── Graph Vector Quantizer (GVQ)
    │
    └──▶ Write Master (Decoder × N layers)
            Each layer contains:
            ├── CSA: Compressed Sparse Attention (GQA + LoRA Q/O)
            │       └── CIF Compressor → dynamic KV compression
            ├── SSM: Mamba-2 State Space Recurrence (SSD scan)
            ├── MoE: Grey-Box Mixture of Experts
            │       ├── Top-K Renormalized Routing (K=2)
            │       ├── Shared Expert (always active)
            │       └── Dual FFN Branch (stable gradient baseline)
            └── mHC: Manifold-Constrained Hyper-Connections
                    └── Sinkhorn doubly-stochastic stream routing

    └──▶ Multi-Token Prediction (MTP)
            ├── t+1 Logits (primary head)
            └── t+2 Logits (auxiliary head)
```

### Key Innovations

| Component | What it does |
|-----------|-------------|
| **ESCM** (Graph VQ Memory) | Pools context into discrete codebook concepts via RVQ + differentiable graph message passing |
| **CIF Compressor** | Dynamic KV compression using Continuous Integrate-and-Fire boundary detection |
| **Mamba-2 SSM** | Parallel chunked SSD scan with learned ZOH decay; complements attention for long sequences |
| **Grey-Box MoE** | DeepSeek-V4-style routing: top-K renormalized + shared expert + dense dual FFN baseline |
| **mHC Streams** | Replaces residual addition with Sinkhorn-projected doubly-stochastic stream mixing |
| **MTP** | Predicts t+1 and t+2 in one forward pass, improving generation quality and throughput |
| **Muon Optimizer** | 5th-order Newton-Schulz iteration for 2D weight matrices, enforcing near-orthogonality |

---

## 📁 Repository Structure

```
Lasmoid/
│
├── README.md                    ← You are here
├── LICENSE
│
├── configs/                     ← All configuration files
│   ├── generation_config.json
│   ├── tokenizer/
│   │   └── tokenizer_config.json
│   └── model/
│       ├── config_10m.json      ← 10M  (dim=256,  layers=6)
│       ├── config_100m.json     ← 100M (dim=384,  layers=12)  ← default Kaggle target
│       ├── config_300m.json     ← 300M (dim=640,  layers=18)
│       ├── config_500m.json     ← 500M (dim=768,  layers=24)
│       ├── config_1b.json       ← 1B   (dim=1024, layers=28)
│       ├── config_1b_2m.json    ← 1B distilled variant
│       ├── config_gemma4_100m.json  ← 100M tuned for Gemma-4 distillation
│       └── config_gemma4_1b.json    ← 1B  tuned for Gemma-4 distillation
│
├── inference/                   ← Core model code (load this for training too)
│   ├── model.py                 ← Entry point: Lasmoid class + ModelArgs
│   ├── lasmoid.py               ← Main model graph: encoder → blocks → MTP
│   ├── attention.py             ← CSA: Compressed Sparse Attention (GQA + LoRA)
│   ├── ssm.py                   ← Mamba-2 SSD State Space Recurrence
│   ├── moe.py                   ← Grey-Box MoE + routing
│   ├── mhc.py                   ← Manifold-Constrained Hyper-Connections
│   ├── concept_memory.py        ← ElasticSparseConceptMemory (ESCM)
│   ├── vq.py                    ← Vector Quantizer + Graph VQ
│   ├── compressor.py            ← CIF context compressor
│   ├── mtp.py                   ← Multi-Token Prediction heads
│   ├── kernel.py                ← Custom CUDA / triton kernels
│   ├── kv_cache.py              ← KV cache management
│   ├── generate.py              ← Autoregressive generation
│   ├── sampler.py               ← Sampling strategies (top-p, top-k, beam)
│   ├── loss.py                  ← All loss functions (MoE, VQ, MTP, etc.)
│   ├── config.py                ← ModelArgs dataclass definition
│   └── requirements.txt
│
├── train/                       ← Training scripts
│   ├── train.py                 ← Main pretraining loop
│   ├── optimizer.py             ← Muon + AdamW dual optimizer
│   ├── scheduler.py             ← WSD (Warmup-Stable-Decay) schedule
│   ├── pretrain.py              ← Pretraining entry point
│   ├── prepare_data.py          ← Data tokenization + packing
│   ├── evaluate.py              ← Evaluation harness
│   ├── grpo_stability.py        ← GRPO reinforcement learning
│   ├── long_context_finetune.py ← Long-context fine-tuning
│   └── mopd.py                  ← Multi-objective preference distillation
│
├── scripts/                     ← Cloud + experiment scripts
│   ├── distill_gemma4.py        ← Gemma-4-12B teacher distillation (Azure)
│   ├── gpu_train.py             ← GPU-optimized training entry
│   ├── gpu_generate.py          ← GPU generation script
│   ├── gpu_benchmark.py         ← Throughput benchmarking
│   ├── azure_auto_deploy.sh     ← One-command Azure VM deploy
│   └── azure_setup.sh           ← Azure environment bootstrap
│
├── notebooks/                   ← Jupyter notebooks
│   ├── distillation/
│   │   └── lasmoid_kaggle_v3.ipynb                   ← 🔥 Gemma-4 KD (Kaggle T4 x2)
│   ├── training/
│   │   ├── lasmoid_kaggle_train_100m.ipynb            ← 100M pretraining (Kaggle)
│   │   ├── lasmoid_kaggle_train_10m.ipynb             ← 10M pretraining (Kaggle)
│   │   └── train_lasmoid_kaggle.ipynb                 ← General Kaggle launcher
│   └── validation/
│       └── lasmoid_10m_validation.ipynb               ← Model validation suite
│
├── datasets/                    ← Local datasets and tokenized binaries
│   ├── input.txt                ← Raw text corpus
│   ├── test_prompts.txt
│   ├── gsm8k_train.bin / gsm8k_val.bin
│   └── tinystories_train.bin / tinystories_val.bin
│
├── docs/                        ← Documentation
│   ├── architecture/
│   │   ├── ARCHITECTURE.md      ← Mathematical deep-dive of all components
│   │   └── NEXTGEN_DESIGN.md    ← Next-gen design proposals
│   ├── training/
│   │   └── TRAINING.md          ← Training guide + hyperparameter reference
│   └── deployment/
│       ├── DEPLOYMENT.md        ← Cloud deployment guide
│       └── MIGRATIONS.md        ← Breaking changes / migration notes
│
├── encoding/                    ← Custom BPE tokenizer
│   └── encoding_lasmoid.py
│
├── tokenizer.json               ← BPE tokenizer (129,286 tokens)
└── checkpoints/                 ← Saved model weights
    ├── current/
    └── pretrain/
```

---

## 🚀 Quick Start

### Training on Kaggle (Free T4 x2)

Open `notebooks/distillation/lasmoid_kaggle_v3.ipynb` on Kaggle.

Requirements:
1. Enable **T4 x2 GPU** in Notebook Settings
2. Add `HF_TOKEN` in Kaggle → Add-ons → Secrets
3. Accept [Gemma-4-12B terms](https://huggingface.co/google/gemma-4-12B)

### Local Training

```bash
# Install deps
pip install -r inference/requirements.txt

# Prepare data
python train/prepare_data.py --input datasets/input.txt --output datasets/

# Train 100M from scratch (pretraining)
python train/train.py \
    --config configs/model/config_100m.json \
    --data   datasets/input.txt \
    --out    checkpoints/

# Distill from Gemma-4 (requires HF token)
export HF_TOKEN=hf_...
python scripts/distill_gemma4.py \
    --student-config configs/model/config_100m.json
```

### Inference

```bash
python inference/generate.py \
    --checkpoint checkpoints/current/ \
    --config     configs/model/config_100m.json \
    --prompt     "The laws of thermodynamics state that"
```

---

## 📊 Model Scale Reference

| Config | Params | dim | Layers | Heads | MoE Experts | VRAM (bf16) | Target Platform |
|--------|--------|-----|--------|-------|-------------|-------------|-----------------|
| `config_10m`   | ~10M  | 256 | 6  | 4  | 4+1 | ~0.2 GB | CPU / any |
| `config_100m`  | ~100M | 384 | 12 | 6  | 6+1 | ~1.0 GB | T4 (free) |
| `config_300m`  | ~300M | 640 | 18 | 10 | 6+1 | ~2.5 GB | T4 / A10  |
| `config_500m`  | ~500M | 768 | 24 | 12 | 8+1 | ~4.0 GB | A10 / A100 |
| `config_1b`    | ~1B   |1024 | 28 | 16 | 8+1 | ~8.0 GB | A100 (40GB) |

---

## 🎓 Distillation

Lasmoid uses **teacher-student knowledge distillation** from Gemma-4-12B:

```
Gemma-4-12B (Teacher, frozen 4-bit NF4)
        │  soft logits (top-4096 sparse KL)
        ▼
Lasmoid-100M (Student, trainable bf16)

Loss = (1-α)·CE(student, hard_labels) + α·T²·SparseKL(teacher_topK || student_topK)

α schedule: warmup 0.5→0.9, then cosine decay 0.9→0.15
T schedule: cosine anneal 4.0→1.0
```

See `docs/architecture/ARCHITECTURE.md` for mathematical details.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 🔗 Related

- [ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) — Full mathematical specification
- [TRAINING.md](docs/training/TRAINING.md) — Training guide
- [DEPLOYMENT.md](docs/deployment/DEPLOYMENT.md) — Cloud deployment
