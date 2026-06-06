# Lasmoid: Hybrid Concept Transformer

Lasmoid is a sovereign AI architecture designed for low-VRAM cognitive operations. It integrates **Multi-Head Latent Attention (MLA)**, **Manifold-Constrained Hyper-Connections (mHC)**, and **Multi-Token Prediction (MTP)** with advanced Graph-based Concept Memory and Grey-box Mixture of Experts routing.

Use `train/train.py` to write fresh checkpoints into `checkpoints/current/` by default. For inference, point `generate.py` at that directory or another explicitly current run directory; the loader now rejects stale incompatible checkpoints unless `--allow-partial-load` is passed.

## Key Innovations

1. **Graph-based Vector Quantizer**: Combines vector quantization with differentiable Directed Graph message passing over raw concepts.
2. **Causal Dynamic Concept Memory**: Step-by-step semantic accumulator that prevents forward temporal leakage in reasoning.
3. **Hybrid Concept Attention**: Blends low-rank local self-attention with sparse Top-K concept memory cross-attention.
4. **DeepSeek-V4 Grey-Box MoE**: Fuses attention concept routing weights with hidden states to guide load-balanced expert selection.
5. **Manifold-Constrained Hyper-Connections**: Distributes representation updates across parallel residual streams using doubly stochastic Sinkhorn routing.

## Directory Structure

```
Lasmoid/
├── config.json
├── generation_config.json
├── tokenizer_config.json
├── tokenizer.json
├── inference/
│   ├── model.py
│   ├── generate.py
│   ├── kernel.py
│   ├── convert.py
│   └── requirements.txt
├── encoding/
│   ├── encoding_lasmoid.py
│   └── test_encoding_lasmoid.py
└── train/
    └── train.py
```
