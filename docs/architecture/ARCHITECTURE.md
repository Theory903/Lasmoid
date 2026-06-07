# Lasmoid Architecture & Technical Specifications

This document provides a comprehensive mathematical and structural specification of the **Lasmoid** model. Lasmoid is a hybrid sequence-modeling architecture combining **Compressed Sparse Attention (CSA/HCA)**, **parallel State Space Recurrence (Mamba-2/SSD)**, and **Manifold-Constrained Hyper-Connections (mHC)** with a **Grey-Box Mixture of Experts (MoE)** routing framework.

---

## 1. High-Level System Architecture

Lasmoid operates under a **Read Replica (Encoder) / Write Master (Decoder)** CQRS paradigm:
1. **Read Replica (Encoder)**: Processes the incoming context using a dense representation layer and pools it into a set of discrete, vector-quantized concepts via the **ElasticSparseConceptMemory (ESCM)**.
2. **Write Master (Decoder)**: Employs interleaved hybrid transformer/SSM blocks containing parallel attention, Mamba recurrence, and manifold hyper-connections.
3. **Multi-Token Prediction (MTP)**: Predicts two tokens at each step ($t+1$ and $t+2$) by fusing embedding streams with residual hidden states.

```mermaid
graph TD
    Input([Input Tokens]) --> Embed[Token Embedding]
    
    subgraph Read Replica (Encoder)
        Embed --> EncAttn[Encoder Attention]
        EncAttn --> ESCM[Elastic Sparse Concept Memory]
        ESCM --> ConceptDB[(Concept Database)]
        ESCM --> MemoryState[(Memory State)]
    end
    
    subgraph Write Master (Decoder Layers)
        Embed --> DecLayers[Lasmoid Blocks 1..N]
        ConceptDB -.-> DecLayers
        MemoryState -.-> DecLayers
        DecLayers --> MHC[mHC Streams Router]
    end
    
    MHC --> OutputHead[Output Projector]
    OutputHead --> Logits1[t+1 Logits]
    
    subgraph Multi-Token Prediction (MTP)
        MHC --> MTP[MTP Block]
        Logits1 -.-> MTP
        MTP --> Logits2[t+2 Logits]
    end
```

---

## 2. Mathematical Deep-Dive of Components

### A. RMSNorm (Root Mean Square Normalization)
To enforce training stability and scale activations without shifting the mean, Lasmoid uses RMSNorm:
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \mathbf{\gamma}$$
Where $\mathbf{\gamma}$ is a learnable scaling parameter initialized to 1.

---

### B. YaRN Rotary Position Embeddings (RoPE)
YaRN (Yet another RoPE extensioN) dynamically scales the rotary frequency dimensions to support context window extension without performance degradation. For a position index $t$ and dimension frequency index $i$:
$$\mathbf{q}_t = \mathbf{W}_q \mathbf{x}_t, \quad \mathbf{k}_t = \mathbf{W}_k \mathbf{x}_t$$
$$\mathbf{q}_t^{\text{rot}} = \mathbf{R}_{\Theta, t}^{\text{YaRN}} \mathbf{q}_t, \quad \mathbf{k}_t^{\text{rot}} = \mathbf{R}_{\Theta, t}^{\text{YaRN}} \mathbf{k}_t$$
Where $\mathbf{R}_{\Theta, t}^{\text{YaRN}}$ dynamically scales the base frequency $\theta_i = b^{-2i/d}$ using interpolation bounds:
$$\theta_i^{\text{YaRN}} = \theta_i \cdot r(i)$$
The ramp function $r(i)$ smoothly transitions from high-frequency (non-interpolated) to low-frequency (fully interpolated) dimensions:
$$r(i) = (1 - \lambda(i)) \cdot \frac{1}{\text{factor}} + \lambda(i)$$

---

### C. ElasticSparseConceptMemory (ESCM)
The ESCM pools raw sequence representations into a codebook centroid graph.
1. **Perceiver Pooling**: Multi-head attention queries pool the hidden states $\mathbf{H}_{enc}$ across episodic, semantic, and global queries:
   $$\mathbf{Q}_{pool} = [\mathbf{Q}_{epi}; \mathbf{Q}_{sem}; \mathbf{Q}_{glo}]$$
   $$\mathbf{H}_{pooled} = \text{Attention}(\mathbf{Q}_{pool}, \mathbf{H}_{enc}, \mathbf{H}_{enc})$$
2. **Residual Vector Quantization (RVQ)**: Projects pooled representations onto codebook centroids and minimizes the commitment objective:
   $$\mathbf{z}_q = \mathbf{e}_k \quad \text{where } k = \arg\min_j \lVert \mathbf{H}_{pooled} - \mathbf{e}_j \rVert_2$$
   $$L_{vq} = \lVert \text{sg}[\mathbf{H}_{pooled}] - \mathbf{z}_q \rVert_2^2 + \beta \lVert \mathbf{H}_{pooled} - \text{sg}[\mathbf{z}_q] \rVert_2^2$$
3. **Graph Vector Quantizer (GVQ)**: Fuses codebook embeddings with a differentiable directed graph message-passing step over concept adjacencies:
   $$\mathbf{E}_{graph} = \mathbf{E} + \text{softmax}(\mathbf{A}) \mathbf{E} \mathbf{W}_g$$

---

### D. Continuous Integrate-and-Fire (CIF) Compressor
The `Compressor` replaces standard static context compression with a dynamic **Continuous Integrate-and-Fire (CIF)** semantic boundary loop:
1. **Event Boundary Detection**: A linear projection determines boundary probability scores $\alpha_t$:
   $$\alpha_t = \text{softplus}(\mathbf{w}_{\text{event}} \mathbf{h}_t + b_{\text{event}})$$
2. **Sequential Accumulator**: Gated KV pairs and boundary scores are accumulated. When $\sum \alpha_t \ge 1.0$, a semantic event is "fired" and written to the cache:
   $$\mathbf{kv}_t = \mathbf{kv}_t \cdot \alpha_t, \quad \mathbf{g}_t = \mathbf{g}_t \cdot \alpha_t$$
   $$\mathbf{x}_{\text{accum\_kv}} \leftarrow \mathbf{x}_{\text{accum\_kv}} + (1 - p_{\text{accum}}) \cdot \mathbf{kv}_t$$
   $$\mathbf{x}_{\text{accum\_gate}} \leftarrow \mathbf{x}_{\text{accum\_gate}} + (1 - p_{\text{accum}}) \cdot \mathbf{g}_t$$
   $$\mathbf{kv}_{\text{fired}} = \frac{\mathbf{x}_{\text{accum\_kv}}}{\mathbf{x}_{\text{accum\_gate}} + \epsilon}$$
   The remainder $p_{\text{accum}} + \alpha_t - 1.0$ is carried over to the next step.
3. **Autoregressive Step**: In generation mode, boundaries are computed step-by-step. If a fire threshold is crossed, the accumulated representation is committed to the cache.

---

### E. Upgraded State Space Recurrence (SSM / Mamba-2)
The StateSpaceRecurrence module implements a parallel chunked scan using a learnable decay parameter $\mathbf{A}$:
1. **Discretization**: Time step parameter $\Delta_t$ (dt) is softplus-mapped and clamped to $[\Delta_{min}, \Delta_{max}]$ to ensure numerical stability:
   $$\Delta_t = \text{clamp}(\text{softplus}(\delta_t + \delta_{\text{bias}}), \Delta_{min}, \Delta_{max})$$
2. **Learned Decay**: The discrete decay matrix $\overline{\mathbf{A}}_t$ is computed using Zero-Order Hold (ZOH):
   $$\overline{\mathbf{A}}_t = \exp(\Delta_t \otimes \mathbf{A})$$
3. **Input Scaling**: The input $\mathbf{v}$ is correctly scaled by $\Delta_t$ ($\overline{\mathbf{B}}_t v_t = \Delta_t B_t v_t$):
   $$\mathbf{v}_{\text{scaled}} = \mathbf{v}_t \cdot \Delta_t$$
   $$\mathbf{h}_t = \overline{\mathbf{A}}_t \odot \mathbf{h}_{t-1} + \mathbf{v}_{\text{scaled}} \otimes \mathbf{B}_t$$
   $$\mathbf{y}_t = \sum (\mathbf{h}_t \odot \mathbf{C}_t)$$
   During training, a chunked parallel scan (`ssm_chunk_scan`) is executed to parallelize this recurrence across the sequence dimension.

---

### F. Grey-Box Mixture of Experts (MoE) & Gate
Lasmoid utilizes a **DeepSeek-V4 inspired MoE** layout to route tokens to specialized networks:
1. **Top-K Renormalized Routing**: Gating scores are projected via a router weight matrix $\mathbf{W}_r$:
   $$\mathbf{s} = \text{softplus}(\mathbf{W}_r \mathbf{x} + \mathbf{b}_{\text{bias}}).^{0.5}$$
   Renormalization is performed over the selected top-$k$ experts:
   $$\mathbf{w}_{\text{routed}} = \text{Renormalize}(\mathbf{s}_{\text{top-k}})$$
2. **Dual FFN Branch (Gemma-4 mlp2)**: To guarantee stable gradients, a compact **dense FFN** runs in parallel with the routed MoE, ensuring every token has a baseline gradient pathway:
   $$\mathbf{y} = \sum_{i=1}^k w_i \cdot \text{Expert}_i(\mathbf{x}) + \mathbf{w}_{\text{shared}} \cdot \text{SharedExpert}(\mathbf{x}) + \text{DenseFFN}(\text{RMSNorm}(\mathbf{x}))$$
3. **Smooth Bias updates**: Gating balance is maintained during training using an EMA bias routing fraction:
   $$\mathbf{b}_{\text{bias}} \leftarrow \mathbf{b}_{\text{bias}} + \eta_{\text{ema}} \cdot (f_{\text{target}} - f_{\text{actual}})$$

---

### G. Manifold-Constrained Hyper-Connections (mHC)
Standard residual additions ($\mathbf{x} + \text{Layer}(\mathbf{x})$) are replaced by doubly stochastic projections across $M$ parallel residual streams:
1. **Birkhoff Polytope Sinkhorn Projection**: The log-domain connection weights $B_{raw}$ are projected onto the set of doubly stochastic matrices:
   $$\mathbf{M} = \exp(\mathbf{B}_{\text{raw}})$$
   Iteratively normalized across columns and rows for 20 steps:
   $$\mathbf{M} \leftarrow \text{ColNorm}(\mathbf{M}), \quad \mathbf{M} \leftarrow \text{RowNorm}(\mathbf{M})$$
   Yielding a matrix $\mathbf{B}_l$ where each row and column sums to exactly $1.0$.
2. **Stream Mixing**:
   $$\mathbf{x}_{\text{out}} = \mathbf{A}_l \odot \mathbf{x} + \mathbf{B}_l \mathbf{x} + \mathbf{C}_l \odot \text{Layer}(\mathbf{x})$$
   Where $\mathbf{A}_l, \mathbf{C}_l$ are Sigmoid-gated scaling matrices.

---

### H. Multi-Token Prediction (MTP)
To improve performance and speed up decoding, Lasmoid predicts both $t+1$ and $t+2$ tokens:
1. **Stream Fusion**:
   $$\mathbf{h}_{\text{fused}} = \text{RMSNorm}(\mathbf{W}_e \mathbf{e}_{t+1} + \mathbf{W}_h \mathbf{h}_t)$$
2. **MTP Block**: Processes $\mathbf{h}_{\text{fused}}$ using a parallel decoder block and projects it to vocabulary logits:
   $$\mathbf{y}_{t+2} = \text{Linear}(\mathbf{h}_{\text{fused}})$$

---

## 3. Training & RL Pipelines

### A. The Muon Optimizer (Newton-Schulz Iteration)
For 2D parameters (linear projection weights), Lasmoid uses the **Muon** optimizer to enforce weight orthogonality:
1. **Momentum Scaling**:
   $$\mathbf{X}_0 = \frac{\mathbf{G}_{\text{momentum}}}{\lVert \mathbf{G}_{\text{momentum}} \rVert_F + \epsilon}$$
2. **Newton-Schulz 5th-Order Iteration**:
   For $k \in [0, 4]$:
   $$\mathbf{A} = \mathbf{X}_k \mathbf{X}_k^T, \quad \mathbf{B} = \mathbf{A} \mathbf{X}_k$$
   $$\mathbf{X}_{k+1} = 3.4445 \mathbf{X}_k - 4.7750 \mathbf{B} + 2.0315 \mathbf{A} \mathbf{B}$$
3. **Weight Update**:
   $$\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \mathbf{X}_{\text{final}}$$

This constraints weights to be orthogonal, stabilizing intermediate layer scaling and accelerating training.

---

### B. Group Relative Policy Optimization (GRPO)
Lasmoid's alignment phase uses **GRPO** to align outputs with logical and formatting constraints without a separate critic model:
1. **Group Rollouts**: For each prompt, we generate $G$ candidate outputs.
2. **Reward Function**: Grades the response using:
   - **Reasoning Structure**: Formatting matches `<think>...</think>` and `<Parallel>...</Parallel>`.
   - **Cognitive OS Formatting**: Proper JSON tool payloads and escape constraints.
   - **Rambling Penalty**: Penalizes completions exceeding maximum length bounds.
3. **Advantage Objective**:
   $$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
   $$L(\theta) = - \frac{1}{G} \sum_{i=1}^G \left( \min\left( r_i(\theta) A_i, \, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta \parallel \pi_{\text{ref}}) \right)$$
   Where $r_i(\theta) = \exp(\log \pi_\theta(\mathbf{o}_i \mid \mathbf{q}) - \log \pi_{\text{ref}}(\mathbf{o}_i \mid \mathbf{q}))$.
