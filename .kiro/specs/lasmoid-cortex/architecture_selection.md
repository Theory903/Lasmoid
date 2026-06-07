# Architecture Selection: lasmoid-cortex

## Recommended Architecture: Hierarchical Sparse Cortex + Tool Sidecar (Candidate B)

### Rationale
Candidate B layers a **hierarchical sparse router** (domain columns over the existing experts) and a **tool/EDA sidecar** onto the current block, keeping the model core and the agentic runtime as separate components communicating only through the token stream and a typed result envelope. It has the lowest cross-cutting-requirement share (routing concerns stay inside the gate; tool concerns stay in the sidecar) and zero synchronous cycles. The trade-off: domain semantics are not isolated in their own module (they live inside the expert `Gate`), so a future move to per-domain *representation tracks* (Candidate C) would touch the gate again. If multi-track scientific representation (AlphaFold-style) becomes the priority, Candidate C is preferable.

### Components
| Component | Owned State | Responsibility |
|-----------|-------------|----------------|
| DomainCortexRouter | domain router weights, cross-domain affinity logits | Select top-k domain columns per token; emit expert eligibility mask + load-balance aux |
| Gate (extended) | expert router weights, EMA balance bias | Expert top-k within active columns; compose cortex mask with group routing |
| DeepSeekMoE (host) | experts, shared/dense branches, per-expert scale | Execute experts; aggregate cortex + MoE aux losses |
| ToolRegistry | tool schemas, handlers | Register/validate/dispatch tools (ML + EDA) |
| EDAToolset | none (pure functions over data refs) | Derive structured findings from raw data |
| ReasoningController | step budget, best score, transcript | Drive propose→verify→refine over generate + tools |
| EpisodicMemory (reuse) | episode centroids, KV tiers | Cluster long streams; bound working memory |

### Information Flow
| From \ To | Cortex | Gate | MoE | ToolRegistry | EDAToolset | ReasoningCtrl |
|-----------|--------|------|-----|--------------|-----------|---------------|
| Cortex | — | → mask | | | | |
| Gate | | — | → weights/idx | | | |
| MoE | | ← call | — | | | |
| ToolRegistry | | | | — | → dispatch | ← call |
| EDAToolset | | | | ← result | — | |
| ReasoningCtrl | ↘ domain_steer | | | → call | | — |

No cycles: model core → tokens → controller → tools → tokens (acyclic per step).

### Requirement Allocation
| Requirement | Component(s) |
|-------------|--------------|
| R1 Sparse domain routing | DomainCortexRouter, Gate |
| R2 Cross-domain affinity | DomainCortexRouter |
| R3 Domain steering | DomainCortexRouter (←ReasoningController) |
| R4 Tool registry/dispatch | ToolRegistry |
| R5 EDA toolset | EDAToolset |
| R6 Reasoning loop | ReasoningController, Verifier |
| R7 Episodic memory | EpisodicMemory (reused KV stack) |
| R8 Back-compat/verification | all (config-gated) |

### Key Design-Induced Invariants
- **Eligibility ⊇ activation**: `domain_topk × experts_per_domain ≥ n_activated_experts` (else expert top-k starves). Enforced at construction.
- **Mask-before-select**: cortex masking happens strictly before expert top-k, so no out-of-column expert can ever receive a token.
- **Acyclic step**: tools never call the model synchronously within a forward pass; they return envelopes consumed on the next generate step.
- **Gate-off identity**: with `use_domain_cortex=False` the gate path is byte-for-byte the pre-cortex path.

### Alternatives Considered
| Candidate | Strength | Weakness | Why Not Selected |
|-----------|----------|----------|------------------|
| A. Monolithic domain-MoE (fold domains into existing `n_group` routing) | Minimal new code | No dedicated/steerable domain semantics; domain & expert balance entangled; can't be supervised or forced | Domain routing not first-class; weak controllability |
| **B. Hierarchical Sparse Cortex + Tool Sidecar (SELECTED)** | First-class, steerable domains; clean tool boundary; acyclic; minimal blast radius | Domain logic lives in the gate, not its own track | — |
| C. Multi-track Evoformer cortex (separate per-domain representation tracks + Sinkhorn cross-track mixing + triangle updates) | Richest scientific representation; explicit relational track | High flow density, large new state, higher compute; touches block + streams + losses | Over-engineered for first iteration; revisit when relational reasoning is the bottleneck |

### Metrics Summary
| Metric | Selected (B) | Alt A | Alt C |
|--------|--------------|-------|-------|
| Cross-cutting reqs % | ~25% | ~50% | ~50% |
| Cross-cutting invariants % | low | medium | high |
| Flow density | 0.14 | 0.10 | 0.40 |
| God object score | ~25% (MoE host) | ~45% (gate) | ~35% (trunk) |
| Sync cycles | 0 | 0 | 0 |
| Max fan-in | 2 (MoE) | 3 (gate) | 4 (trunk) |
| Max fan-out | 2 (controller) | 2 | 5 |
| Evolvability cost | low (~1.3 comp/REQ) | medium | high (~2.5) |

Candidate B minimises cross-cutting concerns and keeps a clean, acyclic separation between the sparse model core and the tool/EDA runtime, at the cost of not yet isolating domain representation into its own track (deferred to Candidate C if needed).

> **Update:** Candidate C's relational track was subsequently implemented as an *optional, additive* refinement (`RelationalCortex`, `relational.py`) rather than a core replacement — it refines the small concept-anchor set (bounded K²/K³ cost), is identity-at-init, and is config-gated (`use_relational_cortex`). This captures Candidate C's relational-reasoning strength without incurring its flow-density/compute cost on the main token path.
