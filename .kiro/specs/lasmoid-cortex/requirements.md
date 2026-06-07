# Requirements Document

## Introduction

This document specifies the **Lasmoid Cortex** capability: redesigning Lasmoid into a brain-like, sparsely-activated reasoning core that (a) routes computation across scientific *domain columns* (mathematics, physics, chemistry, biology/medical, astronomy, computer-science, data-analysis, general), (b) uses ML/EDA *tools* to derive information from raw data, and (c) verifies and refines its conclusions. It builds **on top of** the existing Lasmoid architecture (hybrid transformer-SSM, DeepSeekMoE, mHC residual streams, elastic concept memory, multimodal encoders) rather than replacing it.

The design is grounded in techniques mined from the NEXUS reference repos: cortical-column sparse routing (MoE + group routing), doubly-stochastic cross-track mixing (mHC `sinkhorn_log`), iterative multi-track refinement (AlphaFold-3 Evoformer/Pairformer), episodic clustering + attention-importance memory (ml-epicache), and propose→verify→refine reasoning loops (reasoning-from-scratch).

## Glossary

- **Domain_Cortex**: The top-level sparse router that partitions MoE experts into domain columns and activates only the top-k relevant columns per token.
- **Cortical_Column**: A contiguous group of MoE routed experts specialised to one scientific domain.
- **Domain_Steer**: An external additive bias over domain logits that lets a tool/agent force or suppress specific domains.
- **Cross_Domain_Affinity**: A doubly-stochastic (Sinkhorn-normalised) matrix that lets related columns (e.g. physics↔mathematics) softly co-activate.
- **Tool_Registry**: The catalogue of callable tools (ML estimators, EDA primitives, calculators, search) with typed schemas.
- **EDA_Toolset**: Tools that derive information from raw data (describe, correlate, fit, test, reduce-dimensionality, cluster, plot-to-text).
- **Reasoning_Loop**: The propose→verify→refine controller that drives multi-step tool-augmented analysis.
- **Verifier**: A scorer (heuristic, log-prob, or external checker) that accepts a refinement only if it does not worsen the score.

## Requirements

### Requirement 1: Sparse Domain Cortex Routing  *(IMPLEMENTED)*

**User Story:** As a model architect, I want tokens routed to only the relevant scientific domain columns, so that computation is brain-like sparse and domain-specialised.

#### Acceptance Criteria
1. THE Domain_Cortex SHALL partition `n_routed_experts` into `n_domains` contiguous Cortical_Columns and require `n_routed_experts` to be divisible by `n_domains`.
2. WHEN routing a token, THE Domain_Cortex SHALL select exactly `domain_topk` columns via a dedicated learned domain router operating on the normalised hidden state.
3. WHEN columns are selected, THE Domain_Cortex SHALL mask all experts outside the active columns to `-inf` before expert top-k selection, so only in-column experts can fire.
4. FOR ALL configurations, THE Domain_Cortex SHALL guarantee `domain_topk × experts_per_domain ≥ n_activated_experts`.
5. THE Domain_Cortex SHALL expose a load-balance auxiliary loss that encourages uniform column utilisation, weighted by `cortex_load_balance_coeff`.
6. WHEN `use_domain_cortex` is False, THE model SHALL behave identically to the pre-cortex baseline (zero added computation).

### Requirement 2: Cross-Domain Affinity  *(IMPLEMENTED)*

**User Story:** As a model architect, I want related domains to share information stably, so multi-disciplinary problems (e.g. astrophysics) engage several columns coherently.

#### Acceptance Criteria
1. THE Domain_Cortex SHALL compute a Cross_Domain_Affinity matrix via log-domain Sinkhorn normalisation that is row- and column-stochastic within tolerance 1e-3.
2. THE Domain_Cortex SHALL smooth raw domain probabilities through the affinity matrix before column selection.
3. THE affinity matrix SHALL initialise near identity so columns begin independent and learn couplings during training.

### Requirement 3: Domain Steering  *(IMPLEMENTED)*

**User Story:** As an agent/operator, I want to force or suppress specific domains, so external context (e.g. a chemistry tool result) can direct the cortex.

#### Acceptance Criteria
1. WHEN a Domain_Steer vector is supplied, THE Domain_Cortex SHALL add it to domain logits prior to selection.
2. WHEN a column's steer bias dominates, THE Domain_Cortex SHALL include that column in the active set.
3. WHEN no Domain_Steer is supplied, THE Domain_Cortex SHALL route purely from learned domain logits.

### Requirement 4: Tool Registry & Typed Dispatch  *(IMPLEMENTED)*

**User Story:** As a developer, I want a typed tool registry, so the model can call ML/EDA tools safely and deterministically.

#### Acceptance Criteria
1. THE Tool_Registry SHALL register tools with a name, JSON parameter schema, and a callable handler.
2. WHEN the model emits a DSML tool call, THE dispatcher SHALL validate arguments against the schema before invoking the handler.
3. IF arguments fail validation, THEN THE dispatcher SHALL return a structured error result without invoking the handler.
4. THE dispatcher SHALL serialise tool results into the existing Lasmoid tool-result envelope for re-encoding into context.

### Requirement 5: EDA Toolset  *(IMPLEMENTED)*

**User Story:** As a scientist, I want the model to derive information from raw data, so it can perform exploratory data analysis across domains.

#### Acceptance Criteria
1. THE EDA_Toolset SHALL provide primitives for: summary statistics, correlation, distribution tests, curve/model fitting, dimensionality reduction, and clustering.
2. WHEN given a raw dataset reference, THE EDA_Toolset SHALL return structured, text-serialisable findings (not just plots).
3. THE EDA_Toolset SHALL bound output size so large datasets do not overflow the context window.

### Requirement 6: Reasoning Loop (Propose→Verify→Refine)  *(IMPLEMENTED)*

**User Story:** As an operator, I want tool-augmented reasoning that self-verifies, so long-running analyses converge to correct answers.

#### Acceptance Criteria
1. THE Reasoning_Loop SHALL generate a candidate analysis, optionally invoking tools, then score it with a Verifier.
2. THE Reasoning_Loop SHALL accept a refinement only if its Verifier score is ≥ the current best (monotonic non-worsening).
3. THE Reasoning_Loop SHALL terminate on convergence, a step budget, or a confidence threshold.
4. WHEN a tool result is available, THE Verifier SHALL prefer data-grounded scores over heuristic scores.

### Requirement 7: Long-Running Episodic Memory  *(IMPLEMENTED)*

**User Story:** As an operator, I want bounded working memory over long data/observation streams, so analyses run for extended sessions without OOM.

#### Acceptance Criteria
1. THE system SHALL cluster long context into topical episodes and route queries to the nearest episode(s).
2. THE system SHALL bound KV growth via attention-importance eviction with sink-token retention.
3. THE system SHALL preserve the existing tiered KV / eviction / compaction mechanisms.

### Requirement 8: Backward Compatibility & Verification  *(IMPLEMENTED)*

#### Acceptance Criteria
1. ALL new components SHALL be config-gated and default to off.
2. THE existing test suite SHALL pass unchanged when all cortex flags are off.
3. EACH new component SHALL ship with unit tests covering shape, sparsity, gradient flow, and integration.
