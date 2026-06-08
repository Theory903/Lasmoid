# Design Document

## Overview

This design specifies an **audit-and-harden** effort across the entire Lasmoid
codebase: ~45 modules in `inference/`, plus the `train/` and `encoding/` packages.
Lasmoid is a hybrid concept transformer (CSA/HCA/MLA attention + Mamba-2/SSD state
space recurrence + manifold-constrained hyper-connections + grey-box MoE + elastic
sparse concept memory + multimodal vision/audio + tools/reasoning + a Muon/GRPO
training stack). The effort verifies every module is mathematically correct, removes
placeholder/`Faked_Math`, eliminates redundant computation, stabilizes numerics, and
proves core math against the in-workspace reference repositories.

The work is **not** a rewrite. It is a disciplined, per-module pass that applies the
same methodology everywhere: read the code, identify defects, fix them, add a
CPU-runnable correctness/smoke test, and verify against a reference where one exists.
All verification in this iteration runs on **CPU** under `config_100m.json`
(`Tiny_Config`). GPU training/generation/benchmark scripts are authored and statically
syntax-checked but not executed here.

Correctness and reference parity take priority over backward compatibility. Breaking
changes to checkpoint/config schemas are permitted and accompanied by migration notes
where practical.

### Concrete defects already identified (seed the audit)

These were found during design discovery and anchor the methodology:

- **`mhc.py` hardcodes Sinkhorn iterations on the live path; the config-driven path is
  dead code.** `block.py` (the active `LasmoidBlock`) constructs
  `ManifoldConstrainedHyperConnection(args.dim, args.num_residual_streams)` — a class whose
  `__init__` takes *no* iteration argument and whose `forward` loops a literal `range(20)`.
  So `hc_sinkhorn_iters` (=8 in `Tiny_Config`) is silently ignored wherever mHC actually
  runs. The *parametrized* path does exist — `MHCBlock` correctly threads
  `self.hc_sinkhorn_iters` into the `hc_split_sinkhorn` kernel, and `config.py` exposes it
  via the `mhc_config` property / `MHCConfig.sinkhorn_iters` — but nothing in
  `Lasmoid/inference` constructs `MHCBlock` or reads `mhc_config`, leaving `MHCBlock`,
  `MHCConfig`, and `mhc_config` as dead code / double maintenance. Fix: drive the live
  class from `hc_sinkhorn_iters` and either delete or re-wire the unused `MHCBlock` path
  (Req 4.1, Req 18).
- **`ssm.py` `ssm_chunk_scan` is not a parallel scan.** Despite the "Mamba-3 style chunked
  associative scan" docstring it is a `@torch.jit.script` function whose body is a nested
  `for c in range(n_chunks): for t in range(t0, t1):` sequential loop running the identical
  recurrence `curr_s = dt * curr_s + vt * Bt` as `ssm_recurrence_loop`. It is selected only
  when `S > chunk_size` (the short-sequence path uses `ssm_recurrence_loop` directly), so
  the two routes differ only by an outer chunk loop that changes nothing computationally —
  same O(S) sequential dependency, none of the claimed training efficiency. It trivially
  equals the sequential recurrence (Req 3.3); the finding is both redundant work and
  misleading math (Req 3, Req 18).
- **`attention.py` `precompute_freqs_cis` YaRN ramp is bypassed under `Tiny_Config`, and is
  untested.** The YaRN interpolation activates only when `original_seq_len > 0`; the MLA
  forward passes `original_seq_len = args.original_seq_len` *only* when `self.compress_ratio`
  is truthy and otherwise forces `original_seq_len = 0`. With `Tiny_Config`'s
  `original_seq_len: 0` the ramp never runs on CPU. Note this *mirrors* the
  `DeepSeek-V4-Pro` reference, which likewise disables YaRN at `original_seq_len == 0`
  ("disable YaRN and use base rope_theta in pure sliding-window attention"), so this is a
  **test-coverage gap rather than a divergence**: parity tests must pin the active
  (no-YaRN) branch *and* exercise the YaRN branch explicitly with a non-zero
  `original_seq_len` fixture against DeepSeek-V4-Pro (Req 1.3, Req 21).
- **`moe.py` has two mutually exclusive routing modes.** `DeepSeekMoE.forward` dispatches to
  `_adaptive_forward` when `adaptive_routing` is set (top-p / nucleus *variable-k*
  selection via `_adaptive_select`) and to the fixed top-k path otherwise. The "exactly
  top-k experts" guarantee (Req 6.1) holds only on the fixed path; the adaptive path
  selects a variable expert count bounded by the configured maximum. Audit must verify gate
  renormalization (sum-to-1.0) on *both* paths and pin which mode `Tiny_Config` exercises
  (Req 6.1, Req 18).

These illustrate the defect classes the audit hunts: `Faked_Math`/misleading
implementations, redundant/duplicated computation, config-vs-code drift, and dead /
double-maintained code paths.

## Architecture

Two pipelines drive the effort: an **audit pipeline** (how each module is hardened) and
a **test harness** (the shared utilities that make hardening verifiable on CPU).

### Audit pipeline (per module)

Every audited module passes through the same six stages:

```
  read ──▶ classify defects ──▶ fix ──▶ test ──▶ reference-parity ──▶ verify
   │            │                 │       │            │                │
 source     Faked_Math?        minimal  CPU corr.   compare vs       run suite,
 + docs     redundant?         change   + smoke     Reference_Repo   confirm green
            unstable?          only      test       (if applicable)
            config drift?
```

1. **Read** the module and its `Tiny_Config` usage; note documented intent vs actual math.
2. **Classify defects** into: `Faked_Math` (placeholder/hardcoded/incorrect), redundant
   work (Python loops that should vectorize, recomputed intermediates, duplicate
   modules), numerical instability (missing eps, unclamped exp/softplus, div-by-zero),
   and config drift (literals where config values belong).
3. **Fix** with the smallest change that restores correctness; preserve public call
   signatures unless a breaking change is required (then record a migration note).
4. **Test**: add or extend a CPU correctness test and a smoke test for the module.
5. **Reference-parity**: where a `Reference_Repo` analog exists, add a parity test with a
   documented tolerance.
6. **Verify**: run the module's tests plus the global smoke suite; confirm no
   regressions via the output-equality guard (Req 18.1).

### Test harness architecture

A single shared package, `inference/_harness.py` (new), provides reusable checks the
whole suite composes. It is CPU-only and `Tiny_Config`-driven.

```
                 ┌─────────────────────── _harness.py ───────────────────────┐
                 │ tiny_config()      → load config_100m.json as ModelArgs    │
                 │ nan_inf_guard()    → assert finite; report module+tensor   │
                 │ assert_shape()     → shape contract checks                 │
                 │ grad_flow()        → backward; assert finite nonzero grads │
                 │ determinism()      → run twice w/ seed; assert equal       │
                 │ expert_coverage()  → batch>experts; assert all experts hit │
                 │ tool_dispatch()    → every registered tool → valid envelope│
                 │ equiv_guard()      → refactor old vs new within atol       │
                 │ ref_parity()       → max|Δ| vs Reference_Repo within atol  │
                 └────────────────────────────────────────────────────────────┘
                          ▲              ▲              ▲
          test_model_parts.py   test_embedding_fusion.py   test_<subsystem>.py (new)
```

The harness centralizes tolerances and seeds so individual tests stay declarative. Tests
import `_harness`, build modules from `tiny_config()`, and assert via the shared checks.

### Reference-parity strategy

Parity compares a Lasmoid component against the corresponding implementation in an
in-workspace `Reference_Repo`, feeding identical inputs and (where weights matter)
copied weights, then asserting `max|Δ|` within a documented tolerance under float32.

| Lasmoid component | Reference_Repo source | Tolerance (fp32) |
|---|---|---|
| `RMSNorm` (`_common.py`) | `gemma/gemma/gm/nn/_layers.py` `RMSNorm` | 1e-5 |
| RoPE / YaRN freqs (`attention.py precompute_freqs_cis`, `_common.apply_rotary_emb`) | `DeepSeek-V4-Pro/inference/model.py` `precompute_freqs_cis` / `apply_rotary_emb` | 1e-5 |
| Attention scores + softmax (`attention.py`) | `gpt-oss/gpt_oss/torch/model.py` `sdpa` | 1e-4 |
| MLA latent projection (`attention.py`) | `DeepSeek-V4-Pro/inference/model.py` `MLA` | 1e-4 |
| SSM ZOH decay `exp(dt*A)` (`ssm.py`) | reference Mamba-2 formula (closed form) | 1e-5 |
| SSM chunked scan vs sequential (`ssm.py`) | internal `ssm_recurrence_loop` | 1e-4 |
| MoE top-k routing + renorm (`moe.py` `Gate`) | `DeepSeek-V4-Pro/inference/model.py` `Gate` | 1e-5 |
| Sinkhorn doubly-stochastic (`mhc.py`) | closed-form row/col-sum check | 1e-3 |

**Parity caveats (verified against the reference sources):**

- **gpt-oss `sdpa` uses attention sinks.** `gpt-oss/gpt_oss/torch/model.py:sdpa` appends a
  learned sink logit column `S` to `QK` before `softmax` and slices it off afterward. A
  faithful parity comparison must either feed Lasmoid's matching sink mechanism or disable
  sinks on both sides; otherwise the softmax denominators differ and the comparison is
  meaningless. Pin this explicitly in the parity fixture.
- **DeepSeek-V4-Pro `precompute_freqs_cis` disables YaRN when `original_seq_len == 0`,**
  exactly as Lasmoid does. The parity test therefore needs *two* fixtures: one with
  `original_seq_len == 0` (active CPU branch, must match the reference's no-YaRN path) and
  one with `original_seq_len > 0` (exercises the YaRN ramp, otherwise untested).
- **gemma exposes several `RMSNorm` variants** (`gm/nn/_layers.py`, `gm/nn/gemma4/_layers.py`,
  `gm/nn/gemma3n/_layers.py`, `gm/nn/gemma4/vision/_norms.py`). Use the top-level
  `gm/nn/_layers.py` as the canonical reference and note that gemma applies gamma as
  `(1 + weight)` — align the weight convention before comparing or the 1e-5 tolerance will
  not hold.

Where Lasmoid intentionally deviates from a reference (e.g. heavy-tail SSM decay,
manifold-constrained hyper-connections, grey-box concept experts), the deviation and its
rationale are documented in the per-subsystem notes below and the parity test is replaced
by a self-consistency / closed-form check (Req 21.3).

### Tolerance and determinism conventions

- All parity and equivalence checks run in **float32** on CPU regardless of the config
  `dtype` (`bf16`), to isolate algorithmic error from precision error.
- Determinism uses a fixed seed (default `1234`) applied to `torch.manual_seed` and any
  module-local RNG before each run.
- Default tolerances: `1e-5` for pure-math/parity, `1e-4` for multi-step scans and
  cache-vs-full comparisons, `1e-3` for Sinkhorn doubly-stochastic sums.

## Components and Interfaces

Grouped by subsystem. Each module lists its audit focus, specific known/likely
defects/risks, and the verification added. Public interfaces are preserved unless a
breaking change is noted.

### Core math (`_common.py`)

- **`RMSNorm`** — Computes `x * rsqrt(mean(x^2) + eps) * weight` in float32 then casts
  back. Audit confirms eps placement inside the sqrt and gamma applied after
  normalization. Risk: `weight` stored as float32 while activations are bf16 — verify the
  cast path. Verification: formula property + `gemma` parity (1e-5). (Req 1.1, 1.2)
- **`apply_rotary_emb`** — Complex-multiply rotation. Audit the rank-handling branches
  (3D/4D, freq ndim 2/3) for correct broadcasting and the `inverse` conjugation.
  Verification: relative-position dot-product invariance property. (Req 1.4)
- **`Linear`** — fp8/fp4/bf16 dispatch + einsum path. Audit: ensure CPU/bf16 path is exact
  and the einsum equation `...d,od->...o` matches `F.linear`. Verification: equivalence
  guard between einsum and `F.linear` paths. (Req 18.1)

### RoPE construction (`attention.py: precompute_freqs_cis`)

- YaRN frequency scaling via `find_correction_range` + `linear_ramp_factor`, activating
  only when `original_seq_len > 0`. The MLA forward passes
  `original_seq_len = args.original_seq_len` only when `self.compress_ratio` is truthy and
  otherwise forces `0`; with `Tiny_Config`'s `original_seq_len: 0` the ramp is bypassed on
  CPU. **This matches the DeepSeek-V4-Pro reference** (which also disables YaRN at
  `original_seq_len == 0`), so it is a **coverage gap, not drift**. Fix: keep the behavior,
  add an explicit non-zero `original_seq_len` fixture so the YaRN branch is exercised and
  parity-checked against `DeepSeek-V4-Pro/inference/model.py precompute_freqs_cis`, and add
  a second fixture pinning the active no-YaRN branch. (Req 1.3, 21)

### Attention subsystem (`attention.py`, `attnres.py`, `attention_indexer.py`, `ring.py`)

- **`attention.py`** — CSA/HCA/MLA forward. Audit: output shape equals input hidden
  shape; causal mask zeroes future positions; logit soft-cap applied pre-softmax; MLA
  latent compression reconstructs `n_heads × head_dim`. Risk: soft-cap constant vs config;
  mask construction off-by-one in decode. Verification: shape, causal, soft-cap, MLA-shape
  properties + KV-cache-vs-full equivalence. (Req 2.1–2.5)
- **`attention_indexer.py`** — lightning top-k block selection
  (`lightning_topk_blocks`). Audit for correct block indexing and that masked blocks
  contribute zero. Verification: causal property reuse + shape smoke.
- **`attnres.py` / `ring.py`** — residual attention and ring/streaming attention. Audit
  for redundant recomputation of scores across streams; ring buffer wraparound
  correctness. Verification: equivalence guard vs non-ring path (1e-4). (Req 18)

### State space recurrence (`ssm.py`)

- **`StateSpaceRecurrence`** — dt via `softplus(delta + dt_bias)` clamped to
  `[dt_min, dt_max]`; ZOH decay `exp(dt*A)`; B/C L2-normalized; D-skip bypass
  (`ssm_d_skip: true`). The module self-identifies as "Mamba-3" and adds an optional
  `heavy_tail_decay(x, alpha)` (rational decay, `alpha == 1.0` reduces toward standard
  exponential) gated by `use_heavy_tail`.
- **Defect:** `ssm_chunk_scan` (a `@torch.jit.script` function) is a nested sequential
  `for c ... for t` loop, not an associative scan, and is selected only when
  `S > chunk_size` (`ssm_chunk_size: 64`); `S <= chunk_size` routes through
  `ssm_recurrence_loop`. Remove the misleading "parallel/associative" framing and either
  (a) implement a true chunked scan or (b) document it as a reference sequential
  implementation and ensure the training path picks an efficient route. Either way, scan
  output must equal `ssm_recurrence_loop` within 1e-4. (Req 3.3, 18)
- **Risk:** `heavy_tail_decay` deviates from standard `exp` when `alpha != 1.0`; gated by
  `use_heavy_tail`. Document deviation; parity-check that the `alpha == 1.0` /
  heavy-tail-disabled path equals `exp(dt*A)` within tolerance (note: the rational form is
  only approximately equal to `exp` near `alpha == 1.0`, so the bit-compatible baseline is
  the `use_heavy_tail == False` branch, which the caller must select for ZOH parity).
- Verification: dt-clamp property, ZOH-decay parity, scan equivalence, dt-scaling
  metamorphic property, NaN guard with chunk index. (Req 3.1–3.5)

### Manifold-constrained hyper-connections (`mhc.py`)

- **Defect (live path):** `block.py` builds `ManifoldConstrainedHyperConnection`, whose
  `forward` hardcodes `range(20)` Sinkhorn iterations and accepts no iteration argument, so
  `hc_sinkhorn_iters` is ignored where mHC actually runs. **Dead path:** the parametrized
  `MHCBlock` (threads `hc_sinkhorn_iters` into `hc_split_sinkhorn`) plus `MHCConfig` and the
  `mhc_config` property are unused in `Lasmoid/inference`. Fix: drive
  `ManifoldConstrainedHyperConnection` from `hc_sinkhorn_iters`, then either delete the
  unused `MHCBlock`/`MHCConfig`/`mhc_config` or re-wire `block.py` to use `MHCBlock` as the
  single canonical implementation. (Req 4.1, 18)
- **Risk:** alternating `F.normalize(..., p=1, dim=1/dim=2)` must converge to a doubly
  stochastic matrix (row & col sums = 1.0 ± 1e-3). The `stream_count == 1` config must
  reduce to plain residual add.
- Verification: doubly-stochastic property, residual-mix example, single-stream edge case.
  (Req 4.2–4.4)

### Multi-token prediction (`mtp.py`)

- Stream fusion: project next-token embedding + hidden state, fuse through RMSNorm.
  Audit: t+2 logits vocab dim equals `vocab_size`; the MTP path must not mutate the
  primary t+1 logits (independence). Verification: vocab-dim shape property + primary-head
  independence metamorphic property. (Req 5.1–5.3)

### MoE subsystem (`moe.py`)

- **`Gate`** — top-k selection + gate renormalization (sum to 1.0 ± 1e-5); EMA routing
  bias update (`ema_bias_lr`, target load fraction); router z-loss. Audit
  `apply_pending_updates` for correct EMA direction toward target load (it applies queued
  `pending_bias_updates` under `no_grad`).
- **`DeepSeekMoE`** — must sum routed-expert + shared-expert + dense-FFN contributions
  (`moe_dual_ffn: true`). `forward` branches on `adaptive_routing`: when set it calls
  `_adaptive_forward` (top-p / nucleus **variable-k** selection via `_adaptive_select`,
  bounded by the configured max), otherwise the fixed top-k path. The "exactly top-k"
  guarantee holds only on the fixed path; both paths must still renormalize gates to 1.0.
  Audit `_adaptive_select` / `_adaptive_forward` for capacity handling and dead-expert
  risk; ensure every expert is reachable. Pin which mode `Tiny_Config` runs.
- **`ConceptExpert`** — grey-box expert (SwiGLU FFN with optional concept `weights`); audit
  for `Faked_Math` in the concept gating.
- Verification: top-k-renorm property (conditioned on routing mode), gate-sum-to-1.0
  property covering both modes, three-branch-sum property, expert-coverage over a batch
  larger than expert count, EMA-bias monotonicity property, per-expert nonzero-grad
  property. (Req 6.1–6.5)

### Concept memory / VQ / compression (`concept_memory.py`, `vq.py`, `compressor.py`, `compaction.py`, `episodic.py`, `relational.py`)

- **`concept_memory.py` (ESCM)** — perceiver pooling of episodic/semantic/global queries
  to `num_concepts`. Audit attention pooling shapes.
- **`vq.py` (RVQ/GVQ)** — RVQ nearest-centroid selection + commitment/codebook losses;
  GVQ softmax-normalized adjacency message passing. Audit for placeholder distance/loss
  math (Req 7.6 explicitly forbids `Faked_Math`).
- **`compressor.py` (CIF)** — accumulate boundary scores, fire a semantic event at
  accumulated score ≥ 1.0, carry remainder. AR-mode events must equal parallel-mode within
  1e-4.
- **`compaction.py` / `episodic.py` / `relational.py`** — memory eviction/episodic
  store/relational edges; audit for redundant recomputation and shape contracts.
- Verification: pooling concept-count property, RVQ nearest+loss property, CIF fire/carry
  property, CIF AR-vs-parallel equivalence, no-`Faked_Math` audit. (Req 7.1–7.6)

### Multimodal (`vision.py`, `audio.py`, fusion via `test_embedding_fusion.py`)

- **`vision.py`** — patch projection to text `dim`; output token count equals requested
  length; factorized x/y position embeddings added; vision modality id tagged on every
  token; adaptive-pooling fallback when grid doesn't divide evenly (no error).
- **`audio.py`** — waveform → mel-spectrogram conversion; conformer subsampling reduces
  time dim by ×4; output feature dim equals text `dim`; audio modality id tagged.
- **Fusion** — text/vision/audio concatenated into a uniform-feature-dim sequence with a
  modality-id sequence whose length equals the fused token count.
- Verification: vision dim/count + tag properties, grid-fallback edge case, audio
  subsample + dim/tag properties, fusion uniform-dim + id-length properties, existing
  `test_embedding_fusion.py` extended under `Tiny_Config`. (Req 8, 9, 10)

### Tools (`tools.py`)

- **`ToolRegistry`** — `validate` returns an error description for missing/type-mismatched
  args and `None` for valid; `dispatch`/`dispatch_call` returns a `ToolResult` envelope
  with tool name + output; unregistered name → error result.
- **Statistical tools** (`eda_describe`, `eda_correlate`, `eda_fit_model`,
  `eda_hypothesis_test`, `eda_reduce_dim`, `eda_cluster`) — must compute real statistics
  from provided data (no `Faked_Math`); verified against a NumPy reference.
- Verification: validate property, dispatch-envelope property, unknown-tool edge case,
  every-registered-tool example sweep, statistical-output parity vs NumPy. (Req 11.1–11.5)

### Reasoning (`reasoning.py`, `cortex.py`, `curiosity.py`)

- Parse reasoning markup into reasoning segments + final answer; parse tool-call payloads
  into name + argument map; malformed input returns a parse-error indicator rather than
  raising. Audit `cortex.py`/`curiosity.py` for redundant passes over the same tokens.
- Verification: reasoning-extract roundtrip property, tool-payload roundtrip property,
  malformed-input no-exception property. (Req 12.1–12.3)

### Sampling & generation (`sampler.py`, `generate.py`)

- Deterministic under fixed seed; temperature/top-k/top-p restrict the sampled token set;
  final logit soft-cap bounds logits before sampling; non-finite logits routed through the
  NaN/Inf guard rather than sampled.
- Verification: determinism property, parameter-restriction property, final soft-cap
  bound property, non-finite-logit edge case. (Req 13.1–13.4)

### Loss (`loss.py`)

- Primary cross-entropy ignores padded positions (`ignore_index`); aggregate combines
  primary + MTP + VQ-commitment + MoE-balance losses by configured weights; backward
  yields finite grads for participating params.
- Verification: ignore-index metamorphic property, weighted-aggregate property,
  gradient-flow smoke. (Req 14.1–14.3)

### Optimizer (`train/optimizer.py`, `train/mopd.py`)

- **Muon** — 5th-order Newton-Schulz orthogonalization for 2D params using documented
  coefficients; non-2D params fall back to the configured optimizer; zero-Frobenius-norm
  momentum gets eps stabilization to avoid div-by-zero. Audit `mopd.py` auxiliary
  optimizer for the same eps discipline.
- Verification: orthogonality property (singular values → 1.0), non-2D fallback edge case,
  zero-norm eps edge case, coefficient example. (Req 15.1–15.4)

### Training pipeline (`train/pretrain.py`, `train/train.py`, `train/grpo_stability.py`, `train/reward.py`, `train/prepare_data.py`, `train/long_context_finetune.py`, `train/scheduler.py`, `inference/recovery.py`)

- **GRPO** (`grpo_stability.py`, `reward.py`) — group advantages normalized by group
  mean/std with eps; clipped probability ratio + KL penalty against the reference policy;
  malformed completions get a defined penalty (no exception).
- **Data prep** (`prepare_data.py`) — produced batches match configured seq-len/batch.
- **Scheduler** (`scheduler.py`) — lr follows configured warmup/decay schedule.
- **Recovery** (`inference/recovery.py`) — audit checkpoint recovery for state-shape
  consistency.
- Verification: advantage-normalization property, GRPO-objective example, data-shape
  property, scheduler-lr property, malformed-completion penalty edge case. (Req 16.1–16.5)

### Encoding (`encoding/encoding_lasmoid.py`)

- Encode→decode roundtrip reproduces the original text for in-vocabulary input; batch
  encoding yields ids within `[0, vocab_size)`.
- Verification: roundtrip property, id-range property. (Req 17.1, 17.2)

### Authored GPU scripts (not executed)

Three new scripts are delivered under `train/` / repo root, statically syntax-checked
(`python -m py_compile`) but not run:

- **`scripts/gpu_train.py`** — accepts `--config <path>` and `--device <cuda:N>`; wraps the
  existing `train/pretrain.py` loop for full-scale training.
- **`scripts/gpu_generate.py`** — loads a checkpoint, takes `--prompt`, produces sampled
  output via `inference/generate.py` + `sampler.py`.
- **`scripts/gpu_benchmark.py`** — reports tokens/sec throughput and peak memory.

Each includes an inline usage docstring documenting required args and hardware
assumptions. (Req 20.1–20.5)

## Data Models

### Configuration (`Tiny_Config` = `config_100m.json`)

Key fields the harness reads (verified present): `dim: 384`, `n_layers: 12`,
`n_heads: 6`, `head_dim: 48`, `vocab_size: 129280`, `max_seq_len: 512`,
`max_batch_size: 4`, `dtype: bf16`, `norm_eps: 1e-6`, `rope_theta: 10000.0`,
`original_seq_len: 0` (YaRN bypass note above), `num_residual_streams: 4`,
`hc_sinkhorn_iters: 8`, `n_routed_experts: 6`, `n_shared_experts: 1`,
`n_activated_experts: 2`, `num_concepts: 64`, `codebook_size: 256`, `ssm_heads: 6`,
`ssm_state_dim: 16`, `ssm_chunk_size: 64`, `ssm_dt_min: 0.001`, `ssm_dt_max: 0.1`.

### Tolerance table (single source of truth in `_harness.py`)

```python
TOLERANCES = {
    "pure_math":   1e-5,  # RMSNorm, RoPE, ZOH decay, MoE renorm
    "multi_step":  1e-4,  # SSM scan, KV-cache vs full, CIF AR vs parallel
    "sinkhorn":    1e-3,  # doubly-stochastic row/col sums
    "refactor":    1e-5,  # output-equality guard for redundant-work removal
}
SEED = 1234
```

### Test fixtures

- `tiny_model()` — builds the full `Lasmoid_Core` from `Tiny_Config` on CPU in float32.
- `random_hidden(B, S)` — `(B, S, dim)` normal tensor for module forward checks.
- `mixed_multimodal_batch()` — text + small image + short waveform for fusion tests.
- `reward_group()` — synthetic reward vectors for GRPO advantage tests.
- `eda_dataset()` — small numeric arrays with a known closed-form NumPy answer for tool
  parity.
- Reference fixtures loaded from `gemma/gemma`, `DeepSeek-V4-Pro`, `gpt-oss/gpt_oss`.

## Error Handling

- **NaN/Inf:** `nan_inf_guard(tensor, module_name, tensor_name)` raises a structured
  `NonFiniteError` naming the module and tensor (Req 1.5, 3.5, 13.4, 19.4). The SSM scan
  guard additionally reports the offending chunk index.
- **Division by zero:** RMSNorm/SSM dt/Muon orthogonalization use eps stabilization;
  optimizer guards zero-Frobenius-norm momentum (Req 15.4); GRPO advantage normalization
  adds eps to group std (Req 16.1).
- **Parse errors:** reasoning/tool-payload parsing returns a parse-error indicator instead
  of raising (Req 12.3); unregistered tool names return an error envelope (Req 11.3);
  malformed reward completions return a defined penalty (Req 16.5).
- **Shape mismatches:** `assert_shape` and `apply_rotary_emb`'s existing rank/length
  checks raise descriptive `ValueError`s rather than failing silently.
- **Graceful fallbacks:** vision adaptive-pooling fallback for indivisible grids
  (Req 8.4); non-2D Muon fallback optimizer (Req 15.3).

## Testing Strategy

Four complementary layers, all CPU + `Tiny_Config`:

1. **CPU correctness tests** — per-module unit tests asserting documented math
   (formulas, shapes, masking, bounds). Extend `inference/test_model_parts.py` and add
   `inference/test_<subsystem>.py` files.
2. **Smoke tests** — forward-pass shape check for every audited module, gradient-flow
   check (finite, nonzero grads for participating params), determinism check (two seeded
   runs equal), NaN/Inf scan over outputs, expert-activation + tool-dispatch checks
   (Req 19).
3. **Reference-parity tests** — compare against `gemma/gemma`, `DeepSeek-V4-Pro`,
   `gpt-oss/gpt_oss` per the parity table; report `max|Δ|` and assert within tolerance
   (Req 21). Intentional deviations (heavy-tail SSM, mHC, concept experts) use closed-form
   self-consistency checks instead and are documented.
4. **Authored GPU scripts** — static `py_compile` syntax check only; not executed
   (Req 20.4).

### Dual approach

- **Property tests** cover universal behavior across generated inputs (≥100 iterations
  each, seeded for reproducibility), tagged `Feature: lasmoid-hardening, Property N: ...`.
- **Unit/example tests** cover specific scenarios, integration points, and edge cases.

### Redundant-work safety net

Before any performance refactor, capture the module's output on a fixed seeded input;
after refactor, the `equiv_guard` asserts equality within `1e-5` (Req 18.1) and the
module's existing correctness tests must still pass (Req 18.4).

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid
executions of a system — a formal statement about what the system should do. Properties
bridge human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: RMSNorm matches its definition and reference

For any input tensor and gamma weights, the Norm_Module output equals
`x * rsqrt(mean(x^2, last_dim) + eps) * gamma`, and matches the `gemma` RMSNorm within
1e-5 under float32.

**Validates: Requirements 1.1, 1.2**

### Property 2: RoPE preserves relative position

For any two positions p and q and any query/key vectors, the rotated dot product depends
only on the offset (p − q), not on absolute positions.

**Validates: Requirements 1.4**

### Property 3: Attention preserves hidden shape

For any valid hidden-state input under Tiny_Config, the Attention_Subsystem output tensor
has the same shape as the input.

**Validates: Requirements 2.1**

### Property 4: Causal attention ignores the future

For any sequence in causal decode mode, attention weight assigned to any position later
than the query position is zero.

**Validates: Requirements 2.2**

### Property 5: Attention logits respect the soft cap

For any inputs, pre-softmax attention logits remain within the configured soft-cap bound.

**Validates: Requirements 2.3**

### Property 6: MLA reconstructs correct head shapes

For any input projected through the MLA latent compression dimension, the reconstructed
per-head key and value tensors match the configured head count and head dimension.

**Validates: Requirements 2.4**

### Property 7: Incremental decode equals full forward

For any token sequence, decoding step-by-step with a KV cache produces outputs equal
within 1e-4 to a single full-sequence forward pass over the same tokens.

**Validates: Requirements 2.5**

### Property 8: SSM dt is bounded

For any delta input, the discrete time step equals `softplus(delta + dt_bias)` clamped to
`[dt_min, dt_max]`.

**Validates: Requirements 3.1, 3.4**

### Property 9: SSM decay uses ZOH

For any dt and state matrix A, with the heavy-tail path disabled (`use_heavy_tail == False`),
the discrete decay equals `exp(dt * A)`. (The rational `heavy_tail_decay` form only
approximates `exp` near `alpha == 1.0`, so the ZOH baseline is the heavy-tail-disabled
branch.)

**Validates: Requirements 3.2**

### Property 10: SSM chunked scan equals sequential recurrence

For any input sequence, the chunked scan output equals the sequential recurrence output
within 1e-4.

**Validates: Requirements 3.3**

### Property 11: Sinkhorn yields a doubly stochastic matrix

For any input matrix, after the configured (`hc_sinkhorn_iters`) Sinkhorn iterations every
row sum and every column sum equals 1.0 within 1e-3.

**Validates: Requirements 4.1, 4.2**

### Property 12: Single-stream mHC is a plain residual add

For any input when the configured stream count is 1, the MHC_Module reduces to standard
residual addition.

**Validates: Requirements 4.4**

### Property 13: MTP t+2 logits have vocabulary width

For any input under Tiny_Config, the MTP t+2 logits tensor has a vocabulary dimension
equal to the configured vocabulary size.

**Validates: Requirements 5.2**

### Property 14: MTP does not perturb the primary head

For any hidden states, the primary t+1 logits are identical whether or not the MTP
computation runs.

**Validates: Requirements 5.3**

### Property 15: MoE gates are top-k and renormalized

For any router logits, when fixed-top-k routing is active exactly the configured top-k
experts are selected; and on both the fixed-top-k and adaptive (nucleus variable-k) paths
the selected experts' gating weights sum to 1.0 within 1e-5.

**Validates: Requirements 6.1**

### Property 16: MoE output sums all three branches

For any token, the MoE layer output equals the sum of the routed-expert, shared-expert,
and dense-FFN contributions.

**Validates: Requirements 6.2**

### Property 17: Every expert is reachable

For any batch sized larger than the expert count, every expert receives at least one
routed token, and each activated expert's parameters receive a nonzero gradient after a
backward pass.

**Validates: Requirements 6.3, 6.5**

### Property 18: EMA routing bias moves toward target load

For any routing imbalance, the EMA routing-bias update shifts the bias toward the target
load fraction.

**Validates: Requirements 6.4**

### Property 19: Concept pooling produces the configured concept count

For any encoder hidden states, perceiver pooling produces pooled representations whose
count equals the configured number of concepts.

**Validates: Requirements 7.1**

### Property 20: RVQ selects the nearest centroid

For any pooled vector, RVQ selects the codebook centroid minimizing distance and computes
defined, finite commitment and codebook losses.

**Validates: Requirements 7.2, 7.6**

### Property 21: CIF fires on threshold and carries remainder

For any boundary-score stream, the compressor fires a semantic event exactly when the
accumulated score reaches or exceeds 1.0 and carries the remainder to the next step.

**Validates: Requirements 7.4**

### Property 22: CIF autoregressive equals parallel mode

For any input sequence, CIF fired events in autoregressive mode equal those in parallel
mode within 1e-4.

**Validates: Requirements 7.5**

### Property 23: Vision embeddings match text space and length

For any image batch, vision embeddings have a feature dimension equal to the text model
dimension and a token count equal to the requested output length, and every produced
token carries the vision modality identifier.

**Validates: Requirements 8.1, 8.2**

### Property 24: Audio subsampling and projection are correct

For any waveform input, conformer subsampling reduces the time dimension by a factor of 4,
the output feature dimension equals the text model dimension, and every produced token
carries the audio modality identifier.

**Validates: Requirements 9.2, 9.3**

### Property 25: Multimodal fusion is uniform and consistently tagged

For any mixed text-vision-audio input, the fused sequence has a uniform feature dimension
across modalities and a modality-identifier sequence whose length equals the fused token
count.

**Validates: Requirements 10.1, 10.2**

### Property 26: Tool validation accepts valid and rejects invalid arguments

For any registered schema and argument set, validation returns an error description when an
argument is missing or type-mismatched and returns no error for a valid set.

**Validates: Requirements 11.1**

### Property 27: Valid dispatch returns a well-formed envelope

For any valid tool call, dispatch executes the registered function and returns a result
envelope containing the tool name and output.

**Validates: Requirements 11.2**

### Property 28: Statistical tools compute real statistics

For any input dataset, each statistical tool's output equals an independent NumPy
reference computation within tolerance (no Faked_Math).

**Validates: Requirements 11.5**

### Property 29: Reasoning and tool-payload parsing round-trip

For any well-formed completion, the Reasoning_Subsystem extracts reasoning segments and
the final answer as separate fields and parses any tool-call payload into a structured
name and argument map; for any malformed input it returns a parse-error indicator without
raising.

**Validates: Requirements 12.1, 12.2, 12.3**

### Property 30: Sampling is deterministic under a fixed seed

For any inputs and parameters with a fixed seed, two sampling runs produce identical token
sequences.

**Validates: Requirements 13.1**

### Property 31: Sampling respects its parameters and soft cap

For any logits and any temperature/top-k/top-p settings, sampling is restricted to the
permitted token set, and the final logits are bounded within the configured soft cap
before sampling.

**Validates: Requirements 13.2, 13.3**

### Property 32: Loss ignores padding and aggregates by weight

For any labels containing padded positions, the primary cross-entropy ignores those
positions, and the aggregate loss equals the configured weighted sum of the primary, MTP,
VQ-commitment, and MoE-balance losses.

**Validates: Requirements 14.1, 14.2**

### Property 33: Muon orthogonalizes 2D updates

For any 2D momentum matrix, the Newton-Schulz iteration produces a result whose singular
values approach 1.0 within the documented tolerance.

**Validates: Requirements 15.1, 15.2**

### Property 34: GRPO advantages are group-normalized

For any group of rewards, GRPO advantages are normalized by the group mean and standard
deviation with epsilon stabilization, yielding approximately zero mean and unit standard
deviation.

**Validates: Requirements 16.1**

### Property 35: Prepared batches and scheduler match configuration

For any prepared training batch, token shapes match the configured sequence length and
batch size; and for any training step, the scheduler produces a learning rate following
the configured warmup and decay schedule.

**Validates: Requirements 16.3, 16.4**

### Property 36: Encoding round-trips and stays in range

For any in-vocabulary text, encoding then decoding reproduces the original text, and for
any batch every produced token id falls within the configured vocabulary range.

**Validates: Requirements 17.1, 17.2**

### Property 37: Refactors preserve outputs

For any input under Tiny_Config, a refactored module's output equals its pre-refactor
output within 1e-5.

**Validates: Requirements 18.1, 18.3**

### Property 38: Reference-parity comparisons stay within tolerance

For any component with a Reference_Repo analog, given identical inputs and weights, the
maximum absolute difference between the Lasmoid output and the reference output falls
within the documented tolerance.

**Validates: Requirements 21.1, 21.2**

### Edge-case and error-condition checks (covered by example/edge tests, not properties)

These are verified by targeted tests rather than universally-quantified properties:

- NaN/Inf guard reports the affected module and tensor name for non-finite input to Norm,
  RoPE, SSM scan (with chunk index), and Sampler (Req 1.5, 3.5, 13.4).
- mHC residual mixing composition and Sinkhorn iteration count behavior (Req 4.1, 4.3).
- RoPE YaRN frequency-scaling construction with a non-zero `original_seq_len` fixture
  (Req 1.3).
- Vision adaptive-pooling fallback for an indivisible patch grid (Req 8.4).
- Waveform→mel conversion branch and GVQ adjacency message passing (Req 7.3, 9.1).
- Unknown-tool error envelope and every-registered-tool dispatch sweep (Req 11.3, 11.4).
- Muon non-2D fallback and zero-Frobenius-norm eps stabilization (Req 15.3, 15.4).
- GRPO clipped-ratio + KL objective and malformed-completion penalty (Req 16.2, 16.5).
- Loss backward gradient-flow smoke (Req 14.3).

## Migration Notes Mechanism

Breaking changes are permitted (Req 22). Each is recorded in a running
`MIGRATIONS.md` at the spec root with a consistent entry format:

```
## <date> — <module>: <short title>
- Change: what changed in the checkpoint/config schema or math.
- Reason: the correctness defect it fixes (link to Requirement).
- Impact: which checkpoints/configs break.
- Migration: concrete steps to update affected artifacts (where practical),
  or an explicit note that no automated migration is provided.
```

Anticipated entries from defects already found:

- **`mhc.py` Sinkhorn iterations now config-driven** — `ManifoldConstrainedHyperConnection`
  switches from a hardcoded 20 iterations to `hc_sinkhorn_iters`. Under `Tiny_Config` this
  changes the live count from 20 → 8, so outputs shift slightly (within the doubly-stochastic
  1e-3 tolerance); checkpoints are unaffected, but configs relying on the implicit `20`
  should set `hc_sinkhorn_iters` explicitly.
- **Consolidation of duplicate mHC implementations** — `MHCBlock`, `MHCConfig`, and the
  `mhc_config` property are unused on the live path. If they are removed (or `block.py` is
  re-wired to use `MHCBlock` as canonical), any config or checkpoint referencing the removed
  class path needs its module path updated; steps documented per change.
- **`ssm.py` chunk-scan semantics clarified** — no weight change; if a true associative
  scan replaces the sequential loop, numerical outputs stay within 1e-4 (no checkpoint
  migration needed).


## Goals

### Primary Goals (Gating — definition of done)

Requirements 1–23 are the gating definition of done. The effort is complete only when:

1. All CPU correctness and smoke tests pass under `Tiny_Config` (Req 23.2, 19).
2. Every reference-parity comparison for core math (RMSNorm, RoPE/YaRN, attention, MLA,
   SSM decay/scan, MoE routing, Sinkhorn) falls within its documented tolerance
   (Req 23.3, 21).
3. No audited module contains `Faked_Math` in its primary computation paths — explicitly
   including RVQ/GVQ/CIF (Req 7.6) and the statistical tools (Req 11.5) (Req 23.4).
4. All MoE experts activate, all registered tools dispatch, and the multimodal fusion path
   operates under `Tiny_Config` (Req 23.5).
5. Redundant computation is removed without changing outputs beyond the refactor tolerance
   (Req 18), and breaking changes carry migration notes (Req 22).

## Moonshot Goals (Archived — Non-Gating Stretch Targets)

Recorded per Req 24; these do **not** block the definition of done and are not verified in
this CPU iteration. They are run by the user on GPU hardware.

1. **Training loss** — define a target validation loss under a full-scale configuration
   (`config_1b.json` / `config_1b_2m.json`) as a stretch objective (Req 24.2).
2. **Throughput** — define a target tokens-per-second figure on the user's GPU hardware,
   measured by `scripts/gpu_benchmark.py` (Req 24.3).
3. **Generation quality** — define target reasoning and tool-use quality metrics
   (Req 24.4).
4. **Expert balance** — define a target expert load-balance distribution across routed
   experts (Req 24.5).
5. **Multimodal quality** — define target vision and audio understanding metrics
   (Req 24.6).

These are archived intentionally so ambition is recorded without gating completion of the
hardening work.
