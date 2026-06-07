# Implementation Plan: Lasmoid Hardening

## Overview

This plan executes the design's per-module **audit pipeline** (read → classify defects →
fix → test → reference-parity → verify) across all ~45 `inference/` modules plus `train/`
and `encoding/`. Work is grouped by subsystem and ordered so each step builds on the
previous: the shared CPU test harness comes first, then core math, then each subsystem,
then cross-cutting passes (redundant-work elimination, reference parity), then authored
GPU scripts, migration notes, and a final full-suite run.

**Conventions baked into every task (workspace steering + design):**
- Implementation language is **Python 3.14**; all verification runs on **CPU** in float32
  under `Tiny_Config` (`config_100m.json`). Code must NOT assume CUDA — Apple MPS is the
  dev device and CPU is the fallback.
- **Config-flag gating is a hard convention.** Any fix that changes default outputs (e.g.
  mHC Sinkhorn `20 → 8` iterations, a true SSM associative scan) MUST be gated behind a
  `ModelArgs` flag wired through `inference/config.py` with a conservative default that
  preserves current behavior, unless the existing behavior is outright broken. Never change
  default behavior implicitly.
- Tests live beside code as `test_*.py` in `inference/` or `encoding/` and compose the
  shared `inference/_harness.py` utilities.
- Run tests from the `Lasmoid/` directory:
  `DYLD_LIBRARY_PATH=$(brew --prefix expat)/lib python3 -m unittest test_model_parts -v`
  and `python3 -m pytest inference/test_model_parts.py -v`.
- `ARCHITECTURE.md` is the source of truth for the math. Lint with `ruff check .`.
- Sub-tasks marked `*` are optional tests; the core implementation sub-tasks are not.

## Tasks

- [x] 1. Build shared CPU test harness (`inference/_harness.py`)
  - [x] 1.1 Create `inference/_harness.py` foundation
    - Implement `tiny_config()` loading `config_100m.json` into `ModelArgs`; `tiny_model()`
      building `Lasmoid_Core` on CPU in float32; fixtures `random_hidden(B,S)`,
      `mixed_multimodal_batch()`, `reward_group()`, `eda_dataset()`
    - Add the single-source-of-truth `TOLERANCES` table (`pure_math` 1e-5, `multi_step`
      1e-4, `sinkhorn` 1e-3, `refactor` 1e-5) and `SEED = 1234`
    - _Requirements: 19.1, 23.2_
  - [x] 1.2 Implement the composable harness checks
    - `nan_inf_guard()` (structured `NonFiniteError` with module+tensor, chunk index for
      SSM), `assert_shape()`, `grad_flow()`, `determinism()`, `expert_coverage()`,
      `tool_dispatch()`, `equiv_guard()`, `ref_parity()` (max|Δ| vs reference loaders for
      `gemma/gemma`, `DeepSeek-V4-Pro`, `gpt-oss/gpt_oss`)
    - _Requirements: 19.2, 19.3, 19.4, 21.1, 21.2, 18.1_
  - [ ]* 1.3 Write a smoke test for the harness itself
    - Verify `tiny_config()`/`tiny_model()` build on CPU and each check passes/fails on
      crafted inputs
    - _Requirements: 19.1_

- [x] 2. Core math audit (`inference/_common.py`)
  - [x] 2.1 Audit and fix `RMSNorm`
    - Confirm `x * rsqrt(mean(x^2, last_dim) + eps) * gamma` with eps inside the sqrt and
      gamma applied after normalization; verify the float32-compute / bf16-cast path and the
      `weight`-dtype handling
    - _Requirements: 1.1, 1.2_
  - [x] 2.2 Audit and fix `apply_rotary_emb`
    - Verify YaRN frequency scaling (base/dim/interpolation), the 3D/4D and freq-ndim 2/3
      broadcasting branches, and the `inverse` conjugation path
    - _Requirements: 1.3, 1.4_
  - [x] 2.3 Audit `Linear` fp8/fp4/bf16 dispatch and gate the einsum path
    - Ensure the CPU/bf16 path is exact and the einsum equation `...d,od->...o` matches
      `F.linear`; keep einsum behind its existing config flag (e.g. `use_einsum=False`)
      defaulting off
    - _Requirements: 18.1_
  - [ ]* 2.4 Write property test — RMSNorm definition + gemma parity
    - **Property 1: RMSNorm matches its definition and reference**
    - **Validates: Requirements 1.1, 1.2**
  - [ ]* 2.5 Write property test — RoPE relative-position invariance
    - **Property 2: RoPE preserves relative position**
    - **Validates: Requirements 1.4**
  - [ ]* 2.6 Write NaN/Inf guard edge tests for Norm and RoPE
    - Non-finite input is reported with the affected module and tensor name
    - _Requirements: 1.5_

- [x] 3. Attention subsystem + YaRN coverage (`attention.py`, `attnres.py`, `attention_indexer.py`, `ring.py`)
  - [x] 3.1 Audit and fix `attention.py` CSA/HCA/MLA forward
    - Output shape equals input hidden shape; causal mask zeroes future positions (check
      decode off-by-one); logit soft-cap applied pre-softmax from config; MLA latent
      compression reconstructs `n_heads × head_dim`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x] 3.2 Audit `precompute_freqs_cis` YaRN ramp and close the coverage gap
    - Keep the no-YaRN behavior at `original_seq_len == 0` (matches DeepSeek-V4-Pro); add a
      non-zero `original_seq_len` fixture so the YaRN branch is actually exercised; this is a
      coverage gap, not drift — do not change default behavior
    - _Requirements: 1.3, 21.1, 21.3_
  - [x] 3.3 Audit `attention_indexer.py`, `attnres.py`, `ring.py`
    - `lightning_topk_blocks` block indexing with masked blocks contributing zero; remove
      redundant score recomputation across residual/ring streams; ring buffer wraparound
      correctness (equiv vs non-ring path within 1e-4)
    - _Requirements: 2.1, 2.2, 18.2, 18.3_
  - [ ]* 3.4 Write property test — attention preserves hidden shape
    - **Property 3: Attention preserves hidden shape**
    - **Validates: Requirements 2.1**
  - [ ]* 3.5 Write property test — causal attention ignores the future
    - **Property 4: Causal attention ignores the future**
    - **Validates: Requirements 2.2**
  - [ ]* 3.6 Write property test — attention logits respect the soft cap
    - **Property 5: Attention logits respect the soft cap**
    - **Validates: Requirements 2.3**
  - [ ]* 3.7 Write property test — MLA reconstructs correct head shapes
    - **Property 6: MLA reconstructs correct head shapes**
    - **Validates: Requirements 2.4**
  - [ ]* 3.8 Write property test — incremental decode equals full forward
    - **Property 7: Incremental decode equals full forward**
    - **Validates: Requirements 2.5**

- [x] 4. State space recurrence audit (`inference/ssm.py`)
  - [x] 4.1 Fix the `ssm_chunk_scan` chunk-scan defect
    - Remove the misleading "parallel/associative scan" framing on the `@torch.jit.script`
      sequential loop; either implement a true chunked associative scan **gated behind a new
      `ModelArgs` flag (default = current sequential behavior)** or document it as a reference
      sequential implementation; output must equal `ssm_recurrence_loop` within 1e-4
    - _Requirements: 3.3, 18.2, 22.1, 22.2_
  - [x] 4.2 Audit dt mapping, ZOH decay, and heavy-tail gating
    - `dt = softplus(delta + dt_bias)` clamped to `[dt_min, dt_max]`; ZOH decay `exp(dt*A)`;
      B/C L2-normalization; D-skip; input scaled by dt; keep `heavy_tail_decay` behind
      `use_heavy_tail` and document the `alpha != 1.0` deviation
    - _Requirements: 3.1, 3.2, 3.4, 21.3_
  - [ ]* 4.3 Write property test — SSM dt is bounded
    - **Property 8: SSM dt is bounded**
    - **Validates: Requirements 3.1, 3.4**
  - [ ]* 4.4 Write property test — SSM decay uses ZOH
    - **Property 9: SSM decay uses ZOH** (heavy-tail-disabled baseline)
    - **Validates: Requirements 3.2**
  - [ ]* 4.5 Write property test — chunked scan equals sequential recurrence
    - **Property 10: SSM chunked scan equals sequential recurrence**
    - **Validates: Requirements 3.3**
  - [ ]* 4.6 Write SSM NaN-guard edge test with chunk index
    - Non-finite scan state is reported with the offending chunk index
    - _Requirements: 3.5_

- [x] 5. mHC Sinkhorn defect + dead-code consolidation (`inference/mhc.py`, config-gated)
  - [x] 5.1 Drive the live mHC path from `hc_sinkhorn_iters`
    - Make `ManifoldConstrainedHyperConnection` (built by `block.py`) read the configured
      iteration count instead of the hardcoded `range(20)`; wire through `inference/config.py`
      with a conservative default that preserves current behavior unless the existing
      behavior is broken (under `Tiny_Config` this shifts the live count `20 → 8` within the
      1e-3 doubly-stochastic tolerance — record in MIGRATIONS.md)
    - _Requirements: 4.1, 4.2, 18.1, 22.1, 22.2_
  - [x] 5.2 Consolidate the duplicate / dead mHC implementations
    - Resolve the `MHCBlock` / `MHCConfig` / `mhc_config` dead path: either delete it or
      re-wire `block.py` to use `MHCBlock` as the single canonical implementation; ensure
      `stream_count == 1` reduces to a plain residual add
    - _Requirements: 4.3, 4.4, 18.3, 22.2_
  - [ ]* 5.3 Write property test — Sinkhorn yields a doubly stochastic matrix
    - **Property 11: Sinkhorn yields a doubly stochastic matrix**
    - **Validates: Requirements 4.1, 4.2**
  - [ ]* 5.4 Write property test — single-stream mHC is a plain residual add
    - **Property 12: Single-stream mHC is a plain residual add**
    - **Validates: Requirements 4.4**
  - [ ]* 5.5 Write mHC residual-mixing composition edge test
    - Gated identity + doubly-stochastic mixing + gated layer output combine as specified
    - _Requirements: 4.3_

- [x] 6. Multi-token prediction audit (`inference/mtp.py`)
  - [x] 6.1 Audit and fix MTP stream fusion + primary-head independence
    - Project next-token embedding + hidden state, fuse through RMSNorm; ensure the MTP path
      does not mutate the primary t+1 logits
    - _Requirements: 5.1, 5.2, 5.3_
  - [ ]* 6.2 Write property test — MTP t+2 logits have vocabulary width
    - **Property 13: MTP t+2 logits have vocabulary width**
    - **Validates: Requirements 5.2**
  - [ ]* 6.3 Write property test — MTP does not perturb the primary head
    - **Property 14: MTP does not perturb the primary head**
    - **Validates: Requirements 5.3**

- [x] 7. MoE subsystem audit (`inference/moe.py`)
  - [x] 7.1 Audit and fix `Gate` top-k selection + dual routing modes
    - Fixed top-k selects exactly top-k and renormalizes gates to 1.0 ± 1e-5; `_adaptive_select`
      / `_adaptive_forward` (nucleus variable-k) also renormalizes; pin which mode `Tiny_Config`
      exercises and ensure every expert is reachable on both paths
    - _Requirements: 6.1_
  - [x] 7.2 Audit and fix `DeepSeekMoE` three-branch summation
    - Layer output sums routed-expert + shared-expert + dense-FFN contributions
      (`moe_dual_ffn`); audit `_adaptive_forward` capacity handling for dead-expert risk
    - _Requirements: 6.2_
  - [x] 7.3 Audit EMA routing-bias update and `ConceptExpert` math
    - `apply_pending_updates` shifts EMA bias toward the target load fraction; verify
      `ConceptExpert` SwiGLU + concept gating contains no `Faked_Math`
    - _Requirements: 6.4, 23.4_
  - [ ]* 7.4 Write property test — MoE gates are top-k and renormalized
    - **Property 15: MoE gates are top-k and renormalized** (both routing modes)
    - **Validates: Requirements 6.1**
  - [ ]* 7.5 Write property test — MoE output sums all three branches
    - **Property 16: MoE output sums all three branches**
    - **Validates: Requirements 6.2**
  - [ ]* 7.6 Write property test — every expert is reachable + nonzero grad
    - **Property 17: Every expert is reachable** (batch > expert count, per-expert nonzero grad)
    - **Validates: Requirements 6.3, 6.5**
  - [ ]* 7.7 Write property test — EMA routing bias moves toward target load
    - **Property 18: EMA routing bias moves toward target load**
    - **Validates: Requirements 6.4**

- [x] 8. Concept memory / VQ / compressor audit (`concept_memory.py`, `vq.py`, `compressor.py`, `compaction.py`, `episodic.py`, `relational.py`)
  - [x] 8.1 Audit and fix ESCM perceiver pooling (`concept_memory.py`)
    - Attend episodic/semantic/global query sets over encoder hidden states; produce
      `num_concepts` pooled representations with correct shapes
    - _Requirements: 7.1_
  - [x] 8.2 Audit and fix RVQ/GVQ (`vq.py`) — no `Faked_Math`
    - RVQ nearest-centroid selection + real commitment/codebook losses; GVQ softmax-normalized
      adjacency message passing; remove any placeholder distance/loss math
    - _Requirements: 7.2, 7.3, 7.6, 23.4_
  - [x] 8.3 Audit and fix the CIF compressor (`compressor.py`)
    - Accumulate boundary scores, fire a semantic event at accumulated score ≥ 1.0, carry the
      remainder; AR-mode events must equal parallel-mode within 1e-4; no `Faked_Math`
    - _Requirements: 7.4, 7.5, 7.6, 23.4_
  - [x] 8.4 Audit `compaction.py`, `episodic.py`, `relational.py`
    - Memory eviction / episodic store / relational edges: remove redundant recomputation and
      confirm shape contracts
    - _Requirements: 18.2, 18.3_
  - [ ]* 8.5 Write property test — concept pooling produces the configured concept count
    - **Property 19: Concept pooling produces the configured concept count**
    - **Validates: Requirements 7.1**
  - [ ]* 8.6 Write property test — RVQ selects the nearest centroid
    - **Property 20: RVQ selects the nearest centroid** (defined finite losses, no Faked_Math)
    - **Validates: Requirements 7.2, 7.6**
  - [ ]* 8.7 Write property test — CIF fires on threshold and carries remainder
    - **Property 21: CIF fires on threshold and carries remainder**
    - **Validates: Requirements 7.4**
  - [ ]* 8.8 Write property test — CIF autoregressive equals parallel mode
    - **Property 22: CIF autoregressive equals parallel mode**
    - **Validates: Requirements 7.5**

- [x] 9. Multimodal audit (`vision.py`, `audio.py`, fusion)
  - [x] 9.1 Audit and fix the vision encoder (`vision.py`)
    - Patch projection to text `dim`; output token count equals requested length; factorized
      x/y position embeddings added per patch; vision modality id tagged on every token;
      adaptive-pooling fallback when the grid doesn't divide evenly (no error raised)
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 9.2 Audit and fix the audio encoder (`audio.py`)
    - Waveform → mel-spectrogram conversion; conformer subsampling reduces time dim ×4; output
      feature dim equals text `dim`; audio modality id tagged on every token
    - _Requirements: 9.1, 9.2, 9.3_
  - [x] 9.3 Audit and fix embedding fusion; extend `test_embedding_fusion.py`
    - Text/vision/audio fuse into a uniform-feature-dim sequence with a modality-id sequence
      whose length equals the fused token count; extend the fusion test under `Tiny_Config`
    - _Requirements: 10.1, 10.2, 10.3_
  - [ ]* 9.4 Write property test — vision embeddings match text space and length
    - **Property 23: Vision embeddings match text space and length**
    - **Validates: Requirements 8.1, 8.2**
  - [ ]* 9.5 Write property test — audio subsampling and projection are correct
    - **Property 24: Audio subsampling and projection are correct**
    - **Validates: Requirements 9.2, 9.3**
  - [ ]* 9.6 Write property test — multimodal fusion is uniform and consistently tagged
    - **Property 25: Multimodal fusion is uniform and consistently tagged**
    - **Validates: Requirements 10.1, 10.2**
  - [ ]* 9.7 Write edge tests — vision grid fallback + waveform→mel branch
    - Indivisible patch grid falls back to adaptive pooling without error; raw waveform is
      converted to mel before encoding
    - _Requirements: 8.4, 9.1_

- [x] 10. Checkpoint — Ensure all tests pass
  - Run the CPU suite from `Lasmoid/`; ensure all tests pass, ask the user if questions arise.

- [x] 11. Tools audit (`inference/tools.py`)
  - [x] 11.1 Audit and fix `ToolRegistry` validation and dispatch
    - `validate` returns an error description for missing/type-mismatched args and `None` for
      valid; `dispatch`/`dispatch_call` returns a `ToolResult` envelope (tool name + output);
      unregistered name returns an error result identifying the unknown tool
    - _Requirements: 11.1, 11.2, 11.3_
  - [x] 11.2 Audit and fix the statistical tools — no `Faked_Math`
    - `eda_describe`, `eda_correlate`, `eda_fit_model`, `eda_hypothesis_test`,
      `eda_reduce_dim`, `eda_cluster` compute real statistics from provided data
    - _Requirements: 11.5, 23.4_
  - [ ]* 11.3 Write property test — tool validation accepts valid and rejects invalid args
    - **Property 26: Tool validation accepts valid and rejects invalid arguments**
    - **Validates: Requirements 11.1**
  - [ ]* 11.4 Write property test — valid dispatch returns a well-formed envelope
    - **Property 27: Valid dispatch returns a well-formed envelope**
    - **Validates: Requirements 11.2**
  - [ ]* 11.5 Write property test — statistical tools compute real statistics vs NumPy
    - **Property 28: Statistical tools compute real statistics**
    - **Validates: Requirements 11.5**
  - [ ]* 11.6 Write edge tests — unknown-tool envelope + every-registered-tool sweep
    - Unknown tool returns an error envelope; every default-registry tool dispatches to a
      valid envelope for a representative valid input
    - _Requirements: 11.3, 11.4_

- [x] 12. Reasoning audit (`reasoning.py`, `cortex.py`, `curiosity.py`)
  - [x] 12.1 Audit and fix reasoning/tool-payload parsing + redundancy
    - Extract reasoning segments and final answer as separate fields; parse tool-call payloads
      into name + argument map; malformed input returns a parse-error indicator instead of
      raising; remove redundant token passes in `cortex.py`/`curiosity.py`
    - _Requirements: 12.1, 12.2, 12.3, 18.3_
  - [ ]* 12.2 Write property test — reasoning and tool-payload parsing round-trip
    - **Property 29: Reasoning and tool-payload parsing round-trip** (incl. malformed-input no-raise)
    - **Validates: Requirements 12.1, 12.2, 12.3**

- [ ] 13. Sampling & generation audit (`sampler.py`, `generate.py`)
  - [x] 13.1 Audit and fix the sampler and generation loop
    - Deterministic under a fixed seed; temperature/top-k/top-p restrict the sampled token
      set; final logit soft-cap bounds logits before sampling; non-finite logits routed
      through the NaN/Inf guard rather than sampled
    - _Requirements: 13.1, 13.2, 13.3, 13.4_
  - [ ]* 13.2 Write property test — sampling is deterministic under a fixed seed
    - **Property 30: Sampling is deterministic under a fixed seed**
    - **Validates: Requirements 13.1**
  - [ ]* 13.3 Write property test — sampling respects its parameters and soft cap
    - **Property 31: Sampling respects its parameters and soft cap**
    - **Validates: Requirements 13.2, 13.3**
  - [ ]* 13.4 Write edge test — non-finite logits routed through the guard
    - Non-finite logit tensor is reported via the NaN_Inf_Guard rather than sampled
    - _Requirements: 13.4_

- [x] 14. Loss audit (`inference/loss.py`)
  - [x] 14.1 Audit and fix loss computation and aggregation
    - Primary cross-entropy ignores padded positions (`ignore_index`); aggregate combines
      primary + MTP + VQ-commitment + MoE-balance losses by configured weights
    - _Requirements: 14.1, 14.2_
  - [ ]* 14.2 Write property test — loss ignores padding and aggregates by weight
    - **Property 32: Loss ignores padding and aggregates by weight**
    - **Validates: Requirements 14.1, 14.2**
  - [ ]* 14.3 Write gradient-flow smoke test under Tiny_Config
    - Backward yields finite gradients for all participating trainable params
    - _Requirements: 14.3_

- [x] 15. Optimizer audit (`train/optimizer.py`, `train/mopd.py`)
  - [x] 15.1 Audit and fix Muon Newton-Schulz + `mopd.py`
    - 5th-order Newton-Schulz orthogonalization with documented coefficients for the
      configured steps; non-2D params use the configured fallback optimizer; zero-Frobenius-
      norm momentum gets eps stabilization; apply the same eps discipline in `mopd.py`
    - _Requirements: 15.1, 15.3, 15.4_
  - [ ]* 15.2 Write property test — Muon orthogonalizes 2D updates
    - **Property 33: Muon orthogonalizes 2D updates** (singular values → 1.0)
    - **Validates: Requirements 15.1, 15.2**
  - [ ]* 15.3 Write edge tests — non-2D fallback + zero-norm eps stabilization
    - Non-2D parameter uses the fallback optimizer; zero-Frobenius-norm momentum avoids div-by-zero
    - _Requirements: 15.3, 15.4_

- [x] 16. Training pipeline + GRPO audit (`train/grpo_stability.py`, `reward.py`, `prepare_data.py`, `scheduler.py`, `inference/recovery.py`)
  - [x] 16.1 Audit and fix GRPO advantages, objective, and reward robustness
    - Group advantages normalized by group mean/std with eps; clipped probability ratio + KL
      penalty against the reference policy; malformed completions get a defined penalty (no
      unhandled exception)
    - _Requirements: 16.1, 16.2, 16.5_
  - [x] 16.2 Audit and fix data prep, scheduler, and recovery
    - `prepare_data.py` produces batches matching configured seq-len/batch; `scheduler.py` lr
      follows the configured warmup/decay; `inference/recovery.py` checkpoint recovery keeps
      state-shape consistency
    - _Requirements: 16.3, 16.4_
  - [ ]* 16.3 Write property test — GRPO advantages are group-normalized
    - **Property 34: GRPO advantages are group-normalized**
    - **Validates: Requirements 16.1**
  - [ ]* 16.4 Write property test — prepared batches and scheduler match configuration
    - **Property 35: Prepared batches and scheduler match configuration**
    - **Validates: Requirements 16.3, 16.4**
  - [ ]* 16.5 Write edge tests — GRPO clipped-ratio/KL objective + malformed-completion penalty
    - Clipped-ratio + KL objective computed as specified; malformed completion yields the
      defined penalty without raising
    - _Requirements: 16.2, 16.5_

- [x] 17. Encoding audit (`encoding/encoding_lasmoid.py`)
  - [x] 17.1 Audit and fix tokenizer round-trip and id range
    - Encode→decode reproduces in-vocabulary text; batch encoding yields ids within
      `[0, vocab_size)`
    - _Requirements: 17.1, 17.2_
  - [ ]* 17.2 Write property test — encoding round-trips and stays in range
    - **Property 36: Encoding round-trips and stays in range**
    - **Validates: Requirements 17.1, 17.2**

- [x] 18. Checkpoint — Ensure all per-subsystem tests pass
  - Run the CPU suite from `Lasmoid/`; ensure all tests pass, ask the user if questions arise.

- [x] 19. Redundant-work elimination pass (with `equiv_guard`)
  - [x] 19.1 Remove redundant/duplicated computation across audited modules
    - Capture seeded pre-refactor outputs, then vectorize Python loops, dedupe recomputed
      intermediates, and remove duplicate modules; each refactor must keep outputs within
      1e-5 and keep existing correctness tests green
    - _Requirements: 18.1, 18.2, 18.3, 18.4_
  - [ ]* 19.2 Write property test — refactors preserve outputs
    - **Property 37: Refactors preserve outputs**
    - **Validates: Requirements 18.1, 18.3**

- [x] 20. Reference-parity tests (vs `gemma`, `DeepSeek-V4-Pro`, `gpt-oss`)
  - [x] 20.1 Implement parity fixtures per the design's parity table
    - Compare RMSNorm (gemma, align `(1 + weight)` gamma convention), RoPE/YaRN (DeepSeek-V4-Pro,
      two fixtures: `original_seq_len == 0` and `> 0`), attention scores/softmax (gpt-oss,
      disable or match attention sinks on both sides), MLA, SSM ZOH decay, MoE Gate routing,
      Sinkhorn; report max|Δ| within documented tolerances; use closed-form self-consistency
      checks for intentional deviations (heavy-tail SSM, mHC, concept experts)
    - _Requirements: 21.1, 21.2, 21.3_
  - [ ]* 20.2 Write property test — reference-parity comparisons stay within tolerance
    - **Property 38: Reference-parity comparisons stay within tolerance**
    - **Validates: Requirements 21.1, 21.2**

- [x] 21. Author GPU scripts (AUTHOR-ONLY — DO NOT EXECUTE)
  - [x] 21.1 Author `scripts/gpu_train.py` (author-only / do-not-execute)
    - Accepts `--config <path>` and `--device <cuda:N>`; wraps the existing `train/pretrain.py`
      loop; inline usage docstring documenting required args and hardware assumptions. Do NOT
      run on CPU/MPS — authored for the user's GPU.
    - _Requirements: 20.1, 20.5_
  - [x] 21.2 Author `scripts/gpu_generate.py` (author-only / do-not-execute)
    - Loads a checkpoint, takes `--prompt`, produces sampled output via `inference/generate.py`
      + `sampler.py`; inline usage docstring. Do NOT execute in this iteration.
    - _Requirements: 20.2, 20.5_
  - [x] 21.3 Author `scripts/gpu_benchmark.py` (author-only / do-not-execute)
    - Reports tokens/sec throughput and peak memory; inline usage docstring. Do NOT execute in
      this iteration.
    - _Requirements: 20.3, 20.5_
  - [ ]* 21.4 Static syntax-check the GPU scripts (`py_compile` only — DO NOT RUN)
    - Run `python -m py_compile` on each authored script; never execute the scripts themselves
    - _Requirements: 20.4_

- [x] 22. Record breaking changes in `MIGRATIONS.md`
  - [x] 22.1 Create `MIGRATIONS.md` at the spec root with entries for all breaking changes
    - Use the design's entry format (Change / Reason / Impact / Migration); seed with the
      mHC Sinkhorn `20 → 8` config-driven change, the mHC dead-code consolidation, and the
      SSM chunk-scan semantics clarification
    - _Requirements: 22.1, 22.2_

- [x] 23. Final checkpoint — full CPU suite + ruff
  - Run the full CPU correctness/smoke/parity suite from `Lasmoid/`
    (`DYLD_LIBRARY_PATH=$(brew --prefix expat)/lib python3 -m unittest test_model_parts -v`
    and `python3 -m pytest inference/ -v`), then `ruff check .`. Confirm all gating criteria
    (Req 23): tests pass, parity within tolerance, no Faked_Math, all experts activate, all
    tools dispatch, multimodal fusion operates. Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster pass;
  core implementation sub-tasks are never optional.
- Every output-changing fix is gated behind a `ModelArgs` config flag wired through
  `inference/config.py` with a conservative default preserving current behavior (hard
  project convention), unless the existing behavior is outright broken.
- Each task references the specific requirement clauses it satisfies and, where applicable,
  the design's numbered correctness property.
- All verification runs on CPU under `Tiny_Config`; GPU scripts (task 21) are authored and
  `py_compile`-checked only — never executed in this iteration.
- Checkpoints (tasks 10, 18, 23) provide incremental validation; `MIGRATIONS.md` (task 22)
  records all breaking changes per Req 22.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1", "3.1", "3.3", "4.1", "5.1", "6.1", "7.1", "8.1", "8.2", "8.3", "8.4", "9.1", "9.2", "11.1", "12.1", "13.1", "14.1", "15.1", "16.1", "16.2", "17.1"] },
    { "id": 3, "tasks": ["2.2", "3.2", "4.2", "5.2", "7.2", "9.3", "11.2"] },
    { "id": 4, "tasks": ["2.3", "7.3"] },
    { "id": 5, "tasks": ["2.4", "3.4", "4.3", "5.3", "6.2", "7.4", "8.5", "9.4", "11.3", "12.2", "13.2", "14.2", "15.2", "16.3", "17.2"] },
    { "id": 6, "tasks": ["2.5", "3.5", "4.4", "5.4", "6.3", "7.5", "8.6", "9.5", "11.4", "13.3", "14.3", "15.3", "16.4"] },
    { "id": 7, "tasks": ["2.6", "3.6", "4.5", "5.5", "7.6", "8.7", "9.6", "11.5", "13.4", "16.5"] },
    { "id": 8, "tasks": ["3.7", "4.6", "7.7", "8.8", "9.7", "11.6"] },
    { "id": 9, "tasks": ["3.8"] },
    { "id": 10, "tasks": ["19.1", "20.1", "21.1", "21.2", "21.3", "22.1"] },
    { "id": 11, "tasks": ["19.2", "20.2", "21.4"] }
  ]
}
```
