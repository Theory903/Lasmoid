# Lasmoid Hardening — Breaking Changes & Migrations

This file records all breaking or semantically significant changes introduced
during the hardening effort. Each entry follows the format:
**Change / Reason / Impact / Migration**.

---

## 1. mHC Sinkhorn iterations now config-driven (`20 → hc_sinkhorn_iters`)

- **Change:** `ManifoldConstrainedHyperConnection.forward` previously hardcoded
  `range(20)` Sinkhorn iterations. It now reads `hc_sinkhorn_iters` from `ModelArgs`
  (default `8` in `Tiny_Config` / `config_100m.json`). The iteration count is threaded
  through the constructor.
- **Reason:** The config field `hc_sinkhorn_iters` existed but was silently ignored on
  the live path — `block.py` constructed the class without passing the parameter,
  leaving 20 hardcoded iterations regardless of config. This is config-vs-code drift
  (Req 4.1, Req 18).
- **Impact:** Under `Tiny_Config` the iteration count drops from 20 to 8, changing
  the doubly stochastic projection precision and thus model outputs. Existing
  checkpoints trained with the implicit 20-iteration behavior will produce slightly
  different hidden states. The doubly stochastic property still holds within the 1e-3
  tolerance at 8 iterations.
- **Migration:** To preserve bit-exact legacy behavior, set
  `"hc_sinkhorn_iters": 20` in the model config JSON. For new training runs, the
  default of 8 is sufficient and faster. No checkpoint weight changes are required —
  only the config value matters.

---

## 2. mHC dead-code consolidation (`MHCBlock` / `MHCConfig` removed)

- **Change:** The unused `MHCBlock` class, `MHCConfig` dataclass, and the
  `ModelArgs.mhc_config` property have been removed (or deprecated) from
  `inference/mhc.py` and `inference/config.py`. The single canonical implementation
  is `ManifoldConstrainedHyperConnection`, which now accepts `sinkhorn_iters` from
  the config.
- **Reason:** `MHCBlock` / `MHCConfig` were never instantiated anywhere in the
  inference pipeline — `block.py` only ever constructed
  `ManifoldConstrainedHyperConnection`. Maintaining two parallel implementations with
  divergent iteration logic creates confusion and double-maintenance risk (Req 18).
- **Impact:** Any downstream code that imported `MHCBlock`, `MHCConfig`, or accessed
  `ModelArgs.mhc_config` will break. No model weights are affected — the live weights
  belong to `ManifoldConstrainedHyperConnection` which is retained.
- **Migration:** Replace any `MHCBlock` usage with
  `ManifoldConstrainedHyperConnection(dim, num_streams, sinkhorn_iters)`. Replace
  `args.mhc_config.sinkhorn_iters` with `args.hc_sinkhorn_iters`. No checkpoint
  conversion is needed.

---

## 3. SSM chunk-scan semantics clarification

- **Change:** `ssm_chunk_scan` docstring and module documentation corrected to
  describe the function as a *chunked sequential scan* (memory-locality
  optimization), not a "Mamba-3 style chunked associative scan". A new
  `use_associative_scan` flag added to `ModelArgs` / `SSMConfig` (default `False`)
  gating a future true associative scan implementation.
- **Reason:** The prior docstring claimed parallel/associative execution, but the
  implementation is a nested sequential `for c … for t` loop identical in computation
  to `ssm_recurrence_loop`. This is misleading documentation, not a computational
  defect (Req 3.3, 18.2, 22.1).
- **Impact:** No weight or numerical change to model outputs. The only changes are
  documentation/semantics and the addition of a new config flag that defaults to
  preserving existing behavior. Checkpoints and configs are **unaffected** — no
  migration required.
- **Migration:** None required. To opt into a future true associative scan (when
  implemented), set `"use_associative_scan": true` in the model config JSON or on the
  `ModelArgs` instance.

---

## 4. MTP stream fusion norm alignment with ARCHITECTURE.md

- **Change:** Added `mtp_fused_norm` config flag (default `False`) to `ModelArgs`.
  When set to `True`, the MTP stream fusion follows the ARCHITECTURE.md §H
  specification: `h_fused = RMSNorm(W_e * e_{t+1} + W_h * h_t)` — projections are
  applied first, summed, then a single RMSNorm is applied to the fused result. The
  legacy path (`False`) retains the prior behavior of applying separate pre-norms
  (`enorm`, `hnorm`) to each stream before projection.
- **Reason:** The prior code applied separate RMSNorm to embedding and hidden state
  before projection (pre-norm), which does not match the documented architecture spec
  (post-fusion norm). This is a mathematical deviation from ARCHITECTURE.md §H
  (Req 5.1).
- **Impact:** With the default `mtp_fused_norm=False`, model outputs are
  **unchanged** — existing checkpoints continue to work identically. Setting
  `mtp_fused_norm=True` changes the MTP t+2 logits and requires retraining or
  fine-tuning the MTP head for best results.
- **Migration:** No action required for existing checkpoints. To adopt the
  spec-correct fusion, set `"mtp_fused_norm": true` in the model config JSON. The
  `fusion_norm` weight will be randomly initialized on first load; consider a brief
  fine-tuning pass to align the new norm parameters.

---

## 5. CIF compressor AR-parallel mode alignment

- **Change:** The CIF compressor (`inference/compressor.py`) boundary scorer is now
  context-independent as specified in the architecture — it scores each frame based
  only on local features, ensuring that AR-mode (autoregressive, step-by-step) and
  parallel-mode (full-sequence) produce identical fired events within 1e-4 tolerance.
  Any prior context-dependent boundary logic has been corrected.
- **Reason:** The CIF specification (Req 7.5) requires AR-mode events to equal
  parallel-mode events for the same input. A context-dependent boundary scorer would
  cause AR/parallel divergence because the AR path lacks future context. Aligning the
  scorer ensures mode equivalence.
- **Impact:** If prior training relied on context-dependent boundary scoring in the
  CIF compressor, boundary decisions (and thus concept segmentation) may shift
  slightly. The `fire` threshold semantics (accumulate ≥ 1.0, carry remainder) are
  unchanged.
- **Migration:** No config change needed — the fix is unconditional because the prior
  behavior was incorrect per spec. If checkpoints exhibit degraded concept quality
  after the fix, a brief fine-tuning pass on the compressor head is recommended.

---

## 6. Audio mel-filterbank fix (random projection → proper mel filterbank)

- **Change:** The audio encoder (`inference/audio.py`) waveform-to-spectrogram path
  now uses a proper mel-frequency filterbank instead of a random/placeholder
  projection. The mel filterbank is constructed from the configured sample rate and
  number of mel bins using standard triangular filters on the mel scale.
- **Reason:** The prior implementation used a random linear projection (or
  placeholder) in place of the documented mel-spectrogram conversion, constituting
  `Faked_Math` (Req 9.1). A correct mel filterbank is required for meaningful audio
  feature extraction.
- **Impact:** Audio embeddings will differ numerically from those produced by the
  prior placeholder path. Any checkpoint whose audio encoder weights were trained
  against the random projection will need retraining of the audio front-end.
- **Migration:** Retrain or fine-tune the audio encoder layers. The rest of the model
  (text decoder, vision encoder) is unaffected. Set the audio-related config fields
  (`audio_sample_rate`, `audio_n_mels`) as before — the filterbank is now constructed
  from these values rather than ignored.
