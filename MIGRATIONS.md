# Lasmoid — Breaking Changes & Migrations

This file records all breaking or semantically significant changes introduced
during the hardening effort. Each entry follows the format:
Change / Reason / Impact / Migration.

---

## 2025-07-13 — ssm.py: Chunk-scan semantics clarification

- **Change:** `ssm_chunk_scan` docstring and module documentation corrected to
  describe the function as a *chunked sequential scan* (memory-locality
  optimisation), not a "Mamba-3 style chunked associative scan". A new
  `use_associative_scan` flag added to `ModelArgs` / `SSMConfig` (default
  `False`) gating a future true associative scan implementation.
- **Reason:** The prior docstring claimed parallel/associative execution, but
  the implementation is a nested sequential `for c … for t` loop identical in
  computation to `ssm_recurrence_loop`. This is misleading documentation, not a
  computational defect (Req 3.3, 18.2, 22.1).
- **Impact:** No weight or numerical change to model outputs. The only changes
  are documentation/semantics and the addition of a new config flag that
  defaults to preserving existing behaviour. Checkpoints and configs are
  **unaffected** — no migration required.
- **Migration:** None required. To opt into a future true associative scan
  (when implemented), set `"use_associative_scan": true` in the model config
  JSON or on the `ModelArgs` instance.

---

## 2025-07-13 — mtp.py: Stream fusion norm alignment with ARCHITECTURE.md

- **Change:** Added `mtp_fused_norm` config flag (default `False`) to
  `ModelArgs`. When set to `True`, the MTP stream fusion follows the
  ARCHITECTURE.md §H specification: `h_fused = RMSNorm(W_e * e_{t+1} + W_h * h_t)`
  — projections are applied first, summed, then a single RMSNorm is applied to
  the fused result. The legacy path (`False`) retains the prior behavior of
  applying separate pre-norms (`enorm`, `hnorm`) to each stream before
  projection.
- **Reason:** The prior code applied separate RMSNorm to embedding and hidden
  state before projection (pre-norm), which does not match the documented
  architecture spec (post-fusion norm). This is a mathematical deviation from
  ARCHITECTURE.md §H (Req 5.1).
- **Impact:** With the default `mtp_fused_norm=False`, model outputs are
  **unchanged** — existing checkpoints continue to work identically. Setting
  `mtp_fused_norm=True` changes the MTP t+2 logits and requires retraining or
  fine-tuning the MTP head for best results.
- **Migration:** No action required for existing checkpoints. To adopt the
  spec-correct fusion, set `"mtp_fused_norm": true` in the model config JSON.
  The `fusion_norm` weight will be randomly initialized on first load; consider
  a brief fine-tuning pass to align the new norm parameters.

---

## 2025-07-13 — mhc.py: Dead code consolidation + single-stream shortcut

- **Change:** Removed the dead `MHCBlock` class from `inference/mhc.py` and the
  `MHCConfig` dataclass + `mhc_config` property from `inference/config.py`.
  Added a `stream_count == 1` fast path in `ManifoldConstrainedHyperConnection`
  that reduces to plain residual addition (identity gating) with no Sinkhorn
  overhead.
- **Reason:** `MHCBlock`, `MHCConfig`, and the `mhc_config` property were never
  used in the active `LasmoidBlock` path — `block.py` exclusively constructs
  `ManifoldConstrainedHyperConnection`. Maintaining two implementations with
  different parametrisations (one using `hc_split_sinkhorn` kernel, the other
  using inline Sinkhorn-Knopp) created confusion and double maintenance burden
  (Req 4.3, 4.4, 18.3, 22.2).
- **Impact:** No weight or numerical change for existing configs where
  `num_residual_streams >= 2` (all shipped configs use 4). The `MHCBlock` class
  was never instantiated on the active path so its removal cannot break any
  existing checkpoint. Code that imports `MHCBlock` or `MHCConfig` from
  `inference/mhc.py` / `inference/config.py` will get an `ImportError`; this
  only affects out-of-tree code that was using the dead path.
- **Migration:** If any downstream script was importing `MHCBlock`, switch to
  `ManifoldConstrainedHyperConnection` (the canonical class). The old
  `hc_split_sinkhorn` kernel function remains in `kernel.py` for potential
  future use but is no longer imported by `mhc.py`. To exercise the single-
  stream shortcut, set `"num_residual_streams": 1` in the model config JSON.
