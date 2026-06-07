"""
Lasmoid — train/run_ablations.py
======================================================================
Automated train → evaluate → record loop over feature-flag variants.

For each variant in the sweep:
  1. build a (tiny) Lasmoid from base config + the variant's overrides
  2. optionally run a few SGD steps on real token data
  3. evaluate (perplexity, tool success, cortex/adaptive/curiosity diagnostics)
  4. append one JSON row to results.jsonl  (resumable; skips completed variants)

This is the engine that converts "is this feature fluff?" into a measured number.
On a real corpus + more steps it produces a trustworthy leaderboard; on the tiny
fixture it proves the pipeline end-to-end.

Usage:
  python train/run_ablations.py --sweep train/sweep.json --train-steps 20 \
      --data train/tinystories_train.bin --val train/tinystories_val.bin
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_INF_DIR = os.path.join(os.path.dirname(_HERE), "inference")
sys.path.insert(0, _INF_DIR)
sys.path.insert(0, _HERE)

from config import ModelArgs  # noqa: E402
from lasmoid import Lasmoid  # noqa: E402
from loss import compute_loss  # noqa: E402
from optimizer import build_optimizers, clip_grad_global_norm  # noqa: E402
import evaluate as ev  # noqa: E402


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def quick_train(model, tokens, steps, seq_len, batch, lr, device):
    """A few steps of the *full* objective (CE + every auxiliary the model
    produces) so all components — experts, cortex, curiosity, concept memory —
    actually receive gradient. Uses the Muon+AdamW split."""
    if tokens is None or steps <= 0:
        return
    model.train()
    opts = build_optimizers(model, muon_lr=lr * 6, adamw_lr=lr)
    N = tokens.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (min(batch, N),))
        x = tokens[idx].to(device)
        out = model(x, x, start_pos=0)
        logits, _mtp, _cdb, _mem, routing_maps, _ci, adjs, event_probs = out
        loss = compute_loss(
            logits[:, :-1],
            x[:, 1:],
            routing_maps,
            [model.last_vq_loss],
            adjs,
            event_probs,
            moe_aux_loss=model.last_moe_loss,
            commit_loss=getattr(model, "last_commit_loss", None),
            curiosity_loss=getattr(model, "last_curiosity_loss", None),
            label_smoothing=0.05,
        )
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_global_norm(model, 1.0)
        for o in opts:
            o.step()


def run(sweep_path, data_path, val_path, train_steps, out_path, lr):
    device = _device()
    with open(sweep_path) as f:
        sweep = json.load(f)
    base = sweep["base"]

    # Size vocab to the data so the tiny model stays small and index-safe.
    seq_len = base.get("max_seq_len", 128)
    train_tok = ev.load_tokens(data_path, seq_len) if data_path else None
    val_tok = ev.load_tokens(val_path, seq_len) if val_path else None
    if val_tok is None:  # empty/missing val → reuse train tokens for the smoke metric
        val_tok = train_tok
    vocab = 256
    for t in (train_tok, val_tok):
        if t is not None:
            vocab = max(vocab, int(t.max().item()) + 1)

    # Resume support: skip variants already in the results file.
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["variant"])
                except Exception:
                    pass

    results = []
    for v in sweep["variants"]:
        name = v["name"]
        if name in done:
            print(f"[skip] {name} (already recorded)")
            continue
        cfg = dict(base)
        cfg.update(v.get("overrides", {}))
        cfg["vocab_size"] = vocab
        args = ModelArgs(**cfg)
        torch.manual_seed(0)
        print(f"[run ] {name} ...")
        model = Lasmoid(args).to(device)
        quick_train(model, train_tok, train_steps, seq_len, args.max_batch_size, lr, device)
        metrics = ev.evaluate_model(model, args, val_tok, device)
        n_params = sum(p.numel() for p in model.parameters())
        row = {"variant": name, "params": n_params, **metrics}
        results.append(row)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"        {json.dumps(metrics)}")
        del model
    print(f"\nWrote results → {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=os.path.join(_HERE, "sweep.json"))
    ap.add_argument("--data", default=os.path.join(_HERE, "tinystories_train.bin"))
    ap.add_argument("--val", default=os.path.join(_HERE, "tinystories_val.bin"))
    ap.add_argument("--train-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(_HERE, "ablation_results.jsonl"))
    a = ap.parse_args()
    run(a.sweep, a.data, a.val, a.train_steps, a.out, a.lr)
