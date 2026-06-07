"""
Lasmoid — train/report.py
======================================================================
Turn ablation_results.jsonl into a readable markdown leaderboard and an
automatic keep/cut verdict per feature (Δ perplexity vs. the baseline).

Usage:
  python train/report.py --results train/ablation_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_report(rows):
    if not rows:
        return "No results.\n"
    # Column union (stable order)
    cols = ["variant", "params", "perplexity", "toolcall_success", "avg_experts", "domain_entropy", "curiosity"]
    cols = [c for c in cols if any(c in r for r in rows)]

    lines = ["# Ablation Leaderboard\n", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    by_name = {r["variant"]: r for r in rows}
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "—")) for c in cols) + " |")

    # Verdict vs baseline on perplexity (lower is better).
    base = by_name.get("baseline")
    if base and "perplexity" in base:
        lines.append("\n## Verdict (Δ perplexity vs. baseline; lower is better)\n")
        bp = base["perplexity"]
        lines.append("| feature | Δ ppl | verdict |")
        lines.append("|---|---|---|")
        for r in rows:
            if r["variant"] == "baseline" or "perplexity" not in r:
                continue
            d = r["perplexity"] - bp
            verdict = "KEEP (helps)" if d < -1e-3 else ("CUT (no help)" if d > 1e-3 else "NEUTRAL")
            lines.append(f"| {r['variant']} | {d:+.4f} | {verdict} |")
    lines.append(
        "\n> Note: on the tiny fixture these numbers only prove the pipeline runs. "
        "Run on a real corpus with more --train-steps and multiple seeds for a "
        "trustworthy verdict.\n"
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results", default=os.path.join(here, "ablation_results.jsonl"))
    ap.add_argument("--out", default=os.path.join(here, "ablation_report.md"))
    a = ap.parse_args()
    rows = load_rows(a.results)
    report = build_report(rows)
    with open(a.out, "w") as f:
        f.write(report)
    print(report)
