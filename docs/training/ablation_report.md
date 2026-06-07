# Ablation Leaderboard

| variant | params | perplexity | toolcall_success | avg_experts | domain_entropy | curiosity |
|---|---|---|---|---|---|---|
| baseline | 8481095 | 54840.6239 | 1.0000 | 2.0000 | — | — |
| domain_cortex | 8482279 | 54366.4860 | 1.0000 | 2.0000 | 1.3862 | — |
| adaptive_k | 8481095 | 54893.4798 | 1.0000 | 4.0000 | — | — |
| relational | 8483367 | 60458.6635 | 1.0000 | 2.0000 | — | — |
| curiosity | 8518087 | 55520.3311 | 1.0000 | 2.0000 | — | 0.0008 |
| all_on | 8521543 | 55372.5330 | 1.0000 | 4.0000 | 1.3863 | 0.0008 |

## Verdict (Δ perplexity vs. baseline; lower is better)

| feature | Δ ppl | verdict |
|---|---|---|
| domain_cortex | -474.1378 | KEEP (helps) |
| adaptive_k | +52.8560 | CUT (no help) |
| relational | +5618.0396 | CUT (no help) |
| curiosity | +679.7072 | CUT (no help) |
| all_on | +531.9091 | CUT (no help) |

> Note: on the tiny fixture these numbers only prove the pipeline runs. Run on a real corpus with more --train-steps and multiple seeds for a trustworthy verdict.

