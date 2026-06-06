"""
Lasmod — OMP-based KV cache compaction (Phase A4).

Uses Orthogonal Matching Pursuit to greedily select a subset of KV
positions that best approximate full attention output. Returns compacted
(C1, beta, C2) representation.

Reference: OMP compaction (ICLR 2026)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class OMPCompactionConfig:
    enabled: bool = False
    target_ratio: float = 0.5
    k_choice: int = 1
    ridge_lambda: float = 0.0
    beta_method: str = "nnls"
    nnls_iters: int = 0


def _compute_exp_scores(
    k: torch.Tensor, queries: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = k.shape[1]
    inv = (1.0 / d) ** 0.5
    s = (queries @ k.T).to(torch.float32) * inv
    mx = s.max(dim=1, keepdim=True)[0]
    ex = torch.exp(s - mx)
    return ex, ex.sum(dim=1)


def _nnls_solve(m: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if m.shape[1] == 0:
        return torch.zeros(0, dtype=torch.float32, device=m.device)
    sol = torch.linalg.lstsq(m, target.unsqueeze(1)).solution
    return sol[: m.shape[1], 0].clamp(min=1e-12).to(torch.float32)


def omp_select_indices(
    k: torch.Tensor,
    queries: torch.Tensor,
    config: OMPCompactionConfig,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Select KV positions via OMP greedy selection.

    Returns (selected_indices, beta_weights).
    """
    seq_len = k.shape[0]
    target_size = max(1, min(int(seq_len * config.target_ratio), seq_len))
    device = k.device
    n_q = queries.shape[0]

    exp_s, target = _compute_exp_scores(k, queries)

    current = torch.zeros(n_q, dtype=torch.float32, device=device)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    selected: list = []

    while len(selected) < target_size:
        residual = target - current
        corr = (exp_s * residual.unsqueeze(1)).sum(dim=0)
        corr[mask] = -float("inf")
        if not (corr > -float("inf")).any():
            break

        keep = min(config.k_choice, target_size - len(selected))
        _, top = torch.topk(corr, keep, largest=True)
        for idx in top.cpu().tolist():
            if not mask[idx]:
                mask[idx] = True
                selected.append(idx)

        if not selected:
            break
        sel_t = torch.tensor(selected, dtype=torch.long, device=device)
        m = exp_s[:, sel_t]
        beta = _nnls_solve(m, target)
        current = m @ beta

    if not selected:
        return torch.zeros(0, dtype=torch.long, device=device), torch.zeros(
            0, device=device
        )

    sel_final = torch.tensor(sorted(selected), dtype=torch.long, device=device)
    m_final = exp_s[:, sel_final]
    beta_final = _nnls_solve(m_final, target)
    return sel_final, beta_final


def omp_compact(
    k: torch.Tensor,
    v: torch.Tensor,
    queries: torch.Tensor,
    config: OMPCompactionConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact KV cache using OMP greedy key selection.

    Returns (C1, beta_log, C2) where:
      C1 = selected keys  (batch, keep, head_dim)
      beta_log = log(beta)
      C2 = merged values (batch, keep, head_dim)
    """
    B, S, D = k.shape
    device = k.device

    C1_list, beta_list, C2_list = [], [], []

    for b in range(B):
        sel, w = omp_select_indices(k[b], queries[b], config)
        keep = sel.shape[0]
        if keep == 0:
            C1_list.append(torch.zeros(0, D, device=device))
            beta_list.append(torch.zeros(0, device=device))
            C2_list.append(torch.zeros(0, D, device=device))
            continue

        k_sel = k[b, sel]
        v_sel = v[b, sel]

        C2 = v_sel
        C1_list.append(k_sel.unsqueeze(0))
        beta_list.append(torch.log(w.clamp(min=1e-12)).unsqueeze(0))
        C2_list.append(C2.unsqueeze(0))

    return (
        torch.cat(C1_list, dim=0),
        torch.cat(beta_list, dim=0),
        torch.cat(C2_list, dim=0),
    )
