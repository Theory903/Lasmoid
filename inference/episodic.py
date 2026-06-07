"""
Lasmoid — episodic.py
======================================================================
EpisodicMemory: cluster a long stream of context windows into topical
*episodes* and route a query to the most relevant episode(s).

Adapted from ml-epicache (ClusterManager): embed fixed-size windows, k-means
cluster them into episodes, rank windows by cosine-to-centroid, and retrieve a
bounded working set for an incoming query.  Combined with the existing tiered-KV
/ SnapKV eviction / OMP compaction stack, this gives Lasmoid bounded working
memory over very long data/observation streams.

Dependency-light: torch only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch


def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def kmeans(
    x: torch.Tensor, k: int, iters: int = 25, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's k-means. Returns (labels [N], centroids [k, D])."""
    n = x.shape[0]
    k = max(1, min(int(k), n))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)[:k]
    centroids = x[perm].clone()
    labels = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(int(iters)):
        d2 = torch.cdist(x, centroids) ** 2
        new = d2.argmin(dim=1)
        if torch.equal(new, labels):
            labels = new
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centroids[c] = x[m].mean(dim=0)
    return labels, centroids


@dataclass
class Episode:
    episode_id: int
    member_indices: List[int]
    centroid: torch.Tensor
    payloads: List[Any] = field(default_factory=list)


class EpisodicMemory:
    """Topical episodic memory over embedded context windows.

    Usage:
        mem = EpisodicMemory(n_episodes=8)
        mem.build(window_embeddings, payloads=window_texts)
        ep_ids = mem.retrieve(query_embedding, top_k=2)
        ctx    = mem.working_set(query_embedding, top_k=2, max_windows=16)
    """

    def __init__(self, n_episodes: int = 8, seed: int = 0):
        self.n_episodes = n_episodes
        self.seed = seed
        self.episodes: List[Episode] = []
        self._centroids: Optional[torch.Tensor] = None  # [E, D] L2-normalised
        self._embeddings: Optional[torch.Tensor] = None  # [N, D]
        self._payloads: List[Any] = []

    def build(self, embeddings: torch.Tensor, payloads: Optional[List[Any]] = None) -> "EpisodicMemory":
        """Build episodic memory from window embeddings.

        Shape contract:
          embeddings: [N, D] — N context window embeddings of dimension D
          payloads:   list of length N (or None → range(N))
          After build: self._embeddings is [N, D], self._centroids is [E, D] (L2-normed)
        """
        emb = torch.as_tensor(embeddings, dtype=torch.float32)
        if emb.ndim != 2:
            raise ValueError("embeddings must be 2D [N, D]")
        n = emb.shape[0]
        payloads = payloads or list(range(n))
        if len(payloads) != n:
            raise ValueError("payloads length must match number of embeddings")

        labels, centroids = kmeans(emb, self.n_episodes, seed=self.seed)
        self._embeddings = emb
        self._payloads = list(payloads)
        self._centroids = _l2norm(centroids)

        self.episodes = []
        for c in range(centroids.shape[0]):
            members = (labels == c).nonzero(as_tuple=True)[0].tolist()
            self.episodes.append(
                Episode(
                    episode_id=c,
                    member_indices=members,
                    centroid=centroids[c],
                    payloads=[payloads[i] for i in members],
                )
            )
        return self

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 1) -> List[int]:
        """Return episode ids nearest to the query by cosine similarity."""
        if self._centroids is None:
            return []
        q = _l2norm(torch.as_tensor(query_embedding, dtype=torch.float32).reshape(1, -1))
        sim = (q @ self._centroids.T).squeeze(0)  # [E]
        k = max(1, min(int(top_k), sim.shape[0]))
        return sim.topk(k).indices.tolist()

    def working_set(
        self, query_embedding: torch.Tensor, top_k: int = 2, max_windows: int = 16
    ) -> List[Any]:
        """Bounded working set: payloads from the nearest episodes, ranked by
        cosine-to-centroid and capped at ``max_windows`` (prevents context
        overflow over long sessions)."""
        if self._embeddings is None:
            return []
        ep_ids = self.retrieve(query_embedding, top_k=top_k)

        # Vectorized: gather all candidate indices and their centroids, compute
        # cosine similarities in a single batched operation instead of per-member
        # Python loops (Req 18.2, 18.3).
        all_indices: List[int] = []
        centroid_for_idx: List[torch.Tensor] = []
        for eid in ep_ids:
            ep = self.episodes[eid]
            cen = _l2norm(ep.centroid.reshape(1, -1))  # [1, D]
            for idx in ep.member_indices:
                all_indices.append(idx)
                centroid_for_idx.append(cen)

        if not all_indices:
            return []

        # Batch-normalize all member embeddings at once
        idx_tensor = torch.tensor(all_indices, dtype=torch.long)
        member_embs = _l2norm(self._embeddings[idx_tensor])  # [M, D]

        # Stack centroids and batch-compute dot products
        centroids_stacked = torch.cat(centroid_for_idx, dim=0)  # [M, D]
        sims = (member_embs * centroids_stacked).sum(dim=-1)  # [M]

        # Sort by similarity descending, cap at max_windows
        order = sims.argsort(descending=True)
        result: List[Any] = []
        for i in order[:max_windows].tolist():
            result.append(self._payloads[all_indices[i]])
        return result


__all__ = ["EpisodicMemory", "Episode", "kmeans"]
