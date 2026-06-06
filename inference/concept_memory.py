"""
Elastic Sparse Concept Memory — HCM / Causal Dynamic Memory
=============================================================
Extracted from monolithic model.py during Phase 0 SOLiD refactoring.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple

try:
    from ._common import Linear
    from .vq import ResidualVQ
    from .config import ModelArgs
except ImportError:
    from _common import Linear
    from vq import ResidualVQ
    from config import ModelArgs


class ElasticSparseConceptMemory(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.num_concepts = args.num_concepts
        self.num_abstract_concepts = args.num_abstract_concepts
        self.num_global_concepts = args.num_global_concepts
        self.total_concepts = (
            args.num_concepts + args.num_abstract_concepts + args.num_global_concepts
        )

        self.hcm_ema_alpha = args.hcm_ema_alpha
        self.entropy_threshold = args.entropy_threshold
        self.max_blocks = 16

        self.episodic_queries = nn.Parameter(
            torch.empty(1, args.num_concepts, args.dim)
        )
        self.semantic_queries = nn.Parameter(
            torch.empty(1, args.num_abstract_concepts, args.dim)
        )
        self.global_queries = nn.Parameter(
            torch.empty(1, args.num_global_concepts, args.dim)
        )

        nn.init.normal_(self.episodic_queries, 0.0, 0.02)
        nn.init.normal_(self.semantic_queries, 0.0, 0.02)
        nn.init.normal_(self.global_queries, 0.0, 0.02)

        self.concept_blocks = nn.ModuleList([self._create_block(args)])

        self.register_buffer(
            "slot_ema", torch.zeros(1, self.total_concepts, args.dim), persistent=True
        )
        self.register_buffer(
            "slot_db", torch.zeros(1, self.total_concepts, args.dim), persistent=True
        )
        self.register_buffer(
            "meta_centroids", torch.zeros(1, args.dim), persistent=False
        )

        self.lightning_indexer = Linear(args.dim, args.dim)

    def _create_block(self, args: ModelArgs):
        return ResidualVQ(
            codebook_size=args.codebook_size,
            dim=args.dim,
            commitment_cost=args.hcm_commit_loss_coeff,
        )

    def process_chunk(
        self, encoder_hidden: torch.Tensor, block_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_enc, D = encoder_hidden.shape

        # Perceiver pooling heads
        n_heads = 8
        head_dim = self.dim // n_heads
        KV_4d = encoder_hidden.view(B, N_enc, n_heads, head_dim).transpose(1, 2)

        # 1. Pool episodic
        Q_epi = self.episodic_queries.expand(B, -1, -1)
        Q_epi_4d = (
            Q_epi.view(B, self.num_concepts, n_heads, head_dim)
            .transpose(1, 2)
            .to(encoder_hidden.dtype)
        )
        pooled_epi = (
            F.scaled_dot_product_attention(Q_epi_4d, KV_4d, KV_4d)
            .transpose(1, 2)
            .contiguous()
            .view(B, self.num_concepts, self.dim)
        )

        # 2. Pool semantic
        Q_sem = self.semantic_queries.expand(B, -1, -1)
        Q_sem_4d = (
            Q_sem.view(B, self.num_abstract_concepts, n_heads, head_dim)
            .transpose(1, 2)
            .to(encoder_hidden.dtype)
        )
        pooled_sem = (
            F.scaled_dot_product_attention(Q_sem_4d, KV_4d, KV_4d)
            .transpose(1, 2)
            .contiguous()
            .view(B, self.num_abstract_concepts, self.dim)
        )

        # 3. Pool global
        Q_glo = self.global_queries.expand(B, -1, -1)
        Q_glo_4d = (
            Q_glo.view(B, self.num_global_concepts, n_heads, head_dim)
            .transpose(1, 2)
            .to(encoder_hidden.dtype)
        )
        pooled_glo = (
            F.scaled_dot_product_attention(Q_glo_4d, KV_4d, KV_4d)
            .transpose(1, 2)
            .contiguous()
            .view(B, self.num_global_concepts, self.dim)
        )

        # Concatenate slots
        pooled = torch.cat([pooled_epi, pooled_sem, pooled_glo], dim=1)

        quantized, loss, indices, adj = self.concept_blocks[0](pooled)

        with torch.no_grad():
            mean_quant = quantized.detach().mean(dim=0, keepdim=True)
            self.slot_ema.data.copy_(
                self.hcm_ema_alpha * self.slot_ema.data
                + (1.0 - self.hcm_ema_alpha) * mean_quant
            )
            self.slot_db.data[0].copy_(self.slot_ema.data[0])
            self.meta_centroids.data[0].copy_(torch.mean(self.slot_db[0], dim=0))

        return quantized, loss

    def lightning_retrieve(
        self, decoder_query: torch.Tensor, top_k_blocks: int = 1
    ) -> torch.Tensor:
        proj_q = self.lightning_indexer(decoder_query)
        pooled_q = torch.mean(proj_q, dim=1)

        # Safe normalize L2
        proj_q_norm = pooled_q / (pooled_q.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        centroids_norm = self.meta_centroids.to(proj_q_norm.dtype)
        centroids_norm = centroids_norm / (
            centroids_norm.norm(p=2, dim=-1, keepdim=True) + 1e-8
        )

        sim = torch.matmul(proj_q_norm, centroids_norm.t())
        k = min(top_k_blocks, self.meta_centroids.size(0))
        topk_scores, topk_idxs = torch.topk(sim, k, dim=-1)

        gathered = self.slot_db.to(decoder_query.dtype)[topk_idxs]
        concept_db = gathered.flatten(1, 2)
        return concept_db
