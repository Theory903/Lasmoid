"""
Vector Quantizers — VQ / Residual VQ
======================================
Extracted from monolithic model.py during Phase 0 SOLiD refactoring.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        flat_x_f32 = flat_x.float()

        E = self.embedding.weight
        distances = (
            torch.sum(flat_x_f32**2, dim=-1, keepdim=True)
            + torch.sum(E.float() ** 2, dim=-1)
            - 2 * torch.matmul(flat_x_f32, E.float().t())
        )

        encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(-1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.codebook_size,
            device=x.device,
            dtype=x.dtype,
        )
        encodings.scatter_(1, encoding_indices, 1.0)

        quantized = torch.matmul(encodings, E.to(x.dtype)).view(B, L, D)

        # Codebook loss: moves the embeddings toward the encoder outputs
        codebook_loss = F.mse_loss(quantized, x.detach())
        # Commitment loss: moves the encoder outputs toward the embeddings
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices.view(B, L)


class ResidualVQ(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        dim: int,
        num_quantizers: int = 3,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.commitment_cost = commitment_cost
        self.vqs = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, dim, commitment_cost)
                for _ in range(num_quantizers)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        quantized_out = torch.zeros_like(x)
        residual = x
        total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        indices_list = []

        for vq in self.vqs:
            quantized, loss, indices = vq(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_loss = total_loss + loss
            indices_list.append(indices)

        first_indices = indices_list[0]
        dummy_adj = torch.zeros(1, 1, device=x.device, dtype=x.dtype)
        return quantized_out, total_loss, first_indices, dummy_adj
