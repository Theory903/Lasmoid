"""
Vector Quantizers — VQ / Residual VQ / Graph VQ
=================================================
Extracted from monolithic model.py during Phase 0 SOLiD refactoring.

Implements:
  - VectorQuantizer: Basic VQ with real L2 nearest-centroid selection and
    commitment/codebook losses.
  - GraphVectorQuantizer (GVQ): Fuses codebook embeddings via softmax-normalized
    directed adjacency message passing before quantization.
  - ResidualVQ: Multi-level residual quantization that optionally applies GVQ
    graph message-passing (gated by `use_gvq` config flag).
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple


class VectorQuantizer(nn.Module):
    """Basic Vector Quantizer with real L2 nearest-centroid selection.

    Computes:
      - Nearest centroid via L2 distance: k = argmin_j ||z - e_j||^2
      - Codebook loss: ||sg(z) - e||^2  (moves embeddings toward encoder outputs)
      - Commitment loss: ||z - sg(e)||^2  (moves encoder outputs toward embeddings)
      - Straight-through estimator for gradient flow
    """

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

        # Real L2 distance: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x·e^T
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

        # Codebook loss: ||sg(z) - e||^2 — moves embeddings toward encoder outputs
        codebook_loss = F.mse_loss(quantized, x.detach())
        # Commitment loss: ||z - sg(e)||^2 — moves encoder outputs toward embeddings
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices.view(B, L)


class GraphVectorQuantizer(nn.Module):
    """Graph Vector Quantizer — fuses codebook embeddings via softmax-normalized
    directed adjacency message passing before nearest-centroid quantization.

    Per ARCHITECTURE.md:
      E_graph = E + softmax(A, dim=-1) @ (E @ W_g)

    The adjacency matrix A is a learnable [codebook_size, codebook_size] parameter.
    Softmax normalization along rows makes each row a probability distribution over
    neighbor contributions (row-stochastic message weights).

    Quantization is performed against the graph-enhanced embeddings E_graph, while
    the distance lookup uses the raw embeddings E for stable centroid assignment.
    """

    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost

        # Raw codebook node embeddings
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # Learnable directed adjacency matrix: [codebook_size, codebook_size]
        self.adjacency = nn.Parameter(torch.zeros(codebook_size, codebook_size))

        # Graph relation projection W_g: transforms embeddings for message passing
        self.relation_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with GVQ graph message-passing.

        Args:
            x: Input tensor of shape (B, L, D)

        Returns:
            quantized: Graph-enhanced quantized output (B, L, D)
            loss: Combined codebook + commitment loss (scalar)
            indices: Encoding indices (B, L)
            adjacency: The raw adjacency parameter for external use (e.g. L1 sparsity)
        """
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        flat_x_f32 = flat_x.float()

        E = self.embedding.weight

        # Graph message passing: E_graph = E + softmax(A) @ (E @ W_g)
        relation_E = self.relation_proj(E)  # [codebook_size, dim]
        graph_weights = F.softmax(self.adjacency, dim=-1)  # [codebook_size, codebook_size]
        E_graph = E + torch.matmul(graph_weights, relation_E)  # [codebook_size, dim]

        # Nearest-centroid selection uses raw embeddings E for stable assignment
        # Real L2 distance: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x·e^T
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

        # Quantize using graph-enhanced embeddings E_graph
        quantized = torch.matmul(encodings, E_graph.to(x.dtype)).view(B, L, D)

        # Codebook loss: ||sg(z) - e_graph||^2 — moves graph-enhanced embeddings toward inputs
        codebook_loss = F.mse_loss(quantized, x.detach())
        # Commitment loss: ||z - sg(e_graph)||^2 — moves encoder outputs toward embeddings
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices.view(B, L), self.adjacency


class ResidualVQ(nn.Module):
    """Residual Vector Quantization with optional Graph VQ message-passing.

    When `use_gvq=True`, the first quantizer level uses GraphVectorQuantizer
    (softmax-normalized adjacency message passing over codebook embeddings).
    Subsequent residual levels use plain VectorQuantizer.

    When `use_gvq=False` (default), all levels use plain VectorQuantizer and
    no adjacency matrix is maintained — preserving backward compatibility.
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        num_quantizers: int = 3,
        commitment_cost: float = 0.25,
        use_gvq: bool = False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.commitment_cost = commitment_cost
        self.use_gvq = use_gvq

        quantizers = []
        for i in range(num_quantizers):
            if i == 0 and use_gvq:
                # First level uses GVQ with graph message-passing
                quantizers.append(
                    GraphVectorQuantizer(codebook_size, dim, commitment_cost)
                )
            else:
                quantizers.append(
                    VectorQuantizer(codebook_size, dim, commitment_cost)
                )
        self.vqs = nn.ModuleList(quantizers)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through residual VQ levels.

        Returns:
            quantized_out: Sum of quantized residuals (B, L, D)
            total_loss: Accumulated VQ loss across all levels (scalar)
            first_indices: Encoding indices from the first quantizer level (B, L)
            adjacency: GVQ adjacency matrix if use_gvq=True, else zeros (for interface compat)
        """
        B, L, D = x.shape
        quantized_out = torch.zeros_like(x)
        residual = x
        total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        indices_list = []
        adjacency = torch.zeros(
            self.codebook_size, self.codebook_size, device=x.device, dtype=x.dtype
        )

        for i, vq in enumerate(self.vqs):
            if i == 0 and self.use_gvq:
                # GVQ returns adjacency as 4th element
                quantized, loss, indices, adj = vq(residual)
                adjacency = adj
            else:
                quantized, loss, indices = vq(residual)

            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_loss = total_loss + loss
            indices_list.append(indices)

        first_indices = indices_list[0]
        return quantized_out, total_loss, first_indices, adjacency
