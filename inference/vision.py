"""
Lasmoid — vision.py
===================
Implements the Gemma-4/SigLIP-style Vision Encoder with patch projection,
position embeddings, bidirectional attention layers, and adaptive token budgets.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._common import RMSNorm
except ImportError:
    from _common import RMSNorm

# Modality constants
MODALITY_TEXT = 0
MODALITY_VISION = 1
MODALITY_AUDIO = 2


def avg_pool_by_positions(
    x: torch.Tensor,
    positions_xy: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """
    2D spatial average pooling according to patch coordinates.
    Weights each patch projection back into target token length.
    """
    B, L, D = x.shape
    k = int((L // length) ** 0.5)
    # Check if length matches exactly
    if k * k * length != L or k == 0:
        # Fallback to standard 1D pooling if spatial grid is not a perfect match.
        # Run on CPU to avoid MPS limitations for non-divisible adaptive pooling.
        device = x.device
        x_cpu = x.cpu()
        out_cpu = F.adaptive_avg_pool1d(x_cpu.transpose(1, 2), length).transpose(1, 2)
        return out_cpu.to(device)
    
    # max_x is W_p
    max_x = positions_xy[..., 0].max(dim=-1, keepdim=True).values + 1
    kernel_idxs = torch.div(positions_xy, k, rounding_mode="floor")
    
    # flat_kernel_idx
    num_cols = torch.clamp(max_x // k, min=1)
    flat_kernel_idx = kernel_idxs[..., 0] + num_cols * kernel_idxs[..., 1]
    
    # clamp flat_kernel_idx to [0, length - 1] to prevent out of bounds
    flat_kernel_idx = torch.clamp(flat_kernel_idx, 0, length - 1)
    
    # one-hot in PyTorch
    weights = F.one_hot(flat_kernel_idx.long(), num_classes=length).float() / (k * k)
    
    # weighted average
    output = torch.einsum('bLl,bLd->bld', weights.to(x.dtype), x)
    return output


class BidirectionalTransformerLayer(nn.Module):
    """
    Standard transformer encoder layer with bidirectional self-attention.
    Robustly casts to float32 on MPS backends to prevent precision crashes.
    """

    def __init__(self, dim: int, n_heads: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Safeguard for MPS MultiheadAttention dtype mismatch
        x_fp32 = x.float()
        self.self_attn.to(torch.float32)

        attn_out, _ = self.self_attn(x_fp32, x_fp32, x_fp32)
        attn_out = attn_out.to(dtype)

        x = self.norm1(x + attn_out)
        ffn_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ffn_out)
        return x


class LasmoidVisionEncoder(nn.Module):
    """
    LasmoidVisionEncoder (Gemma-4 SigLIP-Style).
    Maps pixel values to text token sequence space.
    """

    def __init__(
        self,
        vision_dim: int,
        dim: int,
        n_layers: int = 16,
        patch_size: int = 14,
        norm_eps: float = 1e-6,
        standardize_embeddings: bool = False,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = dim
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.standardize_embeddings = standardize_embeddings

        self.patch_embed = nn.Conv2d(
            3, vision_dim, kernel_size=patch_size, stride=patch_size
        )

        # Factorized position embeddings for X and Y coordinates (Gemma-4 style)
        self.pos_emb = nn.Parameter(torch.randn(10240, 2, vision_dim) * 0.02)

        self.vision_layers = nn.ModuleList(
            [BidirectionalTransformerLayer(vision_dim) for _ in range(n_layers)]
        )

        self.proj = nn.Sequential(
            nn.Linear(vision_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            RMSNorm(dim, norm_eps),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_length: int = 280,
        positions_xy: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes image inputs and returns projected embeddings and modality IDs.

        Args:
            pixel_values: Tensor of shape [B, 3, H, W]
            output_length: Number of tokens to pool into (e.g. 70, 140, 280, 560, 1120)
            positions_xy: Optional patch positions coordinate grid [B, L, 2]

        Returns:
            vision_embeddings: [B, output_length, dim]
            modality_ids: [B, output_length]
        """
        B = pixel_values.shape[0]

        # 1. Patch projection
        pixel_values = 2.0 * (pixel_values - 0.5)
        patches = self.patch_embed(pixel_values.to(self.patch_embed.weight.dtype))
        # Shape: [B, vision_dim, H_p, W_p]
        H_p, W_p = patches.shape[2], patches.shape[3]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, vision_dim]

        # 2. Factorized Position embedding addition
        if positions_xy is None:
            x_coords = torch.arange(W_p, device=pixel_values.device)
            y_coords = torch.arange(H_p, device=pixel_values.device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            positions_xy = torch.stack([grid_x, grid_y], dim=-1).flatten(0, 1)  # [num_patches, 2]
            positions_xy = positions_xy.unsqueeze(0).expand(B, -1, -1)  # [B, num_patches, 2]

        x_coords = torch.clamp(positions_xy[..., 0], 0, self.pos_emb.size(0) - 1).long()
        y_coords = torch.clamp(positions_xy[..., 1], 0, self.pos_emb.size(0) - 1).long()

        x_emb = self.pos_emb[x_coords, 0, :]
        y_emb = self.pos_emb[y_coords, 1, :]
        pos_embed = x_emb + y_emb
        patches = patches + pos_embed

        # 3. Bidirectional self-attention layers
        for layer in self.vision_layers:
            patches = layer(patches)

        # 4. 2D spatial pooling according to patch positions
        pooled = avg_pool_by_positions(patches, positions_xy, output_length)

        # 5. Project to text dimension
        embeddings = self.proj(pooled) * math.sqrt(self.text_dim)

        # 6. Optional standardization
        if self.standardize_embeddings:
            embeddings = embeddings / (
                embeddings.norm(dim=-1, keepdim=True) + 1e-6
            )

        # 7. Modality IDs
        modality_ids = torch.full(
            (B, output_length),
            MODALITY_VISION,
            dtype=torch.long,
            device=embeddings.device,
        )

        return embeddings, modality_ids
