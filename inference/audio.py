"""
Lasmoid — audio.py
==================
Implements the Gemma-4/Conformer-style Audio Encoder with subsampling blocks,
macaron-style conformer layers, projection layers, and mel-spectrogram preprocessing.
"""

import math
from typing import Optional, Tuple, Union

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


def preprocess_audio(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 128,
    win_length: int = 320,
    hop_length: int = 160,
    n_fft: int = 512,
) -> torch.Tensor:
    """
    Preprocesses raw audio waveforms into mel-spectrograms.
    Uses torchaudio if available, otherwise falls back to native PyTorch STFT.

    Args:
        waveform: Tensor of shape [B, samples] or [samples]
        sample_rate: Expected sample rate (default: 16000)
        n_mels: Number of mel bands (default: 128)
        win_length: Analysis window length (default: 320)
        hop_length: Frame shift (default: 160)
        n_fft: FFT window size (default: 512)

    Returns:
        mel_spectrogram: Tensor of shape [B, time_steps, n_mels]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.dtype == torch.int16:
        waveform = waveform.float() / 32768.0
    elif waveform.dtype == torch.int32:
        waveform = waveform.float() / 2147483648.0
    elif waveform.dtype == torch.uint8:
        waveform = (waveform.float() - 128.0) / 128.0
    else:
        waveform = waveform.float()

    try:
        import torchaudio
        import torchaudio.transforms as T

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(waveform.device)
        mel = mel_spectrogram(waveform)  # [B, n_mels, T_mel]
        return mel.transpose(1, 2)  # [B, T_mel, n_mels]

    except ImportError:
        # Fallback to standard PyTorch STFT
        B, S = waveform.shape
        window = torch.hann_window(win_length, device=waveform.device)

        # STFT return_complex=True yields [B, n_fft//2 + 1, T_mel]
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        mag = torch.abs(stft)  # [B, n_fft//2 + 1, T_mel]

        # Use a simple linear layer to project STFT magnitude bins to n_mels
        # Initialize weights with simple scaling factor
        proj = nn.Linear(n_fft // 2 + 1, n_mels, bias=False, device=waveform.device)
        nn.init.uniform_(proj.weight, 0.0, 1.0 / (n_fft // 2 + 1))

        mel = proj(mag.transpose(1, 2))  # [B, T_mel, n_mels]
        return mel


class ConformerSubsampling(nn.Module):
    """
    Conformer subsampling block.
    Reduces time dimension by 4x and projects frequency bands to conformer_dims.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)

        # Stride=2 convolutions halve the frequency bands twice
        f_conv1 = (in_dim + 2 * 1 - 3) // 2 + 1
        f_conv2 = (f_conv1 + 2 * 1 - 3) // 2 + 1

        self.proj = nn.Linear(out_dim * f_conv2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = F.gelu(self.conv1(x))  # [B, out_dim, T/2, F/2]
        x = F.gelu(self.conv2(x))  # [B, out_dim, T/4, F/4]

        B, D, T, freq_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, D * freq_dim)
        x = self.proj(x)
        return self.norm(x)


class ConformerFFN(nn.Module):
    """
    Feed Forward module with macaron-style scaling (0.5).
    """

    def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear2(self.activation(self.linear1(x)))
        x = self.dropout(x)
        return residual + 0.5 * x


class ConformerConvModule(nn.Module):
    """
    Convolution module for ConformerBlock.
    Uses LayerNorm instead of BatchNorm for single-sequence/inference robustness.
    """

    def __init__(self, dim: int, kernel_size: int = 31):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
        )
        self.norm_layer = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        x = x.transpose(1, 2)  # [B, dim, T]
        x = self.pointwise_conv1(x)  # [B, dim * 2, T]
        x = F.glu(x, dim=1)  # [B, dim, T]

        x = self.depthwise_conv(x)  # [B, dim, T]

        x = x.transpose(1, 2)  # [B, T, dim]
        x = self.norm_layer(x)
        x = self.activation(x)

        x = x.transpose(1, 2)  # [B, dim, T]
        x = self.pointwise_conv2(x)  # [B, dim, T]
        x = x.transpose(1, 2)  # [B, T, dim]

        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """
    Conformer Block combining macaron FFNs, self-attention, and conv module.
    Casts to float32 on MPS backends during attention to avoid dtype mismatch.
    """

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.ffn1 = ConformerFFN(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.conv = ConformerConvModule(dim)
        self.ffn2 = ConformerFFN(dim)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn1(x)

        # Self-Attention
        residual = x
        norm_x = self.attn_norm(x)

        # Safeguard for MPS MultiheadAttention
        dtype = norm_x.dtype
        norm_x_fp32 = norm_x.float()
        self.self_attn.to(torch.float32)
        attn_out, _ = self.self_attn(norm_x_fp32, norm_x_fp32, norm_x_fp32)
        attn_out = attn_out.to(dtype)

        x = residual + self.attn_dropout(attn_out)

        x = self.conv(x)
        x = self.ffn2(x)
        return self.final_norm(x)


class LasmoidAudioEncoder(nn.Module):
    """
    LasmoidAudioEncoder (Gemma-4 Conformer-Style).
    Maps raw audio features/waveforms to sequence embeddings and modality IDs.
    """

    def __init__(
        self,
        audio_feature_dim: int,
        conformer_dims: int,
        lm_model_dims: int,
        dim: int,
        n_layers: int = 12,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.feature_dim = audio_feature_dim
        self.conformer_dims = conformer_dims
        self.lm_model_dims = lm_model_dims
        self.text_dim = dim

        self.subsampling = ConformerSubsampling(
            in_dim=audio_feature_dim, out_dim=conformer_dims
        )

        self.blocks = nn.ModuleList(
            [ConformerBlock(dim=conformer_dims) for _ in range(n_layers)]
        )

        self.lm_proj = nn.Linear(conformer_dims, lm_model_dims)

        self.proj = nn.Sequential(
            nn.Linear(lm_model_dims, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            RMSNorm(dim, norm_eps),
        )

    def forward(
        self, waveform_or_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform_or_features: Waveforms of shape [B, samples] or mel-spectrogram [B, T, feature_dim]

        Returns:
            audio_embeddings: [B, num_tokens, dim]
            modality_ids: [B, num_tokens]
        """
        # If waveform is passed (2D), convert to mel spectrogram features
        if waveform_or_features.dim() == 2:
            features = preprocess_audio(
                waveform_or_features, n_mels=self.feature_dim
            )
        else:
            features = waveform_or_features

        # 1. Conformer subsampling (4x reduction)
        x = self.subsampling(features.to(self.subsampling.conv1.weight.dtype))

        # 2. Conformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Project to LM dimensions
        x = self.lm_proj(x)

        # 4. Project to text dimension
        embeddings = self.proj(x)

        # 5. Modality IDs
        B, num_tokens, _ = embeddings.shape
        modality_ids = torch.full(
            (B, num_tokens),
            MODALITY_AUDIO,
            dtype=torch.long,
            device=embeddings.device,
        )

        return embeddings, modality_ids
