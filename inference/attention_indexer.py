import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from ._common import Linear, apply_rotary_emb
    from .compressor import Compressor
    from .config import ModelArgs
except ImportError:
    from _common import Linear, apply_rotary_emb
    from compressor import Compressor
    from config import ModelArgs


class Indexer(nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.indexer_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        max_batch_size_comp = args.max_batch_size * args.num_residual_streams
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                max_batch_size_comp,
                max(1, args.max_seq_len // compress_ratio),
                self.head_dim,
            ),
            persistent=False,
        )
        self.freqs_cis = None

    def resize_buffers(self, bsz: int, device: Optional[torch.device] = None):
        if bsz > self.kv_cache.shape[0]:
            if device is None:
                device = self.kv_cache.device
            new_kv_cache = torch.zeros(
                bsz,
                self.kv_cache.shape[1],
                self.kv_cache.shape[2],
                device=device,
                dtype=self.kv_cache.dtype,
            )
            new_kv_cache[: self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_kv_cache, persistent=False)
            self.compressor.kv_cache = self.kv_cache
            self.compressor.resize_buffers(bsz, device=device)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Dynamic resizing of indexer kv_cache for larger batch size (e.g. GRPO)
        self.resize_buffers(bsz, device=x.device)

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        q_nope = q[..., :-rd].contiguous()
        q_rope = apply_rotary_emb(q[..., -rd:].contiguous(), freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1).contiguous()
        q = q.to(qr.dtype)
        self.compressor(x, start_pos)
        cache_len = max(
            1,
            min(
                self.compressor.cache_write_ptr[:bsz].max().item(),
                self.kv_cache.shape[1],
            ),
        )
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[:bsz, :cache_len].to(q.dtype)
        )
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            # Compute fired_positions once and reuse (Req 18.3)
            fired_positions = self.compressor.fired_indices_buf[:bsz, :cache_len]
            query_positions = torch.arange(seqlen, device=x.device).view(1, seqlen, 1)
            mask = fired_positions.unsqueeze(1) > query_positions
            index_score = index_score.masked_fill(mask, float("-inf"))
        topk_idxs = index_score.topk(min(self.index_topk, cache_len), dim=-1)[1]
        if start_pos == 0:
            # Reuse fired_positions computed above instead of re-slicing the buffer
            fired_steps = torch.gather(
                fired_positions.unsqueeze(1).expand(-1, seqlen, -1), 2, topk_idxs.long()
            )
            valids = fired_steps <= query_positions
            topk_idxs = torch.where(valids, topk_idxs + offset, -1)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs
