"""
Lasmoid — kv_cache.py (Phase A1: FP8 quantisation support)
===========================================================
Adds QuantKVCache (FP8 on-disk, BF16 on-read) alongside the existing
plain KVCache and SlidingWindowKVCache.
"""

from typing import Optional

import torch
import torch.nn as nn
from config import ModelArgs

try:
    from .kernels.quant import KVQuantConfig, quantize_kv, dequantize_kv
except ImportError:
    from kernels.quant import KVQuantConfig, quantize_kv, dequantize_kv


class KVCache(nn.Module):
    """Plain BF16 KV cache."""

    def __init__(
        self,
        max_batch: int,
        max_seq: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.head_dim = head_dim
        self._bsz = max_batch
        self.register_buffer(
            "cache",
            torch.zeros(max_batch, max_seq, head_dim, dtype=dtype),
            persistent=False,
        )

    def register(self, device: torch.device) -> torch.Tensor:
        self.cache = self.cache.to(device)
        return self.cache

    def resize(self, bsz: int) -> None:
        if bsz > self.cache.shape[0]:
            new = torch.zeros(
                bsz,
                *self.cache.shape[1:],
                device=self.cache.device,
                dtype=self.cache.dtype,
            )
            new[: self.cache.shape[0]] = self.cache
            self.register_buffer("cache", new, persistent=False)
        self._bsz = bsz

    def reset(self) -> None:
        self.cache.detach_().zero_()


class QuantKVCache(nn.Module):
    """
    FP8-quantised KV cache.

    Stores KV as ``torch.float8_e4m3fn`` + per-block fp32 scales.
    Dequantises to BF16 on read.  Provides 2× memory savings on the
    KV data plane (~0.8 % overhead for scales).

    Shape convention matches ``KVCache``: ``(batch, seq, head_dim)``.
    """

    def __init__(
        self,
        max_batch: int,
        max_seq: int,
        head_dim: int,
        quant_config: Optional[KVQuantConfig] = None,
    ):
        super().__init__()
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.head_dim = head_dim
        self._bsz = max_batch
        qc = quant_config or KVQuantConfig()
        self.block_size = qc.block_size
        self.scale_dtype = qc.torch_scale_dtype
        n_blocks = (head_dim + self.block_size - 1) // self.block_size

        self.register_buffer(
            "cache_q",
            torch.zeros(max_batch, max_seq, head_dim, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        self.register_buffer(
            "cache_s",
            torch.zeros(max_batch, max_seq, n_blocks, dtype=self.scale_dtype),
            persistent=False,
        )

    def register(self, device: torch.device) -> None:
        self.cache_q = self.cache_q.to(device)
        self.cache_s = self.cache_s.to(device)

    def resize(self, bsz: int) -> None:
        if bsz > self.cache_q.shape[0]:
            new_q = torch.zeros(
                bsz,
                *self.cache_q.shape[1:],
                device=self.cache_q.device,
                dtype=self.cache_q.dtype,
            )
            new_s = torch.zeros(
                bsz,
                *self.cache_s.shape[1:],
                device=self.cache_s.device,
                dtype=self.cache_s.dtype,
            )
            new_q[: self.cache_q.shape[0]] = self.cache_q
            new_s[: self.cache_s.shape[0]] = self.cache_s
            self.register_buffer("cache_q", new_q, persistent=False)
            self.register_buffer("cache_s", new_s, persistent=False)
        self._bsz = bsz

    def reset(self) -> None:
        self.cache_q.detach_().zero_()
        self.cache_s.detach_().zero_()

    def write(self, kv: torch.Tensor, slot_start: int, slot_end: int) -> None:
        """Quantise *kv* and store in ``cache_q[:, start:end]`` / ``cache_s``."""
        q, s = quantize_kv(kv.contiguous(), self.block_size, self.scale_dtype)
        self.cache_q[:, slot_start:slot_end] = q
        self.cache_s[:, slot_start:slot_end] = s

    def read(self, slot_start: int, slot_end: int) -> torch.Tensor:
        """Dequantise ``cache_q[:, start:end]`` back to BF16."""
        return dequantize_kv(
            self.cache_q[:, slot_start:slot_end],
            self.cache_s[:, slot_start:slot_end],
            self.block_size,
        )


class SlidingWindowKVCache(KVCache):
    def __init__(
        self,
        max_batch: int,
        max_seq: int,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(max_batch, window_size, head_dim, dtype)
        self.window_size = window_size

    def update(self, kv: torch.Tensor, start_pos: int, seqlen: int) -> None:
        B = kv.shape[0]
        self.resize(B)
        win = self.window_size

        if start_pos == 0:
            self.reset()
            if seqlen <= win:
                self.cache[:B, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.cache[:B, cutoff:win], self.cache[:B, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
        else:
            slot = start_pos % win
            self.cache[:B, slot] = kv[:, 0] if kv.dim() == 3 else kv

    def read(self, start_pos: int, seqlen: int) -> torch.Tensor:
        if start_pos == 0:
            return None
        return self.cache[: self._bsz]


class TieredKVCache(nn.Module):
    """
    Hot (BF16) + Main (FP8) tiered KV cache.

    Keeps the first *hot_size* tokens in BF16 (no quantisation loss) and
    the remaining tokens in FP8 (2× memory savings).  On read the FP8
    portion is transparently dequantised so callers always see BF16.

    Shape: ``(batch, seq, head_dim)`` — same as ``KVCache``.
    """

    def __init__(
        self,
        max_batch: int,
        max_seq: int,
        head_dim: int,
        hot_size: int = 512,
        quant_config: Optional[KVQuantConfig] = None,
    ):
        super().__init__()
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.head_dim = head_dim
        self.hot_size = min(hot_size, max_seq)
        self._bsz = max_batch

        self.hot = KVCache(max_batch, self.hot_size, head_dim)
        self.main = QuantKVCache(
            max_batch, max_seq - self.hot_size, head_dim, quant_config
        )

    def register(self, device: torch.device) -> None:
        self.hot.register(device)
        self.main.register(device)

    def resize(self, bsz: int) -> None:
        if bsz > self._bsz:
            self.hot.resize(bsz)
            self.main.resize(bsz)
        self._bsz = bsz

    def reset(self) -> None:
        self.hot.reset()
        self.main.reset()

    def write(self, kv: torch.Tensor, slot_start: int, slot_end: int) -> None:
        B = kv.shape[0]
        self.resize(B)

        if slot_start < self.hot_size:
            hs = self.hot_size
            hot_end = min(slot_end, hs)
            self.hot.cache[:B, slot_start:hot_end] = kv[:, : hot_end - slot_start]

            if slot_end > hs:
                main_kv = kv[:, hot_end - slot_start :]
                self.main.write(main_kv.contiguous(), 0, slot_end - hs)
        else:
            main_start = slot_start - self.hot_size
            main_end = slot_end - self.hot_size
            self.main.write(kv.contiguous(), main_start, main_end)

    def read(self, slot_start: int, slot_end: int) -> torch.Tensor:
        B = self._bsz
        hs = self.hot_size
        if slot_end <= hs:
            return self.hot.cache[:B, slot_start:slot_end]
        if slot_start >= hs:
            return self.main.read(slot_start - hs, slot_end - hs)
        hot_part = self.hot.cache[:B, slot_start:hs]
        main_part = self.main.read(0, slot_end - hs)
        return torch.cat([hot_part, main_part], dim=1)


class BufferedKVCache(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BufferedKVCache not yet implemented")


class AdaptiveQuantizedKVCacheSlice:
    """
    A sliced view proxy of AdaptiveQuantizedKVCache.
    Used for the compressor zone: kv_cache[:, win:].
    """
    def __init__(self, parent: "AdaptiveQuantizedKVCache", start_idx: int):
        self.parent = parent
        self.start_idx = start_idx

    @property
    def shape(self):
        p_shape = self.parent.shape
        return (p_shape[0], p_shape[1] - self.start_idx, p_shape[2])

    @property
    def device(self):
        return self.parent.device

    @property
    def dtype(self):
        return self.parent.dtype

    def __len__(self):
        return self.shape[1]

    def __setitem__(self, key, value):
        # Map the key to the parent indices
        if isinstance(key, tuple):
            batch_key = key[0]
            seq_key = key[1]
            if isinstance(seq_key, slice):
                start = (seq_key.start if seq_key.start is not None else 0) + self.start_idx
                stop = (seq_key.stop if seq_key.stop is not None else self.shape[1]) + self.start_idx
                parent_key = (batch_key, slice(start, stop, seq_key.step))
            elif isinstance(seq_key, int):
                parent_key = (batch_key, seq_key + self.start_idx)
            else:
                raise TypeError(f"Unsupported sequence key type: {type(seq_key)}")
        elif isinstance(key, slice):
            start = (key.start if key.start is not None else 0) + self.start_idx
            stop = (key.stop if key.stop is not None else self.shape[1]) + self.start_idx
            parent_key = slice(start, stop, key.step)
        elif isinstance(key, int):
            parent_key = key + self.start_idx
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

        self.parent[parent_key] = value

    def __getitem__(self, key):
        # Map the key to the parent indices and read
        if isinstance(key, tuple):
            batch_key = key[0]
            seq_key = key[1]
            if isinstance(seq_key, slice):
                start = (seq_key.start if seq_key.start is not None else 0) + self.start_idx
                stop = (seq_key.stop if seq_key.stop is not None else self.shape[1]) + self.start_idx
                parent_key = (batch_key, slice(start, stop, seq_key.step))
            elif isinstance(seq_key, int):
                parent_key = (batch_key, seq_key + self.start_idx)
            else:
                raise TypeError(f"Unsupported sequence key type: {type(seq_key)}")
        elif isinstance(key, slice):
            start = (key.start if key.start is not None else 0) + self.start_idx
            stop = (key.stop if key.stop is not None else self.shape[1]) + self.start_idx
            parent_key = slice(start, stop, key.step)
        elif isinstance(key, int):
            parent_key = key + self.start_idx
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

        return self.parent[parent_key]


class AdaptiveQuantizedKVCache(nn.Module):
    """
    Unified KV cache coordinating raw storage, FP8 tiered storage,
    TurboQuant simulations, SnapKV eviction, and OMP compaction.
    """
    def __init__(
        self,
        max_batch: int,
        max_seq: int,
        head_dim: int,
        args: ModelArgs,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.head_dim = head_dim
        self._bsz = max_batch
        
        # Read gating flags from args
        self.use_fp8_kv = getattr(args, "use_fp8_kv", False)
        self.use_turboquant = getattr(args, "use_turboquant", False)
        self.use_kv_eviction = getattr(args, "use_kv_eviction", False)
        self.use_compaction = getattr(args, "use_compaction", False)
        self.window_size = getattr(args, "sliding_window_size", 512)
        
        # Primary local storage (BF16 representation)
        self.register_buffer(
            "cache",
            torch.zeros(max_batch, max_seq, head_dim, dtype=dtype),
            persistent=False,
        )
        
        # Initialize FP8 Tiered Cache if gated
        if self.use_fp8_kv:
            self.tiered_cache = TieredKVCache(
                max_batch=max_batch,
                max_seq=max_seq,
                head_dim=head_dim,
                hot_size=self.window_size,
            )
            
        # Initialize TurboQuant modules if gated
        if self.use_turboquant:
            try:
                from turboquant.quantizer import TurboQuantProd
                self.turboquant_module = TurboQuantProd(
                    dim=head_dim,
                    bits=3,
                    dtype=torch.float32,
                )
            except Exception as e:
                print(f"Error importing TurboQuant: {e}")
                self.turboquant_module = None
                
        # Initialize eviction / compaction configs
        try:
            from eviction import SnapKVConfig
            from compaction import OMPCompactionConfig
        except ImportError:
            from .eviction import SnapKVConfig
            from .compaction import OMPCompactionConfig
        
        self.snapkv_config = SnapKVConfig(
            enabled=self.use_kv_eviction,
            sink_size=getattr(args, "snapkv_sink_size", 4),
            window_size=getattr(args, "snapkv_window_size", 64),
            max_keep_size=getattr(args, "snapkv_max_keep_size", 512),
            observation_length=getattr(args, "snapkv_observation_length", 32),
            topk_ratio=getattr(args, "snapkv_topk_ratio", 0.5),
        )
        
        self.omp_config = OMPCompactionConfig(
            enabled=self.use_compaction,
            target_ratio=getattr(args, "omp_target_ratio", 0.5),
            k_choice=getattr(args, "omp_k_choice", 1),
            ridge_lambda=getattr(args, "omp_ridge_lambda", 0.0),
            beta_method=getattr(args, "omp_beta_method", "nnls"),
        )
        
        # kept_mask stores whether a compressed KV slot is kept (True) or evicted (False)
        self.register_buffer(
            "kept_mask",
            torch.ones(max_batch, max_seq, dtype=torch.bool),
            persistent=False,
        )
        
        self.max_written_idx = 0
        self.queries = None

    def set_queries(self, queries: torch.Tensor):
        self.queries = queries

    def register(self, device: torch.device) -> None:
        self.cache = self.cache.to(device)
        self.kept_mask = self.kept_mask.to(device)
        if self.use_fp8_kv:
            self.tiered_cache.register(device)
        if self.use_turboquant and self.turboquant_module is not None:
            self.turboquant_module = self.turboquant_module.to(device)
        return self

    def resize(self, bsz: int) -> None:
        if bsz > self.cache.shape[0]:
            device = self.cache.device
            new_cache = torch.zeros(
                bsz,
                *self.cache.shape[1:],
                device=device,
                dtype=self.cache.dtype,
            )
            new_cache[: self.cache.shape[0]] = self.cache
            self.register_buffer("cache", new_cache, persistent=False)
            
            new_mask = torch.ones(
                bsz,
                *self.kept_mask.shape[1:],
                device=device,
                dtype=torch.bool,
            )
            new_mask[: self.kept_mask.shape[0]] = self.kept_mask
            self.register_buffer("kept_mask", new_mask, persistent=False)
            
            if self.use_fp8_kv:
                self.tiered_cache.resize(bsz)
        self._bsz = bsz

    def reset(self) -> None:
        self.cache.detach_().zero_()
        self.kept_mask.fill_(True)
        self.max_written_idx = 0
        self.queries = None
        if self.use_fp8_kv:
            self.tiered_cache.reset()

    def detach_(self):
        self.cache.detach_()
        return self

    def zero_(self):
        self.cache.zero_()
        self.kept_mask.fill_(True)
        self.max_written_idx = 0
        self.queries = None
        if self.use_fp8_kv:
            self.tiered_cache.reset()
        return self

    def to(self, *args, **kwargs):
        self.cache = self.cache.to(*args, **kwargs)
        self.kept_mask = self.kept_mask.to(*args, **kwargs)
        if self.use_fp8_kv:
            # tiered_cache needs register called with a device
            device = self.cache.device
            self.tiered_cache.register(device)
        if self.use_turboquant and self.turboquant_module is not None:
            self.turboquant_module = self.turboquant_module.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return self.cache.device

    @property
    def dtype(self):
        return self.cache.dtype

    @property
    def shape(self):
        return self.cache.shape

    def size(self, dim=None):
        if dim is not None:
            return self.cache.size(dim)
        return self.cache.size()

    def get_full_tensor(self):
        if self.use_fp8_kv:
            return self.tiered_cache.read(0, self.cache.shape[1])
        return self.cache

    def __getitem__(self, key):
        # Check if it's a slice for the compressor
        if isinstance(key, tuple) and len(key) >= 2:
            batch_key, seq_key = key[0], key[1]
            if isinstance(seq_key, slice) and seq_key.start is not None and seq_key.start > 0:
                return AdaptiveQuantizedKVCacheSlice(self, seq_key.start)
        
        full = self.get_full_tensor()
        return full[key]

    def __setitem__(self, key, value):
        # 1. Simulate TurboQuant roundtrip if enabled
        if self.use_turboquant and self.turboquant_module is not None:
            val_device = value.device
            val_dtype = value.dtype
            tq_val = value.to(device=self.turboquant_module.S.device, dtype=self.turboquant_module.S.dtype)
            tq_val = self.turboquant_module(tq_val)
            value = tq_val.to(device=val_device, dtype=val_dtype)
            
        # 2. Write to local backing cache
        self.cache[key] = value
        
        # 3. Extract seq slots and write to FP8 Tiered Cache if enabled
        slot_start, slot_end = None, None
        
        if isinstance(key, tuple):
            batch_key = key[0]
            seq_key = key[1]
            
            if isinstance(seq_key, slice):
                slot_start = seq_key.start if seq_key.start is not None else 0
                slot_end = seq_key.stop if seq_key.stop is not None else self.cache.shape[1]
            elif isinstance(seq_key, int):
                slot_start = seq_key
                slot_end = seq_key + 1
        elif isinstance(key, slice):
            slot_start = key.start if key.start is not None else 0
            slot_end = key.stop if key.stop is not None else self.cache.shape[1]
            
        if slot_start is not None and slot_end is not None:
            # Track maximum written index in the compressed region
            if slot_end > self.window_size:
                self.max_written_idx = max(self.max_written_idx, slot_end - self.window_size)
                
            if self.use_fp8_kv:
                write_val = value
                if write_val.dim() == 2:
                    write_val = write_val.unsqueeze(1)
                elif write_val.dim() == 1:
                    write_val = write_val.unsqueeze(0).unsqueeze(1)
                self.tiered_cache.write(write_val.contiguous(), slot_start, slot_end)
                
            # 4. Trigger eviction or compaction if enabled and we have writes to the compressed area
            if (self.use_kv_eviction or self.use_compaction) and slot_end > self.window_size:
                self.run_eviction_or_compaction()

    def run_eviction_or_compaction(self):
        win = self.window_size
        active_len = self.max_written_idx
        if active_len <= 0:
            return
            
        # Retrieve the compressed keys/values from our backing store
        comp_kv = self.cache[:, win:win + active_len]
        B = comp_kv.shape[0]
        
        self.kept_mask[:, :active_len] = True
        
        if self.use_kv_eviction:
            try:
                from eviction import snapkv_evict
            except ImportError:
                from .eviction import snapkv_evict
            _, _, kept_indices = snapkv_evict(comp_kv, comp_kv, self.snapkv_config)
            
            self.kept_mask[:, :active_len] = False
            for b in range(B):
                self.kept_mask[b, kept_indices[b]] = True
                
        elif self.use_compaction:
            try:
                from compaction import omp_select_indices
            except ImportError:
                from .compaction import omp_select_indices
            self.kept_mask[:, :active_len] = False
            for b in range(B):
                if self.queries is not None and self.queries.shape[1] > 0:
                    sel, _ = omp_select_indices(comp_kv[b], self.queries[b], self.omp_config)
                    self.kept_mask[b, sel] = True
                else:
                    self.kept_mask[b, :active_len] = True

    def filter_topk_idxs(self, topk_idxs: torch.Tensor, start_pos: int, win: int) -> torch.Tensor:
        if not (self.use_kv_eviction or self.use_compaction):
            return topk_idxs
            
        B, S, topk = topk_idxs.shape
        filtered_idxs = topk_idxs.clone()
        
        for b in range(B):
            mask_b = filtered_idxs[b] >= win
            if mask_b.any():
                cache_idxs = filtered_idxs[b][mask_b] - win
                max_cache_len = self.kept_mask.shape[1]
                clamped_cache_idxs = cache_idxs.clamp(0, max_cache_len - 1)
                
                # Check which ones are kept
                kept = self.kept_mask[b, clamped_cache_idxs]
                
                # Set evicted ones to -1
                update_val = torch.where(kept, cache_idxs + win, -1)
                filtered_idxs[b][mask_b] = update_val.to(filtered_idxs.dtype)
                
        return filtered_idxs

