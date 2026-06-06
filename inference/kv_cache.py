import torch
import torch.nn as nn
from config import ModelArgs


class KVCache(nn.Module):
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


class BufferedKVCache(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BufferedKVCache not yet implemented")
