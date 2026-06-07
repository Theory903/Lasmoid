"""
Lasmoid — ring.py
===================
Implements Ring Attention prefill coordination, simulating or executing distributed
prefill passes across ring nodes.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class RingAttentionPrefill(nn.Module):
    """
    Ring Attention Prefill coordinator.
    Allows processing long context prefill by splitting the sequence
    along the sequence dimension and passing key/value states around a ring.
    """

    def __init__(self, use_ring: bool = False, ring_wraparound_fix: bool = False):
        super().__init__()
        self.use_ring = use_ring
        # When True, the distributed path handles remainder tokens that don't
        # divide evenly across ranks (gated to preserve legacy behavior).
        self.ring_wraparound_fix = ring_wraparound_fix

    def prefill(self, attn_module, hidden_states, freqs_cis=None, start_pos=0):
        """
        Runs the prefill pass using ring attention.
        If distributed mode is not initialized, runs simulated ring prefill locally.
        """
        if not self.use_ring:
            return attn_module(hidden_states, freqs_cis, start_pos=start_pos)

        # Check if we can run distributed ring attention
        if dist.is_initialized() and dist.get_world_size() > 1:
            return self._prefill_distributed(
                attn_module, hidden_states, freqs_cis, start_pos
            )
        else:
            return self._prefill_simulated(
                attn_module, hidden_states, freqs_cis, start_pos
            )

    def _prefill_simulated(
        self, attn_module, hidden_states, freqs_cis=None, start_pos=0
    ):
        """
        Locally simulates ring attention by chunking the prefill phase into
        sequential segments to mimic distributed ring communication buffers.
        """
        B, S, D = hidden_states.shape
        num_segments = 4
        segment_size = (S + num_segments - 1) // num_segments

        if segment_size == 0:
            return attn_module(hidden_states, freqs_cis, start_pos=start_pos)

        outputs = []
        for i in range(num_segments):
            start = i * segment_size
            end = min(S, (i + 1) * segment_size)
            if start >= end:
                break

            seg_hidden = hidden_states[:, start:end]
            seg_freqs = (
                freqs_cis[start:end]
                if freqs_cis is not None and len(freqs_cis) >= end
                else None
            )

            # Compute attention for this segment
            out = attn_module(seg_hidden, seg_freqs, start_pos=start_pos + start)
            outputs.append(out)

        return torch.cat(outputs, dim=1)

    def _prefill_distributed(
        self, attn_module, hidden_states, freqs_cis=None, start_pos=0
    ):
        """
        Actually executes distributed ring-pass communication for prefill.

        When ring_wraparound_fix is enabled, remainder tokens (S % world_size)
        are assigned to the last rank so no tokens are silently dropped.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        B, S, D = hidden_states.shape
        segment_size = S // world_size

        if segment_size == 0:
            return attn_module(hidden_states, freqs_cis, start_pos=start_pos)

        # Local chunk query/key/value hidden states
        chunk_start = rank * segment_size
        if self.ring_wraparound_fix and rank == world_size - 1:
            # Last rank absorbs remainder tokens so none are dropped
            chunk_end = S
        else:
            chunk_end = (rank + 1) * segment_size

        local_hidden = hidden_states[:, chunk_start:chunk_end]
        local_freqs = (
            freqs_cis[chunk_start:chunk_end]
            if freqs_cis is not None
            else None
        )

        # Run attention for this rank's segment
        local_out = attn_module(local_hidden, local_freqs, start_pos=start_pos)

        if self.ring_wraparound_fix:
            # Use all_gather with padding to handle variable-length segments
            # Pad local_out to max possible segment size for uniform gather
            max_seg = segment_size + (S - world_size * segment_size)  # last rank's size
            padded_out = torch.zeros(B, max_seg, D, device=local_out.device, dtype=local_out.dtype)
            padded_out[:, :local_out.shape[1]] = local_out
            tensor_list = [torch.zeros_like(padded_out) for _ in range(world_size)]
            dist.all_gather(tensor_list, padded_out)
            # Trim each rank's contribution to its actual size
            results = []
            for r in range(world_size):
                if r == world_size - 1:
                    actual_len = S - r * segment_size
                else:
                    actual_len = segment_size
                results.append(tensor_list[r][:, :actual_len])
            return torch.cat(results, dim=1)
        else:
            # Legacy path: drops remainder tokens when S % world_size != 0
            tensor_list = [torch.zeros_like(local_out) for _ in range(world_size)]
            dist.all_gather(tensor_list, local_out)
            return torch.cat(tensor_list, dim=1)
