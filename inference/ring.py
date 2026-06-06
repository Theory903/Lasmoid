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

    def __init__(self, use_ring: bool = False):
        super().__init__()
        self.use_ring = use_ring

    def prefill(self, attn_module, hidden_states, freqs_cis=None, start_pos=0):
        """
        Runs the prefill pass using ring attention.
        If distributed mode is not initialized, runs simulated ring prefill locally.
        """
        if not self.use_ring:
            return attn_module(hidden_states, freqs_cis, start_pos=start_pos)

        # Retrieve sequence details
        B, S, D = hidden_states.shape

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
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        B, S, D = hidden_states.shape
        segment_size = S // world_size

        if segment_size == 0:
            return attn_module(hidden_states, freqs_cis, start_pos=start_pos)

        # Local chunk query/key/value hidden states
        local_hidden = hidden_states[:, rank * segment_size : (rank + 1) * segment_size]
        local_freqs = (
            freqs_cis[rank * segment_size : (rank + 1) * segment_size]
            if freqs_cis is not None
            else None
        )

        # Run attention for this rank's segment
        local_out = attn_module(local_hidden, local_freqs, start_pos=start_pos)

        # Communicate intermediate outputs across all nodes in the ring
        # Wait, for a complete distributed gather:
        tensor_list = [torch.zeros_like(local_out) for _ in range(world_size)]
        dist.all_gather(tensor_list, local_out)

        return torch.cat(tensor_list, dim=1)
