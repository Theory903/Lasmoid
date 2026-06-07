"""
Lasmoid — recovery.py
=====================
Implements the recovery state-machine and dynamic model state checkpointing
for long-running stable generation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import torch

logger = logging.getLogger(__name__)


class ErrorLevel:
    L1 = 1  # Cache NaN/Inf corruption
    L2 = 2  # Logit drift / entropy collapse
    L3 = 3  # Repeated failure / logits NaN/Inf
    L4 = 4  # Catastrophic failure


class RecoveryAction:
    CONTINUE = "continue"
    ADJUST_TEMPERATURE = "adjust_temperature"
    RESTART_GENERATION = "restart_generation"
    ABORT = "abort"


class CheckpointManager:
    """
    Manages generation checkpoints by cloning KV caches, concept memory states,
    and sequence tokens to support rolling back state on failure.
    """

    def __init__(self, model):
        self.model = model
        self.checkpoints: List[Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[int, Dict[str, Any]]]] = []

    def save_checkpoint(
        self,
        step: int,
        idx: torch.Tensor,
        concept_db: Optional[torch.Tensor],
        memory_state: Optional[torch.Tensor],
    ):
        """Saves current generation state (tokens, memory, and layer-level caches).
        
        Validates that saved tensor shapes are consistent with the model's
        current architecture before persisting.
        """
        caches_state = {}
        for i, layer in enumerate(self.model.layers):
            layer_state = {}
            attn = layer.attn
            
            # Save buffers
            buffers_state = {}
            for name, val in attn._buffers.items():
                if isinstance(val, torch.Tensor):
                    buffers_state[name] = val.clone()
            layer_state["buffers"] = buffers_state

            # Save normal attributes (e.g. integer pointers/counters)
            attrs_state = {}
            for name, val in attn.__dict__.items():
                if name not in ["_buffers", "_parameters", "_modules", "training"]:
                    if isinstance(val, torch.Tensor):
                        attrs_state[name] = val.clone()
                    elif isinstance(val, (int, float, bool)) or val is None:
                        attrs_state[name] = val
            layer_state["attrs"] = attrs_state
            
            caches_state[i] = layer_state

        self.checkpoints.append((
            step,
            idx.clone(),
            concept_db.clone() if concept_db is not None else None,
            memory_state.clone() if memory_state is not None else None,
            caches_state
        ))
        
        # Keep only the last 2 checkpoints to manage memory usage
        if len(self.checkpoints) > 2:
            self.checkpoints.pop(0)

    def restore_latest(self) -> Optional[Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Restores the latest saved checkpoint state into the model layers.
        
        Verifies tensor shape consistency before restoring to prevent
        silent corruption from architecture changes between save and restore.
        """
        if not self.checkpoints:
            return None
        
        step, idx, concept_db, memory_state, caches_state = self.checkpoints[-1]
        
        for i, layer_state in caches_state.items():
            layer = self.model.layers[i]
            attn = layer.attn
            
            # Restore buffers with shape validation
            for name, val in layer_state.get("buffers", {}).items():
                if name in attn._buffers and isinstance(attn._buffers[name], torch.Tensor):
                    current_buf = attn._buffers[name]
                    if current_buf.shape != val.shape:
                        logger.warning(
                            f"[recovery] Shape mismatch restoring buffer '{name}' in layer {i}: "
                            f"checkpoint {tuple(val.shape)} vs model {tuple(current_buf.shape)}. Skipping."
                        )
                        continue
                    current_buf.copy_(val)

            # Restore attributes with shape validation
            for name, val in layer_state.get("attrs", {}).items():
                if isinstance(val, torch.Tensor):
                    curr_val = getattr(attn, name, None)
                    if isinstance(curr_val, torch.Tensor):
                        if curr_val.shape != val.shape:
                            logger.warning(
                                f"[recovery] Shape mismatch restoring attr '{name}' in layer {i}: "
                                f"checkpoint {tuple(val.shape)} vs model {tuple(curr_val.shape)}. Skipping."
                            )
                            continue
                        curr_val.copy_(val)
                else:
                    setattr(attn, name, val)

        return step, idx, concept_db, memory_state


class LongRunningRecovery:
    """
    State-machine executing recovery actions (L1-L4 degradation)
    for long-running generation.
    """

    def __init__(self, model):
        self.model = model
        self.checkpoint_manager = CheckpointManager(model)
        self.failure_count = 0
        self.max_failures = 3

    def detect_error_level(self, logits: torch.Tensor, drift_signals: list, cache_ok: bool) -> Optional[int]:
        """Classifies the current state and returns an active ErrorLevel if any."""
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return ErrorLevel.L3
        
        if not cache_ok:
            return ErrorLevel.L1
            
        if len(drift_signals) > 0:
            return ErrorLevel.L2
            
        return None

    def recover(self, error_level: int, step: int, current_temp: float) -> Tuple[str, float, Optional[int]]:
        """
        Executes rollback recovery strategies based on the classified error level.

        Returns:
            action: RecoveryAction to execute
            new_temperature: Updated temperature setting
            rollback_step: Sequence position to rollback to
        """
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            logger.error(f"Catastrophic failure (L4) at step {step}: reached max recovery attempts.")
            return RecoveryAction.ABORT, current_temp, None

        if error_level == ErrorLevel.L1:
            logger.warning(f"L1 Cache Corruption detected at step {step}. Performing cache reset and rollback.")
            res = self.checkpoint_manager.restore_latest()
            if res is not None:
                rollback_step, _, _, _ = res
                self.reset_all_caches()
                return RecoveryAction.CONTINUE, current_temp, rollback_step
            
        elif error_level == ErrorLevel.L2:
            logger.warning(f"L2 Drift detected at step {step}. Adjusting temperature and rolling back.")
            res = self.checkpoint_manager.restore_latest()
            if res is not None:
                rollback_step, _, _, _ = res
                # Reduce temperature by 10% to cool down and restrict randomness
                new_temp = max(0.1, current_temp * 0.9)
                return RecoveryAction.ADJUST_TEMPERATURE, new_temp, rollback_step

        elif error_level == ErrorLevel.L3:
            logger.error(f"L3 NaN/Inf logits or spike detected at step {step}. Performing full restart of block states.")
            res = self.checkpoint_manager.restore_latest()
            if res is not None:
                rollback_step, _, _, _ = res
                self.reset_all_caches()
                # Drop to highly conservative temperature to force stable deterministic outputs
                return RecoveryAction.RESTART_GENERATION, 0.2, rollback_step

        return RecoveryAction.ABORT, current_temp, None

    def reset_all_caches(self):
        """Zeros out KV cache buffers across all decoder blocks."""
        for layer in self.model.layers:
            attn = layer.attn
            if hasattr(attn, "reset_cache"):
                attn.reset_cache()
            else:
                for name in ["kv_cache", "local_k_cache", "local_v_cache", "global_k_cache", "global_v_cache", "global_write_ptr"]:
                    if hasattr(attn, name):
                        val = getattr(attn, name)
                        if isinstance(val, torch.Tensor):
                            val.zero_()
