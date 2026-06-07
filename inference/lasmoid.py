"""
Lasmoid — Main Model Class (CQRS Hybrid Concept Transformer)
==============================================================
Extracted from monolithic model.py during Phase 0 SOLiD refactoring.
"""

import torch
import contextlib

# ── PyTorch checkpoint debug/symbolizer crash guard ────────────────────────────
# Root cause: In Kaggle/Colab, `torch.utils.checkpoint._checkpoint_debug_enabled`
# may be set to True by the environment, overriding `debug=False` and forcibly
# activating LoggingTensorMode → capture_logs → symbolize_tracebacks (C ext)
# → ValueError: stoi.
#
# Strategy:
#   1. Monkey-patch set_checkpoint_debug_enabled to a permanent no-op so
#      runtime re-enable by the platform is ignored.
#   2. Directly force _checkpoint_debug_enabled = False (handles pre-existing True).
#   3. Patch symbolize_tracebacks as a belt-and-suspenders fallback.
try:
    import torch.utils.checkpoint as _cp
    from contextlib import contextmanager as _ctxmgr

    @_ctxmgr
    def _noop_checkpoint_debug(enabled=None):
        """Permanent no-op: prevents Kaggle/Colab from re-enabling the debug flag."""
        yield

    # 1. Replace the setter with a no-op (preserves context-manager interface).
    _cp.set_checkpoint_debug_enabled = _noop_checkpoint_debug
    # 2. Force the variable directly (handles any prior value including True).
    _cp._checkpoint_debug_enabled = False
except Exception:
    pass

try:
    # 3. Belt-and-suspenders: patch symbolize_tracebacks in-place so any path
    #    that still reaches it gets a safe fallback instead of crashing.
    import torch.testing._internal.logging_tensor as _lt

    _orig_symbolize = _lt.symbolize_tracebacks

    def _safe_symbolize(tracebacks_list):
        try:
            return _orig_symbolize(tracebacks_list)
        except (ValueError, Exception):
            return [[] for _ in tracebacks_list]

    _lt.symbolize_tracebacks = _safe_symbolize
    # Also patch via sys.modules key in case of alternate import paths.
    import sys as _sys

    for _mod_name, _mod in list(_sys.modules.items()):
        if _mod is not None and hasattr(_mod, "symbolize_tracebacks"):
            if getattr(_mod, "symbolize_tracebacks") is _orig_symbolize:
                setattr(_mod, "symbolize_tracebacks", _safe_symbolize)
except Exception:
    pass

import torch.nn.functional as F
from torch import nn
from typing import Any, List, Optional, Tuple

try:
    from ._common import RMSNorm, Linear, set_dtype
    from .attention import MLAAttention, precompute_freqs_cis
    from .block import LasmoidBlock
    from .concept_memory import ElasticSparseConceptMemory
    from .moe import Gate
    from .mtp import MTPBlock
    from .config import ModelArgs
    from .stability import (
        DriftDetector,
        AdaptiveTemperatureScheduler,
        KVCacheIntegrityChecker,
    )
except ImportError:
    from _common import RMSNorm, Linear, set_dtype
    from attention import MLAAttention, precompute_freqs_cis
    from block import LasmoidBlock
    from concept_memory import ElasticSparseConceptMemory
    from moe import Gate
    from mtp import MTPBlock
    from config import ModelArgs
    from stability import (
        DriftDetector,
        AdaptiveTemperatureScheduler,
        KVCacheIntegrityChecker,
    )


class Lasmoid(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # ══ Einsum parameterization (Gemma-4) ══
        if getattr(args, "use_einsum", False):
            try:
                from ._common import set_use_einsum
            except ImportError:
                from _common import set_use_einsum

            set_use_einsum(True)
        self.max_seq_len = args.max_seq_len
        self.hc_mult = args.num_residual_streams
        self.emb = nn.Embedding(args.vocab_size, args.dim)
        self.external_embedding_proj = None
        self.external_embedding_norm = None
        if getattr(args, "external_embedding_dim", 0) > 0:
            self.external_embedding_proj = Linear(
                args.external_embedding_dim, args.dim, dtype=torch.bfloat16
            )
            self.external_embedding_norm = RMSNorm(args.dim, args.norm_eps)
        self.encoder_attn = MLAAttention(args, layer_id=0)
        self.encoder_norm = RMSNorm(args.dim, args.norm_eps)
        self.memory = ElasticSparseConceptMemory(args)

        # Relational Cortex (AlphaFold-3 Evoformer-style pairwise reasoning over
        # concept anchors). Config-gated; identity at init.
        self.relational_cortex = None
        if getattr(args, "use_relational_cortex", False):
            try:
                from .relational import RelationalCortex
            except ImportError:
                from relational import RelationalCortex
            self.relational_cortex = RelationalCortex(
                dim=args.dim,
                pair_dim=getattr(args, "relational_pair_dim", 16),
                opm_chan=getattr(args, "relational_opm_chan", 4),
                n_iters=getattr(args, "relational_iters", 2),
                eps=args.norm_eps,
            )

        # Curiosity Expert (intrinsic-curiosity questioning over concept memory).
        # Config-gated; identity at init (zero-init read-back).
        self.curiosity_expert = None
        if getattr(args, "use_curiosity_expert", False):
            try:
                from .curiosity import CuriosityExpert
            except ImportError:
                from curiosity import CuriosityExpert
            self.curiosity_expert = CuriosityExpert(
                dim=args.dim,
                n_questions=getattr(args, "curiosity_n_questions", 4),
                eps=args.norm_eps,
            )
        self.last_curiosity_loss = torch.tensor(0.0)

        # WRITE MASTER (Decoder) — sequence processing blocks
        self.layers = nn.ModuleList(
            [LasmoidBlock(i, args) for i in range(args.n_layers)]
        )
        self.decoder_norm = RMSNorm(args.dim, args.norm_eps)

        self.head = Linear(args.dim, args.vocab_size, dtype=torch.bfloat16)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        # Superhuman alignment head for attribute steering
        self.superhuman_alignment_head = Linear(
            args.dim, len(args.steering_attributes), dtype=torch.bfloat16
        )

        # Output head Hyper-Connections reducer
        hc_mult = args.num_residual_streams
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.hc_head_fn, 0, 0.02)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)

        # Multimodal input projections (Gemma4)
        self.vision_proj = None
        self.vision_norm = None
        if getattr(args, "vision_dim", 0) > 0:
            self.vision_proj = Linear(args.vision_dim, args.dim, dtype=torch.bfloat16)
            self.vision_norm = RMSNorm(args.dim, args.norm_eps)

        self.audio_proj = None
        self.audio_norm = None
        if getattr(args, "audio_dim", 0) > 0:
            self.audio_proj = Linear(args.audio_dim, args.dim, dtype=torch.bfloat16)
            self.audio_norm = RMSNorm(args.dim, args.norm_eps)

        # Multimodal Encoders
        self.vision_encoder = None
        if getattr(args, "vision_dim", 0) > 0:
            try:
                from .vision import LasmoidVisionEncoder
            except ImportError:
                from vision import LasmoidVisionEncoder
            self.vision_encoder = LasmoidVisionEncoder(
                vision_dim=args.vision_dim,
                dim=args.dim,
                n_layers=getattr(args, "n_vision_layers", 16),
                norm_eps=args.norm_eps,
            )

        self.audio_encoder = None
        if getattr(args, "audio_dim", 0) > 0:
            try:
                from .audio import LasmoidAudioEncoder
            except ImportError:
                from audio import LasmoidAudioEncoder
            self.audio_encoder = LasmoidAudioEncoder(
                audio_feature_dim=getattr(args, "audio_feature_dim", 80),
                conformer_dims=getattr(args, "audio_conformer_dims", 1024),
                lm_model_dims=getattr(args, "audio_lm_dims", 1536),
                dim=args.dim,
                n_layers=getattr(args, "n_audio_layers", 12),
                norm_eps=args.norm_eps,
            )

        # Per-layer modality parameters
        per_layer_dim = getattr(args, "per_layer_input_dim", 64)
        self.per_layer_embeddings = nn.Parameter(
            torch.randn(3, args.n_layers, per_layer_dim, dtype=torch.bfloat16) * 0.02
        )
        self.per_layer_proj = Linear(
            args.dim, args.n_layers * per_layer_dim, dtype=torch.bfloat16
        )
        self.per_layer_norm = RMSNorm(per_layer_dim, args.norm_eps)

        # Multi-Token Prediction (MTP)
        self.mtp = nn.ModuleList()
        for i in range(args.n_mtp_layers):
            blk = MTPBlock(args.n_layers + i, args)
            blk.embed = self.emb
            blk.head = self.head
            self.mtp.append(blk)

        # YaRN RoPE cache (cap at 2M+1024 to prevent OOM)
        freqs_seqlen = min(args.max_seq_len + 1024, 2097152 + 1024)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.rope_head_dim,
                freqs_seqlen,
                args.original_seq_len,
                args.rope_theta,
                args.rope_factor,
                args.beta_fast,
                args.beta_slow,
            ),
            persistent=False,
        )
        self.last_pred_loss = torch.tensor(0.0)

        self.gradient_checkpointing = False
        self.last_z_loss = torch.tensor(0.0)
        self.last_commit_loss = torch.tensor(0.0)
        self.last_vq_loss = torch.tensor(0.0)
        self.last_moe_loss = torch.tensor(0.0)
        self.last_token_concept_loss = torch.tensor(0.0)

        # Stability system (config-flag gated)
        stab_cfg = self.args.stability_config
        self.stability_enabled = stab_cfg.enabled
        if self.stability_enabled:
            self.drift_detector = DriftDetector(stab_cfg)
            self.temp_scheduler = AdaptiveTemperatureScheduler(stab_cfg)
            self.cache_checker = KVCacheIntegrityChecker(stab_cfg)

        self.create_kv_cache_sharing_patterns()

        # Check and apply post-training quantization (PTQ) to MoE experts
        if hasattr(self.args, "quant_config") and self.args.quant_config is not None:
            try:
                from .moe import DeepSeekMoE
            except ImportError:
                from moe import DeepSeekMoE

            for layer in self.layers:
                if hasattr(layer, "moe_layer") and isinstance(
                    layer.moe_layer, DeepSeekMoE
                ):
                    if self.args.quant_config.moe_route_dtype == "nvfp4":
                        layer.moe_layer.quantize_routed_experts_to_nvfp4()
                    elif self.args.quant_config.moe_route_dtype == "fp8":
                        layer.moe_layer.quantize_routed_experts_to_fp8()

                    if self.args.quant_config.moe_shared_dtype == "fp8":
                        layer.moe_layer.quantize_shared_expert_to_fp8()

    def create_kv_cache_sharing_patterns(self):
        frac = getattr(self.args, "frac_shared_layers", 0.0)
        if frac <= 0.0:
            return

        # Group layers by attention class type
        by_type = {}
        for idx, layer in enumerate(self.layers):
            attn_type = type(layer.attn)
            if attn_type not in by_type:
                by_type[attn_type] = []
            by_type[attn_type].append(idx)

        for attn_type, indices in by_type.items():
            num_layers = len(indices)
            if num_layers <= 1:
                continue
            num_unshared = max(1, int(num_layers * (1.0 - frac)))
            for i in range(num_unshared, num_layers):
                shared_idx = indices[i]
                target_idx = indices[i % num_unshared]

                shared_attn = self.layers[shared_idx].attn
                target_attn = self.layers[target_idx].attn

                # List of potential cache buffer names
                buffer_names = [
                    "kv_cache",
                    "local_k_cache",
                    "local_v_cache",
                    "global_k_cache",
                    "global_v_cache",
                    "global_write_ptr",
                ]
                for name in buffer_names:
                    if hasattr(target_attn, name):
                        target_tensor = getattr(target_attn, name)
                        setattr(shared_attn, name, target_tensor)
                        if name in shared_attn._buffers:
                            shared_attn._buffers[name] = target_tensor

    def embed_tokens(
        self,
        token_ids: torch.Tensor,
        external_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed token ids and optionally fuse aligned external embeddings."""
        token_emb = self.emb(token_ids).to(torch.bfloat16)
        if external_embeddings is None:
            return token_emb
        if self.external_embedding_proj is None:
            raise ValueError(
                "external_embeddings were provided but args.external_embedding_dim is 0"
            )
        if external_embeddings.shape[:2] != token_ids.shape:
            raise ValueError(
                "external_embeddings must be shaped [batch, seq, external_embedding_dim] "
                f"with batch/seq matching token ids; got {tuple(external_embeddings.shape)} vs {tuple(token_ids.shape)}"
            )
        if external_embeddings.shape[-1] != self.args.external_embedding_dim:
            raise ValueError(
                f"external embedding dim mismatch: got {external_embeddings.shape[-1]}, "
                f"expected {self.args.external_embedding_dim}"
            )

        ext = external_embeddings.to(device=token_ids.device, dtype=torch.bfloat16)
        if self.args.external_embedding_norm:
            ext = F.normalize(ext.float(), dim=-1, eps=1e-6).to(torch.bfloat16)
        ext = self.external_embedding_proj(ext)
        if self.external_embedding_norm is not None:
            ext = self.external_embedding_norm(ext)
        return token_emb + ext.to(token_emb.dtype) * self.args.external_embedding_scale

    def embed_multimodal(
        self,
        x_dec: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        external_embeddings: Optional[torch.Tensor] = None,
        vision_output_length: int = 280,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Base text/external embedding
        fused_embeddings = self.embed_tokens(x_dec, external_embeddings).to(
            torch.bfloat16
        )
        B, N_dec, dim = fused_embeddings.shape

        modality_ids = torch.full(
            (B, N_dec),
            0,  # MODALITY_TEXT
            dtype=torch.long,
            device=x_dec.device,
        )

        # 2. Interleave vision embeddings if present
        IMAGE_PLACEHOLDER = 258880
        if pixel_values is not None and self.vision_encoder is not None:
            # Shape: [B, V_len, dim], [B, V_len]
            vision_embeddings, vision_modality_ids = self.vision_encoder(
                pixel_values, output_length=vision_output_length
            )
            for b in range(B):
                idx_v = (x_dec[b] == IMAGE_PLACEHOLDER).nonzero(as_tuple=True)[0]
                if len(idx_v) > 0:
                    min_len = min(len(idx_v), vision_embeddings.size(1))
                    fused_embeddings[b, idx_v[:min_len]] = vision_embeddings[
                        b, :min_len
                    ].to(fused_embeddings.dtype)
                    modality_ids[b, idx_v[:min_len]] = 1  # MODALITY_VISION

        # 3. Interleave audio embeddings if present
        AUDIO_PLACEHOLDER = 258881
        if audio_values is not None and self.audio_encoder is not None:
            # Shape: [B, A_len, dim], [B, A_len]
            audio_embeddings, audio_modality_ids = self.audio_encoder(audio_values)
            for b in range(B):
                idx_a = (x_dec[b] == AUDIO_PLACEHOLDER).nonzero(as_tuple=True)[0]
                if len(idx_a) > 0:
                    min_len = min(len(idx_a), audio_embeddings.size(1))
                    fused_embeddings[b, idx_a[:min_len]] = audio_embeddings[
                        b, :min_len
                    ].to(fused_embeddings.dtype)
                    modality_ids[b, idx_a[:min_len]] = 2  # MODALITY_AUDIO

        # 4. Compute per-layer modality features
        layer_proj_out = self.per_layer_proj(
            fused_embeddings
        )  # [B, N_dec, n_layers * per_layer_input_dim]
        layer_proj_out = layer_proj_out.view(B, N_dec, self.args.n_layers, -1)

        mod_embeddings = self.per_layer_embeddings[
            modality_ids
        ]  # [B, N_dec, n_layers, per_layer_input_dim]

        layer_feats = layer_proj_out + mod_embeddings
        layer_feats = self.per_layer_norm(layer_feats)

        return fused_embeddings, layer_feats

    def apply_pending_bias_updates(self):
        for m in self.modules():
            if isinstance(m, Gate):
                m.apply_pending_updates()

    def clear_saved_checkpoint_states(self):
        for m in self.modules():
            if hasattr(m, "clear_saved_checkpoint_state"):
                m.clear_saved_checkpoint_state()

    def _hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        B, S, hc, D = shape
        xf = x.flatten(2)
        mean_sq = xf.square().mean(-1, keepdim=True).float()
        rsqrt = torch.rsqrt(mean_sq + 1e-6).to(dtype)
        mixes = F.linear(xf, self.hc_head_fn.to(dtype)) * rsqrt
        pre = (
            torch.sigmoid(mixes.float() * self.hc_head_scale + self.hc_head_base) + 1e-6
        )
        y = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)
        return y

    def forward(
        self,
        x_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        concept_db: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        external_embeddings: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        steering_vector: Optional[Any] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        vision_output_length: int = 280,
        domain_steer: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        if self.training:
            self.clear_saved_checkpoint_states()
            self.apply_pending_bias_updates()

        B, N_dec = x_dec.shape
        freqs_cis_dec = self.freqs_cis[start_pos : start_pos + N_dec]

        # ── 1. READ REPLICA ───────────────────────────────────────────
        if start_pos == 0 or concept_db is None or memory_state is None:
            assert x_enc is not None, "x_enc must be provided at start_pos == 0"
            _, N_enc = x_enc.shape
            freqs_cis_enc = self.freqs_cis[:N_enc]
            H_enc, _ = self.embed_multimodal(
                x_enc,
                pixel_values=pixel_values,
                audio_values=audio_values,
                external_embeddings=external_embeddings,
                vision_output_length=vision_output_length,
            )
            enc_out = self.encoder_attn(
                self.encoder_norm(H_enc), freqs_cis_enc, start_pos=0
            )
            memory_state, commit_loss = self.memory.process_chunk(enc_out, block_idx=0)
            self.last_commit_loss = commit_loss
            concept_db = self.memory.lightning_retrieve(
                H_enc, top_k_blocks=self.args.lightning_topk_blocks
            )

        # ── 2. WRITE MASTER ───────────────────────────────────────────
        H_dec_external = external_embeddings
        if external_embeddings is not None and external_embeddings.shape[1] != N_dec:
            H_dec_external = external_embeddings[:, -N_dec:, :]
        H_dec, layer_feats = self.embed_multimodal(
            x_dec,
            pixel_values=pixel_values,
            audio_values=audio_values,
            external_embeddings=H_dec_external,
            vision_output_length=vision_output_length,
        )

        # Parallel streams routing concept embedding representations
        # Relational Cortex: refine concept anchors via pairwise (Evoformer-style)
        # reasoning before they are broadcast into the residual streams.
        if self.relational_cortex is not None and concept_db is not None:
            concept_db = self.relational_cortex(concept_db)

        # Curiosity Expert: notice where new content is unpredictable from prior
        # knowledge and bridge to the most relevant concept slots (self-questioning).
        if self.curiosity_expert is not None:
            H_dec, _curiosity, curiosity_loss = self.curiosity_expert(H_dec, concept_db)
            self.last_curiosity_loss = curiosity_loss
        else:
            self.last_curiosity_loss = torch.tensor(
                0.0, device=x_dec.device, dtype=torch.float32
            )

        H_memory = torch.mean(memory_state, dim=1, keepdim=True).expand(-1, N_dec, -1)
        H_concept = torch.mean(concept_db, dim=1, keepdim=True).expand(-1, N_dec, -1)

        hc = self.hc_mult
        streams_list = [H_dec, H_memory, H_concept]
        if hc < len(streams_list):
            streams_list = streams_list[:hc]
        elif hc > len(streams_list):
            streams_list += [torch.zeros_like(H_dec)] * (hc - len(streams_list))
        streams = torch.stack(streams_list, dim=2)

        # cos & sin removed (relying on freqs_cis)

        total_z_loss = torch.tensor(0.0, device=x_dec.device, dtype=torch.float32)
        total_vq_loss = torch.tensor(0.0, device=x_dec.device, dtype=torch.float32)
        routing_maps = []
        concept_indices = []
        adjacencies = []
        event_probs = []

        # ── Chain-of-Thought Budget (Reasoning State-Machine) ──────────────
        reasoning_steps = getattr(self.args, "reasoning_steps", 1)

        for r_step in range(reasoning_steps):
            for layer in self.layers:
                layer_feats_slice = layer_feats[:, :, layer.layer_id, :]
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(
                        module,
                        freqs_cis_dec,
                        start_pos,
                        x_dec,
                        layer_feats_slice,
                        domain_steer,
                        r_step,
                    ):
                        def custom_forward(streams):
                            return module(
                                streams,
                                freqs_cis_dec,
                                start_pos,
                                x_dec,
                                layer_feats=layer_feats_slice,
                                domain_steer=domain_steer,
                                r_step=r_step,
                            )

                        return custom_forward

                    streams, z_loss, vq_loss, routing, indices, adj, event_prob = (
                        torch.utils.checkpoint.checkpoint(
                            create_custom_forward(
                                layer,
                                freqs_cis_dec,
                                start_pos,
                                x_dec,
                                layer_feats_slice,
                                domain_steer,
                                r_step,
                            ),
                            streams,
                            use_reentrant=False,  # safe for multi-loss backward (CE+KL+VQ+CIF)
                        )
                    )
                else:
                    streams, z_loss, vq_loss, routing, indices, adj, event_prob = layer(
                        streams,
                        freqs_cis_dec,
                        start_pos,
                        x_dec,
                        layer_feats=layer_feats_slice,
                        domain_steer=domain_steer,
                        r_step=r_step,
                    )

                if r_step == reasoning_steps - 1:
                    total_z_loss = total_z_loss + z_loss
                    total_vq_loss = total_vq_loss + vq_loss
                    routing_maps.append(routing)
                    concept_indices.append(indices)
                    adjacencies.append(adj)
                    if self.training and start_pos == 0 and event_prob is not None:
                        event_probs.append(event_prob)

        self.last_z_loss = total_z_loss
        self.last_moe_loss = total_z_loss
        self.last_vq_loss = total_vq_loss
        self.last_pred_loss = sum(layer.last_pred_loss for layer in self.layers)
        self.last_confidences = []

        h_final = self._hc_head_reduce(streams)
        h_normed = self.decoder_norm(h_final)

        if self.training and hasattr(self, "memory") and self.memory is not None:
            try:
                cb = self.memory.concept_blocks[0].vqs[0].embedding.weight
                h_flat = h_normed.view(-1, h_normed.size(-1)).float()
                cb_norm = F.normalize(cb.float(), dim=-1)
                h_norm = F.normalize(h_flat, dim=-1)
                sim = torch.mm(h_norm, cb_norm.t())
                p = F.softmax(sim * 4.0, dim=-1)
                entropy = -(p * torch.log(p + 1e-8)).sum(-1).mean()
                avg_p = p.mean(dim=0)
                diversity = -(avg_p * torch.log(avg_p + 1e-8)).sum()
                token_concept = entropy - diversity
            except (IndexError, AttributeError, RuntimeError):
                token_concept = torch.tensor(
                    0.0, device=x_dec.device, dtype=torch.float32
                )
        else:
            token_concept = torch.tensor(0.0, device=x_dec.device, dtype=torch.float32)
        self.last_token_concept_loss = token_concept

        # Predict attributes for the sequence from final representations
        self.last_predicted_attributes = self.superhuman_alignment_head(h_normed)

        # Apply attribute steering logit biasing if steering_vector is provided
        if steering_vector is not None:
            w_dtype = self.superhuman_alignment_head.weight.dtype
            if isinstance(steering_vector, dict):
                vector_list = [
                    steering_vector.get(attr, 0.0)
                    for attr in self.args.steering_attributes
                ]
                steering_vector = torch.tensor(
                    vector_list, device=h_normed.device, dtype=w_dtype
                )
            elif isinstance(steering_vector, (list, tuple)):
                steering_vector = torch.tensor(
                    steering_vector, device=h_normed.device, dtype=w_dtype
                )
            elif isinstance(steering_vector, torch.Tensor):
                steering_vector = steering_vector.to(
                    device=h_normed.device, dtype=w_dtype
                )

            if steering_vector.ndim == 1:
                steering_vector = steering_vector.unsqueeze(0)

            delta_h = torch.matmul(
                steering_vector, self.superhuman_alignment_head.weight
            )
            if delta_h.ndim == 2:
                delta_h = delta_h.unsqueeze(1)

            h_normed = h_normed + delta_h

        logits = F.linear(h_normed.float(), self.head.weight.float())

        # Final logit softcap (Gemma-4 style: tanh(logits / softcap) * softcap)
        # Prevents logit explosion during long generation; config-flag-gated.
        final_softcap = self.args.final_logit_softcap
        if final_softcap is not None:
            logits = torch.tanh(logits / final_softcap) * final_softcap

        # 3. Multi-Token Prediction (t+2 prediction)
        mtp_logits = None
        if self.training and N_dec > 1 and len(self.mtp) > 0:
            mtp_logits, mtp_z, mtp_vq = self.mtp[0](
                streams[:, :-1],
                freqs_cis_dec[:-1],
                x_dec[:, 1:],
                start_pos,
            )
            self.last_z_loss = self.last_z_loss + mtp_z
            self.last_vq_loss = self.last_vq_loss + mtp_vq

        return (
            logits,
            mtp_logits,
            concept_db,
            memory_state,
            routing_maps,
            concept_indices,
            adjacencies,
            event_probs,
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 0,
        pad_token: int = 1,
        max_len: Optional[int] = None,
        steering_vector: Optional[Any] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        vision_output_length: int = 280,
        domain_steer: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        device = idx.device
        actual_max_len = max_len if max_len is not None else self.max_seq_len
        cond_len = idx.shape[1]

        if cond_len < actual_max_len:
            padding = torch.full(
                (idx.shape[0], actual_max_len - cond_len),
                pad_token,
                dtype=idx.dtype,
                device=device,
            )
            idx_padded = torch.cat([padding, idx], dim=1)
        else:
            idx_padded = idx[:, -actual_max_len:]

        # Single Read pass to freeze prompt representations
        logits, _, concept_db, memory_state, *_ = self(
            idx_padded,
            idx_padded,
            start_pos=0,
            steering_vector=steering_vector,
            pixel_values=pixel_values,
            audio_values=audio_values,
            vision_output_length=vision_output_length,
            domain_steer=domain_steer,
        )

        # Autoregressive decode sampling
        last_logits = logits[:, -1, :]

        # Stability system: drift detection + adaptive temp (config-flag gated)
        if self.stability_enabled:
            drift_signals = self.drift_detector.check(logits)
            temperature = self.temp_scheduler.get_temperature(
                actual_max_len, drift_signals
            )
        else:
            # Frontier Self-Evolution: adjust temperature based on symbolic guard confidence
            if hasattr(self, "last_confidences") and self.last_confidences:
                mean_conf = (
                    torch.stack([c.mean() for c in self.last_confidences]).mean().item()
                )
                # Dynamic temperature: confidence 1.0 -> temp 0.2, confidence 0.0 -> temp 1.5
                temperature = max(0.2, min(1.5, 1.5 - mean_conf * 1.3))

        if temperature > 0:
            last_logits = last_logits / temperature
        if top_k > 0:
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[:, [-1]]] = -10000.0
        probs = torch.softmax(last_logits, dim=-1, dtype=torch.float32)
        idx_next = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(
            dim=-1, keepdim=True
        )
        idx = torch.cat([idx, idx_next], dim=1)

        current_pos = actual_max_len
        for step in range(max_new_tokens - 1):
            logits, _, _, _, *_ = self(
                x_enc=None,
                x_dec=idx_next,
                concept_db=concept_db,
                memory_state=memory_state,
                start_pos=current_pos,
                steering_vector=steering_vector,
                domain_steer=domain_steer,
            )

            last_logits = logits[:, -1, :]

            # Stability system: drift + adaptive temp (config-flag gated)
            if self.stability_enabled:
                drift_signals = self.drift_detector.check(logits)
                temperature = self.temp_scheduler.get_temperature(
                    current_pos + 1, drift_signals
                )
            else:
                # Recalculate dynamic temperature for current token
                if hasattr(self, "last_confidences") and self.last_confidences:
                    mean_conf = (
                        torch.stack([c.mean() for c in self.last_confidences])
                        .mean()
                        .item()
                    )
                    temperature = max(0.2, min(1.5, 1.5 - mean_conf * 1.3))

            if temperature > 0:
                last_logits = last_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = -10000.0

            probs = torch.softmax(last_logits, dim=-1, dtype=torch.float32)
            idx_next = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(
                dim=-1, keepdim=True
            )
            idx = torch.cat([idx, idx_next], dim=1)
            current_pos += 1

        self.train()
        return idx
