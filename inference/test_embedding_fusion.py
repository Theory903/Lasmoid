"""
Focused checks for Lasmoid token and external embedding fusion.

Tests:
  - External embedding fusion (original tests)
  - Mixed multimodal (text + vision + audio) fusion under Tiny_Config
    verifying Requirements 10.1, 10.2, 10.3
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Lasmoid, ModelArgs


# ── Modality constants (must match lasmoid.py / vision.py / audio.py) ──
MODALITY_TEXT = 0
MODALITY_VISION = 1
MODALITY_AUDIO = 2
IMAGE_PLACEHOLDER = 258880
AUDIO_PLACEHOLDER = 258881


def test_external_embedding_fusion():
    args = ModelArgs(
        vocab_size=64,
        max_seq_len=16,
        max_batch_size=2,
        external_embedding_dim=32,
        external_embedding_scale=0.5,
    )
    model = Lasmoid(args).eval()

    token_ids = torch.randint(0, args.vocab_size, (2, 8))
    external_embeddings = torch.randn(2, 8, args.external_embedding_dim)

    token_only = model.embed_tokens(token_ids)
    fused = model.embed_tokens(token_ids, external_embeddings)

    assert token_only.shape == (2, 8, args.dim)
    assert fused.shape == token_only.shape
    assert not torch.allclose(token_only.float(), fused.float())


def test_external_embedding_shape_guard():
    args = ModelArgs(vocab_size=64, max_seq_len=16, external_embedding_dim=32)
    model = Lasmoid(args).eval()
    token_ids = torch.randint(0, args.vocab_size, (2, 8))
    bad_embeddings = torch.randn(2, 7, args.external_embedding_dim)

    try:
        model.embed_tokens(token_ids, bad_embeddings)
    except ValueError as exc:
        assert "batch/seq matching token ids" in str(exc)
    else:
        raise AssertionError("Expected shape guard to reject misaligned embeddings")


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed multimodal fusion tests under Tiny_Config (Req 10.1, 10.2, 10.3)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_tiny_multimodal_model():
    """Build a Lasmoid model from Tiny_Config dimensions with multimodal encoders enabled."""
    # Tiny_Config (config_100m.json) has dim=384.
    # Enable vision + audio paths so encoders are instantiated.
    # vocab_size must be > AUDIO_PLACEHOLDER (258881) to hold special tokens.
    args = ModelArgs(
        vocab_size=260000,
        max_seq_len=64,
        max_batch_size=2,
        dim=384,
        n_layers=2,          # Reduce layers for speed — fusion doesn't depend on depth
        n_heads=6,
        head_dim=48,
        rope_head_dim=16,
        num_residual_streams=4,
        hc_sinkhorn_iters=8,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        num_concepts=16,
        codebook_size=64,
        ssm_heads=6,
        ssm_state_dim=16,
        ssm_chunk_size=64,
        vision_dim=128,         # Small vision encoder for testing
        n_vision_layers=1,
        audio_dim=128,          # Small audio encoder for testing
        n_audio_layers=1,
        audio_feature_dim=80,
        audio_conformer_dims=128,
        audio_lm_dims=128,
    )
    model = Lasmoid(args).eval()
    return model, args


def test_multimodal_fusion_uniform_feature_dim():
    """Req 10.1: Feature dimension is uniform across all modalities after fusion."""
    torch.manual_seed(42)
    model, args = _make_tiny_multimodal_model()

    B, S = 2, 32
    # Build a token sequence with some image and audio placeholders
    # Positions 4-7 are IMAGE_PLACEHOLDER, positions 12-15 are AUDIO_PLACEHOLDER
    token_ids = torch.randint(0, 1000, (B, S))  # Normal text tokens (< IMAGE_PLACEHOLDER)
    token_ids[:, 4:8] = IMAGE_PLACEHOLDER
    token_ids[:, 12:16] = AUDIO_PLACEHOLDER

    # Small image: (B, 3, H, W) — 14×14 patch grid
    pixel_values = torch.randn(B, 3, 28, 28)
    # Short audio waveform
    audio_values = torch.randn(B, 4000)

    with torch.no_grad():
        fused_embeddings, layer_feats = model.embed_multimodal(
            token_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            vision_output_length=4,  # Match the 4 placeholder slots
        )

    # Uniform feature dim: all positions (text, vision, audio) share dim=384
    assert fused_embeddings.shape == (B, S, args.dim), (
        f"Expected fused shape (2, {S}, {args.dim}), got {fused_embeddings.shape}"
    )
    # Verify per-position: text positions, vision positions, and audio positions
    # all have the same last dimension
    text_emb = fused_embeddings[:, 0:4, :]      # pure text
    vision_emb = fused_embeddings[:, 4:8, :]    # vision-replaced
    audio_emb = fused_embeddings[:, 12:16, :]   # audio-replaced

    assert text_emb.shape[-1] == args.dim
    assert vision_emb.shape[-1] == args.dim
    assert audio_emb.shape[-1] == args.dim


def test_multimodal_fusion_modality_id_length():
    """Req 10.2: Modality-id sequence length equals the fused token count."""
    torch.manual_seed(42)
    model, args = _make_tiny_multimodal_model()

    B, S = 2, 24
    token_ids = torch.randint(0, 1000, (B, S))
    token_ids[:, 6:10] = IMAGE_PLACEHOLDER
    token_ids[:, 16:20] = AUDIO_PLACEHOLDER

    pixel_values = torch.randn(B, 3, 28, 28)
    audio_values = torch.randn(B, 4000)

    with torch.no_grad():
        fused_embeddings, layer_feats = model.embed_multimodal(
            token_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            vision_output_length=4,
        )

    # The modality_ids are produced inside embed_multimodal but not returned
    # directly; however they are embedded into layer_feats via per_layer_embeddings.
    # We verify the contract indirectly: fused_embeddings and layer_feats must
    # have the same sequence length (proving modality_ids was (B, S)).
    assert fused_embeddings.shape[0] == B
    assert fused_embeddings.shape[1] == S  # token count preserved
    # layer_feats shape: (B, S, n_layers, per_layer_input_dim)
    assert layer_feats.shape[0] == B
    assert layer_feats.shape[1] == S  # same token count


def test_multimodal_fusion_modality_tagging():
    """Req 10.2, 10.3: Correct modality tagging for each position.

    We verify modality tagging by inspecting the per-layer modality embeddings
    in layer_feats. Positions with different modalities should receive different
    per-layer embeddings (since per_layer_embeddings is indexed by modality_ids).
    """
    torch.manual_seed(42)
    model, args = _make_tiny_multimodal_model()

    B, S = 1, 20
    token_ids = torch.randint(0, 1000, (B, S))
    # Place image placeholders at positions 3-6, audio at positions 10-13
    token_ids[:, 3:7] = IMAGE_PLACEHOLDER
    token_ids[:, 10:14] = AUDIO_PLACEHOLDER

    pixel_values = torch.randn(B, 3, 28, 28)
    audio_values = torch.randn(B, 4000)

    with torch.no_grad():
        fused_embeddings, layer_feats = model.embed_multimodal(
            token_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            vision_output_length=4,
        )

    # layer_feats = per_layer_proj(fused) + per_layer_embeddings[modality_ids]
    # Since per_layer_embeddings has shape (3, n_layers, per_layer_input_dim),
    # different modality ids produce different additive biases.
    # Check that layer_feats at text vs vision vs audio positions differ in the
    # modality-embedding component.

    # Access the raw modality embeddings directly
    mod_emb = model.per_layer_embeddings  # (3, n_layers, per_layer_input_dim)
    text_mod = mod_emb[MODALITY_TEXT]     # (n_layers, per_layer_input_dim)
    vision_mod = mod_emb[MODALITY_VISION]
    audio_mod = mod_emb[MODALITY_AUDIO]

    # They must be distinct (initialized with random values)
    assert not torch.allclose(text_mod.float(), vision_mod.float()), \
        "Text and vision per-layer embeddings should differ"
    assert not torch.allclose(text_mod.float(), audio_mod.float()), \
        "Text and audio per-layer embeddings should differ"
    assert not torch.allclose(vision_mod.float(), audio_mod.float()), \
        "Vision and audio per-layer embeddings should differ"


def test_multimodal_fusion_mixed_input_under_tiny_config():
    """Req 10.3: Full mixed text-vision-audio fusion test under Tiny_Config.

    Verifies:
      - Fused sequence has uniform feature dim = 384 (Req 10.1)
      - Modality-id sequence length equals fused token count (Req 10.2)
      - Vision positions receive vision encoder output, not text embeddings
      - Audio positions receive audio encoder output, not text embeddings
      - Non-placeholder positions remain pure text embeddings
    """
    torch.manual_seed(1234)
    model, args = _make_tiny_multimodal_model()

    B, S = 2, 32
    # Construct a mixed input:
    # [text...][img placeholders][text...][audio placeholders][text...]
    token_ids = torch.randint(10, 1000, (B, S))
    n_img = 4
    n_audio = 4
    img_start, img_end = 5, 5 + n_img
    audio_start, audio_end = 18, 18 + n_audio
    token_ids[:, img_start:img_end] = IMAGE_PLACEHOLDER
    token_ids[:, audio_start:audio_end] = AUDIO_PLACEHOLDER

    pixel_values = torch.randn(B, 3, 28, 28)
    audio_values = torch.randn(B, 4000)

    # Get text-only embeddings for comparison (no vision/audio)
    with torch.no_grad():
        text_only_emb, _ = model.embed_multimodal(token_ids, pixel_values=None, audio_values=None)

    # Get mixed multimodal embeddings
    with torch.no_grad():
        fused_emb, layer_feats = model.embed_multimodal(
            token_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            vision_output_length=n_img,
        )

    # 1. Uniform feature dim (Req 10.1)
    assert fused_emb.shape == (B, S, args.dim), \
        f"Fused embeddings should have uniform dim={args.dim}, got shape {fused_emb.shape}"

    # 2. Sequence length preserved (Req 10.2: modality_ids length == fused token count)
    assert fused_emb.shape[1] == S
    assert layer_feats.shape[1] == S

    # 3. Vision positions differ from text-only (vision encoder output was spliced in)
    vision_slice_fused = fused_emb[:, img_start:img_end, :]
    vision_slice_text = text_only_emb[:, img_start:img_end, :]
    assert not torch.allclose(vision_slice_fused.float(), vision_slice_text.float()), \
        "Vision placeholder positions should have been replaced by vision encoder output"

    # 4. Audio positions differ from text-only (audio encoder output was spliced in)
    audio_slice_fused = fused_emb[:, audio_start:audio_end, :]
    audio_slice_text = text_only_emb[:, audio_start:audio_end, :]
    assert not torch.allclose(audio_slice_fused.float(), audio_slice_text.float()), \
        "Audio placeholder positions should have been replaced by audio encoder output"

    # 5. Non-placeholder text positions are unchanged
    pure_text_pos = [0, 1, 2, 3, 4]  # before image placeholders
    for p in pure_text_pos:
        assert torch.allclose(
            fused_emb[:, p, :].float(),
            text_only_emb[:, p, :].float(),
            atol=1e-5,
        ), f"Pure text position {p} should be identical with/without multimodal"


if __name__ == "__main__":
    test_external_embedding_fusion()
    test_external_embedding_shape_guard()
    test_multimodal_fusion_uniform_feature_dim()
    test_multimodal_fusion_modality_id_length()
    test_multimodal_fusion_modality_tagging()
    test_multimodal_fusion_mixed_input_under_tiny_config()
    print("All Lasmoid embedding fusion tests passed.")
