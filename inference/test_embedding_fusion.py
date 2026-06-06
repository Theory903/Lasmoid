"""
Focused checks for Lasmoid token and external embedding fusion.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Lasmoid, ModelArgs


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


if __name__ == "__main__":
    test_external_embedding_fusion()
    test_external_embedding_shape_guard()
    print("Lasmoid embedding fusion tests passed.")
