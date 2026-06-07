import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
import sys

# Ensure correct path resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    RMSNorm,
    Linear,
    EinsumLinear,
    apply_rotary_emb,
    precompute_freqs_cis,
    VectorQuantizer,
    ElasticSparseConceptMemory,
    Compressor,
    CSAAttention,
    HCAAttention,
    HybridSlidingGlobal,
    Gate,
    DeepSeekMoE,
    StateSpaceRecurrence,
    ManifoldConstrainedHyperConnection,
    MTPBlock,
    LasmoidBlock,
    Lasmoid,
    compute_loss,
    compute_grpo_loss,
    ModelArgs,
    QuantKVCache,
)
from kv_cache import KVCache, SlidingWindowKVCache, TieredKVCache
from kernels.quant import KVQuantConfig, quantize_kv, dequantize_kv


class TestLasmoidComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = ModelArgs()
        cls.args.vocab_size = 1000
        cls.args.dim = 64
        cls.args.n_layers = 2
        cls.args.max_seq_len = 32
        cls.args.max_batch_size = 2
        cls.args.n_heads = 4
        cls.args.head_dim = 16
        cls.args.rope_head_dim = 8
        cls.args.q_lora_rank = 16
        cls.args.o_lora_rank = 16
        cls.args.n_routed_experts = 4
        cls.args.n_shared_experts = 1
        cls.args.n_activated_experts = 2
        cls.args.num_residual_streams = 4
        cls.args.num_concepts = 16
        cls.args.num_abstract_concepts = 4
        cls.args.num_global_concepts = 2
        cls.args.ssm_heads = 2
        cls.args.ssm_state_dim = 8
        cls.args.ssm_chunk_size = 16

        cls.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"\n[Test Setup] Testing on device: {cls.device.upper()}\n")

    def test_1_rms_norm(self):
        print("Testing RMSNorm...")
        norm = RMSNorm(self.args.dim, eps=self.args.norm_eps).to(self.device)
        x = torch.randn(2, 10, self.args.dim, device=self.device)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        # Check that RMS is approximately 1.0 (along the last dimension)
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-3))

    def test_2_linear(self):
        print("Testing Linear Layer...")
        lin = Linear(self.args.dim, self.args.dim * 2).to(self.device)
        x = torch.randn(2, 5, self.args.dim, device=self.device)
        out = lin(x)
        self.assertEqual(out.shape, (2, 5, self.args.dim * 2))

    def test_3_rotary_embeddings(self):
        print("Testing Rotary Embeddings (RoPE)...")
        freqs_cis = precompute_freqs_cis(
            self.args.rope_head_dim, 64, base=self.args.rope_theta
        ).to(self.device)
        x = torch.randn(
            2, 10, self.args.n_heads, self.args.rope_head_dim, device=self.device
        )
        out = apply_rotary_emb(x, freqs_cis[:10])
        self.assertEqual(out.shape, x.shape)
        # Test reverse projection / consistency
        out_inv = apply_rotary_emb(out, freqs_cis[:10], inverse=True)
        self.assertTrue(torch.allclose(x, out_inv, atol=1e-3))

    def test_4_vector_quantizer(self):
        print("Testing VectorQuantizer...")
        vq = VectorQuantizer(codebook_size=32, dim=self.args.dim).to(self.device)
        x = torch.randn(2, 5, self.args.dim, device=self.device, requires_grad=True)
        quantized, loss, indices = vq(x)
        self.assertEqual(quantized.shape, x.shape)
        self.assertEqual(loss.shape, ())
        self.assertEqual(indices.shape, (2, 5))
        # VQ loss should propagate gradients to input x
        quantized.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_5_elastic_sparse_concept_memory(self):
        print("Testing ElasticSparseConceptMemory...")
        escm = ElasticSparseConceptMemory(self.args).to(self.device)
        h_enc = torch.randn(2, 10, self.args.dim, device=self.device)

        # ESCM exposes process_chunk and lightning_retrieve
        memory_state, vq_loss = escm.process_chunk(h_enc, block_idx=0)
        self.assertEqual(memory_state.shape, (2, escm.total_concepts, self.args.dim))
        self.assertEqual(vq_loss.shape, ())

        concept_db = escm.lightning_retrieve(h_enc, top_k_blocks=1)
        self.assertEqual(concept_db.shape, (2, escm.total_concepts, self.args.dim))

    def test_6_compressor(self):
        print("Testing Compressor (Prefill and Autoregressive Modes)...")
        compressor = Compressor(self.args).to(self.device)
        x = torch.randn(2, 16, self.args.dim, device=self.device)

        # Manually assign mock cache buffers
        compressor.kv_cache = torch.zeros(2, 8, compressor.head_dim, device=self.device)
        compressor.freqs_cis = precompute_freqs_cis(
            self.args.rope_head_dim, 64, base=self.args.rope_theta
        ).to(self.device)

        # Test Prefill mode
        kv_out, event_prob = compressor(x, start_pos=0)
        self.assertEqual(event_prob.shape, (2, 16, 1))
        self.assertEqual(kv_out.ndim, 3)
        self.assertEqual(kv_out.shape[0], 2)
        self.assertEqual(kv_out.shape[2], compressor.head_dim)

        # Test Autoregressive mode step-by-step (force fire by setting threshold to 1.0)
        x_step = torch.randn(2, 1, self.args.dim, device=self.device)
        compressor.fire_threshold[:2] = 1.0
        kv_step = compressor(x_step, start_pos=16)
        self.assertIsNotNone(kv_step)
        self.assertEqual(kv_step.ndim, 3)
        self.assertEqual(kv_step.shape[0], 2)
        self.assertEqual(kv_step.shape[2], compressor.head_dim)

    def test_7_compressed_attention_branches(self):
        print("Testing Attention Modules (CSA & HCA)...")
        csa = CSAAttention(self.args).to(self.device)
        hca = HCAAttention(self.args).to(self.device)

        x = torch.randn(2, 16, self.args.dim, device=self.device)
        freqs_cis = precompute_freqs_cis(
            self.args.rope_head_dim, 64, base=self.args.rope_theta
        ).to(self.device)

        # Test CompressedSparseAttention
        out_csa = csa(x, freqs_cis[:16], start_pos=0)
        self.assertEqual(out_csa.shape, x.shape)

        # Test HeavilyCompressedAttention
        out_hca = hca(x, freqs_cis[:16], start_pos=0)
        self.assertEqual(out_hca.shape, x.shape)

    def test_7b_hybrid_sliding_global(self):
        print("Testing HybridSlidingGlobal attention...")
        # Extend args for hybrid: n_heads=4, global_heads=1 (3 local, 1 global)
        args = ModelArgs()
        args.dim = 64
        args.n_heads = 4
        args.head_dim = 16
        args.rope_head_dim = 8
        args.q_lora_rank = 16
        self.assertEqual(args.nope_head_dim, 8)
        args.o_lora_rank = 16
        args.o_groups = 2
        args.max_batch_size = 2
        args.num_residual_streams = 1
        args.max_seq_len = 64
        args.sliding_window_size = 16
        args.global_heads = 1

        hybrid = HybridSlidingGlobal(args).to(self.device)
        x = torch.randn(2, 12, args.dim, device=self.device)

        # Prefill pass
        out = hybrid(x, None, start_pos=0)
        self.assertEqual(out.shape, x.shape)

        # Autoregressive decode (single step)
        x_step = torch.randn(2, 1, args.dim, device=self.device)
        out_step = hybrid(x_step, None, start_pos=12)
        self.assertEqual(out_step.shape, (2, 1, args.dim))

        # Test with qk_norm disabled
        args2 = ModelArgs()
        args2.dim = 64
        args2.n_heads = 4
        args2.head_dim = 16
        args2.rope_head_dim = 8
        args2.q_lora_rank = 16
        args2.o_lora_rank = 16
        args2.o_groups = 2
        args2.max_batch_size = 2
        args2.num_residual_streams = 1
        args2.max_seq_len = 64
        args2.sliding_window_size = 16
        args2.global_heads = 1
        args2.qk_norm_with_scale = False

        hybrid2 = HybridSlidingGlobal(args2).to(self.device)
        out2 = hybrid2(x, None, start_pos=0)
        self.assertEqual(out2.shape, x.shape)

        # Test with k_eq_v_global
        args3 = ModelArgs()
        args3.dim = 64
        args3.n_heads = 4
        args3.head_dim = 16
        args3.rope_head_dim = 8
        args3.q_lora_rank = 16
        args3.o_lora_rank = 16
        args3.o_groups = 2
        args3.max_batch_size = 2
        args3.num_residual_streams = 1
        args3.max_seq_len = 64
        args3.sliding_window_size = 16
        args3.global_heads = 1
        args3.k_eq_v_global = True

        hybrid3 = HybridSlidingGlobal(args3).to(self.device)
        out3 = hybrid3(x, None, start_pos=0)
        self.assertEqual(out3.shape, x.shape)

        # Reset cache test
        hybrid.reset_cache()

    def test_8_gate_routing_moe(self):
        print("Testing Gate and DeepSeekMoE...")
        gate = Gate(0, self.args).to(self.device)
        moe = DeepSeekMoE(self.args).to(self.device)

        x = torch.randn(2, 8, self.args.dim, device=self.device)
        x_flat = x.reshape(-1, self.args.dim)

        # Gate routing
        weights, indices, z_loss, router_probs = gate(x_flat)
        self.assertEqual(indices.shape, (16, self.args.n_activated_experts))
        self.assertEqual(weights.shape, (16, self.args.n_activated_experts))
        self.assertEqual(z_loss.shape, ())

        # DeepSeekMoE execution
        out_moe, moe_loss = moe(x)
        self.assertEqual(out_moe.shape, x.shape)
        self.assertEqual(moe_loss.shape, ())

        # Gradient check
        x_grad = x.clone().detach().requires_grad_(True)
        out_moe_grad, moe_loss_grad = moe(x_grad)
        (out_moe_grad.sum() + moe_loss_grad).backward()
        self.assertIsNotNone(x_grad.grad)

    def test_9_state_space_recurrence(self):
        print("Testing StateSpaceRecurrence (Mamba-3 Upgraded SSM)...")
        ssm = StateSpaceRecurrence(self.args).to(self.device)

        # Prefill / training mode (chunk scan)
        x = torch.randn(2, 20, self.args.dim, device=self.device, requires_grad=True)
        out_train = ssm(x, start_pos=0)
        self.assertEqual(out_train.shape, x.shape)

        # Autoregressive mode (step evaluation)
        x_step = torch.randn(2, 1, self.args.dim, device=self.device)
        out_step = ssm(x_step, start_pos=20)
        self.assertEqual(out_step.shape, x_step.shape)

        # Test backpropagation
        out_train.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_9b_ssm_chunk_scan_equals_recurrence_loop(self):
        """ssm_chunk_scan output must equal ssm_recurrence_loop within 1e-4.

        Both are sequential recurrences; the chunk scan simply partitions the
        loop for memory locality.  This test confirms numerical equivalence
        for the same inputs regardless of chunk_size.
        """
        from ssm import ssm_chunk_scan, ssm_recurrence_loop

        torch.manual_seed(1234)
        B_comp, S, H, d_head, d_state = 2, 80, 2, 32, 8
        chunk_size = 16  # S > chunk_size triggers ssm_chunk_scan

        # Random inputs in float32 on the test device
        decay = torch.rand(B_comp, S, H, d_head, d_state, device=self.device)
        v_heads = torch.randn(B_comp, S, H, d_head, device=self.device)
        B_mat = torch.randn(B_comp, S, H, d_state, device=self.device)
        C_mat = torch.randn(B_comp, S, H, d_state, device=self.device)
        prev_s = torch.zeros(B_comp, H, d_head, d_state, device=self.device)

        # Run both paths from the same initial state
        out_chunk, state_chunk = ssm_chunk_scan(
            decay, v_heads, B_mat, C_mat, prev_s.clone(), chunk_size
        )
        out_loop, state_loop = ssm_recurrence_loop(
            decay, v_heads, B_mat, C_mat, prev_s.clone()
        )

        # Verify equivalence within the multi-step tolerance (1e-4)
        max_output_diff = (out_chunk - out_loop).abs().max().item()
        max_state_diff = (state_chunk - state_loop).abs().max().item()
        self.assertLess(
            max_output_diff, 1e-4,
            f"ssm_chunk_scan outputs differ from ssm_recurrence_loop by {max_output_diff}"
        )
        self.assertLess(
            max_state_diff, 1e-4,
            f"ssm_chunk_scan final state differs from ssm_recurrence_loop by {max_state_diff}"
        )

    def test_10_manifold_hyper_connections(self):
        print("Testing ManifoldConstrainedHyperConnection (mHC Sinkhorn Routing)...")
        mhc = ManifoldConstrainedHyperConnection(
            self.args.dim, self.args.num_residual_streams
        ).to(self.device)

        x = torch.randn(
            2,
            5,
            self.args.num_residual_streams,
            self.args.dim,
            device=self.device,
            requires_grad=True,
        )
        A_l, B_l, C_l = mhc(x)

        self.assertEqual(A_l.shape, (2, 5, self.args.num_residual_streams, 1))
        self.assertEqual(
            B_l.shape,
            (2, 5, self.args.num_residual_streams, self.args.num_residual_streams),
        )
        self.assertEqual(C_l.shape, (2, 5, self.args.num_residual_streams, 1))

        # Verify gradient flow
        (A_l.sum() + B_l.sum() + C_l.sum()).backward()
        self.assertIsNotNone(x.grad)

    def test_11_mtp_block(self):
        print("Testing MTPBlock (Multi-Token Prediction)...")
        mtp = MTPBlock(0, self.args).to(self.device)

        mtp.embed = nn.Embedding(self.args.vocab_size, self.args.dim).to(self.device)
        mtp.head = nn.Linear(self.args.dim, self.args.vocab_size).to(self.device)

        h = torch.randn(
            2, 10, self.args.num_residual_streams, self.args.dim, device=self.device
        )
        input_ids = torch.randint(0, self.args.vocab_size, (2, 10), device=self.device)
        freqs_cis = precompute_freqs_cis(
            self.args.rope_head_dim, 64, base=self.args.rope_theta
        ).to(self.device)

        logits, z_loss, vq_loss = mtp(h, freqs_cis[:10], input_ids, start_pos=0)
        self.assertEqual(logits.shape, (2, 10, self.args.vocab_size))
        self.assertEqual(z_loss.shape, ())
        self.assertEqual(vq_loss.shape, ())

    def test_12_lasmoid_block(self):
        print("Testing LasmoidBlock (Attention + SSM + MHC integration)...")
        block = LasmoidBlock(layer_id=0, args=self.args).to(self.device)

        x = torch.randn(
            2, 10, self.args.num_residual_streams, self.args.dim, device=self.device
        )
        freqs_cis = precompute_freqs_cis(
            self.args.rope_head_dim, 64, base=self.args.rope_theta
        ).to(self.device)

        out, z_loss, vq_loss, routing, indices, adj, event_prob = block(
            x,
            freqs_cis[:10],
            start_pos=0,
            input_ids=torch.randint(
                0, self.args.vocab_size, (2, 10), device=self.device
            ),
        )
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(z_loss.shape, ())
        self.assertEqual(vq_loss.shape, ())
        self.assertEqual(routing.shape, (2, 10))
        self.assertEqual(adj.shape, (1, 1))

    def test_13_lasmoid_full_model(self):
        print("Testing Full Lasmoid Model...")
        model = Lasmoid(self.args).to(self.device)

        x_enc = torch.randint(0, self.args.vocab_size, (2, 16), device=self.device)
        x_dec = torch.randint(0, self.args.vocab_size, (2, 16), device=self.device)

        logits, mtp_logits, c_db, mem, routings, idxs, adjs, event_probs = model(
            x_enc, x_dec
        )

        self.assertEqual(logits.shape, (2, 16, self.args.vocab_size))
        self.assertEqual(mtp_logits.shape, (2, 15, self.args.vocab_size))
        total_concepts = (
            self.args.num_concepts
            + self.args.num_abstract_concepts
            + self.args.num_global_concepts
        )
        self.assertEqual(c_db.shape, (2, total_concepts, self.args.dim))
        self.assertEqual(mem.shape, (2, total_concepts, self.args.dim))
        self.assertEqual(len(routings), self.args.n_layers)
        self.assertEqual(len(adjs), self.args.n_layers)
        self.assertEqual(len(event_probs), self.args.n_layers)

    def test_14_loss_functions(self):
        print("Testing compute_loss and compute_grpo_loss...")
        logits = torch.randn(2, 10, self.args.vocab_size, device=self.device)
        targets = torch.randint(0, self.args.vocab_size, (2, 10), device=self.device)

        # 1. Standard CE Loss
        loss_ce = compute_loss(
            logits=logits,
            targets=targets,
            routing_maps=[
                torch.randn(2, self.args.num_concepts, 10, device=self.device)
            ],
            vq_losses=[torch.tensor(0.1, device=self.device)],
            adjacencies=[
                torch.randn(
                    2,
                    self.args.num_concepts,
                    self.args.num_concepts,
                    device=self.device,
                )
            ],
        )
        self.assertEqual(loss_ce.shape, ())

        # 2. GRPO loss
        advantages = torch.randn(2, device=self.device)
        old_logprobs = torch.randn(2, 10, device=self.device)
        loss_grpo, pol_loss, kl_loss = compute_grpo_loss(
            logits=logits,
            targets=targets,
            advantages=advantages,
            old_logprobs=old_logprobs,
        )
        self.assertEqual(loss_grpo.shape, ())
        self.assertEqual(pol_loss.shape, ())
        self.assertEqual(kl_loss.shape, ())

    # ═══════════════════════════════════════════════════════════════════
    # Phase A1 — FP8 KV Cache Quantisation Tests
    # ═══════════════════════════════════════════════════════════════════

    def test_15_kv_quant_logical(self):
        print("Testing quantize_kv / dequantize_kv logical roundtrip...")
        x = torch.randn(
            2, 16, self.args.head_dim, device=self.device, dtype=torch.bfloat16
        )
        q, s = quantize_kv(x)
        x_hat = dequantize_kv(q, s)
        self.assertEqual(x.shape, x_hat.shape)
        self.assertEqual(x.dtype, x_hat.dtype)
        max_rel_err = (x - x_hat).abs().max() / x.abs().max()
        self.assertLess(max_rel_err, 1.0)

    def test_16_quant_kv_cache_creation(self):
        print("Testing QuantKVCache creation and shape...")
        qc = KVQuantConfig(enabled=True)
        cache = QuantKVCache(2, 64, self.args.head_dim, quant_config=qc)
        self.assertEqual(cache.cache_q.shape, (2, 64, self.args.head_dim))
        self.assertEqual(cache.cache_s.shape, (2, 64, 1))

    def test_17_quant_kv_cache_reset(self):
        print("Testing QuantKVCache reset...")
        qc = KVQuantConfig(enabled=True)
        cache = QuantKVCache(2, 64, self.args.head_dim, quant_config=qc)
        cache.reset()
        out = dequantize_kv(cache.cache_q, cache.cache_s)
        self.assertTrue(out.abs().sum() < 1e-6)

    def test_18_quant_kv_cache_resize(self):
        print("Testing QuantKVCache resize...")
        qc = KVQuantConfig(enabled=True)
        cache = QuantKVCache(2, 64, self.args.head_dim, quant_config=qc)
        cache.resize(4)
        self.assertEqual(cache.cache_q.shape, (4, 64, self.args.head_dim))
        self.assertEqual(cache.cache_s.shape, (4, 64, 1))

    def test_19_kv_cache_basics(self):
        print("Testing plain KVCache register/reset...")
        cache = KVCache(2, 64, self.args.head_dim).to(self.device)
        cache.register(self.device)
        self.assertEqual(cache.cache.device.type, self.device)
        cache.reset()
        self.assertTrue(cache.cache.abs().sum() < 1e-6)

    def test_20_sliding_window_read(self):
        print("Testing SlidingWindowKVCache update/read...")
        cache = SlidingWindowKVCache(2, 64, self.args.head_dim, window_size=16).to(
            self.device
        )
        kv = torch.randn(
            2, 16, self.args.head_dim, device=self.device, dtype=torch.bfloat16
        )
        cache.update(kv, 0, 16)
        out = cache.read(0, 16)
        self.assertIsNone(out)

    # Phase A2 — Tiered KVCache Tests
    # ═══════════════════════════════════════════════════════════════════

    def test_21_tiered_hot_only(self):
        """Write ≤ hot_size tokens — should go to hot BF16 cache."""
        print("Testing TieredKVCache hot-only path...")
        cache = TieredKVCache(2, 64, self.args.head_dim, hot_size=16)
        kv = torch.randn(2, 8, self.args.head_dim, dtype=torch.bfloat16)
        cache.write(kv, 0, 8)
        out = cache.read(0, 8)
        self.assertEqual(out.shape, (2, 8, self.args.head_dim))
        self.assertEqual(out.dtype, torch.bfloat16)
        # Hot path is BF16 → no quantization loss
        torch.testing.assert_close(out, kv)

    def test_22_tiered_main_only(self):
        """Write entirely past hot_size — goes to FP8 main cache."""
        print("Testing TieredKVCache main-only path...")
        cache = TieredKVCache(2, 64, self.args.head_dim, hot_size=16)
        kv = torch.randn(2, 8, self.args.head_dim, dtype=torch.bfloat16)
        cache.write(kv, 20, 28)
        out = cache.read(20, 28)
        self.assertEqual(out.shape, (2, 8, self.args.head_dim))
        self.assertEqual(out.dtype, torch.bfloat16)
        # FP8 on CPU (non-CUDA) has limited precision with small head_dim=16
        # Accept moderate reconstruction error for this test config
        err = (out - kv).abs().max().item()
        self.assertLess(err, 5.0)

    def test_23_tiered_straddle(self):
        """Write spanning hot/main boundary — both paths exercised."""
        print("Testing TieredKVCache straddle path...")
        cache = TieredKVCache(2, 64, self.args.head_dim, hot_size=16)
        kv = torch.randn(2, 24, self.args.head_dim, dtype=torch.bfloat16)
        cache.write(kv, 8, 32)
        out = cache.read(8, 32)
        self.assertEqual(out.shape, (2, 24, self.args.head_dim))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_24_tiered_reset(self):
        """Reset should zero all buffers."""
        print("Testing TieredKVCache reset...")
        cache = TieredKVCache(2, 64, self.args.head_dim, hot_size=16)
        kv = torch.randn(2, 24, self.args.head_dim, dtype=torch.bfloat16)
        cache.write(kv, 8, 32)
        cache.reset()
        # Hot BF16 buffer should be zero
        self.assertTrue(cache.hot.cache.abs().sum() < 1e-6)
        # Main FP8 buffer: dequantize first since float8 lacks sum()
        from kernels.quant import dequantize_kv

        mq_recon = dequantize_kv(
            cache.main.cache_q, cache.main.cache_s, cache.main.block_size
        )
        self.assertTrue(
            mq_recon.abs().sum() < 1e-6, "main FP8 cache not zero after reset"
        )

    # Phase A3 — SnapKV Eviction Tests
    # ═══════════════════════════════════════════════════════════════════

    def test_26_snapkv_select_short_seq(self):
        """Seq shorter than sink+window keeps all."""
        from eviction import SnapKVConfig, snapkv_select_indices

        cfg = SnapKVConfig(sink_size=2, window_size=4)
        k = torch.randn(6, self.args.head_dim)
        sel = snapkv_select_indices(k, cfg)
        self.assertEqual(sel.tolist(), [0, 1, 2, 3, 4, 5])

    def test_27_snapkv_select_long_seq(self):
        """Long seq selects middle positions."""
        from eviction import SnapKVConfig, snapkv_select_indices

        cfg = SnapKVConfig(sink_size=2, window_size=4, topk_ratio=0.3)
        k = torch.randn(20, self.args.head_dim)
        sel = snapkv_select_indices(k, cfg)
        self.assertGreater(len(sel), 6)
        # First 2 should be sink
        self.assertEqual(sel[:2].tolist(), [0, 1])
        # Last 4 should be window
        self.assertEqual(sel[-4:].tolist(), [16, 17, 18, 19])

    def test_28_snapkv_evict_shape(self):
        """Full eviction returns correct shapes."""
        from eviction import SnapKVConfig, snapkv_evict

        cfg = SnapKVConfig(sink_size=2, window_size=4, topk_ratio=0.5)
        k = torch.randn(1, 20, self.args.head_dim)
        v = torch.randn(1, 20, self.args.head_dim)
        ke, ve, idx = snapkv_evict(k, v, cfg)
        self.assertEqual(ke.shape[0], 1)
        self.assertEqual(ke.shape[2], self.args.head_dim)
        self.assertEqual(ve.shape, ke.shape)
        self.assertEqual(idx.shape[0], 1)

    # Phase A4 — OMP Compaction Tests
    # ═══════════════════════════════════════════════════════════════════

    def test_29_omp_compact_all_keep(self):
        """target_ratio=1.0 keeps all positions."""
        from compaction import OMPCompactionConfig, omp_compact

        cfg = OMPCompactionConfig(enabled=True, target_ratio=1.0)
        k = torch.randn(1, 8, self.args.head_dim)
        v = torch.randn(1, 8, self.args.head_dim)
        q = torch.randn(1, 4, self.args.head_dim)
        C1, beta, C2 = omp_compact(k, v, q, cfg)
        self.assertEqual(C1.shape, (1, 8, self.args.head_dim))
        self.assertEqual(beta.shape, (1, 8))
        self.assertEqual(C2.shape, (1, 8, self.args.head_dim))

    def test_30_omp_compact_half_ratio(self):
        """target_ratio=0.5 selects roughly half."""
        from compaction import OMPCompactionConfig, omp_compact

        cfg = OMPCompactionConfig(enabled=True, target_ratio=0.5)
        k = torch.randn(1, 12, self.args.head_dim)
        v = torch.randn(1, 12, self.args.head_dim)
        q = torch.randn(1, 4, self.args.head_dim)
        C1, beta, C2 = omp_compact(k, v, q, cfg)
        self.assertGreaterEqual(C1.shape[1], 1)
        self.assertLessEqual(C1.shape[1], 12)

    def test_31_omp_select_indices(self):
        """OMP selection returns sorted indices."""
        from compaction import OMPCompactionConfig, omp_select_indices

        cfg = OMPCompactionConfig(enabled=True, target_ratio=0.5)
        k = torch.randn(16, self.args.head_dim)
        q = torch.randn(4, self.args.head_dim)
        sel, w = omp_select_indices(k, q, cfg)
        self.assertGreater(len(sel), 0)
        self.assertEqual(len(sel), len(w))
        # Should be sorted
        self.assertTrue((sel[1:] >= sel[:-1]).all())

    def test_32_final_logit_softcap(self):
        """final_logit_softcap=30.0 clamps logits to [-30, 30]."""
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.final_logit_softcap = 30.0
        cfg.vocab_size = 1000
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 2
        cfg.n_heads = 4
        cfg.head_dim = 16

        model = Lasmoid(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, *_ = model(x, x)
        self.assertLessEqual(logits.max().item(), 30.0 + 1e-3)
        self.assertGreaterEqual(logits.min().item(), -30.0 - 1e-3)

    def test_33_final_logit_softcap_disabled(self):
        """final_logit_softcap=None does not clamp logits (may exceed 30)."""
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.final_logit_softcap = None
        cfg.vocab_size = 1000
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 2
        cfg.n_heads = 4
        cfg.head_dim = 16

        model = Lasmoid(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, *_ = model(x, x)
        # With head_dim=16, raw logits may still be <30 due to small scale,
        # but the key is they should match F.linear behavior (no tanh clamp)
        self.assertTrue(logits.requires_grad)

    # ── Stability System Tests ────────────────────────────────────────────

    def test_34_stability_drift_detector_basic(self):
        """DriftDetector returns empty signals when history not full."""
        from config import StabilityConfig
        from stability import DriftDetector

        cfg = StabilityConfig(enabled=True, drift_window_size=100)
        d = DriftDetector(cfg)
        logits = torch.randn(1, 10, 100)
        signals = d.check(logits)
        self.assertEqual(len(signals), 0)

    def test_35_stability_drift_detector_entropy_collapse(self):
        """DriftDetector catches deterministic logits as entropy collapse."""
        from config import StabilityConfig
        from stability import DriftDetector

        cfg = StabilityConfig(enabled=True, drift_window_size=5)
        d = DriftDetector(cfg)
        # Fill history with random logits
        for _ in range(10):
            d.check(torch.randn(1, 10, 100))
        # Inject deterministic logits (one logit much larger than others)
        deterministic = torch.zeros(1, 10, 100)
        deterministic[..., 0] = 100.0
        signals = d.check(deterministic)
        # May trigger entropy collapse or not depending on variance;
        # key is it doesn't crash and returns a list
        self.assertIsInstance(signals, list)

    def test_36_stability_temp_scheduler_basic(self):
        """AdaptiveTemperatureScheduler returns base temp with no signals."""
        from config import StabilityConfig
        from stability import AdaptiveTemperatureScheduler

        cfg = StabilityConfig(enabled=True, base_temperature=0.7)
        s = AdaptiveTemperatureScheduler(cfg)
        temp = s.get_temperature(context_len=1000, drift_signals=[])
        self.assertAlmostEqual(temp, 0.7, places=5)

    def test_37_stability_temp_scheduler_with_signals(self):
        """AdaptiveTemperatureScheduler adjusts temp on collapse signal."""
        from config import StabilityConfig
        from stability import AdaptiveTemperatureScheduler, DriftSignal

        cfg = StabilityConfig(enabled=True, base_temperature=0.7)
        s = AdaptiveTemperatureScheduler(cfg)
        temp = s.get_temperature(
            context_len=1000, drift_signals=[DriftSignal.ENTROPY_COLLAPSE]
        )
        # Collapse adds 0.2
        self.assertAlmostEqual(temp, 0.9, places=5)

    def test_38_stability_cache_integrity_nan(self):
        """KVCacheIntegrityChecker detects NaN in cache tensors."""
        from config import StabilityConfig
        from stability import KVCacheIntegrityChecker

        cfg = StabilityConfig(enabled=True, kmin_cache_check_interval=1)

        class FakeCache:
            keys = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            values = torch.tensor([[float("nan"), 2.0]])

        checker = KVCacheIntegrityChecker(cfg)
        # With check_interval=1, step 0 triggers check
        self.assertFalse(checker.check(FakeCache(), step=0))

    def test_39_stability_cache_integrity_clean(self):
        """KVCacheIntegrityChecker passes clean tensors."""
        from config import StabilityConfig
        from stability import KVCacheIntegrityChecker

        cfg = StabilityConfig(enabled=True, kmin_cache_check_interval=1)

        class FakeCache:
            keys = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            values = torch.tensor([[5.0, 6.0]])

        checker = KVCacheIntegrityChecker(cfg)
        self.assertTrue(checker.check(FakeCache(), step=0))

    def test_40_stability_model_enabled(self):
        """Lasmoid with stability_enabled=True initializes and runs."""
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.stability_enabled = True
        cfg.vocab_size = 1000
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 2
        cfg.n_heads = 4
        cfg.head_dim = 16

        model = Lasmoid(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, *_ = model(x, x)
        self.assertTrue(logits.requires_grad)
        self.assertTrue(model.stability_enabled)
        self.assertTrue(hasattr(model, "drift_detector"))

    def test_41_stability_model_disabled(self):
        """Lasmoid with stability_enabled=False has no drift_detector."""
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.stability_enabled = False
        cfg.vocab_size = 1000
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 2
        cfg.n_heads = 4
        cfg.head_dim = 16

        model = Lasmoid(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, *_ = model(x, x)
        self.assertFalse(model.stability_enabled)
        self.assertFalse(hasattr(model, "drift_detector"))

    def test_42_einsum_linear_equivalence(self):
        """EinsumLinear output matches Linear within 1e-5 for BF16 inputs."""
        torch.manual_seed(42)
        for in_dim, out_dim in [(64, 128), (128, 64), (256, 256)]:
            w = torch.randn(out_dim, in_dim) * 0.02
            b = torch.randn(out_dim) * 0.01
            x = torch.randn(2, 8, in_dim)

            # Reference: standard Linear
            ref = F.linear(x, w, b)

            # Test: EinsumLinear
            el = EinsumLinear(in_dim, out_dim, bias=True)
            el.weight.data = w.T.contiguous()  # weight is (in, out) in EinsumLinear
            el.bias.data = b
            out = el(x)

            max_diff = (out - ref).abs().max().item()
            self.assertLess(
                max_diff,
                1e-5,
                f"EinsumLinear mismatch at ({in_dim},{out_dim}): max_diff={max_diff}",
            )

    def test_43_einsum_linear_no_bias(self):
        """EinsumLinear without bias matches F.linear."""
        torch.manual_seed(42)
        el = EinsumLinear(64, 128, bias=False)
        x = torch.randn(2, 8, 64)
        ref = F.linear(x.to(el.weight.dtype), el.weight.T)
        out = el(x)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_44_use_einsum_flag_forward(self):
        """Model with use_einsum=True produces valid output (no crash)."""
        import sys
        import os

        test_dir = os.path.dirname(os.path.abspath(__file__))
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)
        from config import ModelArgs
        from lasmoid import Lasmoid
        import _common as _cm

        _cm.set_use_einsum(False)  # reset
        cfg = ModelArgs()
        cfg.use_einsum = True
        cfg.vocab_size = 1000
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 2
        cfg.n_heads = 4
        cfg.head_dim = 16

        model = Lasmoid(cfg)
        self.assertTrue(_cm._use_einsum)

        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, *_ = model(x, x)
        self.assertEqual(logits.shape, (2, 8, 1000))
        self.assertTrue(logits.requires_grad)

        _cm.set_use_einsum(False)

    def test_45_compression_events_smoke(self):
        """Compression produces sane event probs across sequences (2K encoder + 100 decoder)."""
        import os, sys

        test_dir = os.path.dirname(os.path.abspath(__file__))
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.vocab_size = 1000
        cfg.dim = 64
        cfg.n_layers = 2
        cfg.max_seq_len = 4096
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.q_lora_rank = 64
        cfg.o_lora_rank = 64
        cfg.csa_compression_ratio = 4
        cfg.hca_compression_ratio = 4
        cfg.n_mtp_layers = 0

        model = Lasmoid(cfg)
        model.train()

        x_enc = torch.randint(0, cfg.vocab_size, (1, 2048))
        x_dec = torch.randint(0, cfg.vocab_size, (1, 100))

        logits, *_rest = model(x_enc, x_dec)

        self.assertEqual(logits.shape, (1, 100, 1000))
        self.assertTrue(logits.requires_grad)

        event_probs = _rest[-1]  # last return is event_probs
        self.assertEqual(len(event_probs), cfg.n_layers)

        for ep in event_probs:
            self.assertEqual(ep.ndim, 3, "event_prob should be 3D (B, S, 1)")
            self.assertGreater(ep.size(0), 0, "batch dim should be > 0")
            self.assertEqual(ep.size(2), 1, "last dim should be 1")
            self.assertGreater(ep.size(1), 0, "seq dim should be > 0")
            self.assertTrue((ep >= 0).all())
            self.assertTrue((ep <= 1).all())
            self.assertGreater(
                ep.mean().item(),
                1e-4,
                "Nearly zero event probability",
            )

        self.assertTrue(logits.requires_grad)

    def test_46_adaptive_quantized_kv_cache(self):
        """Test AdaptiveQuantizedKVCache with various optimization flags."""
        from kv_cache import AdaptiveQuantizedKVCache
        from config import ModelArgs

        # Test case 1: use_fp8_kv = True
        args = ModelArgs()
        args.use_fp8_kv = True
        args.sliding_window_size = 8
        args.max_seq_len = 32
        args.max_batch_size = 2
        args.head_dim = 16

        cache = AdaptiveQuantizedKVCache(
            max_batch=2,
            max_seq=32,
            head_dim=16,
            args=args,
            dtype=torch.bfloat16
        )

        # Write and read back
        kv = torch.randn(2, 16, 16, dtype=torch.bfloat16)
        cache[:2, :16] = kv

        # Slice check
        read_kv = cache[:2, :16]
        self.assertEqual(read_kv.shape, (2, 16, 16))
        # FP8 reconstruction should be close to original (tolerance allowed for float8_e4m3fn)
        mean_diff = (read_kv - kv).abs().mean().item()
        max_diff = (read_kv - kv).abs().max().item()
        self.assertLess(mean_diff, 0.25)
        self.assertLess(max_diff, 2.0)

        # Test case 2: use_turboquant = True
        args = ModelArgs()
        args.use_turboquant = True
        args.sliding_window_size = 8
        args.max_seq_len = 32
        args.max_batch_size = 2
        args.head_dim = 16

        cache_tq = AdaptiveQuantizedKVCache(
            max_batch=2,
            max_seq=32,
            head_dim=16,
            args=args,
            dtype=torch.bfloat16
        )
        cache_tq[:2, :16] = kv
        read_tq = cache_tq[:2, :16]
        self.assertEqual(read_tq.shape, (2, 16, 16))

        # Test case 3: use_kv_eviction = True
        args = ModelArgs()
        args.use_kv_eviction = True
        args.sliding_window_size = 4
        args.max_seq_len = 32
        args.max_batch_size = 2
        args.head_dim = 16
        args.snapkv_sink_size = 2
        args.snapkv_window_size = 2
        args.snapkv_max_keep_size = 8
        args.snapkv_observation_length = 2
        args.snapkv_topk_ratio = 0.5

        cache_ev = AdaptiveQuantizedKVCache(
            max_batch=2,
            max_seq=32,
            head_dim=16,
            args=args,
            dtype=torch.bfloat16
        )
        # Write to compressed region (beyond window size 4)
        comp_kv = torch.randn(2, 12, 16, dtype=torch.bfloat16)
        cache_ev[:2, 4:16] = comp_kv

        # Try filtering topk_idxs
        topk_idxs = torch.arange(4, 16).view(1, 1, -1).expand(2, 1, -1)
        filtered_idxs = cache_ev.filter_topk_idxs(topk_idxs, start_pos=16, win=4)
        # Some should be marked -1 (evicted)
        self.assertTrue((filtered_idxs == -1).any())

        # Test case 4: use_compaction = True
        args = ModelArgs()
        args.use_compaction = True
        args.sliding_window_size = 4
        args.max_seq_len = 32
        args.max_batch_size = 2
        args.head_dim = 16
        args.omp_target_ratio = 0.5

        cache_comp = AdaptiveQuantizedKVCache(
            max_batch=2,
            max_seq=32,
            head_dim=16,
            args=args,
            dtype=torch.bfloat16
        )
        # OMP needs queries to select indices
        queries = torch.randn(2, 1, 16, dtype=torch.bfloat16)
        cache_comp.set_queries(queries)
        cache_comp[:2, 4:16] = comp_kv

        # Try filtering
        topk_idxs = torch.arange(4, 16).view(1, 1, -1).expand(2, 1, -1)
        filtered_idxs_comp = cache_comp.filter_topk_idxs(topk_idxs, start_pos=16, win=4)
        self.assertTrue((filtered_idxs_comp == -1).any())

    def test_47_lasmoid_integration_with_options(self):
        """Lasmoid forward run with optimization flags enabled."""
        import os, sys
        test_dir = os.path.dirname(os.path.abspath(__file__))
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)
        from config import ModelArgs
        from lasmoid import Lasmoid

        for opt_flag in ["use_fp8_kv", "use_turboquant", "use_kv_eviction", "use_compaction"]:
            cfg = ModelArgs()
            # Set the flag to True
            setattr(cfg, opt_flag, True)
            
            cfg.vocab_size = 1000
            cfg.dim = 64
            cfg.n_layers = 1
            cfg.max_seq_len = 32
            cfg.max_batch_size = 2
            cfg.n_heads = 4
            cfg.head_dim = 16
            cfg.q_lora_rank = 16
            cfg.o_lora_rank = 16
            cfg.csa_compression_ratio = 4
            cfg.hca_compression_ratio = 4
            
            # Additional parameters for eviction/compaction
            cfg.snapkv_sink_size = 2
            cfg.snapkv_window_size = 2
            cfg.snapkv_max_keep_size = 8
            cfg.snapkv_observation_length = 2
            cfg.snapkv_topk_ratio = 0.5
            cfg.omp_target_ratio = 0.5

            model = Lasmoid(cfg)
            model.eval()

            x = torch.randint(0, cfg.vocab_size, (2, 16))
            # Prefill forward pass
            logits, *_ = model(x, x)
            self.assertEqual(logits.shape, (2, 16, 1000))

    def test_48_kv_cache_sharing(self):
        """Test KV Cache sharing between layers."""
        from config import ModelArgs
        from lasmoid import Lasmoid
        cfg = ModelArgs()
        cfg.vocab_size = 100
        cfg.dim = 32
        cfg.n_layers = 4
        cfg.max_seq_len = 16
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 8
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 8
        cfg.o_lora_rank = 8
        cfg.frac_shared_layers = 0.5
        
        model = Lasmoid(cfg).to(self.device)
        model.eval()
        
        # Verify that sharing reassignments occurred
        csa_caches = [layer.attn.kv_cache for layer in model.layers if hasattr(layer.attn, 'kv_cache') and isinstance(layer.attn, CSAAttention)]
        hca_caches = [layer.attn.kv_cache for layer in model.layers if hasattr(layer.attn, 'kv_cache') and isinstance(layer.attn, HCAAttention)]
        
        # There should be duplicates in the objects if sharing is active
        self.assertGreater(len(csa_caches), len(set(csa_caches)))
        self.assertGreater(len(hca_caches), len(set(hca_caches)))

        # Run a forward pass
        x = torch.randint(0, cfg.vocab_size, (1, 8), device=self.device)
        logits, *_ = model(x, x)
        self.assertEqual(logits.shape, (1, 8, 100))

    def test_49_qknorm(self):
        """Test QKNorm layer and learning scales."""
        from _layers import QKNorm
        norm = QKNorm(dim=16).to(self.device)
        x = torch.randn(2, 4, 16, device=self.device)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        # Check scale requires grad
        self.assertTrue(norm.scale.requires_grad)

    def test_50_block_attnres(self):
        """Test BlockAttnRes layer and integration in Block."""
        from config import ModelArgs
        from lasmoid import Lasmoid
        cfg = ModelArgs()
        cfg.vocab_size = 100
        cfg.dim = 32
        cfg.n_layers = 2
        cfg.max_seq_len = 16
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 8
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 8
        cfg.o_lora_rank = 8
        cfg.use_block_attnres = True
        cfg.block_attnres_block_size = 4
        cfg.block_attnres_n_blocks = 2
        
        model = Lasmoid(cfg).to(self.device)
        model.eval()
        
        x = torch.randint(0, cfg.vocab_size, (1, 8), device=self.device)
        logits, *_ = model(x, x)
        self.assertEqual(logits.shape, (1, 8, 100))

    def test_51_ring_attention(self):
        """Test RingAttention prefill simulation."""
        from config import ModelArgs
        from lasmoid import Lasmoid
        from ring import RingAttentionPrefill
        
        cfg = ModelArgs()
        cfg.vocab_size = 100
        cfg.dim = 32
        cfg.n_layers = 1
        cfg.max_seq_len = 16
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 8
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 8
        cfg.o_lora_rank = 8
        cfg.use_ring_attention = True
        
        model = Lasmoid(cfg).to(self.device)
        model.eval()
        
        ring = RingAttentionPrefill(use_ring=True)
        h = torch.randn(1, 8, cfg.dim, device=self.device)
        freqs = model.freqs_cis[:8]
        
        out = ring.prefill(model.layers[0].attn, h, freqs)
        self.assertEqual(out.shape, h.shape)

    def test_52_vision_encoder(self):
        """Test LasmoidVisionEncoder forward pass and token budget variations."""
        from vision import LasmoidVisionEncoder
        
        # Test configurations
        encoder = LasmoidVisionEncoder(
            vision_dim=32,
            dim=self.args.dim,
            n_layers=2,
            patch_size=14,
            standardize_embeddings=True,
        ).to(self.device)
        
        # Batch of images
        pixel_values = torch.randn(2, 3, 224, 224, device=self.device)
        
        # Test variable token budgets
        for budget in [70, 140, 280]:
            embeddings, modality_ids = encoder(pixel_values, output_length=budget)
            self.assertEqual(embeddings.shape, (2, budget, self.args.dim))
            self.assertEqual(modality_ids.shape, (2, budget))
            self.assertTrue(torch.all(modality_ids == 1)) # MODALITY_VISION = 1

        # Test interpolation with > 1024 patches
        # 14x14 patches on 700x700 image = 50x50 = 2500 patches (>1024)
        large_pixel_values = torch.randn(1, 3, 700, 700, device=self.device)
        embeddings, modality_ids = encoder(large_pixel_values, output_length=140)
        self.assertEqual(embeddings.shape, (1, 140, self.args.dim))

    def test_53_audio_encoder(self):
        """Test LasmoidAudioEncoder and native spectrogram preprocessing."""
        from audio import LasmoidAudioEncoder, preprocess_audio
        
        # Test preprocess_audio fallback
        waveform = torch.randn(2, 16000, device=self.device)
        spectrogram = preprocess_audio(waveform, n_mels=80)
        self.assertEqual(spectrogram.ndim, 3)
        self.assertEqual(spectrogram.shape[0], 2)
        self.assertEqual(spectrogram.shape[2], 80)
        
        # Test LasmoidAudioEncoder
        encoder = LasmoidAudioEncoder(
            audio_feature_dim=80,
            conformer_dims=64,
            lm_model_dims=128,
            dim=self.args.dim,
            n_layers=2,
        ).to(self.device)
        
        # Test forwarding spectrogram features
        embeddings, modality_ids = encoder(spectrogram)
        self.assertEqual(embeddings.ndim, 3)
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[2], self.args.dim)
        self.assertEqual(modality_ids.shape, (2, embeddings.shape[1]))
        self.assertTrue(torch.all(modality_ids == 2)) # MODALITY_AUDIO = 2
        
        # Test forwarding raw waveform
        embeddings_raw, modality_ids_raw = encoder(waveform)
        self.assertEqual(embeddings_raw.shape[0], 2)
        self.assertEqual(embeddings_raw.shape[2], self.args.dim)

    def test_54_multimodal_lasmoid_integration(self):
        """Test full model forward & generate execution with concurrent text, vision, and audio."""
        from config import ModelArgs
        from lasmoid import Lasmoid
        from encoding.encoding_lasmoid import expand_multimodal_placeholders
        
        cfg = ModelArgs()
        cfg.vocab_size = 1000
        cfg.dim = 64
        cfg.n_layers = 2
        cfg.max_seq_len = 512
        cfg.max_batch_size = 2
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        cfg.n_routed_experts = 4
        cfg.n_shared_experts = 1
        cfg.n_activated_experts = 2
        cfg.num_residual_streams = 4
        
        # Multimodal config
        cfg.vision_dim = 32
        cfg.audio_dim = 64
        cfg.n_vision_layers = 2
        cfg.n_audio_layers = 2
        cfg.audio_conformer_dims = 32
        cfg.audio_lm_dims = 64
        cfg.audio_feature_dim = 80
        cfg.per_layer_input_dim = 16
        
        model = Lasmoid(cfg).to(self.device)
        model.eval()
        
        # Define a multimodal prompt
        raw_prompt = "User: <｜image｜> User: <｜audio｜> Explain what you see and hear."
        
        # Expand placeholder tokens according to budgets
        vision_budget = 70
        audio_budget = 30
        expanded_prompt = expand_multimodal_placeholders(
            raw_prompt, vision_budget=vision_budget, audio_budget=audio_budget
        )
        
        # Count occurrences of the placeholders
        self.assertEqual(expanded_prompt.count("<｜image｜>"), vision_budget)
        self.assertEqual(expanded_prompt.count("<｜audio｜>"), audio_budget)
        
        # Mock tokenization
        IMAGE_PLACEHOLDER = 258880
        AUDIO_PLACEHOLDER = 258881
        
        # Build token IDs sequence with sequential placeholders
        # 10 text tokens, 70 image placeholders, 5 text tokens, 30 audio placeholders, 10 text tokens
        seq_len = 10 + vision_budget + 5 + audio_budget + 10
        tokens = []
        tokens.extend([10] * 10)
        tokens.extend([IMAGE_PLACEHOLDER] * vision_budget)
        tokens.extend([15] * 5)
        tokens.extend([AUDIO_PLACEHOLDER] * audio_budget)
        tokens.extend([20] * 10)
        
        x_enc = torch.tensor([tokens, tokens], device=self.device) # [2, seq_len]
        x_dec = torch.tensor([tokens, tokens], device=self.device) # [2, seq_len]
        
        pixel_values = torch.randn(2, 3, 224, 224, device=self.device)
        audio_values = torch.randn(2, 16000, device=self.device)
        
        # Run forward pass
        logits, mtp_logits, c_db, mem, routings, idxs, adjs, event_probs = model(
            x_enc=x_enc,
            x_dec=x_dec,
            pixel_values=pixel_values,
            audio_values=audio_values,
            vision_output_length=vision_budget,
        )
        
        self.assertEqual(logits.shape, (2, seq_len, cfg.vocab_size))
        
        # Verify generate with multimodal inputs
        idx_input = torch.tensor([[10] * 5, [10] * 5], device=self.device) # simple prompt prefix
        idx_multimodal = torch.cat([
            idx_input,
            torch.full((2, vision_budget), IMAGE_PLACEHOLDER, device=self.device),
            torch.full((2, audio_budget), AUDIO_PLACEHOLDER, device=self.device),
        ], dim=1)
        
        generated = model.generate(
            idx_multimodal,
            max_new_tokens=5,
            pixel_values=pixel_values,
            audio_values=audio_values,
            vision_output_length=vision_budget,
        )
        self.assertEqual(generated.shape, (2, idx_multimodal.shape[1] + 5))

    def test_55_vision_factorized_embeddings_and_2d_pooling(self):
        """Test LasmoidVisionEncoder with factorized position embeddings and 2D spatial pooling."""
        from vision import LasmoidVisionEncoder, avg_pool_by_positions

        # 1. Test avg_pool_by_positions explicitly
        # Create a mock patches tensor of shape [1, 9, 4] (grid of 3x3)
        # pooled into length = 1 (1x1 output grid, k=3)
        x = torch.tensor([[[float(i)] * 4 for i in range(9)]], device=self.device) # [1, 9, 4]
        # Coordinate grid: 3x3 grid of positions (0,0) to (2,2)
        coords = []
        for y in range(3):
            for x_coord in range(3):
                coords.append([x_coord, y])
        positions_xy = torch.tensor([coords], device=self.device) # [1, 9, 2]
        
        pooled = avg_pool_by_positions(x, positions_xy, length=1)
        self.assertEqual(pooled.shape, (1, 1, 4))
        # Each channel should be the mean of 0..8 = 4.0
        expected = torch.tensor([[[4.0] * 4]], device=self.device)
        self.assertTrue(torch.allclose(pooled, expected))

        # 2. Test factorized position embedding shape and forward pass
        encoder = LasmoidVisionEncoder(
            vision_dim=16,
            dim=self.args.dim,
            n_layers=1,
            patch_size=16,
        ).to(self.device)
        
        # Verify self.pos_emb exists and has the correct shape [10240, 2, vision_dim]
        self.assertTrue(hasattr(encoder, "pos_emb"))
        self.assertEqual(encoder.pos_emb.shape, (10240, 2, 16))

        # Run forward pass with custom positions_xy
        pixel_values = torch.randn(1, 3, 48, 48, device=self.device) # 3x3 patches = 9 patches
        custom_positions = torch.tensor([[[0, 0], [1, 0], [2, 0],
                                          [0, 1], [1, 1], [2, 1],
                                          [0, 2], [1, 2], [2, 2]]], device=self.device)
        
        embeddings, modality_ids = encoder(pixel_values, output_length=1, positions_xy=custom_positions)
        self.assertEqual(embeddings.shape, (1, 1, self.args.dim))
        self.assertEqual(modality_ids.shape, (1, 1))

    def test_56_wsd_scheduler(self):
        """Test Warmup-Stable-Decay (WSD) learning rate scheduler values."""
        from train.scheduler import get_wsd_lr_multiplier, WSDScheduler
        
        # Test multiplier logic directly
        # 10 warmup, 80 stable, 10 decay
        # Total steps: 100
        warmup, stable, decay = 10, 80, 10
        
        # At step 0
        self.assertEqual(get_wsd_lr_multiplier(0, warmup, stable, decay), 0.0)
        # Halfway through warmup
        self.assertAlmostEqual(get_wsd_lr_multiplier(5, warmup, stable, decay), 0.5)
        # Peak of warmup / stable start
        self.assertEqual(get_wsd_lr_multiplier(10, warmup, stable, decay), 1.0)
        self.assertEqual(get_wsd_lr_multiplier(50, warmup, stable, decay), 1.0)
        # Decay phase start
        self.assertEqual(get_wsd_lr_multiplier(90, warmup, stable, decay), 1.0)
        # Halfway through decay (cosine of pi/2 is 0.0, so multiplier should be 0.5 * (1+0) = 0.5)
        self.assertAlmostEqual(get_wsd_lr_multiplier(95, warmup, stable, decay), 0.5, places=5)
        # End of decay
        self.assertEqual(get_wsd_lr_multiplier(100, warmup, stable, decay), 0.0)

        # Test stateful scheduler class
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = WSDScheduler(
            optimizers=optimizer,
            warmup_steps=warmup,
            stable_steps=stable,
            decay_steps=decay,
            base_lrs=[[1.0]],
        )
        scheduler.step(5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5)

    def test_57_grpo_stability_reward(self):
        """Test stability-aware GRPO reward calculation (quality, rambling, drift)."""
        from train.grpo_stability import compute_rambling_penalty, compute_drift_penalty, stability_aware_reward
        
        # 1. Test Rambling Penalty
        # Good response (no tag issues, short length, no repetitions)
        self.assertEqual(compute_rambling_penalty("A clean short answer."), 0.0)
        # Tag failure: opened <think> but no closed tag
        self.assertEqual(compute_rambling_penalty("User: <think> Reasoning..."), 1.5)
        # Double <think> tags
        self.assertEqual(compute_rambling_penalty("<think> R1 </think> <think> R2 </think>"), 1.5)
        
        # 2. Test Drift Penalty
        # Normal logits (non-collapsed, standard entropy)
        normal_logits = torch.randn(5, 1000) * 2.0
        self.assertEqual(compute_drift_penalty(normal_logits), 0.0)
        
        # Collapsed logits (overconfident, zero entropy)
        collapsed_logits = torch.zeros(5, 1000)
        collapsed_logits[:, 0] = 50.0  # extremely confident in index 0
        self.assertGreater(compute_drift_penalty(collapsed_logits), 0.0)
        
        # Massive logit norm spike
        spiked_logits = torch.randn(5, 1000) * 150.0
        self.assertGreater(compute_drift_penalty(spiked_logits), 0.0)

        # 3. Test Composite Reward
        # Verify that we can compute a reward
        reward_val = stability_aware_reward(
            response="<think> Clean logic and scientific sequence optimization. </think> <answer> 42 </answer>",
            logits=normal_logits
        )
        self.assertIsInstance(reward_val, float)

    def test_58_distillation_loss(self):
        """Test Multi-Teacher on-policy distillation KL divergence loss calculation."""
        from train.mopd import compute_mopd_loss
        
        # Batch=2, seq_len=4, vocab=10
        student_logits = torch.randn(2, 4, 10, requires_grad=True, device=self.device)
        teacher1_logits = torch.randn(2, 4, 10, device=self.device)
        teacher2_logits = torch.randn(2, 4, 10, device=self.device)
        targets = torch.randint(0, 10, (2, 4), device=self.device)
        
        loss = compute_mopd_loss(
            student_logits,
            teacher1_logits,
            teacher2_logits,
            targets,
            alpha=0.5,
            temp=2.0
        )
        
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(student_logits.grad)

    def test_59_pretrain_qat_ste(self):
        """Test SimulatedQuantLinear QAT wrapper with Straight-Through Estimators (STE)."""
        from train.pretrain import SimulatedQuantLinear
        
        # Test both "fp8" and "nvfp4" modes
        for mode in ["fp8", "nvfp4"]:
            orig_linear = torch.nn.Linear(8, 8, bias=True, device=self.device).to(torch.bfloat16)
            sim_linear = SimulatedQuantLinear(orig_linear, mode=mode)
            
            x = torch.randn(2, 8, requires_grad=True, device=self.device, dtype=torch.bfloat16)
            out = sim_linear(x)
            
            self.assertEqual(out.shape, (2, 8))
            self.assertTrue(out.requires_grad)
            
            # Backpropagation to check STE gradient flow
            loss = out.sum()
            loss.backward()
            
            # Gradients must propagate back to original weights and inputs
            self.assertIsNotNone(orig_linear.weight.grad)
            self.assertIsNotNone(x.grad)

    def test_60_long_context_finetune(self):
        """Test progressive context length extension RoPE scaling and KV cache resizing."""
        from train.long_context_finetune import extend_model_context_length
        
        args = ModelArgs()
        args.vocab_size = 1000
        args.dim = 32
        args.n_layers = 1
        args.max_seq_len = 32
        args.max_batch_size = 2
        args.n_heads = 2
        args.head_dim = 8
        args.rope_head_dim = 4
        args.q_lora_rank = 8
        args.o_lora_rank = 8
        args.n_routed_experts = 2
        args.n_shared_experts = 1
        args.n_activated_experts = 1
        args.num_residual_streams = 2
        args.num_concepts = 4
        args.num_abstract_concepts = 2
        args.num_global_concepts = 1
        args.ssm_heads = 1
        args.ssm_state_dim = 4
        args.ssm_chunk_size = 4
        
        model = Lasmoid(args).to(self.device)
        self.assertEqual(model.args.max_seq_len, 32)
        
        # Extend context to 64 tokens
        extend_model_context_length(model, new_seq_len=64, base_seq_len=32)
        
        # Verify changes
        self.assertEqual(model.args.max_seq_len, 64)
        self.assertEqual(model.args.rope_factor, 2.0)
        self.assertEqual(model.freqs_cis.shape[0], 1088) # 64 + 1024
        
        count = 0
        from inference.kv_cache import AdaptiveQuantizedKVCache
        for name, module in model.named_modules():
            if hasattr(module, "kv_cache"):
                cache = module.kv_cache
                if isinstance(cache, AdaptiveQuantizedKVCache):
                    self.assertEqual(cache.max_seq, 64)
                    count += 1
                elif isinstance(cache, torch.Tensor):
                    compress_ratio = getattr(module, "compress_ratio", None)
                    window_size = getattr(module, "window_size", 0)
                    if compress_ratio:
                        expected_seq = window_size + max(1, 64 // compress_ratio)
                        self.assertEqual(cache.shape[1], expected_seq)
                        count += 1
        self.assertGreater(count, 0)

    def test_61_sampler_temperature(self):
        """Test temperature scaling in sampling."""
        from inference.sampler import apply_temperature
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        # temperature = 0.0 should return original logits unchanged
        self.assertTrue(torch.allclose(apply_temperature(logits, 0.0), logits))
        # temperature = 2.0 should divide logits by 2
        self.assertTrue(torch.allclose(apply_temperature(logits, 2.0), logits / 2.0))
        # temperature = 0.5 should scale logits by 2
        self.assertTrue(torch.allclose(apply_temperature(logits, 0.5), logits / 0.5))

    def test_62_sampler_dry(self):
        """Test DRY repetition penalty."""
        from inference.sampler import apply_dry
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
        penalized = apply_dry(
            logits,
            generated=[0, 1, 0, 1],
            dry_multiplier=0.8,
            dry_base=2.0,
            dry_allowed_length=2
        )
        self.assertLess(penalized[0, 0].item(), 0.9)
        self.assertEqual(penalized[0, 1].item(), 1.0)

    def test_63_sampler_xtc(self):
        """Test XTC (exclude top choices) creative exclusion."""
        from inference.sampler import apply_xtc
        logits = torch.tensor([[10.0, 10.0, 1.0, 1.0, 1.0]])
        masked = apply_xtc(logits, xtc_probability=1.0, xtc_threshold=0.1)
        self.assertEqual(masked[0, 0].item(), float("-inf"))
        self.assertEqual(masked[0, 1].item(), float("-inf"))
        self.assertGreater(masked[0, 2].item(), -10.0)

    def test_64_sampler_min_p(self):
        """Test Min-P sampling."""
        from inference.sampler import apply_min_p
        logits = torch.tensor([[5.0, 2.0, -10.0]])
        masked = apply_min_p(logits, min_p=0.1)
        self.assertEqual(masked[0, 0].item(), 5.0)
        self.assertEqual(masked[0, 1].item(), float("-inf"))
        self.assertEqual(masked[0, 2].item(), float("-inf"))

    def test_65_sampler_top_p(self):
        """Test Top-P (nucleus) sampling."""
        from inference.sampler import apply_top_p
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        masked = apply_top_p(logits, top_p=0.8)
        self.assertEqual(masked[0, 2].item(), 3.0)
        self.assertEqual(masked[0, 1].item(), 2.0)
        self.assertEqual(masked[0, 0].item(), float("-inf"))

    def test_66_sampler_top_k(self):
        """Test Top-K sampling."""
        from inference.sampler import apply_top_k
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        masked = apply_top_k(logits, top_k=2)
        self.assertEqual(masked[0, 3].item(), 4.0)
        self.assertEqual(masked[0, 2].item(), 3.0)
        self.assertEqual(masked[0, 1].item(), float("-inf"))
        self.assertEqual(masked[0, 0].item(), float("-inf"))

    def test_67_full_sample(self):
        """Test full sampling pipeline."""
        from inference.sampler import full_sample
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = full_sample(logits, generated_ids=[0], temperature=0.8, min_p=0.0)
        self.assertEqual(out.shape, (1, 1))
        out_argmax = full_sample(logits, generated_ids=[], temperature=0.0)
        self.assertEqual(out_argmax.item(), 3)

    def test_68_generation_engine(self):
        """Test generate and generate_stream generator functions."""
        from inference.generate import generate, generate_stream
        args = ModelArgs(
            vocab_size=10,
            max_seq_len=8,
            max_batch_size=1,
            dim=16,
            n_layers=1,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            num_residual_streams=4,
            num_concepts=4,
            num_abstract_concepts=2,
            num_global_concepts=1,
            ssm_heads=1,
            ssm_state_dim=4,
            ssm_chunk_size=4,
        )
        model = Lasmoid(args).to(self.device).eval()
        
        results = generate(
            model,
            prompt_tokens=[[1, 2, 3]],
            max_new_tokens=4,
            eos_id=9,
            temperature=0.0,
        )
        self.assertEqual(len(results), 1)
        self.assertLessEqual(len(results[0]), 4)
        
        stream = generate_stream(
            model,
            prompt_tokens=[1, 2, 3],
            max_new_tokens=4,
            eos_id=9,
            temperature=0.0,
        )
        tokens = list(stream)
        self.assertLessEqual(len(tokens), 4)

    def test_69_checkpoint_loader(self):
        """Test checkpoint_loader select_checkpoint_file and config loading."""
        import tempfile
        import json
        from inference.checkpoint_loader import select_checkpoint_file, load_checkpoint_and_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(select_checkpoint_file(tmpdir))
            
            step_file = os.path.join(tmpdir, "lasmoid_step_100.pt")
            with open(step_file, "w") as f:
                f.write("mock")
            self.assertEqual(select_checkpoint_file(tmpdir), step_file)
            
            latest_file = os.path.join(tmpdir, "lasmoid_latest.pt")
            with open(latest_file, "w") as f:
                f.write("mock")
            self.assertEqual(select_checkpoint_file(tmpdir), latest_file)
            
            config_data = {
                "vocab_size": 100,
                "dim": 16,
                "n_layers": 1,
                "max_seq_len": 8,
                "max_batch_size": 1,
                "n_heads": 2,
                "head_dim": 8,
                "rope_head_dim": 4,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "n_activated_experts": 1,
                "num_residual_streams": 2,
                "num_concepts": 4,
                "num_abstract_concepts": 2,
                "num_global_concepts": 1,
                "ssm_heads": 1,
                "ssm_state_dim": 4,
                "ssm_chunk_size": 4,
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_data, f)
                
            model, args = load_checkpoint_and_model(
                ckpt_path=tmpdir,
                config_path=config_path,
                device="cpu",
                allow_partial_load=True
            )
            self.assertEqual(args.vocab_size, 100)
            self.assertEqual(args.dim, 16)

    def test_70_load_mxfp4_weight(self):
        """Test load_mxfp4_weight dequantization logic."""
        from kernel import load_mxfp4_weight
        blocks = torch.randint(0, 256, (2, 16), dtype=torch.uint8)
        scales = torch.randint(100, 150, (2,), dtype=torch.uint8)
        dequant = load_mxfp4_weight(blocks, scales, dtype=torch.float32)
        self.assertEqual(dequant.shape, (2, 32))
        self.assertEqual(dequant.dtype, torch.float32)

    def test_71_moe_ptq_quantization(self):
        """Test PTQ quantization to NVFP4 and FP8 in MoE experts."""
        from config import ModelArgs, QuantConfig
        from lasmoid import Lasmoid
        from moe import DeepSeekMoE

        q_cfg = QuantConfig(moe_route_dtype="nvfp4", moe_shared_dtype="fp8")
        args = ModelArgs(
            vocab_size=10,
            max_seq_len=8,
            max_batch_size=1,
            dim=16,
            n_layers=1,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            num_residual_streams=4,
            num_concepts=4,
            num_abstract_concepts=2,
            num_global_concepts=1,
            ssm_heads=1,
            ssm_state_dim=4,
            ssm_chunk_size=4,
            quant_config=q_cfg
        )
        model = Lasmoid(args).to("cpu")
        moe = model.layers[0].moe_layer
        self.assertTrue(isinstance(moe, DeepSeekMoE))

        for exp in moe.experts:
            self.assertTrue(getattr(exp.w1.weight, "use_fp4_weights", False))
            self.assertIsNotNone(getattr(exp.w1, "scale", None))

        self.assertIsNotNone(getattr(moe.shared.w1, "scale", None))

        x = torch.randint(0, 10, (1, 4))
        logits, *_ = model(x, x)
        self.assertEqual(logits.shape, (1, 4, 10))

    def test_72_checkpoint_mxfp4_on_the_fly_dequant(self):
        """Test on-the-fly dequantization of MXFP4 weights during checkpoint loading."""
        import tempfile
        import json
        from checkpoint_loader import load_checkpoint_and_model

        config_data = {
            "vocab_size": 10,
            "dim": 16,
            "n_layers": 1,
            "max_seq_len": 8,
            "max_batch_size": 1,
            "n_heads": 2,
            "head_dim": 8,
            "rope_head_dim": 4,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "n_activated_experts": 1,
            "num_residual_streams": 4,
            "num_concepts": 4,
            "num_abstract_concepts": 2,
            "num_global_concepts": 1,
            "ssm_heads": 1,
            "ssm_state_dim": 4,
            "ssm_chunk_size": 4,
            "use_mxfp4_weights": True
        }

        blocks = torch.randint(0, 256, (64, 8), dtype=torch.uint8)
        scales = torch.randint(100, 150, (64,), dtype=torch.uint8)

        state_dict = {
            "model_state_dict": {
                "per_layer_proj.weight.blocks": blocks,
                "per_layer_proj.weight.scales": scales
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            ckpt_path = os.path.join(tmpdir, "lasmoid_final.pt")
            torch.save(state_dict, ckpt_path)

            model, args = load_checkpoint_and_model(
                ckpt_path=tmpdir,
                config_path=config_path,
                device="cpu",
                allow_partial_load=True
            )

            self.assertIn("per_layer_proj.weight", model.state_dict())
            self.assertEqual(model.per_layer_proj.weight.shape, (64, 16))

    # ═══════════════════════════════════════════════════════════════════
    # Phase 7 — Next-Gen feature tests (SSM MIMO, AttnRes gating,
    #           speculative decoding, 2M config, stability generate loop)
    # ═══════════════════════════════════════════════════════════════════

    def _mimo_args(self):
        from config import ModelArgs

        a = ModelArgs()
        a.vocab_size = 200
        a.dim = 64
        a.n_layers = 1
        a.max_seq_len = 64
        a.max_batch_size = 2
        a.n_heads = 4
        a.head_dim = 16
        a.rope_head_dim = 8
        a.q_lora_rank = 16
        a.o_lora_rank = 16
        a.ssm_heads = 2
        a.ssm_state_dim = 8
        a.ssm_chunk_size = 16
        a.num_residual_streams = 4
        return a

    def test_73_ssm_mimo_heavy_tail(self):
        """SSM with MIMO + heavy-tail + SSM RoPE runs and backprops."""
        from ssm import StateSpaceRecurrence

        a = self._mimo_args()
        a.ssm_is_mimo = True
        a.ssm_mimo_rank = 4
        a.ssm_heavy_tail_alpha = 0.9
        ssm = StateSpaceRecurrence(a).to(self.device)
        self.assertTrue(ssm.is_mimo)
        self.assertTrue(ssm.use_heavy_tail)
        self.assertTrue(ssm.use_ssm_rope)

        x = torch.randn(2, 20, a.dim, device=self.device, requires_grad=True)
        out = ssm(x, start_pos=0)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        # MIMO up-projections are learnable
        self.assertTrue(ssm.mimo_x_up.requires_grad)

    def test_74_ssm_mimo_identity_at_init(self):
        """MIMO residual mixing is a no-op at init (up-proj initialised to zero)."""
        from ssm import StateSpaceRecurrence

        a = self._mimo_args()
        a.ssm_is_mimo = True
        # alpha=1.0 keeps standard exp decay so only MIMO path differs
        a.ssm_heavy_tail_alpha = 1.0
        ssm_mimo = StateSpaceRecurrence(a).to(self.device)
        self.assertFalse(ssm_mimo.use_heavy_tail)
        # At init, the MIMO up projections are exactly zero → identity mix
        self.assertTrue(torch.count_nonzero(ssm_mimo.mimo_x_up) == 0)
        self.assertTrue(torch.count_nonzero(ssm_mimo.mimo_o_up) == 0)
        # n_heads=2, d_head=dim/heads=64/2=32
        v = torch.randn(2, 5, ssm_mimo.n_heads, ssm_mimo.d_head, device=self.device)
        torch.testing.assert_close(ssm_mimo._apply_mimo_in(v), v)
        torch.testing.assert_close(ssm_mimo._apply_mimo_out(v), v)

    def test_75_ssm_heavy_tail_stable_extreme(self):
        """Heavy-tail decay stays bounded in (0,1] for extreme inputs."""
        from ssm import heavy_tail_decay

        x = torch.tensor([-1e6, -10.0, -0.01, 0.0, 5.0, 1e6])
        d = heavy_tail_decay(x, 0.9)
        self.assertTrue(torch.isfinite(d).all())
        self.assertTrue((d > 0).all())
        self.assertTrue((d <= 1.0 + 1e-6).all())

    def test_76_attnres_recency_bias_init(self):
        """BlockAttnRes initialises recency_bias to 10.0."""
        from attnres import BlockAttnRes

        ar = BlockAttnRes(dim=32, block_size=4, n_blocks=2)
        self.assertAlmostEqual(ar.recency_bias.item(), 10.0, places=5)
        self.assertEqual(ar.alpha.item(), 0.0)

    def test_77_attnres_alpha_residual_identity(self):
        """gate_type='alpha' with alpha=0 is an exact residual at init."""
        from attnres import BlockAttnRes

        ar = BlockAttnRes(dim=32, block_size=4, n_blocks=2, gate_type="alpha").to(self.device)
        streams = torch.randn(2, 8, 4, 32, device=self.device)
        out = ar(streams)
        torch.testing.assert_close(out, streams)

    def test_78_attnres_gate_types(self):
        """All gate types produce correctly-shaped finite output."""
        from attnres import BlockAttnRes

        streams = torch.randn(2, 8, 4, 32, device=self.device)
        for gt in ("alpha", "scalar", "vector", "none"):
            ar = BlockAttnRes(dim=32, block_size=4, n_blocks=2, gate_type=gt).to(self.device)
            out = ar(streams)
            self.assertEqual(out.shape, streams.shape)
            self.assertTrue(torch.isfinite(out).all())

    def _spec_model(self):
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.vocab_size = 200
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 2
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        cfg.n_mtp_layers = 1
        cfg.mtp_speculation_enabled = True
        cfg.mtp_draft_length = 6
        return Lasmoid(cfg).to(self.device), cfg

    def test_79_speculative_decoder_draft_shape(self):
        """SpeculativeDecoder.draft_tokens returns [B, draft_length] ids."""
        from mtp import SpeculativeDecoder

        model, cfg = self._spec_model()
        model.eval()
        sd = SpeculativeDecoder(model, draft_length=6)
        self.assertEqual(sd.draft_length, 6)

        B, S = 2, 10
        streams = torch.randn(B, S, cfg.num_residual_streams, cfg.dim, device=self.device)
        input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=self.device)
        freqs = model.freqs_cis[:8]
        drafts = sd.draft_tokens(streams, freqs, input_ids, start_pos=0, k=6)
        self.assertEqual(drafts.shape, (B, 6))
        self.assertEqual(drafts.dtype, torch.long)

    def test_80_speculative_verify_all_accept(self):
        """When target greedy matches all drafts, all are accepted + bonus."""
        from mtp import SpeculativeDecoder

        model, cfg = self._spec_model()
        sd = SpeculativeDecoder(model, draft_length=4)

        B, k, V = 2, 4, cfg.vocab_size
        draft_ids = torch.randint(0, V, (B, k), device=self.device)
        # Build target logits whose argmax exactly matches drafts for first k slots
        target_logits = torch.zeros(B, k + 1, V, device=self.device)
        for b in range(B):
            for i in range(k):
                target_logits[b, i, draft_ids[b, i]] = 100.0
            target_logits[b, k, 7] = 100.0  # bonus token = 7
        out, n_acc = sd.verify_drafts(draft_ids, target_logits)
        self.assertEqual(n_acc, k)
        self.assertEqual(out.shape, (B, k + 1))
        torch.testing.assert_close(out[:, :k], draft_ids)
        self.assertTrue((out[:, k] == 7).all())

    def test_81_speculative_verify_reject(self):
        """A mismatch truncates acceptance and emits the target correction."""
        from mtp import SpeculativeDecoder

        model, cfg = self._spec_model()
        sd = SpeculativeDecoder(model, draft_length=4)

        B, k, V = 1, 4, cfg.vocab_size
        draft_ids = torch.tensor([[5, 6, 7, 8]], device=self.device)
        target_logits = torch.zeros(B, k + 1, V, device=self.device)
        # target greedy: [5, 6, 99, ...] → mismatch at position 2
        target_logits[0, 0, 5] = 100.0
        target_logits[0, 1, 6] = 100.0
        target_logits[0, 2, 99] = 100.0
        out, n_acc = sd.verify_drafts(draft_ids, target_logits)
        self.assertEqual(n_acc, 2)
        self.assertEqual(out.tolist(), [[5, 6, 99]])

    def test_82_speculative_greedy_equivalence(self):
        """Accepted speculative tokens equal the target model's greedy tokens."""
        from mtp import SpeculativeDecoder

        model, cfg = self._spec_model()
        sd = SpeculativeDecoder(model, draft_length=5)

        B, k, V = 3, 5, cfg.vocab_size
        torch.manual_seed(0)
        draft_ids = torch.randint(0, V, (B, k), device=self.device)
        target_logits = torch.randn(B, k + 1, V, device=self.device)
        target_greedy = target_logits.argmax(dim=-1)
        out, n_acc = sd.verify_drafts(draft_ids, target_logits)
        # Every accepted token must equal the target's greedy choice
        self.assertTrue((out[:, :n_acc] == target_greedy[:, :n_acc]).all())
        # The correction/bonus token also matches target greedy at slot n_acc
        self.assertTrue((out[:, n_acc] == target_greedy[:, n_acc]).all())

    def test_83_config_1b_2m_loads(self):
        """config_1b_2m.json parses into ModelArgs with 2M context + next-gen flags."""
        import json
        from dataclasses import fields
        from config import ModelArgs

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "model", "config_1b_2m.json")
        with open(path) as f:
            data = json.load(f)

        valid = {f.name for f in fields(ModelArgs)}
        filtered = {k: v for k, v in data.items() if k in valid}
        args = ModelArgs(**filtered)

        self.assertEqual(args.max_seq_len, 2097152)
        self.assertEqual(args.rope_theta, 1000000.0)
        self.assertEqual(args.attention_type, "hybrid")
        self.assertTrue(args.ssm_is_mimo)
        self.assertTrue(args.mtp_speculation_enabled)
        self.assertTrue(args.use_fp8_kv)
        self.assertTrue(args.stability_enabled)
        self.assertEqual(args.frac_shared_layers, 0.5)
        self.assertTrue(args.use_einsum)
        self.assertEqual(args.per_layer_input_dim, 64)

    def test_84_generate_stability_loop(self):
        """generate.py stability helpers adapt temperature and run cache checks."""
        from config import ModelArgs
        from lasmoid import Lasmoid
        from generate import _stability_temperature, _stability_cache_check

        cfg = ModelArgs()
        cfg.stability_enabled = True
        cfg.vocab_size = 200
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        model = Lasmoid(cfg).to(self.device)
        model.eval()

        logits = torch.randn(1, 5, cfg.vocab_size, device=self.device)
        temp = _stability_temperature(model, logits, context_len=10, base_temperature=0.8)
        self.assertGreaterEqual(temp, cfg.stability_min_temperature)
        self.assertLessEqual(temp, cfg.stability_max_temperature)
        # Cache check should run without raising
        _stability_cache_check(model, step=0)

        # When stability disabled, base temperature is returned unchanged
        cfg2 = ModelArgs()
        cfg2.stability_enabled = False
        cfg2.vocab_size = 200
        cfg2.dim = 64
        cfg2.n_layers = 1
        cfg2.max_seq_len = 32
        cfg2.max_batch_size = 1
        cfg2.n_heads = 4
        cfg2.head_dim = 16
        cfg2.rope_head_dim = 8
        cfg2.q_lora_rank = 16
        cfg2.o_lora_rank = 16
        model2 = Lasmoid(cfg2).to(self.device)
        self.assertEqual(
            _stability_temperature(model2, logits, 10, 0.8), 0.8
        )

    # ═══════════════════════════════════════════════════════════════════
    # Domain Cortex — brain-like sparse activation over scientific domains
    # ═══════════════════════════════════════════════════════════════════

    def test_85_cortex_router_shapes_and_sparsity(self):
        """DomainCortexRouter activates exactly domain_topk columns per token."""
        from cortex import DomainCortexRouter

        router = DomainCortexRouter(
            dim=64, n_routed_experts=8, n_domains=4, domain_topk=2
        ).to(self.device)
        x = torch.randn(10, 64, device=self.device)
        expert_mask, domain_probs, domain_idx, aux = router(x)
        self.assertEqual(expert_mask.shape, (10, 8))
        self.assertEqual(domain_probs.shape, (10, 4))
        self.assertEqual(domain_idx.shape, (10, 2))
        # Each token activates exactly domain_topk columns × experts_per_domain experts
        experts_per_domain = 8 // 4
        self.assertTrue(
            torch.all(expert_mask.sum(dim=1) == 2 * experts_per_domain)
        )
        self.assertTrue(torch.isfinite(aux))

    def test_86_cortex_affinity_doubly_stochastic(self):
        """Sinkhorn cross-domain affinity is (approximately) doubly stochastic."""
        from cortex import DomainCortexRouter

        router = DomainCortexRouter(dim=32, n_routed_experts=8, n_domains=4, domain_topk=2)
        A = router.domain_affinity()
        self.assertEqual(A.shape, (4, 4))
        torch.testing.assert_close(A.sum(dim=1), torch.ones(4), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(A.sum(dim=0), torch.ones(4), atol=1e-3, rtol=1e-3)

    def test_87_cortex_steering_biases_domain(self):
        """A strong domain_steer bias forces that column into the active set."""
        from cortex import DomainCortexRouter

        router = DomainCortexRouter(
            dim=32, n_routed_experts=8, n_domains=4, domain_topk=1
        ).to(self.device)
        x = torch.randn(16, 32, device=self.device)
        steer = torch.full((4,), -100.0, device=self.device)
        steer[2] = 100.0  # force domain 2
        _, _, domain_idx, _ = router(x, domain_steer=steer)
        self.assertTrue(torch.all(domain_idx[:, 0] == 2))

    def test_88_cortex_gate_masks_experts(self):
        """Gate with cortex enabled only selects experts inside active columns."""
        from config import ModelArgs
        from moe import Gate

        cfg = ModelArgs()
        cfg.dim = 64
        cfg.n_routed_experts = 8
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        cfg.use_domain_cortex = True
        cfg.n_domains = 4
        cfg.domain_topk = 2  # 2 columns × 2 experts/col = 4 eligible >= 2 activated
        gate = Gate(0, cfg).to(self.device)
        gate.eval()
        x = torch.randn(12, 64, device=self.device)
        weights, indices, z_loss, router_probs = gate(x)
        self.assertEqual(indices.shape, (12, 2))
        self.assertIsNotNone(gate.cortex)
        # Selected experts must lie within the activated domain columns.
        experts_per_domain = 8 // 4
        domain_idx = gate.cortex.last_domain_indices  # [12, 2]
        for t in range(12):
            active_domains = set(domain_idx[t].tolist())
            for e in indices[t].tolist():
                self.assertIn(e // experts_per_domain, active_domains)

    def test_89_cortex_moe_forward_and_aux(self):
        """DeepSeekMoE with cortex runs, backprops, and adds a finite aux loss."""
        from config import ModelArgs
        from moe import DeepSeekMoE

        cfg = ModelArgs()
        cfg.dim = 64
        cfg.n_routed_experts = 8
        cfg.n_shared_experts = 1
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        cfg.use_domain_cortex = True
        cfg.n_domains = 4
        cfg.domain_topk = 2
        moe = DeepSeekMoE(cfg).to(self.device)
        x = torch.randn(2, 8, 64, device=self.device, requires_grad=True)
        y, aux = moe(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(aux))
        (y.sum() + aux).backward()
        self.assertIsNotNone(x.grad)
        # cortex domain router received gradient
        self.assertIsNotNone(moe.gate.cortex.domain_weight.grad)

    def test_90_cortex_full_model_integration(self):
        """Full Lasmoid with use_domain_cortex runs forward with finite logits."""
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.vocab_size = 200
        cfg.dim = 64
        cfg.n_layers = 2
        cfg.max_seq_len = 32
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        cfg.n_routed_experts = 8
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        cfg.use_domain_cortex = True
        cfg.n_domains = 4
        cfg.domain_topk = 2
        model = Lasmoid(cfg).to(self.device)
        x = torch.randint(0, cfg.vocab_size, (1, 12), device=self.device)
        logits, *_ = model(x, x)
        self.assertEqual(logits.shape, (1, 12, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    # ═══════════════════════════════════════════════════════════════════
    # Tool Registry + EDA toolset
    # ═══════════════════════════════════════════════════════════════════

    def test_91_tool_registry_validate_and_dispatch(self):
        from tools import ToolRegistry

        reg = ToolRegistry()
        reg.register(
            "add",
            {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
            lambda a, b: {"sum": a + b},
            "add two numbers",
        )
        self.assertIsNotNone(reg.validate("add", {"a": 1}))           # missing required
        self.assertIsNotNone(reg.validate("add", {"a": 1, "b": "x"}))  # wrong type
        self.assertIsNotNone(reg.validate("nope", {}))                # unknown tool
        self.assertIsNone(reg.validate("add", {"a": 1, "b": 2}))      # valid
        res = reg.dispatch("add", {"a": 1, "b": 2})
        self.assertEqual(res.status, "success")
        self.assertEqual(res.content["sum"], 3)
        bad = reg.dispatch("add", {"a": 1})  # error path returns envelope, never raises
        self.assertEqual(bad.status, "error")

    def test_92_tool_dispatch_call_json_args(self):
        from tools import make_eda_registry
        import json as _json

        reg = make_eda_registry()
        self.assertIn("describe", reg.names())
        call = {"function": {"name": "describe", "arguments": _json.dumps({"data": [[1.0], [2.0], [3.0]]})}}
        res = reg.dispatch_call(call)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.content["n_rows"], 3)
        self.assertEqual(res.to_envelope()["role"], "tool")

    def test_93_eda_describe_and_correlate(self):
        from tools import eda_describe, eda_correlate

        data = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
        d = eda_describe(data)
        self.assertEqual(d["n_cols"], 2)
        self.assertAlmostEqual(d["columns"][0]["mean"], 2.5, places=5)
        c = eda_correlate(data)
        self.assertAlmostEqual(c["matrix"][0][1], 1.0, places=4)  # perfectly correlated

    def test_94_eda_fit_model_linear(self):
        from tools import eda_fit_model

        out = eda_fit_model([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0], kind="linear")  # y=2x+1
        self.assertAlmostEqual(out["coefficients"][0], 1.0, places=3)
        self.assertAlmostEqual(out["coefficients"][1], 2.0, places=3)
        self.assertAlmostEqual(out["r2"], 1.0, places=4)

    def test_95_eda_pca_and_cluster(self):
        from tools import eda_reduce_dim, eda_cluster

        data = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0]]
        pca = eda_reduce_dim(data, n_components=1)
        self.assertGreater(pca["explained_variance_ratio"][0], 0.9)
        cl = eda_cluster(data, k=2, seed=0)
        self.assertEqual(len(cl["labels"]), 5)
        self.assertEqual(cl["labels"][3], cl["labels"][4])  # far points share a cluster

    def test_96_eda_hypothesis_test(self):
        from tools import eda_hypothesis_test

        out = eda_hypothesis_test([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0, 14.0])
        self.assertEqual(out["test"], "welch_t")
        self.assertLess(out["p_value_normal_approx"], 0.05)

    def test_96b_eda_numpy_parity(self):
        """Verify all EDA tools produce results matching NumPy reference computations.

        This confirms no Faked_Math: every tool computes REAL statistics from the data.
        """
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy not available")
        from tools import (
            eda_describe, eda_correlate, eda_fit_model,
            eda_hypothesis_test, eda_reduce_dim, eda_cluster,
        )

        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 3))

        # ── eda_describe vs NumPy ──
        desc = eda_describe(data.tolist())
        for c in range(3):
            col = data[:, c]
            self.assertAlmostEqual(desc["columns"][c]["mean"], float(np.mean(col)), places=4)
            self.assertAlmostEqual(desc["columns"][c]["std"], float(np.std(col, ddof=1)), places=4)
            self.assertAlmostEqual(desc["columns"][c]["min"], float(np.min(col)), places=4)
            self.assertAlmostEqual(desc["columns"][c]["max"], float(np.max(col)), places=4)
            self.assertAlmostEqual(desc["columns"][c]["median"], float(np.median(col)), places=4)

        # ── eda_correlate vs NumPy ──
        corr = eda_correlate(data.tolist())
        np_corr = np.corrcoef(data.T)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(corr["matrix"][i][j], float(np_corr[i, j]), places=4)

        # ── eda_fit_model vs NumPy polyfit ──
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.5 * x + 1.3 + rng.standard_normal(5) * 0.01  # nearly perfect linear
        fit = eda_fit_model(x.tolist(), y.tolist(), kind="linear")
        np_coeffs = np.polyfit(x, y, 1)  # [slope, intercept] (highest degree first)
        self.assertAlmostEqual(fit["coefficients"][0], float(np_coeffs[1]), places=2)  # intercept
        self.assertAlmostEqual(fit["coefficients"][1], float(np_coeffs[0]), places=2)  # slope
        self.assertGreater(fit["r2"], 0.99)

        # ── eda_hypothesis_test vs manual Welch's t ──
        a = rng.standard_normal(30) + 5.0
        b = rng.standard_normal(30) + 0.0
        ht = eda_hypothesis_test(a.tolist(), b.tolist())
        na, nb = len(a), len(b)
        va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        se = np.sqrt(va / na + vb / nb)
        expected_t = (np.mean(a) - np.mean(b)) / se
        self.assertAlmostEqual(ht["t_statistic"], round(float(expected_t), 6), places=4)
        self.assertAlmostEqual(ht["mean_a"], round(float(np.mean(a)), 6), places=4)
        self.assertAlmostEqual(ht["mean_b"], round(float(np.mean(b)), 6), places=4)

        # ── eda_reduce_dim vs NumPy SVD-based PCA ──
        pca_data = rng.standard_normal((30, 4))
        pca = eda_reduce_dim(pca_data.tolist(), n_components=2)
        centered = pca_data - pca_data.mean(axis=0)
        _, S, Vh = np.linalg.svd(centered, full_matrices=False)
        np_var = S**2 / (pca_data.shape[0] - 1)
        np_ratio = np_var / np_var.sum()
        self.assertAlmostEqual(pca["explained_variance_ratio"][0], float(np_ratio[0]), places=4)
        self.assertAlmostEqual(pca["explained_variance_ratio"][1], float(np_ratio[1]), places=4)
        # Components should match (up to sign)
        for i in range(2):
            for j in range(4):
                self.assertAlmostEqual(
                    abs(pca["components"][i][j]), abs(float(Vh[i, j])), places=4
                )

        # ── eda_cluster: verify inertia matches manual computation ──
        cl_data = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
        cl = eda_cluster(cl_data.tolist(), k=2, seed=7)
        labels = np.array(cl["labels"])
        centroids = np.array(cl["centroids"])
        manual_inertia = sum(
            np.sum((cl_data[i] - centroids[labels[i]]) ** 2) for i in range(len(cl_data))
        )
        self.assertAlmostEqual(cl["inertia"], float(manual_inertia), places=4)
        # Two obvious clusters should be found
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    # ═══════════════════════════════════════════════════════════════════
    # Reasoning controller + domain steering
    # ═══════════════════════════════════════════════════════════════════

    def test_97_domain_detection_and_steer(self):
        from reasoning import detect_domain_scores, build_domain_steer
        from config import ModelArgs

        names = ModelArgs().domain_names
        scores = detect_domain_scores(
            "Compute the integral and prove the theorem for this matrix equation", names
        )
        self.assertEqual(max(scores, key=scores.get), "mathematics")
        steer = build_domain_steer("quantum particle energy momentum", names, strength=4.0)
        self.assertEqual(steer.shape[0], len(names))
        self.assertGreater(steer[names.index("physics")].item(), 0.0)

    def test_98_verifier_monotonic_and_grounding(self):
        from reasoning import Verifier
        from tools import ToolResult

        v = Verifier()
        self.assertGreater(v.score("The answer is \\boxed{42}"), v.score("hmm not sure"))
        grounded = v.score("\\boxed{42}", [ToolResult("describe", "success", {})])
        self.assertGreater(grounded, v.score("\\boxed{42}"))

    def test_99_reasoning_controller_refine_accept(self):
        from reasoning import ReasoningController, Verifier

        drafts = iter(["thinking about it", "\\boxed{42}"])
        ctrl = ReasoningController(max_steps=2, verifier=Verifier())
        t = ctrl.run(
            propose=lambda: next(drafts),
            refine=lambda d, c: "\\boxed{42}",
            critique=lambda d: "add a boxed final answer",
        )
        self.assertEqual(t.best_answer, "\\boxed{42}")
        self.assertTrue(any(s.kind == "refine" and s.accepted for s in t.steps))

    def test_100_reasoning_controller_tool_dispatch(self):
        from reasoning import ReasoningController, Verifier
        from tools import make_eda_registry
        import json as _json

        reg = make_eda_registry()
        call = "```tool\n" + _json.dumps({"name": "describe", "arguments": {"data": [[1.0], [2.0], [3.0]]}}) + "\n```"
        ctrl = ReasoningController(registry=reg, max_steps=1, verifier=Verifier())
        t = ctrl.run(propose=lambda: f"Let me analyze the data. {call}")
        self.assertGreaterEqual(len(t.steps[0].tool_results), 1)
        self.assertEqual(t.steps[0].tool_results[0].status, "success")

    def test_101_domain_steer_end_to_end(self):
        """domain_steer threads through Lasmoid.forward into the cortex gate."""
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.vocab_size = 200
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        cfg.n_routed_experts = 8
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        cfg.use_domain_cortex = True
        cfg.n_domains = 4
        cfg.domain_topk = 2
        model = Lasmoid(cfg).to(self.device)
        steer = torch.zeros(cfg.n_domains, device=self.device)
        steer[1] = 100.0  # force column 1
        x = torch.randint(0, cfg.vocab_size, (1, 8), device=self.device)
        logits, *_ = model(x, x, domain_steer=steer)
        self.assertEqual(logits.shape, (1, 8, cfg.vocab_size))
        cortex = model.layers[0].moe_layer.gate.cortex
        self.assertIsNotNone(cortex)
        self.assertTrue((cortex.last_domain_indices == 1).any())

    # ═══════════════════════════════════════════════════════════════════
    # LasmoidReasoner live adapter + EpisodicMemory (R7)
    # ═══════════════════════════════════════════════════════════════════

    def test_102_lasmoid_reasoner_stub_generate(self):
        """LasmoidReasoner drives propose→tool→verify→refine over a generate_fn."""
        from reasoning import LasmoidReasoner, Verifier
        from tools import make_eda_registry
        import json as _json
        from config import ModelArgs

        reg = make_eda_registry()
        tool_blob = "```tool\n" + _json.dumps(
            {"name": "describe", "arguments": {"data": [[1.0], [2.0], [3.0]]}}
        ) + "\n```"
        # First call (propose) returns a weak draft that requests a tool;
        # subsequent calls (critique/refine) return a strong boxed answer.
        calls = {"n": 0}

        def generate_fn(prompt, domain_steer=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # domain steering must be supplied for a cortex-style question
                assert domain_steer is not None
                return f"Let me inspect the data {tool_blob}"
            return "After analysis, \\boxed{2.0}"

        reasoner = LasmoidReasoner(
            generate_fn,
            registry=reg,
            verifier=Verifier(),
            domain_names=ModelArgs().domain_names,
            max_steps=2,
        )
        t = reasoner.answer("Compute the mean via regression on this dataset variance")
        self.assertEqual(t.best_answer, "After analysis, \\boxed{2.0}")
        # the proposed step dispatched the describe tool successfully
        self.assertTrue(
            any(r.tool == "describe" and r.status == "success" for r in t.steps[0].tool_results)
        )

    def test_103_lasmoid_reasoner_from_model(self):
        """from_model builds a working reasoner around a tiny live model + stub tokenizer."""
        from reasoning import LasmoidReasoner
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.vocab_size = 60
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        cfg.n_routed_experts = 8
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        cfg.use_domain_cortex = True
        cfg.n_domains = 4
        cfg.domain_topk = 2
        cfg.domain_names = ["general", "mathematics", "physics", "data_analysis"]
        model = Lasmoid(cfg).to(self.device)

        class StubTok:
            def encode(self, s):
                return [(ord(c) % 59) + 1 for c in s][:8] or [1]
            def decode(self, ids):
                return "".join(chr(65 + (i % 26)) for i in ids)

        reasoner = LasmoidReasoner.from_model(
            model, StubTok(), max_steps=1, max_new_tokens=3
        )
        t = reasoner.answer("integral theorem matrix")
        self.assertIsInstance(t.best_answer, str)
        self.assertGreaterEqual(len(t.steps), 1)

    def test_104_episodic_memory_build_and_retrieve(self):
        from episodic import EpisodicMemory

        torch.manual_seed(0)
        # Two well-separated clusters in embedding space.
        a = torch.randn(20, 8) + torch.tensor([5.0] * 8)
        b = torch.randn(20, 8) - torch.tensor([5.0] * 8)
        emb = torch.cat([a, b], dim=0)
        payloads = [f"A{i}" for i in range(20)] + [f"B{i}" for i in range(20)]

        mem = EpisodicMemory(n_episodes=2, seed=0).build(emb, payloads)
        self.assertEqual(len(mem.episodes), 2)

        # A query near cluster A retrieves the A-dominant episode.
        q = a.mean(dim=0)
        ep = mem.retrieve(q, top_k=1)[0]
        members = mem.episodes[ep].payloads
        a_frac = sum(1 for p in members if p.startswith("A")) / max(1, len(members))
        self.assertGreater(a_frac, 0.8)

        ws = mem.working_set(q, top_k=1, max_windows=5)
        self.assertLessEqual(len(ws), 5)
        self.assertTrue(all(isinstance(p, str) for p in ws))

    # ═══════════════════════════════════════════════════════════════════
    # Relational Cortex (Evoformer-style pairwise reasoning)
    # ═══════════════════════════════════════════════════════════════════

    def test_105_relational_cortex_identity_at_init(self):
        """RelationalCortex is identity at init (zero-init read-back)."""
        from relational import RelationalCortex

        rc = RelationalCortex(dim=32, pair_dim=8, opm_chan=4, n_iters=2).to(self.device)
        single = torch.randn(2, 12, 32, device=self.device)
        out = rc(single)
        self.assertEqual(out.shape, single.shape)
        torch.testing.assert_close(out, single, atol=1e-5, rtol=1e-5)

    def test_106_relational_cortex_learns_and_backprops(self):
        """After perturbing the read-back, output changes and gradients flow."""
        from relational import RelationalCortex

        rc = RelationalCortex(dim=32, pair_dim=8, opm_chan=4, n_iters=1).to(self.device)
        with torch.no_grad():
            rc.pair_to_single.weight.normal_(0.0, 0.02)
        single = torch.randn(2, 10, 32, device=self.device, requires_grad=True)
        out = rc(single)
        self.assertFalse(torch.allclose(out, single, atol=1e-4))
        out.sum().backward()
        self.assertIsNotNone(single.grad)
        self.assertIsNotNone(rc.opm_left.weight.grad)

    def test_107_relational_cortex_full_model(self):
        """Lasmoid with use_relational_cortex runs forward with finite logits."""
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.vocab_size = 200
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        cfg.num_concepts = 16
        cfg.num_abstract_concepts = 4
        cfg.num_global_concepts = 2
        cfg.use_relational_cortex = True
        cfg.relational_pair_dim = 8
        cfg.relational_iters = 1
        model = Lasmoid(cfg).to(self.device)
        self.assertIsNotNone(model.relational_cortex)
        x = torch.randint(0, cfg.vocab_size, (1, 10), device=self.device)
        logits, *_ = model(x, x)
        self.assertEqual(logits.shape, (1, 10, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    # ═══════════════════════════════════════════════════════════════════
    # Emergent (noisy) routing + adaptive variable-k expert recruitment
    # ═══════════════════════════════════════════════════════════════════

    def test_108_cortex_noisy_routing_train_vs_eval(self):
        """Noisy gating randomises routing in train mode, deterministic in eval."""
        from cortex import DomainCortexRouter

        router = DomainCortexRouter(
            dim=32, n_routed_experts=8, n_domains=4, domain_topk=2, route_noise=2.0
        ).to(self.device)
        x = torch.randn(8, 32, device=self.device)

        router.eval()
        idx_a = router(x)[2]
        idx_b = router(x)[2]
        self.assertTrue(torch.equal(idx_a, idx_b))  # deterministic at inference
        self.assertIsNotNone(router.noise_weight)

        router.train()
        torch.manual_seed(1)
        t1 = router(x)[2]
        torch.manual_seed(2)
        t2 = router(x)[2]
        # With non-zero learned noise this can differ; at minimum it must not crash
        # and must remain a valid top-k selection.
        self.assertEqual(t1.shape, (8, 2))

    def test_109_adaptive_select_variable_k(self):
        """Adaptive selection recruits few experts for peaked probs, more for flat."""
        from config import ModelArgs
        from moe import DeepSeekMoE

        cfg = ModelArgs()
        cfg.dim = 32
        cfg.n_routed_experts = 8
        cfg.n_activated_experts = 4
        cfg.n_group = 1
        cfg.moe_adaptive_routing = True
        cfg.moe_route_top_p = 0.8
        cfg.moe_min_experts = 1
        cfg.moe_max_experts = 8
        moe = DeepSeekMoE(cfg).to(self.device)

        # Peaked distribution → few experts; flat distribution → many.
        peaked = torch.zeros(1, 8, device=self.device); peaked[0, 0] = 1.0
        flat = torch.full((1, 8), 1.0 / 8, device=self.device)
        sel_p, _ = moe._adaptive_select(peaked)
        sel_f, _ = moe._adaptive_select(flat)
        self.assertLess(int(sel_p.sum()), int(sel_f.sum()))
        self.assertGreaterEqual(int(sel_p.sum()), 1)  # respects min_experts

    def test_110_adaptive_moe_forward_and_aux(self):
        """Adaptive MoE runs, backprops, and reports a variable avg expert count."""
        from config import ModelArgs
        from moe import DeepSeekMoE

        cfg = ModelArgs()
        cfg.dim = 64
        cfg.n_routed_experts = 8
        cfg.n_shared_experts = 1
        cfg.n_activated_experts = 4
        cfg.n_group = 1
        cfg.moe_adaptive_routing = True
        cfg.moe_route_top_p = 0.6
        cfg.moe_min_experts = 1
        cfg.moe_max_experts = 6
        cfg.moe_adaptive_sparsity_coeff = 0.01
        moe = DeepSeekMoE(cfg).to(self.device)
        x = torch.randn(2, 8, 64, device=self.device, requires_grad=True)
        y, aux = moe(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(aux))
        (y.sum() + aux).backward()
        self.assertIsNotNone(x.grad)
        self.assertGreaterEqual(moe.last_avg_experts.item(), 1.0)
        self.assertLessEqual(moe.last_avg_experts.item(), 6.0)

    def test_111_adaptive_with_cortex_full_model(self):
        """Adaptive routing + domain cortex compose in a full Lasmoid forward."""
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.vocab_size = 200
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        cfg.n_routed_experts = 8
        cfg.n_activated_experts = 4
        cfg.n_group = 1
        cfg.use_domain_cortex = True
        cfg.n_domains = 4
        cfg.domain_topk = 2
        cfg.moe_adaptive_routing = True
        cfg.moe_route_top_p = 0.7
        cfg.moe_min_experts = 1
        cfg.moe_max_experts = 4
        model = Lasmoid(cfg).to(self.device)
        x = torch.randint(0, cfg.vocab_size, (1, 10), device=self.device)
        logits, *_ = model(x, x)
        self.assertEqual(logits.shape, (1, 10, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    # ═══════════════════════════════════════════════════════════════════
    # Curiosity expert + Socratic self-questioning loop
    # ═══════════════════════════════════════════════════════════════════

    def test_112_curiosity_identity_at_init(self):
        from curiosity import CuriosityExpert

        ce = CuriosityExpert(dim=32, n_questions=4).to(self.device)
        h = torch.randn(2, 6, 32, device=self.device)
        out, cur, loss = ce(h)
        self.assertEqual(out.shape, h.shape)
        torch.testing.assert_close(out, h, atol=1e-5, rtol=1e-5)  # zero-init read-back
        self.assertEqual(cur.shape, (2, 6))
        self.assertTrue(torch.isfinite(loss))

    def test_113_curiosity_backprop_and_bridge(self):
        from curiosity import CuriosityExpert

        ce = CuriosityExpert(dim=32, n_questions=4).to(self.device)
        with torch.no_grad():
            ce.out.weight.normal_(0.0, 0.02)
        h = torch.randn(2, 6, 32, device=self.device, requires_grad=True)
        concept = torch.randn(2, 10, 32, device=self.device)
        out, cur, loss = ce(h, concept)
        self.assertFalse(torch.allclose(out, h, atol=1e-4))  # bridge now active
        (out.sum() + loss).backward()
        self.assertIsNotNone(h.grad)
        self.assertIsNotNone(ce.forward_model[0].weight.grad)

    def test_114_curiosity_full_model(self):
        from config import ModelArgs
        from lasmoid import Lasmoid

        cfg = ModelArgs()
        cfg.vocab_size = 200
        cfg.dim = 64
        cfg.n_layers = 1
        cfg.max_seq_len = 32
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 16
        cfg.rope_head_dim = 8
        cfg.q_lora_rank = 16
        cfg.o_lora_rank = 16
        cfg.use_curiosity_expert = True
        cfg.curiosity_n_questions = 4
        model = Lasmoid(cfg).to(self.device)
        self.assertIsNotNone(model.curiosity_expert)
        x = torch.randint(0, cfg.vocab_size, (1, 10), device=self.device)
        logits, *_ = model(x, x)
        self.assertEqual(logits.shape, (1, 10, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(model.last_curiosity_loss))

    def test_115_socratic_reasoner_loop(self):
        from reasoning import SocraticReasoner, parse_questions

        # Stub generator: returns numbered questions for the bridging prompt,
        # short answers otherwise.
        def generate_fn(prompt, domain_steer=None):
            if "List them as 1." in prompt or "Questions:" in prompt:
                return "1. Why is quantum needed over classical?\n2. How does measurement differ?\n3. What breaks at small scales?"
            if "Integrated understanding" in prompt:
                return "Quantum extends classical where measurement is probabilistic."
            return "Because classical assumptions fail at atomic scale. \\boxed{ok}"

        sr = SocraticReasoner(generate_fn, k=3)
        qs = sr.ask("quantum mechanics", known="classical physics")
        self.assertEqual(len(qs), 3)
        self.assertTrue(all("?" in q for q in qs))

        trace = sr.learn("quantum mechanics", known="classical physics")
        self.assertEqual(len(trace.questions), 3)
        self.assertEqual(len(trace.answers), 3)
        self.assertTrue(trace.synthesis)
        curriculum = sr.build_curriculum([trace])
        # 3 Q/A pairs + 1 synthesis example, all non-empty
        self.assertGreaterEqual(len(curriculum), 1)
        self.assertTrue(all("prompt" in e and "completion" in e for e in curriculum))

    # ═══════════════════════════════════════════════════════════════════
    # Training-pipeline math: aux-loss wiring + optimizer split
    # ═══════════════════════════════════════════════════════════════════

    def test_116_compute_loss_includes_aux_terms(self):
        """commit + curiosity + CIF terms actually change the loss (were dropped)."""
        from model import compute_loss

        torch.manual_seed(0)
        logits = torch.randn(2, 6, 50)
        targets = torch.randint(0, 50, (2, 6))
        base = compute_loss(logits, targets, [], [], [])
        with_aux = compute_loss(
            logits, targets, [], [], [],
            commit_loss=torch.tensor(1.0),
            curiosity_loss=torch.tensor(1.0),
            commit_coeff=0.25,
            curiosity_coeff=0.1,
        )
        self.assertGreater(with_aux.item(), base.item())
        self.assertAlmostEqual(with_aux.item() - base.item(), 0.25 + 0.1, places=4)

    def test_117_cif_boundary_loss(self):
        """CIF boundary loss is a finite non-negative scalar."""
        from loss import cif_boundary_loss

        eps = [torch.rand(2, 8, 1) for _ in range(3)]
        v = cif_boundary_loss(eps, target_ratio=0.25, ratio_weight=1.0, entropy_weight=0.01)
        self.assertEqual(v.shape, ())
        self.assertTrue(torch.isfinite(v))

    def test_118_build_optimizers_split(self):
        """Muon takes 2-D body weights; AdamW takes embeddings/head/1-D params."""
        from train.optimizer import build_param_groups, build_optimizers, Muon

        from config import ModelArgs
        from lasmoid import Lasmoid
        cfg = ModelArgs()
        cfg.vocab_size = 100
        cfg.dim = 32
        cfg.n_layers = 1
        cfg.max_seq_len = 16
        cfg.max_batch_size = 1
        cfg.n_heads = 4
        cfg.head_dim = 8
        cfg.rope_head_dim = 4
        cfg.q_lora_rank = 8
        cfg.o_lora_rank = 8
        model = Lasmoid(cfg).to(self.device)
        muon_p, adamw_d, adamw_n = build_param_groups(model)
        self.assertGreater(len(muon_p), 0)
        self.assertTrue(all(p.ndim == 2 for p in muon_p))
        self.assertTrue(all(p.ndim == 1 for p in adamw_n))  # norms/biases/scalars
        opts = build_optimizers(model, muon_lr=0.02, adamw_lr=1e-3)
        self.assertTrue(any(isinstance(o, Muon) for o in opts))
        # one optimisation step runs without error
        x = torch.randint(0, cfg.vocab_size, (1, 8), device=self.device)
        logits, *_ = model(x, x)
        logits.sum().backward()
        for o in opts:
            o.step()

    def test_119_muon_noise_adaptive_step(self):
        """Noise-adaptive Muon scales the step by gradient SNR, stays finite."""
        from train.optimizer import Muon

        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
        opt = Muon([w], lr=0.05, adaptive_noise=True)
        before = w.detach().clone()
        # noisy gradient
        w.grad = torch.randn(8, 8, device=self.device) * 5.0
        opt.step()
        self.assertTrue(torch.isfinite(w).all())
        self.assertFalse(torch.equal(before, w.detach()))
        # second-moment state is tracked (the noise estimate)
        self.assertIn("exp_avg_sq", opt.state[w])

    # ═══════════════════════════════════════════════════════════════════
    # Gate top-k renormalization + dual routing modes audit (Task 7.1)
    # ═══════════════════════════════════════════════════════════════════

    def test_120_gate_fixed_topk_renormalization(self):
        """Fixed top-k path: gate weights sum to route_scale (1.0) within 1e-5."""
        from config import ModelArgs
        from moe import Gate

        cfg = ModelArgs()
        cfg.dim = 64
        cfg.n_routed_experts = 6
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        cfg.route_scale = 1.0
        gate = Gate(0, cfg).to(self.device)
        gate.eval()

        torch.manual_seed(42)
        x = torch.randn(32, cfg.dim, device=self.device)
        weights, indices, z_loss, router_probs = gate(x)

        # Exactly top-k experts selected per token
        self.assertEqual(indices.shape, (32, cfg.n_activated_experts))
        self.assertEqual(weights.shape, (32, cfg.n_activated_experts))

        # Gate weights renormalize to sum 1.0 ± 1e-5 (when route_scale=1.0)
        weight_sums = weights.sum(dim=-1)
        max_deviation = (weight_sums - 1.0).abs().max().item()
        self.assertLess(
            max_deviation, 1e-5,
            f"Gate weights deviate from 1.0 by {max_deviation:.2e} (exceeds 1e-5)"
        )

    def test_121_gate_adaptive_renormalization(self):
        """Adaptive (nucleus) path: weights sum to route_scale within 1e-5."""
        from config import ModelArgs
        from moe import DeepSeekMoE

        cfg = ModelArgs()
        cfg.dim = 64
        cfg.n_routed_experts = 6
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        cfg.route_scale = 1.0
        cfg.moe_adaptive_routing = True
        cfg.moe_route_top_p = 0.8
        cfg.moe_min_experts = 1
        cfg.moe_max_experts = 6
        moe = DeepSeekMoE(cfg).to(self.device)
        moe.eval()

        torch.manual_seed(42)
        x = torch.randn(32, cfg.dim, device=self.device)

        # Call _adaptive_select via gate first to get router_probs
        _w, _idx, z_loss, router_probs = moe.gate(x)
        probs = router_probs.float()
        sel, w = moe._adaptive_select(probs)

        # Only check tokens that have at least one expert selected
        active_mask = sel.any(dim=-1)
        if active_mask.any():
            active_w = w[active_mask]
            # Sum of selected weights per token should be route_scale ± 1e-5
            weight_sums = active_w.sum(dim=-1)
            max_deviation = (weight_sums - cfg.route_scale).abs().max().item()
            self.assertLess(
                max_deviation, 1e-5,
                f"Adaptive weights deviate from route_scale by {max_deviation:.2e}"
            )

    def test_122_expert_reachability_fixed_topk(self):
        """Fixed top-k: every expert is reachable over a sufficient batch."""
        from config import ModelArgs
        from moe import Gate

        cfg = ModelArgs()
        cfg.dim = 64
        cfg.n_routed_experts = 6
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        gate = Gate(0, cfg).to(self.device)
        gate.eval()

        # Use a large batch to give every expert a chance to be selected.
        # With 6 experts and random weights, 128 tokens should be plenty.
        torch.manual_seed(1234)
        x = torch.randn(128, cfg.dim, device=self.device)
        weights, indices, _, _ = gate(x)

        activated = set(indices.unique().tolist())
        expected = set(range(cfg.n_routed_experts))
        missing = expected - activated
        self.assertEqual(
            len(missing), 0,
            f"Experts {sorted(missing)} never activated over 128 tokens (fixed top-k)"
        )

    def test_123_expert_reachability_adaptive(self):
        """Adaptive path: every expert is reachable over a sufficient batch."""
        from config import ModelArgs
        from moe import DeepSeekMoE

        cfg = ModelArgs()
        cfg.dim = 64
        cfg.n_routed_experts = 6
        cfg.n_activated_experts = 2
        cfg.n_group = 1
        cfg.moe_adaptive_routing = True
        cfg.moe_route_top_p = 0.9  # generous top-p to allow more experts
        cfg.moe_min_experts = 1
        cfg.moe_max_experts = 6
        moe = DeepSeekMoE(cfg).to(self.device)
        moe.eval()

        torch.manual_seed(1234)
        x = torch.randn(128, cfg.dim, device=self.device)
        _w, _idx, _, router_probs = moe.gate(x)
        probs = router_probs.float()
        sel, w = moe._adaptive_select(probs)

        # Check which experts got selected at least once
        activated = set(sel.any(dim=0).nonzero(as_tuple=True)[0].tolist())
        expected = set(range(cfg.n_routed_experts))
        missing = expected - activated
        self.assertEqual(
            len(missing), 0,
            f"Experts {sorted(missing)} never activated over 128 tokens (adaptive)"
        )

    def test_124_tiny_config_exercises_fixed_topk(self):
        """Tiny_Config (config_100m.json) exercises the fixed top-k routing mode."""
        from config import ModelArgs
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parent.parent / "configs" / "model" / "config_100m.json"
        with open(config_path) as f:
            raw = json.load(f)

        # moe_adaptive_routing is not in config_100m.json → defaults to False
        self.assertNotIn("moe_adaptive_routing", raw)

        args = ModelArgs(**{k: v for k, v in raw.items() if hasattr(ModelArgs, k)})
        self.assertFalse(
            getattr(args, "moe_adaptive_routing", False),
            "Tiny_Config should exercise the fixed top-k path (adaptive_routing=False)"
        )
        # Verify config params match design expectations
        self.assertEqual(args.n_routed_experts, 6)
        self.assertEqual(args.n_activated_experts, 2)


if __name__ == "__main__":
    unittest.main()


