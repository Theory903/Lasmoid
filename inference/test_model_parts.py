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
    apply_rotary_emb,
    precompute_freqs_cis,
    VectorQuantizer,
    ElasticSparseConceptMemory,
    Compressor,
    CSAAttention,
    HCAAttention,
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
)


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

        out, z_loss, vq_loss, routing, indices, adj = block(
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


if __name__ == "__main__":
    unittest.main()
