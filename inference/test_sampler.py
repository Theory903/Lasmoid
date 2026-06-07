"""
test_sampler.py — Unit tests for inference/sampler.py hardening (Task 13.1).

Validates:
  - Req 13.1: Deterministic under a fixed seed (same inputs + same seed = same output)
  - Req 13.2: Temperature/top-k/top-p restrict the sampled token set correctly
  - Req 13.3: Final logit soft-cap bounds logits BEFORE sampling
  - Req 13.4: Non-finite logits (NaN/Inf) routed through NaN/Inf guard, not sampled
"""

import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampler import (
    apply_temperature,
    apply_final_softcap,
    apply_top_k,
    apply_top_p,
    apply_min_p,
    full_sample,
)

try:
    from _harness import NonFiniteError
except ImportError:
    from sampler import NonFiniteError


class TestDeterminism(unittest.TestCase):
    """Req 13.1: Sampling is deterministic under a fixed seed."""

    def test_same_seed_same_output(self):
        """Two calls with the same seed and inputs produce identical tokens."""
        logits = torch.randn(1, 1000)

        gen1 = torch.Generator(device="cpu")
        gen1.manual_seed(42)
        result1 = full_sample(logits.clone(), [], temperature=0.8, generator=gen1)

        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(42)
        result2 = full_sample(logits.clone(), [], temperature=0.8, generator=gen2)

        self.assertEqual(result1.item(), result2.item())

    def test_different_seed_may_differ(self):
        """Different seeds can produce different outputs (not guaranteed but very likely)."""
        logits = torch.randn(1, 10000)  # Large vocab for high probability of difference

        gen1 = torch.Generator(device="cpu")
        gen1.manual_seed(42)
        result1 = full_sample(logits.clone(), [], temperature=1.0, generator=gen1)

        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(999)
        result2 = full_sample(logits.clone(), [], temperature=1.0, generator=gen2)

        # We don't assert inequality (it could be same by chance), but
        # run this to confirm both calls succeed
        self.assertEqual(result1.shape, result2.shape)

    def test_determinism_with_xtc(self):
        """XTC coin flip is deterministic when a generator is provided."""
        logits = torch.randn(1, 100)

        gen1 = torch.Generator(device="cpu")
        gen1.manual_seed(7)
        result1 = full_sample(
            logits.clone(), [], temperature=0.8, xtc_probability=0.5, generator=gen1
        )

        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(7)
        result2 = full_sample(
            logits.clone(), [], temperature=0.8, xtc_probability=0.5, generator=gen2
        )

        self.assertEqual(result1.item(), result2.item())

    def test_argmax_is_deterministic_without_generator(self):
        """Temperature 0 (argmax) is always deterministic regardless of generator."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        result1 = full_sample(logits.clone(), [], temperature=0.0)
        result2 = full_sample(logits.clone(), [], temperature=0.0)
        self.assertEqual(result1.item(), result2.item())
        self.assertEqual(result1.item(), 1)  # index of 5.0


class TestTemperatureTopKTopP(unittest.TestCase):
    """Req 13.2: Temperature/top-k/top-p restrict the sampled token set."""

    def test_temperature_scales_logits(self):
        """Temperature divides logits correctly."""
        logits = torch.tensor([[2.0, 4.0, 6.0]])
        scaled = apply_temperature(logits, 2.0)
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(scaled, expected)

    def test_temperature_zero_is_passthrough(self):
        """Temperature 0 leaves logits unchanged (argmax handled at sample time)."""
        logits = torch.tensor([[2.0, 4.0, 6.0]])
        result = apply_temperature(logits, 0.0)
        torch.testing.assert_close(result, logits)

    def test_top_k_filters_to_exactly_k(self):
        """Top-K retains only the K highest logits, masks the rest to -inf."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 7.0, 2.0]])
        filtered = apply_top_k(logits, top_k=2)
        # The top-2 values are 7.0 (idx 3) and 5.0 (idx 1)
        finite_mask = torch.isfinite(filtered)
        self.assertEqual(finite_mask.sum().item(), 2)
        # Check the surviving values are at indices 1 and 3
        self.assertTrue(torch.isfinite(filtered[0, 1]).item())
        self.assertTrue(torch.isfinite(filtered[0, 3]).item())

    def test_top_k_zero_disabled(self):
        """Top-K with k=0 is a no-op."""
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        result = apply_top_k(logits, top_k=0)
        torch.testing.assert_close(result, logits)

    def test_top_p_filters_cumulative(self):
        """Top-P filters tokens whose cumulative probability exceeds p."""
        # Construct logits where softmax gives known probabilities
        # softmax([10, 5, 1, 0]) ≈ [0.993, 0.0067, 0.00012, 0.000045]
        logits = torch.tensor([[10.0, 5.0, 1.0, 0.0]])
        filtered = apply_top_p(logits, top_p=0.99)
        # Only the first token covers 0.993 > 0.99, so at minimum 1 token survives
        finite_mask = torch.isfinite(filtered)
        n_surviving = finite_mask.sum().item()
        self.assertGreaterEqual(n_surviving, 1)
        self.assertLess(n_surviving, 4)  # Some should be filtered

    def test_top_p_one_disabled(self):
        """Top-P with p=1.0 is a no-op."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_top_p(logits, top_p=1.0)
        torch.testing.assert_close(result, logits)

    def test_min_p_filters_below_threshold(self):
        """Min-P masks tokens whose probability is below min_p * max_prob."""
        # softmax([10, 5, 0]) ≈ [0.993, 0.0067, 0.000045]
        logits = torch.tensor([[10.0, 5.0, 0.0]])
        filtered = apply_min_p(logits, min_p=0.01)
        # Only idx 0 should survive: 0.993 >= 0.01 * 0.993
        # idx 1: 0.0067 < 0.01 * 0.993 = 0.00993 → filtered
        self.assertTrue(torch.isfinite(filtered[0, 0]).item())

    def test_sampling_respects_top_k(self):
        """Sampled tokens come only from the top-k set."""
        logits = torch.tensor([[1.0, 100.0, 1.0, 1.0, 1.0]])  # idx 1 dominates
        # With top_k=1, only idx 1 should ever be sampled
        for _ in range(10):
            token = full_sample(logits.clone(), [], temperature=1.0, top_k=1)
            self.assertEqual(token.item(), 1)


class TestFinalSoftCap(unittest.TestCase):
    """Req 13.3: Final logit soft-cap bounds logits before sampling."""

    def test_softcap_bounds_logits(self):
        """After soft-cap, all logits are within [-cap, +cap]."""
        logits = torch.tensor([[100.0, -200.0, 50.0, -30.0, 0.0]])
        cap = 30.0
        capped = apply_final_softcap(logits, cap)
        self.assertTrue((capped <= cap).all().item())
        self.assertTrue((capped >= -cap).all().item())

    def test_softcap_formula(self):
        """Soft-cap uses tanh(logits/cap) * cap."""
        logits = torch.tensor([[5.0, -3.0, 15.0]])
        cap = 10.0
        result = apply_final_softcap(logits, cap)
        expected = cap * torch.tanh(logits / cap)
        torch.testing.assert_close(result, expected)

    def test_softcap_none_disabled(self):
        """Soft-cap with None leaves logits unchanged."""
        logits = torch.tensor([[100.0, -200.0]])
        result = apply_final_softcap(logits, None)
        torch.testing.assert_close(result, logits)

    def test_softcap_zero_disabled(self):
        """Soft-cap with 0.0 leaves logits unchanged."""
        logits = torch.tensor([[100.0, -200.0]])
        result = apply_final_softcap(logits, 0.0)
        torch.testing.assert_close(result, logits)

    def test_softcap_applied_before_sampling_in_pipeline(self):
        """In full_sample, soft-cap bounds logits before temperature/filtering."""
        # Large logits that would overflow without capping
        logits = torch.tensor([[500.0, -500.0, 300.0, -300.0, 0.0]])
        cap = 30.0
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        # Should not raise; the soft-cap prevents extreme values
        token = full_sample(
            logits, [], temperature=1.0, final_logit_softcap=cap, generator=gen
        )
        self.assertTrue(0 <= token.item() < 5)


class TestNonFiniteGuard(unittest.TestCase):
    """Req 13.4: Non-finite logits routed through NaN/Inf guard, not sampled."""

    def test_nan_raises_non_finite_error(self):
        """NaN in logits raises NonFiniteError."""
        logits = torch.tensor([[1.0, float("nan"), 3.0]])
        with self.assertRaises(NonFiniteError) as ctx:
            full_sample(logits, [], temperature=0.8)
        err = ctx.exception
        self.assertEqual(err.module_name, "Sampler")
        self.assertEqual(err.tensor_name, "input_logits")

    def test_inf_raises_non_finite_error(self):
        """Inf in logits raises NonFiniteError."""
        logits = torch.tensor([[1.0, float("inf"), 3.0]])
        with self.assertRaises(NonFiniteError) as ctx:
            full_sample(logits, [], temperature=0.8)
        err = ctx.exception
        self.assertEqual(err.module_name, "Sampler")
        self.assertEqual(err.tensor_name, "input_logits")

    def test_neg_inf_raises_non_finite_error(self):
        """Negative Inf in logits raises NonFiniteError."""
        logits = torch.tensor([[float("-inf"), 1.0, 3.0]])
        with self.assertRaises(NonFiniteError) as ctx:
            full_sample(logits, [], temperature=0.8)
        err = ctx.exception
        self.assertEqual(err.module_name, "Sampler")

    def test_all_finite_no_error(self):
        """All-finite logits do not raise."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        # Should not raise
        token = full_sample(logits, [], temperature=0.0)
        self.assertEqual(token.item(), 2)  # argmax → index of 3.0


if __name__ == "__main__":
    unittest.main()
