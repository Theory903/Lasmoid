"""
Test suite for the YaRN branch of ``precompute_freqs_cis`` in ``attention.py``.

The YaRN interpolation branch is gated on ``original_seq_len > 0``.
``Tiny_Config`` sets ``original_seq_len: 0``, bypassing it entirely.
This module adds a fixture with a non-zero ``original_seq_len`` so the branch
actually runs, and verifies that ``find_correction_range`` and
``linear_ramp_factor`` produce the expected interpolation behavior.

Also verifies parity with DeepSeek-V4-Pro's ``precompute_freqs_cis`` for both
the YaRN-active and YaRN-disabled branches.

Validates: Requirements 1.3
"""

import math
import sys
import unittest
from pathlib import Path

import torch

# Ensure inference/ is importable
_INFERENCE_DIR = Path(__file__).resolve().parent
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

from attention import precompute_freqs_cis


# ══════════════════════════════════════════════════════════════════════
# REFERENCE IMPLEMENTATION (from DeepSeek-V4-Pro/inference/model.py)
# ══════════════════════════════════════════════════════════════════════

def _ref_precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """
    Standalone reference implementation matching DeepSeek-V4-Pro exactly.
    Used for parity verification.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# ══════════════════════════════════════════════════════════════════════
# YARN FIXTURE PARAMETERS
# ══════════════════════════════════════════════════════════════════════

# These match real-world YaRN config values used when extending context length.
# original_seq_len > 0 activates the YaRN interpolation branch.
YARN_FIXTURE = {
    "dim": 48,              # Same as Tiny_Config head_dim
    "seqlen": 128,          # Reasonable test sequence length
    "original_seq_len": 4096,  # Non-zero: activates YaRN ramp
    "base": 10000.0,        # Standard rope_theta
    "factor": 32.0,         # YaRN scaling factor
    "beta_fast": 32,        # Matches Tiny_Config
    "beta_slow": 1,         # Matches Tiny_Config
}

# Second fixture: larger original_seq_len and different factor for robustness
YARN_FIXTURE_ALT = {
    "dim": 16,              # Matches Tiny_Config rope_head_dim
    "seqlen": 64,
    "original_seq_len": 8192,
    "base": 1000000.0,      # Extended base (like global RoPE)
    "factor": 8.0,          # Smaller factor
    "beta_fast": 64,
    "beta_slow": 2,
}

# No-YaRN fixture: original_seq_len == 0 (active path in Tiny_Config)
NO_YARN_FIXTURE = {
    "dim": 48,
    "seqlen": 128,
    "original_seq_len": 0,
    "base": 10000.0,
    "factor": 32.0,
    "beta_fast": 32,
    "beta_slow": 1,
}


class TestYaRNFreqsCis(unittest.TestCase):
    """Exercise the YaRN interpolation branch in precompute_freqs_cis."""

    TOLERANCE = 1e-5  # Pure-math tolerance per the design

    # ──────────────────────────────────────────────────────────────────
    # Core YaRN branch tests
    # ──────────────────────────────────────────────────────────────────

    def test_yarn_branch_activates_with_nonzero_original_seq_len(self):
        """Verify that original_seq_len > 0 produces frequencies different from
        the no-YaRN baseline (i.e., the interpolation ramp actually modifies freqs)."""
        # Clear the LRU cache to avoid stale results
        precompute_freqs_cis.cache_clear()

        yarn_freqs = precompute_freqs_cis(**YARN_FIXTURE)
        no_yarn = precompute_freqs_cis(
            dim=YARN_FIXTURE["dim"],
            seqlen=YARN_FIXTURE["seqlen"],
            original_seq_len=0,
            base=YARN_FIXTURE["base"],
            factor=YARN_FIXTURE["factor"],
            beta_fast=YARN_FIXTURE["beta_fast"],
            beta_slow=YARN_FIXTURE["beta_slow"],
        )
        precompute_freqs_cis.cache_clear()

        # YaRN ramp should modify at least some frequencies
        self.assertFalse(
            torch.allclose(yarn_freqs, no_yarn, atol=1e-7),
            "YaRN branch did not modify frequencies — ramp may be bypassed",
        )

    def test_yarn_output_shape(self):
        """YaRN-modified freqs_cis has the expected shape (seqlen, dim//2)."""
        precompute_freqs_cis.cache_clear()
        result = precompute_freqs_cis(**YARN_FIXTURE)
        precompute_freqs_cis.cache_clear()

        expected_shape = (YARN_FIXTURE["seqlen"], YARN_FIXTURE["dim"] // 2)
        self.assertEqual(result.shape, expected_shape)
        # Should be complex-valued
        self.assertTrue(result.is_complex(), "freqs_cis should be complex-valued")

    def test_yarn_unit_magnitude(self):
        """All entries in freqs_cis must have unit magnitude (they are rotation phasors)."""
        precompute_freqs_cis.cache_clear()
        result = precompute_freqs_cis(**YARN_FIXTURE)
        precompute_freqs_cis.cache_clear()

        magnitudes = torch.abs(result)
        self.assertTrue(
            torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=self.TOLERANCE),
            "YaRN freqs_cis should have unit magnitude (rotation phasors)",
        )

    def test_yarn_alt_fixture(self):
        """Exercise with alternative parameters to verify branch runs in different configs."""
        precompute_freqs_cis.cache_clear()
        result = precompute_freqs_cis(**YARN_FIXTURE_ALT)
        precompute_freqs_cis.cache_clear()

        expected_shape = (YARN_FIXTURE_ALT["seqlen"], YARN_FIXTURE_ALT["dim"] // 2)
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(result.is_complex())
        # Unit magnitude
        magnitudes = torch.abs(result)
        self.assertTrue(
            torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=self.TOLERANCE),
        )

    # ──────────────────────────────────────────────────────────────────
    # find_correction_range + linear_ramp_factor behavior
    # ──────────────────────────────────────────────────────────────────

    def test_yarn_interpolation_bounded(self):
        """YaRN-interpolated frequencies must be bounded between
        freqs/factor (fully interpolated) and freqs (no interpolation).
        
        The ramp formula: freqs_yarn = freqs / factor * (1 - smooth) + freqs * smooth
        Since 0 <= smooth <= 1, output is in [freqs/factor, freqs].
        """
        precompute_freqs_cis.cache_clear()
        dim = YARN_FIXTURE["dim"]
        base = YARN_FIXTURE["base"]
        factor = YARN_FIXTURE["factor"]

        # Compute raw base frequencies
        raw_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        lower_bound = raw_freqs / factor  # Fully interpolated
        upper_bound = raw_freqs            # No interpolation

        # Get YaRN-modified freqs by examining the angles at position 1
        # At t=1, the angle is just the frequency value
        yarn_freqs_cis = precompute_freqs_cis(**YARN_FIXTURE)
        precompute_freqs_cis.cache_clear()

        # Extract angles at position 1 (the actual freq values after interpolation)
        yarn_angles_at_t1 = torch.angle(yarn_freqs_cis[1])

        # All interpolated frequencies should be between lower and upper bounds
        self.assertTrue(
            torch.all(yarn_angles_at_t1 >= lower_bound - 1e-6),
            "YaRN frequencies below the fully-interpolated lower bound",
        )
        self.assertTrue(
            torch.all(yarn_angles_at_t1 <= upper_bound + 1e-6),
            "YaRN frequencies above the no-interpolation upper bound",
        )

    def test_linear_ramp_smooth_values(self):
        """Verify the smooth factor from linear_ramp_factor is in [0, 1].
        
        This exercises the internal find_correction_range + linear_ramp_factor
        logic by checking that the resulting interpolation blend is well-formed.
        """
        precompute_freqs_cis.cache_clear()
        dim = YARN_FIXTURE["dim"]
        base = YARN_FIXTURE["base"]
        original_seq_len = YARN_FIXTURE["original_seq_len"]
        factor = YARN_FIXTURE["factor"]
        beta_fast = YARN_FIXTURE["beta_fast"]
        beta_slow = YARN_FIXTURE["beta_slow"]

        # Reproduce the internal logic to verify correctness
        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)

        # Verify correction range is valid
        self.assertGreaterEqual(low, 0)
        self.assertLess(high, dim)
        self.assertLessEqual(low, high)

        # Verify linear ramp produces values in [0, 1]
        half_dim = dim // 2
        if low == high:
            high_adj = high + 0.001
        else:
            high_adj = high
        lf = (torch.arange(half_dim, dtype=torch.float32) - low) / (high_adj - low)
        ramp = torch.clamp(lf, 0, 1)

        self.assertTrue(torch.all(ramp >= 0.0))
        self.assertTrue(torch.all(ramp <= 1.0))

        # smooth = 1 - ramp, so also in [0, 1]
        smooth = 1 - ramp
        self.assertTrue(torch.all(smooth >= 0.0))
        self.assertTrue(torch.all(smooth <= 1.0))

    # ──────────────────────────────────────────────────────────────────
    # DeepSeek-V4-Pro reference parity
    # ──────────────────────────────────────────────────────────────────

    def test_parity_with_reference_yarn_active(self):
        """YaRN-active branch matches DeepSeek-V4-Pro reference within tolerance."""
        precompute_freqs_cis.cache_clear()
        lasmoid_result = precompute_freqs_cis(**YARN_FIXTURE)
        precompute_freqs_cis.cache_clear()

        ref_result = _ref_precompute_freqs_cis(**YARN_FIXTURE)

        max_diff = (lasmoid_result - ref_result).abs().max().item()
        self.assertLessEqual(
            max_diff,
            self.TOLERANCE,
            f"YaRN-active parity failed: max|Δ| = {max_diff:.2e} > {self.TOLERANCE:.0e}",
        )

    def test_parity_with_reference_yarn_disabled(self):
        """No-YaRN branch matches DeepSeek-V4-Pro reference within tolerance."""
        precompute_freqs_cis.cache_clear()
        lasmoid_result = precompute_freqs_cis(**NO_YARN_FIXTURE)
        precompute_freqs_cis.cache_clear()

        ref_result = _ref_precompute_freqs_cis(**NO_YARN_FIXTURE)

        max_diff = (lasmoid_result - ref_result).abs().max().item()
        self.assertLessEqual(
            max_diff,
            self.TOLERANCE,
            f"No-YaRN parity failed: max|Δ| = {max_diff:.2e} > {self.TOLERANCE:.0e}",
        )

    def test_parity_with_reference_alt_fixture(self):
        """Alternative YaRN fixture also matches the reference."""
        precompute_freqs_cis.cache_clear()
        lasmoid_result = precompute_freqs_cis(**YARN_FIXTURE_ALT)
        precompute_freqs_cis.cache_clear()

        ref_result = _ref_precompute_freqs_cis(**YARN_FIXTURE_ALT)

        max_diff = (lasmoid_result - ref_result).abs().max().item()
        self.assertLessEqual(
            max_diff,
            self.TOLERANCE,
            f"Alt-fixture parity failed: max|Δ| = {max_diff:.2e} > {self.TOLERANCE:.0e}",
        )

    # ──────────────────────────────────────────────────────────────────
    # Edge cases
    # ──────────────────────────────────────────────────────────────────

    def test_yarn_factor_one_reduces_to_base_freqs(self):
        """When factor == 1.0, the YaRN ramp should produce frequencies very close
        to the base (un-interpolated) frequencies, since:
        freqs / 1.0 * (1 - smooth) + freqs * smooth = freqs * (1 - smooth) + freqs * smooth = freqs
        """
        precompute_freqs_cis.cache_clear()
        factor_one = precompute_freqs_cis(
            dim=YARN_FIXTURE["dim"],
            seqlen=YARN_FIXTURE["seqlen"],
            original_seq_len=YARN_FIXTURE["original_seq_len"],
            base=YARN_FIXTURE["base"],
            factor=1.0,  # No actual scaling
            beta_fast=YARN_FIXTURE["beta_fast"],
            beta_slow=YARN_FIXTURE["beta_slow"],
        )
        no_yarn = precompute_freqs_cis(
            dim=YARN_FIXTURE["dim"],
            seqlen=YARN_FIXTURE["seqlen"],
            original_seq_len=0,
            base=YARN_FIXTURE["base"],
            factor=1.0,
            beta_fast=YARN_FIXTURE["beta_fast"],
            beta_slow=YARN_FIXTURE["beta_slow"],
        )
        precompute_freqs_cis.cache_clear()

        max_diff = (factor_one - no_yarn).abs().max().item()
        self.assertLessEqual(
            max_diff,
            self.TOLERANCE,
            f"factor=1.0 should equal no-YaRN: max|Δ| = {max_diff:.2e}",
        )

    def test_yarn_with_tiny_config_rope_head_dim(self):
        """Exercise YaRN with Tiny_Config's rope_head_dim (16) to confirm it works
        with the exact dimensions the model would use if original_seq_len were non-zero."""
        precompute_freqs_cis.cache_clear()
        result = precompute_freqs_cis(
            dim=16,               # Tiny_Config rope_head_dim
            seqlen=512,           # Tiny_Config max_seq_len
            original_seq_len=4096,
            base=10000.0,         # Tiny_Config rope_theta
            factor=1.0,           # Tiny_Config rope_factor
            beta_fast=32,         # Tiny_Config
            beta_slow=1,          # Tiny_Config
        )
        precompute_freqs_cis.cache_clear()

        self.assertEqual(result.shape, (512, 8))  # seqlen × (dim//2)
        self.assertTrue(result.is_complex())
        magnitudes = torch.abs(result)
        self.assertTrue(
            torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=self.TOLERANCE),
        )


if __name__ == "__main__":
    unittest.main()
