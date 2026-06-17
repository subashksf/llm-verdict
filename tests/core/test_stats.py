"""Tests for core/stats.py — all expected values hand-computed and documented."""

import numpy as np
import pytest

from llm_verdict.core.stats import (
    bootstrap_ci,
    consistency_rate,
    cost_per_success,
    cost_per_success_ci,
    latency_percentiles,
    majority_vote,
    mcnemar_test,
    paired_bootstrap_delta,
    wilson_ci,
)


class TestWilsonCI:
    def test_basic(self):
        # Hand computation: 7/10, z=1.96
        # p_hat = 0.7
        # denom = 1 + 1.96^2/10 = 1 + 0.3842 = 1.3842
        # center = (0.7 + 0.3842/2) / 1.3842 = (0.7 + 0.1921) / 1.3842 = 0.6441
        # margin = (1.96/1.3842) * sqrt(0.7*0.3/10 + 1.96^2/400)
        #        = 1.4155 * sqrt(0.021 + 0.009604) = 1.4155 * 0.1749 = 0.2476
        ci = wilson_ci(7, 10)
        assert ci.point == pytest.approx(0.7, abs=0.001)
        assert ci.lower == pytest.approx(0.397, abs=0.01)
        assert ci.upper == pytest.approx(0.892, abs=0.01)

    def test_zero_total(self):
        ci = wilson_ci(0, 0)
        assert ci.point == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_all_pass(self):
        ci = wilson_ci(20, 20)
        assert ci.point == 1.0
        assert ci.upper == 1.0
        assert ci.lower > 0.8

    def test_all_fail(self):
        ci = wilson_ci(0, 20)
        assert ci.point == 0.0
        assert ci.lower == 0.0
        assert ci.upper < 0.2


class TestBootstrapCI:
    def test_known_mean(self):
        # 10 values all equal to 5.0 → mean=5, CI should be tight around 5
        values = np.array([5.0] * 10, dtype=np.float64)
        ci = bootstrap_ci(values, seed=42)
        assert ci.point == 5.0
        assert ci.lower == 5.0
        assert ci.upper == 5.0

    def test_spread_values(self):
        # [0, 1, 2, ..., 9] → mean = 4.5
        values = np.arange(10, dtype=np.float64)
        ci = bootstrap_ci(values, seed=42)
        assert ci.point == pytest.approx(4.5)
        assert ci.lower < 4.5
        assert ci.upper > 4.5
        assert ci.lower > 2.0
        assert ci.upper < 7.0

    def test_empty(self):
        ci = bootstrap_ci(np.array([], dtype=np.float64))
        assert ci.point == 0.0


class TestPairedBootstrapDelta:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = paired_bootstrap_delta(a, a, seed=42)
        assert ci.point == pytest.approx(0.0)
        assert ci.lower == pytest.approx(0.0)
        assert ci.upper == pytest.approx(0.0)

    def test_clear_improvement(self):
        # B always 1 higher than A → delta = 1.0
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        ci = paired_bootstrap_delta(a, b, seed=42)
        assert ci.point == pytest.approx(1.0)
        assert ci.lower == pytest.approx(1.0)
        assert ci.upper == pytest.approx(1.0)

    def test_mismatched_length(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0])
        ci = paired_bootstrap_delta(a, b)
        assert ci.point == 0.0


class TestMcNemar:
    def test_no_discordant(self):
        # Both pass same tasks → no discordance
        a = np.array([True, True, False, False])
        b = np.array([True, True, False, False])
        result = mcnemar_test(a, b)
        assert result.n_discordant == 0
        assert result.p_value == 1.0

    def test_all_discordant_one_direction(self):
        # Hand computation: b=4 (A pass, B fail), c=0 (A fail, B pass)
        # n_disc=4, exact binomial: P(X >= 4 | n=4, p=0.5) two-sided
        a = np.array([True, True, True, True, False, False])
        b = np.array([False, False, False, False, False, False])
        result = mcnemar_test(a, b)
        assert result.n_discordant == 4
        assert result.p_value < 0.15  # exact binomial with small n

    def test_balanced_discordant(self):
        # b=2, c=2 → perfectly balanced, p should be 1.0
        a = np.array([True, True, False, False])
        b = np.array([False, False, True, True])
        result = mcnemar_test(a, b)
        assert result.n_discordant == 4
        assert result.p_value == pytest.approx(1.0, abs=0.01)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            mcnemar_test(np.array([True]), np.array([True, False]))


class TestCostPerSuccess:
    def test_basic(self):
        # $10 total, 5 tasks passed → $2 per success
        assert cost_per_success(10.0, 5) == pytest.approx(2.0)

    def test_zero_passed(self):
        assert cost_per_success(10.0, 0) == float("inf")

    def test_ci(self):
        costs = np.array([1.0, 2.0, 1.5, 3.0, 0.5], dtype=np.float64)
        passed = np.array([True, True, False, True, True], dtype=np.bool_)
        # total=8.0, passed=4 → point=2.0
        ci = cost_per_success_ci(costs, passed, seed=42)
        assert ci.point == pytest.approx(2.0)
        assert ci.lower < 2.0
        assert ci.upper > 2.0


class TestConsistencyRate:
    def test_all_unanimous(self):
        # All tasks have unanimous trials
        trial_passes = [[True, True, True], [False, False, False]]
        assert consistency_rate(trial_passes) == 1.0

    def test_none_unanimous(self):
        trial_passes = [[True, False, True], [False, True, False]]
        assert consistency_rate(trial_passes) == 0.0

    def test_mixed(self):
        # 1 unanimous out of 2 → 50%
        trial_passes = [[True, True, True], [True, False, True]]
        assert consistency_rate(trial_passes) == 0.5

    def test_empty(self):
        assert consistency_rate([]) == 0.0


class TestMajorityVote:
    def test_all_pass(self):
        assert majority_vote([True, True, True]) is True

    def test_majority_pass(self):
        assert majority_vote([True, True, False]) is True

    def test_majority_fail(self):
        assert majority_vote([False, False, True]) is False

    def test_all_fail(self):
        assert majority_vote([False, False, False]) is False


class TestLatencyPercentiles:
    def test_basic(self):
        # [100, 200, 300, ..., 1000] → p50=550, p95=955
        latencies = np.arange(100, 1100, 100, dtype=np.float64)
        p50, p95 = latency_percentiles(latencies)
        assert p50 == pytest.approx(550.0)
        assert p95 == pytest.approx(955.0)

    def test_empty(self):
        p50, p95 = latency_percentiles(np.array([], dtype=np.float64))
        assert p50 == 0.0
        assert p95 == 0.0
