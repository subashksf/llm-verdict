"""Tests for core/verdict.py — pre-registered criteria engine."""

from pathlib import Path

from llm_verdict.core.stats import ConfidenceInterval as CI
from llm_verdict.core.verdict import (
    CriteriaConfig,
    VerdictInput,
    evaluate_verdict,
    load_criteria,
)


class TestEvaluateVerdict:
    def _criteria(self) -> CriteriaConfig:
        return CriteriaConfig(
            category="code_review",
            incumbent="claude-sonnet-4-6",
            adopt_if=[
                {"pass_rate_delta_pp": ">= 5", "ci_overlap": False},
                {
                    "OR": {
                        "pass_rate_within_ci": True,
                        "cost_per_success_reduction_pct": ">= 30",
                    }
                },
            ],
            constraints={"p95_latency_ms": "<= 20000", "refusal_rate_pct": "<= 2"},
        )

    def test_adopt_via_pass_rate_delta(self):
        metrics = VerdictInput(
            pass_rate_ci=CI(point=0.85, lower=0.78, upper=0.92),
            incumbent_pass_rate_ci=CI(point=0.70, lower=0.62, upper=0.77),
            pass_rate_delta_ci=CI(point=0.10, lower=0.03, upper=0.17),
            p95_latency_ms=15000.0,
            refusal_rate_pct=1.0,
        )
        result = evaluate_verdict(self._criteria(), metrics)
        assert result.outcome == "ADOPT"
        assert any("pass_rate_delta_pp" in c for c in result.fired_clauses)

    def test_adopt_via_cost_reduction(self):
        metrics = VerdictInput(
            pass_rate_ci=CI(point=0.72, lower=0.65, upper=0.79),
            incumbent_pass_rate_ci=CI(point=0.70, lower=0.63, upper=0.77),
            pass_rate_delta_ci=CI(point=0.02, lower=-0.05, upper=0.09),
            cost_per_success=1.0,
            incumbent_cost_per_success=2.0,
            p95_latency_ms=10000.0,
            refusal_rate_pct=0.5,
        )
        result = evaluate_verdict(self._criteria(), metrics)
        assert result.outcome == "ADOPT"
        assert any("cost_per_success_reduction" in c for c in result.fired_clauses)

    def test_reject_constraint_violation(self):
        metrics = VerdictInput(
            pass_rate_ci=CI(point=0.90, lower=0.85, upper=0.95),
            incumbent_pass_rate_ci=CI(point=0.70, lower=0.62, upper=0.77),
            pass_rate_delta_ci=CI(point=0.20, lower=0.10, upper=0.30),
            p95_latency_ms=25000.0,  # violates <= 20000
            refusal_rate_pct=1.0,
        )
        result = evaluate_verdict(self._criteria(), metrics)
        assert result.outcome == "REJECT"
        assert any("p95_latency_ms" in c for c in result.fired_clauses)

    def test_insufficient_data_wide_ci(self):
        # CI width = 0.30 - (-0.20) = 0.50 → 50pp wide
        # threshold is 5pp, 2x = 10pp. 50 > 10 → INSUFFICIENT_DATA
        metrics = VerdictInput(
            pass_rate_ci=CI(point=0.60, lower=0.30, upper=0.90),
            incumbent_pass_rate_ci=CI(point=0.55, lower=0.25, upper=0.85),
            pass_rate_delta_ci=CI(point=0.05, lower=-0.20, upper=0.30),
            p95_latency_ms=10000.0,
            refusal_rate_pct=1.0,
        )
        result = evaluate_verdict(self._criteria(), metrics)
        assert result.outcome == "INSUFFICIENT_DATA"
        assert "CI too wide" in result.fired_clauses[0]

    def test_hold_no_condition_met(self):
        metrics = VerdictInput(
            pass_rate_ci=CI(point=0.71, lower=0.68, upper=0.74),
            incumbent_pass_rate_ci=CI(point=0.70, lower=0.67, upper=0.73),
            pass_rate_delta_ci=CI(point=0.01, lower=-0.01, upper=0.03),
            cost_per_success=1.9,
            incumbent_cost_per_success=2.0,
            p95_latency_ms=10000.0,
            refusal_rate_pct=1.0,
        )
        result = evaluate_verdict(self._criteria(), metrics)
        assert result.outcome == "HOLD"


class TestLoadCriteria:
    def test_load_example(self):
        path = Path("configs/criteria/code_review.yaml")
        criteria = load_criteria(path)
        assert criteria.category == "code_review"
        assert criteria.incumbent == "claude-sonnet-4-6"
        assert len(criteria.adopt_if) == 2
        assert "p95_latency_ms" in criteria.constraints
