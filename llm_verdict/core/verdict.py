"""Pre-registered success-criteria engine.

Evaluates criteria YAML and emits ADOPT / HOLD / REJECT / INSUFFICIENT_DATA.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_verdict.core.stats import ConfidenceInterval


@dataclass(frozen=True)
class CriteriaConfig:
    """Parsed pre-registered criteria from YAML."""

    category: str
    incumbent: str | None = None
    adopt_if: list[dict[str, Any]] = field(default_factory=list)
    constraints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VerdictInput:
    """All computed metrics needed for verdict evaluation."""

    pass_rate_ci: ConfidenceInterval | None = None
    incumbent_pass_rate_ci: ConfidenceInterval | None = None
    pass_rate_delta_ci: ConfidenceInterval | None = None
    cost_per_success: float | None = None
    incumbent_cost_per_success: float | None = None
    p95_latency_ms: float | None = None
    refusal_rate_pct: float | None = None


@dataclass(frozen=True)
class VerdictOutput:
    """Result of evaluating criteria against metrics."""

    outcome: str  # ADOPT | HOLD | REJECT | INSUFFICIENT_DATA
    fired_clauses: list[str] = field(default_factory=list)


def load_criteria(path: Path) -> CriteriaConfig:
    """Load a criteria YAML file."""
    raw = yaml.safe_load(path.read_text())
    return CriteriaConfig(
        category=raw["category"],
        incumbent=raw.get("incumbent"),
        adopt_if=raw.get("adopt_if", []),
        constraints=raw.get("constraints", {}),
    )


def evaluate_verdict(criteria: CriteriaConfig, metrics: VerdictInput) -> VerdictOutput:
    """Evaluate pre-registered criteria against computed metrics."""
    fired: list[str] = []

    constraint_violations = _check_constraints(criteria.constraints, metrics)
    if constraint_violations:
        return VerdictOutput(outcome="REJECT", fired_clauses=constraint_violations)

    adopt_met = _check_adopt_conditions(criteria.adopt_if, metrics, fired)
    if adopt_met:
        return VerdictOutput(outcome="ADOPT", fired_clauses=fired)

    if _is_insufficient_data(criteria, metrics):
        return VerdictOutput(
            outcome="INSUFFICIENT_DATA",
            fired_clauses=["CI too wide to decide"],
        )

    return VerdictOutput(outcome="HOLD", fired_clauses=["No adopt condition met"])


def _check_constraints(constraints: dict[str, str], metrics: VerdictInput) -> list[str]:
    """Return list of violated constraint descriptions."""
    violations: list[str] = []

    for key, threshold_str in constraints.items():
        actual = _get_constraint_value(key, metrics)
        if actual is None:
            continue
        if not _eval_comparison(actual, threshold_str):
            msg = f"constraint:{key} ({actual:.2f} violates {threshold_str})"
            violations.append(msg)

    return violations


def _check_adopt_conditions(
    adopt_if: list[dict[str, Any]],
    metrics: VerdictInput,
    fired: list[str],
) -> bool:
    """Check if any adopt condition is satisfied. Populates `fired`."""
    for condition in adopt_if:
        if "OR" in condition:
            if _check_or_block(condition["OR"], metrics, fired):
                return True
        elif _check_single_condition(condition, metrics, fired):
            return True
    return False


def _check_or_block(
    or_conditions: dict[str, Any],
    metrics: VerdictInput,
    fired: list[str],
) -> bool:
    """All sub-conditions in an OR block must be true."""
    all_met = True
    sub_fired: list[str] = []
    for key, expected in or_conditions.items():
        if not _eval_adopt_clause(key, expected, metrics, sub_fired):
            all_met = False
            break
    if all_met:
        fired.extend(sub_fired)
    return all_met


def _check_single_condition(
    condition: dict[str, Any],
    metrics: VerdictInput,
    fired: list[str],
) -> bool:
    """All keys in a single condition dict must be true (AND semantics)."""
    sub_fired: list[str] = []
    for key, expected in condition.items():
        if not _eval_adopt_clause(key, expected, metrics, sub_fired):
            return False
    fired.extend(sub_fired)
    return True


def _eval_adopt_clause(
    key: str, expected: Any, metrics: VerdictInput, fired: list[str]
) -> bool:
    """Evaluate a single adopt clause via dispatch."""
    evaluator = _CLAUSE_EVALUATORS.get(key)
    if evaluator is None:
        return False
    return evaluator(expected, metrics, fired)


def _eval_pass_rate_delta(
    expected: Any, metrics: VerdictInput, fired: list[str]
) -> bool:
    if metrics.pass_rate_delta_ci is None:
        return False
    delta_pp = metrics.pass_rate_delta_ci.point * 100
    if _eval_comparison(delta_pp, str(expected)):
        fired.append(f"pass_rate_delta_pp={delta_pp:.1f}")
        return True
    return False


def _eval_ci_overlap(expected: Any, metrics: VerdictInput, fired: list[str]) -> bool:
    if metrics.pass_rate_ci is None:
        return False
    if metrics.incumbent_pass_rate_ci is None:
        return False
    overlaps = (
        metrics.pass_rate_ci.lower <= metrics.incumbent_pass_rate_ci.upper
        and metrics.incumbent_pass_rate_ci.lower <= metrics.pass_rate_ci.upper
    )
    result = bool(overlaps == expected)
    if result:
        fired.append(f"ci_overlap={overlaps}")
    return result


def _eval_pass_rate_within_ci(
    expected: Any, metrics: VerdictInput, fired: list[str]
) -> bool:
    if metrics.pass_rate_delta_ci is None:
        return False
    delta = metrics.pass_rate_delta_ci
    within = delta.lower <= 0 <= delta.upper
    result = bool(within == expected)
    if result:
        fired.append(f"pass_rate_within_ci={within}")
    return result


def _eval_cost_reduction(
    expected: Any, metrics: VerdictInput, fired: list[str]
) -> bool:
    if metrics.cost_per_success is None:
        return False
    if metrics.incumbent_cost_per_success is None:
        return False
    if metrics.incumbent_cost_per_success == 0:
        return False
    reduction = (
        (metrics.incumbent_cost_per_success - metrics.cost_per_success)
        / metrics.incumbent_cost_per_success
        * 100
    )
    if _eval_comparison(reduction, str(expected)):
        fired.append(f"cost_per_success_reduction_pct={reduction:.1f}")
        return True
    return False


_ClauseEval = Callable[[Any, VerdictInput, list[str]], bool]

_CLAUSE_EVALUATORS: dict[str, _ClauseEval] = {
    "pass_rate_delta_pp": _eval_pass_rate_delta,
    "ci_overlap": _eval_ci_overlap,
    "pass_rate_within_ci": _eval_pass_rate_within_ci,
    "cost_per_success_reduction_pct": _eval_cost_reduction,
}


def _is_insufficient_data(criteria: CriteriaConfig, metrics: VerdictInput) -> bool:
    """CI too wide to decide: width > 2x the decision threshold.

    Only triggers if the CI is wide AND the result is ambiguous (lower bound
    doesn't already clear the threshold in either direction).
    """
    if not criteria.adopt_if:
        return False

    for condition in criteria.adopt_if:
        if "OR" in condition:
            continue
        threshold = condition.get("pass_rate_delta_pp")
        if threshold is None:
            continue
        threshold_val = _parse_threshold_value(str(threshold))
        if threshold_val is None:
            continue
        if metrics.pass_rate_delta_ci is None:
            return True
        ci_width_pp = metrics.pass_rate_delta_ci.width * 100
        lower_pp = metrics.pass_rate_delta_ci.lower * 100
        if ci_width_pp > 2 * abs(threshold_val) and lower_pp < threshold_val:
            return True

    return False


def _get_constraint_value(key: str, metrics: VerdictInput) -> float | None:
    if key == "p95_latency_ms":
        return metrics.p95_latency_ms
    if key == "refusal_rate_pct":
        return metrics.refusal_rate_pct
    return None


_OPS: dict[str, Any] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


def _eval_comparison(actual: float, threshold_str: str) -> bool:
    """Evaluate 'actual <op> value' (e.g. '>= 5' or '<= 20000')."""
    parts = threshold_str.strip().split()
    if len(parts) == 2:
        op, val_str = parts
    elif len(parts) == 1:
        op, val_str = ">=", parts[0]
    else:
        return False

    try:
        val = float(val_str)
    except ValueError:
        return False

    fn = _OPS.get(op)
    return fn(actual, val) if fn else False


def _parse_threshold_value(threshold_str: str) -> float | None:
    """Extract numeric value from a threshold string like '>= 5'."""
    parts = threshold_str.strip().split()
    try:
        return float(parts[-1])
    except (ValueError, IndexError):
        return None
