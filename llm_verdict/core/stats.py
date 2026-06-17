"""Statistical functions — bootstrap CIs, Wilson, McNemar, cost-per-success."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with lower/upper confidence bounds."""

    point: float
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class McNemarResult:
    """Result of McNemar's test on paired binary outcomes."""

    statistic: float
    p_value: float
    n_discordant: int


def wilson_ci(successes: int, total: int, z: float = 1.96) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return ConfidenceInterval(point=0.0, lower=0.0, upper=0.0)

    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    variance = p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)
    margin = (z / denom) * math.sqrt(variance)

    return ConfidenceInterval(
        point=p_hat,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
    )


def bootstrap_ci(
    values: NDArray[np.floating],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> ConfidenceInterval:
    """Bootstrap CI for the mean of `values` (resample rows = tasks)."""
    if len(values) == 0:
        return ConfidenceInterval(point=0.0, lower=0.0, upper=0.0)

    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()

    alpha = 1 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return ConfidenceInterval(point=float(values.mean()), lower=lower, upper=upper)


def paired_bootstrap_delta(
    values_a: NDArray[np.floating],
    values_b: NDArray[np.floating],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> ConfidenceInterval:
    """Bootstrap CI for the mean difference (B - A), resampling paired tasks."""
    if len(values_a) == 0 or len(values_a) != len(values_b):
        return ConfidenceInterval(point=0.0, lower=0.0, upper=0.0)

    rng = np.random.default_rng(seed)
    n = len(values_a)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = values_b[idx].mean() - values_a[idx].mean()

    alpha = 1 - confidence
    lower = float(np.percentile(deltas, 100 * alpha / 2))
    upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    observed = float(values_b.mean() - values_a.mean())
    return ConfidenceInterval(point=observed, lower=lower, upper=upper)


def mcnemar_test(pass_a: NDArray[np.bool_], pass_b: NDArray[np.bool_]) -> McNemarResult:
    """McNemar's test on paired per-task pass/fail (majority vote across trials).

    Compares discordant pairs: tasks where model A passed but B failed (b) and
    vice versa (c). Uses exact binomial when b+c < 25, chi-squared otherwise.
    """
    if len(pass_a) != len(pass_b):
        raise ValueError("Arrays must have same length")

    b = int(np.sum(pass_a & ~pass_b))  # A pass, B fail
    c = int(np.sum(~pass_a & pass_b))  # A fail, B pass
    n_disc = b + c

    if n_disc == 0:
        return McNemarResult(statistic=0.0, p_value=1.0, n_discordant=0)

    if n_disc < 25:
        from scipy.stats import binomtest

        result = binomtest(b, n_disc, 0.5)
        return McNemarResult(
            statistic=float(b),
            p_value=float(result.pvalue),
            n_discordant=n_disc,
        )

    statistic = (b - c) ** 2 / (b + c)
    from scipy.stats import chi2

    p_val = 1.0 - float(chi2.cdf(statistic, df=1))
    return McNemarResult(statistic=statistic, p_value=p_val, n_discordant=n_disc)


def cost_per_success(total_cost: float, tasks_passed: int) -> float:
    """Total cost divided by number of tasks passed."""
    if tasks_passed == 0:
        return float("inf")
    return total_cost / tasks_passed


def cost_per_success_ci(
    task_costs: NDArray[np.floating],
    task_passed: NDArray[np.bool_],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> ConfidenceInterval:
    """Bootstrap CI for cost-per-success, resampling tasks."""
    if len(task_costs) == 0:
        return ConfidenceInterval(point=0.0, lower=0.0, upper=0.0)

    rng = np.random.default_rng(seed)
    n = len(task_costs)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        total = task_costs[idx].sum()
        passed = task_passed[idx].sum()
        estimates[i] = total / passed if passed > 0 else np.inf

    finite = estimates[np.isfinite(estimates)]
    if len(finite) == 0:
        inf = float("inf")
        return ConfidenceInterval(point=inf, lower=inf, upper=inf)

    alpha = 1 - confidence
    lower = float(np.percentile(finite, 100 * alpha / 2))
    upper = float(np.percentile(finite, 100 * (1 - alpha / 2)))
    point = cost_per_success(float(task_costs.sum()), int(task_passed.sum()))
    return ConfidenceInterval(point=point, lower=lower, upper=upper)


def consistency_rate(trial_passes: list[list[bool]]) -> float:
    """Fraction of tasks where all trials agree (unanimous pass or fail)."""
    if not trial_passes:
        return 0.0
    unanimous = sum(1 for trials in trial_passes if all(trials) or not any(trials))
    return unanimous / len(trial_passes)


def majority_vote(trials: list[bool]) -> bool:
    """Return True if majority of trials passed."""
    return sum(trials) > len(trials) / 2


def latency_percentiles(
    latencies_ms: NDArray[np.floating],
) -> tuple[float, float]:
    """Return (p50, p95) latency in ms."""
    if len(latencies_ms) == 0:
        return (0.0, 0.0)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    return (p50, p95)
