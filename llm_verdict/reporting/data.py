"""Data loading for report generation — queries DuckDB and computes stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_verdict.core.stats import (
    ConfidenceInterval,
    McNemarResult,
    bootstrap_ci,
    consistency_rate,
    cost_per_success_ci,
    latency_percentiles,
    majority_vote,
    mcnemar_test,
    paired_bootstrap_delta,
    wilson_ci,
)


@dataclass
class CategoryStats:
    """Computed statistics for one category."""

    category: str
    n_tasks: int
    pass_rate: ConfidenceInterval
    mean_score: ConfidenceInterval
    cost_per_success: ConfidenceInterval
    p50_latency_ms: float
    p95_latency_ms: float
    refusal_rate: float
    consistency: float


@dataclass
class RunReport:
    """All data needed to render a single-model report card."""

    run_id: str
    model_id: str
    model_version: str | None
    run_date: str
    suite_hash: str
    harness_version: str
    judge_config_hash: str | None
    total_cost: float
    wall_clock_seconds: float | None
    categories: list[CategoryStats]
    overall: CategoryStats
    notable_failures: list[str] = field(default_factory=list)


@dataclass
class CompareReport:
    """All data needed to render a head-to-head comparison."""

    run_a: RunReport
    run_b: RunReport
    delta_by_category: dict[str, ConfidenceInterval]
    mcnemar_by_category: dict[str, McNemarResult]
    overall_delta: ConfidenceInterval
    overall_mcnemar: McNemarResult


@dataclass
class TimelineEntry:
    """A single point in the longitudinal timeline."""

    run_id: str
    model_id: str
    model_version: str | None
    run_date: str
    suite_hash: str
    pass_rate: float
    mean_score: float
    total_cost: float


@dataclass
class TimelineReport:
    """All data for a longitudinal timeline report."""

    model_family: str
    entries: list[TimelineEntry] = field(default_factory=list)


def compute_category_stats(
    conn: Any, run_id: str, category: str, task_ids: list[str]
) -> CategoryStats:
    """Compute all stats for a single category within a run."""
    task_scores = _get_task_pass_rates(conn, run_id, task_ids)
    task_costs_arr = _get_task_costs(conn, run_id, task_ids)
    latencies = _get_latencies(conn, run_id, task_ids)
    trial_passes = _get_trial_passes(conn, run_id, task_ids)

    n_tasks = len(task_ids)
    passed_count = sum(1 for v in task_scores if v)
    pass_rate_ci = wilson_ci(passed_count, n_tasks)

    scores_arr = _get_mean_scores(conn, run_id, task_ids)
    mean_score_ci = bootstrap_ci(scores_arr)

    task_passed_arr = np.array([majority_vote(t) for t in trial_passes], dtype=np.bool_)
    cps_ci = cost_per_success_ci(task_costs_arr, task_passed_arr)

    p50, p95 = latency_percentiles(latencies)
    consist = consistency_rate(trial_passes)
    refusal = _compute_refusal_rate(conn, run_id, task_ids)

    return CategoryStats(
        category=category,
        n_tasks=n_tasks,
        pass_rate=pass_rate_ci,
        mean_score=mean_score_ci,
        cost_per_success=cps_ci,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        refusal_rate=refusal,
        consistency=consist,
    )


def compute_run_report(conn: Any, run_id: str) -> RunReport:
    """Build full run report from DB."""
    run = _get_run_record(conn, run_id)
    categories = _get_categories(conn, run_id)
    cat_stats = []
    all_task_ids: list[str] = []

    for cat, task_ids in categories.items():
        all_task_ids.extend(task_ids)
        cat_stats.append(compute_category_stats(conn, run_id, cat, task_ids))

    overall = compute_category_stats(conn, run_id, "overall", all_task_ids)
    overall = CategoryStats(
        category="overall",
        n_tasks=overall.n_tasks,
        pass_rate=overall.pass_rate,
        mean_score=overall.mean_score,
        cost_per_success=overall.cost_per_success,
        p50_latency_ms=overall.p50_latency_ms,
        p95_latency_ms=overall.p95_latency_ms,
        refusal_rate=overall.refusal_rate,
        consistency=overall.consistency,
    )

    total_cost = _get_total_cost(conn, run_id)
    failures = _get_notable_failures(conn, run_id)

    return RunReport(
        run_id=run_id,
        model_id=run["model_id"],
        model_version=run.get("model_version"),
        run_date=str(run["created_at"]),
        suite_hash=run["suite_hash"],
        harness_version=run["harness_version"],
        judge_config_hash=run.get("judge_config_hash"),
        total_cost=total_cost,
        wall_clock_seconds=None,
        categories=cat_stats,
        overall=overall,
        notable_failures=failures,
    )


def compute_compare_report(conn: Any, run_id_a: str, run_id_b: str) -> CompareReport:
    """Build head-to-head comparison report."""
    run_a_rec = _get_run_record(conn, run_id_a)
    run_b_rec = _get_run_record(conn, run_id_b)

    if run_a_rec["suite_hash"] != run_b_rec["suite_hash"]:
        raise ValueError(
            f"Cannot compare runs with different suite hashes: "
            f"{run_a_rec['suite_hash'][:8]} vs {run_b_rec['suite_hash'][:8]}"
        )

    report_a = compute_run_report(conn, run_id_a)
    report_b = compute_run_report(conn, run_id_b)

    categories = _get_categories(conn, run_id_a)
    delta_by_cat: dict[str, ConfidenceInterval] = {}
    mcnemar_by_cat: dict[str, McNemarResult] = {}

    for cat, task_ids in categories.items():
        passes_a = _get_majority_votes(conn, run_id_a, task_ids)
        passes_b = _get_majority_votes(conn, run_id_b, task_ids)
        scores_a = _get_mean_scores(conn, run_id_a, task_ids)
        scores_b = _get_mean_scores(conn, run_id_b, task_ids)

        delta_by_cat[cat] = paired_bootstrap_delta(scores_a, scores_b)
        mcnemar_by_cat[cat] = mcnemar_test(passes_a, passes_b)

    all_tasks = [tid for tids in categories.values() for tid in tids]
    all_scores_a = _get_mean_scores(conn, run_id_a, all_tasks)
    all_scores_b = _get_mean_scores(conn, run_id_b, all_tasks)
    all_passes_a = _get_majority_votes(conn, run_id_a, all_tasks)
    all_passes_b = _get_majority_votes(conn, run_id_b, all_tasks)

    return CompareReport(
        run_a=report_a,
        run_b=report_b,
        delta_by_category=delta_by_cat,
        mcnemar_by_category=mcnemar_by_cat,
        overall_delta=paired_bootstrap_delta(all_scores_a, all_scores_b),
        overall_mcnemar=mcnemar_test(all_passes_a, all_passes_b),
    )


def compute_timeline_report(conn: Any, model_family: str) -> TimelineReport:
    """Build longitudinal timeline for a model family prefix."""
    rows = conn.execute(
        "SELECT run_id, model_id, model_version, created_at, suite_hash "
        "FROM runs WHERE model_id LIKE ? "
        "ORDER BY created_at",
        [f"{model_family}%"],
    ).fetchall()

    entries: list[TimelineEntry] = []
    for run_id, model_id, version, created_at, suite_hash in rows:
        stats_row = conn.execute(
            "SELECT AVG(CASE WHEN s.passed THEN 1.0 ELSE 0.0 END), "
            "AVG(s.score), COALESCE(SUM(t.cost_usd), 0) "
            "FROM scores s "
            "JOIN trials t ON s.run_id = t.run_id "
            "AND s.task_id = t.task_id "
            "AND s.trial_index = t.trial_index "
            "WHERE s.run_id = ?",
            [run_id],
        ).fetchone()
        if stats_row is None or stats_row[0] is None:
            continue
        entries.append(
            TimelineEntry(
                run_id=run_id,
                model_id=model_id,
                model_version=version,
                run_date=str(created_at),
                suite_hash=suite_hash,
                pass_rate=float(stats_row[0]),
                mean_score=float(stats_row[1]),
                total_cost=float(stats_row[2]),
            )
        )

    return TimelineReport(model_family=model_family, entries=entries)


# --- internal query helpers ---


def _get_run_record(conn: Any, run_id: str) -> dict[str, Any]:
    from llm_verdict.store.duck import get_run

    rec = get_run(conn, run_id)
    if rec is None:
        raise ValueError(f"Run not found: {run_id}")
    return rec


def _get_categories(conn: Any, run_id: str) -> dict[str, list[str]]:
    """Get category -> task_ids mapping from scored trials."""
    cat_rows = conn.execute(
        """SELECT DISTINCT task_id,
           COALESCE(
               SPLIT_PART(task_id, '/', 1),
               'default'
           ) as category
        FROM scores WHERE run_id = ?""",
        [run_id],
    ).fetchall()

    categories: dict[str, list[str]] = {}
    for task_id, cat in cat_rows:
        categories.setdefault(cat, []).append(task_id)
    return categories


def _get_task_pass_rates(conn: Any, run_id: str, task_ids: list[str]) -> list[bool]:
    """For each task, majority vote across trials."""
    results = []
    for tid in task_ids:
        rows = conn.execute(
            "SELECT passed FROM scores "
            "WHERE run_id = ? AND task_id = ? "
            "ORDER BY trial_index",
            [run_id, tid],
        ).fetchall()
        passes = [bool(r[0]) for r in rows if r[0] is not None]
        results.append(majority_vote(passes) if passes else False)
    return results


def _get_task_costs(conn: Any, run_id: str, task_ids: list[str]) -> np.ndarray:
    """Sum of trial costs per task."""
    costs = []
    for tid in task_ids:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) "
            "FROM trials WHERE run_id = ? AND task_id = ?",
            [run_id, tid],
        ).fetchone()
        costs.append(float(row[0]) if row else 0.0)
    return np.array(costs, dtype=np.float64)


def _get_latencies(conn: Any, run_id: str, task_ids: list[str]) -> np.ndarray:
    """All trial latencies for the given tasks."""
    if not task_ids:
        return np.array([], dtype=np.float64)
    placeholders = ",".join(["?"] * len(task_ids))
    sql = (
        "SELECT latency_ms_total FROM trials "
        f"WHERE run_id = ? AND task_id IN ({placeholders})"
    )
    rows = conn.execute(sql, [run_id, *task_ids]).fetchall()
    return np.array([float(r[0]) for r in rows], dtype=np.float64)


def _get_trial_passes(conn: Any, run_id: str, task_ids: list[str]) -> list[list[bool]]:
    """For each task, list of per-trial pass booleans."""
    result = []
    for tid in task_ids:
        rows = conn.execute(
            "SELECT passed FROM scores "
            "WHERE run_id = ? AND task_id = ? "
            "ORDER BY trial_index",
            [run_id, tid],
        ).fetchall()
        result.append([bool(r[0]) for r in rows if r[0] is not None])
    return result


def _get_mean_scores(conn: Any, run_id: str, task_ids: list[str]) -> np.ndarray:
    """Mean score per task (averaging across trials)."""
    means = []
    for tid in task_ids:
        row = conn.execute(
            "SELECT AVG(score) FROM scores WHERE run_id = ? AND task_id = ?",
            [run_id, tid],
        ).fetchone()
        means.append(float(row[0]) if row and row[0] is not None else 0.0)
    return np.array(means, dtype=np.float64)


def _get_majority_votes(conn: Any, run_id: str, task_ids: list[str]) -> np.ndarray:
    """Boolean array of majority vote pass/fail per task."""
    votes = []
    for tid in task_ids:
        rows = conn.execute(
            "SELECT passed FROM scores "
            "WHERE run_id = ? AND task_id = ? "
            "ORDER BY trial_index",
            [run_id, tid],
        ).fetchall()
        passes = [bool(r[0]) for r in rows if r[0] is not None]
        votes.append(majority_vote(passes) if passes else False)
    return np.array(votes, dtype=np.bool_)


def _get_total_cost(conn: Any, run_id: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM trials WHERE run_id = ?",
        [run_id],
    ).fetchone()
    return float(row[0]) if row else 0.0


def _compute_refusal_rate(conn: Any, run_id: str, task_ids: list[str]) -> float:
    """Fraction of trials with 'refusal' flag."""
    if not task_ids:
        return 0.0
    placeholders = ",".join(["?"] * len(task_ids))
    total_row = conn.execute(
        f"SELECT COUNT(*) FROM scores WHERE run_id = ? AND task_id IN ({placeholders})",
        [run_id, *task_ids],
    ).fetchone()
    refusal_row = conn.execute(
        f"SELECT COUNT(*) FROM scores WHERE run_id = ? AND task_id IN ({placeholders}) "
        "AND flags LIKE '%refusal%'",
        [run_id, *task_ids],
    ).fetchone()
    total = int(total_row[0]) if total_row else 0
    refusals = int(refusal_row[0]) if refusal_row else 0
    return (refusals / total * 100) if total > 0 else 0.0


def _get_notable_failures(conn: Any, run_id: str, limit: int = 5) -> list[str]:
    """Up to N anonymized failure patterns (category + flag summary, never raw text)."""
    rows = conn.execute(
        """SELECT SPLIT_PART(task_id, '/', 1) as category, flags
        FROM scores
        WHERE run_id = ? AND passed = false AND flags != '[]'
        LIMIT ?""",
        [run_id, limit * 3],
    ).fetchall()

    patterns: list[str] = []
    seen: set[str] = set()
    for cat, flags_raw in rows:
        import json

        flags = json.loads(flags_raw) if isinstance(flags_raw, str) else flags_raw
        if not flags:
            continue
        pattern = f"{cat}: {', '.join(flags)}"
        if pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)
        if len(patterns) >= limit:
            break
    return patterns
