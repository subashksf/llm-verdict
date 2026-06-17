"""Judge calibration — Cohen's kappa between judge and human annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb


@dataclass(frozen=True)
class CalibrationResult:
    """Output of a calibration computation."""

    kappa: float
    agreement_rate: float
    n_samples: int
    per_criterion: dict[str, float]


def compute_cohens_kappa(judge_labels: list[bool], human_labels: list[bool]) -> float:
    """Compute Cohen's kappa for two binary label lists."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("Label lists must be equal length")
    n = len(judge_labels)
    if n == 0:
        return 0.0

    agree = sum(j == h for j, h in zip(judge_labels, human_labels))
    p_o = agree / n

    j_pos = sum(judge_labels) / n
    h_pos = sum(human_labels) / n
    p_e = j_pos * h_pos + (1 - j_pos) * (1 - h_pos)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def get_calibration_samples(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Sample scored trials from a run for human labeling."""
    rows = conn.execute(
        "SELECT t.run_id, t.task_id, t.trial_index, t.response_text, "
        "s.passed as judge_passed, s.score as judge_score "
        "FROM trials t JOIN scores s "
        "ON t.run_id = s.run_id AND t.task_id = s.task_id "
        "AND t.trial_index = s.trial_index "
        "WHERE t.run_id = ? "
        "ORDER BY RANDOM() LIMIT ?",
        [run_id, sample_size],
    ).fetchall()

    return [
        {
            "run_id": r[0],
            "task_id": r[1],
            "trial_index": r[2],
            "response_text": r[3],
            "judge_passed": r[4],
            "judge_score": r[5],
        }
        for r in rows
    ]


def insert_annotation(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    task_id: str,
    trial_index: int,
    annotator: str,
    passed: bool | None,
    notes: str = "",
) -> None:
    """Insert a human annotation record."""
    conn.execute(
        "INSERT INTO annotations (run_id, task_id, trial_index, annotator, "
        "passed, score, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [run_id, task_id, trial_index, annotator, passed, None, notes],
    )


def compute_calibration(
    conn: duckdb.DuckDBPyConnection, run_id: str
) -> CalibrationResult | None:
    """Compute kappa from stored annotations vs judge scores for a run."""
    rows = conn.execute(
        "SELECT a.passed as human_passed, s.passed as judge_passed "
        "FROM annotations a JOIN scores s "
        "ON a.run_id = s.run_id AND a.task_id = s.task_id "
        "AND a.trial_index = s.trial_index "
        "WHERE a.run_id = ? AND a.passed IS NOT NULL AND s.passed IS NOT NULL",
        [run_id],
    ).fetchall()

    if not rows:
        return None

    human_labels = [bool(r[0]) for r in rows]
    judge_labels = [bool(r[1]) for r in rows]

    kappa = compute_cohens_kappa(judge_labels, human_labels)
    agreement = sum(j == h for j, h in zip(judge_labels, human_labels)) / len(rows)

    return CalibrationResult(
        kappa=kappa,
        agreement_rate=agreement,
        n_samples=len(rows),
        per_criterion={},
    )
