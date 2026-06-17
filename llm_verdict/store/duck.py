"""DuckDB schema management — writers, readers, and initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from llm_verdict.core.models import RunManifest, Score, TrialResult

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    model_id VARCHAR NOT NULL,
    model_provider VARCHAR NOT NULL,
    model_version VARCHAR,
    model_params JSON,
    suite_hash VARCHAR NOT NULL,
    harness_version VARCHAR NOT NULL,
    judge_config_hash VARCHAR,
    trials_per_task INTEGER NOT NULL DEFAULT 3,
    temperature DOUBLE NOT NULL DEFAULT 0.0,
    budget_usd DOUBLE NOT NULL,
    seed INTEGER,
    status VARCHAR NOT NULL DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS trials (
    run_id VARCHAR NOT NULL,
    task_id VARCHAR NOT NULL,
    trial_index INTEGER NOT NULL,
    request_hash VARCHAR NOT NULL,
    response_text VARCHAR NOT NULL,
    tool_calls JSON,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd DOUBLE NOT NULL,
    latency_ms_first_token INTEGER,
    latency_ms_total INTEGER NOT NULL,
    error VARCHAR,
    cached BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (run_id, task_id, trial_index)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id VARCHAR NOT NULL,
    task_id VARCHAR NOT NULL,
    trial_index INTEGER NOT NULL,
    grader_name VARCHAR NOT NULL,
    grader_version VARCHAR NOT NULL,
    passed BOOLEAN,
    score DOUBLE NOT NULL,
    rubric_scores JSON,
    judge_reasoning VARCHAR,
    flags JSON NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS verdicts (
    run_id VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    fired_clauses JSON NOT NULL DEFAULT '[]',
    PRIMARY KEY (run_id, category)
);

CREATE TABLE IF NOT EXISTS annotations (
    run_id VARCHAR NOT NULL,
    task_id VARCHAR NOT NULL,
    trial_index INTEGER NOT NULL,
    annotator VARCHAR NOT NULL,
    passed BOOLEAN,
    score DOUBLE,
    notes VARCHAR NOT NULL DEFAULT ''
);
"""

VIEWS_SQL_PATH = Path(__file__).parent / "views.sql"


def init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Initialize database with schema and views."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA_SQL)

    if VIEWS_SQL_PATH.exists():
        views_sql = VIEWS_SQL_PATH.read_text()
        conn.execute(views_sql)

    return conn


def insert_run(
    conn: duckdb.DuckDBPyConnection,
    manifest: RunManifest,
    status: str = "running",
) -> None:
    """Insert a run manifest record."""
    conn.execute(
        "INSERT INTO runs (run_id, created_at, model_id, model_provider, "
        "model_version, model_params, suite_hash, harness_version, "
        "judge_config_hash, trials_per_task, temperature, budget_usd, seed, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            manifest.run_id,
            manifest.created_at,
            manifest.model.model_id,
            manifest.model.provider,
            manifest.model.version,
            json.dumps(manifest.model.params),
            manifest.suite_hash,
            manifest.harness_version,
            manifest.judge_config_hash,
            manifest.trials_per_task,
            manifest.temperature,
            manifest.budget_usd,
            manifest.seed,
            status,
        ],
    )


def update_run_status(
    conn: duckdb.DuckDBPyConnection, run_id: str, status: str
) -> None:
    """Update a run's lifecycle status (the only allowed mutation)."""
    conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", [status, run_id])


def insert_trial(conn: duckdb.DuckDBPyConnection, trial: TrialResult) -> None:
    """Insert a trial result record."""
    conn.execute(
        "INSERT INTO trials (run_id, task_id, trial_index, request_hash, "
        "response_text, tool_calls, tokens_in, tokens_out, cost_usd, "
        "latency_ms_first_token, latency_ms_total, error, cached) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            trial.run_id,
            trial.task_id,
            trial.trial_index,
            trial.request_hash,
            trial.response_text,
            json.dumps(trial.tool_calls) if trial.tool_calls else None,
            trial.tokens_in,
            trial.tokens_out,
            trial.cost_usd,
            trial.latency_ms_first_token,
            trial.latency_ms_total,
            trial.error,
            trial.cached,
        ],
    )


def insert_score(conn: duckdb.DuckDBPyConnection, score: Score) -> None:
    """Insert a score record (append-only)."""
    conn.execute(
        "INSERT INTO scores (run_id, task_id, trial_index, grader_name, "
        "grader_version, passed, score, rubric_scores, judge_reasoning, flags) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            score.run_id,
            score.task_id,
            score.trial_index,
            score.grader_name,
            score.grader_version,
            score.passed,
            score.score,
            json.dumps(score.rubric_scores) if score.rubric_scores else None,
            score.judge_reasoning,
            json.dumps(score.flags),
        ],
    )


def get_completed_trials(
    conn: duckdb.DuckDBPyConnection, run_id: str
) -> set[tuple[str, int]]:
    """Get the set of (task_id, trial_index) already completed for a run."""
    rows = conn.execute(
        "SELECT task_id, trial_index FROM trials WHERE run_id = ?", [run_id]
    ).fetchall()
    return {(row[0], row[1]) for row in rows}


def get_run(conn: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, Any] | None:
    """Load a run record as a dict, or None if not found."""
    row = conn.execute(
        "SELECT run_id, created_at, model_id, model_provider, model_version, "
        "model_params, suite_hash, harness_version, judge_config_hash, "
        "trials_per_task, temperature, budget_usd, seed, status "
        "FROM runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "created_at": row[1],
        "model_id": row[2],
        "model_provider": row[3],
        "model_version": row[4],
        "model_params": row[5],
        "suite_hash": row[6],
        "harness_version": row[7],
        "judge_config_hash": row[8],
        "trials_per_task": row[9],
        "temperature": row[10],
        "budget_usd": row[11],
        "seed": row[12],
        "status": row[13],
    }


def get_run_total_cost(conn: duckdb.DuckDBPyConnection, run_id: str) -> float:
    """Sum of cost_usd for all trials in a run."""
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) FROM trials WHERE run_id = ?",
        [run_id],
    ).fetchone()
    return float(row[0]) if row else 0.0
