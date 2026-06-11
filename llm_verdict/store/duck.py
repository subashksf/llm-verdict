"""DuckDB schema management — writers, readers, and initialization."""

from __future__ import annotations

from pathlib import Path

import duckdb

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
    seed INTEGER
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
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA_SQL)

    if VIEWS_SQL_PATH.exists():
        views_sql = VIEWS_SQL_PATH.read_text()
        conn.execute(views_sql)

    return conn
