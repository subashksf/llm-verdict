"""Tests for the add-from-failure workflow."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from llm_verdict.core.models import ModelConfig, RunManifest, Score, TrialResult
from llm_verdict.store.duck import init_db, insert_run, insert_score, insert_trial
from llm_verdict.tasks.add_from_failure import promote_failure


@pytest.fixture
def db_conn(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    return init_db(db_path)


def _setup_failed_run(conn, run_id: str = "run1") -> None:
    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        model=ModelConfig(model_id="test-model", provider="litellm", params={}),
        suite_hash="abc123",
        harness_version="0.2.0",
        trials_per_task=3,
        temperature=0.0,
        budget_usd=10.0,
    )
    insert_run(conn, manifest, status="completed")

    trial = TrialResult(
        run_id=run_id,
        task_id="coding/hard-task",
        trial_index=0,
        request_hash="h1",
        response_text="wrong answer",
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.01,
        latency_ms_total=500,
        cached=False,
    )
    insert_trial(conn, trial)

    score = Score(
        run_id=run_id,
        task_id="coding/hard-task",
        trial_index=0,
        grader_name="exact_match",
        grader_version="1.0",
        passed=False,
        score=0.0,
        flags=["wrong_answer"],
    )
    insert_score(conn, score)


class TestPromoteFailure:
    def test_creates_yaml_file(self, db_conn, tmp_path: Path):
        _setup_failed_run(db_conn)
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()

        output = promote_failure(db_conn, "run1", "coding/hard-task", suite_dir)
        assert output.exists()
        assert output.suffix == ".yaml"

        data = yaml.safe_load(output.read_text())
        assert data["task_id"] == "coding/hard-task_regression"
        assert data["category"] == "coding"
        assert data["metadata"]["original_task_id"] == "coding/hard-task"
        assert data["metadata"]["source"] == "promoted_failure"

    def test_errors_on_no_failure(self, db_conn, tmp_path: Path):
        _setup_failed_run(db_conn)
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()

        with pytest.raises(ValueError, match="No failed score"):
            promote_failure(db_conn, "run1", "nonexistent/task", suite_dir)

    def test_errors_on_duplicate(self, db_conn, tmp_path: Path):
        _setup_failed_run(db_conn)
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()

        promote_failure(db_conn, "run1", "coding/hard-task", suite_dir)
        with pytest.raises(ValueError, match="already exists"):
            promote_failure(db_conn, "run1", "coding/hard-task", suite_dir)

    def test_errors_on_missing_suite_dir(self, db_conn, tmp_path: Path):
        _setup_failed_run(db_conn)
        with pytest.raises(ValueError, match="does not exist"):
            promote_failure(db_conn, "run1", "coding/hard-task", tmp_path / "nope")
