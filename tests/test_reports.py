"""Tests for report generation — rendering and no task text leakage."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_verdict.core.models import (
    ModelConfig,
    RunManifest,
    Score,
    TrialResult,
)
from llm_verdict.reporting.data import compute_compare_report, compute_run_report
from llm_verdict.reporting.renderer import (
    render_card_html,
    render_card_markdown,
    render_compare_markdown,
)
from llm_verdict.store.duck import init_db, insert_run, insert_score, insert_trial


@pytest.fixture
def db_conn(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    return init_db(db_path)


def _insert_test_run(conn, run_id: str, suite_hash: str = "abc123") -> None:
    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        model=ModelConfig(model_id="test-model", provider="litellm", params={}),
        suite_hash=suite_hash,
        harness_version="0.2.0",
        trials_per_task=3,
        temperature=0.0,
        budget_usd=10.0,
    )
    insert_run(conn, manifest, status="completed")


def _insert_trial_and_score(
    conn,
    run_id: str,
    task_id: str,
    trial_idx: int,
    passed: bool,
    score: float = 1.0,
    cost: float = 0.01,
    latency: int = 500,
    response_text: str = "SECRET_TASK_CONTENT_MUST_NOT_LEAK",
) -> None:
    trial = TrialResult(
        run_id=run_id,
        task_id=task_id,
        trial_index=trial_idx,
        request_hash="hash123",
        response_text=response_text,
        tokens_in=100,
        tokens_out=200,
        cost_usd=cost,
        latency_ms_total=latency,
        cached=False,
    )
    insert_trial(conn, trial)

    score_rec = Score(
        run_id=run_id,
        task_id=task_id,
        trial_index=trial_idx,
        grader_name="exact_match",
        grader_version="1.0",
        passed=passed,
        score=score,
        flags=[],
    )
    insert_score(conn, score_rec)


class TestReportCard:
    def test_renders_basic_card(self, db_conn):
        _insert_test_run(db_conn, "run1")
        for i in range(3):
            _insert_trial_and_score(db_conn, "run1", "coding/t1", i, True)
            _insert_trial_and_score(db_conn, "run1", "coding/t2", i, False, score=0.0)
            _insert_trial_and_score(db_conn, "run1", "reasoning/t3", i, True)

        report = compute_run_report(db_conn, "run1")
        assert report.model_id == "test-model"
        assert report.total_cost > 0
        assert len(report.categories) == 2

        md = render_card_markdown(report)
        assert "test-model" in md
        assert "abc123" in md

    def test_no_task_text_leakage(self, db_conn):
        """Reports must never contain raw task text or completions."""
        _insert_test_run(db_conn, "run1")
        for i in range(3):
            _insert_trial_and_score(
                db_conn,
                "run1",
                "coding/t1",
                i,
                True,
                response_text="THIS_IS_PRIVATE_COMPLETION_TEXT",
            )

        report = compute_run_report(db_conn, "run1")
        md = render_card_markdown(report)
        html = render_card_html(report)

        assert "THIS_IS_PRIVATE_COMPLETION_TEXT" not in md
        assert "THIS_IS_PRIVATE_COMPLETION_TEXT" not in html
        assert "SECRET_TASK_CONTENT" not in md
        assert "SECRET_TASK_CONTENT" not in html

    def test_html_self_contained(self, db_conn):
        _insert_test_run(db_conn, "run1")
        _insert_trial_and_score(db_conn, "run1", "cat/t1", 0, True)

        report = compute_run_report(db_conn, "run1")
        html = render_card_html(report)
        assert "<!DOCTYPE html>" in html
        assert "<style>" in html


class TestCompareReport:
    def test_same_suite_hash_required(self, db_conn):
        _insert_test_run(db_conn, "run_a", suite_hash="hash1")
        _insert_test_run(db_conn, "run_b", suite_hash="hash2")
        _insert_trial_and_score(db_conn, "run_a", "cat/t1", 0, True)
        _insert_trial_and_score(db_conn, "run_b", "cat/t1", 0, True)

        with pytest.raises(ValueError, match="different suite hashes"):
            compute_compare_report(db_conn, "run_a", "run_b")

    def test_renders_comparison(self, db_conn):
        _insert_test_run(db_conn, "run_a", suite_hash="same")
        _insert_test_run(db_conn, "run_b", suite_hash="same")

        for i in range(3):
            _insert_trial_and_score(db_conn, "run_a", "coding/t1", i, True)
            _insert_trial_and_score(db_conn, "run_a", "coding/t2", i, False, score=0.0)
            _insert_trial_and_score(db_conn, "run_b", "coding/t1", i, True)
            _insert_trial_and_score(db_conn, "run_b", "coding/t2", i, True)

        report = compute_compare_report(db_conn, "run_a", "run_b")
        md = render_compare_markdown(report)
        assert "Head-to-Head" in md
        assert "McNemar" in md
