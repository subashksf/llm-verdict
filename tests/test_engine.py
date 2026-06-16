"""Integration tests for the run engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_verdict.core.models import GraderSpec, Task, TaskSuite, TaskType
from llm_verdict.providers.base import CompletionResponse
from llm_verdict.runner.cache import ResponseCache
from llm_verdict.runner.engine import BudgetExceededError, Engine, RunConfig
from llm_verdict.store.duck import (
    get_completed_trials,
    get_run,
    init_db,
)
from tests.mock_client import MockModelClient


def _make_suite() -> TaskSuite:
    """Create a minimal 3-task suite for testing."""
    tasks = [
        Task(
            task_id="test/exact-001",
            category="test",
            prompt="What is the capital of France?",
            task_type=TaskType.EXACT,
            grader=GraderSpec(
                name="exact_match",
                params={"expected": "Paris", "case_sensitive": False},
            ),
        ),
        Task(
            task_id="test/regex-001",
            category="test",
            prompt="Give me an email address.",
            task_type=TaskType.REGEX,
            grader=GraderSpec(
                name="regex",
                params={"pattern": r".+@.+\..+", "should_match": True},
            ),
        ),
        Task(
            task_id="test/tool-001",
            category="test",
            prompt="Use the calculator.",
            task_type=TaskType.TOOL_CALL,
            grader=GraderSpec(
                name="tool_call",
                params={
                    "expected_tool": "calculator",
                    "expected_args": {"a": 1, "b": 2},
                    "match_mode": "subset",
                },
            ),
        ),
    ]
    return TaskSuite(tasks=tasks, suite_hash="testhash123")


def _default_response() -> CompletionResponse:
    return CompletionResponse(
        text="Paris",
        tool_calls=[
            {"name": "calculator", "arguments": '{"a": 1, "b": 2, "op": "add"}'}
        ],
        tokens_in=50,
        tokens_out=10,
        model_version="v1",
        latency_ms_total=100,
        latency_ms_first_token=30,
    )


@pytest.mark.asyncio
async def test_full_run_produces_manifest_trials_scores(tmp_path: Path) -> None:
    """End-to-end: 3 tasks * 3 trials = 9 trials and 9 scores in DuckDB."""
    db_conn = init_db(tmp_path / "test.duckdb")
    client = MockModelClient(default_response=_default_response())
    engine = Engine(client=client, db_conn=db_conn)

    config = RunConfig(
        model_name="mock-model",
        suite=_make_suite(),
        trials_per_task=3,
        budget_usd=10.0,
    )
    manifest = await engine.execute_run(config)

    # Verify manifest in DB
    run_row = get_run(db_conn, manifest.run_id)
    assert run_row is not None
    assert run_row["status"] == "completed"
    assert run_row["suite_hash"] == "testhash123"

    # Verify trials
    trials = db_conn.execute(
        "SELECT COUNT(*) FROM trials WHERE run_id = ?", [manifest.run_id]
    ).fetchone()
    assert trials[0] == 9  # 3 tasks * 3 trials

    # Verify scores
    scores = db_conn.execute(
        "SELECT COUNT(*) FROM scores WHERE run_id = ?", [manifest.run_id]
    ).fetchone()
    assert scores[0] == 9

    # Verify mock client was called 9 times
    assert client._call_count == 9
    db_conn.close()


@pytest.mark.asyncio
async def test_budget_abort(tmp_path: Path) -> None:
    """In-run budget abort stops the run and marks it partial."""
    db_conn = init_db(tmp_path / "test.duckdb")

    # Use modest pricing so the pre-run estimate passes, but responses
    # report high token counts that blow through the tight budget during execution.
    expensive_response = CompletionResponse(
        text="Paris",
        tool_calls=None,
        tokens_in=10_000,
        tokens_out=10_000,
        model_version="v1",
        latency_ms_total=100,
    )
    client = MockModelClient(
        default_response=expensive_response,
        pricing_input=1.0,
        pricing_output=3.0,
        default_max_tokens=100,
    )
    engine = Engine(client=client, db_conn=db_conn)

    # Budget of $0.05 — each call costs ~$0.04 (10k*1/1M + 10k*3/1M = 0.04)
    # so after 1-2 calls the budget is exceeded
    config = RunConfig(
        model_name="mock-model",
        suite=_make_suite(),
        trials_per_task=3,
        budget_usd=0.05,
    )
    manifest = await engine.execute_run(config)

    run_row = get_run(db_conn, manifest.run_id)
    assert run_row is not None
    assert run_row["status"] == "partial"

    # Not all trials should have completed
    trials = db_conn.execute(
        "SELECT COUNT(*) FROM trials WHERE run_id = ?", [manifest.run_id]
    ).fetchone()
    assert trials[0] < 9
    assert trials[0] >= 1  # At least one trial ran before budget hit
    db_conn.close()


@pytest.mark.asyncio
async def test_budget_estimate_abort(tmp_path: Path) -> None:
    """Pre-run cost estimation aborts if estimate exceeds budget."""
    db_conn = init_db(tmp_path / "test.duckdb")
    client = MockModelClient(
        default_response=_default_response(),
        pricing_input=1000.0,
        pricing_output=3000.0,
        default_max_tokens=100000,
    )
    engine = Engine(client=client, db_conn=db_conn)

    config = RunConfig(
        model_name="mock-model",
        suite=_make_suite(),
        trials_per_task=3,
        budget_usd=0.001,
    )
    with pytest.raises(BudgetExceededError, match="Estimated cost"):
        await engine.execute_run(config)

    # No API calls should have been made
    assert client._call_count == 0
    db_conn.close()


@pytest.mark.asyncio
async def test_resume_completes_partial_run(tmp_path: Path) -> None:
    """Resume completes missing trials under the original manifest."""
    db_conn = init_db(tmp_path / "test.duckdb")
    suite = _make_suite()

    # First run: complete only some trials then abort
    client1 = MockModelClient(default_response=_default_response())
    engine1 = Engine(client=client1, db_conn=db_conn)
    config1 = RunConfig(
        model_name="mock-model",
        suite=suite,
        trials_per_task=3,
        budget_usd=10.0,
    )
    manifest = await engine1.execute_run(config1)
    run_id = manifest.run_id

    # Simulate partial: delete some trials to pretend they never ran
    db_conn.execute(
        "DELETE FROM trials WHERE run_id = ? AND trial_index = 2", [run_id]
    )
    db_conn.execute(
        "DELETE FROM scores WHERE run_id = ? AND trial_index = 2", [run_id]
    )
    update_status_sql = "UPDATE runs SET status = 'partial' WHERE run_id = ?"
    db_conn.execute(update_status_sql, [run_id])

    completed_before = get_completed_trials(db_conn, run_id)
    assert len(completed_before) == 6  # 3 tasks * 2 trials remaining

    # Resume the run
    client2 = MockModelClient(default_response=_default_response())
    engine2 = Engine(client=client2, db_conn=db_conn)
    config2 = RunConfig(
        model_name="mock-model",
        suite=suite,
        trials_per_task=3,
        budget_usd=10.0,
        resume_run_id=run_id,
    )
    resumed_manifest = await engine2.execute_run(config2)

    assert resumed_manifest.run_id == run_id

    # All 9 trials should be present now
    completed_after = get_completed_trials(db_conn, run_id)
    assert len(completed_after) == 9

    # Run should be marked completed
    run_row = get_run(db_conn, run_id)
    assert run_row["status"] == "completed"

    # Only 3 new API calls (the missing trial_index=2 for each task)
    assert client2._call_count == 3
    db_conn.close()


@pytest.mark.asyncio
async def test_cache_hit_skips_api_call(tmp_path: Path) -> None:
    """Second run with same params uses cache, no new API calls."""
    db_conn = init_db(tmp_path / "test.duckdb")
    cache = ResponseCache(tmp_path / "cache.duckdb")

    single_task_suite = TaskSuite(
        tasks=[
            Task(
                task_id="test/exact-001",
                category="test",
                prompt="What is 2+2?",
                task_type=TaskType.EXACT,
                grader=GraderSpec(name="exact_match", params={"expected": "4"}),
            )
        ],
        suite_hash="single_hash",
    )

    # First run: populates cache
    client1 = MockModelClient(default_response=_default_response())
    engine1 = Engine(client=client1, db_conn=db_conn, cache=cache)
    config1 = RunConfig(
        model_name="mock-model",
        suite=single_task_suite,
        trials_per_task=1,
        budget_usd=10.0,
    )
    await engine1.execute_run(config1)
    assert client1._call_count == 1

    # Second run: should hit cache
    client2 = MockModelClient(default_response=_default_response())
    engine2 = Engine(client=client2, db_conn=db_conn, cache=cache)
    config2 = RunConfig(
        model_name="mock-model",
        suite=single_task_suite,
        trials_per_task=1,
        budget_usd=10.0,
    )
    manifest2 = await engine2.execute_run(config2)
    assert client2._call_count == 0  # All from cache

    # The cached trial should be flagged
    row = db_conn.execute(
        "SELECT cached FROM trials WHERE run_id = ?", [manifest2.run_id]
    ).fetchone()
    assert row[0] is True

    cache.close()
    db_conn.close()


@pytest.mark.asyncio
async def test_grader_error_does_not_abort_run(tmp_path: Path) -> None:
    """A grader error on one task doesn't stop others from being graded."""
    db_conn = init_db(tmp_path / "test.duckdb")

    suite = TaskSuite(
        tasks=[
            Task(
                task_id="test/good",
                category="test",
                prompt="Say hi",
                task_type=TaskType.EXACT,
                grader=GraderSpec(name="exact_match", params={"expected": "Paris"}),
            ),
            Task(
                task_id="test/bad-grader",
                category="test",
                prompt="Do something",
                task_type=TaskType.EXACT,
                # Missing 'expected' param will cause grader error
                grader=GraderSpec(name="exact_match", params={}),
            ),
        ],
        suite_hash="graceful_hash",
    )

    client = MockModelClient(default_response=_default_response())
    engine = Engine(client=client, db_conn=db_conn)
    config = RunConfig(
        model_name="mock-model",
        suite=suite,
        trials_per_task=1,
        budget_usd=10.0,
    )
    manifest = await engine.execute_run(config)

    # Both trials should exist
    trials_count = db_conn.execute(
        "SELECT COUNT(*) FROM trials WHERE run_id = ?", [manifest.run_id]
    ).fetchone()
    assert trials_count[0] == 2

    # Both scores should exist (one with grader_error flag)
    scores = db_conn.execute(
        "SELECT task_id, flags FROM scores WHERE run_id = ?", [manifest.run_id]
    ).fetchall()
    assert len(scores) == 2

    error_scores = [s for s in scores if "grader_error" in s[1]]
    assert len(error_scores) == 1

    run_row = get_run(db_conn, manifest.run_id)
    assert run_row["status"] == "completed"
    db_conn.close()


@pytest.mark.asyncio
async def test_manifest_written_before_api_calls(tmp_path: Path) -> None:
    """RunManifest must be in DB before any model call happens."""
    db_conn = init_db(tmp_path / "test.duckdb")
    manifest_run_ids: list[str] = []

    class CheckingClient(MockModelClient):
        async def complete(self, messages, **kwargs):  # type: ignore[override]
            # At call time, the manifest should already be in DB
            runs = db_conn.execute("SELECT run_id FROM runs").fetchall()
            manifest_run_ids.extend(r[0] for r in runs)
            return await super().complete(messages, **kwargs)

    client = CheckingClient(default_response=_default_response())
    engine = Engine(client=client, db_conn=db_conn)
    suite = TaskSuite(
        tasks=[
            Task(
                task_id="test/check",
                category="test",
                prompt="hi",
                task_type=TaskType.EXACT,
                grader=GraderSpec(name="exact_match", params={"expected": "Paris"}),
            )
        ],
        suite_hash="check_hash",
    )
    config = RunConfig(
        model_name="mock-model", suite=suite, trials_per_task=1, budget_usd=10.0
    )
    manifest = await engine.execute_run(config)

    assert manifest.run_id in manifest_run_ids
    db_conn.close()
