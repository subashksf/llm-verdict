"""Tests for model serialization round-trips."""

from datetime import datetime, timezone

from llm_verdict.core.models import (
    Annotation,
    GraderSpec,
    Message,
    ModelConfig,
    RunManifest,
    Score,
    Task,
    TaskSuite,
    TaskType,
    TrialResult,
    VerdictResult,
)


def test_task_roundtrip() -> None:
    """Task serializes to JSON and back identically."""
    task = Task(
        task_id="test/roundtrip-001",
        category="test",
        prompt="What is 2+2?",
        task_type=TaskType.EXACT,
        grader=GraderSpec(name="exact_match", params={"expected": "4"}),
        metadata={"difficulty": "trivial"},
        version=1,
    )
    data = task.model_dump(mode="json")
    restored = Task.model_validate(data)
    assert restored == task


def test_task_with_messages_roundtrip() -> None:
    """Task with chat messages round-trips correctly."""
    task = Task(
        task_id="test/chat-001",
        category="test",
        prompt=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ],
        task_type=TaskType.JUDGED,
        grader=GraderSpec(name="llm_judge", params={"rubric_id": "quality"}),
        metadata={},
        version=2,
    )
    data = task.model_dump(mode="json")
    restored = Task.model_validate(data)
    assert restored == task


def test_run_manifest_roundtrip() -> None:
    """RunManifest round-trips through JSON."""
    manifest = RunManifest(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        model=ModelConfig(
            model_id="claude-sonnet-4-6",
            provider="anthropic",
            version="20250514",
            params={"max_tokens": 4096},
        ),
        suite_hash="abc123def456",
        harness_version="0.1.0+abc1234",
        judge_config_hash=None,
        trials_per_task=3,
        temperature=0.0,
        budget_usd=5.0,
        seed=42,
    )
    data = manifest.model_dump(mode="json")
    restored = RunManifest.model_validate(data)
    assert restored == manifest


def test_trial_result_roundtrip() -> None:
    """TrialResult round-trips through JSON."""
    trial = TrialResult(
        run_id="run-1",
        task_id="task-1",
        trial_index=0,
        request_hash="deadbeef",
        response_text="Paris",
        tool_calls=None,
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.001,
        latency_ms_first_token=150,
        latency_ms_total=500,
        error=None,
        cached=False,
    )
    data = trial.model_dump(mode="json")
    restored = TrialResult.model_validate(data)
    assert restored == trial


def test_score_roundtrip() -> None:
    """Score round-trips through JSON."""
    score = Score(
        run_id="run-1",
        task_id="task-1",
        trial_index=0,
        grader_name="exact_match",
        grader_version="1.0.0",
        passed=True,
        score=1.0,
        rubric_scores=None,
        judge_reasoning=None,
        flags=["format_violation"],
    )
    data = score.model_dump(mode="json")
    restored = Score.model_validate(data)
    assert restored == score


def test_verdict_result_roundtrip() -> None:
    """VerdictResult round-trips through JSON."""
    verdict = VerdictResult(
        run_id="run-1",
        category="code_review",
        outcome="ADOPT",
        fired_clauses=["pass_rate_delta_pp >= 5", "ci_overlap: false"],
    )
    data = verdict.model_dump(mode="json")
    restored = VerdictResult.model_validate(data)
    assert restored == verdict


def test_annotation_roundtrip() -> None:
    """Annotation round-trips through JSON."""
    annotation = Annotation(
        run_id="run-1",
        task_id="task-1",
        trial_index=0,
        annotator="human-01",
        passed=True,
        score=0.9,
        notes="Good response",
    )
    data = annotation.model_dump(mode="json")
    restored = Annotation.model_validate(data)
    assert restored == annotation


def test_task_suite_with_hash() -> None:
    """TaskSuite holds tasks and hash."""
    suite = TaskSuite(
        tasks=[
            Task(
                task_id="t1",
                category="test",
                prompt="hi",
                task_type=TaskType.EXACT,
                grader=GraderSpec(name="exact_match", params={"expected": "hi"}),
            )
        ],
        suite_hash="abc123",
    )
    data = suite.model_dump(mode="json")
    restored = TaskSuite.model_validate(data)
    assert restored == suite
    assert len(restored.tasks) == 1
