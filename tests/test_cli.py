"""Tests for CLI commands."""

from typer.testing import CliRunner

from llm_verdict.cli import app

runner = CliRunner()


def test_suite_validate_examples() -> None:
    """verdict suite validate suites/_examples succeeds."""
    result = runner.invoke(app, ["suite", "validate", "suites/_examples"])
    assert result.exit_code == 0
    assert "OK: 6 tasks validated" in result.output
    assert "Suite hash:" in result.output


def test_suite_hash_examples() -> None:
    """verdict suite hash suites/_examples prints a hash."""
    result = runner.invoke(app, ["suite", "hash", "suites/_examples"])
    assert result.exit_code == 0
    assert len(result.output.strip()) == 64  # SHA-256 hex


def test_suite_validate_bad_path() -> None:
    """verdict suite validate with bad path fails."""
    result = runner.invoke(app, ["suite", "validate", "/nonexistent"])
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_run_missing_model_errors() -> None:
    """verdict run with unknown model config errors."""
    result = runner.invoke(
        app, ["run", "--model", "nonexistent", "--suite", "suites/_examples"]
    )
    assert result.exit_code == 1


def test_grade_stub() -> None:
    """verdict grade prints not implemented."""
    result = runner.invoke(app, ["grade", "some-run-id"])
    assert "not implemented" in result.output


def test_judge_calibrate_stub() -> None:
    """verdict judge calibrate prints not implemented."""
    result = runner.invoke(app, ["judge", "calibrate", "--run", "some-run-id"])
    assert "not implemented" in result.output


def test_report_card_stub() -> None:
    """verdict report card prints not implemented."""
    result = runner.invoke(app, ["report", "card", "some-run-id"])
    assert "not implemented" in result.output


def test_report_compare_stub() -> None:
    """verdict report compare prints not implemented."""
    result = runner.invoke(app, ["report", "compare", "run-a", "run-b"])
    assert "not implemented" in result.output


def test_report_timeline_stub() -> None:
    """verdict report timeline prints not implemented."""
    result = runner.invoke(app, ["report", "timeline", "--model-family", "claude"])
    assert "not implemented" in result.output


def test_db_query_stub() -> None:
    """verdict db query prints not implemented."""
    result = runner.invoke(app, ["db", "query", "SELECT 1"])
    assert "not implemented" in result.output


def test_task_add_from_failure_stub() -> None:
    """verdict task add-from-failure prints not implemented."""
    result = runner.invoke(app, ["task", "add-from-failure", "run-1", "task-1"])
    assert "not implemented" in result.output
