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


def test_judge_calibrate_no_db() -> None:
    """verdict judge calibrate errors when no database exists."""
    result = runner.invoke(app, ["judge", "calibrate", "--run", "some-run-id"])
    assert result.exit_code == 1
    assert "No database found" in result.output


def test_report_card_no_db() -> None:
    """verdict report card errors when no DB exists."""
    result = runner.invoke(app, ["report", "card", "some-run-id"])
    assert result.exit_code == 1
    assert "No database found" in result.output


def test_report_compare_no_db() -> None:
    """verdict report compare errors when no DB exists."""
    result = runner.invoke(app, ["report", "compare", "run-a", "run-b"])
    assert result.exit_code == 1
    assert "No database found" in result.output


def test_report_timeline_no_db() -> None:
    """verdict report timeline errors when no DB exists."""
    result = runner.invoke(app, ["report", "timeline", "--model-family", "claude"])
    assert result.exit_code == 1
    assert "No database found" in result.output


def test_db_query_no_db() -> None:
    """verdict db query errors when no DB exists."""
    result = runner.invoke(app, ["db", "query", "SELECT 1"])
    assert result.exit_code == 1
    assert "No database found" in result.output


def test_task_add_from_failure_missing_suite() -> None:
    """verdict task add-from-failure errors when --suite is missing."""
    result = runner.invoke(app, ["task", "add-from-failure", "run-1", "task-1"])
    assert result.exit_code == 2
