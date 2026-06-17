"""Tests for the code_exec grader."""

from __future__ import annotations

from unittest.mock import patch

from llm_verdict.graders.programmatic.code_exec_grader import (
    CodeExecGrader,
    _extract_code,
)
from llm_verdict.runner.sandbox import SandboxResult


class TestExtractCode:
    def test_fenced_python(self):
        text = "Here's the code:\n```python\ndef add(a, b):\n    return a + b\n```"
        assert _extract_code(text) == "def add(a, b):\n    return a + b"

    def test_fenced_no_language(self):
        text = "```\ndef add(a, b):\n    return a + b\n```"
        assert _extract_code(text) == "def add(a, b):\n    return a + b"

    def test_raw_function(self):
        text = "def add(a, b):\n    return a + b"
        assert _extract_code(text) == "def add(a, b):\n    return a + b"

    def test_no_code(self):
        text = "I don't know how to write that function."
        assert _extract_code(text) == ""


class TestCodeExecGrader:
    @patch("llm_verdict.graders.programmatic.code_exec_grader.run_in_sandbox")
    def test_passing_code(self, mock_sandbox):
        mock_sandbox.return_value = SandboxResult(
            stdout="", stderr="", exit_code=0, timed_out=False
        )
        grader = CodeExecGrader()
        result = grader.grade(
            "```python\ndef add(a, b): return a + b\n```",
            None,
            {"test_code": "def test_add(): assert add(1,2)==3", "timeout_seconds": 30},
        )
        assert result.passed is True
        assert result.score == 1.0

    @patch("llm_verdict.graders.programmatic.code_exec_grader.run_in_sandbox")
    def test_failing_code(self, mock_sandbox):
        mock_sandbox.return_value = SandboxResult(
            stdout="", stderr="AssertionError", exit_code=1, timed_out=False
        )
        grader = CodeExecGrader()
        result = grader.grade(
            "```python\ndef add(a, b): return 0\n```",
            None,
            {"test_code": "def test_add(): assert add(1,2)==3", "timeout_seconds": 30},
        )
        assert result.passed is False
        assert result.score == 0.0

    @patch("llm_verdict.graders.programmatic.code_exec_grader.run_in_sandbox")
    def test_timeout(self, mock_sandbox):
        mock_sandbox.return_value = SandboxResult(
            stdout="", stderr="Execution timed out", exit_code=-1, timed_out=True
        )
        grader = CodeExecGrader()
        result = grader.grade(
            "```python\nwhile True: pass\n```",
            None,
            {"test_code": "def test_x(): pass", "timeout_seconds": 5},
        )
        assert result.passed is False
        assert "timeout" in result.flags

    def test_no_code_block(self):
        grader = CodeExecGrader()
        with patch(
            "llm_verdict.graders.programmatic.code_exec_grader.run_in_sandbox"
        ) as mock_sandbox:
            result = grader.grade(
                "I can't write that.",
                None,
                {"test_code": "def test_x(): pass"},
            )
        assert result.passed is False
        assert "no_code_block" in result.flags
        mock_sandbox.assert_not_called()
