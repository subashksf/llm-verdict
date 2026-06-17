"""code_exec grader — runs extracted code in a Docker sandbox."""

from __future__ import annotations

import re
from typing import Any

from llm_verdict.graders.base import GradeResult
from llm_verdict.runner.sandbox import run_in_sandbox


class CodeExecGrader:
    """Extract code from response, run in sandbox with test assertions."""

    name = "code_exec"
    version = "1.0.0"

    def grade(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> GradeResult:
        test_code: str = params["test_code"]
        timeout: int = params.get("timeout_seconds", 30)

        code = _extract_code(response_text)
        if not code:
            return GradeResult(passed=False, score=0.0, flags=["no_code_block"])

        result = run_in_sandbox(code, test_code, timeout_seconds=timeout)

        if result.timed_out:
            return GradeResult(passed=False, score=0.0, flags=["timeout"])

        test_count = len(
            [ln for ln in test_code.splitlines() if ln.strip().startswith("def test_")]
        )
        if test_count == 0:
            test_count = 1

        if result.exit_code == 0:
            return GradeResult(passed=True, score=1.0)

        failed = _count_assertion_errors(result.stderr)
        passed_count = max(0, test_count - failed)
        score = passed_count / test_count
        return GradeResult(passed=False, score=score)


def _extract_code(text: str) -> str:
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    lines = text.strip().splitlines()
    if any(line.strip().startswith("def ") for line in lines):
        return text.strip()
    return ""


def _count_assertion_errors(stderr: str) -> int:
    count = stderr.count("AssertionError")
    return max(count, 1) if stderr.strip() else 0
