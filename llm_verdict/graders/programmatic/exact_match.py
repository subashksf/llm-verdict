"""exact_match grader — normalized string equality."""

from __future__ import annotations

from typing import Any

from llm_verdict.graders.base import GradeResult


class ExactMatchGrader:
    """Normalized string comparison with configurable case/whitespace."""

    name = "exact_match"
    version = "1.0.0"

    def grade(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> GradeResult:
        expected: str = params["expected"]
        case_sensitive: bool = params.get("case_sensitive", True)
        strip_whitespace: bool = params.get("strip_whitespace", True)

        actual = response_text
        target = expected
        if strip_whitespace:
            actual = actual.strip()
            target = target.strip()
        if not case_sensitive:
            actual = actual.lower()
            target = target.lower()

        passed = actual == target
        return GradeResult(passed=passed, score=1.0 if passed else 0.0)
