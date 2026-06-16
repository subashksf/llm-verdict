"""regex grader — pattern presence/absence assertions."""

from __future__ import annotations

import re
from typing import Any

from llm_verdict.graders.base import GradeResult


class RegexGrader:
    """Validate response matches (or doesn't match) a regex pattern."""

    name = "regex"
    version = "1.0.0"

    def grade(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> GradeResult:
        pattern: str = params["pattern"]
        should_match: bool = params.get("should_match", True)

        match = re.search(pattern, response_text) is not None
        passed = match == should_match
        return GradeResult(passed=passed, score=1.0 if passed else 0.0)
