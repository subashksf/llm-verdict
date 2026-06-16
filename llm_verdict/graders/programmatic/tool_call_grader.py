"""tool_call grader — validate tool name + args matching."""

from __future__ import annotations

import json
from typing import Any

from llm_verdict.graders.base import GradeResult


class ToolCallGrader:
    """Validate model emitted expected tool call with matching args."""

    name = "tool_call"
    version = "1.0.0"

    def grade(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> GradeResult:
        expected_tool: str = params["expected_tool"]
        expected_args: dict[str, Any] = params.get("expected_args", {})
        match_mode: str = params.get("match_mode", "exact")

        if not tool_calls:
            return GradeResult(passed=False, score=0.0, flags=["no_tool_call"])

        for tc in tool_calls:
            name = _get_tool_name(tc)
            if name != expected_tool:
                continue
            actual_args = _get_tool_args(tc)
            if _args_match(actual_args, expected_args, match_mode):
                return GradeResult(passed=True, score=1.0)
            return GradeResult(passed=False, score=0.5, flags=["args_mismatch"])

        return GradeResult(passed=False, score=0.0, flags=["wrong_tool"])


def _get_tool_name(tc: dict[str, Any]) -> str:
    if "name" in tc:
        return tc["name"]
    return tc.get("function", {}).get("name", "")


def _get_tool_args(tc: dict[str, Any]) -> dict[str, Any]:
    if "arguments" in tc:
        raw = tc["arguments"]
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    return tc.get("function", {}).get("arguments", {})


def _args_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    mode: str,
) -> bool:
    if mode == "exact":
        return actual == expected
    if mode == "subset":
        return all(
            k in actual and actual[k] == expected[k] for k in expected
        )
    if mode == "semantic_key":
        return all(k in actual for k in expected)
    return actual == expected
