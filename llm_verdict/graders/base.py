"""Grader protocol — the interface all graders implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GradeResult:
    """Output of a grader invocation."""

    passed: bool | None
    score: float
    flags: list[str] = field(default_factory=list)


class Grader(Protocol):
    """Protocol for all graders."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def grade(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> GradeResult: ...
