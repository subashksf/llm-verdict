"""Attempt repair of malformed JSON from LLM judge responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RepairResult:
    """Result of a JSON repair attempt."""

    data: dict[str, Any] | None
    repaired: bool
    raw: str


def attempt_json_repair(text: str) -> RepairResult:
    """Try to extract and repair JSON from judge response text."""
    cleaned = _strip_fences(text)

    parsed = _try_parse(cleaned)
    if parsed is not None:
        return RepairResult(data=parsed, repaired=False, raw=text)

    fixed = _fix_trailing_commas(cleaned)
    parsed = _try_parse(fixed)
    if parsed is not None:
        return RepairResult(data=parsed, repaired=True, raw=text)

    extracted = _extract_json_block(text)
    if extracted:
        parsed = _try_parse(extracted)
        if parsed is not None:
            return RepairResult(data=parsed, repaired=True, raw=text)
        fixed = _fix_trailing_commas(extracted)
        parsed = _try_parse(fixed)
        if parsed is not None:
            return RepairResult(data=parsed, repaired=True, raw=text)

    return RepairResult(data=None, repaired=False, raw=text)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _fix_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _try_parse(text: str) -> dict[str, Any] | None:
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None
