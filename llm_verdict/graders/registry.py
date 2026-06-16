"""Grader registry — name-based lookup, no if/elif dispatch."""

from __future__ import annotations

from llm_verdict.graders.base import Grader

_REGISTRY: dict[str, type] = {}


def register(grader_cls: type) -> type:
    """Register a grader class by its name attribute."""
    _REGISTRY[grader_cls.name] = grader_cls
    return grader_cls


def get_grader(name: str) -> Grader:
    """Look up and instantiate a grader by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown grader: {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def _register_builtins() -> None:
    from llm_verdict.graders.programmatic.exact_match import ExactMatchGrader
    from llm_verdict.graders.programmatic.json_schema_grader import (
        JsonSchemaGrader,
    )
    from llm_verdict.graders.programmatic.regex_grader import RegexGrader
    from llm_verdict.graders.programmatic.tool_call_grader import (
        ToolCallGrader,
    )

    register(ExactMatchGrader)
    register(RegexGrader)
    register(JsonSchemaGrader)
    register(ToolCallGrader)


_register_builtins()
