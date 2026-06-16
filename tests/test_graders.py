"""Unit tests for programmatic graders."""

from __future__ import annotations

from llm_verdict.graders.programmatic.exact_match import ExactMatchGrader
from llm_verdict.graders.programmatic.json_schema_grader import JsonSchemaGrader
from llm_verdict.graders.programmatic.regex_grader import RegexGrader
from llm_verdict.graders.programmatic.tool_call_grader import ToolCallGrader
from llm_verdict.graders.registry import get_grader

# --- Registry ---


def test_registry_returns_all_four_graders() -> None:
    for name in ("exact_match", "regex", "json_schema", "tool_call"):
        grader = get_grader(name)
        assert grader.name == name


# --- ExactMatchGrader ---


def test_exact_match_pass() -> None:
    g = ExactMatchGrader()
    result = g.grade("Paris", None, {"expected": "Paris"})
    assert result.passed is True
    assert result.score == 1.0


def test_exact_match_fail() -> None:
    g = ExactMatchGrader()
    result = g.grade("London", None, {"expected": "Paris"})
    assert result.passed is False
    assert result.score == 0.0


def test_exact_match_case_insensitive() -> None:
    g = ExactMatchGrader()
    result = g.grade("paris", None, {"expected": "Paris", "case_sensitive": False})
    assert result.passed is True


def test_exact_match_whitespace_strip() -> None:
    g = ExactMatchGrader()
    params = {"expected": "Paris", "strip_whitespace": True}
    result = g.grade("  Paris  \n", None, params)
    assert result.passed is True


def test_exact_match_no_strip() -> None:
    g = ExactMatchGrader()
    params = {"expected": "Paris", "strip_whitespace": False}
    result = g.grade("  Paris  ", None, params)
    assert result.passed is False


# --- RegexGrader ---


def test_regex_match_present() -> None:
    g = RegexGrader()
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    result = g.grade(
        "Contact me at test@example.com",
        None,
        {"pattern": email_pattern, "should_match": True},
    )
    assert result.passed is True


def test_regex_no_match() -> None:
    g = RegexGrader()
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    result = g.grade(
        "No email here",
        None,
        {"pattern": email_pattern, "should_match": True},
    )
    assert result.passed is False


def test_regex_should_not_match() -> None:
    g = RegexGrader()
    result = g.grade(
        "No numbers here",
        None,
        {"pattern": r"\d+", "should_match": False},
    )
    assert result.passed is True


def test_regex_should_not_match_but_does() -> None:
    g = RegexGrader()
    result = g.grade(
        "Got 42 items",
        None,
        {"pattern": r"\d+", "should_match": False},
    )
    assert result.passed is False


# --- JsonSchemaGrader ---


def test_json_schema_valid() -> None:
    g = JsonSchemaGrader()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    result = g.grade('{"name": "Alice", "age": 30}', None, {"schema": schema})
    assert result.passed is True
    assert result.score == 1.0


def test_json_schema_invalid_json() -> None:
    g = JsonSchemaGrader()
    schema = {"type": "object"}
    result = g.grade("not json at all", None, {"schema": schema})
    assert result.passed is False
    assert result.score == 0.0
    assert "format_violation" in result.flags


def test_json_schema_valid_json_bad_schema() -> None:
    g = JsonSchemaGrader()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    result = g.grade('{"age": 30}', None, {"schema": schema})
    assert result.passed is False


def test_json_schema_partial_credit() -> None:
    g = JsonSchemaGrader()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean"},
        },
        "required": ["name", "age", "active"],
    }
    # name is valid string, age is valid int, active is wrong type (string not bool)
    result = g.grade(
        '{"name": "Alice", "age": 25, "active": "yes"}',
        None,
        {"schema": schema, "partial_credit": True},
    )
    assert result.passed is False
    assert abs(result.score - 2.0 / 3.0) < 0.01


def test_json_schema_partial_credit_all_valid() -> None:
    g = JsonSchemaGrader()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    result = g.grade(
        '{"name": "Bob", "age": 40}',
        None,
        {"schema": schema, "partial_credit": True},
    )
    assert result.passed is True
    assert result.score == 1.0


# --- ToolCallGrader ---


def test_tool_call_exact_match() -> None:
    g = ToolCallGrader()
    args_json = '{"a": 42, "b": 17, "operation": "multiply"}'
    tool_calls = [{"name": "calculator", "arguments": args_json}]
    result = g.grade(
        "",
        tool_calls,
        {
            "expected_tool": "calculator",
            "expected_args": {"a": 42, "b": 17, "operation": "multiply"},
            "match_mode": "exact",
        },
    )
    assert result.passed is True


def test_tool_call_subset_match() -> None:
    g = ToolCallGrader()
    args_json = '{"a": 42, "b": 17, "operation": "multiply", "extra": true}'
    tool_calls = [{"name": "calculator", "arguments": args_json}]
    result = g.grade(
        "",
        tool_calls,
        {
            "expected_tool": "calculator",
            "expected_args": {"a": 42, "b": 17},
            "match_mode": "subset",
        },
    )
    assert result.passed is True


def test_tool_call_subset_missing_key() -> None:
    g = ToolCallGrader()
    tool_calls = [{"name": "calculator", "arguments": '{"a": 42}'}]
    result = g.grade(
        "",
        tool_calls,
        {
            "expected_tool": "calculator",
            "expected_args": {"a": 42, "b": 17},
            "match_mode": "subset",
        },
    )
    assert result.passed is False
    assert "args_mismatch" in result.flags


def test_tool_call_wrong_tool() -> None:
    g = ToolCallGrader()
    tool_calls = [{"name": "search", "arguments": "{}"}]
    result = g.grade(
        "",
        tool_calls,
        {"expected_tool": "calculator", "expected_args": {}, "match_mode": "exact"},
    )
    assert result.passed is False
    assert "wrong_tool" in result.flags


def test_tool_call_no_calls() -> None:
    g = ToolCallGrader()
    result = g.grade(
        "I cannot use tools",
        None,
        {"expected_tool": "calculator", "match_mode": "exact"},
    )
    assert result.passed is False
    assert "no_tool_call" in result.flags


def test_tool_call_semantic_key_mode() -> None:
    g = ToolCallGrader()
    args_json = '{"a": 99, "b": 1, "operation": "add"}'
    tool_calls = [{"name": "calculator", "arguments": args_json}]
    result = g.grade(
        "",
        tool_calls,
        {
            "expected_tool": "calculator",
            "expected_args": {"a": 42, "b": 17, "operation": "multiply"},
            "match_mode": "semantic_key",
        },
    )
    assert result.passed is True
