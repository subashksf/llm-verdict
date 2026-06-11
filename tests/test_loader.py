"""Tests for the task suite loader."""

from pathlib import Path

import pytest

from llm_verdict.core.hashing import compute_suite_hash
from llm_verdict.tasks.loader import SuiteLoadError, load_suite

EXAMPLES_PATH = Path(__file__).parent.parent / "suites" / "_examples"


def test_load_examples_suite() -> None:
    """The _examples suite loads successfully."""
    suite = load_suite(EXAMPLES_PATH)
    assert len(suite.tasks) == 6
    assert suite.suite_hash != ""


def test_examples_hash_is_order_invariant() -> None:
    """Loading from disk produces an order-invariant hash."""
    suite = load_suite(EXAMPLES_PATH)
    recomputed = compute_suite_hash(list(reversed(suite.tasks)))
    assert suite.suite_hash == recomputed


def test_load_nonexistent_path() -> None:
    """Loading from a missing path raises SuiteLoadError."""
    with pytest.raises(SuiteLoadError, match="not a directory"):
        load_suite(Path("/nonexistent/path"))


def test_load_empty_directory(tmp_path: Path) -> None:
    """Loading from a directory with no YAML files raises."""
    with pytest.raises(SuiteLoadError, match="No YAML files"):
        load_suite(tmp_path)


def test_duplicate_task_ids(tmp_path: Path) -> None:
    """Duplicate task_ids raise SuiteLoadError."""
    content = """
- task_id: "dup-001"
  category: "test"
  prompt: "hello"
  task_type: "exact"
  grader:
    name: "exact_match"
    params:
      expected: "world"
  version: 1
- task_id: "dup-001"
  category: "test"
  prompt: "goodbye"
  task_type: "exact"
  grader:
    name: "exact_match"
    params:
      expected: "world"
  version: 1
"""
    (tmp_path / "tasks.yaml").write_text(content)
    with pytest.raises(SuiteLoadError, match="Duplicate task_id"):
        load_suite(tmp_path)
