"""Load, validate, and hash task suites from disk."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from llm_verdict.core.hashing import compute_suite_hash
from llm_verdict.core.models import Task, TaskSuite


class SuiteLoadError(Exception):
    """Raised when a suite fails to load or validate."""


def _load_task_file(path: Path) -> list[Task]:
    """Load tasks from a single YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return []

    items = raw if isinstance(raw, list) else [raw]
    tasks: list[Task] = []
    for item in items:
        try:
            tasks.append(Task.model_validate(item))
        except ValidationError as e:
            raise SuiteLoadError(
                f"Validation error in {path.name}: {e.error_count()} error(s)"
            ) from e
    return tasks


def load_suite(suite_path: Path) -> TaskSuite:
    """Load all tasks from a suite directory and compute its hash."""
    if not suite_path.is_dir():
        raise SuiteLoadError(f"Suite path is not a directory: {suite_path}")

    yaml_files = sorted(suite_path.glob("*.yaml")) + sorted(suite_path.glob("*.yml"))
    if not yaml_files:
        raise SuiteLoadError(f"No YAML files found in {suite_path}")

    tasks: list[Task] = []
    for yf in yaml_files:
        tasks.extend(_load_task_file(yf))

    if not tasks:
        raise SuiteLoadError(f"No tasks found in {suite_path}")

    _check_duplicate_ids(tasks)

    suite_hash = compute_suite_hash(tasks)
    return TaskSuite(tasks=tasks, suite_hash=suite_hash)


def _check_duplicate_ids(tasks: list[Task]) -> None:
    """Raise if any task_id appears more than once."""
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise SuiteLoadError(f"Duplicate task_id: {task.task_id}")
        seen.add(task.task_id)
