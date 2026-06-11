"""Content-addressing for task suites and configs.

Pure functions — no I/O. Takes data, returns hashes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llm_verdict.core.models import Task


def _canonicalize_value(value: Any) -> Any:
    """Recursively sort dicts by key for canonical JSON."""
    if isinstance(value, dict):
        return {k: _canonicalize_value(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    return value


def canonicalize_task(task: Task) -> str:
    """Produce a canonical JSON string for a single task."""
    data = task.model_dump(mode="json")
    canonical = _canonicalize_value(data)
    return json.dumps(
        canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )


def hash_task(task: Task) -> str:
    """SHA-256 hash of a single task's canonical form."""
    return hashlib.sha256(canonicalize_task(task).encode()).hexdigest()


def compute_suite_hash(tasks: list[Task]) -> str:
    """Compute order-invariant suite hash.

    Hash each task individually, sort the hex digests, then hash the
    concatenation. This ensures reordering tasks does not change the
    suite hash.
    """
    task_hashes = sorted(hash_task(task) for task in tasks)
    combined = "".join(task_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()
