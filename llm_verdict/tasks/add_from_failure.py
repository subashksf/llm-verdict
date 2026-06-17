"""Promote a failed task from a run into a suite as a regression test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def promote_failure(conn: Any, run_id: str, task_id: str, suite_dir: Path) -> Path:
    """Write a new task YAML for a failed task into the target suite."""
    task_data = _get_failed_task_data(conn, run_id, task_id)
    if not suite_dir.is_dir():
        raise ValueError(f"Suite directory does not exist: {suite_dir}")

    safe_name = task_id.replace("/", "_").replace(" ", "_")
    output_path = suite_dir / f"{safe_name}_regression.yaml"
    if output_path.exists():
        raise ValueError(f"File already exists: {output_path}")

    output_path.write_text(yaml.dump(task_data, default_flow_style=False))
    return output_path


def _get_failed_task_data(conn: Any, run_id: str, task_id: str) -> dict[str, Any]:
    """Fetch task metadata from scores and build a task YAML structure."""
    row = conn.execute(
        "SELECT grader_name, flags FROM scores "
        "WHERE run_id = ? AND task_id = ? AND passed = false "
        "LIMIT 1",
        [run_id, task_id],
    ).fetchone()

    if row is None:
        raise ValueError(f"No failed score found for task {task_id} in run {run_id}")

    grader_name = row[0]
    category = task_id.split("/")[0] if "/" in task_id else "general"

    return {
        "task_id": f"{task_id}_regression",
        "category": category,
        "prompt": f"[FILL IN: original prompt for {task_id}]",
        "task_type": _grader_to_task_type(grader_name),
        "grader": {"name": grader_name, "params": {}},
        "metadata": {
            "source": "promoted_failure",
            "original_run_id": run_id,
            "original_task_id": task_id,
        },
        "version": 1,
    }


def _grader_to_task_type(grader_name: str) -> str:
    mapping = {
        "exact_match": "exact",
        "json_schema": "json_schema",
        "regex": "regex",
        "tool_call": "tool_call",
        "code_exec": "code_exec",
        "judge": "judged",
    }
    return mapping.get(grader_name, "exact")
