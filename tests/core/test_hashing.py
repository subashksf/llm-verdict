"""Tests for suite hashing — order invariance and stability."""

from llm_verdict.core.hashing import compute_suite_hash, hash_task
from llm_verdict.core.models import GraderSpec, Task, TaskType


def _make_task(task_id: str, prompt: str = "test prompt") -> Task:
    return Task(
        task_id=task_id,
        category="test",
        prompt=prompt,
        task_type=TaskType.EXACT,
        grader=GraderSpec(name="exact_match", params={"expected": "answer"}),
        metadata={"difficulty": "easy"},
        version=1,
    )


def test_suite_hash_order_invariant() -> None:
    """Reordering tasks must not change the suite hash."""
    tasks = [_make_task(f"task-{i}", f"prompt-{i}") for i in range(5)]
    hash_original = compute_suite_hash(tasks)

    reversed_tasks = list(reversed(tasks))
    hash_reversed = compute_suite_hash(reversed_tasks)

    assert hash_original == hash_reversed


def test_suite_hash_order_invariant_shuffled() -> None:
    """Multiple shuffle orderings all produce the same hash."""
    tasks = [_make_task(f"task-{i}", f"prompt-{i}") for i in range(10)]
    hash_original = compute_suite_hash(tasks)

    import random

    rng = random.Random(42)
    for _ in range(20):
        shuffled = tasks.copy()
        rng.shuffle(shuffled)
        assert compute_suite_hash(shuffled) == hash_original


def test_suite_hash_changes_with_content() -> None:
    """Changing task content must change the suite hash."""
    tasks_a = [_make_task("task-1", "prompt A")]
    tasks_b = [_make_task("task-1", "prompt B")]
    assert compute_suite_hash(tasks_a) != compute_suite_hash(tasks_b)


def test_task_hash_deterministic() -> None:
    """Same task always produces the same hash."""
    task = _make_task("stable-task", "stable prompt")
    assert hash_task(task) == hash_task(task)


def test_task_hash_dict_key_order_irrelevant() -> None:
    """Metadata dict key order does not affect hash."""
    task_a = _make_task("task-1")
    task_a.metadata = {"a": 1, "b": 2, "c": 3}

    task_b = _make_task("task-1")
    task_b.metadata = {"c": 3, "a": 1, "b": 2}

    assert hash_task(task_a) == hash_task(task_b)
