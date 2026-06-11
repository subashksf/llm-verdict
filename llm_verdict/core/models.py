"""Core Pydantic data models for llm-verdict."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported task grading types."""

    EXACT = "exact"
    JSON_SCHEMA = "json_schema"
    REGEX = "regex"
    CODE_EXEC = "code_exec"
    TOOL_CALL = "tool_call"
    JUDGED = "judged"


class Message(BaseModel):
    """A single chat message."""

    role: str
    content: str


class GraderSpec(BaseModel):
    """Grader configuration attached to a task."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """A single evaluation task."""

    task_id: str
    category: str
    prompt: str | list[Message]
    task_type: TaskType
    grader: GraderSpec
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1


class TaskSuite(BaseModel):
    """A collection of tasks loaded from a suite directory."""

    tasks: list[Task]
    suite_hash: str = ""


class ModelConfig(BaseModel):
    """Full resolved model configuration snapshot."""

    model_id: str
    provider: str
    version: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """Written before any API call; the reproducibility anchor."""

    run_id: str
    created_at: datetime
    model: ModelConfig
    suite_hash: str
    harness_version: str
    judge_config_hash: str | None = None
    trials_per_task: int = 3
    temperature: float = 0.0
    budget_usd: float
    seed: int | None = None


class TrialResult(BaseModel):
    """Result of a single model call for one task."""

    run_id: str
    task_id: str
    trial_index: int
    request_hash: str
    response_text: str
    tool_calls: list[dict[str, Any]] | None = None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms_first_token: int | None = None
    latency_ms_total: int
    error: str | None = None
    cached: bool = False


class Score(BaseModel):
    """Grading result for a single trial."""

    run_id: str
    task_id: str
    trial_index: int
    grader_name: str
    grader_version: str
    passed: bool | None = None
    score: float
    rubric_scores: dict[str, Any] | None = None
    judge_reasoning: str | None = None
    flags: list[str] = Field(default_factory=list)


class VerdictResult(BaseModel):
    """Output of the pre-registered criteria engine."""

    run_id: str
    category: str
    outcome: str  # ADOPT | HOLD | REJECT | INSUFFICIENT_DATA
    fired_clauses: list[str] = Field(default_factory=list)


class Annotation(BaseModel):
    """Human label for judge calibration."""

    run_id: str
    task_id: str
    trial_index: int
    annotator: str
    passed: bool | None = None
    score: float | None = None
    notes: str = ""
