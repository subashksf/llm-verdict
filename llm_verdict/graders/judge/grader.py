"""LLM judge grader — uses a separate model to evaluate responses."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from llm_verdict.graders.base import GradeResult
from llm_verdict.graders.judge.config import (
    JudgeConfig,
    build_judge_prompt,
    load_judge_config,
)
from llm_verdict.graders.judge.json_repair import attempt_json_repair
from llm_verdict.providers.base import ModelClient


class LLMJudgeGrader:
    """Grader that uses an LLM judge with rubrics."""

    name = "llm_judge"
    version = "1.0.0"

    def __init__(
        self,
        judge_client: ModelClient | None = None,
        judge_config: JudgeConfig | None = None,
        model_under_test: str | None = None,
    ) -> None:
        self._judge_client = judge_client
        self._judge_config = judge_config
        self._model_under_test = model_under_test

    def configure(
        self,
        judge_client: ModelClient,
        judge_config: JudgeConfig,
        model_under_test: str,
    ) -> None:
        """Set judge configuration post-construction."""
        self._judge_client = judge_client
        self._judge_config = judge_config
        self._model_under_test = model_under_test

    def grade(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> GradeResult:
        if self._judge_client is None or self._judge_config is None:
            raise RuntimeError("LLM judge not configured — call configure() first")

        _validate_not_self_grading(self._judge_client.model_id, self._model_under_test)

        task_prompt = params.get("task_prompt", "")
        criteria = params.get("criteria", [])

        config = self._judge_config
        if criteria:
            from llm_verdict.graders.judge.config import JudgeCriterion

            config = config.model_copy(
                update={"criteria": [JudgeCriterion(**c) for c in criteria]}
            )

        prompt = build_judge_prompt(config, task_prompt, response_text)
        messages = [{"role": "user", "content": prompt}]

        judge_response = asyncio.get_event_loop().run_until_complete(
            self._judge_client.complete(
                messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )

        return _parse_judge_response(judge_response.text, config)


def grade_with_judge_async(
    judge_client: ModelClient,
    judge_config: JudgeConfig,
    model_under_test: str,
    response_text: str,
    task_prompt: str,
    criteria: list[dict[str, Any]] | None = None,
) -> GradeResult:
    """Synchronous wrapper for judge grading within async context."""
    _validate_not_self_grading(judge_client.model_id, model_under_test)

    config = judge_config
    if criteria:
        from llm_verdict.graders.judge.config import JudgeCriterion

        config = config.model_copy(
            update={"criteria": [JudgeCriterion(**c) for c in criteria]}
        )

    prompt = build_judge_prompt(config, task_prompt, response_text)
    messages = [{"role": "user", "content": prompt}]

    loop = asyncio.get_event_loop()
    judge_response = loop.run_until_complete(
        judge_client.complete(
            messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    )

    return _parse_judge_response(judge_response.text, config)


def _validate_not_self_grading(
    judge_model_id: str, model_under_test: str | None
) -> None:
    if model_under_test and judge_model_id == model_under_test:
        raise ValueError(
            f"Judge model cannot be the model under test: {judge_model_id}"
        )


def _parse_judge_response(text: str, config: JudgeConfig) -> GradeResult:
    repair = attempt_json_repair(text)
    flags: list[str] = []

    if repair.data is None:
        return GradeResult(passed=None, score=0.0, flags=["judge_parse_error"])

    if repair.repaired:
        flags.append("format_violation")

    scores_dict = repair.data.get("scores", {})
    overall_passed = repair.data.get("overall_passed")

    if not scores_dict:
        return GradeResult(
            passed=overall_passed, score=0.0, flags=flags + ["no_scores"]
        )

    weighted_sum = 0.0
    total_weight = 0.0
    for criterion in config.criteria:
        entry = scores_dict.get(criterion.name, {})
        raw_score = entry.get("score", 0) if isinstance(entry, dict) else 0
        normalized = max(0.0, min(1.0, (raw_score - 1) / 4.0))
        weighted_sum += normalized * criterion.weight
        total_weight += criterion.weight

    final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    passed = overall_passed if overall_passed is not None else final_score >= 0.6

    return GradeResult(passed=passed, score=final_score, flags=flags)


def load_judge_from_config_path(path: Path) -> JudgeConfig:
    """Convenience: load a judge config from a path."""
    return load_judge_config(path)
