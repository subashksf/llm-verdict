"""Judge configuration — loading, validation, and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class JudgeCriterion(BaseModel):
    """A single rubric criterion."""

    name: str
    description: str
    weight: float = 1.0


class JudgeConfig(BaseModel):
    """Full judge configuration — model + prompt + rubric."""

    judge_model: str
    prompt_template: str = Field(default="")
    criteria: list[JudgeCriterion] = Field(default_factory=list)
    max_tokens: int = 2048
    temperature: float = 0.0

    @property
    def config_hash(self) -> str:
        data = json.dumps(
            {
                "judge_model": self.judge_model,
                "prompt_template": self.prompt_template,
                "criteria": [c.model_dump() for c in self.criteria],
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]


def load_judge_config(path: Path) -> JudgeConfig:
    """Load and validate a judge config from YAML."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return JudgeConfig(**raw)


_DEFAULT_PROMPT = """\
You are an expert evaluator. Grade the following response on each criterion.

## Task Prompt
{task_prompt}

## Response to Evaluate
{response}

## Criteria
{criteria_block}

## Instructions
For each criterion, provide a score from 1 to 5 and a brief justification.
Return your evaluation as a JSON object with this exact structure:
{{
  "scores": {{
    "<criterion_name>": {{"score": <1-5>, "justification": "<brief reason>"}}
  }},
  "overall_passed": <true/false>
}}
"""


def build_judge_prompt(
    config: JudgeConfig,
    task_prompt: str,
    response: str,
) -> str:
    """Build the judge prompt from config, task, and response."""
    template = config.prompt_template or _DEFAULT_PROMPT
    criteria_block = "\n".join(
        f"- **{c.name}**: {c.description} (weight: {c.weight})" for c in config.criteria
    )
    return template.format(
        task_prompt=task_prompt,
        response=response,
        criteria_block=criteria_block,
    )
