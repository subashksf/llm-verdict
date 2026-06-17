"""Tests for the LLM judge grader and supporting modules."""

from __future__ import annotations

import json

import pytest

from llm_verdict.graders.judge.calibration import compute_cohens_kappa
from llm_verdict.graders.judge.config import (
    JudgeConfig,
    JudgeCriterion,
    build_judge_prompt,
)
from llm_verdict.graders.judge.grader import (
    LLMJudgeGrader,
    _parse_judge_response,
)
from llm_verdict.graders.judge.json_repair import attempt_json_repair

# --- JSON Repair Tests ---


class TestJsonRepair:
    def test_valid_json_no_repair(self):
        data = '{"scores": {"accuracy": {"score": 4}}, "overall_passed": true}'
        result = attempt_json_repair(data)
        assert result.data is not None
        assert result.repaired is False

    def test_markdown_fences(self):
        inner = '{"scores": {"accuracy": {"score": 4}}, "overall_passed": true}'
        data = f"```json\n{inner}\n```"
        result = attempt_json_repair(data)
        assert result.data is not None
        assert result.repaired is False

    def test_trailing_comma(self):
        data = '{"scores": {"accuracy": {"score": 4},}, "overall_passed": true}'
        result = attempt_json_repair(data)
        assert result.data is not None
        assert result.repaired is True

    def test_json_embedded_in_text(self):
        inner = '{"scores": {"accuracy": {"score": 4}}, "overall_passed": true}'
        data = f"Here is my evaluation:\n{inner}\nDone."
        result = attempt_json_repair(data)
        assert result.data is not None
        assert result.repaired is True

    def test_completely_invalid(self):
        data = "I cannot provide a JSON evaluation."
        result = attempt_json_repair(data)
        assert result.data is None

    def test_trailing_comma_in_array(self):
        data = '{"items": [1, 2, 3,]}'
        result = attempt_json_repair(data)
        assert result.data == {"items": [1, 2, 3]}
        assert result.repaired is True


# --- Judge Config Tests ---


class TestJudgeConfig:
    def test_config_hash_deterministic(self):
        config = JudgeConfig(
            judge_model="test-model",
            criteria=[
                JudgeCriterion(
                    name="accuracy",
                    description="Is it correct",
                    weight=1.0,
                )
            ],
        )
        h1 = config.config_hash
        h2 = config.config_hash
        assert h1 == h2

    def test_config_hash_changes_with_model(self):
        config1 = JudgeConfig(judge_model="model-a", criteria=[])
        config2 = JudgeConfig(judge_model="model-b", criteria=[])
        assert config1.config_hash != config2.config_hash

    def test_config_hash_changes_with_criteria(self):
        base = JudgeConfig(judge_model="model-a", criteria=[])
        with_criteria = JudgeConfig(
            judge_model="model-a",
            criteria=[JudgeCriterion(name="x", description="y", weight=1.0)],
        )
        assert base.config_hash != with_criteria.config_hash

    def test_build_prompt_includes_criteria(self):
        config = JudgeConfig(
            judge_model="test",
            criteria=[
                JudgeCriterion(
                    name="accuracy",
                    description="Is correct",
                    weight=1.0,
                )
            ],
        )
        prompt = build_judge_prompt(config, "What is 2+2?", "4")
        assert "accuracy" in prompt
        assert "Is correct" in prompt
        assert "What is 2+2?" in prompt
        assert "4" in prompt


# --- Parse Judge Response Tests ---


class TestParseJudgeResponse:
    def _config(self) -> JudgeConfig:
        return JudgeConfig(
            judge_model="test",
            criteria=[
                JudgeCriterion(name="accuracy", description="correct", weight=0.6),
                JudgeCriterion(name="clarity", description="clear", weight=0.4),
            ],
        )

    def test_perfect_scores(self):
        response = json.dumps(
            {
                "scores": {
                    "accuracy": {"score": 5, "justification": "perfect"},
                    "clarity": {"score": 5, "justification": "clear"},
                },
                "overall_passed": True,
            }
        )
        result = _parse_judge_response(response, self._config())
        assert result.passed is True
        assert result.score == 1.0
        assert result.flags == []

    def test_mixed_scores(self):
        response = json.dumps(
            {
                "scores": {
                    "accuracy": {"score": 3, "justification": "ok"},
                    "clarity": {"score": 5, "justification": "good"},
                },
                "overall_passed": True,
            }
        )
        result = _parse_judge_response(response, self._config())
        # accuracy: (3-1)/4 = 0.5, weight 0.6 -> 0.3
        # clarity: (5-1)/4 = 1.0, weight 0.4 -> 0.4
        # total = 0.7
        assert result.passed is True
        assert abs(result.score - 0.7) < 0.001

    def test_minimum_scores(self):
        response = json.dumps(
            {
                "scores": {
                    "accuracy": {"score": 1, "justification": "wrong"},
                    "clarity": {"score": 1, "justification": "unclear"},
                },
                "overall_passed": False,
            }
        )
        result = _parse_judge_response(response, self._config())
        assert result.passed is False
        assert result.score == 0.0

    def test_unparseable_response(self):
        result = _parse_judge_response("no json here", self._config())
        assert result.passed is None
        assert "judge_parse_error" in result.flags

    def test_repaired_json_flagged(self):
        response = (
            '{"scores": {"accuracy": {"score": 4,},'
            ' "clarity": {"score": 4,},},'
            ' "overall_passed": true}'
        )
        result = _parse_judge_response(response, self._config())
        assert result.passed is True
        assert "format_violation" in result.flags


# --- Self-Grading Prevention ---


class TestSelfGradingPrevention:
    def test_judge_refuses_self_grading(self):
        grader = LLMJudgeGrader()

        class FakeClient:
            model_id = "same-model"
            pricing_input_per_mtok = 1.0
            pricing_output_per_mtok = 3.0
            max_concurrency = 4
            default_max_tokens = 4096

        grader.configure(
            judge_client=FakeClient(),
            judge_config=JudgeConfig(judge_model="same-model", criteria=[]),
            model_under_test="same-model",
        )
        with pytest.raises(ValueError, match="cannot be the model under test"):
            grader.grade("response", None, {"task_prompt": "test"})


# --- Cohen's Kappa Tests ---


class TestCohensKappa:
    def test_perfect_agreement(self):
        judge = [True, True, False, False, True]
        human = [True, True, False, False, True]
        assert compute_cohens_kappa(judge, human) == 1.0

    def test_no_agreement_beyond_chance(self):
        judge = [True, True, False, False]
        human = [True, False, True, False]
        kappa = compute_cohens_kappa(judge, human)
        assert abs(kappa - 0.0) < 0.001

    def test_hand_computed_kappa(self):
        # Hand computation:
        # judge: [T, T, T, F, F]  human: [T, T, F, F, F]
        # agree: 4/5 = 0.8
        # p(judge_pos) = 3/5, p(human_pos) = 2/5
        # p_e = (3/5)*(2/5) + (2/5)*(3/5) = 12/25 = 0.48
        # kappa = (0.8 - 0.48) / (1 - 0.48) = 0.32/0.52 = 0.6154
        judge = [True, True, True, False, False]
        human = [True, True, False, False, False]
        kappa = compute_cohens_kappa(judge, human)
        assert abs(kappa - 0.6154) < 0.001

    def test_empty_labels(self):
        assert compute_cohens_kappa([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_cohens_kappa([True], [True, False])
