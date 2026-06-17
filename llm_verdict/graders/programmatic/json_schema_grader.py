"""json_schema grader — JSON parse + schema validation with partial credit."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from llm_verdict.graders.base import GradeResult


class JsonSchemaGrader:
    """Validate response parses as JSON and conforms to a schema."""

    name = "json_schema"
    version = "1.0.0"

    def grade(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> GradeResult:
        schema: dict[str, Any] = params["schema"]
        partial_credit: bool = params.get("partial_credit", False)

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return GradeResult(passed=False, score=0.0, flags=["format_violation"])

        if not partial_credit:
            return self._binary_validate(data, schema)
        return self._partial_validate(data, schema)

    def _binary_validate(self, data: Any, schema: dict[str, Any]) -> GradeResult:
        try:
            jsonschema.validate(data, schema)
            return GradeResult(passed=True, score=1.0)
        except jsonschema.ValidationError:
            return GradeResult(passed=False, score=0.0)

    def _partial_validate(self, data: Any, schema: dict[str, Any]) -> GradeResult:
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        if not required:
            return self._binary_validate(data, schema)

        if not isinstance(data, dict):
            return GradeResult(passed=False, score=0.0)

        fields_valid = 0
        for field_name in required:
            if field_name not in data:
                continue
            field_schema = properties.get(field_name)
            if not field_schema:
                fields_valid += 1
                continue
            try:
                jsonschema.validate(data[field_name], field_schema)
                fields_valid += 1
            except jsonschema.ValidationError:
                pass

        score = fields_valid / len(required)
        passed = fields_valid == len(required)
        return GradeResult(passed=passed, score=score)
