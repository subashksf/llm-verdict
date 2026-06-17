"""LiteLLM adapter implementing the ModelClient protocol."""

from __future__ import annotations

import os
import time
from typing import Any

import litellm

from llm_verdict.providers.base import CompletionResponse


class LiteLLMClient:
    """Config-driven LiteLLM client."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._model_id: str = config["name"]
        self._litellm_model: str = config["litellm_model"]
        self._api_key_env: str = config["api_key_env"]
        self._pricing_in: float = config["pricing"]["input_per_mtok_usd"]
        self._pricing_out: float = config["pricing"]["output_per_mtok_usd"]
        self._max_concurrency: int = config.get("limits", {}).get("max_concurrency", 4)
        self._default_max_tokens: int = config.get("defaults", {}).get(
            "max_tokens", 4096
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def pricing_input_per_mtok(self) -> float:
        return self._pricing_in

    @property
    def pricing_output_per_mtok(self) -> float:
        return self._pricing_out

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def default_max_tokens(self) -> int:
        return self._default_max_tokens

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
    ) -> CompletionResponse:
        api_key = os.environ.get(self._api_key_env)
        start_ns = time.perf_counter_ns()

        kwargs: dict[str, Any] = {
            "model": self._litellm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key,
        }
        if tools:
            kwargs["tools"] = tools
        if seed is not None:
            kwargs["seed"] = seed

        response = await litellm.acompletion(**kwargs)
        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls_raw = choice.message.tool_calls
        parsed_tool_calls = _extract_tool_calls(tool_calls_raw)

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        return CompletionResponse(
            text=text,
            tool_calls=parsed_tool_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model_version=getattr(response, "model", None),
            latency_ms_total=elapsed_ms,
            latency_ms_first_token=None,
        )


def _extract_tool_calls(
    tool_calls_raw: Any,
) -> list[dict[str, Any]] | None:
    if not tool_calls_raw:
        return None
    result: list[dict[str, Any]] = []
    for tc in tool_calls_raw:
        func = tc.function
        result.append(
            {
                "id": tc.id,
                "name": func.name,
                "arguments": func.arguments,
            }
        )
    return result
