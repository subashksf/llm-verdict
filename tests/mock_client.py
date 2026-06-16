"""Mock model client for testing — scripted responses, no network."""

from __future__ import annotations

from typing import Any

from llm_verdict.providers.base import CompletionResponse


class MockModelClient:
    """ModelClient with pre-scripted responses."""

    def __init__(
        self,
        responses: list[CompletionResponse] | None = None,
        default_response: CompletionResponse | None = None,
        model_id: str = "mock-model",
        pricing_input: float = 1.0,
        pricing_output: float = 3.0,
        max_concurrency: int = 4,
        default_max_tokens: int = 4096,
    ) -> None:
        self._model_id = model_id
        self._pricing_in = pricing_input
        self._pricing_out = pricing_output
        self._max_concurrency = max_concurrency
        self._default_max_tokens = default_max_tokens
        self._responses = list(responses or [])
        self._default = default_response
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
                "seed": seed,
            }
        )
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        elif self._default:
            resp = self._default
        else:
            raise RuntimeError(
                f"MockModelClient: no response for call {self._call_count}"
            )
        self._call_count += 1
        return resp
