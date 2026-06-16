"""ModelClient protocol — the provider-agnostic interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CompletionResponse:
    """Normalized response from any model provider."""

    text: str
    tool_calls: list[dict[str, Any]] | None
    tokens_in: int
    tokens_out: int
    model_version: str | None
    latency_ms_total: int
    latency_ms_first_token: int | None = None
    finish_reason: str = "stop"


class ModelClient(Protocol):
    """Protocol for model clients."""

    @property
    def model_id(self) -> str: ...

    @property
    def pricing_input_per_mtok(self) -> float: ...

    @property
    def pricing_output_per_mtok(self) -> float: ...

    @property
    def max_concurrency(self) -> int: ...

    @property
    def default_max_tokens(self) -> int: ...

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
    ) -> CompletionResponse: ...
