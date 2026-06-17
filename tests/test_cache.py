"""Unit tests for the response cache."""

from __future__ import annotations

from pathlib import Path

from llm_verdict.runner.cache import (
    CachedResponse,
    CacheKey,
    ResponseCache,
    compute_params_hash,
    compute_prompt_hash,
)


def test_put_and_get(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.duckdb")
    key = CacheKey(
        model_id="test-model",
        model_version="v1",
        params_hash="abc123",
        prompt_hash="def456",
    )
    response = CachedResponse(
        text="Hello world",
        tool_calls_json=None,
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.001,
        latency_ms_total=200,
    )
    cache.put(key, response)
    result = cache.get(key)
    assert result is not None
    assert result.text == "Hello world"
    assert result.tokens_in == 10
    assert result.tokens_out == 5
    assert result.cost_usd == 0.001
    assert result.latency_ms_total == 200
    cache.close()


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.duckdb")
    key = CacheKey(
        model_id="test-model",
        model_version="v1",
        params_hash="nonexistent",
        prompt_hash="missing",
    )
    assert cache.get(key) is None
    cache.close()


def test_cache_key_determinism() -> None:
    key1 = CacheKey(
        model_id="model-a",
        model_version="v1",
        params_hash="p1",
        prompt_hash="h1",
    )
    key2 = CacheKey(
        model_id="model-a",
        model_version="v1",
        params_hash="p1",
        prompt_hash="h1",
    )
    assert key1.composite() == key2.composite()


def test_different_keys_produce_different_composites() -> None:
    key1 = CacheKey(
        model_id="model-a",
        model_version="v1",
        params_hash="p1",
        prompt_hash="h1",
    )
    key2 = CacheKey(
        model_id="model-b",
        model_version="v1",
        params_hash="p1",
        prompt_hash="h1",
    )
    assert key1.composite() != key2.composite()


def test_params_hash_determinism() -> None:
    h1 = compute_params_hash(0.0, 4096, None)
    h2 = compute_params_hash(0.0, 4096, None)
    assert h1 == h2


def test_prompt_hash_determinism() -> None:
    msgs = [{"role": "user", "content": "hello"}]
    h1 = compute_prompt_hash(msgs)
    h2 = compute_prompt_hash(msgs)
    assert h1 == h2


def test_cache_with_tool_calls(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.duckdb")
    key = CacheKey(model_id="m", model_version=None, params_hash="p", prompt_hash="h")
    response = CachedResponse(
        text="",
        tool_calls_json='[{"name": "calc", "arguments": "{}"}]',
        tokens_in=5,
        tokens_out=10,
        cost_usd=0.002,
        latency_ms_total=100,
    )
    cache.put(key, response)
    result = cache.get(key)
    assert result is not None
    assert result.tool_calls_json == '[{"name": "calc", "arguments": "{}"}]'
    cache.close()
