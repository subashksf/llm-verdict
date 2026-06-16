"""Response cache keyed by (model_id, model_version, params_hash, prompt_hash)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS response_cache (
    cache_key VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    text VARCHAR NOT NULL,
    tool_calls_json VARCHAR,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd DOUBLE NOT NULL,
    latency_ms_total INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass(frozen=True)
class CacheKey:
    """Components that uniquely identify a model request."""

    model_id: str
    model_version: str | None
    params_hash: str
    prompt_hash: str

    def composite(self) -> str:
        parts = (
            f"{self.model_id}|{self.model_version or ''}"
            f"|{self.params_hash}|{self.prompt_hash}"
        )
        return hashlib.sha256(parts.encode()).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    """Stored response data for cache hits."""

    text: str
    tool_calls_json: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms_total: int


def compute_params_hash(temperature: float, max_tokens: int, seed: int | None) -> str:
    data = json.dumps(
        {"temperature": temperature, "max_tokens": max_tokens, "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def compute_prompt_hash(messages: list[dict[str, str]]) -> str:
    data = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class ResponseCache:
    """DuckDB-backed response cache."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db_path))
        self._conn.execute(CACHE_SCHEMA)

    def get(self, key: CacheKey) -> CachedResponse | None:
        row = self._conn.execute(
            "SELECT text, tool_calls_json, tokens_in, tokens_out, "
            "cost_usd, latency_ms_total FROM response_cache WHERE cache_key = ?",
            [key.composite()],
        ).fetchone()
        if row is None:
            return None
        return CachedResponse(
            text=row[0],
            tool_calls_json=row[1],
            tokens_in=row[2],
            tokens_out=row[3],
            cost_usd=row[4],
            latency_ms_total=row[5],
        )

    def put(self, key: CacheKey, response: CachedResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO response_cache "
            "(cache_key, model_id, text, tool_calls_json, tokens_in, "
            "tokens_out, cost_usd, latency_ms_total) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                key.composite(),
                key.model_id,
                response.text,
                response.tool_calls_json,
                response.tokens_in,
                response.tokens_out,
                response.cost_usd,
                response.latency_ms_total,
            ],
        )

    def close(self) -> None:
        self._conn.close()
