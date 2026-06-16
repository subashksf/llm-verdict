"""Run engine — orchestrates trials with concurrency, budget, retries, cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anyio

from llm_verdict.core.models import (
    ModelConfig,
    RunManifest,
    Score,
    Task,
    TaskSuite,
    TrialResult,
)
from llm_verdict.graders.registry import get_grader
from llm_verdict.providers.base import CompletionResponse, ModelClient
from llm_verdict.runner.cache import (
    CachedResponse,
    CacheKey,
    ResponseCache,
    compute_params_hash,
    compute_prompt_hash,
)
from llm_verdict.store.duck import (
    get_completed_trials,
    get_run,
    get_run_total_cost,
    insert_run,
    insert_score,
    insert_trial,
    update_run_status,
)

logger = logging.getLogger(__name__)

_HARNESS_VERSION = "0.2.0"
_BASE_RETRY_MS = 1000
_MAX_RETRIES = 5


class BudgetExceededError(Exception):
    """Raised when budget cap is reached mid-run."""


@dataclass
class RunConfig:
    """Configuration for a single run execution."""

    model_name: str
    suite: TaskSuite
    trials_per_task: int = 3
    budget_usd: float = 10.0
    temperature: float = 0.0
    seed: int | None = None
    no_cache: bool = False
    resume_run_id: str | None = None


@dataclass
class _RunState:
    """Mutable state tracked during a run."""

    spent_usd: float = 0.0
    budget_usd: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    budget_exceeded: bool = False


class Engine:
    """Async run engine with budget enforcement, caching, and retries."""

    def __init__(
        self,
        client: ModelClient,
        db_conn: Any,
        cache: ResponseCache | None = None,
    ) -> None:
        self._client = client
        self._db = db_conn
        self._cache = cache

    async def execute_run(self, config: RunConfig) -> RunManifest:
        """Execute a full evaluation run."""
        manifest = self._resolve_manifest(config)
        work_items = self._determine_work(manifest, config)

        estimate = self._estimate_cost(config.suite.tasks, config.trials_per_task)
        if estimate > config.budget_usd:
            raise BudgetExceededError(
                f"Estimated cost ${estimate:.4f} exceeds "
                f"budget ${config.budget_usd:.2f}"
            )

        if config.resume_run_id is None:
            insert_run(self._db, manifest, status="running")

        state = _RunState(
            spent_usd=get_run_total_cost(self._db, manifest.run_id),
            budget_usd=config.budget_usd,
        )

        await self._run_trials(work_items, manifest, config, state)

        final_status = "partial" if state.budget_exceeded else "completed"
        update_run_status(self._db, manifest.run_id, final_status)
        return manifest

    def _resolve_manifest(self, config: RunConfig) -> RunManifest:
        if config.resume_run_id:
            return self._load_manifest(config.resume_run_id)
        return self._create_manifest(config)

    def _create_manifest(self, config: RunConfig) -> RunManifest:
        from ulid import ULID

        run_id = str(ULID())
        return RunManifest(
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            model=ModelConfig(
                model_id=self._client.model_id,
                provider="litellm",
                params={"max_tokens": self._client.default_max_tokens},
            ),
            suite_hash=config.suite.suite_hash,
            harness_version=_HARNESS_VERSION,
            trials_per_task=config.trials_per_task,
            temperature=config.temperature,
            budget_usd=config.budget_usd,
            seed=config.seed,
        )

    def _load_manifest(self, run_id: str) -> RunManifest:
        row = get_run(self._db, run_id)
        if row is None:
            raise ValueError(f"Run not found: {run_id}")
        params = row["model_params"]
        if isinstance(params, str):
            params = json.loads(params)
        return RunManifest(
            run_id=row["run_id"],
            created_at=row["created_at"],
            model=ModelConfig(
                model_id=row["model_id"],
                provider=row["model_provider"],
                version=row["model_version"],
                params=params or {},
            ),
            suite_hash=row["suite_hash"],
            harness_version=row["harness_version"],
            judge_config_hash=row["judge_config_hash"],
            trials_per_task=row["trials_per_task"],
            temperature=row["temperature"],
            budget_usd=row["budget_usd"],
            seed=row["seed"],
        )

    def _determine_work(
        self, manifest: RunManifest, config: RunConfig
    ) -> list[tuple[Task, int]]:
        all_items = [
            (task, idx)
            for task in config.suite.tasks
            for idx in range(manifest.trials_per_task)
        ]
        if config.resume_run_id is None:
            return all_items
        done = get_completed_trials(self._db, manifest.run_id)
        return [(t, i) for t, i in all_items if (t.task_id, i) not in done]

    def _estimate_cost(self, tasks: list[Task], trials_per_task: int) -> float:
        total_calls = len(tasks) * trials_per_task
        est_in = 500
        est_out = self._client.default_max_tokens
        price_in = self._client.pricing_input_per_mtok
        price_out = self._client.pricing_output_per_mtok
        input_cost = total_calls * est_in * price_in / 1_000_000
        output_cost = total_calls * est_out * price_out / 1_000_000
        return input_cost + output_cost

    async def _run_trials(
        self,
        work_items: list[tuple[Task, int]],
        manifest: RunManifest,
        config: RunConfig,
        state: _RunState,
    ) -> None:
        semaphore = anyio.Semaphore(self._client.max_concurrency)

        async def bounded_trial(task: Task, trial_idx: int) -> None:
            async with semaphore:
                if state.budget_exceeded:
                    return
                await self._execute_and_grade(task, trial_idx, manifest, config, state)

        async with anyio.create_task_group() as tg:
            for task, trial_idx in work_items:
                tg.start_soon(bounded_trial, task, trial_idx)

    async def _execute_and_grade(
        self,
        task: Task,
        trial_idx: int,
        manifest: RunManifest,
        config: RunConfig,
        state: _RunState,
    ) -> None:
        try:
            trial = await self._execute_single_trial(
                task, trial_idx, manifest, config, state
            )
        except BudgetExceededError:
            state.budget_exceeded = True
            return

        insert_trial(self._db, trial)
        score = self._grade_trial(task, trial, manifest)
        insert_score(self._db, score)

    async def _execute_single_trial(
        self,
        task: Task,
        trial_idx: int,
        manifest: RunManifest,
        config: RunConfig,
        state: _RunState,
    ) -> TrialResult:
        messages = self._build_messages(task)
        request_hash = self._compute_request_hash(messages, manifest)

        cached = self._try_cache(messages, manifest, config)
        if cached is not None:
            return self._trial_from_cache(
                cached, task, trial_idx, manifest, request_hash
            )

        response = await self._call_with_retry(messages, manifest)
        cost = self._compute_cost(response.tokens_in, response.tokens_out)

        async with state.lock:
            if state.spent_usd + cost > state.budget_usd:
                raise BudgetExceededError("Budget cap reached")
            state.spent_usd += cost

        self._store_in_cache(messages, manifest, config, response, cost)

        return TrialResult(
            run_id=manifest.run_id,
            task_id=task.task_id,
            trial_index=trial_idx,
            request_hash=request_hash,
            response_text=response.text,
            tool_calls=response.tool_calls,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=cost,
            latency_ms_first_token=response.latency_ms_first_token,
            latency_ms_total=response.latency_ms_total,
            error=None,
            cached=False,
        )

    def _try_cache(
        self,
        messages: list[dict[str, str]],
        manifest: RunManifest,
        config: RunConfig,
    ) -> CachedResponse | None:
        if config.no_cache or self._cache is None:
            return None
        key = self._build_cache_key(messages, manifest)
        return self._cache.get(key)

    def _store_in_cache(
        self,
        messages: list[dict[str, str]],
        manifest: RunManifest,
        config: RunConfig,
        response: CompletionResponse,
        cost: float,
    ) -> None:
        if config.no_cache or self._cache is None:
            return
        key = self._build_cache_key(messages, manifest)
        tc = response.tool_calls
        tool_calls_json = json.dumps(tc) if tc else None
        cached = CachedResponse(
            text=response.text,
            tool_calls_json=tool_calls_json,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=cost,
            latency_ms_total=response.latency_ms_total,
        )
        self._cache.put(key, cached)

    def _trial_from_cache(
        self,
        cached: CachedResponse,
        task: Task,
        trial_idx: int,
        manifest: RunManifest,
        request_hash: str,
    ) -> TrialResult:
        tool_calls = (
            json.loads(cached.tool_calls_json) if cached.tool_calls_json else None
        )
        return TrialResult(
            run_id=manifest.run_id,
            task_id=task.task_id,
            trial_index=trial_idx,
            request_hash=request_hash,
            response_text=cached.text,
            tool_calls=tool_calls,
            tokens_in=cached.tokens_in,
            tokens_out=cached.tokens_out,
            cost_usd=cached.cost_usd,
            latency_ms_first_token=None,
            latency_ms_total=cached.latency_ms_total,
            error=None,
            cached=True,
        )

    async def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        manifest: RunManifest,
    ) -> CompletionResponse:
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._client.complete(
                    messages,
                    temperature=manifest.temperature,
                    max_tokens=manifest.model.params.get("max_tokens", 4096),
                    seed=manifest.seed,
                )
            except Exception as e:
                last_error = e
                delay_ms = _BASE_RETRY_MS * (2**attempt) + random.randint(
                    0, _BASE_RETRY_MS
                )
                logger.warning(
                    "Retry %d/%d after %dms: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay_ms,
                    str(e)[:100],
                )
                await asyncio.sleep(delay_ms / 1000.0)
        raise last_error  # type: ignore[misc]

    def _grade_trial(
        self, task: Task, trial: TrialResult, manifest: RunManifest
    ) -> Score:
        try:
            grader = get_grader(task.grader.name)
            result = grader.grade(
                trial.response_text, trial.tool_calls, task.grader.params
            )
            return Score(
                run_id=manifest.run_id,
                task_id=task.task_id,
                trial_index=trial.trial_index,
                grader_name=grader.name,
                grader_version=grader.version,
                passed=result.passed,
                score=result.score,
                flags=result.flags,
            )
        except Exception as e:
            logger.error("Grader error for %s: %s", task.task_id, e)
            return Score(
                run_id=manifest.run_id,
                task_id=task.task_id,
                trial_index=trial.trial_index,
                grader_name=task.grader.name,
                grader_version="error",
                passed=None,
                score=0.0,
                flags=["grader_error"],
            )

    def _build_messages(self, task: Task) -> list[dict[str, str]]:
        if isinstance(task.prompt, str):
            return [{"role": "user", "content": task.prompt}]
        return [{"role": m.role, "content": m.content} for m in task.prompt]

    def _compute_request_hash(
        self, messages: list[dict[str, str]], manifest: RunManifest
    ) -> str:
        data = json.dumps(
            {
                "messages": messages,
                "model": manifest.model.model_id,
                "temperature": manifest.temperature,
                "seed": manifest.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _build_cache_key(
        self, messages: list[dict[str, str]], manifest: RunManifest
    ) -> CacheKey:
        return CacheKey(
            model_id=self._client.model_id,
            model_version=manifest.model.version,
            params_hash=compute_params_hash(
                manifest.temperature,
                manifest.model.params.get("max_tokens", 4096),
                manifest.seed,
            ),
            prompt_hash=compute_prompt_hash(messages),
        )

    def _compute_cost(self, tokens_in: int, tokens_out: int) -> float:
        input_cost = tokens_in * self._client.pricing_input_per_mtok / 1_000_000
        output_cost = tokens_out * self._client.pricing_output_per_mtok / 1_000_000
        return input_cost + output_cost
