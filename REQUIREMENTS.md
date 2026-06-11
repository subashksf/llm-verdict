# llm-verdict — Provider-Agnostic LLM Evaluation Framework

**Document type:** Requirements specification for implementation
**Status:** v1.0 — ready for build
**Owner:** Subash Keshapragada

---

## 1. Purpose & Vision

llm-verdict is a private, reproducible, provider-agnostic evaluation harness for frontier LLMs. When a new model ships (Anthropic, OpenAI, Google, open-weights via any OpenAI-compatible endpoint), llm-verdict answers one question quickly and credibly:

> **For my task suite, is this model better, worse, cheaper, or faster than the incumbent — with statistical confidence?**

Primary outputs: (1) a queryable results store, (2) a one-page Markdown/HTML "model report card" suitable for publishing, (3) longitudinal comparisons across model versions and dates.

### Design pillars (non-negotiable invariants)

1. **Reproducibility** — every run is fully reconstructable from logged artifacts: task suite hash, harness version, model ID + version string, config snapshot, seeds, judge version.
2. **Immutability** — completed runs are never mutated. Corrections happen via new runs or annotation records, never edits.
3. **Provider-agnosticism** — adding a model is a YAML config entry, never a code change.
4. **Contamination resistance** — the private task suite never leaves the local machine except in aggregate statistics. No task text in published reports.
5. **Statistical honesty** — every comparison carries confidence intervals; the framework refuses to declare a "winner" inside overlapping CIs.
6. **Cost safety** — hard per-run budget caps; the harness halts before exceeding them.

### Non-goals (v1)

- No web UI (CLI + static HTML reports only).
- No public benchmark replication (MMLU, HumanEval, etc.).
- No fine-tuning, training, or RLHF tooling.
- No distributed execution (single machine, async concurrency is sufficient).
- No real-time/streaming evaluation.

---

## 2. Tech Stack & Constraints

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | Owner's primary language |
| Package mgmt | `uv` with `pyproject.toml` | Fast, lockfile-based reproducibility |
| Provider abstraction | `litellm` | Uniform interface; OpenAI-compatible fallback for any endpoint |
| Results store | DuckDB (single file) + Parquet exports | Queryable, zero-ops, portable to Snowflake later |
| Raw artifacts | JSONL on disk, content-addressed directories | Immutable, diff-able, grep-able |
| Config | YAML + Pydantic v2 models for validation | Fail-fast on bad config |
| CLI | `typer` | Typed, self-documenting |
| Stats | `numpy` + `scipy` (bootstrap CIs, McNemar, Cohen's kappa) | No heavyweight deps |
| Reports | Jinja2 → Markdown and self-contained HTML | Publishable artifacts |
| Async | `asyncio` + `anyio` semaphores for concurrency limits | Throughput with rate-limit respect |
| Testing | `pytest` + `pytest-asyncio`; `vcrpy` or recorded fixtures for API mocking | No live API calls in unit tests |
| Lint/format | `ruff` (lint + format), `mypy --strict` on `core/` | Quality gate |

**Code style requirement:** low cognitive complexity. Max cyclomatic complexity 8 per function (enforce via ruff `C901`). Prefer small pure functions, explicit dataclasses/Pydantic models over dicts, and early returns over nesting.

**Naming & packaging:** repo name `llm-verdict`, Python package `llm_verdict`, CLI entry point `verdict` (declared in `pyproject.toml` under `[project.scripts]` as `verdict = "llm_verdict.cli:app"`). The README must include a one-line disambiguation: this project is unrelated to Haize Labs' `verdict` judge-composition library — llm-verdict is a full evaluation harness (runner, graders, statistics, and adoption verdicts), not a judge framework.

---

## 3. Repository Layout

```
llm-verdict/                  # repo root (package dir below is llm_verdict)
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── llm_verdict/
│   ├── core/                 # Pure domain logic — no I/O, no API calls
│   │   ├── models.py         # Pydantic: Task, Trial, RunManifest, Score, ...
│   │   ├── hashing.py        # Content-addressing for suites/configs
│   │   ├── stats.py          # Bootstrap CIs, McNemar, kappa, cost-per-success
│   │   └── verdict.py        # Pre-registered success-criteria engine
│   ├── providers/
│   │   ├── base.py           # ModelClient protocol
│   │   ├── litellm_client.py # Default adapter (covers ~all providers)
│   │   └── registry.py       # Config-driven model registration
│   ├── tasks/
│   │   ├── loader.py         # Load/validate/hash task suites
│   │   └── schemas.py        # Task-type schemas (see §5)
│   ├── graders/
│   │   ├── base.py           # Grader protocol
│   │   ├── programmatic/     # exact_match, json_schema, regex, code_exec, tool_call
│   │   ├── judge/            # LLM-as-judge w/ rubrics + calibration
│   │   └── registry.py       # Plugin registration by name
│   ├── runner/
│   │   ├── engine.py         # Orchestrates trials: retries, concurrency, budget
│   │   ├── sandbox.py        # Subprocess code execution (timeouts, no network)
│   │   └── cache.py          # Response cache keyed by (model, params, prompt hash)
│   ├── store/
│   │   ├── duck.py           # DuckDB schema + writers/readers
│   │   └── artifacts.py      # JSONL raw transcript persistence
│   ├── reporting/
│   │   ├── report_card.py    # Single-model report
│   │   ├── compare.py        # Head-to-head & longitudinal reports
│   │   └── templates/
│   └── cli.py
├── suites/                   # PRIVATE task suites (gitignored content, schema-validated)
│   ├── code_review/
│   ├── tool_use/
│   ├── structured_output/
│   └── _examples/            # Synthetic examples, safe to commit
├── configs/
│   ├── models/               # one YAML per model
│   ├── judges/               # pinned judge configs
│   └── criteria/             # pre-registered success criteria per category
├── runs/                     # Immutable run artifacts (gitignored)
├── reports/                  # Generated report cards
└── tests/
```

---

## 4. Core Data Model

All entities are Pydantic models, serialized to JSONL (artifacts) and DuckDB (analytics).

### Task
```
task_id: str            # stable, human-readable, e.g. "code_review/cr-0042"
category: str           # task category, drives criteria lookup
prompt: str | Message[] # single string or chat messages
task_type: enum         # exact | json_schema | code_exec | tool_call | judged
grader: GraderSpec      # grader name + params (e.g. schema, test code, rubric_id)
metadata: dict          # tags, difficulty, source ("from-failure:run-id" lineage)
version: int            # bumped when task content changes
```

### RunManifest (written before any API call; the reproducibility anchor)
```
run_id: str             # ULID
created_at: datetime
model: ModelConfig      # full resolved config snapshot
suite_hash: str         # SHA-256 over canonical task suite content
harness_version: str    # package version + git SHA
judge_config_hash: str | None
trials_per_task: int    # default 3
temperature: float      # default 0.0
budget_usd: float       # hard cap
seed: int | None
```

### TrialResult
```
run_id, task_id, trial_index
request_hash: str
response_text: str
tool_calls: list | None
tokens_in, tokens_out: int
cost_usd: float         # from provider pricing table in model config
latency_ms_first_token: int | None
latency_ms_total: int
error: str | None       # refusals, timeouts, API errors — categorized
```

### Score
```
run_id, task_id, trial_index
grader_name, grader_version
passed: bool | None     # for binary graders
score: float            # normalized 0–1
rubric_scores: dict | None   # per-criterion for judged tasks
judge_reasoning: str | None
flags: list[str]        # e.g. ["format_violation", "refusal", "judge_low_confidence"]
```

### DuckDB schema
Tables: `runs`, `trials`, `scores`, `verdicts`, `annotations` (append-only human labels for judge calibration). Provide a `views.sql` with convenience views: `model_summary`, `category_summary`, `head_to_head`, `longitudinal`.

---

## 5. Task Types & Graders

### Programmatic graders (tier 1 — prefer always)

| Grader | Behavior |
|---|---|
| `exact_match` | Normalized string equality (configurable: case, whitespace, number tolerance) |
| `regex` | Pattern presence/absence assertions |
| `json_schema` | Response must parse as JSON and validate against provided schema; partial credit option for field-level scoring |
| `code_exec` | Extract code block → run in sandbox (subprocess, 30s timeout, no network, temp dir) → run task-provided pytest assertions. Score = tests passed / total |
| `tool_call` | Validate model emitted the expected tool name with arguments matching an expected-args schema (exact, subset, or semantic-key match modes) |

### LLM-as-judge (tier 2)

- Judge is a **pinned model + pinned prompt + pinned rubric**, hashed together as `judge_config_hash`. Changing any component creates a new judge version; the framework must **refuse to compare scores across judge versions** unless `--allow-judge-mismatch` is passed.
- Rubrics: 3–5 criteria, each scored 1–5 with required written justification, output as structured JSON (use provider structured-output mode where available; otherwise strict JSON prompt + repair-then-fail parsing).
- The judge must never be the model under test (hard validation error).
- Position-bias control for pairwise comparisons: evaluate both orderings, flag disagreements.
- **Calibration command:** `verdict judge calibrate` — sample N graded items, present to human via CLI for labeling, store in `annotations`, report Cohen's kappa and per-criterion agreement. Report cards must display the judge's most recent kappa; if below 0.6 print a prominent warning.

### Refusal / non-answer detection
Lightweight classifier (regex + judge fallback) tagging trials as `refusal`, `partial`, or `answered`. Refusal rate is a first-class metric.

---

## 6. Runner Requirements

- **Concurrency:** async with per-provider semaphore (configurable, default 4). Exponential backoff with jitter on 429/5xx, max 5 retries; retries logged, not silent.
- **Budget enforcement:** estimate cost before run (tasks × trials × est. tokens × price); abort if estimate exceeds budget. During run, track actual spend; hard-stop at cap, mark run `partial`, persist everything completed.
- **Caching:** response cache keyed by `(model_id, model_version, params_hash, prompt_hash)`. Cache hits marked in trial records. `--no-cache` flag for fresh runs. Cache makes grader development free.
- **Determinism:** temperature 0 by default; still run `trials_per_task ≥ 3` (default 3, configurable) because providers are nondeterministic at temp 0.
- **Resumability:** `verdict run --resume <run_id>` completes missing trials of a partial run under the original manifest.
- **Graceful degradation:** a single task failure (grader exception, sandbox crash) is recorded with `flags=["grader_error"]` and never aborts the run.

---

## 7. Statistics & Verdict Engine

Implemented in `core/stats.py` and `core/verdict.py`, fully unit-tested against known values.

1. **Pass-rate CIs:** Wilson score interval per category; bootstrap (10k resamples over tasks, not trials) for mean rubric scores.
2. **Paired comparison:** when two models ran the same suite hash, use McNemar's test on per-task pass/fail (majority vote across trials) and paired bootstrap for score deltas.
3. **Cost-per-success:** `total_cost_usd / tasks_passed`, with bootstrap CI. Defined per category and overall.
4. **Consistency:** per-task trial variance; report % of tasks with unanimous trials.
5. **Latency:** p50/p95 total and first-token, per category.
6. **Verdict engine:** evaluates pre-registered criteria YAML, e.g.:

```yaml
# configs/criteria/code_review.yaml
category: code_review
incumbent: claude-sonnet-4-6
adopt_if:
  - pass_rate_delta_pp: ">= 5"
    ci_overlap: false
  - OR:
      pass_rate_within_ci: true
      cost_per_success_reduction_pct: ">= 30"
constraints:
  p95_latency_ms: "<= 20000"
  refusal_rate_pct: "<= 2"
```

Verdict output is one of: `ADOPT`, `HOLD`, `REJECT`, `INSUFFICIENT_DATA` — with the specific clauses that fired. `INSUFFICIENT_DATA` when CIs are too wide to decide (define: CI width > 2× the decision threshold).

---

## 8. CLI Surface

```
verdict suite validate <path>          # schema-check + hash a task suite
verdict suite hash <path>
verdict run --model <name> --suite <path> [--trials 3] [--budget 10.00] [--dry-run]
verdict run --resume <run_id>
verdict grade <run_id> [--regrade --grader <name>]   # regrade = new score records, old kept
verdict judge calibrate --run <run_id> --sample 30
verdict report card <run_id>                          # single-model report card
verdict report compare <run_id_a> <run_id_b>          # head-to-head w/ McNemar + verdict
verdict report timeline --model-family <prefix>       # longitudinal across versions
verdict db query "<sql>"                              # escape hatch into DuckDB
verdict task add-from-failure <run_id> <task_id>      # promote a failure into the suite
```

`--dry-run` prints task count, cost estimate, and manifest without API calls.

---

## 9. Report Card Requirements

Markdown + self-contained HTML. Must contain:

- Header: model, version string, run date, suite hash (short), harness version, judge version + last kappa, total cost, wall-clock time.
- Per-category table: pass rate ± CI, mean score ± CI, cost-per-success, p50/p95 latency, refusal rate, consistency.
- Head-to-head section (if incumbent specified): deltas with CIs, McNemar p-value, verdict with fired clauses.
- "Notable failures" section: up to 5 anonymized failure *patterns* (category + flag summary — **never raw task text**).
- Footer: methodology disclosure block (trials, temperature, judge model family, category mix percentages) — designed to be safe to publish.

---

## 10. Testing & Quality Requirements

- Unit tests for all of `core/` (stats functions tested against hand-computed values), graders (fixture-driven), hashing (canonicalization edge cases: key order, whitespace, unicode).
- Integration test: full run against a `MockModelClient` with scripted responses → grade → report, asserting end-to-end artifact integrity.
- Sandbox tests: timeout enforcement, network blocked, filesystem isolation.
- No test may make a live API call. CI target: `ruff check && mypy llm_verdict/core && pytest` green.
- Property tests (hypothesis) for: cache key stability, suite hash invariance under task reordering.

---

## 11. Build Phases & Acceptance Criteria

**Phase 1 — Skeleton & data layer.** Repo scaffold, Pydantic models, suite loader + hashing, DuckDB schema, CLI stubs. *Accept:* `verdict suite validate suites/_examples` passes; hash stable across reorderings.

**Phase 2 — Runner + programmatic graders.** LiteLLM client, engine with budget/retry/cache, `exact_match`, `json_schema`, `regex`, `tool_call` graders. *Accept:* full run against mock client produces manifest, trials, scores in DuckDB; budget abort works; resume works.

**Phase 3 — Code execution + judge.** Sandbox, `code_exec` grader, judge with rubric + structured output + calibration command. *Accept:* judge refuses self-grading; kappa computation matches hand-checked fixture; sandbox kills infinite loop at timeout.

**Phase 4 — Stats, verdicts, reports.** Bootstrap/Wilson/McNemar, verdict engine, all three report types. *Accept:* stats unit tests pass against known values; verdict engine returns `INSUFFICIENT_DATA` on a contrived wide-CI fixture; report card renders with no task text leakage (grep check in test).

**Phase 5 — Polish.** `add-from-failure` workflow, longitudinal report, README with a 10-minute quickstart using `_examples`. *Accept:* a new model can be added with a single YAML file and produce a full report card with one `run` + one `report` command.

---

## 12. Explicit "Never Do" List

- Never mutate or delete records in `runs/`, `trials`, `scores` — append-only everywhere (regrades append new score rows with new `grader_version`).
- Never include private task prompts or completions in reports, logs at INFO level, or exceptions.
- Never compare scores across different `judge_config_hash` values without an explicit override flag.
- Never let the judge model equal the model under test.
- Never make a network call from the code-exec sandbox.
- Never exceed `budget_usd`.
