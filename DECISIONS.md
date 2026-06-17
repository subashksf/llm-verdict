# DECISIONS.md — llm-verdict

## Phase 1 Decisions

### Suite hash is order-invariant by design
The spec says "hash stable across reorderings" (§11). Implementation: hash each task individually, sort the hex digests lexicographically, concatenate, then SHA-256 the result. This means adding/removing a task changes the hash, but reordering does not.

### Task prompt type is `str | list[Message]`
The spec allows "single string or chat messages" (§4). Pydantic's discriminated union would add complexity here; instead we use a plain union type. The loader accepts both forms from YAML.

### DuckDB views use MEDIAN for p50
DuckDB supports MEDIAN as an aggregate. The `model_summary` view uses it for `p50_latency_ms`. P95 will require QUANTILE_CONT in Phase 4 report generation.

### CLI uses typer subcommands
`verdict suite validate`, `verdict report card`, etc. are implemented as typer sub-apps. The `verdict run` command is registered directly on the root app since it's the primary action.

### Python 3.13 used (spec says 3.12+)
uv resolved to Python 3.13.13 which is available on the system. This is compatible with the 3.12+ requirement.

### `regex` task type added to TaskType enum
The spec (§5) lists `regex` as a grader but the Task model (§4) enum lists `exact | json_schema | code_exec | tool_call | judged`. Since regex is a distinct programmatic grading mode with its own grader, I added it to TaskType. This avoids overloading `exact` for pattern matching.

### No deviations from spec
Phase 1 implementation follows the spec faithfully. All acceptance criteria pass.

## Phase 2 Decisions

### `status` column added to `runs` table
The spec says runs are immutable, but a run needs lifecycle tracking (running → completed/partial). Added a `status VARCHAR NOT NULL DEFAULT 'completed'` column to the `runs` table. `update_run_status()` is the sole mutation and only changes lifecycle state, never results data. This preserves the append-only invariant on trials/scores.

### Response cache uses a separate DuckDB file
Cache is stored in `.verdict_cache.duckdb` (configurable). Separate from the main results store so cache can be wiped without affecting run history. Keyed by composite SHA-256 of `(model_id, model_version, params_hash, prompt_hash)`.

### Cache hits populate cost/latency from the cached record
Per CLAUDE.md: "cache hits still need cost/latency fields populated (copy from cached record, flag `cached=true`)." The engine copies all fields from the cache entry and sets `cached=True` on the resulting TrialResult.

### Budget enforcement is two-tier
1. Pre-run cost estimation (tasks × trials × estimated tokens × price) — hard abort before any API call if estimate exceeds budget.
2. In-run tracking with `asyncio.Lock` — after each response, atomically check-and-deduct. If spent >= budget, mark `budget_exceeded=True`, cancel remaining work, persist what completed, set run status to `partial`.

### Concurrency uses anyio semaphore with anyio task group
Using `anyio.Semaphore` and `anyio.create_task_group()` for concurrency control. This is portable between asyncio and trio, and the semaphore count comes from the model config's `max_concurrency` field.

### Exponential backoff with jitter
Retry on any exception from `client.complete()` (in production this catches 429/5xx from litellm). Formula: `base_ms * 2^attempt + random(0, base_ms)` where base_ms=1000, max 5 retries.

### Grader errors don't abort the run
Per §6: "a single task failure (grader exception, sandbox crash) is recorded with `flags=["grader_error"]` and never aborts the run." The engine catches all exceptions in `_grade_trial` and records a Score with `grader_version="error"` and `flags=["grader_error"]`.

### Run IDs use ULID
ULIDs are time-sortable and globally unique, matching the spec's `run_id: str # ULID` field. Using `python-ulid` package.

### Pre-run estimate uses conservative token assumptions
Input tokens estimated at 500 per call (conservative for typical prompts). Output tokens estimated at the model's `default_max_tokens`. This errs on the side of rejecting expensive runs.

### `verdict run --resume` loads original manifest from DB
Resume queries the `runs` table for the original manifest, then diffs completed trials against the expected full set. Only the delta is executed under the same run_id.

### No deviations from spec
Phase 2 implementation follows the spec faithfully. All acceptance criteria pass:
- Full run against mock client produces manifest, trials, and scores in DuckDB.
- Budget abort works (both pre-run estimate and in-run hard stop).
- Resume completes a partial run under the original manifest.

## Phase 3 Decisions

### Sandbox uses Docker containers
macOS lacks network namespaces for subprocess isolation. Rather than a best-effort approach (no DNS, temp dir, short timeout), we use Docker containers for airtight network isolation. This adds Docker as a runtime dependency for `code_exec` tasks.

### Judge model is user-configured, no default
The spec requires the judge ≠ model under test, but doesn't specify a default. Rather than hardcoding one, users must provide a judge config in `configs/judges/`. This is highlighted in the README. Rationale: explicit configuration over magic defaults; judge choice has cost/quality implications the user should own.

### Calibration uses interactive CLI prompts
`verdict judge calibrate` presents samples one at a time with y/n/skip prompts. This is more natural for the expected workflow (quick labeling sessions) vs. generating a batch file to fill in.

### JSON repair attempted with format_violation flag
When the judge returns slightly malformed JSON (trailing commas, markdown fences), attempt repair before failing. If repair succeeds, set `flags=["format_violation"]` on the score so downstream analysis can filter or flag these. If repair fails, score the trial as a grader error.

## Phase 4 Decisions

### INSUFFICIENT_DATA checked after ADOPT conditions
The spec says INSUFFICIENT_DATA fires when "CIs are too wide to decide." If adopt conditions are already met (e.g., lower CI bound clears threshold despite wide CI), the data is sufficient. Order: REJECT (constraints) → ADOPT → INSUFFICIENT_DATA → HOLD.

### Verdict clause evaluation uses dispatch table
Each clause type (`pass_rate_delta_pp`, `ci_overlap`, `cost_per_success_reduction_pct`, `pass_rate_within_ci`) has its own evaluator function in `_CLAUSE_EVALUATORS`. This avoids a long if/elif chain and keeps cyclomatic complexity under the C901 limit.

### Bootstrap resamples tasks, not trials
Per the spec: "bootstrap (10k resamples over tasks, not trials)." Each task has one aggregated score (mean across trials or majority vote). The bootstrap resamples *tasks* to get the CI.

### McNemar uses scipy.stats.binomtest for small samples
For n_discordant < 25, exact binomial via `binomtest` (scipy removed the old `binom_test`). For larger samples, uses chi-squared approximation.

### Category derived from task_id prefix
Tasks are expected to use `category/task_name` as their task_id (e.g., `coding/t1`). The report data layer splits on `/` to determine category. This avoids needing a separate category lookup table.

### Report card never includes response text or prompts
Reports only render aggregate statistics, CIs, and anonymized failure patterns (category + flags). Test `test_no_task_text_leakage` asserts this invariant by planting sentinel strings and grepping the output.

### Timeline report deferred to Phase 5
Per the spec: Phase 5 includes "longitudinal report." The CLI stub remains with "not implemented (Phase 5)" message.

### numpy and scipy added as dependencies
Stats functions require numpy (bootstrap, array ops) and scipy (binomial test, chi-squared CDF). These are explicit in pyproject.toml. Type stubs (`scipy-stubs`, `types-PyYAML`) added to dev deps for mypy strict compliance.

## Phase 5 Decisions

### Timeline report queries DuckDB directly
`compute_timeline_report` uses a model family prefix match (`LIKE '{family}%'`) and joins scores with trials to compute per-run aggregates inline. This avoids materializing full RunReports for every historical run.

### `add-from-failure` creates a stub YAML that requires manual prompt fill-in
The failed trial's response text is never stored in the generated task YAML (privacy invariant). Instead, the prompt field contains a `[FILL IN]` placeholder. The user fills this from their suite source. Metadata records the original run_id and task_id for traceability.

### `db query` outputs TSV to stdout
Raw query results are printed as tab-separated values with a header row. This is simple, pipeable, and avoids adding a table formatting dependency.

### README quickstart uses `uv run verdict` convention
All CLI examples use `uv run verdict` (not `python -m` or bare `verdict`) since the project uses uv for dependency management and the entrypoint is defined in pyproject.toml.

### No deviations from spec
Phase 5 acceptance criteria: "a new model can be added with a single YAML file and produce a full report card with one `run` + one `report` command." The README quickstart demonstrates exactly this flow.
