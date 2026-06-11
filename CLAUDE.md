# CLAUDE.md — llm-verdict

## What this project is

llm-verdict is a private, provider-agnostic LLM evaluation harness. It runs a versioned private task suite against any model (via LiteLLM), grades results with programmatic checks and a calibrated LLM judge, applies pre-registered statistical success criteria, and emits publishable model report cards. The authoritative spec is `REQUIREMENTS.md` — read it before structural changes.

## Commands

```bash
uv sync                          # install deps
uv run pytest                    # full test suite (must pass before any commit)
uv run pytest tests/core -x      # fast loop while editing core/
uv run ruff check --fix . && uv run ruff format .
uv run mypy llm_verdict/core         # strict typing on core only
uv run verdict --help            # CLI entrypoint
uv run verdict run --model <m> --suite suites/_examples --dry-run   # safe smoke test
```

## Architecture invariants (do not violate)

1. **`core/` is pure.** No I/O, no network, no DuckDB, no env vars inside `llm_verdict/core/`. It takes data in, returns data out. All side effects live in `providers/`, `store/`, `runner/`.
2. **Append-only data.** Never write code that mutates or deletes rows in `runs`, `trials`, or `scores`, or edits files under `runs/`. Regrades append new `Score` rows with a new `grader_version`.
3. **Manifest before API calls.** `RunManifest` is fully written to disk before the first model request. If you add a config knob, it must be captured in the manifest.
4. **Judge pinning.** Any change to judge model, prompt template, or rubric must change `judge_config_hash`. Cross-judge-version comparisons must raise unless explicitly overridden.
5. **Privacy.** Task prompts and model completions never appear in reports, INFO-level logs, exception messages, or test snapshots committed to git. Suites under `suites/` (except `_examples/`) are gitignored — keep it that way.
6. **Budget cap is sacred.** Any new execution path that calls a provider must flow through the engine's budget accounting.
7. **Sandbox isolation.** Code-exec runs in a subprocess with timeout, temp-dir cwd, and no network. Never relax this for convenience.

## Code style

- Python 3.12+, full type hints everywhere; `mypy --strict` clean in `core/`.
- **Low cognitive complexity is a hard requirement**: max cyclomatic complexity 8 (ruff C901 is enabled and must stay enabled). Prefer early returns, small pure functions, and extracting helpers over nesting.
- Pydantic v2 models for any structured data crossing a module boundary — no bare dicts in public signatures.
- No clever metaprogramming. Boring, explicit code wins.
- Docstrings: one-line summary + args only where non-obvious. No docstring novels.
- New graders/providers register by name in their `registry.py`; never add if/elif dispatch chains.

## Testing rules

- Every PR-sized change: `ruff check && mypy llm_verdict/core && pytest` must be green.
- No live API calls in tests, ever. Use `MockModelClient` or recorded fixtures.
- Stats functions in `core/stats.py` must have tests asserting against hand-computed expected values (document the hand computation in the test).
- When fixing a bug, write the failing test first, then fix.

## Workflow conventions

- Work in the phase order defined in `REQUIREMENTS.md §11`; don't start Phase N+1 until Phase N acceptance criteria pass.
- Conventional commits (`feat:`, `fix:`, `test:`, `refactor:`). One logical change per commit.
- If a requirement is ambiguous, write your interpretation as a comment in `DECISIONS.md` (create it if absent) rather than guessing silently.
- After completing each phase, update `DECISIONS.md` with anything that deviated from spec and why.

## Domain glossary

- **Suite hash** — SHA-256 over canonicalized task suite content; the identity of "what was tested."
- **Trial** — one model call for one task; default 3 trials per task even at temperature 0.
- **Cost-per-success** — total run cost ÷ tasks passed (majority vote across trials). The headline metric.
- **Verdict** — output of pre-registered criteria: ADOPT / HOLD / REJECT / INSUFFICIENT_DATA. (Lowercase `verdict` in commands refers to the CLI; "the verdict engine" refers to `core/verdict.py`. Context disambiguates — when writing docs or errors, prefer "adoption verdict" for the output.)
- **Judge calibration (kappa)** — Cohen's kappa between LLM judge and human annotations; < 0.6 triggers report warnings.
- **Incumbent** — the currently adopted model a challenger is compared against, defined per category in `configs/criteria/`.

## Common pitfalls in this codebase

- Forgetting that cache hits still need cost/latency fields populated (copy from cached record, flag `cached=true`).
- Bootstrapping over trials instead of tasks — resample **tasks**, then aggregate trials within each.
- Letting JSON-repair logic silently mask format violations — a repaired response must still carry the `format_violation` flag.
- Comparing runs with different suite hashes — the compare command must hard-error on mismatch.
