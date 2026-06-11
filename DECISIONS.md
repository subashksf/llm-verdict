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
