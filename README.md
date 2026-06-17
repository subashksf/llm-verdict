# llm-verdict

Provider-agnostic LLM evaluation harness. Runs a versioned task suite against any model (via LiteLLM), grades results with programmatic checks and a calibrated LLM judge, and emits publishable model report cards.

## 10-Minute Quickstart

### 1. Install

```bash
git clone <this-repo> && cd llm-verdict
uv sync
```

### 2. Add a model

Create a YAML file in `configs/models/`. Example for GPT-4o:

```yaml
# configs/models/gpt-4o.yaml
name: gpt-4o
litellm_model: openai/gpt-4o
api_key_env: OPENAI_API_KEY
pricing:
  input_per_mtok_usd: 2.50
  output_per_mtok_usd: 10.00
  as_of: "2024-06-01"
limits:
  max_concurrency: 4
  requests_per_minute: 60
defaults:
  temperature: 0.0
  max_tokens: 4096
metadata:
  provider: openai
  family: gpt-4o
  context_window: 128000
```

### 3. Validate the example suite

```bash
uv run verdict suite validate suites/_examples
```

### 4. Run an evaluation

```bash
export OPENAI_API_KEY=sk-...
uv run verdict run --model gpt-4o --suite suites/_examples --budget 1.00
```

Use `--dry-run` to preview cost without calling APIs:

```bash
uv run verdict run --model gpt-4o --suite suites/_examples --dry-run
```

### 5. Generate a report card

```bash
uv run verdict report card <run_id>
```

This writes a Markdown report to `reports/<run_id>_card.md`. Add `--html` for a self-contained HTML version.

### 6. Compare two models

Run both models on the same suite, then:

```bash
uv run verdict report compare <run_id_a> <run_id_b>
```

Produces a head-to-head report with McNemar's test and per-category score deltas.

### 7. Track progress over time

```bash
uv run verdict report timeline --model-family gpt-4o
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Suite** | A directory of YAML task files. Identified by its content hash. |
| **Trial** | One model call for one task. Default 3 trials per task. |
| **Grader** | Scores a trial: `exact_match`, `json_schema`, `regex`, `tool_call`, `code_exec`, or `judge`. |
| **Verdict** | ADOPT / HOLD / REJECT / INSUFFICIENT_DATA based on pre-registered criteria. |
| **Cost-per-success** | Total run cost / tasks passed. The headline efficiency metric. |

## CLI Reference

```
verdict suite validate <path>         # schema-check + hash a task suite
verdict suite hash <path>             # print suite hash
verdict run --model <name> --suite <path> [--trials 3] [--budget 10.00] [--dry-run]
verdict run --resume <run_id>         # complete a partial run
verdict judge calibrate --run <id>    # calibrate judge vs. human labels
verdict report card <run_id>          # single-model report card
verdict report compare <id_a> <id_b>  # head-to-head comparison
verdict report timeline --model-family <prefix>  # longitudinal
verdict db query "<sql>"              # raw DuckDB query
verdict task add-from-failure <run_id> <task_id> --suite <path>
```

## Writing Tasks

Each task is a YAML file in a suite directory:

```yaml
task_id: "coding/fizzbuzz-001"
category: "coding"
prompt: "Write a Python function that returns FizzBuzz for 1-100."
task_type: "code_exec"
grader:
  name: "code_exec"
  params:
    test_code: |
      result = solution()
      assert result[2] == "Fizz"
      assert result[4] == "Buzz"
      assert result[14] == "FizzBuzz"
metadata:
  difficulty: "easy"
version: 1
```

Supported `task_type` values: `exact`, `json_schema`, `regex`, `code_exec`, `tool_call`, `judged`.

## Pre-registered Criteria

Define adoption criteria in `configs/criteria/`:

```yaml
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

## Development

```bash
uv sync
uv run ruff check --fix . && uv run ruff format .
uv run mypy llm_verdict/core
uv run pytest
```

All tests run offline — no API keys needed. The test suite uses `MockModelClient` with scripted responses.
