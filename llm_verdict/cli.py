"""CLI entry point for llm-verdict."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from llm_verdict.tasks.loader import SuiteLoadError, load_suite

app = typer.Typer(name="verdict", no_args_is_help=True)
suite_app = typer.Typer(name="suite", no_args_is_help=True)
report_app = typer.Typer(name="report", no_args_is_help=True)
judge_app = typer.Typer(name="judge", no_args_is_help=True)
db_app = typer.Typer(name="db", no_args_is_help=True)
task_app = typer.Typer(name="task", no_args_is_help=True)

app.add_typer(suite_app, name="suite")
app.add_typer(report_app, name="report")
app.add_typer(judge_app, name="judge")
app.add_typer(db_app, name="db")
app.add_typer(task_app, name="task")


# --- suite commands ---


@suite_app.command("validate")
def suite_validate(path: Path) -> None:
    """Schema-check and validate a task suite."""
    try:
        suite = load_suite(path)
    except SuiteLoadError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK: {len(suite.tasks)} tasks validated")
    typer.echo(f"Suite hash: {suite.suite_hash}")


@suite_app.command("hash")
def suite_hash(path: Path) -> None:
    """Compute and print the canonical hash of a task suite."""
    try:
        suite = load_suite(path)
    except SuiteLoadError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(suite.suite_hash)


# --- run command ---


@app.command("run")
def run_cmd(
    model: str = typer.Option(..., help="Model name from configs/models/"),
    suite: Path = typer.Option(..., help="Path to task suite directory"),
    trials: int = typer.Option(3, help="Trials per task"),
    budget: float = typer.Option(10.0, help="Budget cap in USD"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print plan without executing"
    ),
    resume: str = typer.Option("", help="Resume a partial run by run_id"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable response cache"),
) -> None:
    """Execute an evaluation run."""
    from llm_verdict.providers.registry import create_client
    from llm_verdict.runner.cache import ResponseCache
    from llm_verdict.runner.engine import BudgetExceededError, Engine, RunConfig
    from llm_verdict.store.duck import init_db

    try:
        task_suite = load_suite(suite)
    except SuiteLoadError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1)

    try:
        client = create_client(model)
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1)
    db_path = Path("verdict.duckdb")
    db_conn = init_db(db_path)

    cache = None
    if not no_cache:
        cache = ResponseCache(Path(".verdict_cache.duckdb"))

    config = RunConfig(
        model_name=model,
        suite=task_suite,
        trials_per_task=trials,
        budget_usd=budget,
        temperature=0.0,
        no_cache=no_cache,
        resume_run_id=resume or None,
    )

    if dry_run:
        engine = Engine(client=client, db_conn=db_conn, cache=cache)
        est = engine._estimate_cost(task_suite.tasks, trials)
        typer.echo(f"Tasks: {len(task_suite.tasks)}")
        typer.echo(f"Total trials: {len(task_suite.tasks) * trials}")
        typer.echo(f"Estimated cost: ${est:.4f}")
        typer.echo(f"Budget: ${budget:.2f}")
        typer.echo(f"Suite hash: {task_suite.suite_hash}")
        return

    engine = Engine(client=client, db_conn=db_conn, cache=cache)
    try:
        manifest = asyncio.run(engine.execute_run(config))
        typer.echo(f"Run complete: {manifest.run_id}")
    except BudgetExceededError as e:
        typer.echo(f"Budget exceeded: {e}", err=True)
        raise typer.Exit(code=1)


# --- grade command ---


@app.command("grade")
def grade_cmd(
    run_id: str = typer.Argument(..., help="Run ID to grade"),
    regrade: bool = typer.Option(False, "--regrade", help="Append new scores"),
    grader: str = typer.Option("", help="Specific grader to use"),
) -> None:
    """Grade or regrade a completed run."""
    typer.echo("not implemented")
    raise typer.Exit(code=0)


# --- judge commands (stubs) ---


def _prompt_for_label() -> str:
    """Prompt user until they enter a valid label (y/n/s)."""
    while True:
        answer = typer.prompt("Your verdict (y/n/s)")
        if answer.lower() in ("y", "n", "s"):
            return answer.lower()
        typer.echo("Please enter y, n, or s")


def _collect_annotations(
    samples: list[dict], run: str, annotator: str, db_conn: object
) -> int:
    """Present samples interactively and collect annotations."""
    from llm_verdict.graders.judge.calibration import insert_annotation

    labeled = 0
    for i, s in enumerate(samples, 1):
        typer.echo(f"--- [{i}/{len(samples)}] Task: {s['task_id']} ---")
        preview = s["response_text"][:300]
        typer.echo(f"Response: {preview}")
        judge_label = "PASS" if s["judge_passed"] else "FAIL"
        typer.echo(f"Judge said: {judge_label}")

        answer = _prompt_for_label()
        if answer == "s":
            continue

        insert_annotation(
            db_conn, run, s["task_id"], s["trial_index"], annotator, answer == "y"
        )
        labeled += 1
    return labeled


@judge_app.command("calibrate")
def judge_calibrate(
    run: str = typer.Option(..., help="Run ID to calibrate against"),
    sample: int = typer.Option(30, help="Number of items to sample"),
    annotator: str = typer.Option("human", help="Annotator name"),
) -> None:
    """Calibrate the LLM judge against human annotations."""
    from llm_verdict.graders.judge.calibration import (
        compute_calibration,
        get_calibration_samples,
    )
    from llm_verdict.store.duck import init_db

    db_path = Path("verdict.duckdb")
    if not db_path.exists():
        typer.echo("ERROR: No database found. Run an evaluation first.", err=True)
        raise typer.Exit(code=1)

    db_conn = init_db(db_path)
    samples = get_calibration_samples(db_conn, run, sample)

    if not samples:
        typer.echo("ERROR: No scored trials found for this run.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Calibrating {len(samples)} samples from run {run}")
    typer.echo("For each response, enter: y=pass, n=fail, s=skip\n")

    labeled = _collect_annotations(samples, run, annotator, db_conn)
    typer.echo(f"\nLabeled {labeled} samples.")

    result = compute_calibration(db_conn, run)
    if result is None:
        typer.echo("Not enough data to compute kappa.")
        return

    typer.echo(f"Cohen's kappa: {result.kappa:.3f}")
    typer.echo(f"Agreement rate: {result.agreement_rate:.1%}")
    if result.kappa < 0.6:
        typer.echo("WARNING: Kappa below 0.6 — judge may not be reliable")


# --- report commands ---


@report_app.command("card")
def report_card(
    run_id: str = typer.Argument(..., help="Run ID"),
    output: Path = typer.Option(Path("reports"), help="Output directory"),
    html: bool = typer.Option(False, "--html", help="Also generate HTML"),
) -> None:
    """Generate a single-model report card."""
    from llm_verdict.reporting.data import compute_run_report
    from llm_verdict.reporting.renderer import render_card_html, render_card_markdown
    from llm_verdict.store.duck import init_db

    db_path = Path("verdict.duckdb")
    if not db_path.exists():
        typer.echo("ERROR: No database found.", err=True)
        raise typer.Exit(code=1)

    conn = init_db(db_path)
    report = compute_run_report(conn, run_id)

    output.mkdir(parents=True, exist_ok=True)
    md_path = output / f"{run_id}_card.md"
    md_path.write_text(render_card_markdown(report))
    typer.echo(f"Report card: {md_path}")

    if html:
        html_path = output / f"{run_id}_card.html"
        html_path.write_text(render_card_html(report))
        typer.echo(f"HTML report: {html_path}")


@report_app.command("compare")
def report_compare(
    run_id_a: str = typer.Argument(..., help="First run ID"),
    run_id_b: str = typer.Argument(..., help="Second run ID"),
    output: Path = typer.Option(Path("reports"), help="Output directory"),
) -> None:
    """Generate a head-to-head comparison report."""
    from llm_verdict.reporting.data import compute_compare_report
    from llm_verdict.reporting.renderer import render_compare_markdown
    from llm_verdict.store.duck import init_db

    db_path = Path("verdict.duckdb")
    if not db_path.exists():
        typer.echo("ERROR: No database found.", err=True)
        raise typer.Exit(code=1)

    conn = init_db(db_path)
    try:
        report = compute_compare_report(conn, run_id_a, run_id_b)
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    md_path = output / f"compare_{run_id_a[:8]}_{run_id_b[:8]}.md"
    md_path.write_text(render_compare_markdown(report))
    typer.echo(f"Comparison report: {md_path}")


@report_app.command("timeline")
def report_timeline(
    model_family: str = typer.Option(..., "--model-family", help="Model family prefix"),
) -> None:
    """Generate a longitudinal timeline report."""
    typer.echo("not implemented (Phase 5)")
    raise typer.Exit(code=0)


# --- db commands (stubs) ---


@db_app.command("query")
def db_query(sql: str = typer.Argument(..., help="SQL query to execute")) -> None:
    """Run a raw SQL query against the DuckDB store."""
    typer.echo("not implemented")
    raise typer.Exit(code=0)


# --- task commands (stubs) ---


@task_app.command("add-from-failure")
def task_add_from_failure(
    run_id: str = typer.Argument(..., help="Source run ID"),
    task_id: str = typer.Argument(..., help="Task ID to promote"),
) -> None:
    """Promote a failure into the task suite."""
    typer.echo("not implemented")
    raise typer.Exit(code=0)
