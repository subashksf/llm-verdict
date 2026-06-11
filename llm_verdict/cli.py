"""CLI entry point for llm-verdict."""

from pathlib import Path

import typer

from llm_verdict.tasks.loader import SuiteLoadError, load_suite

app = typer.Typer(name="verdict", no_args_is_help=True)
suite_app = typer.Typer(name="suite", no_args_is_help=True)
run_app = typer.Typer(name="run", no_args_is_help=True, invoke_without_command=True)
report_app = typer.Typer(name="report", no_args_is_help=True)
judge_app = typer.Typer(name="judge", no_args_is_help=True)
db_app = typer.Typer(name="db", no_args_is_help=True)
task_app = typer.Typer(name="task", no_args_is_help=True)

app.add_typer(suite_app, name="suite")
app.add_typer(report_app, name="report")
app.add_typer(judge_app, name="judge")
app.add_typer(db_app, name="db")
app.add_typer(task_app, name="task")


# --- suite commands (validate and hash are working) ---


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


# --- run commands (stubs) ---


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
    typer.echo("not implemented")
    raise typer.Exit(code=0)


# --- grade commands (stubs) ---


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


@judge_app.command("calibrate")
def judge_calibrate(
    run: str = typer.Option(..., help="Run ID to calibrate against"),
    sample: int = typer.Option(30, help="Number of items to sample"),
) -> None:
    """Calibrate the LLM judge against human annotations."""
    typer.echo("not implemented")
    raise typer.Exit(code=0)


# --- report commands (stubs) ---


@report_app.command("card")
def report_card(run_id: str = typer.Argument(..., help="Run ID")) -> None:
    """Generate a single-model report card."""
    typer.echo("not implemented")
    raise typer.Exit(code=0)


@report_app.command("compare")
def report_compare(
    run_id_a: str = typer.Argument(..., help="First run ID"),
    run_id_b: str = typer.Argument(..., help="Second run ID"),
) -> None:
    """Generate a head-to-head comparison report."""
    typer.echo("not implemented")
    raise typer.Exit(code=0)


@report_app.command("timeline")
def report_timeline(
    model_family: str = typer.Option(..., "--model-family", help="Model family prefix"),
) -> None:
    """Generate a longitudinal timeline report."""
    typer.echo("not implemented")
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
