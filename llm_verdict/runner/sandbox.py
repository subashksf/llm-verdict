"""Docker-based sandbox for code execution — airtight network isolation."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    """Output of a sandboxed code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


_DOCKER_IMAGE = "python:3.12-slim"


def run_in_sandbox(
    code: str,
    test_code: str,
    timeout_seconds: int = 30,
) -> SandboxResult:
    """Run code + tests inside a Docker container with no network access."""
    combined = f"{code}\n\n{test_code}\n"
    test_funcs = [
        line.split("(")[0].replace("def ", "").strip()
        for line in test_code.splitlines()
        if line.strip().startswith("def test_")
    ]
    if test_funcs:
        combined += "\n" + "\n".join(f"{fn}()" for fn in test_funcs)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(combined)
        script_path = Path(f.name)

    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--memory=256m",
                "--cpus=1",
                "--read-only",
                "--tmpfs=/tmp:size=64m",
                "-v",
                f"{script_path}:/app/run.py:ro",
                _DOCKER_IMAGE,
                "python",
                "/app/run.py",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 5,
        )
        return SandboxResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            stdout="",
            stderr="Execution timed out",
            exit_code=-1,
            timed_out=True,
        )
    finally:
        script_path.unlink(missing_ok=True)
