"""Tests for Docker sandbox execution."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from llm_verdict.runner.sandbox import run_in_sandbox


@pytest.fixture
def mock_docker_success():
    """Mock a successful Docker run."""
    result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="All tests passed\n", stderr=""
    )
    with patch("llm_verdict.runner.sandbox.subprocess.run", return_value=result) as m:
        yield m


@pytest.fixture
def mock_docker_failure():
    """Mock a failed Docker run (assertion error)."""
    result = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="",
        stderr="AssertionError: assert 4 == 5\n",
    )
    with patch("llm_verdict.runner.sandbox.subprocess.run", return_value=result) as m:
        yield m


@pytest.fixture
def mock_docker_timeout():
    """Mock a Docker timeout."""
    with patch(
        "llm_verdict.runner.sandbox.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=35),
    ) as m:
        yield m


def test_sandbox_success(mock_docker_success):
    code = "def add(a, b): return a+b"
    test = "def test_add(): assert add(1,2)==3"
    result = run_in_sandbox(code, test)
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout == "All tests passed\n"


def test_sandbox_failure(mock_docker_failure):
    code = "def add(a, b): return a"
    test = "def test_add(): assert add(1,2)==3"
    result = run_in_sandbox(code, test)
    assert result.exit_code == 1
    assert "AssertionError" in result.stderr
    assert result.timed_out is False


def test_sandbox_timeout(mock_docker_timeout):
    result = run_in_sandbox("while True: pass", "def test_inf(): pass")
    assert result.timed_out is True
    assert result.exit_code == -1


def test_docker_called_with_network_none(mock_docker_success):
    run_in_sandbox("x = 1", "def test_x(): assert x == 1")
    call_args = mock_docker_success.call_args[0][0]
    assert "--network=none" in call_args


def test_docker_called_with_memory_limit(mock_docker_success):
    run_in_sandbox("x = 1", "def test_x(): assert x == 1")
    call_args = mock_docker_success.call_args[0][0]
    assert "--memory=256m" in call_args


def test_docker_called_with_read_only(mock_docker_success):
    run_in_sandbox("x = 1", "def test_x(): assert x == 1")
    call_args = mock_docker_success.call_args[0][0]
    assert "--read-only" in call_args
