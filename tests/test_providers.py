"""Tests for provider config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_verdict.providers.registry import load_model_config


def test_load_model_config_existing() -> None:
    config = load_model_config("claude-sonnet-4-6")
    assert config["name"] == "claude-sonnet-4-6"
    assert config["litellm_model"] == "anthropic/claude-sonnet-4-6"
    assert config["pricing"]["input_per_mtok_usd"] == 3.00
    assert config["pricing"]["output_per_mtok_usd"] == 15.00
    assert config["limits"]["max_concurrency"] == 4


def test_load_model_config_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="No model config found"):
        load_model_config("nonexistent-model")


def test_load_model_config_custom_dir(tmp_path: Path) -> None:
    config_file = tmp_path / "my-model.yaml"
    config_file.write_text(
        "name: my-model\n"
        "litellm_model: openai/gpt-4\n"
        "api_key_env: OPENAI_API_KEY\n"
        "pricing:\n"
        "  input_per_mtok_usd: 5.0\n"
        "  output_per_mtok_usd: 15.0\n"
        "limits:\n"
        "  max_concurrency: 2\n"
        "defaults:\n"
        "  max_tokens: 2048\n"
    )
    config = load_model_config("my-model", configs_dir=tmp_path)
    assert config["name"] == "my-model"
    assert config["limits"]["max_concurrency"] == 2
