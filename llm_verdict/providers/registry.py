"""Config-driven model registry — load YAML, create client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_verdict.providers.base import ModelClient
from llm_verdict.providers.litellm_client import LiteLLMClient

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "models"


def load_model_config(name: str, configs_dir: Path | None = None) -> dict[str, Any]:
    """Load a model config YAML by name."""
    base = configs_dir or _CONFIGS_DIR
    path = base / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model config found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def create_client(name: str, configs_dir: Path | None = None) -> ModelClient:
    """Create a ModelClient from a named config."""
    config = load_model_config(name, configs_dir)
    return LiteLLMClient(config)
