"""
Prompt manager — loads and renders prompts from versioned YAML config.

WHY NOT HARDCODE PROMPTS:
  1. Git tracks changes: you can diff prompt v1.0 vs v1.1
  2. You can A/B test prompt versions without changing code
  3. If a prompt change tanks your eval scores, you can revert
  4. Source code stays clean — no giant string literals
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"


class PromptManager:
    """Loads and renders prompt templates from YAML."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG):
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

    @property
    def version(self) -> str:
        return self._config.get("version", "unknown")

    def get_system_prompt(self, task: str) -> str:
        """Get the system prompt for a task (e.g. 'generation')."""
        return self._config[task]["system"].strip()

    def render_user_prompt(self, task: str, **kwargs) -> str:
        """Render a user prompt template with variables."""
        template = self._config[task]["user_template"]
        return template.format(**kwargs).strip()
