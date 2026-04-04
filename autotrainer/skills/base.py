"""Base skill abstraction with YAML frontmatter and JSON Schema tool definition.

Inspired by Claude Code's agent/skill system:
- YAML frontmatter for metadata (name, description, tools, hooks)
- JSON Schema for input validation
- Lifecycle hooks (pre_run / post_run)
- Unified interface for all skills
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class SkillMeta:
    """Metadata parsed from a SKILL.md YAML frontmatter."""

    name: str
    description: str
    version: str = "1.0.0"
    tier: int = 2  # 1 = fast (pattern match), 2 = LLM-assisted
    tools: list[dict[str, Any]] = field(default_factory=list)
    hooks: dict[str, str] = field(default_factory=dict)  # {"pre_run": "module.func"}
    model: str | None = None  # override model for this skill
    tags: list[str] = field(default_factory=list)


def parse_skill_markdown(content: str, skill_dir: str = "") -> tuple[SkillMeta, str]:
    """Parse a SKILL.md file into (metadata, prompt_body).

    Supports YAML frontmatter delimited by ---:
    ---
    name: plan-experiment
    description: ...
    tools:
      - name: suggest_params
        parameters:
          type: object
          properties: ...
    ---

    If no frontmatter, returns default metadata with the full content as prompt.
    """
    # Match YAML frontmatter
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if match:
        fm_data = yaml.safe_load(match.group(1)) or {}
        prompt_body = match.group(2).strip()
    else:
        fm_data = {}
        prompt_body = content.strip()

    # Extract name from first heading if not in frontmatter
    if "name" not in fm_data:
        heading_match = re.match(r"^#\s+Skill:\s*(.+)", prompt_body, re.MULTILINE)
        if heading_match:
            fm_data["name"] = heading_match.group(1).strip().lower().replace(" ", "-")
        elif skill_dir:
            fm_data["name"] = Path(skill_dir).name
        else:
            fm_data["name"] = "unnamed"

    if "description" not in fm_data:
        fm_data["description"] = ""

    meta = SkillMeta(
        name=fm_data.get("name", "unnamed"),
        description=fm_data.get("description", ""),
        version=fm_data.get("version", "1.0.0"),
        tier=fm_data.get("tier", 2),
        tools=fm_data.get("tools", []),
        hooks=fm_data.get("hooks", {}),
        model=fm_data.get("model"),
        tags=fm_data.get("tags", []),
    )

    return meta, prompt_body


class BaseSkill:
    """Base class for all skills. Provides unified interface and lifecycle hooks."""

    def __init__(self, meta: SkillMeta, prompt_body: str, handler_module: Any = None):
        self.meta = meta
        self.prompt_body = prompt_body
        self._handler = handler_module
        self._pre_run_hooks: list[Callable] = []
        self._post_run_hooks: list[Callable] = []

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def system_prompt(self) -> str:
        """Full system prompt for this skill."""
        return self.prompt_body

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return JSON Schema definitions for tools this skill provides."""
        return self.meta.tools

    async def pre_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute pre-run hooks. Returns (possibly modified) context."""
        for hook in self._pre_run_hooks:
            context = await hook(context) if callable(hook) else context
        return context

    async def post_run(self, result: dict[str, Any]) -> dict[str, Any]:
        """Execute post-run hooks. Returns (possibly modified) result."""
        for hook in self._post_run_hooks:
            result = await hook(result) if callable(hook) else result
        return result

    def call_handler_method(self, method_name: str, *args, **kwargs) -> Any:
        """Call a method on the handler module if it exists."""
        if self._handler and hasattr(self._handler, method_name):
            return getattr(self._handler, method_name)(*args, **kwargs)
        return None

    def __repr__(self) -> str:
        return f"BaseSkill(name={self.name!r}, tier={self.meta.tier}, tools={len(self.meta.tools)})"
