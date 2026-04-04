"""Skill registry — discovers, loads, and caches skill definitions.

Inspired by Claude Code's loadAgentsDir.ts:
- YAML frontmatter parsing with schema validation
- Memoized loading (clear on /clear or /compact)
- Unified registry for all skills with tool schema support
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from autotrainer.skills.base import BaseSkill, SkillMeta, parse_skill_markdown

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Discovers, loads, and manages all skills."""

    def __init__(self, skills_dir: str = ""):
        self.skills_dir = Path(skills_dir) if skills_dir else Path(__file__).parent
        self._skills: dict[str, BaseSkill] = {}
        self._loaded = False

    def load_all(self) -> None:
        """Discover and load all skills from the skills directory."""
        if self._loaded:
            return

        if not self.skills_dir.is_dir():
            logger.warning("Skills directory not found: %s", self.skills_dir)
            return

        for child in sorted(self.skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                self._load_skill(child.name, child)
            except Exception as e:
                logger.warning("Failed to load skill %s: %s", child.name, e)

        self._loaded = True
        logger.info("Loaded %d skills: %s", len(self._skills), list(self._skills.keys()))

    def _load_skill(self, skill_name: str, skill_dir: Path) -> BaseSkill:
        """Load a single skill from its directory."""
        skill_md = skill_dir / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")

        meta, prompt_body = parse_skill_markdown(content, str(skill_dir))

        # Import handler module if it exists
        handler_module = None
        handler_path = skill_dir / "handler.py"
        if handler_path.exists():
            module_path = f"autotrainer.skills.{skill_name}.handler"
            try:
                handler_module = importlib.import_module(module_path)
            except Exception as e:
                logger.warning("Failed to import handler for %s: %s", skill_name, e)

        skill = BaseSkill(meta=meta, prompt_body=prompt_body, handler_module=handler_module)
        self._skills[skill_name] = skill
        return skill

    def get(self, skill_name: str) -> BaseSkill | None:
        """Get a skill by name. Lazy-loads if not already loaded."""
        if not self._loaded:
            self.load_all()
        return self._skills.get(skill_name)

    def list_skills(self) -> list[str]:
        """List all available skill names."""
        if not self._loaded:
            self.load_all()
        return sorted(self._skills.keys())

    def get_tool_schemas(self, skill_name: str) -> list[dict[str, Any]]:
        """Get JSON Schema definitions for a skill's tools."""
        skill = self.get(skill_name)
        if skill:
            return skill.get_tool_schemas()
        return []

    def clear_cache(self) -> None:
        """Clear the registry cache (analogous to /clear in Claude Code)."""
        self._skills.clear()
        self._loaded = False

    def get_skill_for_llm(self, skill_name: str) -> dict[str, Any] | None:
        """Get skill info formatted for LLM tool_use."""
        skill = self.get(skill_name)
        if not skill:
            return None

        result = {
            "name": skill.name,
            "description": skill.meta.description,
            "system_prompt": skill.system_prompt,
        }
        if skill.meta.tools:
            result["tools"] = skill.meta.tools
        if skill.meta.model:
            result["model"] = skill.meta.model
        return result
