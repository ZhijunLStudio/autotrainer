"""Skill loader — discovers and loads SKILL.md files for LLM context injection."""

from __future__ import annotations

import os
from pathlib import Path


class SkillLoader:
    """Loads skill definitions from SKILL.md files.

    Skills are directories under autotrainer/skills/ containing:
    - SKILL.md: The prompt/documentation for the skill
    - handler.py: Python logic for the skill
    """

    def __init__(self, skills_dir: str = ""):
        self.skills_dir = skills_dir or str(Path(__file__).parent)
        self._cache: dict[str, str] = {}

    def load(self, skill_name: str) -> str:
        """Load the SKILL.md content for a named skill.

        Raises FileNotFoundError if the skill doesn't exist.
        """
        if skill_name not in self._cache:
            path = os.path.join(self.skills_dir, skill_name, "SKILL.md")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Skill not found: {skill_name} (looked at {path})")
            with open(path, "r") as f:
                self._cache[skill_name] = f.read()
        return self._cache[skill_name]

    def list_skills(self) -> list[str]:
        """List all available skills."""
        skills = []
        try:
            for name in os.listdir(self.skills_dir):
                skill_md = os.path.join(self.skills_dir, name, "SKILL.md")
                if os.path.isdir(os.path.join(self.skills_dir, name)) and os.path.exists(skill_md):
                    skills.append(name)
        except FileNotFoundError:
            pass
        return sorted(skills)

    def get_handler(self, skill_name: str):
        """Import and return the handler module for a skill."""
        import importlib

        module_path = f"autotrainer.skills.{skill_name}.handler"
        module = importlib.import_module(module_path)
        return module
