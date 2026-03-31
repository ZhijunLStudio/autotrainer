"""Data-inspect skill handler — generate conversion script from data samples."""

from __future__ import annotations

import json
from typing import Any


class DataInspectHandler:
    """Given data inspection result, generate a Python conversion script via LLM."""

    SYSTEM_PROMPT_PATH = __file__.replace("handler.py", "SKILL.md")

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._skill_md: str | None = None

    def _load_skill_md(self) -> str:
        if not self._skill_md:
            with open(self.SYSTEM_PROMPT_PATH, "r") as f:
                self._skill_md = f.read()
        return self._skill_md

    async def generate_script(
        self,
        inspection_summary: str,
        input_path: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Generate a conversion script for the given data.

        Returns dict with: format_identified, explanation, script
        """
        if not self.llm:
            return {"error": "LLM client not configured", "script": ""}

        user_prompt = f"""Data to convert:

{inspection_summary}

Input file path context: {input_path}

Write a Python script to convert this data to erniekit JSONL format.
"""
        if context:
            user_prompt += f"\nAdditional context: {context}"

        system = self._load_skill_md()

        result = await self.llm.complete_json(
            system=system,
            user=user_prompt,
            max_tokens=4096,
        )

        return result

    async def generate_script_sync(
        self,
        inspection_summary: str,
        input_path: str,
    ) -> dict[str, Any]:
        """Synchronous wrapper using asyncio."""
        import asyncio
        return asyncio.run(self.generate_script(inspection_summary, input_path))
