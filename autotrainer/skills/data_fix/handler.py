"""Data-fix skill handler — repair broken conversion scripts."""

from __future__ import annotations

import json
from typing import Any


class DataFixHandler:
    """Given a broken script + error, produce a fixed version via LLM."""

    SYSTEM_PROMPT_PATH = __file__.replace("handler.py", "SKILL.md")

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._skill_md: str | None = None

    def _load_skill_md(self) -> str:
        if not self._skill_md:
            with open(self.SYSTEM_PROMPT_PATH, "r") as f:
                self._skill_md = f.read()
        return self._skill_md

    async def fix_script(
        self,
        original_script: str,
        error_summary: str,
        data_samples: list[Any],
        attempt: int = 1,
    ) -> dict[str, Any]:
        """Fix a broken conversion script.

        Returns dict with: root_cause, fix_applied, script
        """
        if not self.llm:
            return {"error": "LLM client not configured", "script": original_script}

        samples_str = json.dumps(data_samples[:5], ensure_ascii=False, indent=2)

        user_prompt = f"""Attempt #{attempt} fix needed.

Error:
{error_summary}

Data samples (first 5):
{samples_str}

Original script:
```python
{original_script}
```

Fix the script.
"""
        system = self._load_skill_md()
        result = await self.llm.complete_json(
            system=system,
            user=user_prompt,
            max_tokens=4096,
        )
        return result
