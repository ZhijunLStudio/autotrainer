"""Skill runner — invokes skills with context injection via two-tier system.

Tier 1: Fast pattern matching (no LLM needed) — for common errors.
Tier 2: LLM-assisted analysis — for complex situations.
"""

from __future__ import annotations

import json
import re
from typing import Any

from autotrainer.context.store import ContextStore
from autotrainer.skills.loader import SkillLoader
from autotrainer.utils.llm_client import LLMClient


class SkillRunner:
    """Invokes skills with context injection.

    Each invocation:
    1. Loads the skill's SKILL.md as system prompt
    2. Assembles relevant context from ContextStore
    3. Either uses fast pattern matching (Tier 1) or LLM (Tier 2)
    4. Returns structured results
    """

    def __init__(
        self,
        context_store: ContextStore,
        skill_loader: SkillLoader,
        llm_client: LLMClient | None = None,
    ):
        self.context = context_store
        self.loader = skill_loader
        self.llm = llm_client

    async def run(
        self,
        skill_name: str,
        user_input: str,
        force_llm: bool = False,
    ) -> dict[str, Any]:
        """Run a skill with the given input.

        Args:
            skill_name: Name of the skill (e.g., "diagnose_training").
            user_input: The input text/data for the skill.
            force_llm: Skip Tier 1 pattern matching and go straight to LLM.

        Returns:
            Structured result dict from the skill.
        """
        if not force_llm:
            # Tier 1: Try fast pattern matching
            tier1_result = self._try_tier1(skill_name, user_input)
            if tier1_result:
                return {"tier": 1, "skill": skill_name, "result": tier1_result}

        # Tier 2: LLM analysis
        if not self.llm:
            return {"tier": 0, "skill": skill_name, "error": "LLM client not configured"}

        return await self._run_tier2(skill_name, user_input)

    def _try_tier1(self, skill_name: str, user_input: str) -> dict | None:
        """Try Tier 1 pattern matching for common cases.

        Returns None if no pattern matches (needs LLM).
        """
        if skill_name == "diagnose_training":
            return self._tier1_diagnose(user_input)
        return None

    def _tier1_diagnose(self, error_text: str) -> dict | None:
        """Fast error pattern matching for diagnose_training skill."""
        patterns = {
            "oom": {
                "patterns": [r"Out of memory", r"CUDA out of memory", r"OOM", r"ResourceExhaustedError"],
                "category": "memory",
                "fix": {"action": "reduce_batch_size", "factor": 0.5, "reason": "GPU OOM detected"},
            },
            "nan": {
                "patterns": [r"loss.*nan", r"NaN loss", r"gradient.*inf"],
                "category": "numerical",
                "fix": {"action": "reduce_learning_rate", "factor": 0.1, "reason": "NaN loss detected"},
            },
            "nccl": {
                "patterns": [r"NCCL.*error", r"NCCL.*timeout", r"nccl.*abort"],
                "category": "distributed",
                "fix": {"action": "increase_nccl_timeout", "reason": "NCCL communication failure"},
            },
            "data_format": {
                "patterns": [r"json\.decode", r"JSONDecodeError", r"KeyError.*messages"],
                "category": "data",
                "fix": {"action": "check_data_format", "reason": "Data format mismatch"},
            },
        }

        for error_type, config in patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, error_text, re.IGNORECASE):
                    return {
                        "error_type": error_type,
                        "category": config["category"],
                        "confidence": 0.9,
                        "fix": config["fix"],
                        "source": "pattern_match",
                    }

        return None

    async def _run_tier2(self, skill_name: str, user_input: str) -> dict:
        """Run Tier 2: LLM-assisted analysis."""
        skill_prompt = self.loader.load(skill_name)
        context_text = self.context.build_prompt()

        user_message = f"Context:\n{context_text}\n\nInput:\n{user_input}"

        try:
            result = await self.llm.complete_json(
                system=skill_prompt,
                user=user_message,
            )
            return {"tier": 2, "skill": skill_name, "result": result}
        except Exception as e:
            return {"tier": 2, "skill": skill_name, "error": str(e)}
