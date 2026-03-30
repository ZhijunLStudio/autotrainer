"""Context budget allocation and token limits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextBudget:
    """Percentage-based token budget allocation for context windows.

    All percentages should sum to 1.0. The budget determines how much
    of the LLM's context window each section is allowed to consume.
    """

    system_pct: float = 0.08      # System prompt, skill definitions
    data_pct: float = 0.12        # Data profiles, dataset summaries
    decision_pct: float = 0.40    # Experiment history, plan-experiment output
    working_pct: float = 0.20     # Current task context
    response_pct: float = 0.20    # Reserved for LLM response

    def __post_init__(self):
        total = self.system_pct + self.data_pct + self.decision_pct + self.working_pct + self.response_pct
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Budget percentages must sum to ~1.0, got {total:.2f}")

    def get_limit(self, section: str, max_tokens: int) -> int:
        """Get token limit for a section given the max context size."""
        pct_map = {
            "system": self.system_pct,
            "data": self.data_pct,
            "decision": self.decision_pct,
            "working": self.working_pct,
            "response": self.response_pct,
        }
        if section not in pct_map:
            raise ValueError(f"Unknown section: {section}. Valid: {list(pct_map.keys())}")
        return int(max_tokens * pct_map[section])
