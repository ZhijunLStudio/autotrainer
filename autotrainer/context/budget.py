"""Context budget allocation, token limits, and budget tracking.

Inspired by Claude Code's tokenBudget.ts:
- Percentage-based budget allocation (existing)
- BudgetTracker: stateful tracker with diminishing returns detection
- Auto-compact trigger when usage exceeds threshold
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


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


# ── Budget Tracker (Claude Code-inspired) ──────────────────────────────────


@dataclass
class BudgetTracker:
    """Stateful token budget tracker with diminishing returns detection.

    Inspired by Claude Code's tokenBudget.ts:
    - Tracks continuation count and token deltas across turns
    - Detects diminishing returns (low new tokens per turn)
    - Decides whether to continue or trigger compaction
    """

    continuation_count: int = 0
    last_total_tokens: int = 0
    last_delta_tokens: int = 0
    started_at: float = field(default_factory=time.time)

    # Thresholds
    COMPLETION_THRESHOLD: float = 0.85  # compact at 85% usage
    DIMINISHING_THRESHOLD: int = 500    # less than 500 new tokens = diminishing
    MIN_CONTINUATIONS: int = 3          # don't check diminishing until 3 turns

    def check_budget(self, total_tokens_used: int, max_tokens: int) -> dict:
        """Check if we should continue or trigger compaction.

        Returns:
            {"action": "continue", ...} — safe to continue
            {"action": "compact", "reason": "..."} — should compact
            {"action": "stop", "reason": "..."} — hard stop
        """
        if max_tokens <= 0:
            return {"action": "continue"}

        usage_pct = total_tokens_used / max_tokens

        # Compute delta
        delta = total_tokens_used - self.last_total_tokens
        self.last_total_tokens = total_tokens_used

        # Hard stop: over 95%
        if usage_pct > 0.95:
            return {
                "action": "stop",
                "reason": f"Context usage {usage_pct:.0%} exceeds 95% hard limit",
                "usage_pct": round(usage_pct, 3),
            }

        # Diminishing returns check
        is_diminishing = (
            self.continuation_count >= self.MIN_CONTINUATIONS
            and delta < self.DIMINISHING_THRESHOLD
            and self.last_delta_tokens < self.DIMINISHING_THRESHOLD
        )

        if is_diminishing:
            return {
                "action": "compact",
                "reason": f"Diminishing returns: only {delta} new tokens after {self.continuation_count} turns",
                "usage_pct": round(usage_pct, 3),
            }

        # Auto-compact threshold
        if usage_pct > self.COMPLETION_THRESHOLD:
            return {
                "action": "compact",
                "reason": f"Context usage {usage_pct:.0%} exceeds {self.COMPLETION_THRESHOLD:.0%} threshold",
                "usage_pct": round(usage_pct, 3),
            }

        # Continue
        self.continuation_count += 1
        self.last_delta_tokens = delta
        return {
            "action": "continue",
            "continuation_count": self.continuation_count,
            "usage_pct": round(usage_pct, 3),
        }

    def reset(self):
        """Reset tracker (e.g., after compaction)."""
        self.continuation_count = 0
        self.last_total_tokens = 0
        self.last_delta_tokens = 0
        self.started_at = time.time()
