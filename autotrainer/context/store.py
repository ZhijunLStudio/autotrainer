"""Context store — manages context window with percentage-based budgets.

The context window is divided into sections, each with a token budget.
Raw logs and large data never enter context directly — they are accessed
through tools (file reads at specific offsets) when the LLM needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autotrainer.context.budget import ContextBudget
from autotrainer.context.token_counter import estimate_tokens, truncate_to_tokens


@dataclass
class ContextSection:
    """A named section of the context window."""

    name: str
    max_tokens: int
    content: str = ""
    items: list[str] = field(default_factory=list)

    @property
    def used_tokens(self) -> int:
        return estimate_tokens(self.content) + sum(estimate_tokens(item) for item in self.items)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def add(self, text: str) -> bool:
        """Add text to this section. Returns False if budget exceeded."""
        new_tokens = estimate_tokens(text)
        if new_tokens > self.remaining_tokens:
            return False
        self.items.append(text)
        self.content = "\n".join(self.items)
        return True

    def add_compressed(self, text: str):
        """Add text, compressing if needed to fit."""
        if estimate_tokens(text) <= self.remaining_tokens:
            self.items.append(text)
        else:
            compressed = truncate_to_tokens(text, self.remaining_tokens)
            self.items.append(compressed)
        self.content = "\n".join(self.items)

    def set(self, text: str):
        """Replace all content in this section."""
        self.items = [text]
        self.content = text

    def clear(self):
        """Clear all content."""
        self.items.clear()
        self.content = ""


class ContextStore:
    """Manages LLM context windows with percentage-based token budgets.

    Sections:
        system   — task goal, constraints, skill definitions
        data     — data profiles, dataset summaries
        decision — experiment history, plan-experiment output
        working  — current task state, recent events
        response — reserved for LLM response (not user-settable)
    """

    def __init__(self, max_tokens: int = 128_000, budget: ContextBudget | None = None):
        self.max_tokens = max_tokens
        self.budget = budget or ContextBudget()
        self._sections: dict[str, ContextSection] = {
            "system": ContextSection("system", self.budget.get_limit("system", max_tokens)),
            "data": ContextSection("data", self.budget.get_limit("data", max_tokens)),
            "decision": ContextSection("decision", self.budget.get_limit("decision", max_tokens)),
            "working": ContextSection("working", self.budget.get_limit("working", max_tokens)),
            "response": ContextSection("response", self.budget.get_limit("response", max_tokens)),
        }

    def get_section(self, name: str) -> ContextSection:
        if name not in self._sections:
            raise ValueError(f"Unknown section: {name}")
        return self._sections[name]

    def set_system(self, text: str):
        """Set the system context (task, constraints, skills)."""
        self._sections["system"].set(text)

    def set_data_profile(self, profile: dict):
        """Set data profile summary in the data section."""
        import json

        text = json.dumps(profile, indent=2, ensure_ascii=False)
        self._sections["data"].set(text)

    def add_experiment_summary(self, experiment_id: str, summary: str):
        """Add a historical experiment summary to the decision section."""
        entry = f"## Experiment: {experiment_id}\n{summary}"
        self._sections["decision"].add_compressed(entry)

    def set_working_context(self, text: str):
        """Set current working context (phase, status, recent events)."""
        self._sections["working"].set(text)

    def build_prompt(self) -> str:
        """Build the full context string from all sections."""
        parts = []
        for name in ["system", "data", "decision", "working"]:
            section = self._sections[name]
            if section.content:
                parts.append(f"=== {name.upper()} ===\n{section.content}")
        return "\n\n".join(parts)

    def get_budget_report(self) -> dict[str, dict]:
        """Get a report of token usage vs budget for all sections."""
        report = {}
        for name, section in self._sections.items():
            report[name] = {
                "budget": section.max_tokens,
                "used": section.used_tokens,
                "remaining": section.remaining_tokens,
                "usage_pct": round((section.used_tokens / section.max_tokens * 100), 1) if section.max_tokens > 0 else 0,
            }
        return report

    def clear_section(self, name: str):
        """Clear a specific section."""
        self._sections[name].clear()

    def update_max_tokens(self, new_max: int):
        """Update max tokens (e.g., when switching models) and resize sections."""
        self.max_tokens = new_max
        for name, section in self._sections.items():
            section.max_tokens = self.budget.get_limit(name, new_max)

    @property
    def total_used_tokens(self) -> int:
        return sum(s.used_tokens for s in self._sections.values())
