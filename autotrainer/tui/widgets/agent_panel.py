"""Agent panel widget — displays LLM thinking, recommendations, and pending actions."""

from __future__ import annotations

from textual.widgets import Static


class AgentPanel(Static):
    """Side panel for LLM agent output.

    Shows:
    - Current thinking/analysis
    - Recommendations with confidence
    - Pending user confirmations
    - Context usage stats
    """

    def __init__(self, *args, **kwargs):
        super().__init__("  [dim]Agent ready[/dim]", *args, **kwargs)
        self._history: list[str] = []

    def set_thinking(self, message: str):
        """Show agent is thinking."""
        self.update(f"  [italic dim]Thinking...[/italic dim]\n\n  {message}")

    def set_recommendation(self, title: str, steps: list[str], confidence: float = 0):
        """Show a recommendation with steps."""
        conf_str = f" (confidence: {confidence:.0%})" if confidence > 0 else ""
        lines = [f"  [bold cyan]{title}[/bold cyan]{conf_str}\n"]
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
        content = "\n".join(lines)
        self.update(content)
        self._history.append(title)

    def set_confirm(self, message: str):
        """Show a confirmation prompt."""
        self.update(
            f"\n  [yellow bold]?[/yellow bold] {message}\n\n"
            f"  [dim]Type 'y' or 'n' in the input bar below[/dim]"
        )

    def add_event(self, event: str):
        """Append an event to the agent panel."""
        current = str(self.renderable) if self.renderable else ""
        self.update(f"{current}\n  [dim]>[/dim] {event}")

    def get_history(self) -> list[str]:
        """Get the history of recommendations."""
        return self._history
