"""Log panel widget — smart-truncated training log display."""

from __future__ import annotations

from textual.widgets import Static

from autotrainer.pf_integration.log_parser import LogParser

# Patterns that indicate important log lines (displayed in smart mode)
_KEY_PATTERNS = [
    "loss",
    "step",
    "epoch",
    "ERROR",
    "WARNING",
    "FATAL",
    "OOM",
    "NaN",
    "NCCL",
    "eval_",
    "Checkpoint saved",
    "throughput",
    "tokens/s",
    "learning_rate",
]


class LogPanel(Static):
    """Smart-truncated log display.

    In 'smart' mode: shows only key lines (errors, metrics, milestones).
    In 'full' mode: shows everything.
    """

    MAX_VISIBLE_LINES = 80

    def __init__(self, *args, **kwargs):
        super().__init__("  [dim]Waiting for training logs...[/dim]", *args, **kwargs)
        self._full_log: list[str] = []
        self._key_lines: list[str] = []
        self._mode: str = "smart"
        self._parser = LogParser()

    def append_line(self, line: str):
        """Add a new log line."""
        self._full_log.append(line)

        if self._is_key_line(line):
            self._key_lines.append(line)
            # Keep key lines bounded
            if len(self._key_lines) > self.MAX_VISIBLE_LINES * 2:
                self._key_lines = self._key_lines[-self.MAX_VISIBLE_LINES:]

        self._render()

    def toggle_mode(self):
        """Switch between smart and full mode."""
        self._mode = "full" if self._mode == "smart" else "smart"
        self._render()

    def _is_key_line(self, line: str) -> bool:
        """Check if a line is important enough to display."""
        return any(p.lower() in line.lower() for p in _KEY_PATTERNS)

    def _render(self):
        """Render the log panel content."""
        if self._mode == "smart":
            lines = self._key_lines[-self.MAX_VISIBLE_LINES :]
        else:
            lines = self._full_log[-self.MAX_VISIBLE_LINES :]

        if not lines:
            return

        formatted = []
        for line in lines:
            if any(e in line for e in ("ERROR", "FATAL")):
                formatted.append(f"[red]{line}[/red]")
            elif any(w in line for w in ("WARNING", "WARN")):
                formatted.append(f"[yellow]{line}[/yellow]")
            elif "loss" in line.lower() and any(c.isdigit() for c in line):
                formatted.append(f"[green]{line}[/green]")
            elif "Checkpoint saved" in line:
                formatted.append(f"[blue]{line}[/blue]")
            else:
                formatted.append(line)

        content = "\n".join(formatted)
        mode_label = "smart" if self._mode == "smart" else "full"
        self.update(f"[dim]── Log ({mode_label}, {len(self._full_log)} lines) ──[/dim]\n{content}")

    def clear(self):
        """Clear all log content."""
        self._full_log.clear()
        self._key_lines.clear()
        self.update("  [dim]Waiting for training logs...[/dim]")
