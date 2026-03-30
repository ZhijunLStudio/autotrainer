"""Status bar widget — displays pipeline phase, GPU stats, progress."""

from __future__ import annotations

from textual.widgets import Static


class StatusBar(Static):
    """Top status bar showing current phase, GPU utilization, and progress."""

    DEFAULT_CONTENT = "[b]Phase:[/b] INIT"

    def __init__(self, *args, **kwargs):
        super().__init__(self.DEFAULT_CONTENT, *args, **kwargs)

    def update_health(
        self,
        phase: str,
        gpu_util: float = 0,
        gpu_mem_pct: float = 0,
        step: int = 0,
        loss: float | None = None,
        eta: str = "",
    ):
        """Update with live health metrics."""
        parts = [f"[b]Phase:[/b] {phase}"]

        if gpu_util > 0:
            parts.append(f"[b]GPU:[/b] {gpu_util:.0f}%")
        if gpu_mem_pct > 0:
            parts.append(f"[b]Mem:[/b] {gpu_mem_pct:.0f}%")
        if step > 0:
            parts.append(f"[b]Step:[/b] {step}")
        if loss is not None:
            parts.append(f"[b]Loss:[/b] {loss:.4f}")
        if eta:
            parts.append(f"[b]ETA:[/b] {eta}")

        self.update("  |  ".join(parts))
