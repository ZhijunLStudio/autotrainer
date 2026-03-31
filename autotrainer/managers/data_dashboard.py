"""DataProcessingDashboard — Rich Live real-time progress dashboard.

Shows all datasets in a table with live status, current phase,
sample counts, and elapsed time. Thread-safe for parallel processing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box


class Phase(str, Enum):
    PENDING = "pending"
    EXPLORING = "exploring"
    GENERATING = "generating"
    VALIDATING = "validating"
    RUNNING = "running"
    CLEANING = "cleaning"
    PROFILING = "profiling"
    SPLITTING = "splitting"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


_PHASE_COLOR = {
    Phase.PENDING:    "dim",
    Phase.EXPLORING:  "cyan",
    Phase.GENERATING: "cyan",
    Phase.VALIDATING: "yellow",
    Phase.RUNNING:    "yellow",
    Phase.CLEANING:   "blue",
    Phase.PROFILING:  "blue",
    Phase.SPLITTING:  "blue",
    Phase.COMPLETED:  "green",
    Phase.FAILED:     "red",
    Phase.SKIPPED:    "dim",
}

_PHASE_ICON = {
    Phase.PENDING:    "⏳",
    Phase.EXPLORING:  "🔍",
    Phase.GENERATING: "✍",
    Phase.VALIDATING: "⚡",
    Phase.RUNNING:    "⚙",
    Phase.CLEANING:   "🧹",
    Phase.PROFILING:  "📊",
    Phase.SPLITTING:  "✂",
    Phase.COMPLETED:  "✓",
    Phase.FAILED:     "✗",
    Phase.SKIPPED:    "⟳",
}


@dataclass
class DatasetStatus:
    """Thread-safe status for one dataset."""
    name: str
    source_path: str
    phase: Phase = Phase.PENDING
    message: str = ""          # current step detail
    log_tail: list[str] = field(default_factory=list)  # last N lines
    samples: int = 0
    images: int = 0
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, phase: Phase | None = None, message: str = "", log_line: str = ""):
        with self._lock:
            if phase:
                self.phase = phase
                if phase == Phase.EXPLORING and not self.started_at:
                    self.started_at = time.time()
                if phase in (Phase.COMPLETED, Phase.FAILED, Phase.SKIPPED):
                    self.finished_at = time.time()
            if message:
                self.message = message
            if log_line:
                self.log_tail.append(log_line.rstrip())
                if len(self.log_tail) > 6:
                    self.log_tail.pop(0)

    def elapsed(self) -> str:
        if not self.started_at:
            return ""
        end = self.finished_at or time.time()
        secs = int(end - self.started_at)
        if secs < 60:
            return f"{secs}s"
        return f"{secs // 60}m{secs % 60:02d}s"

    def status_str(self) -> str:
        icon = _PHASE_ICON.get(self.phase, "?")
        color = _PHASE_COLOR.get(self.phase, "white")
        label = self.phase.value
        return f"[{color}]{icon} {label}[/{color}]"


class DataProcessingDashboard:
    """Rich Live dashboard for parallel dataset processing."""

    def __init__(self, dataset_paths: list[str], parallel: int = 1):
        self.datasets: list[DatasetStatus] = [
            DatasetStatus(name=p.rstrip("/").split("/")[-1], source_path=p)
            for p in dataset_paths
        ]
        self.parallel = parallel
        self._lock = threading.Lock()
        self._live: Live | None = None
        self._console = Console()
        self._running = False

    def get_status(self, source_path: str) -> DatasetStatus | None:
        for ds in self.datasets:
            if ds.source_path == source_path:
                return ds
        return None

    def start(self):
        self._running = True
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()

    def stop(self):
        self._running = False
        if self._live:
            self._live.update(self._render())
            self._live.stop()

    def refresh(self):
        if self._live:
            self._live.update(self._render())

    def _render(self):
        """Build the full dashboard layout."""
        total = len(self.datasets)
        completed = sum(1 for d in self.datasets if d.phase == Phase.COMPLETED)
        failed = sum(1 for d in self.datasets if d.phase == Phase.FAILED)
        running = sum(1 for d in self.datasets if d.phase in (Phase.EXPLORING, Phase.GENERATING, Phase.VALIDATING, Phase.RUNNING, Phase.CLEANING, Phase.PROFILING, Phase.SPLITTING))
        pending = total - completed - failed - running

        # Header
        header = (
            f"[bold]AutoTrainer · Data Processing[/bold]  "
            f"[dim]{total} datasets[/dim]  "
            f"[green]✓ {completed}[/green]  "
            f"[yellow]⚙ {running}[/yellow]  "
            f"[red]✗ {failed}[/red]  "
            f"[dim]⏳ {pending} pending[/dim]  "
            f"[dim]workers={self.parallel}[/dim]"
        )

        # Main table
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("#", width=4, no_wrap=True)
        table.add_column("Dataset", width=42, no_wrap=True)
        table.add_column("Status", width=16, no_wrap=True)
        table.add_column("Phase / Detail", width=38, no_wrap=True)
        table.add_column("Samples", width=9, justify="right")
        table.add_column("Time", width=7, justify="right")

        for i, ds in enumerate(self.datasets, 1):
            with ds._lock:
                phase = ds.phase
                msg = ds.message[:36] if ds.message else ""
                samples = str(ds.samples) if ds.samples else "-"
                elapsed = ds.elapsed()
                error_hint = ds.error[:36] if ds.error else ""

                detail = msg or error_hint or ""
                status_cell = ds.status_str()

            table.add_row(
                str(i),
                ds.name[:40],
                status_cell,
                detail,
                samples,
                elapsed,
            )

        # Active log tails (datasets currently running)
        active_logs = []
        for ds in self.datasets:
            with ds._lock:
                if ds.phase in (Phase.RUNNING, Phase.VALIDATING, Phase.GENERATING, Phase.EXPLORING):
                    if ds.log_tail:
                        active_logs.append(f"[cyan]{ds.name[:30]}[/cyan]")
                        for line in ds.log_tail[-3:]:
                            active_logs.append(f"  [dim]{line[:90]}[/dim]")

        log_section = "\n".join(active_logs) if active_logs else "[dim]No active output[/dim]"

        from rich.columns import Columns
        from rich.text import Text

        layout = Layout()
        layout.split_column(
            Layout(Panel(header, padding=(0, 1)), size=3),
            Layout(table, ratio=3),
            Layout(Panel(log_section, title="Live Output", padding=(0, 1)), size=min(12, len(active_logs) + 3)),
        )
        return layout
