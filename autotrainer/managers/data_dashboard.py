"""DataProcessingDashboard — Rich Live real-time progress dashboard.

Shows all datasets in a table with live status, current phase,
sample counts, elapsed time, and step-by-step detail.
Thread-safe for parallel processing.
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
from rich.text import Text
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


# ── Display styles ────────────────────────────────────────────

_PHASE_STYLE = {
    Phase.PENDING:    {"icon": "⏳", "color": "dim",        "label": "pending"},
    Phase.EXPLORING:  {"icon": "🔎", "color": "cyan",       "label": "exploring"},
    Phase.GENERATING: {"icon": "✍",  "color": "cyan",       "label": "generating script"},
    Phase.VALIDATING: {"icon": "⚡", "color": "yellow",     "label": "validating"},
    Phase.RUNNING:    {"icon": "⚙",  "color": "yellow",     "label": "running script"},
    Phase.CLEANING:   {"icon": "🧹", "color": "blue",       "label": "cleaning"},
    Phase.PROFILING:  {"icon": "📊", "color": "blue",       "label": "profiling"},
    Phase.SPLITTING:  {"icon": "✂",  "color": "blue",       "label": "splitting"},
    Phase.COMPLETED:  {"icon": "✓",  "color": "green",      "label": "completed"},
    Phase.FAILED:     {"icon": "✗",  "color": "red",        "label": "failed"},
    Phase.SKIPPED:    {"icon": "⟳",  "color": "dim",        "label": "skipped"},
}

# Spinner frames for active states
_SPIN_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_SPIN_IDX = 0


def _spin_icon(phase: Phase) -> str:
    """Return a spinner for active phases, static icon for completed ones."""
    global _SPIN_IDX
    style = _PHASE_STYLE.get(phase, {})
    if phase in (Phase.COMPLETED, Phase.FAILED, Phase.SKIPPED):
        return style.get("icon", "?")
    # Spinning icon
    icon = _SPIN_FRAMES[_SPIN_IDX % len(_SPIN_FRAMES)]
    return icon


@dataclass
class StepRecord:
    """One step in the processing timeline (like Claude Code's tool_use entries)."""
    action: str        # e.g. "shell ls -la", "python inspect schema", "sandbox run attempt 1"
    started_at: float = 0.0
    finished_at: float = 0.0
    result: str = ""   # e.g. "OK (510 rows)", "timeout", ""

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return max(0, end - self.started_at)

    @property
    def elapsed_str(self) -> str:
        s = int(self.elapsed)
        if s < 60:
            return f"{s}s"
        return f"{s // 60}m{s % 60:02d}s"


@dataclass
class DatasetStatus:
    """Thread-safe status for one dataset, with step-level tracking."""
    name: str
    source_path: str
    phase: Phase = Phase.PENDING
    message: str = ""          # one-line current detail
    steps: list[StepRecord] = field(default_factory=list)  # completed steps
    current_step: StepRecord | None = None  # in-progress step
    log_tail: list[str] = field(default_factory=list)
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
                if phase in (Phase.EXPLORING, Phase.GENERATING) and not self.started_at:
                    self.started_at = time.time()
                if phase in (Phase.COMPLETED, Phase.FAILED, Phase.SKIPPED):
                    self.finished_at = time.time()
            if message:
                self.message = message
            if log_line:
                self.log_tail.append(log_line.rstrip())
                if len(self.log_tail) > 10:
                    self.log_tail.pop(0)

    def step_start(self, action: str):
        """Start a new processing step."""
        with self._lock:
            # Finish previous step if still running
            if self.current_step:
                self.current_step.finished_at = time.time()
                self.steps.append(self.current_step)
                if len(self.steps) > 20:
                    self.steps = self.steps[-20:]
            self.current_step = StepRecord(action=action, started_at=time.time())

    def step_finish(self, result: str = "OK"):
        """Mark the current step as finished."""
        with self._lock:
            if self.current_step:
                self.current_step.finished_at = time.time()
                self.current_step.result = result
                self.steps.append(self.current_step)
                if len(self.steps) > 20:
                    self.steps = self.steps[-20:]
                self.current_step = None

    def elapsed(self) -> str:
        if not self.started_at:
            return ""
        end = self.finished_at or time.time()
        secs = int(end - self.started_at)
        if secs < 60:
            return f"{secs}s"
        return f"{secs // 60}m{secs % 60:02d}s"

    def status_display(self) -> tuple[str, str]:
        """Return (status_colored, detail_colored) for the table."""
        style = _PHASE_STYLE.get(self.phase, {})
        icon = _spin_icon(self.phase)
        color = style.get("color", "white")
        label = style.get("label", self.phase.value)

        status = f"[{color}]{icon} {label}[/{color}]"

        # Detail: current action + elapsed for current step
        if self.current_step and not self.finished_at:
            step_secs = int(time.time() - self.current_step.started_at)
            detail = f"{self.current_step.action[:35]} ({step_secs}s)"
        elif self.message:
            detail = self.message[:40]
        elif self.error:
            detail = f"[red]{self.error[:40]}[/red]"
        elif self.samples:
            detail = f"done — {self.samples} samples"
        else:
            detail = ""

        return status, detail


class DataProcessingDashboard:
    """Rich Live dashboard for parallel dataset processing."""

    def __init__(self, dataset_paths: list[str], parallel: int = 1):
        global _SPIN_IDX
        _SPIN_IDX = 0
        self.datasets: list[DatasetStatus] = [
            DatasetStatus(name=p.rstrip("/").split("/")[-1], source_path=p)
            for p in dataset_paths
        ]
        self.parallel = parallel
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
            refresh_per_second=2,
            screen=False,
        )
        self._live.start()

    def stop(self):
        self._running = False
        if self._live:
            self._live.update(self._render())
            self._live.stop()

    def refresh(self):
        global _SPIN_IDX
        _SPIN_IDX += 1
        if self._live:
            self._live.update(self._render())

    def _render(self):
        """Build the full dashboard layout."""
        total = len(self.datasets)
        completed = sum(1 for d in self.datasets if d.phase == Phase.COMPLETED)
        failed = sum(1 for d in self.datasets if d.phase == Phase.FAILED)
        running = sum(1 for d in self.datasets if d.phase not in (Phase.PENDING, Phase.COMPLETED, Phase.FAILED, Phase.SKIPPED))
        pending = total - completed - failed - running - sum(1 for d in self.datasets if d.phase == Phase.SKIPPED)

        header = Text()
        header.append("AutoTrainer · Data Processing ", style="bold")
        header.append(f"{total} datasets  ", style="dim")
        header.append(f"✓ {completed}  ", style="green")
        header.append(f"⚙ {running}  ", style="yellow")
        header.append(f"✗ {failed}  ", style="red")
        header.append(f"⏳ {pending}  ", style="dim")
        header.append(f"workers={self.parallel}", style="dim")

        # ── Main table ────────────────────────────────────────
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            expand=True,
            show_edge=False,
        )
        table.add_column("#", width=3, no_wrap=True)
        table.add_column("Dataset", width=40, no_wrap=True)
        table.add_column("Status", width=20, no_wrap=True)
        table.add_column("Current Step / Detail", width=44, no_wrap=True)
        table.add_column("Samples", width=8, justify="right")
        table.add_column("Time", width=8, justify="right")

        for i, ds in enumerate(self.datasets, 1):
            status_cell, detail_cell = ds.status_display()
            samples = str(ds.samples) if ds.samples else "-"
            table.add_row(
                str(i),
                ds.name[:38],
                status_cell,
                detail_cell,
                samples,
                ds.elapsed(),
            )

        # ── Step timeline for active datasets ─────────────────
        timeline_lines = []
        for ds in self.datasets:
            with ds._lock:
                if ds.phase not in (Phase.EXPLORING, Phase.GENERATING, Phase.VALIDATING,
                                     Phase.RUNNING, Phase.CLEANING, Phase.PROFILING, Phase.SPLITTING):
                    continue

                n_steps = len(ds.steps)
                style = _PHASE_STYLE.get(ds.phase, {})
                icon = style.get("icon", "?")
                color = style.get("color", "white")

                # Header
                timeline_lines.append(f"[bold {color}]{icon} {ds.name[:35]}[/bold {color}]")

                # Completed steps (last 3)
                for step in ds.steps[-3:]:
                    elapsed = step.elapsed_str
                    result_color = "green" if "OK" in step.result else ("red" if step.result else "dim")
                    timeline_lines.append(
                        f"  [dim]✓ {step.action[:50]}[/dim]  "
                        f"[{result_color}]{step.result[:30]}[/{result_color}]  "
                        f"[dim]{elapsed}[/dim]"
                    )

                # Current step (in progress)
                if ds.current_step:
                    step_secs = int(time.time() - ds.current_step.started_at)
                    timeline_lines.append(
                        f"  [yellow]▸ {ds.current_step.action[:50]}[/yellow]  "
                        f"[yellow]running {step_secs}s[/yellow]"
                    )

                # Recent log lines (from sandbox execution)
                for log in ds.log_tail[-2:]:
                    timeline_lines.append(f"  [dim]│ {log[:85]}[/dim]")

                timeline_lines.append("")  # blank line between datasets

        if not timeline_lines:
            timeline_lines.append("[dim]Waiting for datasets...[/dim]")

        timeline_text = "\n".join(timeline_lines)

        # ── Assemble layout ───────────────────────────────────
        layout = Layout()
        layout.split_column(
            Layout(Panel(header, padding=(0, 1)), size=3),
            Layout(table, ratio=3),
            Layout(Panel(timeline_text, title="Step Timeline", padding=(0, 1)),
                   size=min(16, len(timeline_lines) + 3)),
        )
        return layout
