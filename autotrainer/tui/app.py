"""TUI application — Textual-based terminal UI for autotrainer."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static

from autotrainer.tui.widgets.status_bar import StatusBar
from autotrainer.tui.widgets.log_panel import LogPanel
from autotrainer.tui.widgets.agent_panel import AgentPanel


CSS = """
Screen {
    layout: grid;
    grid-size: 2 3;
    grid-rows: 3 1fr 8;
    grid-columns: 3fr 2fr;
}

#status-bar {
    column-span: 2;
    height: 3;
    dock: top;
    background: $primary;
    color: $text;
}

#main-panel {
    row-span: 1;
    overflow-y: scroll;
    padding: 0 1;
}

#agent-panel {
    row-span: 1;
    overflow-y: scroll;
    border-left: solid $accent;
    padding: 0 1;
}

#log-panel {
    column-span: 2;
    height: 8;
    overflow-y: scroll;
    background: $surface;
    padding: 0 1;
    border-top: solid $accent;
}

#input-bar {
    column-span: 2;
    height: 3;
    dock: bottom;
    background: $surface;
    padding: 0 1;
}

.collapsible-header {
    background: $primary-darken-2;
    padding: 0 1;
    height: 1;
}

.panel-content {
    padding: 0 1;
}
"""


class AutoTrainerApp(App):
    """Main TUI application for the automated training system."""

    CSS = CSS
    TITLE = "AutoTrainer"
    BINDINGS = [
        Binding("tab", "toggle_agent_panel", "Toggle Agent", priority=True),
        Binding("l", "toggle_log_mode", "Log Mode", priority=True),
        Binding("s", "show_status", "Status"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, task: str, gpu_ids: list[int], work_dir: str, resume: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.gpu_ids = gpu_ids
        self.work_dir = work_dir
        self.resume = resume

        # Components (initialized on mount)
        self.status_bar: StatusBar | None = None
        self.log_panel: LogPanel | None = None
        self.agent_panel: AgentPanel | None = None
        self.main_panel: Static | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        self.status_bar = StatusBar(id="status-bar")
        yield self.status_bar

        self.main_panel = Static(id="main-panel")
        yield self.main_panel

        self.agent_panel = AgentPanel(id="agent-panel")
        yield self.agent_panel

        self.log_panel = LogPanel(id="log-panel")
        yield self.log_panel

        yield Input(placeholder="Press Enter to confirm, or type a command...", id="input-bar")
        yield Footer()

    def on_mount(self):
        """Initialize the app on mount."""
        self.status_bar.update(
            f"[b]Phase:[/b] INIT  |  [b]Task:[/b] {self.task}  |  [b]GPUs:[/b] {','.join(str(g) for g in self.gpu_ids)}"
        )
        self.main_panel.update(
            f"\n  [bold]AutoTrainer v0.1.0[/bold]\n"
            f"  Task: {self.task}\n"
            f"  GPUs: {self.gpu_ids}\n"
            f"  Work dir: {self.work_dir}\n\n"
            f"  Initializing pipeline..."
        )
        self.agent_panel.update("[dim]Agent waiting...[/dim]")

    def action_toggle_agent_panel(self):
        """Toggle agent panel visibility."""
        if self.agent_panel:
            self.agent_panel.visible = not self.agent_panel.visible

    def action_toggle_log_mode(self):
        """Toggle between smart and full log modes."""
        if self.log_panel:
            self.log_panel.toggle_mode()

    def action_show_status(self):
        """Show status info in the main panel."""
        from autotrainer.config import AutoTrainerConfig

        cfg = AutoTrainerConfig.from_env()
        self.main_panel.update(
            f"\n  [bold]Status[/bold]\n"
            f"  Task: {self.task}\n"
            f"  GPUs: {self.gpu_ids}\n"
            f"  Work dir: {self.work_dir}\n"
            f"  PaddleFormers: {cfg.paddleformers_root or 'not detected'}\n"
            f"  GPU count: {cfg.detect_gpu_count()}\n"
        )

    def update_phase(self, phase: str, message: str = ""):
        """Update the current phase display."""
        self.status_bar.update(
            f"[b]Phase:[/b] {phase}  |  [b]Task:[/b] {self.task}  |  [b]GPUs:[/b] {','.join(str(g) for g in self.gpu_ids)}"
        )
        if message:
            self.main_panel.update(f"\n  [bold]{phase}[/bold]\n\n  {message}")

    def add_log_line(self, line: str):
        """Add a log line to the log panel."""
        if self.log_panel:
            self.log_panel.append_line(line)

    def set_agent_message(self, message: str):
        """Set a message in the agent panel."""
        if self.agent_panel:
            self.agent_panel.update(message)

    def update_experiment_status(self, scheduler_status):
        """Update main panel with experiment queue status."""
        s = scheduler_status
        lines = [
            f"\n  [bold]Experiment Queue[/bold]  ({s.completed}/{s.total} completed, {s.failed} failed)",
            "",
        ]
        for phase_name, breakdown in s.phase_breakdown.items():
            status_str = f"{breakdown['done']}/{breakdown['total']}"
            if breakdown.get("failed", 0) > 0:
                status_str += f" ({breakdown['failed']} failed)"
            lines.append(f"  [bold]{phase_name}:[/bold] {status_str}")

        if s.current:
            lines.extend([
                "",
                f"  [yellow]Running:[/yellow] {s.current.id}",
            ])

        if self.main_panel:
            self.main_panel.update("\n".join(lines))

    def show_confirm_prompt(self, message: str, callback: Any = None):
        """Show a confirmation prompt to the user."""
        self.agent_panel.update(f"\n  [yellow bold]?[/yellow bold] {message}\n\n  Press [y]es or [n]o in the input bar.")


def run_tui(task: str, gpu_ids: list[int], work_dir: str, resume: bool = False):
    """Launch the TUI application."""
    app = AutoTrainerApp(task=task, gpu_ids=gpu_ids, work_dir=work_dir, resume=resume)
    app.run()
