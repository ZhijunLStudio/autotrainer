"""TUI application — Textual-based terminal UI for autotrainer."""

from __future__ import annotations

import threading
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
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

    def __init__(self, task: str, gpu_ids: list[int], work_dir: str,
                 data_dir: str = "", data_path: str = "", eval_data_path: str = "",
                 skip_ablation: bool = False, resume: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._task = task
        self.gpu_ids = gpu_ids
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.data_path = data_path
        self.eval_data_path = eval_data_path
        self.skip_ablation = skip_ablation
        self.resume = resume

        # Components
        self.status_bar: StatusBar | None = None
        self.log_panel: LogPanel | None = None
        self.agent_panel: AgentPanel | None = None
        self.main_panel: Static | None = None

        # Pipeline
        self._pipeline_thread: threading.Thread | None = None
        self._pipeline_running = False

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

        yield Input(placeholder="Press Enter to start training...", id="input-bar")
        yield Footer()

    def on_mount(self):
        """Initialize the app, then auto-start the training pipeline."""
        self.status_bar.update(
            f"[b]Phase:[/b] INIT  |  [b]Task:[/b] {self._task}  |  [b]GPUs:[/b] {','.join(str(g) for g in self.gpu_ids)}"
        )
        self.main_panel.update(
            f"\n  [bold]AutoTrainer v0.1.0[/bold]\n"
            f"  Task: {self._task}\n"
            f"  GPUs: {self.gpu_ids}\n"
            f"  Work dir: {self.work_dir}\n\n"
            f"  Starting pipeline..."
        )
        self.agent_panel.update("[dim]Agent ready[/dim]")

        # Auto-start pipeline in background thread
        self._start_pipeline()

    def _start_pipeline(self):
        """Start the training pipeline in a background thread."""
        if self._pipeline_running:
            return
        self._pipeline_running = True
        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline, daemon=True, name="pipeline"
        )
        self._pipeline_thread.start()

    def _run_pipeline(self):
        """Run the training pipeline, pushing updates to the TUI."""
        from autotrainer.config import AutoTrainerConfig
        from autotrainer.orchestrator.pipeline_v2 import PipelineOrchestratorV2

        cfg = AutoTrainerConfig.from_env()

        orch = PipelineOrchestratorV2(
            config=cfg,
            task=self._task,
            gpu_ids=self.gpu_ids,
            resume=self.resume,
            data_dir=self.data_dir,
            skip_ablation=self.skip_ablation,
        )

        # Wire callbacks
        orch.ctx.on_phase_change = self._on_phase_change
        orch.ctx.on_user_confirm = self._on_confirm
        orch.ctx.on_user_input = self._on_input

        if self.data_path:
            orch.ctx.data_path = self.data_path
        if self.eval_data_path:
            orch.ctx.eval_data_path = self.eval_data_path

        try:
            orch.run()
            self.call_from_thread(self._on_complete)
        except Exception as e:
            self.call_from_thread(lambda: self._on_error(str(e)))

    def _on_phase_change(self, phase: str, msg: str):
        """Called from pipeline thread when phase changes."""
        self.call_from_thread(lambda: self.update_phase(phase, msg))
        if self.log_panel:
            self.call_from_thread(lambda: self.log_panel.append_line(f"[{phase}] {msg}"))

    def _on_confirm(self, message: str, context: dict | None = None) -> bool:
        """Called from pipeline thread for user confirmation. Auto-confirms."""
        self.call_from_thread(lambda: self.show_confirm_prompt(message))
        return True  # Auto-confirm in TUI mode

    def _on_input(self, prompt: str, choices: list[str]) -> str:
        """Called from pipeline thread for user input. Returns first choice."""
        return choices[0] if choices else ""

    def _on_complete(self):
        """Called when pipeline completes successfully."""
        self.main_panel.update(
            f"\n  [bold green]Training Complete![/bold green]\n\n"
            f"  Check {self.work_dir} for results."
        )
        self.status_bar.update("[b]Phase:[/b] COMPLETED")
        self.agent_panel.update("[bold green]All phases completed![/bold green]")

    def _on_error(self, error: str):
        """Called when pipeline fails."""
        self.main_panel.update(f"\n  [bold red]Training Failed[/bold red]\n\n  {error}")
        self.status_bar.update("[b]Phase:[/b] FAILED")
        self.agent_panel.update(f"[bold red]Error:[/bold red] {error}")

    def on_input_submitted(self, event: Input.Submitted):
        """Handle user input from the input bar."""
        value = event.value.strip().lower()
        if value in ("q", "quit", "exit"):
            self.exit()
        elif value in ("y", "yes"):
            self.agent_panel.update("[dim]Confirmed.[/dim]")
        elif value in ("n", "no"):
            self.agent_panel.update("[dim]Declined.[/dim]")
        elif value == "s":
            self.action_show_status()
        elif value:
            self.agent_panel.update(f"[dim]Unknown command: {value}[/dim]")

    def action_toggle_agent_panel(self):
        if self.agent_panel:
            self.agent_panel.display = not self.agent_panel.display

    def action_toggle_log_mode(self):
        if self.log_panel:
            self.log_panel.toggle_mode()

    def action_show_status(self):
        from autotrainer.config import AutoTrainerConfig
        cfg = AutoTrainerConfig.from_env()
        self.main_panel.update(
            f"\n  [bold]Status[/bold]\n"
            f"  Task: {self._task}\n"
            f"  GPUs: {self.gpu_ids}\n"
            f"  Work dir: {self.work_dir}\n"
            f"  PaddleFormers: {cfg.paddleformers_root or 'not detected'}\n"
            f"  GPU count: {cfg.detect_gpu_count()}\n"
        )

    def update_phase(self, phase: str, message: str = ""):
        self.status_bar.update(
            f"[b]Phase:[/b] {phase}  |  [b]Task:[/b] {self._task}  |  [b]GPUs:[/b] {','.join(str(g) for g in self.gpu_ids)}"
        )
        if message and self.main_panel:
            self.main_panel.update(f"\n  [bold]{phase}[/bold]\n\n  {message}")

    def add_log_line(self, line: str):
        if self.log_panel:
            self.log_panel.append_line(line)

    def set_agent_message(self, message: str):
        if self.agent_panel:
            self.agent_panel.update(message)

    def update_experiment_status(self, scheduler_status):
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
            lines.extend(["", f"  [yellow]Running:[/yellow] {s.current.id}"])
        if self.main_panel:
            self.main_panel.update("\n".join(lines))

    def show_confirm_prompt(self, message: str, callback: Any = None):
        self.agent_panel.update(f"\n  [yellow bold]?[/yellow bold] {message}\n\n  Type y/n in input bar.")


def run_tui(task: str, gpu_ids: list[int], work_dir: str,
            data_dir: str = "", data_path: str = "", eval_data_path: str = "",
            skip_ablation: bool = False, resume: bool = False):
    """Launch the TUI application."""
    app = AutoTrainerApp(
        task=task, gpu_ids=gpu_ids, work_dir=work_dir,
        data_dir=data_dir, data_path=data_path, eval_data_path=eval_data_path,
        skip_ablation=skip_ablation, resume=resume,
    )
    app.run()
