"""Core interfaces — all protocol/ABC definitions for the autotrainer system.

Every major component depends on these interfaces, not on concrete implementations.
This makes testing, extension, and replacement straightforward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ── Phase types ──────────────────────────────────────────────────────────────


class Phase(Enum):
    """Training pipeline phases."""
    TASK_CONFIRM = 0
    DATA_PREPARE = 1
    ENV_CHECK = 2
    ABLATION = 3
    FULL_TRAINING = 4
    EVALUATION = 5
    REPORT = 6
    COMPLETED = 7


class PhaseStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ── Callback types ───────────────────────────────────────────────────────────

OnPhaseChange = Callable[[str, str], None]
OnLogLine = Callable[[str], None]
OnMetrics = Callable[..., None]  # LogMetrics from pf_integration
OnHealthUpdate = Callable[..., None]  # HealthStatus from orchestrator
OnUserConfirm = Callable[[str, dict | None], bool]
OnUserInput = Callable[[str, list[str]], str]


# ── Pipeline context (passed through all phases) ─────────────────────────────


@dataclass
class PipelineContext:
    """Mutable context object passed through every phase.

    Each phase reads/writes the fields it needs. The orchestrator
    injects this into every PhaseHandler.execute() call.
    """

    task: str = ""
    goal: str = ""
    gpu_ids: list[int] = field(default_factory=list)
    work_dir: str = ""

    # Data
    data_path: str = ""
    eval_data_path: str = ""
    data_dir: str = ""
    data_profile: dict = field(default_factory=dict)
    multi_dataset_info: list[dict] = field(default_factory=list)

    # Ablation
    ablation_config: dict = field(default_factory=dict)
    ablation_results: list[dict] = field(default_factory=list)
    best_ablation_config: dict = field(default_factory=dict)

    # Training
    full_training_result: dict = field(default_factory=dict)
    eval_result: dict = field(default_factory=dict)

    # Callbacks (set by runner before pipeline starts)
    on_phase_change: OnPhaseChange | None = None
    on_log_line: OnLogLine | None = None
    on_metrics: OnMetrics | None = None
    on_health: OnHealthUpdate | None = None
    on_user_confirm: OnUserConfirm | None = None
    on_user_input: OnUserInput | None = None

    # ── Convenience methods (use callbacks under the hood) ──

    def notify(self, phase: str, msg: str):
        """Notify phase change via callback."""
        if self.on_phase_change:
            self.on_phase_change(phase, msg)

    def confirm(self, message: str, context: dict | None = None) -> bool:
        """Request user confirmation. Default: auto-confirm."""
        if self.on_user_confirm:
            return self.on_user_confirm(message, context or {})
        return True

    def prompt(self, prompt: str, choices: list[str]) -> str:
        """Request user input. Default: first choice."""
        if self.on_user_input:
            return self.on_user_input(prompt, choices)
        return choices[0] if choices else ""


# ── Phase result ─────────────────────────────────────────────────────────────


@dataclass
class PhaseResult:
    """Result returned by every PhaseHandler.execute()."""

    phase: Phase
    status: PhaseStatus
    message: str = ""
    data: dict = field(default_factory=dict)
    next_phase: Phase | None = None  # None = default linear progression


# ── Phase handler interface ──────────────────────────────────────────────────


class PhaseHandler(ABC):
    """Interface for all pipeline phase handlers.

    Each handler is stateless (all state lives in ctx and Store).
    """

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PhaseResult:
        """Execute this phase. Read/write ctx and Store as needed."""
        ...


# ── Task plugin interface ────────────────────────────────────────────────────


@dataclass
class TaskSpec:
    """Declares a trainable task for the registry."""

    name: str
    description: str = ""
    # Hyperparameter space for ablation
    hyperparam_space: dict = field(default_factory=dict)
    # Format requirements
    supported_data_formats: list[str] = field(default_factory=list)
    # Launcher command template
    launch_command: str = ""
    launch_env: dict = field(default_factory=dict)


# ── Recording types (Store read/write) ───────────────────────────────────────


@dataclass
class ExperimentRecord:
    """An experiment stored in the database."""

    id: str
    phase: str
    status: str  # pending, running, completed, failed
    config_diff: dict = field(default_factory=dict)
    config_path: str = ""
    result: dict = field(default_factory=dict)
    created_at: str = ""
    completed_at: str = ""


@dataclass
class CheckpointRecord:
    """A training checkpoint tracked in the database."""

    id: int = 0
    experiment_id: str = ""
    path: str = ""
    step: int = 0
    loss: float | None = None
    created_at: str = ""
