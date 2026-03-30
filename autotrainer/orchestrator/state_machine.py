"""Phase state machine — drives the training pipeline through phases.

This is the central orchestration component. All state transitions are
persisted to disk before execution (WAL-style) for crash recovery.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable

from autotrainer.utils.file_utils import atomic_write_json, safe_read_json


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
    """Status of a single phase."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseManager:
    """State machine that drives the training pipeline through phases.

    Persists state to disk on every transition for crash recovery.
    """

    def __init__(self, state_file: str, task_name: str = ""):
        self.state_file = state_file
        self.task_name = task_name
        self.current = Phase.TASK_CONFIRM
        self.phase_states: dict[int, dict] = {}
        self.metadata: dict[str, Any] = {}
        self._on_transition: list[Callable[[Phase, Phase], None]] = []

    def on_transition(self, callback: Callable[[Phase, Phase], None]):
        """Register a callback for phase transitions."""
        self._on_transition.append(callback)

    def can_transition_to(self, target: Phase) -> bool:
        """Check if a transition to target phase is valid."""
        # Can always go to completed
        if target == Phase.COMPLETED:
            return True
        # Can only advance one phase at a time
        if target.value == self.current.value + 1:
            return True
        # Can go back for retry
        if target.value < self.current.value:
            return True
        return False

    def transition_to(self, target: Phase, metadata: dict[str, Any] | None = None) -> bool:
        """Transition to a new phase. Returns True if successful.

        Persists state BEFORE transitioning (WAL-style).
        """
        if not self.can_transition_to(target):
            return False

        # Mark current phase completed
        self._set_phase_status(self.current, PhaseStatus.COMPLETED)

        # Mark target as in_progress
        self._set_phase_status(target, PhaseStatus.IN_PROGRESS, metadata)

        # Persist BEFORE executing
        old_phase = self.current
        self.current = target
        self.save_state()

        # Notify callbacks
        for cb in self._on_transition:
            try:
                cb(old_phase, target)
            except Exception:
                pass

        return True

    def mark_failed(self, phase: Phase, error: str):
        """Mark a phase as failed with an error message."""
        self._set_phase_status(phase, PhaseStatus.FAILED, {"error": error})
        self.save_state()

    def mark_skipped(self, phase: Phase):
        """Mark a phase as skipped."""
        self._set_phase_status(phase, PhaseStatus.SKIPPED)
        self.save_state()

    def get_phase_state(self, phase: Phase) -> dict:
        """Get the state dict for a phase."""
        return self.phase_states.get(phase.value, {"status": "pending"})

    def is_completed(self, phase: Phase) -> bool:
        """Check if a phase has been completed."""
        state = self.get_phase_state(phase)
        return state.get("status") == "completed"

    def get_progress(self) -> dict:
        """Get overall pipeline progress."""
        total = len(Phase) - 1  # Exclude COMPLETED
        completed = sum(1 for p in Phase if p != Phase.COMPLETED and self.is_completed(p))
        return {
            "current_phase": self.current.name,
            "current_phase_index": self.current.value,
            "completed_phases": completed,
            "total_phases": total,
            "progress_pct": round(completed / total * 100, 1) if total > 0 else 0,
        }

    def get_pending_phases(self) -> list[Phase]:
        """Get phases that haven't been completed yet."""
        return [p for p in Phase if p != Phase.COMPLETED and not self.is_completed(p)]

    def get_failed_phases(self) -> list[Phase]:
        """Get phases that failed."""
        return [
            p
            for p in Phase
            if p != Phase.COMPLETED and self.get_phase_state(p).get("status") == "failed"
        ]

    def save_state(self):
        """Persist state to disk atomically."""
        state = {
            "task_name": self.task_name,
            "current_phase": self.current.value,
            "phase_states": {str(k): v for k, v in self.phase_states.items()},
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat(),
        }
        atomic_write_json(self.state_file, state)

    def load_state(self) -> bool:
        """Load state from disk. Returns True if state was loaded."""
        data = safe_read_json(self.state_file)
        if not data:
            return False

        try:
            self.task_name = data.get("task_name", "")
            self.current = Phase(data["current_phase"])
            self.phase_states = {int(k): v for k, v in data.get("phase_states", {}).items()}
            self.metadata = data.get("metadata", {})
            return True
        except (KeyError, ValueError):
            return False

    def _set_phase_status(self, phase: Phase, status: PhaseStatus, metadata: dict | None = None):
        """Set the status of a phase."""
        state = self.phase_states.get(phase.value, {})
        state["status"] = status.value

        if status == PhaseStatus.IN_PROGRESS:
            state["started_at"] = datetime.now().isoformat()
        elif status in (PhaseStatus.COMPLETED, PhaseStatus.FAILED):
            state["completed_at"] = datetime.now().isoformat()

        if metadata:
            state.update(metadata)

        self.phase_states[phase.value] = state
