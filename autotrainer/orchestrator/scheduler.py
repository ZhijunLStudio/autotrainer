"""Experiment scheduler — manages a persistent queue of training experiments.

Supports:
- Adding experiments in batches
- Resuming from where we left off (experiment-level granularity)
- Getting overall progress status
- Checkpoint garbage collection
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable

from autotrainer.utils.file_utils import atomic_write_json, safe_read_json


QUEUE_FILE = "experiment_queue.json"


@dataclass
class ExperimentSpec:
    """Specification and state for a single experiment."""

    id: str
    phase: str                    # "ablation_hyperparams" | "ablation_ratio" | "full_training"
    config: dict                  # Full PaddleFormers config
    config_diff: dict             # Changes relative to base
    status: str = "pending"       # pending | running | completed | failed
    result: dict | None = None
    checkpoint_path: str = ""
    created_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentSpec:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SchedulerStatus:
    """Overall scheduler progress."""

    total: int
    completed: int
    failed: int
    pending: int
    current: ExperimentSpec | None
    phase_breakdown: dict         # {"ablation_hyperparams": {"total":5, "done":3}, ...}


class ExperimentScheduler:
    """Manages a persistent queue of experiments with resume support."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self._queue_file = os.path.join(work_dir, QUEUE_FILE)
        self._experiments: list[ExperimentSpec] = []
        self.loaded = False
        self._load()

    def _load(self):
        """Load queue from disk if it exists."""
        data = safe_read_json(self._queue_file)
        if data and "experiments" in data:
            self._experiments = [ExperimentSpec.from_dict(e) for e in data["experiments"]]
            self.loaded = True

    def _save(self):
        """Persist queue to disk atomically."""
        os.makedirs(self.work_dir, exist_ok=True)
        data = {
            "experiments": [e.to_dict() for e in self._experiments],
            "updated_at": datetime.now().isoformat(),
        }
        atomic_write_json(self._queue_file, data)

    def add_experiments(self, experiments: list[ExperimentSpec]):
        """Add experiments to the queue. Preserves existing completed/running state."""
        existing = {e.id: e for e in self._experiments}
        for exp in experiments:
            if exp.id in existing:
                continue
            if not exp.created_at:
                exp.created_at = datetime.now().isoformat()
            self._experiments.append(exp)
        self._save()

    def get_next_pending(self) -> ExperimentSpec | None:
        """Get the next pending experiment, or None if all done."""
        for exp in self._experiments:
            if exp.status == "pending":
                return exp
        return None

    def mark_running(self, exp_id: str):
        """Mark an experiment as running."""
        for exp in self._experiments:
            if exp.id == exp_id:
                exp.status = "running"
                break
        self._save()

    def mark_completed(self, exp_id: str, result: dict, checkpoint_path: str = ""):
        """Mark an experiment as completed with result."""
        for exp in self._experiments:
            if exp.id == exp_id:
                exp.status = "completed"
                exp.result = result
                exp.checkpoint_path = checkpoint_path
                exp.completed_at = datetime.now().isoformat()
                break
        self._save()

    def mark_failed(self, exp_id: str, result: dict | None = None):
        """Mark an experiment as failed."""
        for exp in self._experiments:
            if exp.id == exp_id:
                exp.status = "failed"
                exp.result = result
                exp.completed_at = datetime.now().isoformat()
                break
        self._save()

    def get_status(self) -> SchedulerStatus:
        """Get overall scheduler progress."""
        total = len(self._experiments)
        completed = sum(1 for e in self._experiments if e.status == "completed")
        failed = sum(1 for e in self._experiments if e.status == "failed")
        pending = sum(1 for e in self._experiments if e.status == "pending")

        current = None
        for e in self._experiments:
            if e.status == "running":
                current = e
                break

        phase_breakdown: dict[str, dict] = {}
        for exp in self._experiments:
            if exp.phase not in phase_breakdown:
                phase_breakdown[exp.phase] = {"total": 0, "done": 0, "failed": 0}
            phase_breakdown[exp.phase]["total"] += 1
            if exp.status == "completed":
                phase_breakdown[exp.phase]["done"] += 1
            elif exp.status == "failed":
                phase_breakdown[exp.phase]["failed"] += 1

        return SchedulerStatus(
            total=total,
            completed=completed,
            failed=failed,
            pending=pending,
            current=current,
            phase_breakdown=phase_breakdown,
        )

    def get_experiments_by_phase(self, phase: str) -> list[ExperimentSpec]:
        """Get all experiments for a given phase."""
        return [e for e in self._experiments if e.phase == phase]

    def clear(self):
        """Clear all experiments from the queue."""
        self._experiments.clear()
        if os.path.exists(self._queue_file):
            os.remove(self._queue_file)
