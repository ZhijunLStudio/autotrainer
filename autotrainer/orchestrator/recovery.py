"""Recovery manager — handles crash recovery and checkpoint resumption.

Provides WAL-style state persistence so that any point of failure
can be recovered from by reading the last known good state.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from autotrainer.utils.file_utils import atomic_write_json, safe_read_json


@dataclass
class RecoveryState:
    """Serializable recovery state persisted to disk."""

    task_name: str = ""
    experiment_id: str = ""
    current_phase: int = 0
    phase_states: dict = field(default_factory=dict)
    gpu_ids: list[int] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    last_training_step: int = 0
    last_config_path: str = ""
    data_path: str = ""
    eval_data_path: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RecoveryState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RecoveryManager:
    """Handles crash recovery and state restoration.

    State is persisted in two places:
    1. recovery_state.json — main state file (atomic writes)
    2. experiment_index.json — experiment records (separate for readability)
    """

    STATE_FILE = "recovery_state.json"

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.state_file = os.path.join(work_dir, self.STATE_FILE)
        os.makedirs(work_dir, exist_ok=True)

    def save(self, state: RecoveryState):
        """Save recovery state atomically."""
        state.updated_at = datetime.now().isoformat()
        if not state.created_at:
            state.created_at = state.updated_at
        atomic_write_json(self.state_file, state.to_dict())

    def load(self) -> RecoveryState | None:
        """Load recovery state. Returns None if no state exists."""
        data = safe_read_json(self.state_file)
        if not data:
            return None
        try:
            return RecoveryState.from_dict(data)
        except Exception:
            return None

    def exists(self) -> bool:
        """Check if a recovery state file exists."""
        return os.path.exists(self.state_file)

    def clear(self):
        """Remove the recovery state file (after successful completion)."""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        lock_file = self.state_file + ".lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

    def find_last_checkpoint(self, output_dir: str) -> str | None:
        """Find the most recent PaddleFormers checkpoint in an output directory.

        Looks for checkpoint-N directories and returns the path with the highest N.
        """
        p = Path(output_dir)
        if not p.exists():
            return None

        checkpoints = []
        for d in p.iterdir():
            if d.is_dir() and d.name.startswith("checkpoint-"):
                try:
                    step = int(d.name.split("-")[1])
                    checkpoints.append((step, str(d)))
                except (IndexError, ValueError):
                    continue

        if not checkpoints:
            return None

        checkpoints.sort(reverse=True)
        return checkpoints[0][1]

    def cleanup_stale_processes(self):
        """Clean up any stale training processes from a previous run.

        Looks for orphaned processes that might still be running.
        """
        import signal

        pids_file = os.path.join(self.work_dir, ".active_pids")
        if not os.path.exists(pids_file):
            return

        with open(pids_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pid = int(line)
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass

        os.remove(pids_file)

    def register_pid(self, pid: int):
        """Register a process PID for cleanup on recovery."""
        pids_file = os.path.join(self.work_dir, ".active_pids")
        with open(pids_file, "a") as f:
            f.write(f"{pid}\n")

    def get_experiment_index_path(self) -> str:
        """Path to the experiment index file."""
        return os.path.join(self.work_dir, "experiment_index.json")

    def load_experiment_index(self) -> dict:
        """Load the experiment index."""
        return safe_read_json(self.get_experiment_index_path()) or {"experiments": [], "rankings": []}

    def save_experiment_index(self, index: dict):
        """Save the experiment index atomically."""
        atomic_write_json(self.get_experiment_index_path(), index)

    def add_experiment(self, experiment: dict):
        """Add an experiment record to the index."""
        index = self.load_experiment_index()

        # Update if exists, append if not
        existing_ids = {e["id"] for e in index.get("experiments", [])}
        exp_id = experiment.get("id", "")
        if exp_id in existing_ids:
            for i, e in enumerate(index["experiments"]):
                if e["id"] == exp_id:
                    index["experiments"][i] = experiment
                    break
        else:
            index["experiments"].append(experiment)

        self.save_experiment_index(index)

    def update_experiment_status(self, exp_id: str, status: str, result: dict | None = None):
        """Update an experiment's status and result."""
        index = self.load_experiment_index()
        for exp in index.get("experiments", []):
            if exp.get("id") == exp_id:
                exp["status"] = status
                if result:
                    exp["result"] = result
                if status == "completed":
                    exp["completed_at"] = datetime.now().isoformat()
                break
        self.save_experiment_index(index)
