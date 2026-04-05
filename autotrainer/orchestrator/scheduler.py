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

    def run_next(
        self,
        train_manager,
        gpu_ids: list[int] | None = None,
    ):
        """Run the next pending experiment. Returns TrainingResult or None."""
        exp = self.get_next_pending()
        if exp is None:
            return None

        self.mark_running(exp.id)

        # Find resume checkpoint
        resume_from = self._find_resume_checkpoint(exp)

        # Extract paths from config
        subset_path = exp.config.get("data", {}).get("train_dataset_path", "")
        max_steps = exp.config.get("finetuning", {}).get("max_steps", 1000)

        try:
            result = train_manager.run_single_ablation(
                base_config=exp.config,
                factor_changes=exp.config_diff,
                subset_path=subset_path,
                max_steps=max_steps,
                experiment_id=exp.id,
                gpu_ids=gpu_ids,
                resume_from_checkpoint=resume_from,
            )
        except Exception as e:
            self.mark_failed(exp.id, {"error": str(e)})
            return None

        if result.status == "completed":
            self.mark_completed(exp.id, result.to_dict(), result.checkpoint_path)
        else:
            self.mark_failed(exp.id, result.to_dict())

        return result

    def run_all(
        self,
        train_manager,
        gpu_ids: list[int] | None = None,
        on_progress=None,
    ) -> list:
        """Run all pending experiments sequentially."""
        results = []
        while True:
            result = self.run_next(train_manager, gpu_ids)
            if result is None:
                break
            results.append(result)
            if on_progress:
                on_progress(self.get_status(), result)
        return results

    def _find_resume_checkpoint(self, exp: ExperimentSpec) -> str | None:
        """Find the latest checkpoint-N in exp's output dir."""
        output_dir = exp.config.get("finetuning", {}).get("output_dir", "")
        if not output_dir or not os.path.exists(output_dir):
            return None
        checkpoints = []
        for d in os.listdir(output_dir):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, os.path.join(output_dir, d)))
                except (IndexError, ValueError):
                    pass
        if not checkpoints:
            return None
        checkpoints.sort(reverse=True)
        return checkpoints[0][1]

    # ── Checkpoint Garbage Collection ──

    def cleanup_phase_checkpoints(self, phase: str):
        """Remove checkpoint-N dirs for completed experiments in a phase.
        Keeps result.json and config.yaml."""
        for exp in self._experiments:
            if exp.phase != phase or exp.status != "completed":
                continue
            cp_path = exp.checkpoint_path
            if not cp_path or not os.path.isdir(cp_path):
                continue
            for item in os.listdir(cp_path):
                item_path = os.path.join(cp_path, item)
                if item.startswith("checkpoint-") and os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    def cleanup_full_training(self, keep_best: int = 1, keep_last: int = 1):
        """Keep only best + last checkpoints for full training."""
        ckpt_dir = os.path.join(self.work_dir, "checkpoints", "full-training")
        if not os.path.exists(ckpt_dir):
            return

        checkpoints = []
        for d in os.listdir(ckpt_dir):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, d))
                except (IndexError, ValueError):
                    pass

        if len(checkpoints) <= keep_best + keep_last:
            return

        checkpoints.sort()

        # Last ones
        keep = {c[1] for c in checkpoints[-keep_last:]}

        # Best ones — read from result.json
        best_ones = self._find_best_checkpoints(ckpt_dir, keep_best)
        keep |= best_ones

        for _, dirname in checkpoints:
            if dirname not in keep:
                shutil.rmtree(os.path.join(ckpt_dir, dirname))

    def _find_best_checkpoints(self, ckpt_dir: str, count: int) -> set[str]:
        """Find the best checkpoints by eval_loss from result.json."""
        result_path = os.path.join(ckpt_dir, "result.json")
        if not os.path.exists(result_path):
            return set()

        data = safe_read_json(result_path) or {}
        eval_losses = data.get("eval_losses_by_step", {})
        if not eval_losses:
            return set()

        sorted_steps = sorted(eval_losses.items(), key=lambda x: x[1])
        best_steps = {int(s) for s, _ in sorted_steps[:count]}

        best_dirs = set()
        for d in os.listdir(ckpt_dir):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    if step in best_steps:
                        best_dirs.add(d)
                except (IndexError, ValueError):
                    pass
        return best_dirs

    def clear(self):
        """Clear all experiments from the queue."""
        self._experiments.clear()
        if os.path.exists(self._queue_file):
            os.remove(self._queue_file)
