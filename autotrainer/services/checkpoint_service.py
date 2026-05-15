"""CheckpointService — checkpoint discovery, cleanup, and disk management."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from autotrainer.core.store import PipelineStore


class CheckpointService:
    """Manage training checkpoints: find latest, cleanup ablation, keep best."""

    def __init__(self, store: PipelineStore, run_id: str, work_dir: str):
        self._store = store
        self._run_id = run_id
        self._work_dir = work_dir

    def find_latest(self, output_dir: str) -> str | None:
        """Find the most recent checkpoint-N directory."""
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

    def cleanup_phase_checkpoints(self, phase: str):
        """Remove checkpoint dirs for completed experiments in a phase. Keeps result.json and config.yaml."""
        experiments = self._store.get_experiments_by_phase(self._run_id, phase)
        for exp in experiments:
            if exp.get("status") != "completed":
                continue
            cp_path = exp.get("result", {}).get("checkpoint_path") or exp.get("checkpoint_path", "")
            if not cp_path or not os.path.isdir(cp_path):
                continue
            for item in os.listdir(cp_path):
                item_path = os.path.join(cp_path, item)
                if item.startswith("checkpoint-") and os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    def cleanup_full_training(self, keep_best: int = 1, keep_last: int = 1):
        """Keep only best + last checkpoints for full training."""
        ckpt_dir = os.path.join(self._work_dir, "checkpoints", "full-training")
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
        keep = {c[1] for c in checkpoints[-keep_last:]}
        for _, dirname in checkpoints:
            if dirname not in keep:
                shutil.rmtree(os.path.join(ckpt_dir, dirname))

    def track_checkpoint(self, experiment_id: str, path: str, step: int = 0, loss: float | None = None):
        self._store.add_checkpoint(self._run_id, experiment_id, path, step, loss)

    def get_checkpoints(self, experiment_id: str) -> list[dict]:
        return self._store.get_checkpoints(self._run_id, experiment_id)

    def get_latest_checkpoint(self, experiment_id: str) -> dict | None:
        return self._store.get_latest_checkpoint(self._run_id, experiment_id)
