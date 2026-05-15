"""ExperimentService — single source of truth for experiment lifecycle management.

Replaces:
  - TrainManager._save_experiment_record()
  - RecoveryManager.add_experiment()
  - RecoveryManager.update_experiment_status()
  - RecoveryManager.load_experiment_index()
  - RecoveryManager.save_experiment_index()
"""

from __future__ import annotations

from datetime import datetime

from autotrainer.core.interfaces import ExperimentRecord
from autotrainer.core.store import PipelineStore


class ExperimentService:
    """Unified experiment CRUD. All experiment writes go through this service."""

    def __init__(self, store: PipelineStore, run_id: str):
        self._store = store
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def create(self, exp_id: str, phase: str, config_diff: dict | None = None,
               config_path: str = "", result: dict | None = None) -> ExperimentRecord:
        record = ExperimentRecord(
            id=exp_id,
            phase=phase,
            status="pending",
            config_diff=config_diff or {},
            config_path=config_path,
            result=result or {},
            created_at=datetime.now().isoformat(),
        )
        self._store.add_experiment(self._run_id, record)
        return record

    def mark_running(self, exp_id: str):
        self._store.update_experiment(self._run_id, exp_id, "running")

    def mark_completed(self, exp_id: str, result: dict):
        self._store.update_experiment(self._run_id, exp_id, "completed", result)

    def mark_failed(self, exp_id: str, result: dict | None = None):
        self._store.update_experiment(self._run_id, exp_id, "failed", result)

    def get(self, exp_id: str) -> dict | None:
        return self._store.get_experiment(self._run_id, exp_id)

    def list_by_phase(self, phase: str) -> list[dict]:
        return self._store.get_experiments_by_phase(self._run_id, phase)

    def list_all(self) -> list[dict]:
        return self._store.list_experiments(self._run_id)

    def get_pending(self, phase: str) -> list[dict]:
        exps = self.list_by_phase(phase)
        return [e for e in exps if e.get("status") == "pending"]

    def add_batch(self, experiments: list[ExperimentRecord]):
        for exp in experiments:
            self._store.add_experiment(self._run_id, exp)
