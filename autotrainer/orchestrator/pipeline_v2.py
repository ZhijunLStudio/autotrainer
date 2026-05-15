"""Pipeline orchestrator V2 — thin dispatch loop over PhaseHandlers.

Replaces the 887-line PipelineOrchestrator with a ~60-line orchestrator
that delegates all phase logic to independent PhaseHandler classes.

State persistence: PipelineStore (SQLite) — single source of truth.
"""

from __future__ import annotations

import os
from datetime import datetime

from autotrainer.config import AutoTrainerConfig
from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler
from autotrainer.core.store import PipelineStore
from autotrainer.core.registry import TaskRegistry
from autotrainer.core.legacy_importer import migrate_to_store
from autotrainer.phases import (
    TaskConfirmHandler,
    DataPrepareHandler,
    EnvCheckHandler,
    AblationHandler,
    FullTrainingHandler,
    EvaluationHandler,
    ReportHandler,
)

# Phase execution order
PIPELINE_PHASES = [
    Phase.TASK_CONFIRM,
    Phase.DATA_PREPARE,
    Phase.ENV_CHECK,
    Phase.ABLATION,
    Phase.FULL_TRAINING,
    Phase.EVALUATION,
    Phase.REPORT,
]


class PipelineOrchestratorV2:
    """Thin orchestrator: creates context, instantiates handlers, runs the loop."""

    def __init__(self, config: AutoTrainerConfig, task: str = "paddleocr-vl",
                 gpu_ids: list[int] | None = None, resume: bool = False,
                 data_dir: str = "", skip_ablation: bool = False):
        self.config = config
        self.task = task
        gpu_ids = gpu_ids or list(range(config.detect_gpu_count()))
        self.work_dir = os.path.join(config.work_dir, f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(self.work_dir, exist_ok=True)

        # Unified storage
        store_path = os.path.join(self.work_dir, "pipeline.db")
        self.run_id = f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.store = PipelineStore(store_path)

        # Try legacy migration
        if resume:
            legacy_run_id = migrate_to_store(self.store, self.work_dir)
            if legacy_run_id:
                self.run_id = legacy_run_id

        # Create run
        self.store.create_run(
            run_id=self.run_id, task_name=task,
            gpu_ids=gpu_ids, data_path=data_dir,
        )

        # Shared context
        self.ctx = PipelineContext(
            task=task, gpu_ids=gpu_ids,
            work_dir=self.work_dir, data_dir=data_dir,
        )

        self.registry = TaskRegistry()
        self.skip_ablation = skip_ablation

        # Phase handlers
        pf_root = config.paddleformers_root
        self.handlers: dict[Phase, PhaseHandler] = {
            Phase.TASK_CONFIRM: TaskConfirmHandler(),
            Phase.DATA_PREPARE: DataPrepareHandler(),
            Phase.ENV_CHECK: EnvCheckHandler(paddleformers_root=pf_root),
            Phase.ABLATION: AblationHandler(paddleformers_root=pf_root, task_registry=self.registry),
            Phase.FULL_TRAINING: FullTrainingHandler(
                paddleformers_root=pf_root,
                health_poll_interval=config.health_poll_interval,
                health_hang_timeout=config.health_hang_timeout,
            ),
            Phase.EVALUATION: EvaluationHandler(paddleformers_root=pf_root),
            Phase.REPORT: ReportHandler(store=self.store),
        }

    def run(self):
        """Execute the full pipeline."""
        for phase in PIPELINE_PHASES:
            if phase == Phase.ABLATION and self.skip_ablation:
                self.store.update_run_phase(self.run_id, phase, PhaseStatus.SKIPPED, "User skipped ablation")
                continue

            # Check if already completed (resume)
            if self.store.is_phase_completed(self.run_id, phase):
                continue

            handler = self.handlers.get(phase)
            if handler is None:
                continue

            # Update store: phase starting
            self.store.update_run_phase(self.run_id, phase, PhaseStatus.IN_PROGRESS)

            try:
                result = handler.execute(self.ctx)

                if result.status == PhaseStatus.COMPLETED:
                    self.store.update_run_phase(self.run_id, phase, PhaseStatus.COMPLETED, result.message)
                elif result.status == PhaseStatus.FAILED:
                    self.store.fail_run(self.run_id, result.message)
                    raise RuntimeError(f"Phase {phase.name} failed: {result.message}")
                elif result.status == PhaseStatus.SKIPPED:
                    self.store.update_run_phase(self.run_id, phase, PhaseStatus.SKIPPED, result.message)

            except Exception as e:
                self.store.fail_run(self.run_id, str(e))
                raise

        self.store.complete_run(self.run_id)
        return self.store.get_full_snapshot(self.run_id)
