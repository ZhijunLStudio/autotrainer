"""Phase 4: Full Training — best config on full data with health watchdog.

Supports step-level checkpoint resume and auto-recovery on training hang.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from autotrainer.config import AutoTrainerConfig
from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler
from autotrainer.core.registry import TaskRegistry
from autotrainer.managers.train_manager import TrainManager
from autotrainer.orchestrator.health_monitor import HealthMonitor
from autotrainer.pf_integration.config_builder import ConfigBuilder
from autotrainer.skills.diagnose_training.handler import DiagnoseTrainingHandler
from autotrainer.services.checkpoint_service import CheckpointService

# Emergency notification log
NOTIFY_LOG = os.path.join(os.path.dirname(__file__), "..", "..", "notifications.log")


def _notify(level: str, message: str, work_dir: str = ""):
    """Write emergency notification to a persistent log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "work_dir": work_dir,
    }
    try:
        with open(NOTIFY_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _find_latest_checkpoint(output_dir: str) -> tuple[str | None, int]:
    """Find the most recent checkpoint-N directory in output_dir.

    Returns (checkpoint_path, step_number) or (None, 0).
    """
    p = Path(output_dir)
    if not p.exists():
        return None, 0
    best = None
    best_step = 0
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                if step > best_step:
                    best_step = step
                    best = str(d)
            except (IndexError, ValueError):
                continue
    return best, best_step


class FullTrainingHandler(PhaseHandler):
    """Execute full-scale training with the best config from ablation.

    Features:
    - Step-level checkpoint resume
    - Health watchdog with auto-kill on hang
    - Disk space monitoring
    - Emergency notification log
    """

    def __init__(self, paddleformers_root: str = "",
                 health_poll_interval: float = 5.0, health_hang_timeout: float = 300.0):
        self.pf_root = paddleformers_root
        self.health_poll_interval = health_poll_interval
        self.health_hang_timeout = health_hang_timeout

    def execute(self, ctx: PipelineContext) -> PhaseResult:
        def notify(msg: str):
            ctx.notify("FULL_TRAINING", msg)

        config_builder = ConfigBuilder()
        train_mgr = TrainManager(paddleformers_root=self.pf_root, work_dir=ctx.work_dir)
        diagnose_skill = DiagnoseTrainingHandler()
        checkpoint_svc = CheckpointService(None, "", ctx.work_dir)  # type: ignore

        config = ctx.best_ablation_config
        if not config:
            registry = TaskRegistry()
            task_spec = registry.get(ctx.task)
            if task_spec:
                model_path = AutoTrainerConfig.detect_model_path(task_spec.model_name_or_path)
                config = config_builder.build_task_config(
                    task_spec=task_spec,
                    train_data_path=ctx.data_path,
                    eval_data_path=ctx.eval_data_path,
                    overrides={"model": {"model_name_or_path": model_path},
                               "finetuning": {"learning_rate": 1e-5, "warmup_ratio": 0.01,
                                              "weight_decay": 0.1, "sharding": "stage2"}},
                )
            else:
                return PhaseResult(Phase.FULL_TRAINING, PhaseStatus.FAILED, f"Unknown task: {ctx.task}")

        # Ensure data paths
        if ctx.data_path:
            config.setdefault("data", {})["train_dataset_path"] = ctx.data_path
        if ctx.eval_data_path:
            config.setdefault("data", {})["eval_dataset_path"] = ctx.eval_data_path

        output_dir = os.path.join(ctx.work_dir, "checkpoints", "full-training")
        config.setdefault("finetuning", {})["output_dir"] = output_dir
        config["finetuning"]["save_steps"] = 500
        config["finetuning"]["eval_steps"] = 500

        # Step-level resume: find latest checkpoint
        latest_ckpt, resume_step = _find_latest_checkpoint(output_dir)
        if latest_ckpt and resume_step > 0:
            config["finetuning"]["resume_from_checkpoint"] = latest_ckpt
            notify(f"Resuming from checkpoint at step {resume_step}")
            _notify("INFO", f"Resume from step {resume_step}", ctx.work_dir)

        # Health monitoring with auto-actions
        log_path = os.path.join(ctx.work_dir, "logs", "train-full-training.log")
        health_monitor = HealthMonitor(
            gpu_ids=ctx.gpu_ids,
            poll_interval=self.health_poll_interval,
            hang_timeout=self.health_hang_timeout,
            work_dir=ctx.work_dir,
        )
        health_monitor.set_log_path(log_path)
        health_monitor.set_process_checker(
            lambda: train_mgr.subprocess_mgr.is_alive("full-training")
        )
        # Auto-kill on hang, disk full, etc.
        health_monitor.set_process_killer(
            lambda: train_mgr.subprocess_mgr.kill("full-training")
        )
        health_monitor.on_anomaly(self._make_health_handler(ctx))
        health_monitor.on_action(self._make_action_handler(ctx))
        health_monitor.start()

        notify("Starting full training with best config...")

        try:
            result = train_mgr.run_full_training(
                config=config, experiment_id="full-training", gpu_ids=ctx.gpu_ids,
            )
            ctx.full_training_result = result.to_dict()

            if result.status == "failed":
                # Check if process was killed by watchdog (recoverable)
                if any(e.get("error_type") == "process_killed_by_watchdog" for e in result.errors):
                    _notify("WARN", "Training killed by watchdog - may resume from checkpoint", ctx.work_dir)
                    return PhaseResult(Phase.FULL_TRAINING, PhaseStatus.FAILED,
                                       "Training killed by health watchdog. Re-run to resume from checkpoint.",
                                       data={"recoverable": True, "last_step": result.total_steps})

                diagnosis = diagnose_skill.diagnose(
                    self._format_errors(result),
                    training_context={"learning_rate": config.get("finetuning", {}).get("learning_rate", 1e-4)},
                )
                _notify("ERROR", f"Training failed: {diagnosis.get('root_cause', 'unknown')}", ctx.work_dir)
                notify(f"Training failed. Diagnosis: {diagnosis.get('root_cause', 'unknown')}")
                return PhaseResult(Phase.FULL_TRAINING, PhaseStatus.FAILED,
                                   f"Training failed: {diagnosis.get('root_cause', 'unknown')}",
                                   data={"diagnosis": diagnosis})

        finally:
            health_monitor.stop()

        _notify("INFO", f"Full training complete. Final loss={result.final_loss}", ctx.work_dir)
        notify(f"Full training complete. Loss={result.final_loss}")
        return PhaseResult(Phase.FULL_TRAINING, PhaseStatus.COMPLETED,
                           f"Full training complete. Loss={result.final_loss}")

    @staticmethod
    def _format_errors(result) -> str:
        return "\n".join(f"[{e.get('error_type', '?')}] {e.get('line', '')[:200]}" for e in result.errors[:10])

    @staticmethod
    def _make_health_handler(ctx: PipelineContext):
        """Notification-only handler — logs anomalies but doesn't act."""
        def handler(status):
            anomaly = status.anomaly or ""
            disk_info = ""
            if status.disk_free_gb > 0:
                disk_info = f" disk_free={status.disk_free_gb:.1f}GB"

            msg_map = {
                "process_dead": "Training process died unexpectedly!",
                "hang_detected": f"Training hung >{status.last_log_age_seconds:.0f}s — watchdog will kill",
                "low_gpu_utilization": "GPU utilization sustained <5%.",
                "disk_full": f"Disk usage >95%!{disk_info}",
                "disk_low": f"Disk low: {status.disk_free_gb:.1f}GB free{disk_info}",
                "process_killed_by_watchdog": "Training process killed by watchdog (hung).",
            }
            for key, msg in msg_map.items():
                if key in anomaly:
                    ctx.notify("HEALTH", msg)
                    _notify("WARN", msg, ctx.work_dir)
                    break
            if ctx.on_health:
                ctx.on_health(status)
        return handler

    @staticmethod
    def _make_action_handler(ctx: PipelineContext):
        """Auto-recovery action handler."""
        def handler(status):
            anomaly = status.anomaly or ""
            if "hang_detected" in anomaly:
                ctx.notify("HEALTH", "Auto-action: killing hung training process")
                _notify("ACTION", "Watchdog auto-killing hung training process", ctx.work_dir)
            elif "oom_risk" in anomaly:
                ctx.notify("HEALTH", f"OOM risk on {anomaly} — consider reducing batch size")
                _notify("WARN", f"GPU OOM risk: {anomaly}", ctx.work_dir)
            elif "disk_full" in anomaly or "disk_low" in anomaly:
                _notify("CRITICAL", f"Disk space critical: {status.disk_free_gb:.1f}GB free", ctx.work_dir)
        return handler
