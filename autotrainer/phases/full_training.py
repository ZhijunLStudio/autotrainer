"""Phase 4: Full Training — best config on full data with health watchdog."""

from __future__ import annotations

import os

from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler
from autotrainer.managers.train_manager import TrainManager
from autotrainer.orchestrator.health_monitor import HealthMonitor
from autotrainer.pf_integration.config_builder import ConfigBuilder
from autotrainer.skills.diagnose_training.handler import DiagnoseTrainingHandler
from autotrainer.services.checkpoint_service import CheckpointService


class FullTrainingHandler(PhaseHandler):
    """Execute full-scale training with the best config from ablation."""

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
            model_path = self._detect_model_path("PaddlePaddle/PaddleOCR-VL")
            config = config_builder.build_paddleocr_vl_config(
                model_path=model_path,
                train_data=ctx.data_path,
                eval_data=ctx.eval_data_path,
            )

        # Ensure data paths
        if ctx.data_path:
            config.setdefault("data", {})["train_dataset_path"] = ctx.data_path
        if ctx.eval_data_path:
            config.setdefault("data", {})["eval_dataset_path"] = ctx.eval_data_path

        config.setdefault("finetuning", {})["output_dir"] = os.path.join(ctx.work_dir, "checkpoints", "full-training")
        config["finetuning"]["save_steps"] = 500
        config["finetuning"]["eval_steps"] = 500

        # Health monitoring
        log_path = os.path.join(ctx.work_dir, "logs", "train-full-training.log")
        health_monitor = HealthMonitor(
            gpu_ids=ctx.gpu_ids,
            poll_interval=self.health_poll_interval,
            hang_timeout=self.health_hang_timeout,
        )
        health_monitor.set_log_path(log_path)
        health_monitor.set_process_checker(lambda: train_mgr.subprocess_mgr.is_alive("full-training"))
        health_monitor.on_anomaly(self._make_health_handler(ctx))
        health_monitor.start()

        notify("Starting full training with best config...")

        try:
            result = train_mgr.run_full_training(
                config=config, experiment_id="full-training", gpu_ids=ctx.gpu_ids,
            )
            ctx.full_training_result = result.to_dict()

            if result.status == "failed":
                diagnosis = diagnose_skill.diagnose(
                    self._format_errors(result),
                    training_context={"learning_rate": config.get("finetuning", {}).get("learning_rate", 1e-4)},
                )
                notify(f"Training failed. Diagnosis: {diagnosis.get('root_cause', 'unknown')}")
                return PhaseResult(Phase.FULL_TRAINING, PhaseStatus.FAILED,
                                   f"Training failed: {diagnosis.get('root_cause', 'unknown')}",
                                   data={"diagnosis": diagnosis})

        finally:
            health_monitor.stop()

        notify(f"Full training complete. Loss={result.final_loss}")
        return PhaseResult(Phase.FULL_TRAINING, PhaseStatus.COMPLETED, f"Full training complete. Loss={result.final_loss}")

    @staticmethod
    def _format_errors(result) -> str:
        return "\n".join(f"[{e.get('error_type', '?')}] {e.get('line', '')[:200]}" for e in result.errors[:10])

    @staticmethod
    def _make_health_handler(ctx: PipelineContext):
        def handler(status):
            msg_map = {
                "process_dead": "Training process died unexpectedly!",
                "hang_detected": "Training appears hung. Consider restarting.",
                "low_gpu_utilization": "GPU utilization sustained <5%.",
            }
            for key, msg in msg_map.items():
                if status.anomaly and key in (status.anomaly or ""):
                    ctx.notify("HEALTH", msg)
                    break
            if ctx.on_health:
                ctx.on_health(status)
        return handler

    @staticmethod
    def _detect_model_path(model_id: str) -> str:
        from pathlib import Path
        import os
        custom_root = os.environ.get("AUTOTRAINER_MODELS_DIR")
        if custom_root:
            local_path = Path(custom_root) / model_id
            if (local_path / "config.json").exists():
                return str(local_path)
        return model_id
