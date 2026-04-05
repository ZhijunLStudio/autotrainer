"""Training manager — launches and manages PaddleFormers training subprocesses.

Orchestrates the full training lifecycle: config generation, launch,
monitoring, log parsing, and result collection.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from autotrainer.pf_integration.config_builder import ConfigBuilder
from autotrainer.pf_integration.launcher import PaddleFormersLauncher
from autotrainer.pf_integration.log_parser import LogParser, LogMetrics
from autotrainer.utils.subprocess_mgr import SubprocessManager
from autotrainer.utils.file_utils import atomic_write_json


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    experiment_id: str
    config_path: str
    output_dir: str
    gpu_ids: list[int] = field(default_factory=list)
    max_steps: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of a completed training run."""

    experiment_id: str = ""
    final_loss: float | None = None
    total_steps: int = 0
    total_time_seconds: float = 0
    checkpoint_path: str = ""
    log_path: str = ""
    eval_loss: float | None = None
    throughput: float | None = None
    metrics: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)  # {steps:[], train_loss:[], eval_loss:[], learning_rate:[]}
    errors: list[dict] = field(default_factory=list)
    status: str = "completed"  # completed, failed, hung

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "final_loss": self.final_loss,
            "total_steps": self.total_steps,
            "total_time_seconds": round(self.total_time_seconds, 1),
            "checkpoint_path": self.checkpoint_path,
            "log_path": self.log_path,
            "eval_loss": self.eval_loss,
            "throughput": self.throughput,
            "metrics": self.metrics,
            "history": self.history,
            "errors": self.errors,
            "status": self.status,
        }


class TrainManager:
    """Manages the full training lifecycle."""

    def __init__(
        self,
        paddleformers_root: str,
        work_dir: str,
        on_log_line: Callable[[str], None] | None = None,
        on_metrics: Callable[[LogMetrics], None] | None = None,
        on_error: Callable[[dict], None] | None = None,
    ):
        self.pf_root = paddleformers_root
        self.work_dir = work_dir
        self.config_builder = ConfigBuilder()
        self.log_parser = LogParser()
        self.subprocess_mgr = SubprocessManager()
        self.launcher = PaddleFormersLauncher(paddleformers_root, self.subprocess_mgr)

        # Callbacks
        self._on_log_line = on_log_line
        self._on_metrics = on_metrics
        self._on_error = on_error

        # State
        self._current_exp_id: str = ""
        self._start_time: float = 0
        self._latest_metrics: LogMetrics | None = None
        self._collected_errors: list[dict] = []

    def run_single_ablation(
        self,
        base_config: dict,
        factor_changes: dict[str, Any],
        subset_path: str,
        max_steps: int,
        experiment_id: str,
        gpu_ids: list[int] | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingResult:
        """Run a single ablation experiment and return the result.

        Used by the intelligent ablation loop in the pipeline.
        """
        # Build config
        exp_config = self.config_builder.build_ablation_config(
            base=base_config,
            factor_changes=factor_changes,
            subset_path=subset_path,
            max_steps=max_steps,
        )

        if resume_from_checkpoint:
            exp_config["finetuning"]["resume_from_checkpoint"] = resume_from_checkpoint

        output_dir = os.path.join(self.work_dir, "checkpoints", experiment_id)
        exp_config["finetuning"]["output_dir"] = output_dir

        # Write config
        config_path = os.path.join(self.work_dir, "configs", f"{experiment_id}.yaml")
        self.config_builder.to_yaml(exp_config, config_path)

        # Save experiment record
        experiment_record = {
            "id": experiment_id,
            "phase": "ablation",
            "config_diff": factor_changes,
            "config_path": config_path,
            "data_subset": subset_path,
            "max_steps": max_steps,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        self._save_experiment_record(experiment_record)

        # Run training
        result = self._run_single(
            experiment_id=experiment_id,
            config_path=config_path,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
        )

        # Update record
        experiment_record["status"] = result.status
        experiment_record["result"] = result.to_dict()
        experiment_record["completed_at"] = datetime.now().isoformat()
        self._save_experiment_record(experiment_record)

        return result

    def run_ablation(
        self,
        base_config: dict,
        factors: dict[str, list],
        subset_path: str,
        max_steps: int = 1000,
        gpu_ids: list[int] | None = None,
    ) -> list[TrainingResult]:
        """Run a set of ablation experiments sequentially.

        Each experiment varies one factor from the base config.
        Returns results for all runs.
        """
        results = []
        exp_num = 0

        for factor_name, values in factors.items():
            for value in values:
                exp_num += 1
                exp_id = f"abl-{exp_num:03d}-{factor_name}={value}"
                factor_changes = {factor_name: value}

                # Build config
                exp_config = self.config_builder.build_ablation_config(
                    base=base_config,
                    factor_changes=factor_changes,
                    subset_path=subset_path,
                    max_steps=max_steps,
                )

                output_dir = os.path.join(self.work_dir, "checkpoints", exp_id)
                exp_config["finetuning"]["output_dir"] = output_dir

                # Write config
                config_path = os.path.join(self.work_dir, "configs", f"{exp_id}.yaml")
                self.config_builder.to_yaml(exp_config, config_path)

                # Save experiment record
                experiment_record = {
                    "id": exp_id,
                    "phase": "ablation",
                    "config_diff": factor_changes,
                    "config_path": config_path,
                    "data_subset": subset_path,
                    "max_steps": max_steps,
                    "status": "pending",
                    "created_at": datetime.now().isoformat(),
                }
                self._save_experiment_record(experiment_record)

                # Run training
                result = self._run_single(
                    experiment_id=exp_id,
                    config_path=config_path,
                    output_dir=output_dir,
                    gpu_ids=gpu_ids,
                )

                # Update record
                experiment_record["status"] = result.status
                experiment_record["result"] = result.to_dict()
                experiment_record["completed_at"] = datetime.now().isoformat()
                self._save_experiment_record(experiment_record)

                results.append(result)

        return results

    def run_full_training(
        self,
        config: dict,
        experiment_id: str = "full-training",
        gpu_ids: list[int] | None = None,
    ) -> TrainingResult:
        """Run full-scale training with the given config."""
        output_dir = os.path.join(self.work_dir, "checkpoints", experiment_id)
        config["finetuning"]["output_dir"] = output_dir

        config_path = os.path.join(self.work_dir, "configs", f"{experiment_id}.yaml")
        self.config_builder.to_yaml(config, config_path)

        return self._run_single(
            experiment_id=experiment_id,
            config_path=config_path,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
        )

    def _run_single(
        self,
        experiment_id: str,
        config_path: str,
        output_dir: str,
        gpu_ids: list[int] | None = None,
    ) -> TrainingResult:
        """Run a single training experiment."""
        self._current_exp_id = experiment_id
        self._start_time = time.time()
        self._collected_errors = []
        os.makedirs(output_dir, exist_ok=True)

        log_path = os.path.join(self.work_dir, "logs", f"train-{experiment_id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        log_file = open(log_path, "w")

        # Per-step training history for curve plots
        history_steps: list[int] = []
        history_train_loss: list[float] = []
        history_eval_loss: list[float] = []
        history_lr: list[float] = []

        def _on_stdout(line: str):
            log_file.write(line + "\n")
            log_file.flush()
            if self._on_log_line:
                self._on_log_line(line)

            metrics = self.log_parser.parse_line(line)
            if metrics:
                self._latest_metrics = metrics
                if self._on_metrics:
                    self._on_metrics(metrics)

                # Collect per-step history
                if metrics.step is not None:
                    history_steps.append(metrics.step)
                    history_train_loss.append(metrics.loss if metrics.loss is not None else 0.0)
                    history_eval_loss.append(metrics.eval_loss if metrics.eval_loss is not None else 0.0)
                    history_lr.append(metrics.lr if metrics.lr is not None else 0.0)

            error = self.log_parser.detect_error(line)
            if error:
                err_dict = {
                    "line": line,
                    "error_type": error.error_type,
                    "severity": error.severity,
                }
                self._collected_errors.append(err_dict)
                if self._on_error:
                    self._on_error(err_dict)

        try:
            self.launcher.launch_training(
                config_path=config_path,
                gpu_ids=gpu_ids,
                log_dir=os.path.dirname(log_path),
                process_name=experiment_id,
                on_stdout=_on_stdout,
            )

            # Wait for training to complete
            while self.launcher.is_training_alive(experiment_id):
                time.sleep(2)

            exit_code = self.subprocess_mgr.get_exit_code(experiment_id)

        except Exception as e:
            exit_code = -1
            self._collected_errors.append({"error_type": "launch_failed", "message": str(e)})

        finally:
            log_file.close()

        elapsed = time.time() - self._start_time

        # Parse final metrics from log
        final_metrics = self.log_parser.extract_final_metrics(log_path)

        # Determine status: completed if exit_code == 0, or if we ran most/all steps
        # (eval may crash on IterableDataset but training itself succeeded)
        final_step = final_metrics.get("final_step", 0)
        expected_steps = final_metrics.get("max_steps", 0)
        if exit_code == 0:
            status = "completed"
        elif final_step > 0 and (expected_steps <= 0 or final_step >= expected_steps * 0.9):
            status = "completed"  # Training finished, eval may have failed
        else:
            status = "failed"

        result = TrainingResult(
            experiment_id=experiment_id,
            final_loss=final_metrics.get("final_loss"),
            total_steps=final_step,
            total_time_seconds=elapsed,
            checkpoint_path=output_dir,
            log_path=log_path,
            eval_loss=final_metrics.get("final_eval_loss"),
            throughput=final_metrics.get("throughput"),
            metrics=final_metrics,
            history={
                "steps": history_steps,
                "train_loss": history_train_loss,
                "eval_loss": history_eval_loss,
                "learning_rate": history_lr,
            },
            errors=self._collected_errors,
            status=status,
        )

        # Save result
        result_path = os.path.join(output_dir, "result.json")
        atomic_write_json(result_path, result.to_dict())

        return result

    def _save_experiment_record(self, record: dict):
        """Save/update an experiment record in the index."""
        index_path = os.path.join(self.work_dir, "experiment_index.json")
        from autotrainer.utils.file_utils import safe_read_json

        index = safe_read_json(index_path) or {"experiments": [], "rankings": []}

        # Update or append
        existing_ids = {e["id"] for e in index.get("experiments", [])}
        if record["id"] in existing_ids:
            for i, e in enumerate(index["experiments"]):
                if e["id"] == record["id"]:
                    index["experiments"][i] = record
                    break
        else:
            index["experiments"].append(record)

        atomic_write_json(index_path, index)

    def get_latest_metrics(self) -> LogMetrics | None:
        """Get the most recently parsed metrics."""
        return self._latest_metrics
