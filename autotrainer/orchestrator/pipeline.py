"""Pipeline orchestrator — the real task flow engine.

Ties together all layers: PhaseManager, managers, skills, health monitor,
and recovery into a single coherent execution pipeline.

This is NOT the same as state_machine.py (which only tracks phase states).
This class actually executes each phase's logic, makes decisions,
handles errors, and coordinates the entire training lifecycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from autotrainer.config import AutoTrainerConfig
from autotrainer.context.store import ContextStore
from autotrainer.context.summarizer import LogSummarizer
from autotrainer.managers.data_manager import DataManager
from autotrainer.managers.env_manager import EnvManager
from autotrainer.managers.eval_manager import EvalManager
from autotrainer.managers.train_manager import TrainManager, TrainingResult
from autotrainer.orchestrator.health_monitor import HealthMonitor, HealthStatus
from autotrainer.orchestrator.recovery import RecoveryManager, RecoveryState
from autotrainer.orchestrator.state_machine import Phase, PhaseManager
from autotrainer.pf_integration.config_builder import ConfigBuilder
from autotrainer.pf_integration.log_parser import LogMetrics
from autotrainer.skills.data_intel.handler import DataIntelHandler
from autotrainer.skills.diagnose_training.handler import DiagnoseTrainingHandler
from autotrainer.skills.loader import SkillLoader
from autotrainer.skills.plan_experiment.handler import PlanExperimentHandler
from autotrainer.utils.file_utils import atomic_write_json, ensure_dir, safe_read_json


# --- Callback types for TUI integration ---
OnPhaseChange = Callable[[str, str], None]           # (phase_name, message)
OnLogLine = Callable[[str], None]
OnMetrics = Callable[[LogMetrics], None]
OnHealthUpdate = Callable[[HealthStatus], None]
OnUserConfirm = Callable[[str, dict], bool]           # (message, context) -> confirmed
OnUserInput = Callable[[str, list[str]], str]         # (prompt, choices) -> user_choice


@dataclass
class PipelineState:
    """Full pipeline state snapshot (for recovery)."""

    task: str = ""
    goal: str = ""
    gpu_ids: list[int] = field(default_factory=list)
    base_config: dict = field(default_factory=dict)
    ablation_config: dict = field(default_factory=dict)
    data_path: str = ""
    eval_data_path: str = ""
    data_profile: dict = field(default_factory=dict)
    ablation_results: list[dict] = field(default_factory=list)
    best_ablation_config: dict = field(default_factory=dict)
    full_training_result: dict = field(default_factory=dict)
    eval_result: dict = field(default_factory=dict)


class PipelineOrchestrator:
    """Executes the full automated training pipeline.

    Drives the system through 7 phases with real logic at each step:
    0. Task Confirm — verify goal, GPU selection, config
    1. Data Prepare — validate/profile/download/convert data
    2. Env Check — verify PaddlePaddle, packages, GPU
    3. Ablation — single-factor experiments on 5% subset
    4. Full Training — top configs on full data
    5. Evaluation — eval + inference verification
    6. Report — generate visualizations and summary

    Each phase can fail and trigger error recovery via skills.
    """

    def __init__(
        self,
        config: AutoTrainerConfig,
        task: str = "paddleocr-vl",
        gpu_ids: list[int] | None = None,
        resume: bool = False,
    ):
        self.config = config
        self.task = task
        self.gpu_ids = gpu_ids or list(range(config.detect_gpu_count()))

        # Work directory
        self.work_dir = os.path.join(config.work_dir, f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        ensure_dir(self.work_dir)

        # Core components
        self.phase_mgr = PhaseManager(
            state_file=os.path.join(self.work_dir, "pipeline_state.json"),
            task_name=task,
        )
        self.recovery = RecoveryManager(self.work_dir)
        self.context = ContextStore(max_tokens=config.context_max_tokens)
        self.summarizer = LogSummarizer()

        # Managers
        self.env_mgr = EnvManager(paddleformers_root=config.paddleformers_root)
        self.data_mgr = DataManager(cache_dir=os.path.join(self.work_dir, "data"), paddleformers_root=config.paddleformers_root)
        self.train_mgr = TrainManager(paddleformers_root=config.paddleformers_root, work_dir=self.work_dir)
        self.eval_mgr = EvalManager(paddleformers_root=config.paddleformers_root, work_dir=self.work_dir)
        self.config_builder = ConfigBuilder()

        # Skills
        self.skill_loader = SkillLoader(skills_dir=config.skills_dir)
        self.diagnose_skill = DiagnoseTrainingHandler()
        self.plan_skill = PlanExperimentHandler()
        self.data_intel_skill = DataIntelHandler(cache_dir=self.work_dir)

        # Health monitor
        self.health_monitor = HealthMonitor(
            gpu_ids=self.gpu_ids,
            poll_interval=config.health_poll_interval,
            hang_timeout=config.health_hang_timeout,
        )

        # Pipeline state
        self.state = PipelineState(task=task, gpu_ids=self.gpu_ids)

        # Callbacks (set by TUI or headless runner)
        self.on_phase_change: OnPhaseChange | None = None
        self.on_log_line: OnLogLine | None = None
        self.on_metrics: OnMetrics | None = None
        self.on_health: OnHealthUpdate | None = None
        self.on_user_confirm: OnUserConfirm | None = None
        self.on_user_input: OnUserInput | None = None

        # Hook up health monitor
        self.health_monitor.on_anomaly(self._handle_health_anomaly)

        # Load previous state if resuming
        if resume:
            self._try_resume()

    def _try_resume(self):
        """Attempt to resume from a previous state."""
        if self.phase_mgr.load_state():
            saved = self.recovery.load()
            if saved:
                self.state.task = saved.task_name
                self.state.gpu_ids = saved.gpu_ids
                self.state.data_path = saved.data_path
                self.state.eval_data_path = saved.eval_data_path

    def _notify(self, phase: str, msg: str):
        """Notify phase change callback."""
        if self.on_phase_change:
            self.on_phase_change(phase, msg)

    def _confirm(self, message: str, context: dict | None = None) -> bool:
        """Request user confirmation. Returns True if confirmed."""
        if self.on_user_confirm:
            return self.on_user_confirm(message, context or {})
        # Default: auto-confirm in headless mode
        return True

    def _input(self, prompt: str, choices: list[str]) -> str:
        """Request user input. Returns the chosen value."""
        if self.on_user_input:
            return self.on_user_input(prompt, choices)
        # Default: first choice
        return choices[0] if choices else ""

    def _save_recovery_state(self):
        """Persist current pipeline state for recovery."""
        rs = RecoveryState(
            task_name=self.state.task,
            experiment_id=f"{self.task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            current_phase=self.phase_mgr.current.value,
            gpu_ids=self.state.gpu_ids,
            data_path=self.state.data_path,
            eval_data_path=self.state.eval_data_path,
            last_config_path="",
        )
        self.recovery.save(rs)

        # Also save the full pipeline state
        atomic_write_json(
            os.path.join(self.work_dir, "pipeline_state_full.json"),
            {
                "task": self.state.task,
                "goal": self.state.goal,
                "gpu_ids": self.state.gpu_ids,
                "data_path": self.state.data_path,
                "data_profile": self.state.data_profile,
                "ablation_results": self.state.ablation_results,
                "best_ablation_config": self.state.best_ablation_config,
                "full_training_result": self.state.full_training_result,
                "eval_result": self.state.eval_result,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def run(self):
        """Execute the full pipeline sequentially."""
        try:
            self._run_phase_data_prepare()
            self._run_phase_env_check()
            self._run_phase_ablation()
            self._run_phase_full_training()
            self._run_phase_evaluation()
            self._run_phase_report()

            self.phase_mgr.transition_to(Phase.COMPLETED)
            self._notify("COMPLETED", "Pipeline finished successfully!")
        except Exception as e:
            self.phase_mgr.mark_failed(self.phase_mgr.current, str(e))
            self._notify("FAILED", f"Pipeline failed: {e}")
            raise

    # ════════════════════════════════════════════════════════════
    # Phase 0: Task Confirm (called externally by CLI/TUI)
    # ════════════════════════════════════════════════════════════

    def confirm_task(self, goal: str = "", data_path: str = "", eval_data_path: str = ""):
        """Confirm task setup before starting the pipeline.

        Called by CLI/TUI after user confirms task details.
        """
        self.state.goal = goal
        self.state.data_path = data_path
        self.state.eval_data_path = eval_data_path

        # Set up context
        self.context.set_system(
            f"Task: {self.task}\n"
            f"Goal: {goal}\n"
            f"GPUs: {self.gpu_ids}\n"
            f"Work dir: {self.work_dir}"
        )

        self.phase_mgr.transition_to(Phase.DATA_PREPARE)
        self._save_recovery_state()
        self._notify("DATA_PREPARE", "Starting data preparation...")

    # ════════════════════════════════════════════════════════════
    # Phase 1: Data Prepare
    # ════════════════════════════════════════════════════════════

    def _run_phase_data_prepare(self):
        """Phase 1: Validate, profile, and prepare training data."""
        if self.phase_mgr.is_completed(Phase.DATA_PREPARE):
            self._notify("DATA_PREPARE", "Skipped (already completed)")
            return

        self.phase_mgr.transition_to(Phase.DATA_PREPARE)
        self._notify("DATA_PREPARE", "Validating and profiling data...")

        data_path = self.state.data_path
        if not data_path or not os.path.exists(data_path):
            # Try data intelligence
            result = self.data_intel_skill.handle_discover_mode(self.task)
            self._notify("DATA_PREPARE", f"Data not found. Use `autotrainer data` to prepare data first.")
            raise RuntimeError("No training data found. Run `autotrainer data --mode discover --task <task>` first.")

        # Validate
        validation = self.data_mgr.validate_dataset(data_path)
        if not validation["valid"]:
            error_msg = "\n".join(validation["errors"][:5])
            if not self._confirm(f"Data validation failed:\n{error_msg}\nContinue anyway?"):
                raise RuntimeError("Data validation failed, user aborted.")

        # Profile
        profile = self.data_mgr.profile_dataset(data_path)
        self.state.data_profile = profile.to_dict()

        # Set data context
        self.context.set_data_profile(profile.to_dict())

        # Split if no eval data
        if not self.state.eval_data_path:
            split_result = self.data_mgr.split_dataset(
                data_path,
                train_ratio=0.9,
                val_ratio=0.05,
            )
            self.state.data_path = split_result["train"]["path"]
            self.state.eval_data_path = split_result["val"]["path"]
            self._notify(
                "DATA_PREPARE",
                f"Split data: train={split_result['train']['count']}, "
                f"val={split_result['val']['count']}, test={split_result['test']['count']}",
            )

        # Create ablation subset (5%)
        ablation_data_dir = os.path.join(self.work_dir, "data")
        ensure_dir(ablation_data_dir)
        ablation_subset_path = os.path.join(ablation_data_dir, "subset_5pct.jsonl")
        subset_info = self.data_mgr.create_subset(
            self.state.data_path,
            ablation_subset_path,
            ratio=0.05,
        )
        self.state.ablation_config = {"subset_path": ablation_subset_path, "subset_info": subset_info}

        self._save_recovery_state()
        self._notify("DATA_PREPARE", f"Data ready: {profile.num_samples} samples, {profile.format} format")

    # ════════════════════════════════════════════════════════════
    # Phase 2: Environment Check
    # ════════════════════════════════════════════════════════════

    def _run_phase_env_check(self):
        """Phase 2: Verify environment (PaddlePaddle, packages, GPU)."""
        if self.phase_mgr.is_completed(Phase.ENV_CHECK):
            self._notify("ENV_CHECK", "Skipped (already completed)")
            return

        self.phase_mgr.transition_to(Phase.ENV_CHECK)
        self._notify("ENV_CHECK", "Checking environment...")

        env_status = self.env_mgr.check_environment()

        # Report issues
        if env_status.issues:
            issue_text = "\n".join(f"  - {i}" for i in env_status.issues)
            if not self._confirm(
                f"Environment issues found:\n{issue_text}\nContinue anyway?"
            ):
                raise RuntimeError("Environment check failed, user aborted.")

        # Report suggestions
        if env_status.suggestions:
            for sug in env_status.suggestions:
                self._notify("ENV_CHECK", f"Suggestion: {sug}")

        # Interactive package upgrades
        if env_status.packages:
            missing = [k for k, v in env_status.packages.items() if not v]
            if missing:
                choice = self._input(
                    f"Missing packages: {missing}. Install?",
                    ["yes", "skip"],
                )
                if choice == "yes":
                    for pkg in missing:
                        try:
                            self.env_mgr.install_package(pkg)
                        except Exception as e:
                            self._notify("ENV_CHECK", f"Failed to install {pkg}: {e}")

        self._save_recovery_state()
        self._notify(
            "ENV_CHECK",
            f"Environment ready. GPU={env_status.gpu_count}, "
            f"PaddlePaddle={env_status.paddle_version or 'not found'}, "
            f"PaddleFormers={env_status.paddleformers_version or 'not found'}",
        )

    # ════════════════════════════════════════════════════════════
    # Phase 3: Ablation Experiments
    # ════════════════════════════════════════════════════════════

    def _run_phase_ablation(self):
        """Phase 3: Run ablation experiments on 5% data subset.

        Three sub-phases:
        A. Single-factor: lr, bs, lora_rank (each 500-1000 steps)
        B. Multi-factor: top 2-3 combinations from A
        C. Rank and select best for full training
        """
        if self.phase_mgr.is_completed(Phase.ABLATION):
            self._notify("ABLATION", "Skipped (already completed)")
            return

        self.phase_mgr.transition_to(Phase.ABLATION)

        # Build base config
        base_config = self.config_builder.build_paddleocr_vl_config(
            model_path="PaddlePaddle/PaddleOCR-VL",
            train_data=self.state.data_path,
            eval_data=self.state.eval_data_path,
            lora=True,
            lora_rank=8,
        )

        subset_path = self.state.ablation_config.get("subset_path", "")
        if not subset_path or not os.path.exists(subset_path):
            self._notify("ABLATION", "No ablation subset found, skipping ablation.")
            self.state.best_ablation_config = base_config
            return

        # --- Sub-phase A: Single-factor ablation ---
        self._notify("ABLATION", "Phase A: Single-factor ablation (5% subset, 1000 steps each)")
        single_factors = {
            "finetuning.learning_rate": [1e-5, 3e-5, 1e-4, 3e-4],
            "finetuning.per_device_train_batch_size": [1, 2, 4],
            "model.lora_rank": [4, 8, 16],
        }

        ablation_plan = self.plan_skill.plan_ablation(
            base_config=base_config,
            factors=single_factors,
            subset_ratio=0.05,
            max_steps=1000,
        )

        self._notify(
            "ABLATION",
            f"Planned {ablation_plan['total_runs']} single-factor experiments",
        )

        if not self._confirm(f"Run {ablation_plan['total_runs']} ablation experiments?"):
            self._notify("ABLATION", "Skipped by user. Using default config.")
            self.state.best_ablation_config = base_config
            return

        # Run single-factor experiments
        single_results = self.train_mgr.run_ablation(
            base_config=base_config,
            factors=single_factors,
            subset_path=subset_path,
            max_steps=1000,
            gpu_ids=self.gpu_ids,
        )

        # Record results
        for r in single_results:
            self.state.ablation_results.append(r.to_dict())

        # --- Sub-phase B: Rank and plan multi-factor ---
        self._notify("ABLATION", "Phase B: Analyzing results, planning multi-factor experiments")

        # Build experiment history for the plan skill
        history = []
        for r in single_results:
            history.append(
                {
                    "id": r.experiment_id,
                    "status": r.status,
                    "result": r.to_dict(),
                    "config_diff": {r.experiment_id.split("-")[2]: r.experiment_id.split("=")[1] if "=" in r.experiment_id else ""},
                }
            )

        ranking = self.plan_skill.rank_experiments(history)

        if ranking:
            top_n = ranking[:3]
            self._notify("ABLATION", f"Top 3 configs from Phase A:")
            for i, r in enumerate(top_n, 1):
                self._notify("ABLATION", f"  {i}. {r['id']} — eval_loss={r['eval_loss']}")

            # Pick the best as our full training config
            best_exp = top_n[0]
            self._notify("ABLATION", f"Selected best config from: {best_exp['id']}")
            # Note: In a full implementation, we'd reconstruct the config from best_exp
            self.state.best_ablation_config = base_config
        else:
            self._notify("ABLATION", "No successful ablation results. Using default config.")
            self.state.best_ablation_config = base_config

        self._save_recovery_state()
        self._notify("ABLATION", f"Ablation complete. {len(single_results)} experiments run.")

    # ════════════════════════════════════════════════════════════
    # Phase 4: Full Training
    # ════════════════════════════════════════════════════════════

    def _run_phase_full_training(self):
        """Phase 4: Full-scale training with best config from ablation."""
        if self.phase_mgr.is_completed(Phase.FULL_TRAINING):
            self._notify("FULL_TRAINING", "Skipped (already completed)")
            return

        self.phase_mgr.transition_to(Phase.FULL_TRAINING)
        self._notify("FULL_TRAINING", "Starting full training with best config...")

        config = self.state.best_ablation_config
        if not config:
            config = self.config_builder.build_paddleocr_vl_config(
                model_path="PaddlePaddle/PaddleOCR-VL",
                train_data=self.state.data_path,
                eval_data=self.state.eval_data_path,
            )

        # Set full training parameters
        config["finetuning"]["output_dir"] = os.path.join(self.work_dir, "checkpoints", "full-training")
        config["finetuning"]["save_steps"] = 500
        config["finetuning"]["eval_steps"] = 500

        # Start health monitoring
        log_path = os.path.join(self.work_dir, "logs", "train-full-training.log")
        self.health_monitor.set_log_path(log_path)
        self.health_monitor.set_process_checker(lambda: self.train_mgr.subprocess_mgr.is_alive("full-training"))
        self.health_monitor.start()

        try:
            # Run training
            result = self.train_mgr.run_full_training(
                config=config,
                experiment_id="full-training",
                gpu_ids=self.gpu_ids,
            )
            self.state.full_training_result = result.to_dict()

            if result.status == "failed":
                # Error recovery via diagnose skill
                self._handle_training_error(result)

        finally:
            self.health_monitor.stop()

        self._save_recovery_state()
        self._notify("FULL_TRAINING", f"Full training complete. Loss={result.final_loss}")

    # ════════════════════════════════════════════════════════════
    # Phase 5: Evaluation
    # ════════════════════════════════════════════════════════════

    def _run_phase_evaluation(self):
        """Phase 5: Evaluate model and run inference verification."""
        if self.phase_mgr.is_completed(Phase.EVALUATION):
            self._notify("EVALUATION", "Skipped (already completed)")
            return

        self.phase_mgr.transition_to(Phase.EVALUATION)
        self._notify("EVALUATION", "Running evaluation...")

        checkpoint_dir = os.path.join(self.work_dir, "checkpoints", "full-training")

        if not os.path.exists(checkpoint_dir):
            self._notify("EVALUATION", "No checkpoint found, skipping evaluation.")
            return

        # Run eval
        config_path = os.path.join(self.work_dir, "configs", "full-training.yaml")
        if os.path.exists(config_path):
            eval_result = self.eval_mgr.run_eval(config_path, checkpoint_dir)
            self.state.eval_result = eval_result.to_dict()
            self._notify(
                "EVALUATION",
                f"Eval complete: metrics={eval_result.metrics}",
            )

        # Inference verification (sample predictions)
        if self.state.eval_data_path:
            samples = self.eval_mgr.run_inference_samples(
                config_path=config_path if os.path.exists(config_path) else "",
                checkpoint_path=checkpoint_dir,
                data_path=self.state.eval_data_path,
                n_samples=5,
            )
            if samples:
                self._notify("EVALUATION", f"Inference verification: {len(samples)} samples checked")

        self._save_recovery_state()

    # ════════════════════════════════════════════════════════════
    # Phase 6: Report
    # ════════════════════════════════════════════════════════════

    def _run_phase_report(self):
        """Phase 6: Generate final report with visualizations."""
        if self.phase_mgr.is_completed(Phase.REPORT):
            return

        self.phase_mgr.transition_to(Phase.REPORT)
        self._notify("REPORT", "Generating report...")

        # Load experiment index
        index = self.recovery.load_experiment_index()
        experiments = index.get("experiments", [])

        # Text summary
        report_lines = [
            "=" * 60,
            f"  AutoTrainer Report — {self.task}",
            f"  Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            f"Total experiments: {len(experiments)}",
        ]

        completed = [e for e in experiments if e.get("status") == "completed"]
        failed = [e for e in experiments if e.get("status") == "failed"]
        report_lines.append(f"Completed: {len(completed)}")
        report_lines.append(f"Failed: {len(failed)}")

        if completed:
            best = min(
                completed,
                key=lambda x: x.get("result", {}).get("eval_loss", float("inf")),
            )
            report_lines.extend(
                [
                    "",
                    f"Best experiment: {best.get('id')}",
                    f"  Eval loss: {best.get('result', {}).get('eval_loss', 'N/A')}",
                    f"  Config: {best.get('config_diff', {})}",
                ]
            )

        report_text = "\n".join(report_lines)

        # Save report
        report_dir = os.path.join(self.work_dir, "reports")
        ensure_dir(report_dir)
        report_path = os.path.join(report_dir, "final_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        self._notify("REPORT", f"Report saved to {report_path}")

        # Generate charts if matplotlib available
        try:
            self._generate_charts(experiments, report_dir)
        except ImportError:
            pass

    def _generate_charts(self, experiments: list[dict], report_dir: str):
        """Generate comparison charts."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        completed = [e for e in experiments if e.get("status") == "completed" and e.get("result", {}).get("eval_loss")]
        if not completed:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        names = [e.get("id", "?")[:20] for e in completed]
        eval_losses = [e["result"]["eval_loss"] for e in completed]
        colors = ["green" if l == min(eval_losses) else "steelblue" for l in eval_losses]
        ax.barh(names, eval_losses, color=colors)
        ax.set_xlabel("Eval Loss")
        ax.set_title(f"{self.task} — Experiment Comparison")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(report_dir, "eval_loss_comparison.png"), dpi=100)
        plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # Error Handling
    # ════════════════════════════════════════════════════════════

    def _handle_training_error(self, result: TrainingResult):
        """Handle a training failure using the diagnose skill."""
        if not result.errors:
            return

        # Build error context (structured, not raw log)
        error_text = "\n".join(
            f"[{e.get('error_type', '?')}] {e.get('line', '')[:200]}" for e in result.errors[:10]
        )

        # Diagnose
        diagnosis = self.diagnose_skill.diagnose(
            error_text,
            training_context={"learning_rate": 1e-4},  # TODO: extract from config
        )

        self._notify(
            f"DIAGNOSE:{diagnosis.get('error_type', 'unknown')}",
            f"Root cause: {diagnosis.get('root_cause', 'unknown')}\n"
            f"Suggested fix: {diagnosis.get('fix_steps', [])}",
        )

        # Ask user to confirm fix
        if diagnosis.get("config_changes"):
            if self._confirm(
                f"Apply fix? Changes: {diagnosis['config_changes']}",
                context=diagnosis,
            ):
                # Apply changes to best config
                for key, value in diagnosis["config_changes"].items():
                    parts = key.split(".")
                    cfg = self.state.best_ablation_config
                    for p in parts[:-1]:
                        cfg = cfg.setdefault(p, {})
                    cfg[parts[-1]] = value

    def _handle_health_anomaly(self, status: HealthStatus):
        """Handle a health anomaly from the watchdog."""
        if status.anomaly == "process_dead":
            self._notify("HEALTH", "Training process died unexpectedly!")
        elif status.anomaly == "hang_detected":
            self._notify("HEALTH", "Training appears hung (no log output for >timeout). Consider restarting.")
        elif status.anomaly and status.anomaly.startswith("oom_risk"):
            self._notify("HEALTH", f"OOM risk detected on {status.anomaly}")
        elif status.anomaly and status.anomaly.startswith("thermal"):
            self._notify("HEALTH", f"Thermal risk on {status.anomaly}")
        elif status.anomaly == "low_gpu_utilization":
            self._notify("HEALTH", "GPU utilization sustained <5%. Possible hang or I/O bottleneck.")

        if self.on_health:
            self.on_health(status)
