"""Phase 3: Ablation — hyperparameter + data ratio experiments on 5% subset."""

from __future__ import annotations

import copy
import os

from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler
from autotrainer.core.registry import TaskRegistry
from autotrainer.managers.train_manager import TrainManager
from autotrainer.orchestrator.scheduler import ExperimentScheduler, ExperimentSpec
from autotrainer.pf_integration.config_builder import ConfigBuilder
from autotrainer.skills.data_ratio_ablation.handler import DataRatioAblationHandler, DatasetInfo


class AblationHandler(PhaseHandler):
    """Intelligent ablation: hyperparameter tuning + data ratio optimization."""

    def __init__(self, paddleformers_root: str = "", task_registry: TaskRegistry | None = None):
        self.pf_root = paddleformers_root
        self.registry = task_registry or TaskRegistry()

    def execute(self, ctx: PipelineContext) -> PhaseResult:
        task_spec = self.registry.get(ctx.task)
        if not task_spec:
            return PhaseResult(Phase.ABLATION, PhaseStatus.FAILED, f"Unknown task: {ctx.task}")

        def notify(msg: str):
            ctx.notify("ABLATION", msg)

        config_builder = ConfigBuilder()
        train_mgr = TrainManager(paddleformers_root=self.pf_root, work_dir=ctx.work_dir)
        scheduler = ExperimentScheduler(work_dir=ctx.work_dir)
        ratio_handler = DataRatioAblationHandler()

        # Build base config
        model_path = self._detect_model_path("PaddlePaddle/PaddleOCR-VL")
        base_config = config_builder.build_paddleocr_vl_config(
            model_path=model_path,
            train_data=ctx.data_path,
            eval_data=ctx.eval_data_path,
            lora=True,
            lora_rank=8,
        )

        subset_path = ctx.ablation_config.get("subset_path", "")
        if not subset_path or not os.path.exists(subset_path):
            notify("No ablation subset found, skipping ablation.")
            ctx.best_ablation_config = base_config
            return PhaseResult(Phase.ABLATION, PhaseStatus.COMPLETED, "Skipped (no ablation subset)")

        # ── Step A: Hyperparameter ablation ──
        existing_hyperparam = scheduler.get_experiments_by_phase("ablation_hyperparams")
        if not existing_hyperparam:
            notify("=== Step A: Hyperparameter Ablation ===")
            self._build_hyperparam_experiments(scheduler, base_config, subset_path, task_spec, config_builder)

        scheduler.run_all(train_manager=train_mgr, gpu_ids=ctx.gpu_ids,
                          on_progress=lambda s, r: notify(
                              f"[{s.completed}/{s.total}] {r.experiment_id}: loss={r.final_loss or '?'} status={r.status}"
                          ))

        best_config = self._pick_best_hyperparams(scheduler, base_config)
        notify("Step A complete. Best hyperparams applied.")
        scheduler.cleanup_phase_checkpoints("ablation_hyperparams")

        # ── Step B: Data ratio ablation (multi-dataset only) ──
        multi_ds = ctx.multi_dataset_info
        if len(multi_ds) > 1:
            notify("=== Step B: Data Ratio Ablation ===")
            existing_ratio = scheduler.get_experiments_by_phase("ablation_ratio")
            if not existing_ratio:
                datasets = [DatasetInfo(name=d["name"], subset_path=d["subset_path"],
                                        sample_count=d["sample_count"]) for d in multi_ds]
                ratio_specs = ratio_handler.build_experiment_specs(
                    datasets=datasets, base_config=best_config,
                    config_builder=config_builder,
                    subset_dir=os.path.join(ctx.work_dir, "data"),
                )
                scheduler.add_experiments(ratio_specs)

            scheduler.run_all(train_manager=train_mgr, gpu_ids=ctx.gpu_ids,
                              on_progress=lambda s, r: notify(
                                  f"[{s.completed}/{s.total}] {r.experiment_id}: loss={r.final_loss or '?'}"
                              ))
            best_ratio_config = self._pick_best_ratio(scheduler, best_config)
            ctx.best_ablation_config = best_ratio_config
            notify("Step B complete. Best data ratio applied.")
            scheduler.cleanup_phase_checkpoints("ablation_ratio")
        else:
            ctx.best_ablation_config = best_config

        return PhaseResult(Phase.ABLATION, PhaseStatus.COMPLETED, "Ablation complete.")

    def _build_hyperparam_experiments(self, scheduler: ExperimentScheduler, base_config: dict,
                                      subset_path: str, task_spec, config_builder: ConfigBuilder):
        """Build hyperparameter ablation experiment specs from task manifest."""
        hp_space = task_spec.hyperparam_space
        specs = []
        for dotted_key, hp_def in hp_space.items():
            for value in hp_def.get("initial", []):
                exp_id = f"abl-hyper-{dotted_key.split('.')[-1]}={value}"
                exp_config = config_builder.build_ablation_config(
                    base=base_config, factor_changes={dotted_key: value},
                    subset_path=subset_path, max_steps=1000,
                )
                specs.append(ExperimentSpec(
                    id=exp_id, phase="ablation_hyperparams",
                    config=exp_config, config_diff={dotted_key: value},
                ))
        scheduler.add_experiments(specs)

    def _pick_best_hyperparams(self, scheduler: ExperimentScheduler, base_config: dict) -> dict:
        best_config = copy.deepcopy(base_config)
        experiments = scheduler.get_experiments_by_phase("ablation_hyperparams")
        completed = [e for e in experiments if e.status == "completed" and e.result]
        if not completed:
            return best_config

        factor_best: dict[str, tuple] = {}
        for exp in completed:
            factor_key = list(exp.config_diff.keys())[0] if exp.config_diff else ""
            if not factor_key:
                continue
            loss = exp.result.get("eval_loss") or exp.result.get("final_loss") or float("inf")
            value = exp.config_diff[factor_key]
            if factor_key not in factor_best or loss < factor_best[factor_key][1]:
                factor_best[factor_key] = (value, loss)

        for factor_key, (value, _) in factor_best.items():
            parts = factor_key.split(".")
            target = best_config
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = value

        return best_config

    def _pick_best_ratio(self, scheduler: ExperimentScheduler, hyperparams_config: dict) -> dict:
        best_config = copy.deepcopy(hyperparams_config)
        experiments = scheduler.get_experiments_by_phase("ablation_ratio")
        completed = [e for e in experiments if e.status == "completed" and e.result]
        if not completed:
            return best_config
        best_exp = min(completed, key=lambda e: e.result.get("eval_loss") or e.result.get("final_loss") or float("inf"))
        if best_exp.config and isinstance(best_exp.config, dict):
            data = best_exp.config.get("data", {})
            best_config.setdefault("data", {})["train_dataset_path"] = data.get("train_dataset_path", "")
            best_config.setdefault("data", {})["train_dataset_prob"] = data.get("train_dataset_prob", "1.0")
        return best_config

    @staticmethod
    def _detect_model_path(model_id: str) -> str:
        from pathlib import Path
        import os
        custom_root = os.environ.get("AUTOTRAINER_MODELS_DIR")
        if custom_root:
            local_path = Path(custom_root) / model_id
            if (local_path / "config.json").exists():
                return str(local_path)
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        for suffix in (model_id, model_id.replace("/", "--")):
            candidate = hf_cache / suffix
            if (candidate / "config.json").exists():
                return str(candidate)
        return model_id
