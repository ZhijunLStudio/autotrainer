"""Parallel experiment agent — runs multiple experiments concurrently.

Inspired by Claude Code's forkedAgent.ts:
- Each experiment gets an isolated context (config copy, log path, state)
- Experiments run in parallel threads (training is I/O-bound waiting for GPU)
- Results are collected and merged back
- GPU allocation distributes available GPUs across experiments
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from autotrainer.managers.train_manager import TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    """Isolated context for a single parallel experiment."""

    experiment_id: str
    factor: str  # dotted key of the factor being varied
    value: float | int  # the value to test
    config_diff: dict[str, Any]  # {dotted_key: value}
    gpu_ids: list[int]  # GPUs allocated to this experiment
    status: str = "pending"
    result: TrainingResult | None = None
    error: str | None = None


@dataclass
class ParallelResult:
    """Result of running multiple experiments in parallel."""

    experiments: list[ExperimentContext] = field(default_factory=list)
    total_time_seconds: float = 0
    succeeded: int = 0
    failed: int = 0

    @property
    def completed(self) -> list[ExperimentContext]:
        return [e for e in self.experiments if e.result is not None and e.result.status == "completed"]

    @property
    def best(self) -> ExperimentContext | None:
        """Return the experiment with the lowest loss."""
        completed = self.completed
        if not completed:
            return None
        return min(
            completed,
            key=lambda e: (e.result.eval_loss or e.result.final_loss or float("inf")),
        )


class ParallelExperimentAgent:
    """Runs multiple experiments in parallel with isolated contexts.

    Usage:
        agent = ParallelExperimentAgent(gpu_ids=[0,1,2,3])
        result = agent.run_experiments(
            base_config=cfg,
            factor_values=[("finetuning.learning_rate", v) for v in [1e-5, 3e-5, 1e-4]],
            run_fn=my_train_fn,
        )
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        max_parallel: int = 0,  # 0 = auto (len(gpu_ids) // gpus_per_exp)
        gpus_per_experiment: int = 1,
    ):
        self.gpu_ids = gpu_ids or []
        self.gpus_per_experiment = max(1, gpus_per_experiment)
        self.max_parallel = max_parallel or max(1, len(self.gpu_ids) // self.gpus_per_experiment)

    def _allocate_gpus(self, slot: int) -> list[int]:
        """Allocate GPUs for a parallel slot."""
        if not self.gpu_ids:
            return []
        start = (slot * self.gpus_per_experiment) % len(self.gpu_ids)
        return [self.gpu_ids[(start + i) % len(self.gpu_ids)] for i in range(self.gpus_per_experiment)]

    def run_experiments(
        self,
        factor: str,
        values: list[float | int],
        run_fn: Callable[[dict, dict, str, list[int]], TrainingResult],
        base_config: dict,
        subset_path: str,
        max_steps: int,
    ) -> ParallelResult:
        """Run multiple experiments in parallel.

        Args:
            factor: Dotted key of the factor being varied
            values: List of values to test
            run_fn: Function to call for each experiment: run_fn(base_config, factor_changes, subset_path, gpu_ids) -> TrainingResult
            base_config: Base training config
            subset_path: Path to data subset
            max_steps: Max training steps per experiment

        Returns:
            ParallelResult with all experiment results
        """
        import time

        contexts = []
        for i, value in enumerate(values):
            exp_id = f"par-{factor.split('.')[-1]}={value}"
            gpu_ids = self._allocate_gpus(i)
            contexts.append(ExperimentContext(
                experiment_id=exp_id,
                factor=factor,
                value=value,
                config_diff={factor: value},
                gpu_ids=gpu_ids,
            ))

        start_time = time.time()

        def _run_one(ctx: ExperimentContext) -> ExperimentContext:
            """Run a single experiment in its isolated context."""
            try:
                ctx.status = "running"
                logger.info("Starting parallel experiment %s on GPUs %s", ctx.experiment_id, ctx.gpu_ids)
                result = run_fn(base_config, ctx.config_diff, subset_path, ctx.gpu_ids)
                ctx.result = result
                ctx.status = result.status
                logger.info(
                    "Experiment %s finished: status=%s loss=%s steps=%d",
                    ctx.experiment_id, result.status, result.final_loss, result.total_steps,
                )
            except Exception as e:
                ctx.status = "failed"
                ctx.error = str(e)
                logger.error("Experiment %s failed: %s", ctx.experiment_id, e)
            return ctx

        # Run in parallel
        parallelism = min(self.max_parallel, len(contexts))
        if parallelism <= 1:
            # Fallback to sequential
            for ctx in contexts:
                _run_one(ctx)
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = {executor.submit(_run_one, ctx): ctx for ctx in contexts}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error("Parallel experiment error: %s", e)

        elapsed = time.time() - start_time

        succeeded = sum(1 for c in contexts if c.result and c.result.status == "completed")
        failed = sum(1 for c in contexts if c.status == "failed")

        return ParallelResult(
            experiments=contexts,
            total_time_seconds=elapsed,
            succeeded=succeeded,
            failed=failed,
        )
