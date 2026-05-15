"""Phase 5: Evaluation — run evaluation metrics and inference verification."""

from __future__ import annotations

import os

from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler
from autotrainer.managers.eval_manager import EvalManager


class EvaluationHandler(PhaseHandler):
    """Evaluate trained model and run inference sample verification."""

    def __init__(self, paddleformers_root: str = ""):
        self.pf_root = paddleformers_root

    def execute(self, ctx: PipelineContext) -> PhaseResult:
        def notify(msg: str):
            ctx.notify("EVALUATION", msg)

        eval_mgr = EvalManager(paddleformers_root=self.pf_root, work_dir=ctx.work_dir)
        checkpoint_dir = os.path.join(ctx.work_dir, "checkpoints", "full-training")

        if not os.path.exists(checkpoint_dir):
            notify("No checkpoint found, skipping evaluation.")
            return PhaseResult(Phase.EVALUATION, PhaseStatus.COMPLETED, "Skipped (no checkpoint)")

        config_path = os.path.join(ctx.work_dir, "configs", "full-training.yaml")

        # Run evaluation
        if os.path.exists(config_path):
            eval_result = eval_mgr.run_eval(config_path, checkpoint_dir)
            ctx.eval_result = eval_result.to_dict()
            notify(f"Eval complete: metrics={eval_result.metrics}")

        # Inference verification
        if ctx.eval_data_path and os.path.exists(config_path):
            samples = eval_mgr.run_inference_samples(
                config_path=config_path,
                checkpoint_path=checkpoint_dir,
                data_path=ctx.eval_data_path,
                n_samples=5,
            )
            if samples:
                notify(f"Inference verification: {len(samples)} samples checked")

        return PhaseResult(Phase.EVALUATION, PhaseStatus.COMPLETED, "Evaluation complete.",
                           data={"eval_result": ctx.eval_result})
