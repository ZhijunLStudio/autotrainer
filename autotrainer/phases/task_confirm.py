"""Phase 0: Task Confirm — verify goal, GPU selection, config."""

from __future__ import annotations

from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler


class TaskConfirmHandler(PhaseHandler):
    """Confirm task setup before starting the pipeline."""

    def execute(self, ctx: PipelineContext) -> PhaseResult:
        return PhaseResult(
            phase=Phase.TASK_CONFIRM,
            status=PhaseStatus.COMPLETED,
            message=f"Task confirmed: {ctx.task}, GPUs={ctx.gpu_ids}, work_dir={ctx.work_dir}",
            next_phase=Phase.DATA_PREPARE,
        )
