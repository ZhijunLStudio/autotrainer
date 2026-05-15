"""Phase 2: Environment Check — verify PaddlePaddle, packages, GPU."""

from __future__ import annotations

from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler
from autotrainer.managers.env_manager import EnvManager


class EnvCheckHandler(PhaseHandler):
    """Verify environment: PaddlePaddle, packages, GPU availability."""

    def __init__(self, paddleformers_root: str = ""):
        self.pf_root = paddleformers_root

    def execute(self, ctx: PipelineContext) -> PhaseResult:
        env_mgr = EnvManager(paddleformers_root=self.pf_root)

        def notify(msg: str):
            ctx.notify("ENV_CHECK", msg)

        notify("Checking environment...")
        env_status = env_mgr.check_environment()

        # Report issues
        if env_status.issues:
            issue_text = "\n".join(f"  - {i}" for i in env_status.issues)
            if not ctx.confirm(f"Environment issues found:\n{issue_text}\nContinue anyway?"):
                return PhaseResult(Phase.ENV_CHECK, PhaseStatus.FAILED, "Environment check failed, user aborted.")

        # Report suggestions
        for sug in env_status.suggestions:
            notify(f"Suggestion: {sug}")

        # Interactive package upgrades
        if env_status.packages:
            missing = [k for k, v in env_status.packages.items() if not v]
            if missing:
                choice = ctx.prompt("Missing packages. Install?", ["yes", "skip"])
                if choice == "yes":
                    for pkg in missing:
                        try:
                            env_mgr.install_package(pkg)
                        except Exception as e:
                            notify(f"Failed to install {pkg}: {e}")

        notify(
            f"Environment ready. GPU={env_status.gpu_count}, "
            f"PaddlePaddle={env_status.paddle_version or 'not found'}, "
            f"PaddleFormers={env_status.paddleformers_version or 'not found'}"
        )
        return PhaseResult(Phase.ENV_CHECK, PhaseStatus.COMPLETED, "Environment check complete.")
