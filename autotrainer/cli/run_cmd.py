"""Run command — starts the automated training pipeline (V2 architecture)."""

from __future__ import annotations

import click


def run_command(
    task: str,
    gpus: str | None,
    resume: bool,
    no_tui: bool,
    config_override: str | None,
):
    """Execute the training pipeline."""
    from autotrainer.config import AutoTrainerConfig

    cfg = AutoTrainerConfig.from_env()

    # Parse GPU IDs
    gpu_ids = None
    if gpus:
        if gpus.lower() == "all":
            gpu_ids = list(range(cfg.detect_gpu_count()))
        else:
            gpu_ids = [int(g.strip()) for g in gpus.split(",")]

    if not gpu_ids:
        gpu_ids = list(range(cfg.detect_gpu_count()))

    click.echo(f"Task: {task}")
    click.echo(f"GPUs: {gpu_ids}")
    click.echo(f"Work dir: {cfg.work_dir}")
    click.echo(f"PaddleFormers: {cfg.paddleformers_root or 'not detected'}")
    click.echo(f"Resume: {resume}")
    click.echo()

    if not no_tui:
        try:
            from autotrainer.tui.app import run_tui
            run_tui(task=task, gpu_ids=gpu_ids, work_dir=cfg.work_dir, resume=resume)
            return
        except ImportError:
            click.echo("TUI not available, falling back to headless mode.")

    # Headless mode — uses V2 pipeline
    _run_headless(task=task, gpu_ids=gpu_ids, cfg=cfg, resume=resume)


def _run_headless(task: str, gpu_ids: list[int], cfg, resume: bool):
    """Run the pipeline without TUI using PipelineOrchestratorV2."""
    from autotrainer.orchestrator.pipeline_v2 import PipelineOrchestratorV2

    # Initialize V2 orchestrator
    orch = PipelineOrchestratorV2(
        config=cfg,
        task=task,
        gpu_ids=gpu_ids,
        resume=resume,
    )

    # Wire up click output as context callbacks
    orch.ctx.on_phase_change = lambda phase, msg: click.echo(f"\n[{phase}] {msg}")

    def _on_confirm(message: str, context: dict | None = None) -> bool:
        click.echo(f"\n  ? {message}")
        return click.confirm("  Continue?", default=True)

    def _on_input(prompt: str, choices: list[str]) -> str:
        click.echo(f"\n  {prompt}")
        for i, c in enumerate(choices, 1):
            click.echo(f"  [{i}] {c}")
        return choices[0] if choices else ""

    orch.ctx.on_user_confirm = _on_confirm
    orch.ctx.on_user_input = _on_input

    try:
        click.echo(f"\nStarting pipeline: {task}")
        click.echo(f"Run ID: {orch.run_id}")
        click.echo(f"Work dir: {orch.work_dir}")
        orch.run()
        click.echo(f"\nPipeline completed successfully!")
        click.echo(f"Run ID: {orch.run_id}")
    except Exception as e:
        click.echo(f"\nPipeline failed: {e}", err=True)
        raise SystemExit(1)
