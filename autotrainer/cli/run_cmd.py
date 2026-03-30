"""Run command — starts the automated training pipeline."""

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

    # Headless mode
    _run_headless(task=task, gpu_ids=gpu_ids, cfg=cfg, resume=resume, config_override=config_override)


def _run_headless(task: str, gpu_ids: list[int], cfg, resume: bool, config_override: str | None = None):
    """Run the pipeline without TUI using the PipelineOrchestrator."""
    from autotrainer.orchestrator.pipeline import PipelineOrchestrator

    # Initialize orchestrator
    orch = PipelineOrchestrator(
        config=cfg,
        task=task,
        gpu_ids=gpu_ids,
        resume=resume,
    )

    # Wire up click output as callbacks
    orch.on_phase_change = lambda phase, msg: click.echo(f"\n[{phase}] {msg}")

    def _on_confirm(message: str, context: dict) -> bool:
        click.echo(f"\n  ? {message}")
        return click.confirm("  Confirm?", default=True)

    def _on_input(prompt: str, choices: list[str]) -> str:
        click.echo(f"\n  {prompt}")
        for i, c in enumerate(choices, 1):
            click.echo(f"  [{i}] {c}")
        return choices[0]  # Default first choice in headless

    orch.on_user_confirm = _on_confirm
    orch.on_user_input = _on_input

    # Phase 0: Confirm task
    goal = click.prompt("  Training goal (e.g., 'accuracy > 0.85 on DocVQA')", default="", show_default=False)
    data_path = click.prompt("  Data path", default="", show_default=False)

    orch.confirm_task(goal=goal, data_path=data_path)

    # Run remaining phases
    try:
        orch.run()
        click.echo("\nPipeline completed successfully!")
    except Exception as e:
        click.echo(f"\nPipeline failed: {e}", err=True)
        raise SystemExit(1)
