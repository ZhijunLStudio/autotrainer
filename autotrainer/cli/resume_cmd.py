"""Resume command — resume a previously interrupted pipeline."""

from __future__ import annotations

import os

import click


def resume_command(work_dir: str | None):
    """Resume a previously interrupted training pipeline."""
    from autotrainer.config import AutoTrainerConfig
    from autotrainer.orchestrator.recovery import RecoveryManager

    cfg = AutoTrainerConfig.from_env()
    work_dir = work_dir or cfg.work_dir

    recovery = RecoveryManager(work_dir)

    if not recovery.exists():
        click.echo("No recovery state found. Nothing to resume.")
        click.echo(f"Work dir: {work_dir}")
        raise SystemExit(1)

    state = recovery.load()
    if not state:
        click.echo("Recovery state file is corrupted. Cannot resume.")
        raise SystemExit(1)

    click.echo(f"Resuming task: {state.task_name}")
    click.echo(f"Experiment: {state.experiment_id}")
    click.echo(f"Phase: {state.current_phase}")
    click.echo(f"Last step: {state.last_training_step}")
    click.echo(f"GPUs: {state.gpu_ids}")
    click.echo()

    # Find last checkpoint
    if state.checkpoint_paths:
        click.echo("Checkpoints found:")
        for cp in state.checkpoint_paths:
            click.echo(f"  {cp}")

    click.echo("\nStarting resume... (use TUI for full interactive experience)")
    click.echo("Run: autotrainer run --task <task> --resume")
