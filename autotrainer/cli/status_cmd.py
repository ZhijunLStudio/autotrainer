"""Status command — show current training status."""

from __future__ import annotations

import json
import os

import click


def status_command(work_dir: str | None):
    """Display current pipeline and experiment status."""
    from autotrainer.config import AutoTrainerConfig

    cfg = AutoTrainerConfig.from_env()
    work_dir = work_dir or cfg.work_dir

    state_file = os.path.join(work_dir, "pipeline_state.json")
    index_file = os.path.join(work_dir, "experiment_index.json")

    # Pipeline state
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
        click.echo("Pipeline Status:")
        click.echo(f"  Task: {state.get('task_name', 'unknown')}")
        click.echo(f"  Current phase: {state.get('current_phase', '?')}")

        phase_states = state.get("phase_states", {})
        for phase_id, phase_state in sorted(phase_states.items()):
            status = phase_state.get("status", "?")
            click.echo(f"  Phase {phase_id}: {status}")
    else:
        click.echo("No active pipeline found.")
        click.echo(f"  Work dir: {work_dir}")

    # Experiment index
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)

        experiments = index.get("experiments", [])
        if experiments:
            click.echo(f"\nExperiments ({len(experiments)} total):")
            click.echo(f"  {'ID':<25} {'Status':<12} {'Eval Loss':<12} {'Steps':<8}")
            click.echo(f"  {'-' * 57}")

            for exp in experiments[-20:]:  # Show last 20
                exp_id = exp.get("id", "?")
                status = exp.get("status", "?")
                result = exp.get("result", {})
                eval_loss = result.get("eval_loss", "-")
                steps = result.get("total_steps", "-")

                if isinstance(eval_loss, float):
                    eval_loss = f"{eval_loss:.4f}"

                click.echo(f"  {exp_id:<25} {status:<12} {str(eval_loss):<12} {str(steps):<8}")
    else:
        click.echo("\nNo experiments recorded yet.")
