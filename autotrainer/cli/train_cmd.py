"""Train command — launch training from processed data (V2 architecture)."""

from __future__ import annotations

import os
import sys

import click


def train_command(
    data_dir: str,
    data_path: str,
    task: str,
    gpus: str | None,
    skip_ablation: bool,
    goal: str,
    resume: bool,
    no_tui: bool,
):
    """Launch training pipeline from processed data.

    Two-step workflow:
      Step 1: autotrainer data --path <raw_data> --output-dir ./output
      Step 2: autotrainer train --data-dir ./output --gpus 0,1
    """
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

    # Auto-detect: if data_dir has train.jsonl but no data_index.json, set data_path
    if data_dir and not data_path and os.path.isdir(data_dir):
        index_path = os.path.join(data_dir, "data_index.json")
        if not os.path.exists(index_path):
            # Look for train/val JSONL files directly
            train_jsonl = os.path.join(data_dir, "train.jsonl")
            if os.path.exists(train_jsonl):
                data_path = train_jsonl
                click.echo(f"Auto-detected train.jsonl in data-dir: {train_jsonl}")
            # Also find val.jsonl
            val_jsonl = os.path.join(data_dir, "val.jsonl")
            if os.path.exists(val_jsonl):
                cfg._eval_data_path = val_jsonl

    click.echo(f"Task: {task}")
    click.echo(f"GPUs: {gpu_ids}")
    click.echo(f"Data dir: {data_dir or '(none)'}")
    click.echo(f"Data path: {data_path or '(none)'}")
    click.echo(f"Skip ablation: {skip_ablation}")
    click.echo(f"Goal: {goal or '(none)'}")
    click.echo()

    if not no_tui:
        try:
            from autotrainer.tui.app import run_tui
            run_tui(
                task=task, gpu_ids=gpu_ids, work_dir=cfg.work_dir,
                data_dir=data_dir, data_path=data_path,
                eval_data_path=getattr(cfg, '_eval_data_path', ''),
                skip_ablation=skip_ablation, resume=resume,
            )
            return
        except ImportError:
            click.echo("TUI not available, falling back to headless mode.")

    _run_headless(
        task=task,
        gpu_ids=gpu_ids,
        cfg=cfg,
        resume=resume,
        data_dir=data_dir,
        data_path=data_path,
        eval_data_path=getattr(cfg, '_eval_data_path', ''),
        skip_ablation=skip_ablation,
        goal=goal,
    )


def _run_headless(
    task: str,
    gpu_ids: list[int],
    cfg,
    resume: bool,
    data_dir: str,
    data_path: str,
    eval_data_path: str,
    skip_ablation: bool,
    goal: str,
):
    """Run training pipeline headless using PipelineOrchestratorV2."""
    from autotrainer.orchestrator.pipeline_v2 import PipelineOrchestratorV2

    orch = PipelineOrchestratorV2(
        config=cfg,
        task=task,
        gpu_ids=gpu_ids,
        resume=resume,
        data_dir=data_dir,
        skip_ablation=skip_ablation,
    )

    # Wire callbacks
    orch.ctx.on_phase_change = lambda phase, msg: click.echo(f"\n[{phase}] {msg}")

    def _on_confirm(message: str, context: dict | None = None) -> bool:
        click.echo(f"\n  ? {message}")
        if not sys.stdin.isatty():
            click.echo("  Auto-confirming (non-TTY)...")
            return True
        return click.confirm("  Continue?", default=True)

    def _on_input(prompt: str, choices: list[str]) -> str:
        click.echo(f"\n  {prompt}")
        for i, c in enumerate(choices, 1):
            click.echo(f"  [{i}] {c}")
        return choices[0] if choices else ""

    orch.ctx.on_user_confirm = _on_confirm
    orch.ctx.on_user_input = _on_input
    orch.ctx.goal = goal
    orch.ctx.data_dir = data_dir
    if data_path:
        orch.ctx.data_path = data_path
    if eval_data_path:
        orch.ctx.eval_data_path = eval_data_path

    try:
        click.echo(f"Starting training: {task}")
        click.echo(f"Run ID: {orch.run_id}")
        orch.run()
        click.echo(f"\nTraining completed successfully!")
        click.echo(f"Run ID: {orch.run_id}")

        # Show brief summary
        snapshot = orch.store.get_full_snapshot(orch.run_id)
        exps = snapshot.get("experiments", [])
        completed = [e for e in exps if e.get("status") == "completed"]
        click.echo(f"\nExperiments: {len(exps)} total, {len(completed)} completed")

        progress = snapshot.get("progress", {})
        click.echo(f"Phases: {progress.get('completed_phases', 0)}/{progress.get('total_phases', 7)} completed")
    except Exception as e:
        click.echo(f"\nTraining failed: {e}", err=True)
        raise SystemExit(1)
