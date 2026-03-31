"""Data command — full pipeline: search → select → download → clean → convert → profile → split."""

from __future__ import annotations

import json
import os

import click


def data_command(
    mode: str,
    task: str,
    data_path: str | None,
    output_dir: str | None,
    query: str | None = None,
):
    from autotrainer.config import AutoTrainerConfig
    from autotrainer.managers.data_pipeline import DataPipeline

    cfg = AutoTrainerConfig.from_env()
    cache_dir = output_dir or os.path.join(cfg.work_dir, "data")

    # Load optional keys from config
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    modelscope_token = os.environ.get("MODELSCOPE_API_TOKEN", "")
    if not tavily_key or not modelscope_token:
        try:
            import yaml
            with open(os.path.expanduser("~/.autotrainer/config.yaml")) as f:
                file_cfg = yaml.safe_load(f) or {}
            tavily_key = tavily_key or file_cfg.get("tavily_api_key", "")
            modelscope_token = modelscope_token or file_cfg.get("modelscope_token", "")
        except Exception:
            pass

    pipeline = DataPipeline(cache_dir=cache_dir, paddleformers_root=cfg.paddleformers_root)

    if mode == "fixed":
        _handle_fixed(pipeline, data_path, task, output_dir)
    else:
        # discover / expand: always interactive
        search_query = query or f"{task} OCR training dataset"
        pipeline.run_full_pipeline(
            query=search_query,
            task=task,
            target_format="erniekit",
            tavily_key=tavily_key,
            modelscope_token=modelscope_token,
            output_dir=output_dir,
        )


def _handle_fixed(pipeline, data_path: str | None, task: str, output_dir: str | None):
    """Mode fixed: clean → profile → split existing data."""
    if not data_path:
        click.echo("Error: --data-path is required for fixed mode.", err=True)
        raise SystemExit(1)

    click.echo(f"Processing: {data_path}")
    fmt = pipeline.detect_format(data_path)
    click.echo(f"  Format: {fmt}")

    out_dir = output_dir or os.path.dirname(data_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    cleaned = os.path.join(out_dir, "cleaned_" + os.path.basename(data_path))

    click.echo(f"\n[1/3] Cleaning (dedup, bad JSON, empty)...")
    stats = pipeline.clean(data_path, cleaned)
    click.echo(
        f"  {stats['input_lines']} in → "
        f"dupes={stats['duplicates']}, bad_json={stats['json_errors']}, empty={stats['empty_content']} → "
        f"{stats['output_lines']} out"
    )

    click.echo(f"\n[2/3] Profiling...")
    prof = pipeline.profile(cleaned)
    tl = prof.get("text_lengths", {})
    click.echo(f"  {prof.get('num_samples', 0)} samples, {prof.get('size_mb', 0)} MB, images={prof.get('image_count', 0)}")
    if tl:
        click.echo(f"  text: min={tl['min']}, avg={tl['avg']}, p95={tl['p95']}, max={tl['max']}")

    click.echo(f"\n[3/3] Splitting train/val/test (90/5/5)...")
    split = pipeline.split(cleaned)
    click.echo(
        f"  train={split['train']['count']} → {split['train']['path']}\n"
        f"  val  ={split['val']['count']}   → {split['val']['path']}\n"
        f"  test ={split['test']['count']}  → {split['test']['path']}"
    )
