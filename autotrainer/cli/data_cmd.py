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
    full_pipeline: bool = False,
):
    """Execute the data management command."""
    from autotrainer.config import AutoTrainerConfig
    from autotrainer.managers.data_pipeline import DataPipeline

    cfg = AutoTrainerConfig.from_env()
    cache_dir = output_dir or os.path.join(cfg.work_dir, "data")

    # Read Tavily key from config
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        try:
            import yaml

            with open(os.path.expanduser("~/.autotrainer/config.yaml")) as f:
                tavily_key = (yaml.safe_load(f) or {}).get("tavily_api_key", "")
        except Exception:
            pass

    pipeline = DataPipeline(cache_dir=cache_dir, paddleformers_root=cfg.paddleformers_root)

    if mode == "fixed":
        _handle_fixed(pipeline, data_path, task, output_dir)

    elif mode in ("expand", "discover"):
        search_query = query or f"{task} OCR training dataset"

        if full_pipeline:
            # Full pipeline: search → select → download → clean → convert → profile → split
            results = pipeline.run_full_pipeline(
                query=search_query,
                task=task,
                target_format="erniekit",
                tavily_key=tavily_key,
                output_dir=output_dir,
            )
            if results:
                click.echo(f"\n  Pipeline complete: {len(results)} dataset(s) processed.")
                for r in results:
                    click.echo(f"    {r.dataset_name}: {r.status}")
        else:
            # Search-only mode
            _handle_search(pipeline, search_query, task, tavily_key)


def _handle_fixed(pipeline, data_path: str | None, task: str, output_dir: str | None):
    """Mode 1: Validate, clean, profile, and split existing data."""
    if not data_path:
        click.echo("Error: --data-path is required for fixed mode.", err=True)
        raise SystemExit(1)

    click.echo(f"Processing existing data: {data_path}")

    # Detect format
    fmt = pipeline.detect_format(data_path)
    click.echo(f"  Format detected: {fmt}")

    if fmt == "unknown":
        click.echo("  [WARN] Unknown format. Trying to process anyway...")

    # Clean
    if output_dir:
        clean_dir = output_dir
    else:
        clean_dir = os.path.dirname(data_path) or "."

    cleaned_path = os.path.join(clean_dir, "cleaned_" + os.path.basename(data_path))
    click.echo(f"\n  [1/3] Cleaning (dedup, bad rows)...")
    stats = pipeline.clean(data_path, cleaned_path)
    click.echo(
        f"    Input: {stats['input_lines']}, "
        f"Duplicates: {stats['duplicates']}, "
        f"JSON errors: {stats['json_errors']}, "
        f"Empty: {stats['empty_content']}, "
        f"Output: {stats['output_lines']}"
    )

    # Profile
    click.echo(f"\n  [2/3] Profiling...")
    prof = pipeline.profile(cleaned_path)
    click.echo(f"    Samples: {prof.get('num_samples', 0)}")
    click.echo(f"    Size: {prof.get('size_mb', 0)} MB")
    click.echo(f"    Has images: {prof.get('has_images', False)}")
    tl = prof.get("text_lengths", {})
    if tl:
        click.echo(f"    Text lengths: min={tl.get('min', 0)}, avg={tl.get('avg', 0)}, max={tl.get('max', 0)}")

    # Split
    click.echo(f"\n  [3/3] Splitting train/val/test...")
    split = pipeline.split(cleaned_path)
    click.echo(
        f"    train={split.get('train', {}).get('count', 0)}, "
        f"val={split.get('val', {}).get('count', 0)}, "
        f"test={split.get('test', {}).get('count', 0)}"
    )

    click.echo(f"\n  Cleaned data: {cleaned_path}")
    click.echo(f"  Train: {split.get('train', {}).get('path', '')}")
    click.echo(f"  Val:   {split.get('val', {}).get('path', '')}")
    click.echo(f"  Test:  {split.get('test', {}).get('path', '')}")


def _handle_search(pipeline, search_query: str, task: str, tavily_key: str):
    """Search-only mode (no download)."""
    click.echo(f"\n  Searching: {search_query}")
    candidates = pipeline.search(search_query, tavily_key=tavily_key)

    if not candidates:
        click.echo("  No datasets found.")
        click.echo(f"  Try: autotrainer data --mode discover --task {task} --query '<your search terms>' --full")
        return

    click.echo(f"\n  Found {len(candidates)} candidates:")
    click.echo(f"  {'#':<5} {'Name':<46} {'Source':<13} {'Info'}")
    click.echo(f"  {'-' * 90}")
    for i, c in enumerate(candidates, 1):
        click.echo(c.display(i))

    click.echo(f"\n  To run the full download+clean+convert pipeline:")
    click.echo(f"    autotrainer data --mode discover --task {task} --query '{search_query}' --full")
