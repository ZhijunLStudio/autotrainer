"""Data command — LLM-driven data processing pipeline.

Usage:
  # Process one or more data files/directories
  autotrainer data --path /data/ocr1.jsonl --path /data/ocr2/
  autotrainer data --path /data/ocr/ --output-dir /data/processed

  # Profile existing processed data (no LLM needed)
  autotrainer data --path /data/processed.jsonl --profile-only

  # Split existing JSONL
  autotrainer data --path /data/cleaned.jsonl --split-only
"""

from __future__ import annotations

import os

import click


def data_command(
    paths: list[str],
    output_dir: str | None,
    profile_only: bool = False,
    split_only: bool = False,
):
    """Execute the data agent pipeline."""
    from autotrainer.config import AutoTrainerConfig
    from autotrainer.managers.data_agent import DataAgent
    from autotrainer.managers.data_pipeline import DataPipeline

    cfg = AutoTrainerConfig.from_env()
    work_dir = output_dir or os.path.join(cfg.work_dir, "data")

    # Validate paths, make absolute, and expand "collection directories"
    valid_paths = []
    for p in paths:
        abs_p = os.path.abspath(p)          # always resolve to absolute path
        if not os.path.exists(abs_p):
            click.echo(f"  [WARN] Path not found: {abs_p}", err=True)
            continue
        expanded = _expand_path(abs_p)
        valid_paths.extend(expanded)

    if not valid_paths:
        click.echo("Error: no valid data paths provided.", err=True)
        raise SystemExit(1)

    dp = DataPipeline(cache_dir=work_dir)

    # Profile-only mode (no LLM, no conversion)
    if profile_only:
        for path in valid_paths:
            click.echo(f"\nProfiling: {path}")
            if not path.endswith(".jsonl"):
                click.echo("  [WARN] profile-only works on JSONL files")
                continue
            prof = dp.profile(path)
            _print_profile(prof)
        return

    # Split-only mode
    if split_only:
        for path in valid_paths:
            click.echo(f"\nSplitting: {path}")
            result = dp.split(path)
            click.echo(
                f"  train={result['train']['count']} → {result['train']['path']}\n"
                f"  val  ={result['val']['count']}   → {result['val']['path']}\n"
                f"  test ={result['test']['count']}  → {result['test']['path']}"
            )
        return

    # Full agent pipeline
    llm_client = _build_llm_client(cfg)
    if not llm_client:
        click.echo(
            "\n  [INFO] No LLM configured. Running without auto-conversion.\n"
            "  Add LLM settings to ~/.autotrainer/config.yaml for automatic\n"
            "  format conversion of non-JSONL data.\n"
        )

    from autotrainer.context.store import ContextStore
    context = ContextStore(max_tokens=cfg.context_max_tokens)

    agent = DataAgent(
        work_dir=work_dir,
        llm_client=llm_client,
        context_store=context,
    )

    agent.run(valid_paths)


def _expand_path(path: str) -> list[str]:
    """Expand a path to a list of paths to process.

    Rules:
    - If it's a file → return [path]
    - If it's a directory that contains data files at the top level → return [path]
    - If it's a directory whose top-level entries are ALL subdirectories
      (e.g. a downloads collection) → return each subdirectory
    """
    from pathlib import Path

    p = Path(path)
    if not p.is_dir():
        return [path]

    # Data file extensions we care about
    DATA_EXT = {".jsonl", ".json", ".csv", ".tsv", ".parquet", ".xml"}

    top_files = [e for e in p.iterdir() if e.is_file() and e.suffix.lower() in DATA_EXT]
    top_dirs = [e for e in p.iterdir() if e.is_dir() and not e.name.startswith(".")]

    if top_files:
        # Has data files at top level → treat as a single dataset
        return [path]

    if top_dirs and not top_files:
        # No data files at top level, only subdirs → treat each subdir as a dataset
        # Filter out obviously empty or non-data directories
        dataset_dirs = []
        for d in sorted(top_dirs):
            # Check that the subdir actually has data files somewhere inside
            has_data = any(
                f.suffix.lower() in DATA_EXT
                for f in d.rglob("*") if f.is_file()
            )
            if has_data:
                dataset_dirs.append(str(d))

        if dataset_dirs:
            click.echo(
                f"\n  Detected collection directory with {len(dataset_dirs)} datasets."
            )
            click.echo("  Will process each subdirectory as a separate dataset:\n")
            for i, d in enumerate(dataset_dirs, 1):
                click.echo(f"    [{i:>2}] {os.path.basename(d)}")
            click.echo()
            return dataset_dirs

    return [path]


def _build_llm_client(cfg):
    """Build LLM client from config if available."""
    if not cfg.llm_base_url or not cfg.llm_model:
        return None
    try:
        from autotrainer.utils.llm_client import LLMClient
        return LLMClient(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
        )
    except Exception:
        return None


def _print_profile(prof: dict):
    """Print a dataset profile summary."""
    click.echo(f"  Format:  {prof.get('format', '?')}")
    click.echo(f"  Samples: {prof.get('num_samples', 0)}")
    click.echo(f"  Size:    {prof.get('size_mb', 0)} MB")
    click.echo(f"  Images:  {prof.get('image_count', 0)}")
    tl = prof.get("text_lengths", {})
    if tl:
        click.echo(f"  Text:    avg={tl.get('avg', 0)} p95={tl.get('p95', 0)} max={tl.get('max', 0)}")
    fc = prof.get("field_coverage", {})
    if fc:
        click.echo(f"  Fields:  {fc}")
    if prof.get("sample_preview"):
        import json
        preview = json.dumps(prof["sample_preview"][0], ensure_ascii=False)
        click.echo(f"  Preview: {preview[:250]}")
