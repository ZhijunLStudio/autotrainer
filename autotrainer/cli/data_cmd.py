"""Data command — manage datasets in 3 modes."""

from __future__ import annotations

import json
import os

import click


def data_command(mode: str, task: str, data_path: str | None, output_dir: str | None, query: str | None = None):
    """Execute the data management command."""
    from autotrainer.config import AutoTrainerConfig
    from autotrainer.skills.data_intel.handler import DataIntelHandler

    cfg = AutoTrainerConfig.from_env()
    cache_dir = output_dir or cfg.work_dir
    handler = DataIntelHandler(cache_dir=cache_dir)

    if mode == "fixed":
        _handle_fixed(handler, data_path, task, output_dir)
    elif mode in ("expand", "discover"):
        _handle_search(handler, task, data_path, cfg, expand=(mode == "expand"), custom_query=query)


def _handle_fixed(handler, data_path: str | None, task: str, output_dir: str | None):
    """Mode 1: Validate, profile, and prepare existing data."""
    if not data_path:
        click.echo("Error: --data-path is required for fixed mode.", err=True)
        raise SystemExit(1)

    click.echo(f"Validating and profiling: {data_path}")
    result = handler.handle_fixed_mode(data_path, task)

    # Validation
    validation = result.get("validation", {})
    if validation.get("valid"):
        click.echo("  [OK] Data format is valid")
    else:
        click.echo("  [FAIL] Data format errors:")
        for err in validation.get("errors", [])[:10]:
            click.echo(f"    - {err}")

    for warn in validation.get("warnings", [])[:5]:
        click.echo(f"  [WARN] {warn}")

    # Profile
    profile = result.get("profile", {})
    click.echo(f"\n  Format: {profile.get('format', 'unknown')}")
    click.echo(f"  Samples: {profile.get('num_samples', 0)}")
    click.echo(f"  Has images: {profile.get('has_images', False)}")
    click.echo(f"  Size: {profile.get('total_size_mb', 0):.1f} MB")

    text_lens = profile.get("text_lengths", {})
    if text_lens:
        click.echo(
            f"  Text lengths: min={text_lens.get('min', 0)}, avg={text_lens.get('avg', 0)}, max={text_lens.get('max', 0)}"
        )

    for rec in result.get("recommendations", []):
        click.echo(f"  [RECOMMEND] {rec}")

    # Save profile
    if output_dir:
        profile_path = os.path.join(output_dir, "data_profile.json")
        with open(profile_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        click.echo(f"\n  Profile saved to: {profile_path}")


def _handle_search(
    handler,
    task: str,
    data_path: str | None,
    cfg,
    expand: bool = False,
    custom_query: str | None = None,
):
    """Mode 2/3: Search for datasets (expand or discover).

    If custom_query is provided, use that instead of the default task-based query.
    """
    from autotrainer.managers.data_manager import DataManager

    dm = DataManager(cache_dir=cfg.work_dir, paddleformers_root=cfg.paddleformers_root)

    # Build search query
    if custom_query:
        search_query = custom_query
        click.echo(f"Searching with custom query: {search_query}")
    else:
        search_query = f"{task} OCR training dataset"
        if expand and data_path:
            click.echo(f"Searching for additional datasets to complement: {data_path}")
        else:
            click.echo(f"Searching for datasets: {search_query}")

    # Source 1: HuggingFace Hub
    click.echo(f"\n  [1/3] Searching HuggingFace Hub for: {search_query}")
    hf_results = dm.search_hf(query=search_query, limit=10)
    click.echo(f"        Found {len(hf_results)} results")

    # Source 2: Tavily (if configured)
    tavily_results = []
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        try:
            import yaml

            with open(os.path.expanduser("~/.autotrainer/config.yaml")) as f:
                tavily_key = (yaml.safe_load(f) or {}).get("tavily_api_key", "")
        except Exception:
            pass

    click.echo(f"  [2/3] Searching Tavily...")
    if tavily_key:
        tavily_results = dm.search_tavily(query=f"{search_query} huggingface dataset", api_key=tavily_key)
        click.echo(f"        Found {len(tavily_results)} results")
    else:
        click.echo("        Skipped (no API key)")
        click.echo("        To enable: pip install autotrainer[search]")
        click.echo("        Then add tavily_api_key to ~/.autotrainer/config.yaml")

    # Source 3: Show results
    click.echo(f"  [3/3] Compiling results...")

    all_candidates = []
    for r in hf_results:
        all_candidates.append({**r, "source": "huggingface"})
    for r in tavily_results:
        if "error" not in r:
            all_candidates.append(r)

    click.echo(f"\n  Total candidates: {len(all_candidates)}")
    click.echo(f"  {'#':<4} {'ID/Title':<40} {'Source':<14} {'Info'}")
    click.echo(f"  {'-' * 80}")

    for i, c in enumerate(all_candidates[:15], 1):
        name = c.get("id") or c.get("title", "?")[:38]
        source = c.get("source", "?")
        if source == "huggingface":
            info = f"downloads={c.get('downloads', 0)}"
        else:
            info = c.get("snippet", "")[:40]
        click.echo(f"  {i:<4} {name:<40} {source:<14} {info}")

    # Download instructions
    if all_candidates:
        click.echo(f"\n  To download a dataset:")
        click.echo(f"    HuggingFace: huggingface-cli download <repo_id>")
        click.echo(f"    Then run:    autotrainer data --mode fixed --task {task} --data-path <path>")
