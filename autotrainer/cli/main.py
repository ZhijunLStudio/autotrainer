"""CLI root — Click-based command group for autotrainer."""

from __future__ import annotations

import click

from autotrainer import __version__


@click.group()
@click.version_option(version=__version__, prog_name="autotrainer")
def cli():
    """PaddleFormers Automated Training System.

    LLM-driven automated training for PaddleFormers models.
    Supports environment setup, data management, training, ablation experiments,
    evaluation, and reporting.
    """
    pass


@cli.command()
def init():
    """Initialize AutoTrainer config file (~/.autotrainer/config.yaml)."""
    from autotrainer.config import init_config_file

    path = init_config_file()
    click.echo(f"Config file created at: {path}")
    click.echo("Edit it to set paddleformers_root, LLM API, and training defaults.")


@cli.command()
@click.option("--task", type=click.Choice(["paddleocr-vl"]), required=True, help="Task to run")
@click.option("--gpus", type=str, default=None, help='GPU IDs (e.g., "0,1,2,3" or "all")')
@click.option("--resume", is_flag=True, default=False, help="Resume from last checkpoint")
@click.option("--no-tui", is_flag=True, default=False, help="Run without TUI (headless mode)")
@click.option("--config-override", type=click.Path(exists=True), default=None, help="Config YAML override file")
def run(task: str, gpus: str | None, resume: bool, no_tui: bool, config_override: str | None):
    """Start automated training pipeline."""
    from autotrainer.cli.run_cmd import run_command

    run_command(task=task, gpus=gpus, resume=resume, no_tui=no_tui, config_override=config_override)


@cli.command()
@click.option("--path", "paths", multiple=True, required=True, help="Data file or directory (can repeat)")
@click.option("--output-dir", type=str, default=None, help="Output directory (default: ./autotrainer_output/)")
@click.option("--profile-only", is_flag=True, default=False, help="Only profile existing JSONL (no conversion)")
@click.option("--split-only", is_flag=True, default=False, help="Only split existing JSONL into train/val/test")
@click.option("--parallel", type=int, default=1, show_default=True,
              help="Number of concurrent dataset workers")
@click.option("--script", "custom_script", type=click.Path(exists=True), default=None,
              help="Use a pre-written conversion script (for manual fix iteration)")
def data(paths: tuple[str, ...], output_dir: str | None, profile_only: bool, split_only: bool,
         parallel: int, custom_script: str | None):
    """Process training data with LLM-driven format conversion.

    \b
    Accepts any format: JSONL, JSON, CSV, TSV, Parquet, XML, ZIP, directories.
    The LLM inspects your data samples and writes a conversion script
    to transform it into erniekit JSONL format automatically.
    Two-phase execution: quick 100-row validation, then full run with auto-scaled timeout.

    \b
    Examples:
      autotrainer data --path /data/ocr_annotations/
      autotrainer data --path /data/a.parquet --path /data/b.csv
      autotrainer data --path /data/processed.jsonl --profile-only
      autotrainer data --path /data/cleaned.jsonl --split-only
      # After a failure, edit the saved script and re-run:
      autotrainer data --path /data/dataset/ --script ./autotrainer_output/dataset/convert_script.py
    """
    from autotrainer.cli.data_cmd import data_command
    data_command(paths=list(paths), output_dir=output_dir, profile_only=profile_only,
                 split_only=split_only, custom_script=custom_script, parallel=parallel)


@cli.command()
@click.option("--work-dir", type=str, default=None, help="Work directory to check")
def status(work_dir: str | None):
    """Show current training status."""
    from autotrainer.cli.status_cmd import status_command

    status_command(work_dir=work_dir)


@cli.command()
@click.option("--work-dir", type=str, default=None, help="Work directory to resume from")
def resume(work_dir: str | None):
    """Resume a previously interrupted training pipeline."""
    from autotrainer.cli.resume_cmd import resume_command

    resume_command(work_dir=work_dir)


@cli.command()
@click.option("--work-dir", type=str, default=None, help="Work directory containing experiments")
@click.option("--format", "fmt", type=click.Choice(["text", "html", "json"]), default="text", help="Report format")
def report(work_dir: str | None, fmt: str):
    """Generate experiment report."""
    from autotrainer.cli.report_cmd import report_command

    report_command(work_dir=work_dir, fmt=fmt)
