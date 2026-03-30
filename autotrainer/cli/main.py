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
@click.option("--mode", type=click.Choice(["fixed", "expand", "discover"]), required=True, help="Data mode")
@click.option("--task", type=str, required=True, help="Task name (e.g., paddleocr-vl)")
@click.option("--data-path", type=str, default=None, help="Path to existing data (for fixed mode)")
@click.option("--output-dir", type=str, default=None, help="Output directory for processed data")
@click.option("--query", type=str, default=None, help="Custom search query (for discover/expand mode)")
def data(mode: str, task: str, data_path: str | None, output_dir: str | None, query: str | None):
    """Manage datasets in 3 modes."""
    from autotrainer.cli.data_cmd import data_command

    data_command(mode=mode, task=task, data_path=data_path, output_dir=output_dir, query=query)


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
