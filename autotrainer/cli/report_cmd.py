"""Report command — generate experiment reports."""

from __future__ import annotations

import json
import os

import click


def report_command(work_dir: str | None, fmt: str):
    """Generate experiment report."""
    from autotrainer.config import AutoTrainerConfig

    cfg = AutoTrainerConfig.from_env()
    work_dir = work_dir or cfg.work_dir

    index_file = os.path.join(work_dir, "experiment_index.json")
    if not os.path.exists(index_file):
        click.echo("No experiment index found. No report to generate.")
        raise SystemExit(1)

    with open(index_file, "r") as f:
        index = json.load(f)

    experiments = index.get("experiments", [])
    if not experiments:
        click.echo("No experiments recorded.")
        raise SystemExit(0)

    if fmt == "json":
        click.echo(json.dumps(index, indent=2, ensure_ascii=False))
        return

    if fmt == "text":
        _print_text_report(experiments)
        return

    if fmt == "html":
        _generate_html_report(experiments, work_dir)
        return


def _print_text_report(experiments: list[dict]):
    """Print a text report."""
    click.echo("=" * 70)
    click.echo("  Experiment Report")
    click.echo("=" * 70)

    completed = [e for e in experiments if e.get("status") == "completed"]
    failed = [e for e in experiments if e.get("status") == "failed"]

    click.echo(f"\nTotal experiments: {len(experiments)}")
    click.echo(f"Completed: {len(completed)}")
    click.echo(f"Failed: {len(failed)}")

    # Best result
    if completed:
        best = min(
            completed,
            key=lambda x: x.get("result", {}).get("eval_loss", float("inf")),
        )
        click.echo(f"\nBest experiment: {best.get('id')}")
        result = best.get("result", {})
        click.echo(f"  Eval loss: {result.get('eval_loss', 'N/A')}")
        click.echo(f"  Steps: {result.get('total_steps', 'N/A')}")
        click.echo(f"  Time: {result.get('total_time_seconds', 0):.0f}s")
        click.echo(f"  Config diff: {best.get('config_diff', {})}")

    # All experiments table
    click.echo(f"\n{'ID':<25} {'Status':<12} {'Eval Loss':<12} {'Steps':<8} {'Time':<10}")
    click.echo("-" * 67)
    for exp in experiments:
        exp_id = exp.get("id", "?")
        status = exp.get("status", "?")
        result = exp.get("result", {})
        eval_loss = result.get("eval_loss", "-")
        steps = result.get("total_steps", "-")
        time_s = result.get("total_time_seconds", 0)

        if isinstance(eval_loss, float):
            eval_loss = f"{eval_loss:.4f}"
        if isinstance(time_s, (int, float)) and time_s > 0:
            time_str = f"{time_s:.0f}s"
        else:
            time_str = "-"

        click.echo(f"{exp_id:<25} {status:<12} {str(eval_loss):<12} {str(steps):<8} {time_str:<10}")


def _generate_html_report(experiments: list[dict], work_dir: str):
    """Generate an HTML report with basic charts."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        click.echo("matplotlib not available. Install with: pip install matplotlib")
        raise SystemExit(1)

    report_dir = os.path.join(work_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Loss comparison chart
    completed = [e for e in experiments if e.get("status") == "completed" and e.get("result", {}).get("eval_loss")]
    if completed:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [e.get("id", "?")[:20] for e in completed]
        eval_losses = [e["result"]["eval_loss"] for e in completed]
        colors = ["green" if l == min(eval_losses) else "steelblue" for l in eval_losses]
        ax.barh(names, eval_losses, color=colors)
        ax.set_xlabel("Eval Loss")
        ax.set_title("Experiment Comparison")
        ax.invert_yaxis()
        plt.tight_layout()

        chart_path = os.path.join(report_dir, "eval_loss_comparison.png")
        fig.savefig(chart_path, dpi=100)
        plt.close(fig)
        click.echo(f"Chart saved: {chart_path}")

    # HTML report
    html_path = os.path.join(report_dir, "report.html")
    html = _build_html(experiments)
    with open(html_path, "w") as f:
        f.write(html)
    click.echo(f"Report saved: {html_path}")


def _build_html(experiments: list[dict]) -> str:
    """Build a simple HTML report."""
    rows = []
    for exp in experiments:
        result = exp.get("result", {})
        eval_loss = result.get("eval_loss", "-")
        rows.append(
            f"<tr>"
            f"<td>{exp.get('id', '?')}</td>"
            f"<td>{exp.get('status', '?')}</td>"
            f"<td>{eval_loss}</td>"
            f"<td>{result.get('total_steps', '-')}</td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html>
<head><title>AutoTrainer Report</title></head>
<body>
<h1>Experiment Report</h1>
<p>Total experiments: {len(experiments)}</p>
<img src="eval_loss_comparison.png" alt="Eval Loss Comparison" style="max-width:800px">
<h2>Experiments</h2>
<table border="1" cellpadding="8" cellspacing="0">
<tr><th>ID</th><th>Status</th><th>Eval Loss</th><th>Steps</th></tr>
{''.join(rows)}
</table>
</body>
</html>"""
