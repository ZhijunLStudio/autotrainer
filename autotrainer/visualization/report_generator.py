# autotrainer/visualization/report_generator.py
"""Complete HTML report generator combining static and interactive charts."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from autotrainer.visualization.charts import generate_all_charts
from autotrainer.visualization.interactive import generate_interactive_report


def generate_full_report(
    experiments: list[dict[str, Any]],
    output_dir: str,
    training_histories: dict[str, dict] | None = None,
) -> str:
    """Generate a complete HTML report with all charts.

    Args:
        experiments: List of experiment result dicts.
        output_dir: Directory to write report files.
        training_histories: Optional {exp_id: {steps:[], train_loss:[], eval_loss:[]}}.

    Returns:
        Path to the generated HTML report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    charts_dir = os.path.join(output_dir, "charts")
    interactive_dir = os.path.join(output_dir, "interactive")

    static_charts = generate_all_charts(experiments, charts_dir, training_histories)
    interactive_charts = generate_interactive_report(experiments, interactive_dir, training_histories)

    html = _build_html(experiments, static_charts, interactive_charts)

    report_path = os.path.join(output_dir, "ocr_training_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path


def _build_html(
    experiments: list[dict[str, Any]],
    static_charts: dict[str, str],
    interactive_charts: dict[str, str],
) -> str:
    """Build the complete HTML report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Overview cards
    overview_cards = ""
    for exp in experiments:
        exp_id = exp.get("experiment_id", "unknown")
        status = exp.get("status", "unknown")
        tr = exp.get("text_recognition", {})
        ned = tr.get("ned", "-")
        cer = tr.get("cer", "-")
        em = tr.get("exact_match", "-")
        eval_loss = exp.get("training", {}).get("eval_loss", "-")

        status_color = "#2ecc71" if status == "completed" else "#e74c3c"
        overview_cards += f"""
        <div class="card">
            <h3>{exp_id}</h3>
            <span class="status" style="background:{status_color}">{status}</span>
            <table>
                <tr><td>NED</td><td>{ned if ned == '-' else f'{ned:.4f}'}</td></tr>
                <tr><td>CER</td><td>{cer if cer == '-' else f'{cer:.4f}'}</td></tr>
                <tr><td>Exact Match</td><td>{em if em == '-' else f'{em:.4f}'}</td></tr>
                <tr><td>Eval Loss</td><td>{eval_loss if eval_loss == '-' else f'{eval_loss:.4f}'}</td></tr>
            </table>
        </div>"""

    # Chart sections
    def _chart_section(title: str, chart_keys: list[str]) -> str:
        imgs = ""
        for key in chart_keys:
            if key in static_charts:
                rel = os.path.relpath(static_charts[key],
                                      os.path.dirname(list(static_charts.values())[0])
                                      if static_charts else ".")
                imgs += f'<img src="{rel}" alt="{key}" style="max-width:100%;margin:10px 0;">\n'
        if not imgs:
            return ""
        return f"<section><h2>{title}</h2>{imgs}</section>"

    radar_section = _chart_section("Radar Chart — Multi-dimensional Capability",
                                   [k for k in static_charts if k == "radar"])
    bar_section = _chart_section("Bar Charts — Metric Comparison",
                                 [k for k in static_charts if k.startswith("bar_")])
    curves_section = _chart_section("Training Curves",
                                    [k for k in static_charts if k.startswith("curves_")])
    heatmap_section = _chart_section("Heatmap — Metrics × Experiments",
                                     [k for k in static_charts if k == "heatmap"])

    # Interactive links
    interactive_links = ""
    for name, path in sorted(interactive_charts.items()):
        rel = os.path.relpath(path, os.path.dirname(list(interactive_charts.values())[0])
                              if interactive_charts else ".")
        interactive_links += f'<li><a href="{rel}" target="_blank">{name}</a></li>\n'

    # Detail table
    detail_rows = ""
    all_keys = set()
    for exp in experiments:
        _collect_keys(exp, "", all_keys)
    sorted_keys = sorted(all_keys)

    header = "<tr><th>Experiment</th>" + "".join(f"<th>{k}</th>" for k in sorted_keys) + "</tr>"
    for exp in experiments:
        exp_id = exp.get("experiment_id", "unknown")
        cells = ""
        for key in sorted_keys:
            val = _nested_get(exp, key)
            cells += f"<td>{val:.4f}</td>" if isinstance(val, float) else f"<td>{val or '-'}</td>"
        detail_rows += f"<tr><td><b>{exp_id}</b></td>{cells}</tr>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OCR Training Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; }}
h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
.card {{ display: inline-block; background: white; border-radius: 8px; padding: 15px; margin: 10px;
         box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 200px; vertical-align: top; }}
.card h3 {{ margin-top: 0; }}
.status {{ color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
section {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
a {{ color: #3498db; }}
.interactive-links {{ columns: 2; }}
</style>
</head>
<body>
<h1>OCR Training Report</h1>
<p>Generated: {timestamp} | Experiments: {len(experiments)}</p>

<section><h2>Overview</h2>{overview_cards}</section>

{radar_section}
{bar_section}
{curves_section}
{heatmap_section}

<section><h2>Interactive Charts</h2>
<ul class="interactive-links">{interactive_links}</ul>
</section>

<section><h2>Detail Table</h2>
<table>{header}{detail_rows}</table>
</section>
</body>
</html>"""


def _nested_get(d: dict, key_path: str, default=None):
    val = d
    for part in key_path.split("."):
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return default
    return val


def _collect_keys(d: dict, prefix: str, result: set):
    for k, v in d.items():
        if k in ("experiment_id", "timestamp", "status"):
            continue
        full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            _collect_keys(v, full_key, result)
        elif isinstance(v, (int, float)):
            result.add(full_key)
