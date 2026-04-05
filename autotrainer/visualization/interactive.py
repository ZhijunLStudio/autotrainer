# autotrainer/visualization/interactive.py
"""Interactive chart generation using plotly."""

from __future__ import annotations

import os
from typing import Any, Sequence

import plotly.graph_objects as go


def generate_interactive_radar(
    experiments: dict[str, dict[str, float]],
    output_path: str,
    title: str = "OCR Model Capability Radar",
) -> str:
    """Generate an interactive radar chart with plotly."""
    if not experiments:
        return ""

    first_exp = next(iter(experiments.values()))
    categories = list(first_exp.keys()) + [list(first_exp.keys())[0]]

    fig = go.Figure()
    for exp_name, metrics in experiments.items():
        values = [metrics.get(cat, 0.0) for cat in first_exp.keys()]
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill="toself", name=exp_name,
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title, showlegend=True,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def generate_interactive_bars(
    experiments: list[dict[str, Any]],
    metrics_keys: list[str],
    output_path: str,
) -> str:
    """Generate interactive grouped bar chart."""
    exp_ids = [exp.get("experiment_id", f"exp_{i}") for i, exp in enumerate(experiments)]

    fig = go.Figure()
    for metric_key in metrics_keys:
        values = []
        for exp in experiments:
            val = _nested_get(exp, metric_key)
            values.append(val if isinstance(val, (int, float)) else 0.0)
        fig.add_trace(go.Bar(name=metric_key, x=exp_ids, y=values,
                             text=[f"{v:.4f}" for v in values]))

    fig.update_layout(
        barmode="group", title="Metrics Comparison",
        xaxis_title="Experiment", yaxis_title="Value",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def generate_interactive_curves(
    steps: Sequence[int],
    train_losses: Sequence[float],
    eval_losses: Sequence[float] | None,
    output_path: str,
    title: str = "Training Curves",
) -> str:
    """Generate interactive training curve plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(steps), y=list(train_losses),
                             mode="lines", name="Train Loss",
                             line=dict(color="blue")))
    if eval_losses:
        eval_steps = list(steps[::max(1, len(steps) // len(eval_losses))][:len(eval_losses)])
        fig.add_trace(go.Scatter(x=eval_steps, y=list(eval_losses),
                                 mode="lines", name="Eval Loss",
                                 line=dict(color="red")))

    fig.update_layout(title=title, xaxis_title="Step", yaxis_title="Loss",
                      hovermode="x unified")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def generate_interactive_heatmap(
    experiments: list[dict[str, Any]],
    metrics_keys: list[str],
    output_path: str,
    title: str = "Metrics × Experiments Heatmap",
) -> str:
    """Generate interactive heatmap."""
    exp_ids = [exp.get("experiment_id", f"exp_{i}") for i, exp in enumerate(experiments)]
    data = []
    for metric_key in metrics_keys:
        row = []
        for exp in experiments:
            val = _nested_get(exp, metric_key)
            row.append(val if isinstance(val, (int, float)) else 0.0)
        data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=data, x=exp_ids, y=metrics_keys,
        colorscale="RdYlGn",
        text=[[f"{v:.4f}" for v in row] for row in data],
        texttemplate="%{text}",
        hovertemplate="Exp: %{x}<br>Metric: %{y}<br>Value: %{text}<extra></extra>",
    ))
    fig.update_layout(title=title)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def generate_interactive_report(
    experiments: list[dict[str, Any]],
    output_dir: str,
    training_histories: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Generate all interactive charts. Returns dict of chart_name -> file_path."""
    charts: dict[str, str] = {}

    bar_metrics = [
        "text_recognition.ned", "text_recognition.cer", "text_recognition.wer",
        "text_recognition.exact_match", "table_recognition.teds",
        "layout_analysis.map", "training.eval_loss",
    ]

    bar_path = generate_interactive_bars(experiments, bar_metrics,
                                         os.path.join(output_dir, "interactive_bars.html"))
    if bar_path:
        charts["bars"] = bar_path

    available = [m for m in bar_metrics
                 if any(_nested_get(exp, m) is not None for exp in experiments)]
    if available and len(experiments) > 1:
        hm_path = generate_interactive_heatmap(experiments, available,
                                               os.path.join(output_dir, "interactive_heatmap.html"))
        if hm_path:
            charts["heatmap"] = hm_path

    if len(experiments) > 1:
        radar_data = {}
        for exp in experiments:
            exp_id = exp.get("experiment_id", "unknown")
            radar_data[exp_id] = _normalize_for_radar(exp)
        radar_path = generate_interactive_radar(radar_data,
                                                os.path.join(output_dir, "interactive_radar.html"))
        if radar_path:
            charts["radar"] = radar_path

    if training_histories:
        for exp_id, history in training_histories.items():
            if history.get("steps") and history.get("train_loss"):
                curve_path = generate_interactive_curves(
                    steps=history["steps"],
                    train_losses=history["train_loss"],
                    eval_losses=history.get("eval_loss"),
                    output_path=os.path.join(output_dir, f"interactive_curves_{exp_id}.html"),
                    title=f"Training Curves: {exp_id}",
                )
                if curve_path:
                    charts[f"curves_{exp_id}"] = curve_path

    return charts


def _nested_get(d: dict, key_path: str, default=None):
    val = d
    for part in key_path.split("."):
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return default
    return val


def _normalize_for_radar(experiment: dict) -> dict[str, float]:
    radar = {}
    mappings = {
        "NED": ("text_recognition.ned", True),
        "CER": ("text_recognition.cer", True),
        "WER": ("text_recognition.wer", True),
        "Exact Match": ("text_recognition.exact_match", False),
        "TEDS": ("table_recognition.teds", False),
        "Layout mAP": ("layout_analysis.map", False),
    }
    for label, (path, invert) in mappings.items():
        val = _nested_get(experiment, path)
        if val is not None:
            radar[label] = max(0.0, 1.0 - val) if invert else val
    return radar
