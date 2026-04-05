# autotrainer/visualization/charts.py
"""Static chart generation using matplotlib."""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_radar_chart(
    experiments: dict[str, dict[str, float]],
    output_path: str,
    title: str = "OCR Model Capability Radar",
) -> str:
    """Generate a radar chart comparing multiple experiments across metrics."""
    if not experiments:
        return ""

    first_exp = next(iter(experiments.values()))
    categories = list(first_exp.keys())
    n = len(categories)
    if n < 3:
        return ""

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    for idx, (exp_name, metrics) in enumerate(experiments.items()):
        values = [metrics.get(cat, 0.0) for cat in categories]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=exp_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_bar_charts(
    experiments: list[dict[str, Any]],
    metrics_keys: list[str],
    output_dir: str,
) -> dict[str, str]:
    """Generate bar charts comparing experiments across each metric."""
    saved = {}
    for metric_key in metrics_keys:
        fig, ax = plt.subplots(figsize=(10, max(4, len(experiments) * 0.5)))

        exp_ids = []
        values = []
        for exp in experiments:
            val = _nested_get(exp, metric_key)
            if isinstance(val, (int, float)):
                exp_ids.append(exp.get("experiment_id", "unknown"))
                values.append(val)

        if not values:
            plt.close(fig)
            continue

        colors = ["#2ecc71" if v == max(values) else "#3498db" for v in values]
        bars = ax.barh(exp_ids, values, color=colors)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

        ax.set_xlabel(metric_key)
        ax.set_title(f"Comparison: {metric_key}")
        ax.invert_yaxis()

        safe_name = metric_key.replace(".", "_")
        path = os.path.join(output_dir, f"bar_{safe_name}.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved[metric_key] = path

    return saved


def generate_training_curves(
    steps: Sequence[int],
    train_losses: Sequence[float],
    eval_losses: Sequence[float] | None,
    learning_rates: Sequence[float] | None,
    output_path: str,
    title: str = "Training Curves",
) -> str:
    """Generate training curve line plots with dual-axis for LR."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(steps, train_losses, "b-", linewidth=1.5, label="Train Loss", alpha=0.8)
    if eval_losses:
        eval_steps = steps[::max(1, len(steps) // len(eval_losses))][:len(eval_losses)]
        ax1.plot(eval_steps, eval_losses, "r-", linewidth=1.5, label="Eval Loss", alpha=0.8)

        # Shade overfitting region
        if len(train_losses) >= len(eval_losses):
            aligned_train = [train_losses[min(i * max(1, len(train_losses) // len(eval_losses)),
                                              len(train_losses) - 1)]
                             for i in range(len(eval_losses))]
            for i in range(len(eval_losses)):
                if aligned_train[i] < eval_losses[i]:
                    ax1.axvspan(eval_steps[max(0, i-1)], eval_steps[i],
                                alpha=0.1, color="red")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    if learning_rates:
        ax2 = ax1.twinx()
        ax2.plot(steps[:len(learning_rates)], learning_rates, "g--",
                 linewidth=1, label="Learning Rate", alpha=0.5)
        ax2.set_ylabel("Learning Rate", color="green")
        ax2.legend(loc="upper right")

    ax1.set_title(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_heatmap(
    experiments: list[dict[str, Any]],
    metrics_keys: list[str],
    output_path: str,
    title: str = "Metrics × Experiments Heatmap",
) -> str:
    """Generate a heatmap of metrics × experiments."""
    data = []
    exp_ids = [exp.get("experiment_id", f"exp_{i}") for i, exp in enumerate(experiments)]

    for metric_key in metrics_keys:
        row = []
        for exp in experiments:
            val = _nested_get(exp, metric_key)
            row.append(val if isinstance(val, (int, float)) else 0.0)
        data.append(row)

    if not data:
        return ""

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(max(8, len(exp_ids) * 1.5), max(4, len(metrics_keys) * 0.6)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(exp_ids)))
    ax.set_xticklabels(exp_ids, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(metrics_keys)))
    ax.set_yticklabels(metrics_keys, fontsize=9)

    for i in range(len(metrics_keys)):
        for j in range(len(exp_ids)):
            val = data[i, j]
            color = "white" if abs(val - data.mean()) > data.std() else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontweight="bold")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_charts(
    experiments: list[dict[str, Any]],
    output_dir: str,
    training_histories: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Generate all static charts. Returns dict of chart_name -> file_path."""
    charts: dict[str, str] = {}

    bar_metrics = [
        "text_recognition.ned",
        "text_recognition.cer",
        "text_recognition.wer",
        "text_recognition.exact_match",
        "table_recognition.teds",
        "layout_analysis.map",
        "training.eval_loss",
    ]

    # Bar charts
    bar_paths = generate_bar_charts(experiments, bar_metrics, output_dir)
    charts.update({f"bar_{k}": v for k, v in bar_paths.items()})

    # Heatmap
    available_metrics = [m for m in bar_metrics
                         if any(_nested_get(exp, m) is not None for exp in experiments)]
    if available_metrics and len(experiments) > 1:
        heatmap_path = generate_heatmap(experiments, available_metrics,
                                        os.path.join(output_dir, "heatmap_metrics.png"))
        if heatmap_path:
            charts["heatmap"] = heatmap_path

    # Radar chart
    if len(experiments) > 1:
        radar_data = {}
        for exp in experiments:
            exp_id = exp.get("experiment_id", "unknown")
            radar_data[exp_id] = _normalize_for_radar(exp)
        radar_path = generate_radar_chart(radar_data,
                                          os.path.join(output_dir, "radar_comparison.png"))
        if radar_path:
            charts["radar"] = radar_path

    # Training curves
    if training_histories:
        for exp_id, history in training_histories.items():
            if history.get("steps") and history.get("train_loss"):
                curve_path = generate_training_curves(
                    steps=history["steps"],
                    train_losses=history["train_loss"],
                    eval_losses=history.get("eval_loss"),
                    learning_rates=history.get("learning_rate"),
                    output_path=os.path.join(output_dir, f"curves_{exp_id}.png"),
                    title=f"Training Curves: {exp_id}",
                )
                if curve_path:
                    charts[f"curves_{exp_id}"] = curve_path

    return charts


def _nested_get(d: dict, key_path: str, default=None):
    """Navigate nested dict with dot-separated key."""
    val = d
    for part in key_path.split("."):
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return default
    return val


def _normalize_for_radar(experiment: dict) -> dict[str, float]:
    """Normalize experiment metrics to [0, 1] for radar chart."""
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
