# autotrainer/metrics/compute.py
"""Unified entry point for computing all metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from autotrainer.metrics.ocr_metrics import (
    OCRMetrics,
    compute_ocr_text_metrics,
    compute_teds_batch,
)
from autotrainer.metrics.efficiency import (
    EfficiencyMetrics,
    compute_efficiency_metrics,
)


@dataclass
class AllMetrics:
    """Complete metrics container."""
    text_recognition: OCRMetrics | None = None
    table_teds: float | None = None
    layout_map: float | None = None
    efficiency: EfficiencyMetrics | None = None
    training: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.text_recognition:
            result["text_recognition"] = self.text_recognition.to_dict()
        if self.table_teds is not None:
            result["table_recognition"] = {"teds": round(self.table_teds, 6)}
        if self.layout_map is not None:
            result["layout_analysis"] = {"map": round(self.layout_map, 6)}
        if self.efficiency:
            result["efficiency"] = self.efficiency.to_dict()
        if self.training:
            result["training"] = {k: round(v, 6) if isinstance(v, float) else v
                                  for k, v in self.training.items()}
        return result


def compute_all_metrics(
    predictions: Sequence[str] | None = None,
    references: Sequence[str] | None = None,
    pred_htmls: Sequence[str] | None = None,
    ref_htmls: Sequence[str] | None = None,
    layout_map: float | None = None,
    train_losses: Sequence[float] | None = None,
    eval_losses: Sequence[float] | None = None,
    steps: Sequence[int] | None = None,
    total_time_seconds: float = 0.0,
    eval_loss: float | None = None,
    train_loss: float | None = None,
) -> AllMetrics:
    """Compute all available metrics from provided data."""
    metrics = AllMetrics()

    if predictions is not None and references is not None:
        metrics.text_recognition = compute_ocr_text_metrics(predictions, references)

    if pred_htmls is not None and ref_htmls is not None:
        metrics.table_teds = compute_teds_batch(pred_htmls, ref_htmls)

    if layout_map is not None:
        metrics.layout_map = layout_map

    if train_losses and steps:
        metrics.efficiency = compute_efficiency_metrics(
            train_losses=train_losses,
            eval_losses=eval_losses or [],
            steps=steps,
            total_time_seconds=total_time_seconds,
        )

    if eval_loss is not None:
        metrics.training["eval_loss"] = eval_loss
    if train_loss is not None:
        metrics.training["train_loss"] = train_loss
    if train_loss is not None and eval_loss is not None:
        metrics.training["overfitting_gap"] = train_loss - eval_loss

    return metrics
