# autotrainer/metrics/efficiency.py
"""Derived efficiency metrics for training analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class EfficiencyMetrics:
    """Container for training efficiency metrics."""
    best_step: int | None = None
    loss_reduction_rate: float = 0.0
    overfitting_gap: float = 0.0
    training_time_per_step: float = 0.0
    convergence_speed: float = 0.0  # steps to reach 90% of final improvement

    def to_dict(self) -> dict:
        return {
            "best_step": self.best_step,
            "loss_reduction_rate": round(self.loss_reduction_rate, 8),
            "overfitting_gap": round(self.overfitting_gap, 6),
            "training_time_per_step": round(self.training_time_per_step, 4),
            "convergence_speed": self.convergence_speed,
        }


def compute_efficiency_metrics(
    train_losses: Sequence[float],
    eval_losses: Sequence[float],
    steps: Sequence[int],
    total_time_seconds: float,
) -> EfficiencyMetrics:
    """Compute derived efficiency metrics from training history.

    Args:
        train_losses: Per-step training loss values.
        eval_losses: Per-eval-step eval loss values (may be fewer points).
        steps: Step numbers corresponding to train_losses.
        total_time_seconds: Total training wall-clock time.
    """
    result = EfficiencyMetrics()

    if not train_losses or not steps:
        return result

    # best_step: step with lowest eval_loss (or train_loss if no eval)
    losses_for_best = eval_losses if eval_losses else train_losses
    if losses_for_best:
        best_idx = min(range(len(losses_for_best)), key=lambda i: losses_for_best[i])
        if eval_losses and len(steps) >= len(eval_losses):
            eval_steps = steps[::max(1, len(steps) // len(eval_losses))][:len(eval_losses)]
            result.best_step = eval_steps[best_idx] if best_idx < len(eval_steps) else steps[-1]
        else:
            result.best_step = steps[best_idx]

    # loss_reduction_rate: (first_loss - last_loss) / num_steps
    first_loss = train_losses[0]
    last_loss = train_losses[-1]
    num_steps = steps[-1] - steps[0] + 1 if len(steps) > 1 else 1
    result.loss_reduction_rate = (first_loss - last_loss) / num_steps

    # overfitting_gap: train_loss - eval_loss at final step
    if train_losses and eval_losses:
        result.overfitting_gap = train_losses[-1] - eval_losses[-1]

    # training_time_per_step: total time divided by max step number
    if steps[-1] > 0:
        result.training_time_per_step = total_time_seconds / steps[-1]

    # convergence_speed: steps to reach 90% of total improvement
    if len(train_losses) > 1 and first_loss != last_loss:
        target = first_loss - 0.9 * (first_loss - last_loss)
        for i, loss in enumerate(train_losses):
            if first_loss > last_loss:  # loss decreasing
                if loss <= target:
                    result.convergence_speed = steps[i]
                    break
            else:  # loss increasing (unusual)
                if loss >= target:
                    result.convergence_speed = steps[i]
                    break
        else:
            result.convergence_speed = steps[-1]

    return result
