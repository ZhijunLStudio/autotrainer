# OCR 训练指标与可视化系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [  ]`) syntax for tracking.

**Goal:** 建立完整的 OCR 训练指标体系和多维可视化系统，支持多轮训练成果的雷达图、柱状图、折线图、热力图对比分析。

**Architecture:** 在现有 autotrainer 架构上轻量扩展——新增 `metrics/` 模块负责 OCR 指标计算，新增 `visualization/` 模块负责图表生成，与现有 `eval_manager`、`log_parser`、`pipeline`、`report_cmd` 集成。

**Tech Stack:** Python, matplotlib (静态图), plotly (交互式图), python-Levenshtein (编辑距离), numpy

---

## File Structure

```
autotrainer/
  metrics/                          # 新增
    __init__.py                     # 导出 compute_all_metrics
    ocr_metrics.py                  # NED, CER, WER, Exact Match, TEDS
    efficiency.py                   # 衍生效率指标
    compute.py                      # 统一入口
  visualization/                    # 新增
    __init__.py
    charts.py                       # matplotlib 静态图
    interactive.py                  # plotly 交互式图
    report_generator.py             # 完整 HTML 报告
  managers/eval_manager.py          # 修改：集成新指标
  pf_integration/log_parser.py      # 修改：新增 grad_norm 解析
  managers/train_manager.py         # 修改：收集 training history
  cli/report_cmd.py                 # 修改：调用新报告生成器
  orchestrator/pipeline.py          # 修改：串联新流程
```

---

## Task 1: OCR 核心指标计算函数

**Files:**
- Create: `autotrainer/metrics/__init__.py`
- Create: `autotrainer/metrics/ocr_metrics.py`

- [ ] **Step 1: 创建 metrics 包的 `__init__.py`**

```python
# autotrainer/metrics/__init__.py
from autotrainer.metrics.compute import compute_all_metrics

__all__ = ["compute_all_metrics"]
```

- [ ] **Step 2: 创建 `ocr_metrics.py` — NED/CER/WER/Exact Match**

```python
# autotrainer/metrics/ocr_metrics.py
"""OCR-specific evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


def _levenshtein_distance(s: str, t: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s) < len(t):
        return _levenshtein_distance(t, s)
    if not t:
        return len(s)

    prev_row = list(range(len(t) + 1))
    for i, c1 in enumerate(s):
        curr_row = [i + 1]
        for j, c2 in enumerate(t):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def compute_ned(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Normalized Edit Distance. Lower is better. Range [0, 1].

    NED = mean( levenshtein(pred, ref) / max(len(pred), len(ref)) )
    """
    if not predictions:
        return 0.0
    total = 0.0
    for pred, ref in zip(predictions, references):
        dist = _levenshtein_distance(pred, ref)
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            total += dist / max_len
    return total / len(predictions)


def compute_cer(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Character Error Rate. Lower is better.

    CER = total_char_edit_distance / total_ref_chars
    """
    total_dist = 0
    total_ref_chars = 0
    for pred, ref in zip(predictions, references):
        total_dist += _levenshtein_distance(pred, ref)
        total_ref_chars += len(ref)
    if total_ref_chars == 0:
        return 0.0
    return total_dist / total_ref_chars


def compute_wer(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Word Error Rate. Lower is better.

    WER = total_word_edit_distance / total_ref_words
    """
    total_dist = 0
    total_ref_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        total_dist += _levenshtein_distance(pred_words, ref_words) if pred_words or ref_words else 0
        total_ref_words += len(ref_words)
    if total_ref_words == 0:
        return 0.0
    return total_dist / total_ref_words


def compute_exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Exact Match rate. Higher is better. Range [0, 1]."""
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p == r)
    return matches / len(predictions)


@dataclass
class OCRMetrics:
    """Container for all OCR text recognition metrics."""
    ned: float = 0.0
    cer: float = 0.0
    wer: float = 0.0
    exact_match: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "ned": round(self.ned, 6),
            "cer": round(self.cer, 6),
            "wer": round(self.wer, 6),
            "exact_match": round(self.exact_match, 6),
            "num_samples": self.num_samples,
        }


def compute_ocr_text_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
) -> OCRMetrics:
    """Compute all text recognition metrics at once."""
    return OCRMetrics(
        ned=compute_ned(predictions, references),
        cer=compute_cer(predictions, references),
        wer=compute_wer(predictions, references),
        exact_match=compute_exact_match(predictions, references),
        num_samples=len(predictions),
    )
```

- [ ] **Step 3: 创建 `ocr_metrics.py` — TEDS (表格指标)**

在 `ocr_metrics.py` 文件末尾追加：

```python
# --- Table recognition metrics ---

def compute_teds(pred_html: str, ref_html: str) -> float:
    """Tree Edit Distance Similarity for table structure.

    Simplified version: computes edit distance on tag sequences.
    Returns similarity in [0, 1] where 1 = perfect match.

    For production use, consider the dedicated 'teds' PyPI package.
    """
    import re

    def _extract_tags(html: str) -> list[str]:
        return re.findall(r"<(/?\w+)[^>]*>", html)

    pred_tags = _extract_tags(pred_html)
    ref_tags = _extract_tags(ref_html)
    if not ref_tags:
        return 1.0 if not pred_tags else 0.0

    dist = _levenshtein_distance(pred_tags, ref_tags)
    max_len = max(len(pred_tags), len(ref_tags))
    if max_len == 0:
        return 1.0
    return 1.0 - (dist / max_len)


def compute_teds_batch(
    pred_htmls: Sequence[str],
    ref_htmls: Sequence[str],
) -> float:
    """Average TEDS across a batch of table predictions."""
    if not pred_htmls:
        return 0.0
    total = sum(compute_teds(p, r) for p, r in zip(pred_htmls, ref_htmls))
    return total / len(pred_htmls)
```

- [ ] **Step 4: 验证代码可导入**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
from autotrainer.metrics.ocr_metrics import compute_ned, compute_cer, compute_wer, compute_exact_match, compute_teds

# 测试 NED
assert abs(compute_ned(['abc'], ['abc']) - 0.0) < 1e-6  # 完美匹配
assert abs(compute_ned(['abc'], ['abd']) - 1/3) < 1e-6  # 1 个字符不同
assert compute_ned(['abc'], ['xyz']) > 0.5  # 大部分不同

# 测试 CER
assert abs(compute_cer(['hello'], ['hello']) - 0.0) < 1e-6
assert abs(compute_cer(['helo'], ['hello']) - 1/5) < 1e-6

# 测试 WER
assert abs(compute_wer(['hello world'], ['hello world']) - 0.0) < 1e-6
assert abs(compute_wer(['hello earth'], ['hello world']) - 1/2) < 1e-6

# 测试 Exact Match
assert abs(compute_exact_match(['abc', 'def'], ['abc', 'xyz']) - 0.5) < 1e-6

# 测试 TEDS
assert compute_teds('<table></table>', '<table></table>') == 1.0
assert compute_teds('<table></table>', '<div></div>') < 1.0

print('All OCR metric tests passed!')
"
```

- [ ] **Step 5: Commit**

```bash
git add autotrainer/metrics/__init__.py autotrainer/metrics/ocr_metrics.py
git commit -m "feat(metrics): add OCR text recognition and table metrics (NED/CER/WER/EM/TEDS)"
```

---

## Task 2: 效率指标计算

**Files:**
- Create: `autotrainer/metrics/efficiency.py`

- [ ] **Step 1: 创建 `efficiency.py`**

```python
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
        # Map eval index back to step number
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

    # training_time_per_step
    if num_steps > 0:
        result.training_time_per_step = total_time_seconds / num_steps

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
```

- [ ] **Step 2: 验证**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
from autotrainer.metrics.efficiency import compute_efficiency_metrics

m = compute_efficiency_metrics(
    train_losses=[2.0, 1.5, 1.0, 0.5, 0.3],
    eval_losses=[2.1, 1.6, 1.1, 0.6, 0.4],
    steps=[100, 200, 300, 400, 500],
    total_time_seconds=600.0,
)
print(m.to_dict())
assert m.loss_reduction_rate > 0
assert m.overfitting_gap < 0  # train < eval means gap is negative = slight underfitting or noise
assert m.training_time_per_step == 600.0 / 500
print('Efficiency metrics test passed!')
"
```

- [ ] **Step 3: Commit**

```bash
git add autotrainer/metrics/efficiency.py
git commit -m "feat(metrics): add training efficiency metrics (convergence, overfitting gap)"
```

---

## Task 3: 统一指标计算入口

**Files:**
- Create: `autotrainer/metrics/compute.py`

- [ ] **Step 1: 创建 `compute.py`**

```python
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
    # Text recognition inputs
    predictions: Sequence[str] | None = None,
    references: Sequence[str] | None = None,
    # Table recognition inputs
    pred_htmls: Sequence[str] | None = None,
    ref_htmls: Sequence[str] | None = None,
    # Layout analysis (pre-computed)
    layout_map: float | None = None,
    # Training history
    train_losses: Sequence[float] | None = None,
    eval_losses: Sequence[float] | None = None,
    steps: Sequence[int] | None = None,
    total_time_seconds: float = 0.0,
    # Basic training metrics
    eval_loss: float | None = None,
    train_loss: float | None = None,
) -> AllMetrics:
    """Compute all available metrics from provided data.

    Only provide the inputs you have — missing inputs are skipped.
    """
    metrics = AllMetrics()

    # Text recognition
    if predictions is not None and references is not None:
        metrics.text_recognition = compute_ocr_text_metrics(predictions, references)

    # Table recognition
    if pred_htmls is not None and ref_htmls is not None:
        metrics.table_teds = compute_teds_batch(pred_htmls, ref_htmls)

    # Layout analysis
    if layout_map is not None:
        metrics.layout_map = layout_map

    # Efficiency
    if train_losses and steps:
        metrics.efficiency = compute_efficiency_metrics(
            train_losses=train_losses,
            eval_losses=eval_losses or [],
            steps=steps,
            total_time_seconds=total_time_seconds,
        )

    # Basic training
    if eval_loss is not None:
        metrics.training["eval_loss"] = eval_loss
    if train_loss is not None:
        metrics.training["train_loss"] = train_loss
    if train_loss is not None and eval_loss is not None:
        metrics.training["overfitting_gap"] = train_loss - eval_loss

    return metrics
```

- [ ] **Step 2: 验证完整流程**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
import json
from autotrainer.metrics.compute import compute_all_metrics

m = compute_all_metrics(
    predictions=['hello', 'world', 'foo'],
    references=['hello', 'worl', 'bar'],
    train_losses=[2.0, 1.0, 0.5],
    eval_losses=[2.1, 1.1, 0.6],
    steps=[100, 200, 300],
    total_time_seconds=300.0,
    eval_loss=0.6,
    train_loss=0.5,
)
print(json.dumps(m.to_dict(), indent=2))
assert m.text_recognition is not None
assert m.text_recognition.exact_match == 1/3
assert m.efficiency is not None
print('compute_all_metrics test passed!')
"
```

- [ ] **Step 3: Commit**

```bash
git add autotrainer/metrics/compute.py
git commit -m "feat(metrics): add unified compute_all_metrics() entry point"
```

---

## Task 4: 静态图表生成 (matplotlib)

**Files:**
- Create: `autotrainer/visualization/__init__.py`
- Create: `autotrainer/visualization/charts.py`

- [ ] **Step 1: 创建 visualization 包 `__init__.py`**

```python
# autotrainer/visualization/__init__.py
from autotrainer.visualization.charts import generate_all_charts
from autotrainer.visualization.interactive import generate_interactive_report
from autotrainer.visualization.report_generator import generate_full_report

__all__ = ["generate_all_charts", "generate_interactive_report", "generate_full_report"]
```

- [ ] **Step 2: 创建 `charts.py` — matplotlib 静态图**

```python
# autotrainer/visualization/charts.py
"""Static chart generation using matplotlib."""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def generate_radar_chart(
    experiments: dict[str, dict[str, float]],
    output_path: str,
    title: str = "OCR Model Capability Radar",
) -> str:
    """Generate a radar chart comparing multiple experiments across metrics.

    Args:
        experiments: {exp_name: {metric_name: normalized_value_0_to_1, ...}, ...}
        output_path: Path to save PNG.
        title: Chart title.
    Returns:
        Path to the saved PNG file.
    """
    if not experiments:
        return ""

    # Get metric names from first experiment
    first_exp = next(iter(experiments.values()))
    categories = list(first_exp.keys())
    n = len(categories)
    if n < 3:
        return ""

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    for idx, (exp_name, metrics) in enumerate(experiments.items()):
        values = [metrics.get(cat, 0.0) for cat in categories]
        values += values[:1]  # Close
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
    """Generate bar charts comparing experiments across each metric.

    Args:
        experiments: List of experiment dicts with 'experiment_id' and metric values.
        metrics_keys: List of metric key paths (e.g., 'text_recognition.ned').
        output_dir: Directory to save PNGs.
    Returns:
        Dict mapping metric_key -> output PNG path.
    """
    saved = {}
    for metric_key in metrics_keys:
        fig, ax = plt.subplots(figsize=(10, max(4, len(experiments) * 0.5)))

        exp_ids = []
        values = []
        for exp in experiments:
            # Navigate nested dict (e.g., "text_recognition.ned")
            val = exp
            for part in metric_key.split("."):
                val = val.get(part, {}) if isinstance(val, dict) else 0.0
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
    """Generate training curve line plots.

    Dual-axis: loss on left y-axis, learning rate on right y-axis.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(steps, train_losses, "b-", linewidth=1.5, label="Train Loss", alpha=0.8)
    if eval_losses:
        # eval_losses may have fewer points; create matching step indices
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
            val = exp
            for part in metric_key.split("."):
                val = val.get(part, {}) if isinstance(val, dict) else 0.0
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

    # Annotate cells
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

    # Define which metrics to chart
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

    # Radar chart (normalize metrics to 0-1)
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
    """Normalize experiment metrics to [0, 1] for radar chart.

    For error-rate metrics (NED, CER, WER), invert: 1 - value.
    Higher is always "better" in radar.
    """
    radar = {}
    mappings = {
        "NED": ("text_recognition.ned", True),      # invert: lower is better
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
```

- [ ] **Step 3: 验证图表生成**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
from autotrainer.visualization.charts import generate_all_charts
import os, tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    experiments = [
        {
            'experiment_id': 'exp_001',
            'text_recognition': {'ned': 0.05, 'cer': 0.03, 'wer': 0.05, 'exact_match': 0.82},
            'table_recognition': {'teds': 0.88},
            'layout_analysis': {'map': 0.91},
            'training': {'eval_loss': 0.42},
        },
        {
            'experiment_id': 'exp_002',
            'text_recognition': {'ned': 0.08, 'cer': 0.05, 'wer': 0.07, 'exact_match': 0.78},
            'table_recognition': {'teds': 0.85},
            'layout_analysis': {'map': 0.89},
            'training': {'eval_loss': 0.50},
        },
    ]
    charts = generate_all_charts(experiments, tmpdir)
    print(f'Generated {len(charts)} charts:')
    for name, path in sorted(charts.items()):
        size = os.path.getsize(path)
        print(f'  {name}: {path} ({size} bytes)')
    assert len(charts) > 0
    print('Chart generation test passed!')
"
```

- [ ] **Step 4: Commit**

```bash
git add autotrainer/visualization/__init__.py autotrainer/visualization/charts.py
git commit -m "feat(viz): add matplotlib static charts (radar, bar, line, heatmap)"
```

---

## Task 5: 交互式图表生成 (plotly)

**Files:**
- Create: `autotrainer/visualization/interactive.py`

- [ ] **Step 1: 创建 `interactive.py`**

```python
# autotrainer/visualization/interactive.py
"""Interactive chart generation using plotly."""

from __future__ import annotations

import os
from typing import Any, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_interactive_radar(
    experiments: dict[str, dict[str, float]],
    output_path: str,
    title: str = "OCR Model Capability Radar",
) -> str:
    """Generate an interactive radar chart with plotly."""
    if not experiments:
        return ""

    first_exp = next(iter(experiments.values()))
    categories = list(first_exp.keys()) + [list(first_exp.keys())[0]]  # Close

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
            val = exp
            for part in metric_key.split("."):
                val = val.get(part, {}) if isinstance(val, dict) else 0.0
            values.append(val if isinstance(val, (int, float)) else 0.0)
        fig.add_trace(go.Bar(name=metric_key, x=exp_ids, y=values, text=[f"{v:.4f}" for v in values]))

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
                             mode="lines", name="Train Loss", line=dict(color="blue")))
    if eval_losses:
        eval_steps = list(steps[::max(1, len(steps) // len(eval_losses))][:len(eval_losses)])
        fig.add_trace(go.Scatter(x=eval_steps, y=list(eval_losses),
                                 mode="lines", name="Eval Loss", line=dict(color="red")))

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
    import numpy as np

    exp_ids = [exp.get("experiment_id", f"exp_{i}") for i, exp in enumerate(experiments)]
    data = []
    for metric_key in metrics_keys:
        row = []
        for exp in experiments:
            val = exp
            for part in metric_key.split("."):
                val = val.get(part, {}) if isinstance(val, dict) else 0.0
            row.append(val if isinstance(val, (int, float)) else 0.0)
        data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=data, x=exp_ids, y=metrics_keys,
        colorscale="RdYlGn", text=[[f"{v:.4f}" for v in row] for row in data],
        texttemplate="%{text}", hovertemplate="Exp: %{x}<br>Metric: %{y}<br>Value: %{text}<extra></extra>",
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

    # Interactive bar chart
    bar_path = generate_interactive_bars(experiments, bar_metrics,
                                         os.path.join(output_dir, "interactive_bars.html"))
    if bar_path:
        charts["bars"] = bar_path

    # Interactive heatmap
    available = [m for m in bar_metrics
                 if any(_nested_get(exp, m) is not None for exp in experiments)]
    if available and len(experiments) > 1:
        hm_path = generate_interactive_heatmap(experiments, available,
                                               os.path.join(output_dir, "interactive_heatmap.html"))
        if hm_path:
            charts["heatmap"] = hm_path

    # Interactive radar
    if len(experiments) > 1:
        radar_data = {}
        for exp in experiments:
            exp_id = exp.get("experiment_id", "unknown")
            radar_data[exp_id] = _normalize_for_radar(exp)
        radar_path = generate_interactive_radar(radar_data,
                                                os.path.join(output_dir, "interactive_radar.html"))
        if radar_path:
            charts["radar"] = radar_path

    # Interactive training curves
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
```

- [ ] **Step 2: 验证**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
pip install plotly 2>/dev/null || true
python -c "
from autotrainer.visualization.interactive import generate_interactive_report
import os, tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    experiments = [
        {'experiment_id': 'exp_001',
         'text_recognition': {'ned': 0.05, 'cer': 0.03, 'wer': 0.05, 'exact_match': 0.82},
         'table_recognition': {'teds': 0.88}, 'layout_analysis': {'map': 0.91},
         'training': {'eval_loss': 0.42}},
        {'experiment_id': 'exp_002',
         'text_recognition': {'ned': 0.08, 'cer': 0.05, 'wer': 0.07, 'exact_match': 0.78},
         'table_recognition': {'teds': 0.85}, 'layout_analysis': {'map': 0.89},
         'training': {'eval_loss': 0.50}},
    ]
    charts = generate_interactive_report(experiments, tmpdir)
    print(f'Generated {len(charts)} interactive charts:')
    for name, path in sorted(charts.items()):
        size = os.path.getsize(path)
        print(f'  {name}: {os.path.basename(path)} ({size} bytes)')
    assert len(charts) > 0
    print('Interactive chart test passed!')
"
```

- [ ] **Step 3: Commit**

```bash
git add autotrainer/visualization/interactive.py
git commit -m "feat(viz): add plotly interactive charts (radar, bar, heatmap, curves)"
```

---

## Task 6: HTML 报告生成器

**Files:**
- Create: `autotrainer/visualization/report_generator.py`

- [ ] **Step 1: 创建 `report_generator.py`**

```python
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

    # Generate all charts
    static_charts = generate_all_charts(experiments, charts_dir, training_histories)
    interactive_charts = generate_interactive_report(experiments, interactive_dir, training_histories)

    # Build HTML
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

    # --- Overview cards ---
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

    # --- Static chart sections ---
    def _chart_section(title: str, chart_keys: list[str]) -> str:
        imgs = ""
        for key in chart_keys:
            if key in static_charts:
                rel = os.path.relpath(static_charts[key], os.path.dirname(
                    list(static_charts.values())[0]) if static_charts else ".")
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

    # --- Interactive links ---
    interactive_links = ""
    for name, path in sorted(interactive_charts.items()):
        rel = os.path.relpath(path, os.path.dirname(list(interactive_charts.values())[0])
                              if interactive_charts else ".")
        interactive_links += f'<li><a href="{rel}" target="_blank">{name}</a></li>\n'

    # --- Detail table ---
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
```

- [ ] **Step 2: 验证完整报告生成**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
from autotrainer.visualization.report_generator import generate_full_report
import os, tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    experiments = [
        {'experiment_id': 'exp_001', 'status': 'completed',
         'text_recognition': {'ned': 0.05, 'cer': 0.03, 'wer': 0.05, 'exact_match': 0.82},
         'table_recognition': {'teds': 0.88}, 'layout_analysis': {'map': 0.91},
         'training': {'eval_loss': 0.42, 'train_loss': 0.38}},
        {'experiment_id': 'exp_002', 'status': 'completed',
         'text_recognition': {'ned': 0.08, 'cer': 0.05, 'wer': 0.07, 'exact_match': 0.78},
         'table_recognition': {'teds': 0.85}, 'layout_analysis': {'map': 0.89},
         'training': {'eval_loss': 0.50, 'train_loss': 0.45}},
    ]
    histories = {
        'exp_001': {'steps': [100,200,300,400,500], 'train_loss': [2.0,1.5,1.0,0.5,0.38],
                    'eval_loss': [2.1,1.6,1.1,0.6,0.42]},
        'exp_002': {'steps': [100,200,300,400,500], 'train_loss': [2.0,1.6,1.2,0.7,0.45],
                    'eval_loss': [2.1,1.7,1.3,0.8,0.50]},
    }
    report_path = generate_full_report(experiments, tmpdir, histories)
    print(f'Report generated: {report_path}')
    print(f'Size: {os.path.getsize(report_path)} bytes')
    # List all generated files
    for root, dirs, files in os.walk(tmpdir):
        for f in sorted(files):
            full = os.path.join(root, f)
            print(f'  {os.path.relpath(full, tmpdir)}: {os.path.getsize(full)} bytes')
    assert os.path.exists(report_path)
    print('Full report generation test passed!')
"
```

- [ ] **Step 3: Commit**

```bash
git add autotrainer/visualization/report_generator.py
git commit -m "feat(viz): add complete HTML report generator"
```

---

## Task 7: 集成 — 扩展 log_parser 支持 grad_norm

**Files:**
- Modify: `autotrainer/pf_integration/log_parser.py`

- [ ] **Step 1: 在 `LogMetrics` dataclass 中添加 `grad_norm` 字段**

在 `LogMetrics` class (line 15) 中添加字段：

```python
@dataclass
class LogMetrics:
    step: int | None = None
    loss: float | None = None
    lr: float | None = None
    eval_loss: float | None = None
    throughput: float | None = None
    memory_mb: float | None = None
    epoch: int | None = None
    grad_norm: float | None = None  # 新增
    raw_line: str = ""
```

- [ ] **Step 2: 在 `LogParser` 中添加 `grad_norm` 正则和解析**

在 `LogParser` class 的 pattern 区域（约 line 48 之后）添加：

```python
GRAD_NORM_PATTERN = re.compile(r"grad_norm[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)")
```

在 `parse_line` 方法中（约 line 59-115），在已有的 metric 解析逻辑之后添加：

```python
# Parse grad_norm
grad_match = self.GRAD_NORM_PATTERN.search(line)
if grad_match:
    try:
        metrics.grad_norm = float(grad_match.group(1))
    except ValueError:
        pass
```

- [ ] **Step 3: 验证**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
from autotrainer.pf_integration.log_parser import LogParser, LogMetrics
parser = LogParser()
# Test grad_norm parsing
result = parser.parse_line('step=100 loss=0.5 lr=1e-4 grad_norm=1.23 throughput=1000')
print(f'grad_norm: {result.grad_norm if result else None}')
assert result is not None
assert result.grad_norm == 1.23
print('grad_norm parsing test passed!')
"
```

- [ ] **Step 4: Commit**

```bash
git add autotrainer/pf_integration/log_parser.py
git commit -m "feat(metrics): add grad_norm parsing to LogParser"
```

---

## Task 8: 集成 — eval_manager 调用新指标

**Files:**
- Modify: `autotrainer/managers/eval_manager.py`

- [ ] **Step 1: 扩展 `EvalResult` dataclass**

在 `EvalResult`（line 16）中添加 `ocr_metrics` 字段：

```python
@dataclass
class EvalResult:
    experiment_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    ocr_metrics: dict[str, Any] = field(default_factory=dict)  # 新增：完整 OCR 指标
    inference_samples: list[dict] = field(default_factory=list)
    report_path: str = ""
    status: str = "pending"
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "metrics": self.metrics,
            "ocr_metrics": self.ocr_metrics,  # 新增
            "inference_samples": self.inference_samples,
            "report_path": self.report_path,
            "status": self.status,
            "timestamp": self.timestamp,
        }
```

- [ ] **Step 2: 添加 `run_ocr_eval` 方法**

在 `EvalManager` class 末尾（line 204 之前）添加新方法：

```python
def run_ocr_eval(
    self,
    predictions: list[str],
    references: list[str],
    pred_htmls: list[str] | None = None,
    ref_htmls: list[str] | None = None,
) -> dict:
    """Compute OCR metrics from prediction/reference pairs.

    Call this after inference when you have raw predictions and ground truth.
    Returns the full metrics dict suitable for storing in EvalResult.ocr_metrics.
    """
    from autotrainer.metrics.compute import compute_all_metrics

    metrics = compute_all_metrics(
        predictions=predictions,
        references=references,
        pred_htmls=pred_htmls,
        ref_htmls=ref_htmls,
    )
    return metrics.to_dict()
```

- [ ] **Step 3: 验证导入**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
from autotrainer.managers.eval_manager import EvalManager, EvalResult
# Verify EvalResult has ocr_metrics field
r = EvalResult(experiment_id='test', ocr_metrics={'text_recognition': {'ned': 0.05}})
d = r.to_dict()
assert 'ocr_metrics' in d
assert d['ocr_metrics']['text_recognition']['ned'] == 0.05
print('EvalManager integration test passed!')
"
```

- [ ] **Step 4: Commit**

```bash
git add autotrainer/managers/eval_manager.py
git commit -m "feat(metrics): integrate OCR metrics into EvalManager"
```

---

## Task 9: 集成 — train_manager 收集 training history

**Files:**
- Modify: `autotrainer/managers/train_manager.py`

- [ ] **Step 1: 在 `TrainingResult` 中添加 history 字段**

在 `TrainingResult` dataclass（line 36）中添加：

```python
@dataclass
class TrainingResult:
    experiment_id: str
    final_loss: float | None = None
    total_steps: int = 0
    total_time_seconds: float = 0.0
    checkpoint_path: str = ""
    log_path: str = ""
    eval_loss: float | None = None
    throughput: float | None = None
    metrics: dict = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    status: str = "pending"
    history: dict = field(default_factory=dict)  # 新增: {steps:[], train_loss:[], eval_loss:[], learning_rate:[]}

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "final_loss": self.final_loss,
            "total_steps": self.total_steps,
            "total_time_seconds": self.total_time_seconds,
            "checkpoint_path": self.checkpoint_path,
            "log_path": self.log_path,
            "eval_loss": self.eval_loss,
            "throughput": self.throughput,
            "metrics": self.metrics,
            "errors": self.errors,
            "status": self.status,
            "history": self.history,  # 新增
        }
```

- [ ] **Step 2: 在 `_run_single` 中收集 history**

在 `TrainManager._run_single` 方法中，在 `_on_stdout` 回调内（约 line 280-310），每次解析到 `LogMetrics` 后，追加到 history 列表。在方法开始处初始化 history：

```python
# 在 _run_single 方法开头，start_time 之后添加
history_steps = []
history_train_loss = []
history_eval_loss = []
history_lr = []
```

在 `_on_stdout` 回调的 metrics 解析部分（`if metrics:` 分支），追加：

```python
if metrics.step is not None:
    history_steps.append(metrics.step)
if metrics.loss is not None:
    history_train_loss.append(metrics.loss)
if metrics.eval_loss is not None:
    history_eval_loss.append(metrics.eval_loss)
if metrics.lr is not None:
    history_lr.append(metrics.lr)
```

在构建 `TrainingResult` 时（约 line 330），添加 history：

```python
result.history = {
    "steps": history_steps,
    "train_loss": history_train_loss,
    "eval_loss": history_eval_loss,
    "learning_rate": history_lr,
}
```

- [ ] **Step 3: 验证**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
from autotrainer.managers.train_manager import TrainingResult
r = TrainingResult(
    experiment_id='test',
    history={'steps': [100,200], 'train_loss': [1.0, 0.5], 'eval_loss': [], 'learning_rate': []}
)
d = r.to_dict()
assert 'history' in d
assert len(d['history']['steps']) == 2
print('TrainingResult history test passed!')
"
```

- [ ] **Step 4: Commit**

```bash
git add autotrainer/managers/train_manager.py
git commit -m "feat(metrics): collect training history in TrainingResult for curve plots"
```

---

## Task 10: 集成 — report_cmd 和 pipeline 使用新报告

**Files:**
- Modify: `autotrainer/cli/report_cmd.py`
- Modify: `autotrainer/orchestrator/pipeline.py`

- [ ] **Step 1: 修改 `report_cmd.py` — 添加 full report 选项**

在 `report_cmd.py` 的 `report_command` 函数中，在 format 分支添加 `full` 选项（约 line 30）：

```python
# 在现有的 if/elif 分支之后添加
elif fmt == "full":
    _generate_full_report(experiments, work_dir)
```

添加新函数（在文件末尾）：

```python
def _generate_full_report(experiments: list[dict], work_dir: str) -> None:
    """Generate the full OCR report with radar, bar, line, and heatmap charts."""
    from autotrainer.visualization.report_generator import generate_full_report

    report_dir = os.path.join(work_dir, "reports")
    # Load training histories from eval_results if available
    training_histories = {}
    eval_dir = os.path.join(work_dir, "eval_results")
    if os.path.isdir(eval_dir):
        for fname in os.listdir(eval_dir):
            if fname.endswith(".json"):
                with open(os.path.join(eval_dir, fname)) as f:
                    data = json.load(f)
                exp_id = data.get("experiment_id", fname.replace(".json", ""))
                if "history" in data:
                    training_histories[exp_id] = data["history"]

    report_path = generate_full_report(experiments, report_dir, training_histories)
    print(f"Full OCR report generated: {report_path}")
```

- [ ] **Step 2: 修改 `pipeline.py` — _run_phase_report 使用新报告生成器**

在 `pipeline.py` 的 `_run_phase_report` 方法中（约 line 686-742），替换原有的简单图表生成：

在方法体中找到调用 `_generate_charts` 的地方（约 line 740），替换为：

```python
# Generate full OCR report with all chart types
try:
    from autotrainer.visualization.report_generator import generate_full_report

    # Collect all experiments
    all_experiments = []
    for abl in self._state.ablation_results:
        all_experiments.append(abl)
    if self._state.full_training_result:
        all_experiments.append(self._state.full_training_result)

    # Load training histories
    training_histories = {}
    eval_dir = os.path.join(self._work_dir, "eval_results")
    if os.path.isdir(eval_dir):
        for fname in os.listdir(eval_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(eval_dir, fname)
                data = safe_read_json(fpath) or {}
                exp_id = data.get("experiment_id", fname.replace(".json", ""))
                if "history" in data:
                    training_histories[exp_id] = data["history"]

    report_path = generate_full_report(all_experiments, report_dir, training_histories)
    self._notify("report", f"Full OCR report generated: {report_path}")
except Exception as e:
    # Fallback to simple chart generation
    self._notify("report", f"Full report generation failed ({e}), falling back to basic charts")
    self._generate_charts(all_experiments, report_dir)
```

- [ ] **Step 3: 验证导入链**

```bash
cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer
python -c "
# Verify all import chains work
from autotrainer.metrics import compute_all_metrics
from autotrainer.visualization import generate_all_charts, generate_interactive_report, generate_full_report
from autotrainer.managers.eval_manager import EvalManager
from autotrainer.managers.train_manager import TrainManager, TrainingResult
from autotrainer.pf_integration.log_parser import LogParser
print('All import chains verified successfully!')
"
```

- [ ] **Step 4: Commit**

```bash
git add autotrainer/cli/report_cmd.py autotrainer/orchestrator/pipeline.py
git commit -m "feat: integrate full OCR report into report_cmd and pipeline"
```

---

## Spec Coverage Check

| Spec 要求 | 对应 Task |
|-----------|----------|
| NED/CER/WER/Exact Match 计算 | Task 1 |
| TEDS 表格指标 | Task 1 |
| 效率指标 (收敛速度/过拟合) | Task 2 |
| 统一 compute_all_metrics 入口 | Task 3 |
| matplotlib 静态图 (雷达/柱状/折线/热力图) | Task 4 |
| plotly 交互式图 | Task 5 |
| 完整 HTML 报告 | Task 6 |
| log_parser grad_norm | Task 7 |
| eval_manager 集成 | Task 8 |
| train_manager history 收集 | Task 9 |
| report_cmd + pipeline 集成 | Task 10 |
| Layout mAP | Task 3 (compute_all_metrics 接口已预留，后续接入具体计算) |
