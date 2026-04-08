# Multi-Dataset Ratio Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-dataset training ratio optimization, experiment scheduler with step-level resume, checkpoint GC, expanded metric extraction, and progress preview to AutoTrainer.

**Architecture:** Introduce an `ExperimentScheduler` that owns a persistent experiment queue, replacing scattered experiment state in pipeline.py and recovery.py. Pipeline builds experiment specs and delegates execution to the scheduler. Multi-dataset ratio ablation is a new skill that generates experiment specs for the scheduler.

**Tech Stack:** Python 3.11+, dataclasses, JSON persistence, existing autotrainer components (TrainManager, ConfigBuilder, LogParser, PhaseManager), Textual TUI.

---

## File Map

| File | Role |
|---|---|
| `autotrainer/orchestrator/scheduler.py` | **NEW** — ExperimentScheduler + CheckpointGC |
| `autotrainer/skills/data_ratio_ablation/__init__.py` | **NEW** — empty package init |
| `autotrainer/skills/data_ratio_ablation/SKILL.md` | **NEW** — skill prompt |
| `autotrainer/skills/data_ratio_ablation/handler.py` | **NEW** — ratio ablation logic |
| `autotrainer/pf_integration/log_parser.py` | MODIFY — generic eval metric extraction |
| `autotrainer/managers/train_manager.py` | MODIFY — resume_from_checkpoint param |
| `autotrainer/pf_integration/config_builder.py` | MODIFY — multi-dataset config |
| `autotrainer/orchestrator/pipeline.py` | MODIFY — use scheduler, multi-dataset Phase 1, cleanup hooks |
| `autotrainer/tui/widgets/status_bar.py` | MODIFY — experiment progress |
| `autotrainer/tui/app.py` | MODIFY — experiment progress display |

---

### Task 1: LogParser — Expand Metric Extraction

**Files:**
- Modify: `autotrainer/pf_integration/log_parser.py`
- Create: `tests/test_log_parser.py`

**Why first:** All downstream tasks depend on metrics. Making the parser generic early prevents rework.

- [ ] **Step 1: Write failing tests for new LogParser features**

```python
# tests/test_log_parser.py
from autotrainer.pf_integration.log_parser import LogParser, LogMetrics

def test_parse_eval_ppl():
    parser = LogParser()
    m = parser.parse_line("eval_loss: 1.234, eval_ppl: 3.435")
    assert m is not None
    assert m.eval_loss == 1.234
    assert m.eval_ppl == 3.435

def test_parse_generic_eval_metrics():
    parser = LogParser()
    m = parser.parse_line("eval_loss: 0.5, eval_accuracy: 0.92, eval_bleu: 45.6, global_step: 100")
    assert m is not None
    assert m.extra_metrics == {"eval_accuracy": 0.92, "eval_bleu": 45.6}

def test_parse_no_eval_metrics():
    parser = LogParser()
    m = parser.parse_line("some random log line")
    assert m is None

def test_has_nan_detection():
    parser = LogParser()
    m = parser.parse_line("global_step: 50, loss: nan")
    assert m is not None
    assert m.has_nan is True

def test_has_nan_detection_inf():
    parser = LogParser()
    m = parser.parse_line("global_step: 50, loss: inf")
    assert m is not None
    assert m.has_nan is True

def test_loss_std_from_history():
    """loss_std is computed externally from history, not from a single line."""
    parser = LogParser()
    lines = [
        "global_step: 1, loss: 0.5",
        "global_step: 2, loss: 0.6",
        "global_step: 3, loss: 0.4",
    ]
    metrics = []
    for line in lines:
        m = parser.parse_line(line)
        if m:
            metrics.append(m)
    # loss_std is computed on the collection, not per-line
    losses = [m.loss for m in metrics if m.loss is not None]
    assert len(losses) == 3
    import statistics
    std = statistics.stdev(losses)
    assert round(std, 4) == round(0.1, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_log_parser.py -v`
Expected: FAIL — `eval_ppl`, `extra_metrics`, `has_nan` attributes don't exist on LogMetrics

- [ ] **Step 3: Update LogMetrics dataclass**

In `autotrainer/pf_integration/log_parser.py`, update the `LogMetrics` dataclass:

```python
@dataclass
class LogMetrics:
    """Extracted metrics from a training log line."""

    step: int | None = None
    loss: float | None = None
    lr: float | None = None
    eval_loss: float | None = None
    eval_ppl: float | None = None
    throughput: float | None = None
    memory_mb: float | None = None
    epoch: int | None = None
    grad_norm: float | None = None
    extra_metrics: dict = field(default_factory=dict)  # eval_accuracy, eval_bleu, etc.
    has_nan: bool = False
    raw_line: str = ""
```

- [ ] **Step 4: Add EVAL_KV_PATTERN and update parse_line**

Add to `LogParser` class:

```python
EVAL_KV_PATTERN = re.compile(r"eval_(\w+)[:\s]+([\d.eE+-]+)")
NAN_LOSS_PATTERN = re.compile(r"loss[:\s]+(?:nan|inf)", re.IGNORECASE)
```

In `parse_line`, after existing metric extraction, add:

```python
# Generic eval metrics extraction
eval_matches = list(self.EVAL_KV_PATTERN.finditer(line))
for eval_m in eval_matches:
    key = eval_m.group(1)
    try:
        val = float(eval_m.group(2))
        if key == "loss":
            metrics.eval_loss = val
        elif key == "ppl":
            metrics.eval_ppl = val
        else:
            metrics.extra_metrics[f"eval_{key}"] = val
    except ValueError:
        pass

# NaN detection
if self.NAN_LOSS_PATTERN.search(line):
    metrics.has_nan = True
```

Also update the "return None if nothing extracted" check to include `eval_ppl` and `extra_metrics`:

```python
if metrics.step is None and metrics.loss is None and metrics.eval_loss is None and not metrics.extra_metrics:
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_log_parser.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add autotrainer/pf_integration/log_parser.py tests/test_log_parser.py
git commit -m "feat(log_parser): add generic eval metric extraction, eval_ppl, has_nan"
```

---

### Task 2: ExperimentScheduler — Core Data Structures and Queue

**Files:**
- Create: `autotrainer/orchestrator/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing tests for ExperimentSpec, SchedulerStatus, and queue persistence**

```python
# tests/test_scheduler.py
import os
import json
import tempfile
from autotrainer.orchestrator.scheduler import ExperimentSpec, SchedulerStatus, ExperimentScheduler

def test_experiment_spec_defaults():
    exp = ExperimentSpec(id="test-001", phase="ablation", config={}, config_diff={})
    assert exp.status == "pending"
    assert exp.result is None
    assert exp.completed_at == ""

def test_experiment_spec_to_dict_roundtrip():
    exp = ExperimentSpec(
        id="test-001", phase="ablation",
        config={"lr": 0.001}, config_diff={"lr": 0.001},
        status="completed", result={"loss": 0.5},
    )
    d = exp.to_dict()
    restored = ExperimentSpec.from_dict(d)
    assert restored.id == "test-001"
    assert restored.status == "completed"
    assert restored.result == {"loss": 0.5}

def test_scheduler_add_and_get_pending():
    with tempfile.TemporaryDirectory() as tmpdir:
        sched = ExperimentScheduler(work_dir=tmpdir)
        exps = [
            ExperimentSpec(id="e1", phase="ablation", config={}, config_diff={}),
            ExperimentSpec(id="e2", phase="ablation", config={}, config_diff={}),
        ]
        sched.add_experiments(exps)
        pending = sched.get_next_pending()
        assert pending is not None
        assert pending.id == "e1"

def test_scheduler_skip_completed():
    with tempfile.TemporaryDirectory() as tmpdir:
        sched = ExperimentScheduler(work_dir=tmpdir)
        exps = [
            ExperimentSpec(id="e1", phase="ablation", config={}, config_diff={}, status="completed"),
            ExperimentSpec(id="e2", phase="ablation", config={}, config_diff={}),
        ]
        sched.add_experiments(exps)
        pending = sched.get_next_pending()
        assert pending.id == "e2"

def test_scheduler_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        sched1 = ExperimentScheduler(work_dir=tmpdir)
        sched1.add_experiments([
            ExperimentSpec(id="e1", phase="ablation", config={}, config_diff={}, status="completed"),
            ExperimentSpec(id="e2", phase="ablation", config={}, config_diff={}),
        ])
        # Simulate restart
        sched2 = ExperimentScheduler(work_dir=tmpdir)
        assert sched2.loaded is True
        pending = sched2.get_next_pending()
        assert pending.id == "e2"

def test_scheduler_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        sched = ExperimentScheduler(work_dir=tmpdir)
        sched.add_experiments([
            ExperimentSpec(id="e1", phase="ablation_hyper", config={}, config_diff={}, status="completed"),
            ExperimentSpec(id="e2", phase="ablation_hyper", config={}, config_diff={}, status="completed"),
            ExperimentSpec(id="e3", phase="ablation_ratio", config={}, config_diff={}),
            ExperimentSpec(id="e4", phase="ablation_ratio", config={}, config_diff={}),
        ])
        status = sched.get_status()
        assert status.total == 4
        assert status.completed == 2
        assert status.pending == 2
        assert status.phase_breakdown["ablation_hyper"]["done"] == 2
        assert status.phase_breakdown["ablation_ratio"]["total"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_scheduler.py -v`
Expected: FAIL — scheduler module doesn't exist

- [ ] **Step 3: Implement ExperimentSpec, SchedulerStatus, ExperimentScheduler**

```python
# autotrainer/orchestrator/scheduler.py
"""Experiment scheduler — manages a persistent queue of training experiments.

Supports:
- Adding experiments in batches
- Resuming from where we left off (experiment-level granularity)
- Getting overall progress status
- Checkpoint garbage collection
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable

from autotrainer.utils.file_utils import atomic_write_json, safe_read_json


QUEUE_FILE = "experiment_queue.json"


@dataclass
class ExperimentSpec:
    """Specification and state for a single experiment."""

    id: str
    phase: str                    # "ablation_hyperparams" | "ablation_ratio" | "full_training"
    config: dict                  # Full PaddleFormers config
    config_diff: dict             # Changes relative to base
    status: str = "pending"       # pending | running | completed | failed
    result: dict | None = None
    checkpoint_path: str = ""
    created_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentSpec:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SchedulerStatus:
    """Overall scheduler progress."""

    total: int
    completed: int
    failed: int
    pending: int
    current: ExperimentSpec | None
    phase_breakdown: dict         # {"ablation_hyperparams": {"total":5, "done":3}, ...}


class ExperimentScheduler:
    """Manages a persistent queue of experiments with resume support."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self._queue_file = os.path.join(work_dir, QUEUE_FILE)
        self._experiments: list[ExperimentSpec] = []
        self.loaded = False
        self._load()

    def _load(self):
        """Load queue from disk if it exists."""
        data = safe_read_json(self._queue_file)
        if data and "experiments" in data:
            self._experiments = [ExperimentSpec.from_dict(e) for e in data["experiments"]]
            self.loaded = True

    def _save(self):
        """Persist queue to disk atomically."""
        os.makedirs(self.work_dir, exist_ok=True)
        data = {
            "experiments": [e.to_dict() for e in self._experiments],
            "updated_at": datetime.now().isoformat(),
        }
        atomic_write_json(self._queue_file, data)

    def add_experiments(self, experiments: list[ExperimentSpec]):
        """Add experiments to the queue. Preserves existing completed/running state."""
        existing = {e.id: e for e in self._experiments}
        for exp in experiments:
            if exp.id in existing:
                # Keep existing state if already in queue
                continue
            if not exp.created_at:
                exp.created_at = datetime.now().isoformat()
            self._experiments.append(exp)
        self._save()

    def get_next_pending(self) -> ExperimentSpec | None:
        """Get the next pending experiment, or None if all done."""
        for exp in self._experiments:
            if exp.status == "pending":
                return exp
        return None

    def mark_running(self, exp_id: str):
        """Mark an experiment as running."""
        for exp in self._experiments:
            if exp.id == exp_id:
                exp.status = "running"
                break
        self._save()

    def mark_completed(self, exp_id: str, result: dict, checkpoint_path: str = ""):
        """Mark an experiment as completed with result."""
        for exp in self._experiments:
            if exp.id == exp_id:
                exp.status = "completed"
                exp.result = result
                exp.checkpoint_path = checkpoint_path
                exp.completed_at = datetime.now().isoformat()
                break
        self._save()

    def mark_failed(self, exp_id: str, result: dict | None = None):
        """Mark an experiment as failed."""
        for exp in self._experiments:
            if exp.id == exp_id:
                exp.status = "failed"
                exp.result = result
                exp.completed_at = datetime.now().isoformat()
                break
        self._save()

    def get_status(self) -> SchedulerStatus:
        """Get overall scheduler progress."""
        total = len(self._experiments)
        completed = sum(1 for e in self._experiments if e.status == "completed")
        failed = sum(1 for e in self._experiments if e.status == "failed")
        pending = sum(1 for e in self._experiments if e.status == "pending")

        current = None
        for e in self._experiments:
            if e.status == "running":
                current = e
                break

        # Phase breakdown
        phase_breakdown: dict[str, dict] = {}
        for exp in self._experiments:
            if exp.phase not in phase_breakdown:
                phase_breakdown[exp.phase] = {"total": 0, "done": 0, "failed": 0}
            phase_breakdown[exp.phase]["total"] += 1
            if exp.status == "completed":
                phase_breakdown[exp.phase]["done"] += 1
            elif exp.status == "failed":
                phase_breakdown[exp.phase]["failed"] += 1

        return SchedulerStatus(
            total=total,
            completed=completed,
            failed=failed,
            pending=pending,
            current=current,
            phase_breakdown=phase_breakdown,
        )

    def get_experiments_by_phase(self, phase: str) -> list[ExperimentSpec]:
        """Get all experiments for a given phase."""
        return [e for e in self._experiments if e.phase == phase]

    def clear(self):
        """Clear all experiments from the queue."""
        self._experiments.clear()
        if os.path.exists(self._queue_file):
            os.remove(self._queue_file)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_scheduler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autotrainer/orchestrator/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): add ExperimentScheduler with persistent queue and resume"
```

---

### Task 3: ExperimentScheduler — Execution and CheckpointGC

**Files:**
- Modify: `autotrainer/orchestrator/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing tests for run_next and CheckpointGC**

```python
# Add to tests/test_scheduler.py

def test_run_next_calls_train_manager():
    """run_next should call train_manager and update state."""
    # We'll use a mock-like approach
    from unittest.mock import MagicMock
    from autotrainer.managers.train_manager import TrainingResult

    with tempfile.TemporaryDirectory() as tmpdir:
        sched = ExperimentScheduler(work_dir=tmpdir)
        mock_train_mgr = MagicMock()
        mock_result = TrainingResult(
            experiment_id="e1", final_loss=0.5, total_steps=100,
            status="completed", checkpoint_path=os.path.join(tmpdir, "checkpoints", "e1"),
        )
        mock_train_mgr.run_single_ablation.return_value = mock_result
        mock_config_builder = MagicMock()

        sched.add_experiments([
            ExperimentSpec(id="e1", phase="ablation", config={"data": {"train_dataset_path": "/tmp/subset.jsonl"}, "finetuning": {"max_steps": 1000, "output_dir": os.path.join(tmpdir, "checkpoints", "e1")}}, config_diff={"lr": 0.001}),
        ])

        result = sched.run_next(mock_train_mgr, mock_config_builder, gpu_ids=[0])
        assert result is not None
        assert result.experiment_id == "e1"
        status = sched.get_status()
        assert status.completed == 1

def test_checkpoint_gc_ablation():
    """cleanup_phase_checkpoints should remove checkpoint dirs but keep result.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake checkpoint dirs and result files
        exp_dir = os.path.join(tmpdir, "checkpoints", "abl-001")
        os.makedirs(os.path.join(exp_dir, "checkpoint-500"))
        with open(os.path.join(exp_dir, "result.json"), "w") as f:
            json.dump({"loss": 0.5}, f)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            f.write("lr: 0.001")

        sched = ExperimentScheduler(work_dir=tmpdir)
        sched.add_experiments([
            ExperimentSpec(id="abl-001", phase="ablation_hyperparams", config={}, config_diff={}, status="completed", checkpoint_path=exp_dir),
        ])
        sched.cleanup_phase_checkpoints("ablation_hyperparams")

        assert os.path.exists(os.path.join(exp_dir, "result.json"))
        assert os.path.exists(os.path.join(exp_dir, "config.yaml"))
        assert not os.path.exists(os.path.join(exp_dir, "checkpoint-500"))

def test_checkpoint_gc_full_training():
    """cleanup_full_training should keep best + last checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints", "full-training")
        # Create 5 fake checkpoints
        for step in [100, 200, 300, 400, 500]:
            cp = os.path.join(ckpt_dir, f"checkpoint-{step}")
            os.makedirs(cp)
            with open(os.path.join(cp, "model.safetensors"), "w") as f:
                f.write("fake model data")

        # Create result.json with eval_losses
        result_path = os.path.join(ckpt_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump({
                "eval_losses_by_step": {"100": 0.8, "200": 0.6, "300": 0.5, "400": 0.45, "500": 0.44},
            }, f)

        sched = ExperimentScheduler(work_dir=tmpdir)
        sched.cleanup_full_training(keep_best=1, keep_last=1)

        remaining = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
        # Should keep: checkpoint-300 (best eval_loss in result), checkpoint-500 (last)
        # Actually best by eval_losses_by_step: 500 (0.44) is both best and last
        # So we should keep checkpoint-500 and the next-best distinct one
        assert len(remaining) <= 2
        assert "checkpoint-500" in remaining  # last
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_scheduler.py::test_run_next_calls_train_manager tests/test_scheduler.py::test_checkpoint_gc_ablation tests/test_scheduler.py::test_checkpoint_gc_full_training -v`
Expected: FAIL — `run_next` and `cleanup_*` don't exist

- [ ] **Step 3: Implement run_next, run_all, and CheckpointGC methods**

Add to `ExperimentScheduler` in `autotrainer/orchestrator/scheduler.py`:

```python
    def run_next(
        self,
        train_manager,
        config_builder,
        gpu_ids: list[int] | None = None,
    ) -> Any:
        """Run the next pending experiment. Returns TrainingResult or None."""
        from autotrainer.managers.train_manager import TrainingResult

        exp = self.get_next_pending()
        if exp is None:
            return None

        self.mark_running(exp.id)

        # Find resume checkpoint
        resume_from = self._find_resume_checkpoint(exp)

        # Extract paths from config
        subset_path = exp.config.get("data", {}).get("train_dataset_path", "")
        max_steps = exp.config.get("finetuning", {}).get("max_steps", 1000)

        try:
            result = train_manager.run_single_ablation(
                base_config=exp.config,
                factor_changes=exp.config_diff,
                subset_path=subset_path,
                max_steps=max_steps,
                experiment_id=exp.id,
                gpu_ids=gpu_ids,
                resume_from_checkpoint=resume_from,
            )
        except Exception as e:
            self.mark_failed(exp.id, {"error": str(e)})
            return None

        if result.status == "completed":
            self.mark_completed(exp.id, result.to_dict(), result.checkpoint_path)
        else:
            self.mark_failed(exp.id, result.to_dict())

        return result

    def run_all(
        self,
        train_manager,
        config_builder,
        gpu_ids: list[int] | None = None,
        on_progress: Callable | None = None,
    ) -> list:
        """Run all pending experiments sequentially."""
        results = []
        while True:
            result = self.run_next(train_manager, config_builder, gpu_ids)
            if result is None:
                break
            results.append(result)
            if on_progress:
                on_progress(self.get_status(), result)
        return results

    def _find_resume_checkpoint(self, exp: ExperimentSpec) -> str | None:
        """Find the latest checkpoint-N in exp's output dir."""
        output_dir = exp.config.get("finetuning", {}).get("output_dir", "")
        if not output_dir or not os.path.exists(output_dir):
            return None
        checkpoints = []
        for d in os.listdir(output_dir):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, os.path.join(output_dir, d)))
                except (IndexError, ValueError):
                    pass
        if not checkpoints:
            return None
        checkpoints.sort(reverse=True)
        return checkpoints[0][1]

    # ── Checkpoint Garbage Collection ──

    def cleanup_phase_checkpoints(self, phase: str):
        """Remove checkpoint-N dirs for completed experiments in a phase.
        Keeps result.json and config.yaml."""
        for exp in self._experiments:
            if exp.phase != phase or exp.status != "completed":
                continue
            cp_path = exp.checkpoint_path
            if not cp_path or not os.path.isdir(cp_path):
                continue
            for item in os.listdir(cp_path):
                item_path = os.path.join(cp_path, item)
                if item.startswith("checkpoint-") and os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    def cleanup_full_training(self, keep_best: int = 1, keep_last: int = 1):
        """Keep only best + last checkpoints for full training."""
        ckpt_dir = os.path.join(self.work_dir, "checkpoints", "full-training")
        if not os.path.exists(ckpt_dir):
            return

        checkpoints = []
        for d in os.listdir(ckpt_dir):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, d))
                except (IndexError, ValueError):
                    pass

        if len(checkpoints) <= keep_best + keep_last:
            return

        checkpoints.sort()

        # Last ones
        keep = {c[1] for c in checkpoints[-keep_last:]}

        # Best ones — read from result.json
        best_ones = self._find_best_checkpoints(ckpt_dir, keep_best)
        keep |= best_ones

        for _, dirname in checkpoints:
            if dirname not in keep:
                shutil.rmtree(os.path.join(ckpt_dir, dirname))

    def _find_best_checkpoints(self, ckpt_dir: str, count: int) -> set[str]:
        """Find the best checkpoints by eval_loss from result.json."""
        result_path = os.path.join(ckpt_dir, "result.json")
        if not os.path.exists(result_path):
            return set()

        data = safe_read_json(result_path) or {}
        eval_losses = data.get("eval_losses_by_step", {})
        if not eval_losses:
            return set()

        # Sort by eval_loss ascending (lower is better)
        sorted_steps = sorted(eval_losses.items(), key=lambda x: x[1])
        best_steps = {int(s) for s, _ in sorted_steps[:count]}

        # Map steps to checkpoint dir names
        best_dirs = set()
        for d in os.listdir(ckpt_dir):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    if step in best_steps:
                        best_dirs.add(d)
                except (IndexError, ValueError):
                    pass
        return best_dirs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_scheduler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autotrainer/orchestrator/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): add run_next, run_all, and CheckpointGC"
```

---

### Task 4: TrainManager — Add resume_from_checkpoint Support

**Files:**
- Modify: `autotrainer/managers/train_manager.py`
- Modify: `autotrainer/pf_integration/launcher.py`

- [ ] **Step 1: Add resume_from_checkpoint param to run_single_ablation**

In `train_manager.py`, update `run_single_ablation` signature:

```python
def run_single_ablation(
    self,
    base_config: dict,
    factor_changes: dict[str, Any],
    subset_path: str,
    max_steps: int,
    experiment_id: str,
    gpu_ids: list[int] | None = None,
    resume_from_checkpoint: str | None = None,  # NEW
) -> TrainingResult:
```

After building `exp_config` and before writing config, add:

```python
if resume_from_checkpoint:
    exp_config["finetuning"]["resume_from_checkpoint"] = resume_from_checkpoint
```

- [ ] **Step 2: Update _build_command in launcher.py to pass resume_from_checkpoint**

In `launcher.py`, `_build_command` currently builds:

```python
[python, "-m", "paddle.distributed.launch", ..., pf_launcher_path, "train", config_path]
```

The resume flag is passed via the YAML config (as `resume_from_checkpoint`), so `_build_command` doesn't need changes — PaddleFormers reads it from the config file. No launcher change needed.

- [ ] **Step 3: Commit**

```bash
git add autotrainer/managers/train_manager.py
git commit -m "feat(train_manager): add resume_from_checkpoint support"
```

---

### Task 5: ConfigBuilder — Multi-Dataset Config Support

**Files:**
- Modify: `autotrainer/pf_integration/config_builder.py`
- Create: `tests/test_config_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config_builder.py
from autotrainer.pf_integration.config_builder import ConfigBuilder

def test_build_multi_dataset_config():
    cb = ConfigBuilder()
    datasets = [
        ("/data/ds1/train.jsonl", 0.5),
        ("/data/ds2/train.jsonl", 0.3),
        ("/data/ds3/train.jsonl", 0.2),
    ]
    config = cb.build_multi_dataset_config(
        model_path="/models/paddleocr-vl",
        datasets=datasets,
    )
    assert config["data"]["train_dataset_path"] == "/data/ds1/train.jsonl,/data/ds2/train.jsonl,/data/ds3/train.jsonl"
    assert config["data"]["train_dataset_prob"] == "0.5,0.3,0.2"

def test_build_multi_dataset_config_with_overrides():
    cb = ConfigBuilder()
    datasets = [("/data/ds1/train.jsonl", 0.5), ("/data/ds2/train.jsonl", 0.5)]
    config = cb.build_multi_dataset_config(
        model_path="/models/paddleocr-vl",
        datasets=datasets,
        overrides={"finetuning": {"learning_rate": 0.0001}},
    )
    assert config["finetuning"]["learning_rate"] == 0.0001
    assert config["data"]["train_dataset_prob"] == "0.5,0.5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_config_builder.py -v`
Expected: FAIL — `build_multi_dataset_config` doesn't exist

- [ ] **Step 3: Implement build_multi_dataset_config**

Add to `ConfigBuilder`:

```python
def build_multi_dataset_config(
    self,
    model_path: str,
    datasets: list[tuple[str, float]],  # [(path, probability), ...]
    eval_data_path: str = "",
    stage: str = "VL-SFT",
    template: str = "paddleocr_vl",
    output_dir: str = "",
    overrides: dict[str, Any] | None = None,
) -> dict:
    """Build a config for multi-dataset training with proportional sampling."""
    config = self.merge_configs(_PADDLEOCR_VL_DEFAULTS, {})
    config["model"]["model_name_or_path"] = model_path
    config["model"]["stage"] = stage
    config["data"]["dataset_type"] = "erniekit"
    config["data"]["template"] = template

    paths = [d[0] for d in datasets]
    probs = [d[1] for d in datasets]

    config["data"]["train_dataset_path"] = ",".join(paths)
    config["data"]["train_dataset_prob"] = ",".join(str(p) for p in probs)

    if eval_data_path:
        config["data"]["eval_dataset_path"] = eval_data_path
    if output_dir:
        config["finetuning"]["output_dir"] = output_dir
    if overrides:
        config = self.merge_configs(config, overrides)

    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_config_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autotrainer/pf_integration/config_builder.py tests/test_config_builder.py
git commit -m "feat(config_builder): add build_multi_dataset_config with proportional sampling"
```

---

### Task 6: Data Ratio Ablation Skill — Handler

**Files:**
- Create: `autotrainer/skills/data_ratio_ablation/__init__.py` (empty)
- Create: `autotrainer/skills/data_ratio_ablation/handler.py`
- Create: `autotrainer/skills/data_ratio_ablation/SKILL.md`
- Create: `tests/test_ratio_ablation.py`

- [ ] **Step 1: Write failing tests for ratio generation logic**

```python
# tests/test_ratio_ablation.py
from autotrainer.skills.data_ratio_ablation.handler import DataRatioAblationHandler

def test_generate_equal_ratios():
    handler = DataRatioAblationHandler()
    ratios = handler.generate_equal_ratios(3)
    assert len(ratios) == 3
    assert abs(sum(ratios) - 1.0) < 0.001

def test_generate_proportional_ratios():
    handler = DataRatioAblationHandler()
    ratios = handler.generate_proportional_ratios([10000, 5000, 1000])
    assert abs(sum(ratios) - 1.0) < 0.001
    assert ratios[0] > ratios[1] > ratios[2]

def test_generate_leave_one_out():
    handler = DataRatioAblationHandler()
    base_ratios = [0.5, 0.3, 0.2]
    names = ["ds1", "ds2", "ds3"]
    loo_configs = handler.generate_leave_one_out(base_ratios, names)
    assert len(loo_configs) == 3
    # Each config should exclude one dataset
    assert loo_configs[0]["excluded"] == "ds1"
    assert loo_configs[0]["ratios"] == [0.0, 0.6, 0.4]  # renormalized
    assert loo_configs[1]["excluded"] == "ds2"
    assert loo_configs[1]["ratios"] == [0.625, 0.0, 0.375]

def test_generate_ratio_sweep():
    handler = DataRatioAblationHandler()
    sweep = handler.generate_ratio_sweep(top_n=2)
    assert len(sweep) >= 5
    for combo in sweep:
        assert abs(sum(combo) - 1.0) < 0.001

def test_generate_ratio_sweep_three_datasets():
    handler = DataRatioAblationHandler()
    sweep = handler.generate_ratio_sweep(top_n=3)
    assert len(sweep) >= 5
    for combo in sweep:
        assert len(combo) == 3
        assert abs(sum(combo) - 1.0) < 0.001

def test_compute_score():
    handler = DataRatioAblationHandler()
    baseline = {"eval_loss": 1.0, "throughput": 1000}
    result = {"eval_loss": 0.5, "throughput": 1200, "has_nan": False}
    score = handler.compute_score(result, baseline)
    assert score > 1.0  # better than baseline

def test_compute_score_with_nan():
    handler = DataRatioAblationHandler()
    baseline = {"eval_loss": 1.0, "throughput": 1000}
    result = {"eval_loss": 0.5, "throughput": 1200, "has_nan": True}
    score_nan = handler.compute_score(result, baseline)
    result_no_nan = {"eval_loss": 0.5, "throughput": 1200, "has_nan": False}
    score_no_nan = handler.compute_score(result_no_nan, baseline)
    assert score_nan < score_no_nan  # NaN penalty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_ratio_ablation.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement handler.py**

```python
# autotrainer/skills/data_ratio_ablation/handler.py
"""Data Ratio Ablation skill handler.

Generates experiment specs for finding optimal multi-dataset mixing ratios
through leave-one-out ablation and ratio sweeps.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetInfo:
    """Info about a single dataset for ratio ablation."""

    name: str
    subset_path: str
    sample_count: int
    ds_type: str = "unknown"       # text_rich, chart, table, mixed


@dataclass
class RatioExperiment:
    """A single ratio experiment spec."""

    ratios: dict[str, float]       # {"ds1": 0.5, "ds2": 0.3, "ds3": 0.2}
    label: str                     # "baseline", "no-ds1", "sweep-50-30-20"
    description: str = ""


class DataRatioAblationHandler:
    """Handles data ratio ablation: leave-one-out + ratio sweep + scoring."""

    # Score weights (configurable)
    W_LOSS = 0.7
    W_STABILITY = 0.2
    W_SPEED = 0.1

    def generate_equal_ratios(self, n: int) -> list[float]:
        """Generate equal ratios that sum to 1.0."""
        base = 1.0 / n
        ratios = [base] * n
        # Fix floating point: last one gets the remainder
        ratios[-1] = 1.0 - sum(ratios[:-1])
        return ratios

    def generate_proportional_ratios(self, sample_counts: list[int]) -> list[float]:
        """Generate ratios proportional to sample counts."""
        total = sum(sample_counts)
        if total == 0:
            return self.generate_equal_ratios(len(sample_counts))
        ratios = [c / total for c in sample_counts]
        ratios[-1] = 1.0 - sum(ratios[:-1])
        return ratios

    def generate_leave_one_out(
        self,
        base_ratios: list[float],
        names: list[str],
    ) -> list[dict]:
        """Generate leave-one-out configs.

        For each dataset, remove it and renormalize the remaining ratios.
        Returns list of {"excluded": name, "ratios": [renormalized]}.
        """
        configs = []
        for i, name in enumerate(names):
            remaining = [r for j, r in enumerate(base_ratios) if j != i]
            total_remaining = sum(remaining)
            if total_remaining > 0:
                renormalized = [r / total_remaining for r in remaining]
                # Fix floating point
                renormalized[-1] = 1.0 - sum(renormalized[:-1])
            else:
                renormalized = [1.0 / len(remaining)] * len(remaining)

            # Insert 0.0 at position i
            full_ratios = list(renormalized)
            full_ratios.insert(i, 0.0)

            configs.append({
                "excluded": name,
                "ratios": full_ratios,
            })
        return configs

    def generate_ratio_sweep(self, top_n: int = 2) -> list[list[float]]:
        """Generate ratio combinations to sweep for top N datasets.

        Returns list of ratio lists, each summing to 1.0.
        """
        if top_n == 2:
            combos = [
                [0.5, 0.5],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.8, 0.2],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.9, 0.1],
            ]
        elif top_n == 3:
            combos = [
                [0.33, 0.33, 0.34],
                [0.5, 0.3, 0.2],
                [0.6, 0.2, 0.2],
                [0.4, 0.4, 0.2],
                [0.3, 0.5, 0.2],
                [0.7, 0.15, 0.15],
                [0.4, 0.3, 0.3],
            ]
        else:
            # For N > 3, use equal + a few skewed
            equal = self.generate_equal_ratios(top_n)
            combos = [equal]
            # Generate some skewed combos
            for i in range(top_n):
                combo = [0.1] * top_n
                combo[i] = 1.0 - 0.1 * (top_n - 1)
                if combo[i] > 0:
                    combos.append(combo)

        # Validate all sum to 1.0
        for combo in combos:
            combo[-1] = 1.0 - sum(combo[:-1])

        return combos

    def generate_saturation_configs(self) -> list[float]:
        """Downsampling ratios for saturation curve."""
        return [0.25, 0.50, 0.75, 1.00]

    def compute_score(
        self,
        result: dict,
        baseline: dict,
    ) -> float:
        """Compute weighted composite score. Higher = better."""
        # Loss score: baseline_loss / result_loss (higher is better)
        result_loss = result.get("eval_loss", float("inf"))
        baseline_loss = baseline.get("eval_loss", 1.0)
        if result_loss <= 0:
            result_loss = 1e-8
        loss_score = baseline_loss / result_loss

        # Speed score
        result_speed = result.get("throughput", 0)
        baseline_speed = baseline.get("throughput", 1)
        speed_score = result_speed / baseline_speed if baseline_speed > 0 else 1.0

        # Stability score
        stability_score = 1.0
        if result.get("has_nan", False):
            stability_score *= 0.5

        return (
            self.W_LOSS * loss_score
            + self.W_STABILITY * stability_score
            + self.W_SPEED * speed_score
        )

    def rank_datasets_by_importance(
        self,
        baseline_score: float,
        loo_scores: dict[str, float],
    ) -> list[tuple[str, float, str]]:
        """Rank datasets by importance based on leave-one-out results.

        Returns list of (name, delta, verdict) sorted by importance.
        delta = baseline_score - loo_score (positive = important)
        """
        rankings = []
        for name, loo_score in loo_scores.items():
            delta = baseline_score - loo_score
            if delta > 0.05:
                verdict = "important"
            elif delta > -0.02:
                verdict = "neutral"
            else:
                verdict = "noise"  # removing it actually helped
            rankings.append((name, delta, verdict))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def build_experiment_specs(
        self,
        datasets: list[DatasetInfo],
        base_config: dict,
        config_builder,
        subset_dir: str,
        ablation_max_steps: int = 1000,
    ) -> list[dict]:
        """Build the full list of experiment specs for ratio ablation.

        Returns list of ExperimentSpec dicts ready for scheduler.
        """
        from autotrainer.orchestrator.scheduler import ExperimentSpec

        specs = []
        names = [d.name for d in datasets]
        paths = [d.subset_path for d in datasets]

        # B1: Baseline (equal ratios)
        equal_ratios = self.generate_equal_ratios(len(datasets))
        baseline_config = config_builder.build_multi_dataset_config(
            model_path=base_config["model"]["model_name_or_path"],
            datasets=list(zip(paths, equal_ratios)),
            overrides={
                "finetuning": {
                    "learning_rate": base_config["finetuning"].get("learning_rate", 1e-4),
                    "per_device_train_batch_size": base_config["finetuning"].get("per_device_train_batch_size", 2),
                    "max_steps": ablation_max_steps,
                    "num_train_epochs": 1,
                    "do_eval": True,
                },
                "model": {
                    "use_lora": base_config["model"].get("use_lora", True),
                    "lora_rank": base_config["model"].get("lora_rank", 8),
                },
            },
        )
        specs.append(ExperimentSpec(
            id="ratio-b1-baseline",
            phase="ablation_ratio",
            config=baseline_config,
            config_diff={"ratios": dict(zip(names, equal_ratios))},
        ))

        # B2: Leave-one-out
        loo_configs = self.generate_leave_one_out(equal_ratios, names)
        for loo in loo_configs:
            excluded = loo["excluded"]
            remaining_names = [n for n in names if n != excluded]
            remaining_paths = [p for n, p in zip(names, paths) if n != excluded]
            remaining_ratios = [r for r in loo["ratios"] if r > 0]

            config = config_builder.build_multi_dataset_config(
                model_path=base_config["model"]["model_name_or_path"],
                datasets=list(zip(remaining_paths, remaining_ratios)),
                overrides={
                    "finetuning": {
                        "learning_rate": base_config["finetuning"].get("learning_rate", 1e-4),
                        "per_device_train_batch_size": base_config["finetuning"].get("per_device_train_batch_size", 2),
                        "max_steps": ablation_max_steps,
                        "num_train_epochs": 1,
                        "do_eval": True,
                    },
                    "model": {
                        "use_lora": base_config["model"].get("use_lora", True),
                        "lora_rank": base_config["model"].get("lora_rank", 8),
                    },
                },
            )
            ratio_dict = dict(zip(remaining_names, remaining_ratios))
            specs.append(ExperimentSpec(
                id=f"ratio-b2-no-{excluded}",
                phase="ablation_ratio",
                config=config,
                config_diff={"excluded": excluded, "ratios": ratio_dict},
            ))

        # B3: Ratio sweep (top 2 datasets — placeholder, actual top-N determined after B2)
        # Generate sweep for top-2 and top-3
        for top_n in [2, 3]:
            if top_n > len(datasets):
                continue
            sweep_combos = self.generate_ratio_sweep(top_n)
            for i, combo in enumerate(sweep_combos):
                top_names = names[:top_n]
                top_paths = paths[:top_n]
                config = config_builder.build_multi_dataset_config(
                    model_path=base_config["model"]["model_name_or_path"],
                    datasets=list(zip(top_paths, combo)),
                    overrides={
                        "finetuning": {
                            "learning_rate": base_config["finetuning"].get("learning_rate", 1e-4),
                            "per_device_train_batch_size": base_config["finetuning"].get("per_device_train_batch_size", 2),
                            "max_steps": ablation_max_steps,
                            "num_train_epochs": 1,
                            "do_eval": True,
                        },
                        "model": {
                            "use_lora": base_config["model"].get("use_lora", True),
                            "lora_rank": base_config["model"].get("lora_rank", 8),
                        },
                    },
                )
                ratio_str = "-".join(f"{int(r*100)}" for r in combo)
                specs.append(ExperimentSpec(
                    id=f"ratio-b3-top{top_n}-{ratio_str}",
                    phase="ablation_ratio",
                    config=config,
                    config_diff={"ratios": dict(zip(top_names, combo))},
                ))

        return specs
```

- [ ] **Step 4: Create SKILL.md**

```markdown
---
name: data-ratio-ablation
description: Find optimal multi-dataset mixing ratio through systematic ablation
tier: 2
---

# Data Ratio Ablation

You analyze multi-dataset training results to find the optimal mixing ratio.

## What You Receive

- Dataset list with sample counts and types
- Leave-one-out experiment results (which dataset matters most)
- Ratio sweep experiment results (which combination is best)

## Decision Rules

- If removing a dataset INCREASES score by >5% → that dataset is noise, remove it
- If removing a dataset DECREASES score by >5% → that dataset is important, keep it
- Pick the ratio combination with the highest composite score
- If scores are within 2% → prefer simpler ratio (fewer datasets)

## Output Format

Respond with valid JSON:

{
  "recommended_ratios": {"ds1": 0.8, "ds2": 0.2},
  "removed_datasets": ["ds3"],
  "analysis": "explanation",
  "confidence": 0.85
}
```

- [ ] **Step 5: Create __init__.py**

```python
# autotrainer/skills/data_ratio_ablation/__init__.py
```

(empty file)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer && python -m pytest tests/test_ratio_ablation.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add autotrainer/skills/data_ratio_ablation/ tests/test_ratio_ablation.py
git commit -m "feat: add data_ratio_ablation skill with leave-one-out, ratio sweep, scoring"
```

---

### Task 7: Pipeline — Wire Up Scheduler and Multi-Dataset Support

**Files:**
- Modify: `autotrainer/orchestrator/pipeline.py`

- [ ] **Step 1: Add scheduler import and initialization**

In `PipelineOrchestrator.__init__`, add after existing component init:

```python
from autotrainer.orchestrator.scheduler import ExperimentScheduler
from autotrainer.skills.data_ratio_ablation.handler import DataRatioAblationHandler

self.scheduler = ExperimentScheduler(work_dir=self.work_dir)
self.ratio_ablation_skill = DataRatioAblationHandler()
```

- [ ] **Step 2: Add multi-dataset detection in _run_phase_data_prepare**

In `_run_phase_data_prepare`, after the `index_path` check block (Path A), add multi-dataset subset creation:

```python
# After merge_from_index completes and sets self.state.data_path:
# Create per-dataset subsets for ratio ablation
if index_path and os.path.exists(index_path):
    # ... existing merge logic ...

    # NEW: Create per-dataset 5% subsets for ratio ablation
    import json
    with open(index_path, "r") as f:
        index_data = json.load(f)

    completed_datasets = [d for d in index_data.get("datasets", []) if d.get("status") == "completed"]
    if len(completed_datasets) > 1:
        ablation_data_dir = os.path.join(self.work_dir, "data")
        self.state.multi_dataset_info = []
        for ds in completed_datasets:
            ds_name = ds.get("dataset_name", "unknown")
            train_path = ds.get("split", {}).get("train", {}).get("path", "")
            if train_path and os.path.exists(train_path):
                subset_path = os.path.join(ablation_data_dir, f"subset_5pct_{ds_name}.jsonl")
                subset_info = self.data_mgr.create_subset(train_path, subset_path, ratio=0.05)
                self.state.multi_dataset_info.append({
                    "name": ds_name,
                    "subset_path": subset_path,
                    "sample_count": subset_info.get("subset", 0),
                    "total_count": subset_info.get("total", 0),
                })
        self._notify("DATA_PREPARE", f"Created per-dataset subsets for {len(self.state.multi_dataset_info)} datasets")
```

Add `multi_dataset_info` to `PipelineState`:

```python
@dataclass
class PipelineState:
    # ... existing fields ...
    multi_dataset_info: list[dict] = field(default_factory=list)
```

- [ ] **Step 3: Rewrite _run_phase_ablation to use scheduler**

Replace the existing `_run_phase_ablation` with:

```python
def _run_phase_ablation(self):
    """Phase 3: Ablation — hyperparams + data ratio (if multi-dataset)."""
    import copy

    if self.phase_mgr.is_completed(Phase.ABLATION):
        self._notify("ABLATION", "Skipped (already completed)")
        return

    state = self.phase_mgr.get_phase_state(Phase.ABLATION)
    if state.get("status") == "skipped":
        self._notify("ABLATION", "Skipped (skip-ablation)")
        return

    self.phase_mgr.transition_to(Phase.ABLATION)

    # Build base config
    model_path = self.config.detect_model_path("PaddlePaddle/PaddleOCR-VL")
    base_config = self.config_builder.build_paddleocr_vl_config(
        model_path=model_path,
        train_data=self.state.data_path,
        eval_data=self.state.eval_data_path,
        lora=True,
        lora_rank=8,
    )

    subset_path = self.state.ablation_config.get("subset_path", "")
    if not subset_path or not os.path.exists(subset_path):
        self._notify("ABLATION", "No ablation subset found, skipping ablation.")
        self.state.best_ablation_config = base_config
        return

    # ── Step A: Hyperparameter ablation ──
    from autotrainer.skills.plan_experiment.handler import AblationFactorConfig

    step_a_already_done = len(self.scheduler.get_experiments_by_phase("ablation_hyperparams")) > 0
    if not step_a_already_done:
        self._notify("ABLATION", "=== Step A: Hyperparameter Ablation ===")
        self._build_hyperparam_experiments(base_config, subset_path)

    # Run all pending Step A experiments
    self.scheduler.run_all(
        train_manager=self.train_mgr,
        config_builder=self.config_builder,
        gpu_ids=self.gpu_ids,
        on_progress=self._on_ablation_progress,
    )

    # Determine best hyperparams from Step A
    best_config = self._pick_best_hyperparams(base_config)
    self._notify("ABLATION", f"Step A complete. Best hyperparams applied.")

    # Cleanup Step A checkpoints
    self.scheduler.cleanup_phase_checkpoints("ablation_hyperparams")

    # ── Step B: Data ratio ablation (multi-dataset only) ──
    multi_ds = getattr(self.state, "multi_dataset_info", [])
    if len(multi_ds) > 1:
        self._notify("ABLATION", "=== Step B: Data Ratio Ablation ===")
        step_b_already_done = len(self.scheduler.get_experiments_by_phase("ablation_ratio")) > 0
        if not step_b_already_done:
            from autotrainer.skills.data_ratio_ablation.handler import DatasetInfo
            datasets = [
                DatasetInfo(
                    name=d["name"],
                    subset_path=d["subset_path"],
                    sample_count=d["sample_count"],
                )
                for d in multi_ds
            ]
            ratio_specs = self.ratio_ablation_skill.build_experiment_specs(
                datasets=datasets,
                base_config=best_config,
                config_builder=self.config_builder,
                subset_dir=os.path.join(self.work_dir, "data"),
            )
            self.scheduler.add_experiments(ratio_specs)

        # Run all pending Step B experiments
        self.scheduler.run_all(
            train_manager=self.train_mgr,
            config_builder=self.config_builder,
            gpu_ids=self.gpu_ids,
            on_progress=self._on_ablation_progress,
        )

        # Pick best ratio
        best_ratio_config = self._pick_best_ratio(best_config)
        self.state.best_ablation_config = best_ratio_config
        self._notify("ABLATION", "Step B complete. Best data ratio applied.")

        # Cleanup Step B checkpoints
        self.scheduler.cleanup_phase_checkpoints("ablation_ratio")
    else:
        self.state.best_ablation_config = best_config

    self._save_recovery_state()
```

- [ ] **Step 4: Add helper methods to pipeline**

```python
def _build_hyperparam_experiments(self, base_config: dict, subset_path: str):
    """Build hyperparameter ablation experiment specs and add to scheduler."""
    from autotrainer.skills.plan_experiment.handler import AblationFactorConfig

    factors = [
        AblationFactorConfig(
            dotted_key="finetuning.learning_rate",
            initial_values=[1e-5, 3e-5, 1e-4, 3e-4],
            scale="log", discrete=False, min_value=1e-6, max_value=1e-3,
        ),
        AblationFactorConfig(
            dotted_key="finetuning.per_device_train_batch_size",
            initial_values=[1, 2, 4],
            scale="linear", discrete=True, min_value=1, max_value=16,
        ),
        AblationFactorConfig(
            dotted_key="model.lora_rank",
            initial_values=[4, 8, 16],
            scale="linear", discrete=True, min_value=2, max_value=64,
        ),
    ]

    specs = []
    for factor in factors:
        for value in factor.initial_values:
            exp_id = f"abl-hyper-{factor.dotted_key.split('.')[-1]}={value}"
            exp_config = self.config_builder.build_ablation_config(
                base=base_config,
                factor_changes={factor.dotted_key: value},
                subset_path=subset_path,
                max_steps=1000,
            )
            from autotrainer.orchestrator.scheduler import ExperimentSpec
            specs.append(ExperimentSpec(
                id=exp_id,
                phase="ablation_hyperparams",
                config=exp_config,
                config_diff={factor.dotted_key: value},
            ))

    self.scheduler.add_experiments(specs)

def _pick_best_hyperparams(self, base_config: dict) -> dict:
    """Pick best hyperparams from completed Step A experiments."""
    import copy
    best_config = copy.deepcopy(base_config)
    experiments = self.scheduler.get_experiments_by_phase("ablation_hyperparams")
    completed = [e for e in experiments if e.status == "completed" and e.result]

    if not completed:
        return best_config

    # Group by factor, pick best value for each
    factor_best: dict[str, tuple[Any, float]] = {}
    for exp in completed:
        factor_key = list(exp.config_diff.keys())[0] if exp.config_diff else ""
        if not factor_key:
            continue
        loss = exp.result.get("eval_loss") or exp.result.get("final_loss") or float("inf")
        value = exp.config_diff[factor_key]
        if factor_key not in factor_best or loss < factor_best[factor_key][1]:
            factor_best[factor_key] = (value, loss)

    for factor_key, (value, _) in factor_best.items():
        parts = factor_key.split(".")
        target = best_config
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value

    return best_config

def _pick_best_ratio(self, hyperparams_config: dict) -> dict:
    """Pick best ratio from completed Step B experiments."""
    import copy
    best_config = copy.deepcopy(hyperparams_config)
    experiments = self.scheduler.get_experiments_by_phase("ablation_ratio")
    completed = [e for e in experiments if e.status == "completed" and e.result]

    if not completed:
        return best_config

    best_exp = min(
        completed,
        key=lambda e: e.result.get("eval_loss") or e.result.get("final_loss") or float("inf"),
    )

    # Apply best ratio's paths and probs to config
    best_config["data"]["train_dataset_path"] = best_exp.config.get("data", {}).get("train_dataset_path", "")
    best_config["data"]["train_dataset_prob"] = best_exp.config.get("data", {}).get("train_dataset_prob", "1.0")

    self._notify("ABLATION", f"Best ratio: {best_exp.config_diff.get('ratios', {})}")
    return best_config

def _on_ablation_progress(self, status, result):
    """Callback for ablation progress — forwards to TUI."""
    self._notify("ABLATION",
        f"[{status.completed}/{status.total}] {result.experiment_id}: "
        f"loss={result.final_loss or '?'} status={result.status}"
    )
```

- [ ] **Step 5: Add cleanup hook in _run_phase_full_training**

At the end of `_run_phase_full_training`, add:

```python
# Cleanup ablation checkpoints now that we're moving to full training
self.scheduler.cleanup_phase_checkpoints("ablation_hyperparams")
self.scheduler.cleanup_phase_checkpoints("ablation_ratio")
```

- [ ] **Step 6: Commit**

```bash
git add autotrainer/orchestrator/pipeline.py
git commit -m "feat(pipeline): wire up scheduler, multi-dataset ratio ablation, cleanup hooks"
```

---

### Task 8: TUI — Experiment Progress Display

**Files:**
- Modify: `autotrainer/tui/widgets/status_bar.py`
- Modify: `autotrainer/tui/app.py`

- [ ] **Step 1: Update StatusBar to show experiment progress**

In `status_bar.py`, add method:

```python
def update_experiment_progress(
    self,
    phase: str,
    exp_current: int,
    exp_total: int,
    exp_id: str = "",
    loss: float | None = None,
    gpu_util: float = 0,
):
    """Update with experiment-level progress."""
    parts = [f"[b]Phase:[/b] {phase}"]
    parts.append(f"[b]Exp:[/b] {exp_current}/{exp_total}")
    if exp_id:
        parts.append(f"[b]ID:[/b] {exp_id}")
    if loss is not None:
        parts.append(f"[b]Loss:[/b] {loss:.4f}")
    if gpu_util > 0:
        parts.append(f"[b]GPU:[/b] {gpu_util:.0f}%")
    self.update("  |  ".join(parts))
```

- [ ] **Step 2: Update AutoTrainerApp to display experiment queue status**

In `app.py`, add method:

```python
def update_experiment_status(self, scheduler_status):
    """Update main panel with experiment queue status."""
    s = scheduler_status
    lines = [
        f"\n  [bold]Experiment Queue[/bold]  ({s.completed}/{s.total} completed, {s.failed} failed)",
        "",
    ]
    for phase_name, breakdown in s.phase_breakdown.items():
        status_str = f"{breakdown['done']}/{breakdown['total']}"
        if breakdown.get("failed", 0) > 0:
            status_str += f" ({breakdown['failed']} failed)"
        lines.append(f"  [bold]{phase_name}:[/bold] {status_str}")

    if s.current:
        lines.extend([
            "",
            f"  [yellow]Running:[/yellow] {s.current.id}",
        ])

    if self.main_panel:
        self.main_panel.update("\n".join(lines))
```

- [ ] **Step 3: Commit**

```bash
git add autotrainer/tui/widgets/status_bar.py autotrainer/tui/app.py
git commit -m "feat(tui): add experiment progress display to status bar and main panel"
```

---

## Spec Coverage Check

| Spec Section | Task |
|---|---|
| 1. ExperimentScheduler | Task 2 (core), Task 3 (execution + GC) |
| 2. Multi-Dataset Ratio Ablation | Task 5 (config), Task 6 (handler), Task 7 (pipeline wiring) |
| 3. Step-Level Resume | Task 4 (train_manager) |
| 4. Checkpoint GC | Task 3 (scheduler GC methods), Task 7 (cleanup hooks) |
| 5. Metric Extraction | Task 1 (log_parser) |
| 6. TUI Progress | Task 8 (status_bar + app) |

All spec requirements covered. No gaps.
