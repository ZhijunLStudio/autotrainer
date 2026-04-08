# Multi-Dataset Ratio Optimization, Experiment Scheduler, and Infrastructure Improvements

**Date**: 2026-04-05
**Status**: Draft

## Overview

Add multi-dataset training ratio optimization, an experiment scheduler with step-level resume, checkpoint garbage collection, expanded metric extraction, and progress preview to the AutoTrainer system.

## Problem

The current AutoTrainer pipeline has several gaps when handling multiple datasets:

1. **No data ratio optimization**: `merge_from_index()` concatenates all datasets without weighting — the largest dataset dominates training
2. **Single-metric evaluation**: Only `eval_loss` is used for ablation ranking, ignoring other useful signals
3. **No experiment-level resume**: If ablation is killed at experiment 5/7, the entire phase restarts
4. **No checkpoint management**: All checkpoints accumulate until disk fills up
5. **No ablation progress preview**: The TUI shows phase-level info but not experiment-level progress

## Design

### 1. ExperimentScheduler — Experiment Queue Manager

New component: `autotrainer/orchestrator/scheduler.py`

Manages a persistent experiment queue with resume and cleanup support.

```python
class ExperimentScheduler:
    """Manages a queue of experiments with resume and cleanup support."""

    def __init__(self, work_dir, train_manager, config_builder, gpu_ids):
        # Persists queue to work_dir/experiment_queue.json

    # Queue management
    def add_experiments(self, experiments: list[ExperimentSpec])
    def get_status(self) -> SchedulerStatus
    def get_next_pending(self) -> ExperimentSpec | None

    # Execution
    def run_next(self) -> TrainingResult
    def run_all(self, on_progress=None) -> list[TrainingResult]

    # Cleanup
    def cleanup_phase_checkpoints(self, phase: str)
    def cleanup_full_training(self, keep_best=1, keep_last=1)
```

Data structures:

```python
@dataclass
class ExperimentSpec:
    id: str                    # "abl-001-lr=0.0001"
    phase: str                 # "ablation_hyperparams" | "ablation_ratio" | "full_training"
    config: dict               # Full PaddleFormers config
    config_diff: dict          # Changes relative to base
    status: str = "pending"    # pending | running | completed | failed
    result: dict | None = None # TrainingResult.to_dict()
    checkpoint_path: str = ""
    created_at: str = ""
    completed_at: str = ""

@dataclass
class SchedulerStatus:
    total: int
    completed: int
    failed: int
    pending: int
    current: ExperimentSpec | None
    phase_breakdown: dict      # {"ablation_hyperparams": {"total":5, "done":3}, ...}
```

Replaces `recovery.py` experiment index — scheduler owns all experiment state.

### 2. Multi-Dataset Ratio Ablation

New skill: `autotrainer/skills/data_ratio_ablation/`

#### 2a. Pipeline Phase 1 Changes

When multiple datasets are detected:

- Scan each subdirectory, count samples, classify type (text_rich/chart/table)
- Each dataset gets its own train/val split
- Each dataset gets its own 5% subset
- Generate baseline ratio: equal weight, proportional by sample count, or paper default

#### 2b. Ablation Flow — Two Steps

**Step A: Hyperparameter ablation** (existing, unchanged)

Input: baseline ratio mixed subset. Tunes: learning_rate → batch_size → lora_rank.

**Step B: Data ratio ablation** (new)

Input: best hyperparams from Step A + individual dataset subsets.

Sub-steps:

| Sub-step | Description | Runs |
|---|---|---|
| B1 Baseline | Train with default ratio (equal/proportional) | 1 run |
| B2 Leave-one-out | Remove each dataset, renormalize, train | N runs |
| B3 Ratio sweep | Test 5-7 ratio combos on top 2-3 datasets | 5-7 runs |
| B4 Saturation (optional) | Downsample each dataset 25/50/75/100% | 4 runs |

All runs use 5% subset + 1 epoch. Total: ~8-12 runs.

#### 2c. Config: Multi-Dataset Native Support

PaddleFormers natively supports multiple datasets via comma-separated paths:

```yaml
train_dataset_path: "ds1/train.jsonl,ds2/train.jsonl,ds3/train.jsonl"
train_dataset_prob: "0.5,0.3,0.2"
mix_strategy: "concat"  # or use dynamic proportional sampling
```

`ConfigBuilder` changes:
- New method `build_multi_dataset_config()` accepting list of (path, prob) pairs
- Rewrite `train_dataset_prob` as comma-separated string

#### 2d. Scoring Algorithm

Weighted composite score (higher = better):

```python
def compute_score(result, baseline):
    loss_score = baseline.eval_loss / max(result.eval_loss, 1e-8)
    speed_score = result.throughput / baseline.throughput if baseline.throughput else 1.0
    stability_score = 1.0
    if result.has_nan: stability_score *= 0.5
    if result.loss_std > threshold: stability_score *= 0.8
    return 0.7 * loss_score + 0.2 * stability_score + 0.1 * speed_score
```

Weights configurable, defaults above.

### 3. Step-Level Resume (Three-Layer)

| Layer | Existing | New |
|---|---|---|
| Phase level | PhaseManager.is_completed() | Keep as-is |
| Experiment level | None | scheduler reads experiment_queue.json, skips completed |
| Step level | None | PaddleFormers --resume_from_checkpoint |

In `scheduler.run_next()`:

1. Check experiment status in queue → skip if completed
2. Call `train_manager.run_single_ablation(resume_from_checkpoint=...)`
3. `train_manager` detects existing `checkpoint-N/` in output_dir, passes `--resume_from_checkpoint` to PaddleFormers launcher
4. After completion or failure, immediately persist to `experiment_queue.json`

PaddleFormers natively supports `--resume_from_checkpoint <path>`, so this is a config passthrough.

### 4. Checkpoint Garbage Collection

New component: `autotrainer/orchestrator/scheduler.py` — `CheckpointGC` inner class.

**Cleanup triggers**:

| When | What |
|---|---|
| Step A (hyperparams ablation) complete | Delete all Step A experiment checkpoints, keep result.json |
| Step B (ratio ablation) complete | Delete all Step B experiment checkpoints, keep result.json |
| Full training running | After each save, check if total checkpoints > keep_best + keep_last, delete oldest |
| Full training complete | Keep only best (by eval_loss) + last checkpoint |

**What to keep per experiment**:
- `result.json` (always — small, has all metrics)
- `config.yaml` (always — small, needed for reproducibility)
- `checkpoint-N/` directories (deleted on cleanup)

**Disk budget display**: TUI shows current disk usage vs budget.

### 5. Metric Extraction — LogParser Expansion

Changes to `autotrainer/pf_integration/log_parser.py`:

```python
@dataclass
class LogMetrics:
    # Existing fields...
    eval_ppl: float | None = None          # new
    extra_metrics: dict = field(default_factory=dict)  # new: eval_accuracy, eval_bleu, etc.
    has_nan: bool = False                  # new
    loss_std: float = 0.0                  # new: rolling loss standard deviation
```

New regex pattern for generic eval metrics:
```python
EVAL_KV_PATTERN = re.compile(r"eval_(\w+)[:\s]+([\d.eE+-]+)")
```

PaddleFormers stdout format: `eval_loss: 1.2, eval_ppl: 3.32, eval_accuracy: 0.8`

The parser dynamically captures any `eval_<key>: <value>` pair, not just loss.

### 6. TUI Progress Preview

New widget or expanded `StatusBar`:

```
[Step A] Hyperparameter Ablation — COMPLETED
  lr=0.0001 → loss=0.52 (BEST)
  batch_size=4 → loss=0.54
  lora_rank=16 → loss=0.51 (BEST)

[Step B] Data Ratio Ablation — IN PROGRESS (3/7)
  baseline (33/33/34) → score=0.85  ✓
  no-docvqa (50/50)   → score=0.62  ✓
  no-chartqa (98/2)   → score=0.83  ✓
  no-table (100/0)    → running...
    GPU: 85% | Step: 342/1000 | Loss: 0.68 | ETA: 2min
  (50/30/20)          pending
  (60/20/20)          pending
  (70/15/15)          pending

Checkpoint: 2.3GB / 50GB budget
```

Implementation: `PipelineOrchestrator` passes `SchedulerStatus` callbacks to TUI via existing `OnPhaseChange`/`OnMetrics` mechanism, extended with `OnExperimentProgress`.

## File Changes Summary

| File | Action | Description |
|---|---|---|
| `autotrainer/orchestrator/scheduler.py` | **NEW** | ExperimentScheduler + CheckpointGC |
| `autotrainer/skills/data_ratio_ablation/SKILL.md` | **NEW** | Skill prompt for ratio ablation |
| `autotrainer/skills/data_ratio_ablation/handler.py` | **NEW** | Ratio ablation logic |
| `autotrainer/skills/data_ratio_ablation/__init__.py` | **NEW** | Package init |
| `autotrainer/orchestrator/pipeline.py` | MODIFY | Phase 1: multi-dataset scan. Phase 3: use scheduler. Cleanup hooks |
| `autotrainer/pf_integration/config_builder.py` | MODIFY | `build_multi_dataset_config()`, multi-path support |
| `autotrainer/pf_integration/log_parser.py` | MODIFY | Generic eval metric extraction, eval_ppl, has_nan, loss_std |
| `autotrainer/managers/train_manager.py` | MODIFY | `resume_from_checkpoint` param in `run_single_ablation()` |
| `autotrainer/managers/data_pipeline.py` | MODIFY | Multi-dataset subset creation |
| `autotrainer/tui/app.py` | MODIFY | Experiment progress display |
| `autotrainer/tui/widgets/status_bar.py` | MODIFY | Experiment-level status |

## Dependencies

- PaddleFormers `train_dataset_prob` comma-separated string support (already exists in config defaults)
- PaddleFormers `--resume_from_checkpoint` flag (native support)
- No new Python package dependencies

## Open Questions

- PaddleFormers dynamic proportional sampling vs simple concat: need to verify `mix_strategy` behavior with multiple `train_dataset_prob`
- Exact weight values for composite score (0.7/0.2/0.1) — may need tuning based on real experiments
- Saturation curve (B4) — should it be opt-in or always run? Defaulting to opt-in.
