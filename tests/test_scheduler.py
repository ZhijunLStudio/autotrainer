"""Tests for ExperimentScheduler."""

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
