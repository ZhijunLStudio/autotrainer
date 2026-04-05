"""Tests for ExperimentScheduler."""

import os
import json
import tempfile
from unittest.mock import MagicMock
from autotrainer.orchestrator.scheduler import ExperimentSpec, SchedulerStatus, ExperimentScheduler
from autotrainer.managers.train_manager import TrainingResult


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


def test_run_next_calls_train_manager():
    """run_next should call train_manager and update state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sched = ExperimentScheduler(work_dir=tmpdir)
        mock_train_mgr = MagicMock()
        mock_result = TrainingResult(
            experiment_id="e1", final_loss=0.5, total_steps=100,
            status="completed", checkpoint_path=os.path.join(tmpdir, "checkpoints", "e1"),
        )
        mock_train_mgr.run_single_ablation.return_value = mock_result

        sched.add_experiments([
            ExperimentSpec(
                id="e1", phase="ablation",
                config={"data": {"train_dataset_path": "/tmp/subset.jsonl"}, "finetuning": {"max_steps": 1000, "output_dir": os.path.join(tmpdir, "checkpoints", "e1")}},
                config_diff={"lr": 0.001},
            ),
        ])

        result = sched.run_next(mock_train_mgr, gpu_ids=[0])
        assert result is not None
        assert result.experiment_id == "e1"
        status = sched.get_status()
        assert status.completed == 1


def test_run_all():
    """run_all should run all pending experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sched = ExperimentScheduler(work_dir=tmpdir)
        mock_train_mgr = MagicMock()

        def make_result(experiment_id, **kwargs):
            return TrainingResult(
                experiment_id=experiment_id, final_loss=0.5, total_steps=100,
                status="completed", checkpoint_path="",
            )
        mock_train_mgr.run_single_ablation.side_effect = make_result

        sched.add_experiments([
            ExperimentSpec(id="e1", phase="ablation", config={"data": {}, "finetuning": {"max_steps": 100, "output_dir": ""}}, config_diff={}),
            ExperimentSpec(id="e2", phase="ablation", config={"data": {}, "finetuning": {"max_steps": 100, "output_dir": ""}}, config_diff={}),
        ])

        results = sched.run_all(mock_train_mgr, gpu_ids=[0])
        assert len(results) == 2
        status = sched.get_status()
        assert status.completed == 2


def test_checkpoint_gc_ablation():
    """cleanup_phase_checkpoints should remove checkpoint dirs but keep result.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
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


def test_cleanup_skips_non_completed():
    """cleanup should not touch non-completed experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = os.path.join(tmpdir, "checkpoints", "abl-001")
        os.makedirs(os.path.join(exp_dir, "checkpoint-500"))

        sched = ExperimentScheduler(work_dir=tmpdir)
        sched.add_experiments([
            ExperimentSpec(id="abl-001", phase="ablation_hyper", config={}, config_diff={}, status="failed", checkpoint_path=exp_dir),
        ])
        sched.cleanup_phase_checkpoints("ablation_hyper")

        assert os.path.exists(os.path.join(exp_dir, "checkpoint-500"))
