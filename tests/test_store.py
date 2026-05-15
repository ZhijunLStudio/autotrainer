"""Tests for PipelineStore — unified SQLite storage."""

import os
import tempfile

from autotrainer.core.store import PipelineStore
from autotrainer.core.interfaces import Phase, PhaseStatus, ExperimentRecord


class TestPipelineStore:
    """Tests for the unified SQLite store."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.store = PipelineStore(self.db_path)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_and_get_run(self):
        self.store.create_run("run-1", "paddleocr-vl", gpu_ids=[0, 1])
        run = self.store.get_run("run-1")
        assert run is not None
        assert run["task_name"] == "paddleocr-vl"
        assert run["status"] == "running"
        assert run["gpu_ids"] == [0, 1]

    def test_update_run_phase(self):
        self.store.create_run("run-1", "test")
        self.store.update_run_phase("run-1", Phase.DATA_PREPARE, PhaseStatus.IN_PROGRESS, "starting")
        self.store.update_run_phase("run-1", Phase.DATA_PREPARE, PhaseStatus.COMPLETED, "done")

        assert self.store.is_phase_completed("run-1", Phase.DATA_PREPARE)
        assert not self.store.is_phase_completed("run-1", Phase.ENV_CHECK)

    def test_phase_events_recorded(self):
        self.store.create_run("run-1", "test")
        self.store.update_run_phase("run-1", Phase.DATA_PREPARE, PhaseStatus.COMPLETED, "done")
        self.store.update_run_phase("run-1", Phase.ENV_CHECK, PhaseStatus.FAILED, "GPU not found")

        events = self.store.get_run_phase_events("run-1")
        assert len(events) == 2
        assert events[0]["status"] == "completed"
        assert events[1]["status"] == "failed"

    def test_complete_and_fail_run(self):
        self.store.create_run("run-1", "test")
        self.store.complete_run("run-1")
        assert self.store.get_run("run-1")["status"] == "completed"

        self.store.create_run("run-2", "test")
        self.store.fail_run("run-2", "something broke")
        assert self.store.get_run("run-2")["status"] == "failed"

    def test_experiment_crud(self):
        self.store.create_run("run-1", "test")
        exp = ExperimentRecord(id="exp-1", phase="ablation_hyperparams", status="pending")
        self.store.add_experiment("run-1", exp)

        # Read back
        exps = self.store.get_experiments_by_phase("run-1", "ablation_hyperparams")
        assert len(exps) == 1
        assert exps[0]["id"] == "exp-1"
        assert exps[0]["status"] == "pending"

        # Update
        self.store.update_experiment("run-1", "exp-1", "completed", {"eval_loss": 0.5})
        exps = self.store.get_experiments_by_phase("run-1", "ablation_hyperparams")
        assert exps[0]["status"] == "completed"
        assert exps[0]["result"]["eval_loss"] == 0.5

    def test_experiments_by_phase_isolation(self):
        self.store.create_run("run-1", "test")
        self.store.add_experiment("run-1", ExperimentRecord(id="e1", phase="ablation_hyperparams", status="pending"))
        self.store.add_experiment("run-1", ExperimentRecord(id="e2", phase="ablation_ratio", status="pending"))
        self.store.add_experiment("run-1", ExperimentRecord(id="e3", phase="ablation_ratio", status="pending"))

        assert len(self.store.get_experiments_by_phase("run-1", "ablation_hyperparams")) == 1
        assert len(self.store.get_experiments_by_phase("run-1", "ablation_ratio")) == 2
        assert len(self.store.list_experiments("run-1")) == 3

    def test_checkpoint_tracking(self):
        self.store.create_run("run-1", "test")
        self.store.add_experiment("run-1", ExperimentRecord(id="exp-1", phase="full_training", status="pending"))

        self.store.add_checkpoint("run-1", "exp-1", "/tmp/ckpt/checkpoint-100", step=100, loss=0.5)
        self.store.add_checkpoint("run-1", "exp-1", "/tmp/ckpt/checkpoint-200", step=200, loss=0.3)

        ckpts = self.store.get_checkpoints("run-1", "exp-1")
        assert len(ckpts) == 2
        # Should be ordered by step DESC
        assert ckpts[0]["step"] == 200

        latest = self.store.get_latest_checkpoint("run-1", "exp-1")
        assert latest["step"] == 200
        assert latest["loss"] == 0.3

    def test_get_full_snapshot(self):
        self.store.create_run("run-1", "test")
        self.store.update_run_phase("run-1", Phase.TASK_CONFIRM, PhaseStatus.COMPLETED)
        self.store.add_experiment("run-1", ExperimentRecord(id="e1", phase="ablation", status="pending"))

        snapshot = self.store.get_full_snapshot("run-1")
        assert "run" in snapshot
        assert "phase_events" in snapshot
        assert "experiments" in snapshot
        assert "progress" in snapshot

    def test_find_latest_run(self):
        self.store.create_run("run-a", "test")
        self.store.create_run("run-b", "test")
        latest = self.store.find_latest_run()
        assert latest["id"] == "run-b"

    def test_progress(self):
        self.store.create_run("run-1", "test")
        self.store.update_run_phase("run-1", Phase.TASK_CONFIRM, PhaseStatus.COMPLETED)
        self.store.update_run_phase("run-1", Phase.DATA_PREPARE, PhaseStatus.COMPLETED)

        progress = self.store.get_progress("run-1")
        assert progress["completed_phases"] == 2
        assert progress["progress_pct"] == pytest.approx(28.6, 0.5)

    def test_store_persistence(self):
        """Store survives closing and reopening."""
        self.store.create_run("run-1", "test")
        self.store.update_run_phase("run-1", Phase.DATA_PREPARE, PhaseStatus.COMPLETED)
        self.store.add_experiment("run-1", ExperimentRecord(id="e1", phase="ablation", status="pending"))

        # Reopen
        store2 = PipelineStore(self.db_path)
        run = store2.get_run("run-1")
        assert run is not None
        assert store2.is_phase_completed("run-1", Phase.DATA_PREPARE)
        assert len(store2.list_experiments("run-1")) == 1


import pytest
