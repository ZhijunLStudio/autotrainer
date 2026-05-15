"""Tests for Services layer — ExperimentService and CheckpointService."""

import os
import tempfile

from autotrainer.core.store import PipelineStore
from autotrainer.core.interfaces import ExperimentRecord
from autotrainer.services.experiment_service import ExperimentService
from autotrainer.services.checkpoint_service import CheckpointService


class TestExperimentService:
    """Tests for unified experiment CRUD service."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.store = PipelineStore(self.db_path)
        self.store.create_run("run-1", "test")
        self.svc = ExperimentService(self.store, "run-1")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_and_retrieve(self):
        self.svc.create("exp-1", "ablation_hyperparams", config_diff={"lr": 1e-4})
        exp = self.svc.get("exp-1")
        assert exp is not None
        assert exp["status"] == "pending"
        assert exp["phase"] == "ablation_hyperparams"

    def test_mark_running_completed(self):
        self.svc.create("exp-1", "ablation_hyperparams")
        self.svc.mark_running("exp-1")
        assert self.svc.get("exp-1")["status"] == "running"

        self.svc.mark_completed("exp-1", {"eval_loss": 0.5})
        assert self.svc.get("exp-1")["status"] == "completed"
        assert self.svc.get("exp-1")["result"]["eval_loss"] == 0.5

    def test_mark_failed(self):
        self.svc.create("exp-1", "ablation")
        self.svc.mark_failed("exp-1", {"error": "OOM"})
        assert self.svc.get("exp-1")["status"] == "failed"

    def test_list_by_phase(self):
        self.svc.create("e1", "ablation_hyperparams")
        self.svc.create("e2", "ablation_hyperparams")
        self.svc.create("e3", "ablation_ratio")

        assert len(self.svc.list_by_phase("ablation_hyperparams")) == 2
        assert len(self.svc.list_by_phase("ablation_ratio")) == 1

    def test_get_pending(self):
        self.svc.create("e1", "ablation")
        self.svc.create("e2", "ablation")
        self.svc.mark_completed("e1", {})

        pending = self.svc.get_pending("ablation")
        assert len(pending) == 1
        assert pending[0]["id"] == "e2"

    def test_add_batch(self):
        records = [
            ExperimentRecord(id=f"batch-{i}", phase="ablation", status="pending")
            for i in range(5)
        ]
        self.svc.add_batch(records)
        assert len(self.svc.list_by_phase("ablation")) == 5


class TestCheckpointService:
    """Tests for checkpoint management."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.store = PipelineStore(self.db_path)
        self.store.create_run("run-1", "test")
        self.store.add_experiment("run-1", ExperimentRecord(id="exp-1", phase="full_training", status="pending"))
        self.svc = CheckpointService(self.store, "run-1", self.tmpdir)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_track_and_get_checkpoint(self):
        self.svc.track_checkpoint("exp-1", "/tmp/ckpt/checkpoint-100", step=100, loss=0.3)
        ckpts = self.svc.get_checkpoints("exp-1")
        assert len(ckpts) == 1
        assert ckpts[0]["step"] == 100

    def test_get_latest_checkpoint(self):
        self.svc.track_checkpoint("exp-1", "/tmp/ckpt/checkpoint-100", step=100, loss=0.5)
        self.svc.track_checkpoint("exp-1", "/tmp/ckpt/checkpoint-200", step=200, loss=0.2)

        latest = self.svc.get_latest_checkpoint("exp-1")
        assert latest["step"] == 200

    def test_find_latest_from_disk(self):
        ckpt_dir = os.path.join(self.tmpdir, "checkpoints", "exp-1")
        os.makedirs(os.path.join(ckpt_dir, "checkpoint-50"), exist_ok=True)
        os.makedirs(os.path.join(ckpt_dir, "checkpoint-150"), exist_ok=True)

        path = self.svc.find_latest(ckpt_dir)
        assert path is not None
        assert "checkpoint-150" in path

    def test_find_latest_empty_dir(self):
        empty_dir = os.path.join(self.tmpdir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        assert self.svc.find_latest(empty_dir) is None

    def test_cleanup_phase_checkpoints(self):
        exp = ExperimentRecord(id="exp-1", phase="ablation_hyperparams", status="completed")
        self.store.add_experiment("run-1", exp)
        self.store.update_experiment("run-1", "exp-1", "completed",
                                     {"checkpoint_path": os.path.join(self.tmpdir, "ckpt")})

        ckpt_dir = os.path.join(self.tmpdir, "ckpt")
        os.makedirs(os.path.join(ckpt_dir, "checkpoint-1"), exist_ok=True)
        os.makedirs(os.path.join(ckpt_dir, "checkpoint-2"), exist_ok=True)

        self.svc.cleanup_phase_checkpoints("ablation_hyperparams")
        assert not os.path.exists(os.path.join(ckpt_dir, "checkpoint-1"))
        assert not os.path.exists(os.path.join(ckpt_dir, "checkpoint-2"))
