"""End-to-end pipeline integration tests using PipelineOrchestratorV2."""

import os
import tempfile

from autotrainer.config import AutoTrainerConfig
from autotrainer.core.store import PipelineStore
from autotrainer.core.interfaces import Phase, PhaseStatus, PipelineContext
from autotrainer.orchestrator.pipeline_v2 import PipelineOrchestratorV2


class TestPipelineOrchestratorV2:
    """Integration tests for the V2 orchestrator."""

    def test_orchestrator_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoTrainerConfig(work_dir=tmp)
            orch = PipelineOrchestratorV2(config=config, task="paddleocr-vl", gpu_ids=[0],
                                          skip_ablation=True)
            assert orch.run_id.startswith("paddleocr-vl-")
            assert orch.work_dir.startswith(tmp)
            assert len(orch.handlers) == 7
            assert orch.store is not None
            assert orch.ctx.task == "paddleocr-vl"

    def test_run_creates_store_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoTrainerConfig(work_dir=tmp)
            orch = PipelineOrchestratorV2(config=config, task="paddleocr-vl", gpu_ids=[0],
                                          skip_ablation=True)
            run = orch.store.get_run(orch.run_id)
            assert run is not None
            assert run["task_name"] == "paddleocr-vl"
            assert run["status"] == "running"

    def test_orchestrator_context_is_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoTrainerConfig(work_dir=tmp)
            orch = PipelineOrchestratorV2(
                config=config, task="paddleocr-vl", gpu_ids=[0, 1, 2, 3, 4, 5, 6],
                skip_ablation=True, data_dir="/tmp/data",
            )
            assert orch.ctx.gpu_ids == [0, 1, 2, 3, 4, 5, 6]
            assert orch.ctx.data_dir == "/tmp/data"

    def test_handlers_have_correct_types(self):
        from autotrainer.phases import (
            TaskConfirmHandler, DataPrepareHandler, EnvCheckHandler,
            AblationHandler, FullTrainingHandler, EvaluationHandler, ReportHandler,
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoTrainerConfig(work_dir=tmp)
            orch = PipelineOrchestratorV2(config=config, task="paddleocr-vl", gpu_ids=[0],
                                          skip_ablation=True)
            assert isinstance(orch.handlers[Phase.TASK_CONFIRM], TaskConfirmHandler)
            assert isinstance(orch.handlers[Phase.DATA_PREPARE], DataPrepareHandler)
            assert isinstance(orch.handlers[Phase.ENV_CHECK], EnvCheckHandler)
            assert isinstance(orch.handlers[Phase.ABLATION], AblationHandler)
            assert isinstance(orch.handlers[Phase.FULL_TRAINING], FullTrainingHandler)
            assert isinstance(orch.handlers[Phase.EVALUATION], EvaluationHandler)
            assert isinstance(orch.handlers[Phase.REPORT], ReportHandler)


class TestStoreIntegration:
    """Integration tests between Store and Orchestrator."""

    def test_store_persistence_across_orchestrators(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoTrainerConfig(work_dir=tmp)

            # Create orch 1 - simulates first run
            orch1 = PipelineOrchestratorV2(config=config, task="paddleocr-vl", gpu_ids=[0],
                                           skip_ablation=True)
            run_id = orch1.run_id

            # Re-open the store
            db_path = os.path.join(orch1.work_dir, "pipeline.db")
            store2 = PipelineStore(db_path)
            run = store2.get_run(run_id)
            assert run is not None
            assert run["task_name"] == "paddleocr-vl"
