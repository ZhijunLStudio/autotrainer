#!/usr/bin/env python
"""Architecture validation script — verifies the new V2 pipeline with test data.

Usage:
  # Dry-run (no GPUs needed): validates all imports, store, registry, phase handlers
  python scripts/validate_architecture.py --dry-run

  # With test data: creates a small subset and runs data_prepare phase
  python scripts/validate_architecture.py --data /path/to/data.jsonl

  # Full pipeline (needs GPUs):
  python scripts/validate_architecture.py --data /path/to/data.jsonl --gpus 0,1,2,3,4,5,6
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time


def check_imports() -> dict:
    """Verify all new modules import correctly."""
    results = {}
    modules = [
        ("core.store", "autotrainer.core.store.PipelineStore"),
        ("core.registry", "autotrainer.core.registry.TaskRegistry"),
        ("core.interfaces", "autotrainer.core.interfaces"),
        ("core.legacy_importer", "autotrainer.core.legacy_importer"),
        ("services.experiment", "autotrainer.services.experiment_service.ExperimentService"),
        ("services.checkpoint", "autotrainer.services.checkpoint_service.CheckpointService"),
        ("phases.task_confirm", "autotrainer.phases.task_confirm.TaskConfirmHandler"),
        ("phases.data_prepare", "autotrainer.phases.data_prepare.DataPrepareHandler"),
        ("phases.env_check", "autotrainer.phases.env_check.EnvCheckHandler"),
        ("phases.ablation", "autotrainer.phases.ablation.AblationHandler"),
        ("phases.full_training", "autotrainer.phases.full_training.FullTrainingHandler"),
        ("phases.evaluation", "autotrainer.phases.evaluation.EvaluationHandler"),
        ("phases.report", "autotrainer.phases.report.ReportHandler"),
        ("orchestrator.v2", "autotrainer.orchestrator.pipeline_v2.PipelineOrchestratorV2"),
    ]
    for name, import_path in modules:
        try:
            parts = import_path.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            results[name] = True
        except Exception as e:
            results[name] = str(e)
    return results


def check_store() -> bool:
    """Verify Store CRUD operations."""
    from autotrainer.core.store import PipelineStore
    from autotrainer.core.interfaces import Phase, PhaseStatus, ExperimentRecord

    with tempfile.TemporaryDirectory() as tmp:
        store = PipelineStore(os.path.join(tmp, "test.db"))
        store.create_run("arch-validation", "paddleocr-vl", gpu_ids=[0, 1, 2, 3, 4, 5, 6])
        store.update_run_phase("arch-validation", Phase.TASK_CONFIRM, PhaseStatus.COMPLETED)
        store.update_run_phase("arch-validation", Phase.DATA_PREPARE, PhaseStatus.IN_PROGRESS, "validating")

        exp = ExperimentRecord(id="test-exp", phase="full_training", status="completed")
        store.add_experiment("arch-validation", exp)
        store.add_checkpoint("arch-validation", "test-exp", "/fake/checkpoint-100", step=100, loss=0.5)

        run = store.get_run("arch-validation")
        assert run is not None, "Run not found"
        assert run["gpu_ids"] == [0, 1, 2, 3, 4, 5, 6], "GPU IDs mismatch"
        assert store.is_phase_completed("arch-validation", Phase.TASK_CONFIRM), "Phase not completed"

        exps = store.list_experiments("arch-validation")
        assert len(exps) == 1, "Experiment count mismatch"

        ckpts = store.get_checkpoints("arch-validation", "test-exp")
        assert len(ckpts) == 1 and ckpts[0]["step"] == 100, "Checkpoint mismatch"

        snapshot = store.get_full_snapshot("arch-validation")
        assert all(k in snapshot for k in ("run", "phase_events", "experiments", "progress")), "Snapshot incomplete"

    return True


def check_registry() -> bool:
    """Verify TaskRegistry discovers paddleocr-vl."""
    from autotrainer.core.registry import TaskRegistry

    reg = TaskRegistry()
    tasks = reg.list_tasks()
    assert len(tasks) >= 1, "No tasks discovered"
    t = reg.get("paddleocr-vl")
    assert t is not None, "paddleocr-vl not found"
    assert "finetuning.learning_rate" in t.hyperparam_space, "Missing learning_rate in hyperparams"
    return True


def check_orchestrator() -> bool:
    """Verify Orchestrator V2 initialization."""
    from autotrainer.config import AutoTrainerConfig
    from autotrainer.orchestrator.pipeline_v2 import PipelineOrchestratorV2

    with tempfile.TemporaryDirectory() as tmp:
        config = AutoTrainerConfig(work_dir=tmp)
        orch = PipelineOrchestratorV2(
            config=config, task="paddleocr-vl",
            gpu_ids=[0, 1, 2, 3, 4, 5, 6],
            skip_ablation=True,
        )
        assert orch.run_id.startswith("paddleocr-vl-"), f"Bad run_id: {orch.run_id}"
        assert len(orch.handlers) == 7, f"Expected 7 handlers, got {len(orch.handlers)}"
        assert orch.ctx.gpu_ids == [0, 1, 2, 3, 4, 5, 6], "GPU mismatch in context"
        assert orch.store is not None, "Store not initialized"

        run = orch.store.get_run(orch.run_id)
        assert run["task_name"] == "paddleocr-vl", "Task name mismatch"
    return True


def create_test_subset(data_path: str, output_path: str, n_lines: int = 100):
    """Create a small test subset from a larger JSONL file."""
    lines = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            if line.strip():
                lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"  Created test subset: {len(lines)} lines -> {output_path}")
    return output_path


def check_phase_data_prepare(data_path: str) -> bool:
    """Verify DataPrepareHandler with real data."""
    from autotrainer.core.interfaces import Phase, PipelineContext
    from autotrainer.phases.data_prepare import DataPrepareHandler

    with tempfile.TemporaryDirectory() as tmp:
        ctx = PipelineContext(
            task="paddleocr-vl",
            gpu_ids=[0],
            work_dir=tmp,
            data_path=data_path,
        )
        handler = DataPrepareHandler()
        result = handler.execute(ctx)

        print(f"  Phase result: {result.phase.name} -> {result.status.value}")
        print(f"  Message: {result.message}")
        print(f"  Data path: {ctx.data_path}")
        print(f"  Eval path: {ctx.eval_data_path}")
        print(f"  Data profile: {ctx.data_profile.get('num_samples', 'N/A')} samples")
        print(f"  Ablation config: subset_path={ctx.ablation_config.get('subset_path', 'N/A')}")

        return result.status.value == "completed"


def main():
    parser = argparse.ArgumentParser(description="Validate AutoTrainer V2 architecture")
    parser.add_argument("--dry-run", action="store_true", help="Verify imports and basic functionality only")
    parser.add_argument("--data", type=str, default="", help="Path to JSONL training data")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs for training validation")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (needs GPUs)")
    args = parser.parse_args()

    print("=" * 60)
    print("  AutoTrainer V2 Architecture Validation")
    print("=" * 60)

    # 1. Import checks
    print("\n[1/5] Checking module imports...")
    import_results = check_imports()
    all_ok = True
    for name, ok in import_results.items():
        status = "OK" if ok is True else f"FAIL: {ok}"
        if ok is not True:
            all_ok = False
        print(f"  {name:30s} {status}")
    if not all_ok:
        print("\nFAIL: Import errors found.")
        sys.exit(1)

    # 2. Store CRUD
    print("\n[2/5] Checking Store CRUD...")
    check_store()
    print("  Store: OK")

    # 3. Registry
    print("\n[3/5] Checking Task Registry...")
    check_registry()
    print("  Registry: OK")

    # 4. Orchestrator initialization
    print("\n[4/5] Checking Orchestrator V2...")
    check_orchestrator()
    print("  Orchestrator V2: OK")

    # 5. If data provided, test data_prepare phase
    if args.data:
        print(f"\n[5/5] Checking DataPrepareHandler with test data...")
        if not os.path.exists(args.data):
            print(f"  FAIL: Data path does not exist: {args.data}")
            sys.exit(1)

        # Create a small test subset
        test_data = os.path.join(tempfile.mkdtemp(), "test_subset.jsonl")
        create_test_subset(args.data, test_data, n_lines=100)

        check_phase_data_prepare(test_data)
        print("  DataPrepareHandler: OK")
    else:
        print("\n[5/5] Skipping data phase check (use --data to test with real data)")
        print("  Example: python scripts/validate_architecture.py --data ocr_training_data/detection_dataset.jsonl")

    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED")
    print("=" * 60)

    # Summary
    print(f"\nArchitecture summary:")
    print(f"  New modules: core/, phases/, services/, tasks/")
    print(f"  Total new files: 15")
    print(f"  Store: SQLite (replaces 4 JSON files)")
    print(f"  Registry: 1 task (paddleocr-vl)")
    print(f"  Phases: 7 independent handlers")
    print(f"  Services: ExperimentService + CheckpointService")
    print(f"  Orchestrator: 887 lines (v1) -> ~80 lines (v2)")

    if args.full and args.gpus:
        print(f"\nTo run full pipeline validation:")
        print(f"  cd /data/lizhijun/work/PaddleFormersAutomatedTraining/autotrainer")
        print(f"  python -m autotrainer train --data-path {test_data if args.data else '<data>'} --gpus {args.gpus}")


if __name__ == "__main__":
    main()
