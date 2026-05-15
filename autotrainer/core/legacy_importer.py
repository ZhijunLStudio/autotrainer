"""Migrate legacy JSON state files into the SQLite Store.

Called once on first startup when the Store finds no pipeline_run
but detects existing JSON files on disk.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime

from autotrainer.core.interfaces import ExperimentRecord, Phase, PhaseStatus
from autotrainer.core.store import PipelineStore

logger = logging.getLogger(__name__)

LEGACY_FILES = [
    "pipeline_state.json",
    "pipeline_state_full.json",
    "recovery_state.json",
    "experiment_index.json",
]


def find_legacy_state(work_dir: str) -> dict | None:
    """Find and merge all legacy JSON state files into a single dict. Returns None if nothing found."""
    result: dict = {"found": False, "phase_states": {}, "experiments": [], "data_path": "", "gpu_ids": []}

    # 1. pipeline_state.json (PhaseManager)
    ps_path = os.path.join(work_dir, "pipeline_state.json")
    if os.path.exists(ps_path):
        try:
            with open(ps_path, "r") as f:
                data = json.load(f)
            result["task_name"] = data.get("task_name", "unknown")
            result["current_phase"] = data.get("current_phase", 0)
            result["phase_states"] = data.get("phase_states", {})
            result["found"] = True
        except Exception:
            pass

    # 2. pipeline_state_full.json (PipelineOrchestrator)
    psf_path = os.path.join(work_dir, "pipeline_state_full.json")
    if os.path.exists(psf_path):
        try:
            with open(psf_path, "r") as f:
                data = json.load(f)
            result["task_name"] = result.get("task_name") or data.get("task", "unknown")
            result["gpu_ids"] = data.get("gpu_ids", [])
            result["data_path"] = data.get("data_path", "")
            result["ablation_results"] = data.get("ablation_results", [])
            result["found"] = True
        except Exception:
            pass

    # 3. recovery_state.json
    rc_path = os.path.join(work_dir, "recovery_state.json")
    if os.path.exists(rc_path):
        try:
            with open(rc_path, "r") as f:
                data = json.load(f)
            result["gpu_ids"] = result.get("gpu_ids") or data.get("gpu_ids", [])
            result["data_path"] = result.get("data_path") or data.get("data_path", "")
            result["found"] = True
        except Exception:
            pass

    # 4. experiment_index.json
    ei_path = os.path.join(work_dir, "experiment_index.json")
    if os.path.exists(ei_path):
        try:
            with open(ei_path, "r") as f:
                data = json.load(f)
            result["experiments"] = data.get("experiments", [])
            result["found"] = True
        except Exception:
            pass

    return result if result["found"] else None


def migrate_to_store(store: PipelineStore, work_dir: str) -> str | None:
    """Migrate legacy JSON state to the Store.

    Returns the run_id if migration succeeded, None otherwise.
    """
    legacy = find_legacy_state(work_dir)
    if not legacy:
        return None

    task_name = legacy.get("task_name", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{task_name}-legacy-{timestamp}"

    logger.info("Migrating legacy state to Store: run_id=%s", run_id)

    # Create run
    store.create_run(
        run_id=run_id,
        task_name=task_name,
        gpu_ids=legacy.get("gpu_ids", []),
        data_path=legacy.get("data_path", ""),
        eval_data_path=legacy.get("eval_data_path", ""),
    )

    # Import phase events
    phase_states = legacy.get("phase_states", {})
    for phase_str, state in phase_states.items():
        try:
            phase_num = int(phase_str)
            phase = Phase(phase_num)
            status_str = state.get("status", "pending")
            try:
                status = PhaseStatus(status_str)
            except ValueError:
                status = PhaseStatus.PENDING
            store.update_run_phase(run_id, phase, status, state.get("error", ""))
        except (ValueError, KeyError):
            continue

    # Import experiments
    for exp in legacy.get("experiments", []):
        record = ExperimentRecord(
            id=exp.get("id", "unknown"),
            phase=exp.get("phase", ""),
            status=exp.get("status", "pending"),
            config_diff=exp.get("config_diff", {}),
            config_path=exp.get("config_path", ""),
            result=exp.get("result", {}),
            created_at=exp.get("created_at", ""),
            completed_at=exp.get("completed_at", ""),
        )
        store.add_experiment(run_id, record)

    logger.info("Migration complete: %d experiments imported", len(legacy.get("experiments", [])))
    return run_id
