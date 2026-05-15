"""Unified SQLite store — single source of truth for all pipeline state.

Replaces 4 scattered JSON files:
  - pipeline_state.json (PhaseManager)
  - recovery_state.json (RecoveryManager)
  - pipeline_state_full.json (PipelineOrchestrator)
  - experiment_index.json (TrainManager + RecoveryManager)

Schema versioning is built in for future migrations.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from autotrainer.core.interfaces import (
    CheckpointRecord,
    ExperimentRecord,
    Phase,
    PhaseStatus,
)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id TEXT PRIMARY KEY,                -- e.g. "paddleocr-vl-20260514-120000"
    task_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed
    current_phase INTEGER NOT NULL DEFAULT 0,
    gpu_ids TEXT NOT NULL DEFAULT '[]',
    data_path TEXT DEFAULT '',
    eval_data_path TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS phase_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    phase INTEGER NOT NULL,
    status TEXT NOT NULL,               -- in_progress, completed, failed, skipped
    message TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}',
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,                -- ablation_hyperparams, ablation_ratio, full_training
    status TEXT NOT NULL DEFAULT 'pending',
    config_diff_json TEXT DEFAULT '{}',
    config_path TEXT DEFAULT '',
    result_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    completed_at TEXT DEFAULT '',
    PRIMARY KEY (id, run_id),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    path TEXT NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    loss REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id, run_id) REFERENCES experiments(id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_run ON experiments(run_id);
CREATE INDEX IF NOT EXISTS idx_experiments_phase ON experiments(run_id, phase, status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_exp ON checkpoints(experiment_id, run_id);
"""


class PipelineStore:
    """Unified SQLite-backed store for all pipeline and experiment state."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ── Internal ─────────────────────────────────────────────────────────

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
            # Check/set schema version
            cur = conn.execute("SELECT MAX(version) FROM schema_version")
            row = cur.fetchone()
            current_version = row[0] if row and row[0] is not None else 0
            if current_version < SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
                conn.commit()

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection within the lock."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
            finally:
                conn.close()

    def _now(self) -> str:
        return datetime.now().isoformat()

    # ── Pipeline Run ─────────────────────────────────────────────────────

    def create_run(
        self,
        run_id: str,
        task_name: str,
        gpu_ids: list[int] | None = None,
        data_path: str = "",
        eval_data_path: str = "",
    ):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO pipeline_runs (id, task_name, status, current_phase, gpu_ids,
                   data_path, eval_data_path, created_at, updated_at)
                   VALUES (?, ?, 'running', 0, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    task_name,
                    json.dumps(gpu_ids or []),
                    data_path,
                    eval_data_path,
                    self._now(),
                    self._now(),
                ),
            )
            conn.commit()

    def update_run_phase(self, run_id: str, phase: Phase, status: PhaseStatus, message: str = ""):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE pipeline_runs SET current_phase = ?, updated_at = ? WHERE id = ?",
                (phase.value, self._now(), run_id),
            )
            conn.execute(
                """INSERT INTO phase_events (run_id, phase, status, message, metadata_json, timestamp)
                   VALUES (?, ?, ?, ?, '{}', ?)""",
                (run_id, phase.value, status.value, message, self._now()),
            )
            conn.commit()

    def complete_run(self, run_id: str):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE pipeline_runs SET status = 'completed', updated_at = ? WHERE id = ?",
                (self._now(), run_id),
            )
            conn.commit()

    def fail_run(self, run_id: str, error: str = ""):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE pipeline_runs SET status = 'failed', updated_at = ? WHERE id = ?",
                (self._now(), run_id),
            )
            conn.execute(
                """INSERT INTO phase_events (run_id, phase, status, message, timestamp)
                   VALUES (?, ?, 'failed', ?, ?)""",
                (run_id, -1, error, self._now()),
            )
            conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["gpu_ids"] = json.loads(d.get("gpu_ids", "[]"))
            return d

    def get_run_phase_events(self, run_id: str) -> list[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM phase_events WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def find_latest_run(self) -> dict | None:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM pipeline_runs ORDER BY updated_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["gpu_ids"] = json.loads(d.get("gpu_ids", "[]"))
            return d

    def get_phase_state(self, run_id: str, phase: Phase) -> dict:
        """Get the latest event for a phase in a run."""
        with self._get_conn() as conn:
            cur = conn.execute(
                """SELECT status, message, metadata_json, timestamp
                   FROM phase_events
                   WHERE run_id = ? AND phase = ?
                   ORDER BY id DESC LIMIT 1""",
                (run_id, phase.value),
            )
            row = cur.fetchone()
            if not row:
                return {"status": "pending"}
            return {
                "status": row["status"],
                "message": row["message"],
                "metadata": json.loads(row["metadata_json"]),
                "timestamp": row["timestamp"],
            }

    def is_phase_completed(self, run_id: str, phase: Phase) -> bool:
        state = self.get_phase_state(run_id, phase)
        return state.get("status") == "completed"

    def get_progress(self, run_id: str) -> dict:
        """Get overall pipeline progress for a run."""
        run = self.get_run(run_id)
        if not run:
            return {
                "current_phase": "unknown",
                "current_phase_index": -1,
                "completed_phases": 0,
                "total_phases": 7,
                "progress_pct": 0.0,
            }
        with self._get_conn() as conn:
            cur = conn.execute(
                """SELECT phase FROM phase_events
                   WHERE run_id = ? AND status = 'completed'
                   GROUP BY phase""",
                (run_id,),
            )
            completed = len(cur.fetchall())
        return {
            "current_phase": Phase(run["current_phase"]).name,
            "current_phase_index": run["current_phase"],
            "completed_phases": completed,
            "total_phases": 7,
            "progress_pct": round(completed / 7 * 100, 1),
        }

    # ── Experiments ──────────────────────────────────────────────────────

    def add_experiment(self, run_id: str, record: ExperimentRecord):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (id, run_id, phase, status, config_diff_json, config_path, result_json, created_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    run_id,
                    record.phase,
                    record.status,
                    json.dumps(record.config_diff),
                    record.config_path,
                    json.dumps(record.result),
                    record.created_at or self._now(),
                    record.completed_at,
                ),
            )
            conn.commit()

    def update_experiment(
        self, run_id: str, exp_id: str, status: str, result: dict | None = None
    ):
        with self._get_conn() as conn:
            updates = ["status = ?"]
            params: list = [status]
            if result is not None:
                updates.append("result_json = ?")
                params.append(json.dumps(result))
            if status in ("completed", "failed"):
                updates.append("completed_at = ?")
                params.append(self._now())
            params.extend([exp_id, run_id])
            conn.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE id = ? AND run_id = ?",
                params,
            )
            conn.commit()

    def get_experiments_by_phase(self, run_id: str, phase: str) -> list[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE run_id = ? AND phase = ? ORDER BY id ASC",
                (run_id, phase),
            )
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["config_diff"] = json.loads(d.get("config_diff_json", "{}"))
                d["result"] = json.loads(d.get("result_json", "{}"))
                results.append(d)
            return results

    def get_experiment(self, run_id: str, exp_id: str) -> dict | None:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE id = ? AND run_id = ?",
                (exp_id, run_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["config_diff"] = json.loads(d.get("config_diff_json", "{}"))
            d["result"] = json.loads(d.get("result_json", "{}"))
            return d

    def list_experiments(self, run_id: str) -> list[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM experiments WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            )
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["config_diff"] = json.loads(d.get("config_diff_json", "{}"))
                d["result"] = json.loads(d.get("result_json", "{}"))
                results.append(d)
            return results

    # ── Checkpoints ─────────────────────────────────────────────────────

    def add_checkpoint(
        self,
        run_id: str,
        experiment_id: str,
        path: str,
        step: int = 0,
        loss: float | None = None,
    ):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO checkpoints (experiment_id, run_id, path, step, loss, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (experiment_id, run_id, path, step, loss, self._now()),
            )
            conn.commit()

    def get_checkpoints(self, run_id: str, experiment_id: str) -> list[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? AND experiment_id = ? ORDER BY step DESC",
                (run_id, experiment_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_latest_checkpoint(self, run_id: str, experiment_id: str) -> dict | None:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? AND experiment_id = ? ORDER BY step DESC LIMIT 1",
                (run_id, experiment_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ── Convenience: snapshot for reporting ──────────────────────────────

    def get_full_snapshot(self, run_id: str) -> dict:
        """Return a complete snapshot of a run for reporting."""
        run = self.get_run(run_id)
        return {
            "run": run,
            "phase_events": self.get_run_phase_events(run_id),
            "experiments": self.list_experiments(run_id),
            "progress": self.get_progress(run_id),
        }
