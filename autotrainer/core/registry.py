"""Task registry — plugin discovery and registration for trainable tasks.

Tasks are discovered via:
1. Built-in manifest files (tasks/<name>/manifest.yaml)
2. Python entry_points group "autotrainer.tasks"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from autotrainer.core.interfaces import TaskSpec

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Global registry of trainable tasks.

    Tasks are registered with a TaskSpec that declares:
    - name, description
    - hyperparameter space for ablation
    - supported data formats
    - launcher command template
    """

    def __init__(self):
        self._tasks: dict[str, TaskSpec] = {}
        self._discover_builtin()

    def register(self, spec: TaskSpec) -> None:
        if spec.name in self._tasks:
            logger.warning("Task '%s' is already registered, overwriting.", spec.name)
        self._tasks[spec.name] = spec
        logger.info("Registered task: %s", spec.name)

    def get(self, name: str) -> TaskSpec | None:
        return self._tasks.get(name)

    def list_tasks(self) -> list[TaskSpec]:
        return list(self._tasks.values())

    def task_names(self) -> list[str]:
        return list(self._tasks.keys())

    def _discover_builtin(self) -> None:
        """Discover tasks from built-in tasks/ directory."""
        tasks_dir = Path(__file__).resolve().parent.parent / "tasks"
        if not tasks_dir.is_dir():
            return
        for entry in sorted(tasks_dir.iterdir()):
            manifest = entry / "manifest.yaml"
            if manifest.is_file():
                try:
                    spec = self._load_manifest(manifest)
                    self.register(spec)
                except Exception:
                    logger.warning("Failed to load task manifest: %s", manifest, exc_info=True)

    def _load_manifest(self, path: Path) -> TaskSpec:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return TaskSpec(
            name=data.get("name", path.parent.name),
            description=data.get("description", ""),
            hyperparam_space=data.get("hyperparam_space", {}),
            supported_data_formats=data.get("supported_data_formats", []),
            launch_command=data.get("launch_command", ""),
            launch_env=data.get("launch_env", {}),
        )

    @classmethod
    def discover_from_entry_points(cls) -> "TaskRegistry":
        """Discover tasks from installed Python packages via entry_points."""
        registry = cls()
        try:
            from importlib.metadata import entry_points

            for ep in entry_points(group="autotrainer.tasks"):
                try:
                    spec = ep.load()()
                    if isinstance(spec, TaskSpec):
                        registry.register(spec)
                except Exception:
                    logger.warning("Failed to load task plugin: %s", ep.name, exc_info=True)
        except ImportError:
            pass
        return registry
