"""Tests for TaskRegistry — plugin discovery and registration."""

import tempfile
import os
import yaml

from autotrainer.core.registry import TaskRegistry
from autotrainer.core.interfaces import TaskSpec


class TestTaskRegistry:
    """Tests for the task registry system."""

    def test_register_and_get(self):
        reg = TaskRegistry()
        spec = TaskSpec(name="test-task", description="A test task")
        reg.register(spec)

        assert reg.get("test-task") is not None
        assert reg.get("test-task").description == "A test task"
        assert reg.get("nonexistent") is None

    def test_list_tasks(self):
        reg = TaskRegistry()
        reg.register(TaskSpec(name="task-a"))
        reg.register(TaskSpec(name="task-b"))

        names = reg.task_names()
        assert "task-a" in names
        assert "task-b" in names

    def test_duplicate_registration_warns(self):
        reg = TaskRegistry()
        reg.register(TaskSpec(name="dup"))
        reg.register(TaskSpec(name="dup", description="updated"))
        # Should overwrite without error
        assert reg.get("dup").description == "updated"

    def test_builtin_discovery(self):
        """Registry should discover paddleocr-vl from manifest.yaml."""
        reg = TaskRegistry()
        task = reg.get("paddleocr-vl")
        assert task is not None
        assert task.description != ""
        assert "finetuning.learning_rate" in task.hyperparam_space
        assert "erniekit-jsonl" in task.supported_data_formats

    def test_list_tasks_all_fields(self):
        reg = TaskRegistry()
        tasks = reg.list_tasks()
        for t in tasks:
            assert t.name
            assert isinstance(t.hyperparam_space, dict)
            assert isinstance(t.supported_data_formats, list)


class TestTaskSpec:
    """Tests for TaskSpec dataclass."""

    def test_defaults(self):
        spec = TaskSpec(name="test")
        assert spec.description == ""
        assert spec.hyperparam_space == {}
        assert spec.supported_data_formats == []

    def test_full_spec(self):
        spec = TaskSpec(
            name="my-task",
            description="My custom task",
            hyperparam_space={"lr": {"type": "log", "min": 1e-5, "max": 1e-2, "initial": [1e-4]}},
            supported_data_formats=["jsonl", "parquet"],
            launch_command="train.py",
            launch_env={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        assert spec.name == "my-task"
        assert spec.hyperparam_space["lr"]["type"] == "log"
        assert len(spec.supported_data_formats) == 2
