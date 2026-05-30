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
        # New model config fields
        assert task.model_family == "VL"
        assert task.stage == "VL-SFT"
        assert task.template == "paddleocr_vl"
        assert task.model_name_or_path == "PaddlePaddle/PaddleOCR-VL"
        assert "freeze_vision" in task.freeze_config
        assert isinstance(task.model_defaults, dict)
        assert isinstance(task.finetuning_defaults, dict)

    def test_builtin_discovers_all_eight_tasks(self):
        """Registry should discover all 8 built-in task manifests."""
        reg = TaskRegistry()
        names = reg.task_names()
        assert len(names) == 8
        expected = {"paddleocr-vl", "qwen2_5-vl", "qwen3-vl", "ernie4_5-vl",
                    "qwen3", "deepseek-v3", "ernie4_5", "llama"}
        assert set(names) == expected
        # Verify VL vs LLM classification
        vl_tasks = {"paddleocr-vl", "qwen2_5-vl", "qwen3-vl", "ernie4_5-vl"}
        for name in names:
            spec = reg.get(name)
            if name in vl_tasks:
                assert spec.model_family == "VL", f"{name} should be VL"
                assert "VL" in spec.stage, f"{name} should have VL stage"
                assert spec.freeze_config, f"{name} should have freeze_config"
            else:
                assert spec.model_family == "LLM", f"{name} should be LLM"
                assert spec.stage == "SFT", f"{name} should be SFT"

    def test_list_tasks_all_fields(self):
        reg = TaskRegistry()
        tasks = reg.list_tasks()
        assert len(tasks) >= 8
        for t in tasks:
            assert t.name
            assert isinstance(t.hyperparam_space, dict)
            assert isinstance(t.supported_data_formats, list)
            assert isinstance(t.model_defaults, dict)
            assert isinstance(t.finetuning_defaults, dict)
            assert t.stage in ("VL-SFT", "SFT", "DPO", "PT")
            assert t.model_family in ("VL", "LLM")


class TestTaskSpec:
    """Tests for TaskSpec dataclass."""

    def test_defaults(self):
        spec = TaskSpec(name="test")
        assert spec.description == ""
        assert spec.hyperparam_space == {}
        assert spec.supported_data_formats == []
        # New fields should have sensible defaults
        assert spec.model_name_or_path == ""
        assert spec.model_family == ""
        assert spec.stage == "SFT"
        assert spec.template == ""
        assert spec.max_seq_len == 8192
        assert spec.model_defaults == {}
        assert spec.finetuning_defaults == {}

    def test_full_spec(self):
        spec = TaskSpec(
            name="my-task",
            description="My custom task",
            model_name_or_path="Org/Model",
            model_family="VL",
            stage="VL-SFT",
            template="my_template",
            freeze_config="freeze_vision",
            max_seq_len=16384,
            hyperparam_space={"lr": {"type": "log", "min": 1e-5, "max": 1e-2, "initial": [1e-4]}},
            supported_data_formats=["jsonl", "parquet"],
            launch_command="train.py",
            launch_env={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        assert spec.name == "my-task"
        assert spec.model_family == "VL"
        assert spec.stage == "VL-SFT"
        assert spec.template == "my_template"
        assert spec.freeze_config == "freeze_vision"
        assert spec.max_seq_len == 16384
        assert spec.hyperparam_space["lr"]["type"] == "log"
        assert len(spec.supported_data_formats) == 2
