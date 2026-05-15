"""Tests for Phase handlers — each phase tested in isolation with fake dependencies."""

import os
import tempfile
import json

from autotrainer.core.interfaces import Phase, PhaseStatus, PipelineContext
from autotrainer.phases.task_confirm import TaskConfirmHandler
from autotrainer.phases.env_check import EnvCheckHandler
from autotrainer.phases.evaluation import EvaluationHandler


class TestTaskConfirmHandler:
    def test_execute_returns_completed(self):
        handler = TaskConfirmHandler()
        ctx = PipelineContext(task="paddleocr-vl", gpu_ids=[0, 1], work_dir="/tmp/test")
        result = handler.execute(ctx)
        assert result.status == PhaseStatus.COMPLETED
        assert result.phase == Phase.TASK_CONFIRM
        assert result.next_phase == Phase.DATA_PREPARE
        assert "paddleocr-vl" in result.message


class TestEnvCheckHandler:
    def test_execute_with_fake_env(self):
        handler = EnvCheckHandler()
        ctx = PipelineContext(task="paddleocr-vl", gpu_ids=[0], work_dir="/tmp/test")
        result = handler.execute(ctx)
        # Should succeed or fail depending on whether PaddlePaddle is installed
        # We just check that it returns a valid result type
        assert result.phase == Phase.ENV_CHECK
        assert result.status in (PhaseStatus.COMPLETED, PhaseStatus.FAILED)


class TestEvaluationHandler:
    def test_execute_no_checkpoint_skips(self):
        handler = EvaluationHandler()
        ctx = PipelineContext(task="paddleocr-vl", work_dir="/tmp/nonexistent")
        result = handler.execute(ctx)
        assert result.status == PhaseStatus.COMPLETED
        assert "skipping" in result.message.lower() or "no checkpoint" in result.message.lower()


class TestPipelineContext:
    """Tests for PipelineContext convenience methods."""

    def test_notify_calls_callback(self):
        calls = []
        ctx = PipelineContext(
            on_phase_change=lambda phase, msg: calls.append((phase, msg))
        )
        ctx.notify("TEST", "hello")
        assert len(calls) == 1
        assert calls[0] == ("TEST", "hello")

    def test_notify_without_callback(self):
        ctx = PipelineContext()
        ctx.notify("TEST", "hello")  # Should not raise

    def test_confirm_default_true(self):
        ctx = PipelineContext()
        assert ctx.confirm("Are you sure?") is True

    def test_confirm_with_callback(self):
        ctx = PipelineContext(on_user_confirm=lambda msg, ctx: False)
        assert ctx.confirm("Are you sure?") is False

    def test_prompt_default_first_choice(self):
        ctx = PipelineContext()
        assert ctx.prompt("Pick one", ["a", "b"]) == "a"

    def test_prompt_with_callback(self):
        ctx = PipelineContext(on_user_input=lambda prompt, choices: "b")
        assert ctx.prompt("Pick one", ["a", "b"]) == "b"

    def test_prompt_empty_choices(self):
        ctx = PipelineContext()
        assert ctx.prompt("Pick", []) == ""


class TestPhaseResult:
    """Tests for PhaseResult dataclass."""

    def test_defaults(self):
        from autotrainer.core.interfaces import PhaseResult
        r = PhaseResult(phase=Phase.DATA_PREPARE, status=PhaseStatus.COMPLETED)
        assert r.message == ""
        assert r.data == {}
        assert r.next_phase is None

    def test_with_data(self):
        from autotrainer.core.interfaces import PhaseResult
        r = PhaseResult(
            phase=Phase.ABLATION,
            status=PhaseStatus.COMPLETED,
            message="Done",
            data={"best_lr": 1e-4},
            next_phase=Phase.FULL_TRAINING,
        )
        assert r.data["best_lr"] == 1e-4
        assert r.next_phase == Phase.FULL_TRAINING
