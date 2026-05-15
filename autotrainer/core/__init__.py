"""Core infrastructure layer — no business dependencies."""

from autotrainer.core.interfaces import (
    Phase,
    PhaseStatus,
    PipelineContext,
    PhaseResult,
    PhaseHandler,
    TaskSpec,
    ExperimentRecord,
    CheckpointRecord,
)
from autotrainer.core.store import PipelineStore
from autotrainer.core.registry import TaskRegistry

__all__ = [
    "Phase",
    "PhaseStatus",
    "PipelineContext",
    "PhaseResult",
    "PhaseHandler",
    "TaskSpec",
    "ExperimentRecord",
    "CheckpointRecord",
    "PipelineStore",
    "TaskRegistry",
]
