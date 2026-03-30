"""Custom exception hierarchy for the autotrainer system."""

from __future__ import annotations


class AutoTrainerError(Exception):
    """Base exception for all autotrainer errors."""


# --- Environment errors ---


class EnvironmentError(AutoTrainerError):
    """Environment setup or verification failure."""


class PackageInstallError(EnvironmentError):
    """Failed to install a required package."""


class GPUError(EnvironmentError):
    """GPU detection or allocation failure."""


# --- Data errors ---


class DataError(AutoTrainerError):
    """Base for data-related errors."""


class DatasetNotFoundError(DataError):
    """Requested dataset cannot be found."""


class DatasetFormatError(DataError):
    """Dataset has wrong format or is corrupted."""


class DatasetDownloadError(DataError):
    """Failed to download a dataset."""


class DataConversionError(DataError):
    """Failed to convert data format."""


class AuthenticationRequiredError(DataError):
    """Dataset requires user authentication (login, API key, terms acceptance)."""


# --- Training errors ---


class TrainingError(AutoTrainerError):
    """Base for training-related errors."""


class TrainingLaunchError(TrainingError):
    """Failed to launch training process."""


class TrainingCrashError(TrainingError):
    """Training process crashed unexpectedly."""


class TrainingHangError(TrainingError):
    """Training appears to be hung (no progress)."""


class CheckpointError(TrainingError):
    """Checkpoint corruption or loading failure."""


class OOMError(TrainingError):
    """CUDA out of memory during training."""


# --- Orchestration errors ---


class OrchestrationError(AutoTrainerError):
    """Pipeline orchestration failure."""


class PhaseTransitionError(OrchestrationError):
    """Invalid phase transition."""


class RecoveryError(OrchestrationError):
    """Failed to recover from a previous state."""


# --- Context / LLM errors ---


class ContextOverflowError(AutoTrainerError):
    """Context window budget exceeded."""


class LLMApiError(AutoTrainerError):
    """LLM API call failed."""


class SkillExecutionError(AutoTrainerError):
    """Skill handler execution failed."""
