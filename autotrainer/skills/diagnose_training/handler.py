"""Diagnose Training skill handler — error pattern matching and fix suggestions."""

from __future__ import annotations

from typing import Any


class DiagnoseTrainingHandler:
    """Handles training error diagnosis with two-tier approach."""

    def __init__(self):
        pass

    def diagnose(self, error_text: str, training_context: dict) -> dict:
        """Diagnose an error from text and training context.

        Uses regex pattern matching for common errors.
        """
        import re

        error_lower = error_text.lower()

        # OOM detection
        oom_patterns = [r"out of memory", r"oom", r"resourceexhaustederror"]
        if any(re.search(p, error_lower) for p in oom_patterns):
            return {
                "error_type": "oom",
                "category": "memory",
                "root_cause": "GPU memory exhausted during training",
                "severity": "fatal",
                "confidence": 0.95,
                "fix_steps": [
                    "Reduce per_device_train_batch_size by 50%",
                    "Enable gradient_checkpointing: true",
                    "If still OOM, reduce max_seq_len",
                    "Consider using sharding (stage2)",
                ],
                "config_changes": {
                    "finetuning.gradient_checkpointing": True,
                },
            }

        # NaN detection
        if re.search(r"loss.*nan|nan.*loss|nanloss", error_lower):
            return {
                "error_type": "nan",
                "category": "numerical",
                "root_cause": "Loss became NaN — likely learning rate too high or bad data",
                "severity": "error",
                "confidence": 0.9,
                "fix_steps": [
                    "Reduce learning_rate by 10x",
                    "Add max_grad_norm: 1.0",
                    "Check training data for NaN values",
                ],
                "config_changes": {
                    "finetuning.learning_rate": training_context.get("learning_rate", 1e-4) * 0.1,
                    "finetuning.max_grad_norm": 1.0,
                },
            }

        # NCCL detection
        if re.search(r"nccl", error_lower):
            return {
                "error_type": "nccl",
                "category": "distributed",
                "root_cause": "NCCL communication failure between GPUs",
                "severity": "error",
                "confidence": 0.85,
                "fix_steps": [
                    "Set NCCL_TIMEOUT=1800 in environment",
                    "Check network connectivity between nodes",
                    "Try reducing number of GPUs",
                ],
                "config_changes": {},
            }

        # Data format detection
        if re.search(r"jsondecodeerror|json.*decode|keyerror|unexpected.*format", error_lower):
            return {
                "error_type": "data_format",
                "category": "data",
                "root_cause": "Data format doesn't match expected schema",
                "severity": "error",
                "confidence": 0.9,
                "fix_steps": [
                    "Run: autotrainer data --mode fixed --task <task> --data-path <path>",
                    "Check that JSONL lines have required fields",
                    "Verify image paths are accessible",
                ],
                "config_changes": {},
            }

        # Checkpoint corruption
        if re.search(r"checkpoint.*error|checkpoint.*corrupt|failed.*load", error_lower):
            return {
                "error_type": "checkpoint",
                "category": "checkpoint",
                "root_cause": "Checkpoint file may be corrupted",
                "severity": "error",
                "confidence": 0.8,
                "fix_steps": [
                    "Resume from the previous checkpoint-N",
                    "Delete corrupted checkpoint directory",
                    "Reduce save_steps to have more frequent checkpoints",
                ],
                "config_changes": {},
            }

        # Unknown
        return {
            "error_type": "unknown",
            "category": "other",
            "root_cause": "Could not determine root cause from error text",
            "severity": "error",
            "confidence": 0.3,
            "fix_steps": [
                "Review the full training log for context",
                "Search for similar errors online",
                "Consider LLM-assisted diagnosis (force_llm=True)",
            ],
            "config_changes": {},
        }

    def suggest_fix(self, diagnosis: dict) -> dict:
        """Convert a diagnosis into actionable fix instructions."""
        return {
            "action": diagnosis.get("error_type", "unknown"),
            "steps": diagnosis.get("fix_steps", []),
            "config_changes": diagnosis.get("config_changes", {}),
            "confidence": diagnosis.get("confidence", 0.0),
            "requires_confirmation": True,
        }
