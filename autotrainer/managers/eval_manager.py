"""Evaluation manager — runs evaluation and inference verification."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autotrainer.utils.file_utils import atomic_write_json, safe_read_json


@dataclass
class EvalResult:
    """Result of an evaluation run."""

    experiment_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    ocr_metrics: dict[str, Any] = field(default_factory=dict)
    inference_samples: list[dict] = field(default_factory=list)
    report_path: str = ""
    status: str = "completed"
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "metrics": self.metrics,
            "ocr_metrics": self.ocr_metrics,
            "inference_samples": self.inference_samples,
            "report_path": self.report_path,
            "status": self.status,
            "timestamp": self.timestamp,
        }


class EvalManager:
    """Run evaluation and inference verification.

    Evaluation runs PaddleFormers eval mode on the validation set.
    Inference verification samples predictions to check quality beyond metrics.
    """

    # Python binary for the PaddlePaddle conda environment
    _PADDLE_PYTHON = "/data/lizhijun/anaconda3/envs/paddle/bin/python3"

    def __init__(self, paddleformers_root: str, work_dir: str):
        self.pf_root = paddleformers_root
        self.work_dir = work_dir

    def run_eval(self, config_path: str, checkpoint_path: str) -> EvalResult:
        """Run evaluation using PaddleFormers.

        Subprocesses paddleformers-cli with eval mode.
        """
        from datetime import datetime

        exp_id = Path(checkpoint_path).name
        log_path = os.path.join(self.work_dir, "logs", f"eval-{exp_id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        result = EvalResult(
            experiment_id=exp_id,
            timestamp=datetime.now().isoformat(),
            report_path=log_path,
        )

        cmd = [
            self._PADDLE_PYTHON,
            "-m",
            "paddleformers.cli.cli",
            "train",
            config_path,
            "--do_eval",
            "--resume_from_checkpoint",
            checkpoint_path,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=self.pf_root,
                env={**os.environ, "PYTHONPATH": self.pf_root},
            )

            # Write log
            with open(log_path, "w") as f:
                f.write(proc.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(proc.stderr)

            # Parse eval metrics from stdout
            metrics = self._parse_eval_output(proc.stdout)
            result.metrics = metrics
            result.status = "completed" if proc.returncode == 0 else "failed"

        except subprocess.TimeoutExpired:
            result.status = "timeout"
        except Exception as e:
            result.status = "error"
            result.metrics = {"error": str(e)}

        # Save result
        eval_result_path = os.path.join(self.work_dir, "eval_results", f"{exp_id}.json")
        os.makedirs(os.path.dirname(eval_result_path), exist_ok=True)
        atomic_write_json(eval_result_path, result.to_dict())

        return result

    def run_inference_samples(
        self,
        config_path: str,
        checkpoint_path: str,
        data_path: str,
        n_samples: int = 10,
    ) -> list[dict]:
        """Sample N data points and run inference for qualitative verification.

        This goes beyond metrics to check if the model output is reasonable.
        """
        import random

        # Read samples from dataset
        samples = []
        with open(data_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return []

        n = min(n_samples, len(lines))
        sampled_lines = random.sample(lines, n)

        for line in sampled_lines:
            try:
                data = json.loads(line.strip())
                samples.append(data)
            except json.JSONDecodeError:
                continue

        # Write samples to a temp file for inference
        samples_dir = os.path.join(self.work_dir, "inference_samples")
        os.makedirs(samples_dir, exist_ok=True)

        input_path = os.path.join(samples_dir, "input.jsonl")
        output_path = os.path.join(samples_dir, "output.jsonl")

        with open(input_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # TODO: Run inference via PaddleFormers and collect outputs
        # This would need PaddleFormers to support batch inference
        # For now, return the samples as-is

        return samples

    def compare_experiments(self, results: list[EvalResult]) -> dict:
        """Compare evaluation results across experiments."""
        if not results:
            return {}

        comparison = {
            "best": None,
            "worst": None,
            "rankings": [],
        }

        # Find best and worst by eval_loss if available
        scored = [(r.experiment_id, r.metrics.get("eval_loss", float("inf"))) for r in results]
        scored.sort(key=lambda x: x[1])

        if scored:
            comparison["best"] = {"id": scored[0][0], "eval_loss": scored[0][1]}
            comparison["worst"] = {"id": scored[-1][0], "eval_loss": scored[-1][1]}
            comparison["rankings"] = [{"id": eid, "eval_loss": loss} for eid, loss in scored]

        return comparison

    def run_ocr_eval(
        self,
        predictions: list[str],
        references: list[str],
        pred_htmls: list[str] | None = None,
        ref_htmls: list[str] | None = None,
    ) -> dict:
        """Compute OCR metrics from prediction/reference pairs.

        Call this after inference when you have raw predictions and ground truth.
        Returns the full metrics dict suitable for storing in EvalResult.ocr_metrics.
        """
        from autotrainer.metrics.compute import compute_all_metrics

        metrics = compute_all_metrics(
            predictions=predictions,
            references=references,
            pred_htmls=pred_htmls,
            ref_htmls=ref_htmls,
        )
        return metrics.to_dict()

    def _parse_eval_output(self, output: str) -> dict:
        """Parse eval metrics from PaddleFormers stdout."""
        import re

        metrics = {}

        patterns = {
            "eval_loss": r"eval[_\s].*?loss[:\s]+(\d+\.?\d*)",
            "eval_accuracy": r"(?:eval[_\s].*?accuracy|acc)[:\s]+(\d+\.?\d*)",
            "eval_perplexity": r"eval[_\s].*?perplexity[:\s]+(\d+\.?\d*)",
        }

        for name, pattern in patterns.items():
            m = re.search(pattern, output, re.IGNORECASE)
            if m:
                try:
                    metrics[name] = float(m.group(1))
                except ValueError:
                    pass

        return metrics
