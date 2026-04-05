"""PaddleFormers training log parser.

Parses stdout/stderr from paddleformers-cli train to extract
metrics, errors, and progress information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LogMetrics:
    """Extracted metrics from a training log line."""

    step: int | None = None
    loss: float | None = None
    lr: float | None = None
    eval_loss: float | None = None
    throughput: float | None = None
    memory_mb: float | None = None
    epoch: int | None = None
    grad_norm: float | None = None
    eval_ppl: float | None = None
    extra_metrics: dict = field(default_factory=dict)
    has_nan: bool = False
    raw_line: str = ""


@dataclass
class LogError:
    """Extracted error from a training log."""

    error_type: str
    message: str
    raw_line: str
    severity: str = "error"  # error, warning, fatal


class LogParser:
    """Parse PaddleFormers training output for metrics and errors."""

    # Metric extraction patterns
    LOSS_PATTERN = re.compile(r"loss[:\s]+(\d+\.\d+)")
    STEP_PATTERN = re.compile(r"global_step[:\s]+(\d+)")
    LR_PATTERN = re.compile(r"(?:learning_rate|lr)[:\s]+([\d.eE+-]+)")
    MEM_PATTERN = re.compile(r"(?:allocated_memory|memory)[:\s]+([\d.]+)\s*MB")
    SPEED_PATTERN = re.compile(r"(?:throughput|speed)[:\s]+([\d.]+)\s*tokens/s")
    EVAL_PATTERN = re.compile(r"eval[_\s].*?loss[:\s]+(\d+\.\d+)")
    EPOCH_PATTERN = re.compile(r"epoch[:\s]+(\d+)")
    GRAD_NORM_PATTERN = re.compile(r"grad_norm[=:]\s*([\d.]+(?:[eE][+-]?\d+)?)")
    EVAL_KV_PATTERN = re.compile(r"eval_(\w+)[:\s]+([\d.eE+-]+)")
    NAN_LOSS_PATTERN = re.compile(r"loss[:\s]+(?:nan|inf)", re.IGNORECASE)

    # Error patterns
    OOM_PATTERNS = [
        re.compile(r"(?:CUDA|GPU)\s*(?:out\s*of\s*memory|OOM)", re.IGNORECASE),
        re.compile(r"ResourceExhaustedError", re.IGNORECASE),
    ]
    NAN_PATTERN = re.compile(r"loss.*(?:NaN|nan|inf)", re.IGNORECASE)
    NCCL_PATTERN = re.compile(r"NCCL.*(?:error|timeout|abort)", re.IGNORECASE)
    FATAL_PATTERN = re.compile(r"(?:FATAL|Fatal error|CRITICAL)", re.IGNORECASE)

    def parse_line(self, line: str) -> LogMetrics | None:
        """Try to extract metrics from a single log line."""
        line = line.strip()
        if not line:
            return None

        metrics = LogMetrics(raw_line=line)

        # Try each pattern
        m = self.STEP_PATTERN.search(line)
        if m:
            metrics.step = int(m.group(1))

        m = self.LOSS_PATTERN.search(line)
        if m:
            try:
                metrics.loss = float(m.group(1))
            except ValueError:
                pass

        m = self.LR_PATTERN.search(line)
        if m:
            try:
                metrics.lr = float(m.group(1))
            except ValueError:
                pass

        m = self.MEM_PATTERN.search(line)
        if m:
            try:
                metrics.memory_mb = float(m.group(1))
            except ValueError:
                pass

        m = self.SPEED_PATTERN.search(line)
        if m:
            try:
                metrics.throughput = float(m.group(1))
            except ValueError:
                pass

        m = self.EVAL_PATTERN.search(line)
        if m:
            try:
                metrics.eval_loss = float(m.group(1))
            except ValueError:
                pass

        m = self.EPOCH_PATTERN.search(line)
        if m:
            metrics.epoch = int(m.group(1))

        # Parse grad_norm
        grad_match = self.GRAD_NORM_PATTERN.search(line)
        if grad_match:
            try:
                metrics.grad_norm = float(grad_match.group(1))
            except ValueError:
                pass

        # Generic eval metric extraction
        for eval_match in self.EVAL_KV_PATTERN.finditer(line):
            key = eval_match.group(1)
            try:
                value = float(eval_match.group(2))
            except ValueError:
                continue
            if key == "loss":
                metrics.eval_loss = value
            elif key == "ppl":
                metrics.eval_ppl = value
            else:
                metrics.extra_metrics[f"eval_{key}"] = value

        # NaN/inf detection in loss values
        if self.NAN_LOSS_PATTERN.search(line):
            metrics.has_nan = True

        # Return None if nothing was extracted
        if (metrics.step is None and metrics.loss is None
                and metrics.eval_loss is None and not metrics.extra_metrics):
            return None

        return metrics

    def detect_error(self, line: str) -> LogError | None:
        """Check if a log line contains an error."""
        for pattern in self.OOM_PATTERNS:
            if pattern.search(line):
                return LogError(error_type="OOM", message=line.strip(), raw_line=line, severity="fatal")

        if self.FATAL_PATTERN.search(line):
            return LogError(error_type="FATAL", message=line.strip(), raw_line=line, severity="fatal")

        if self.NAN_PATTERN.search(line):
            return LogError(error_type="NaN", message=line.strip(), raw_line=line, severity="error")

        if self.NCCL_PATTERN.search(line):
            return LogError(error_type="NCCL", message=line.strip(), raw_line=line, severity="error")

        if re.search(r"(?:^|\s)ERROR", line):
            return LogError(error_type="ERROR", message=line.strip(), raw_line=line, severity="error")

        # Only flag meaningful warnings, not NCCL/LAUNCH noise
        if re.search(r"(?:^|\s)WARNING", line, re.IGNORECASE) and not re.search(r"NCCL|LAUNCH|RAS", line):
            return LogError(error_type="WARNING", message=line.strip(), raw_line=line, severity="warning")

        return None

    def extract_metrics(self, log_text: str) -> list[LogMetrics]:
        """Extract all metrics from a log text."""
        result = []
        for line in log_text.splitlines():
            m = self.parse_line(line)
            if m:
                result.append(m)
        return result

    def extract_final_metrics(self, log_path: str) -> dict:
        """Extract final training metrics from a log file.

        Returns dict with: final_loss, final_step, max_steps, total_epochs, etc.
        """
        p = Path(log_path)
        if not p.exists():
            return {}

        # Read last 200 lines for final metrics
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()

        # Extract max_steps from the header (look in first 2000 lines)
        max_steps = 0
        for line in lines[:2000]:
            m = re.search(r"Total optimization steps\s*=\s*([\d,]+)", line)
            if m:
                max_steps = int(m.group(1).replace(",", ""))
                break

        tail = "".join(lines[-200:])

        metrics_list = self.extract_metrics(tail)
        if not metrics_list:
            return {}

        # Find the last metric that has a step (skip summary lines without step)
        last_with_step = None
        for m in reversed(metrics_list):
            if m.step is not None:
                last_with_step = m
                break
        last = metrics_list[-1]  # May have train_loss from summary

        result = {"max_steps": max_steps}
        # Use summary loss if available, otherwise last step loss
        if last.loss is not None and last.step is None:
            result["final_loss"] = last.loss
        elif last_with_step and last_with_step.loss is not None:
            result["final_loss"] = last_with_step.loss
        if last_with_step and last_with_step.step is not None:
            result["final_step"] = last_with_step.step
        if last.eval_loss is not None:
            result["final_eval_loss"] = last.eval_loss
        if last.throughput is not None:
            result["throughput"] = last.throughput

        # Find all eval losses for trend
        eval_losses = [m.eval_loss for m in metrics_list if m.eval_loss is not None]
        if eval_losses:
            result["eval_losses"] = eval_losses
            result["best_eval_loss"] = min(eval_losses)

        return result

    def get_error_summary(self, log_path: str) -> list[dict]:
        """Scan a log file for all errors and return summaries."""
        p = Path(log_path)
        if not p.exists():
            return []

        errors = []
        with open(log_path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 100_000:  # Safety limit
                    break
                err = self.detect_error(line)
                if err and err.severity in ("error", "fatal"):
                    errors.append(
                        {
                            "line_num": i + 1,
                            "error_type": err.error_type,
                            "message": err.message[:200],
                            "severity": err.severity,
                        }
                    )

        return errors
