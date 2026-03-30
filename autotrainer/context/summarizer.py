"""Log and error summarization — compresses large raw logs into structured summaries."""

from __future__ import annotations

import re
from pathlib import Path

from autotrainer.context.token_counter import estimate_tokens, truncate_to_tokens


class LogSummarizer:
    """Extracts structured summaries from raw training logs.

    Key principle: raw logs never enter the LLM context directly.
    This class converts 100KB+ logs into ~1KB structured summaries.
    """

    # Patterns that indicate important log lines
    KEY_PATTERNS = [
        re.compile(r"loss[:\s]+\d+\.?\d*", re.IGNORECASE),
        re.compile(r"global_step[:\s]+\d+", re.IGNORECASE),
        re.compile(r"OOM|OutOfMemory|out of memory", re.IGNORECASE),
        re.compile(r"NaN|nan|NAN", re.IGNORECASE),
        re.compile(r"NCCL|nccl", re.IGNORECASE),
        re.compile(r"[Cc]heckpoint saved", re.IGNORECASE),
        re.compile(r"eval[_\s].*loss", re.IGNORECASE),
        re.compile(r"ERROR|FATAL|Traceback", re.IGNORECASE),
        re.compile(r"throughput[:\s]+[\d.]+\s*tokens/s", re.IGNORECASE),
        re.compile(r"learning_rate[:\s]+[\d.e-]+", re.IGNORECASE),
    ]

    def is_key_line(self, line: str) -> bool:
        """Check if a log line is important enough to keep."""
        return any(pattern.search(line) for pattern in self.KEY_PATTERNS)

    def skeleton_extract(self, raw_log: str, max_lines: int = 100) -> str:
        """Extract a structured skeleton from a raw log string.

        Keeps only key lines + first/last few lines for context.
        Returns a compact string suitable for LLM context.
        """
        lines = raw_log.splitlines()

        if len(lines) <= max_lines:
            return raw_log

        key_lines = []
        for line in lines:
            if self.is_key_line(line):
                key_lines.append(line)

        # Add first 5 and last 5 lines for crash context
        head = lines[:5]
        tail = lines[-5:]

        # Deduplicate while preserving order
        seen = set()
        result = []
        for line in head + key_lines + tail:
            if line not in seen:
                seen.add(line)
                result.append(line)

        if len(result) > max_lines:
            result = result[: max_lines // 2] + ["...[truncated]..."] + result[-(max_lines // 2) :]

        return "\n".join(result)

    def summarize_log_file(self, path: str, max_lines: int = 200) -> str:
        """Read and summarize a log file from disk.

        Uses file offset reading to avoid loading huge files entirely.
        """
        p = Path(path)
        if not p.exists():
            return f"[Log file not found: {path}]"

        file_size = p.stat().st_size

        if file_size < 500_000:  # < 500KB, read all
            content = p.read_text(errors="replace")
        else:
            # Read first 50KB and last 50KB
            with open(path, "r", errors="replace") as f:
                head = f.read(50_000)
                f.seek(-50_000, 2)
                tail = f.read()
            content = head + "\n...[file truncated]...\n" + tail

        return self.skeleton_extract(content, max_lines)

    def summarize_error(self, traceback_text: str) -> dict:
        """Extract structured error information from a traceback.

        Returns:
            Dict with: error_type, message, file, line, key_frames, summary
        """
        lines = traceback_text.strip().splitlines()

        error_type = "UnknownError"
        error_msg = ""
        file_location = ""
        line_number = ""
        key_frames = []

        for line in reversed(lines):
            line = line.strip()
            if line.startswith(("Traceback", "...")):
                continue
            # Match: ErrorType: message
            match = re.match(r"^(\w+(?:Error|Exception)):\s*(.+)$", line)
            if match:
                error_type = match.group(1)
                error_msg = match.group(2)
                continue
            # Match: File "...", line N, in func
            match = re.match(r'File "([^"]+)", line (\d+), in (\S+)', line)
            if match:
                file_location = match.group(1)
                line_number = match.group(2)
                key_frames.append(f"{match.group(3)} ({file_location}:{line_number})")
                if len(key_frames) >= 3:
                    break

        summary = f"{error_type}: {error_msg}"
        if key_frames:
            summary += f"\nFrames: {' -> '.join(key_frames)}"

        return {
            "error_type": error_type,
            "message": error_msg,
            "file": file_location,
            "line": line_number,
            "key_frames": key_frames,
            "summary": truncate_to_tokens(summary, 500),
        }

    def extract_key_metrics(self, log_text: str) -> dict:
        """Extract training metrics from log text.

        Returns dict with: steps, losses, lrs, eval_losses, etc.
        """
        steps = []
        losses = []
        lrs = []
        eval_losses = []

        for line in log_text.splitlines():
            # loss
            m = re.search(r"loss[:\s]+(\d+\.?\d*)", line, re.IGNORECASE)
            if m:
                try:
                    losses.append(float(m.group(1)))
                except ValueError:
                    pass

            # step
            m = re.search(r"global_step[:\s]+(\d+)", line, re.IGNORECASE)
            if m:
                steps.append(int(m.group(1)))

            # learning rate
            m = re.search(r"learning_rate[:\s]+([\d.e-]+)", line, re.IGNORECASE)
            if m:
                try:
                    lrs.append(float(m.group(1)))
                except ValueError:
                    pass

            # eval loss
            m = re.search(r"eval[_\s].*?loss[:\s]+(\d+\.?\d*)", line, re.IGNORECASE)
            if m:
                try:
                    eval_losses.append(float(m.group(1)))
                except ValueError:
                    pass

        result = {}
        if losses:
            result["latest_loss"] = losses[-1]
            result["loss_trend"] = "decreasing" if losses[-1] < losses[0] else "increasing"
            result["loss_values"] = losses[-20:]  # Keep last 20
        if steps:
            result["latest_step"] = steps[-1]
        if lrs:
            result["latest_lr"] = lrs[-1]
        if eval_losses:
            result["latest_eval_loss"] = eval_losses[-1]
            if losses:
                result["overfit_gap"] = eval_losses[-1] - losses[-1]

        return result

    def summarize_experiment(self, experiment_dir: str) -> str:
        """Generate a ~300-word summary of an experiment directory.

        Looks for: config.yaml, train.log, metrics, eval results.
        """
        from autotrainer.utils.file_utils import safe_read_json, safe_read_text

        p = Path(experiment_dir)
        summary_parts = []

        # Config
        config_path = p / "config.yaml"
        if config_path.exists():
            summary_parts.append(f"Config: {safe_read_text(str(config_path))[:500]}")

        # Metrics
        metrics = safe_read_json(str(p / "metrics.json"))
        if metrics:
            summary_parts.append(f"Metrics: {json.dumps(metrics, indent=2)[:500]}")

        # Log summary
        log_path = p / "train.log"
        if log_path.exists():
            log_summary = self.summarize_log_file(str(log_path), max_lines=50)
            summary_parts.append(f"Log:\n{log_summary[:500]}")

        return "\n---\n".join(summary_parts)


import json  # noqa: E402 — used in summarize_experiment
