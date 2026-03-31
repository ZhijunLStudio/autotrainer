"""Sandbox — execute LLM-generated Python scripts in a controlled subprocess.

The script receives INPUT_PATH and OUTPUT_PATH as environment variables.
Captures stdout/stderr, enforces timeout, returns structured result.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field


@dataclass
class SandboxResult:
    """Result of running a script in the sandbox."""

    success: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    output_path: str = ""
    output_rows: int = 0           # number of lines in output file

    @property
    def error_summary(self) -> str:
        """Compact error for LLM context (avoids huge tracebacks)."""
        if self.timed_out:
            return f"Script timed out after {self.elapsed_seconds:.0f}s"

        lines = []

        # Extract last traceback from stderr
        stderr_lines = self.stderr.splitlines()
        # Find last "Traceback" block
        last_tb_start = -1
        for i, line in enumerate(stderr_lines):
            if "Traceback" in line:
                last_tb_start = i
        if last_tb_start >= 0:
            tb_lines = stderr_lines[last_tb_start:]
            lines.extend(tb_lines[-15:])  # last 15 lines of traceback
        else:
            lines.extend(stderr_lines[-10:])

        if self.stdout:
            lines.append(f"[stdout tail]: {self.stdout[-500:]}")

        return "\n".join(lines)


class Sandbox:
    """Execute LLM-generated scripts safely."""

    def __init__(
        self,
        timeout: int = 300,          # seconds per run
        python_bin: str = "python",
    ):
        self.timeout = timeout
        self.python_bin = python_bin

    def run(
        self,
        script: str,
        input_path: str,
        output_path: str,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Write script to a temp file and run it.

        The script receives:
          INPUT_PATH  — path to the raw data file
          OUTPUT_PATH — where to write the erniekit JSONL output
        """
        result = SandboxResult(output_path=output_path)

        # Write script to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            script_path = f.name

        # Build environment
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_path)
        env["OUTPUT_PATH"] = str(output_path)
        env["PYTHONIOENCODING"] = "utf-8"
        if extra_env:
            env.update(extra_env)

        start = time.time()
        try:
            proc = subprocess.run(
                [self.python_bin, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            result.elapsed_seconds = time.time() - start
            result.exit_code = proc.returncode
            result.stdout = proc.stdout[-5000:] if proc.stdout else ""
            result.stderr = proc.stderr[-5000:] if proc.stderr else ""
            result.success = proc.returncode == 0

        except subprocess.TimeoutExpired:
            result.elapsed_seconds = time.time() - start
            result.timed_out = True
            result.success = False
            result.stderr = f"Script exceeded timeout of {self.timeout}s"

        except Exception as e:
            result.elapsed_seconds = time.time() - start
            result.success = False
            result.stderr = str(e)

        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

        # Count output rows if file exists
        if result.success and os.path.exists(output_path):
            try:
                with open(output_path, "r", errors="replace") as f:
                    result.output_rows = sum(1 for line in f if line.strip())
            except OSError:
                pass

        return result

    def validate_script(self, script: str) -> tuple[bool, str]:
        """Syntax-check a script without running it."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = subprocess.run(
                [self.python_bin, "-m", "py_compile", script_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return proc.returncode == 0, proc.stderr
        except Exception as e:
            return False, str(e)
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
