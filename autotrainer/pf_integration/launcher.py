"""PaddleFormers training launcher.

Wraps paddleformers-cli train as a managed subprocess.
Does NOT import PaddleFormers at module level.
"""

from __future__ import annotations

import os
import sys
from typing import Iterator

from autotrainer.utils.subprocess_mgr import SubprocessManager


class PaddleFormersLauncher:
    """Launches PaddleFormers training via subprocess."""

    # Default to paddle conda env where paddlepaddle + paddleformers are installed
    _PADDLE_PYTHON = "/data/lizhijun/anaconda3/envs/paddle/bin/python3"

    def __init__(self, paddleformers_root: str, subprocess_mgr: SubprocessManager | None = None):
        self.pf_root = paddleformers_root
        self._mgr = subprocess_mgr or SubprocessManager()

    @staticmethod
    def _python_bin() -> str:
        """Return the Python binary path for the PaddlePaddle environment."""
        import os
        if os.path.exists(PaddleFormersLauncher._PADDLE_PYTHON):
            return PaddleFormersLauncher._PADDLE_PYTHON
        return sys.executable

    def _build_command(
        self,
        config_path: str,
        gpu_ids: list[int] | None = None,
    ) -> list[str]:
        """Build the command to launch paddleformers-cli train.

        Uses paddle.distributed.launch to wrap training, which manages
        the child process lifecycle properly for both single and multi GPU.
        """
        python = self._python_bin()

        # Use the pip-installed package path in the paddle conda env
        import subprocess
        result = subprocess.run(
            [python, "-c", "import paddleformers.cli.launcher; print(paddleformers.cli.launcher.__file__)"],
            capture_output=True, text=True,
        )
        pf_launcher_path = result.stdout.strip().split("\n")[-1]

        if gpu_ids and len(gpu_ids) > 1:
            gpu_str = ",".join(str(g) for g in gpu_ids)
        else:
            gpu_str = ",".join(str(g) for g in (gpu_ids or [0]))

        return [
            python,
            "-m",
            "paddle.distributed.launch",
            "--gpus",
            gpu_str,
            "--log_dir",
            "/tmp/paddle_distributed_logs",
            pf_launcher_path,
            "train",
            config_path,
        ]

    def build_env(self, gpu_ids: list[int] | None = None) -> dict[str, str]:
        """Build environment variables for the training subprocess."""
        # Don't set PYTHONPATH — use pip-installed paddleformers package directly.
        # Setting PYTHONPATH to pf_root causes import conflicts with site-packages.
        return {}

    def launch_training(
        self,
        config_path: str,
        gpu_ids: list[int] | None = None,
        env_overrides: dict[str, str] | None = None,
        log_dir: str = "",
        process_name: str = "training",
        on_stdout: callable = None,
        on_stderr: callable = None,
    ):
        """Launch PaddleFormers training.

        Returns the Popen process object.
        """
        cmd = self._build_command(config_path, gpu_ids)
        env = self.build_env(gpu_ids)
        if env_overrides:
            env.update(env_overrides)

        cwd = self.pf_root if self.pf_root else None

        return self._mgr.launch(
            name=process_name,
            cmd=cmd,
            env=env,
            cwd=cwd,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )

    def is_training_alive(self, process_name: str = "training") -> bool:
        """Check if the training process is still running."""
        return self._mgr.is_alive(process_name)

    def get_recent_log_lines(self, process_name: str = "training", n: int = 50) -> list[str]:
        """Get the last N stdout lines from the training process."""
        return self._mgr.get_recent_lines(process_name, n)

    def stop_training(self, process_name: str = "training", timeout: int = 30) -> bool:
        """Stop the training process gracefully."""
        return self._mgr.kill(process_name, timeout)

    def add_stdout_callback(self, process_name: str, callback):
        """Add a callback for training stdout lines."""
        self._mgr.add_stdout_callback(process_name, callback)

    def add_stderr_callback(self, process_name: str, callback):
        """Add a callback for training stderr lines."""
        self._mgr.add_stderr_callback(process_name, callback)
