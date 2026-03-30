"""Managed subprocess with output streaming and lifecycle control."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator


@dataclass
class ProcessInfo:
    """Metadata about a managed subprocess."""

    name: str
    process: subprocess.Popen
    started_at: float = field(default_factory=time.time)
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    _stdout_thread: threading.Thread | None = None
    _stderr_thread: threading.Thread | None = None
    _on_stdout: list[Callable] = field(default_factory=list)
    _on_stderr: list[Callable] = field(default_factory=list)


class SubprocessManager:
    """Manages subprocess lifecycle with real-time output streaming."""

    def __init__(self):
        self._processes: dict[str, ProcessInfo] = {}
        self._lock = threading.Lock()

    def launch(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> subprocess.Popen:
        """Launch a managed subprocess with optional output callbacks.

        Args:
            name: Unique name for this process (used for kill/stream).
            cmd: Command as list of strings.
            env: Environment variables (inherits parent if None).
            cwd: Working directory.
            on_stdout: Callback for each stdout line.
            on_stderr: Callback for each stderr line.

        Returns:
            The Popen object.
        """
        with self._lock:
            if name in self._processes:
                if self._processes[name].process.poll() is None:
                    raise RuntimeError(f"Process '{name}' is already running (pid={self._processes[name].process.pid})")

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            cwd=cwd,
            text=True,
            bufsize=1,  # line-buffered
        )

        info = ProcessInfo(name=name, process=proc)
        if on_stdout:
            info._on_stdout.append(on_stdout)
        if on_stderr:
            info._on_stderr.append(on_stderr)

        def _reader(stream, lines_store, callbacks, label):
            """Read lines from stream, store them, and invoke callbacks."""
            try:
                for line in stream:
                    line = line.rstrip("\n")
                    lines_store.append(line)
                    for cb in callbacks:
                        try:
                            cb(line)
                        except Exception:
                            pass
            except ValueError:
                # Stream closed during iteration
                pass

        info._stdout_thread = threading.Thread(
            target=_reader,
            args=(proc.stdout, info.stdout_lines, info._on_stdout, "stdout"),
            daemon=True,
            name=f"{name}-stdout",
        )
        info._stderr_thread = threading.Thread(
            target=_reader,
            args=(proc.stderr, info.stderr_lines, info._on_stderr, "stderr"),
            daemon=True,
            name=f"{name}-stderr",
        )
        info._stdout_thread.start()
        info._stderr_thread.start()

        with self._lock:
            self._processes[name] = info

        return proc

    def add_stdout_callback(self, name: str, callback: Callable[[str], None]):
        """Add a stdout callback to an already-running process."""
        with self._lock:
            info = self._processes.get(name)
        if info:
            info._on_stdout.append(callback)

    def add_stderr_callback(self, name: str, callback: Callable[[str], None]):
        """Add a stderr callback to an already-running process."""
        with self._lock:
            info = self._processes.get(name)
        if info:
            info._on_stderr.append(callback)

    def get_info(self, name: str) -> ProcessInfo | None:
        """Get process info by name."""
        return self._processes.get(name)

    def is_alive(self, name: str) -> bool:
        """Check if a managed process is still running."""
        with self._lock:
            info = self._processes.get(name)
        return info is not None and info.process.poll() is None

    def get_exit_code(self, name: str) -> int | None:
        """Get exit code of a process (None if still running)."""
        with self._lock:
            info = self._processes.get(name)
        return info.process.poll() if info else None

    def get_recent_lines(self, name: str, n: int = 50) -> list[str]:
        """Get the last N stdout lines."""
        with self._lock:
            info = self._processes.get(name)
        return info.stdout_lines[-n:] if info else []

    def kill(self, name: str, timeout: int = 30) -> bool:
        """Kill a process tree gracefully (SIGTERM, then SIGKILL).

        Returns True if process was killed successfully.
        """
        with self._lock:
            info = self._processes.get(name)
        if not info or info.process.poll() is not None:
            return False

        proc = info.process
        pid = proc.pid

        # Try graceful shutdown
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            return True

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)

        return proc.poll() is not None

    def kill_all(self, timeout: int = 30):
        """Kill all managed processes."""
        for name in list(self._processes.keys()):
            self.kill(name, timeout)

    def list_processes(self) -> dict[str, bool]:
        """List all managed processes and their alive status."""
        return {name: self.is_alive(name) for name in self._processes}
