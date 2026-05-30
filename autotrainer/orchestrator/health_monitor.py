"""Health monitor — GPU/process watchdog for long-running training.

Runs in a background thread during training phases, checking:
- GPU utilization and memory
- Training process alive
- Log output freshness (hang detection)
- Disk space
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from autotrainer.utils.gpu_monitor import GPUMonitor
from autotrainer.utils.file_utils import get_file_mtime


@dataclass
class HealthStatus:
    """Current health status snapshot."""

    gpu_available: bool = False
    gpu_util: list[float] = field(default_factory=list)
    gpu_mem_used_pct: list[float] = field(default_factory=list)
    gpu_temp: list[float] = field(default_factory=list)
    process_alive: bool = False
    last_log_age_seconds: float = 0.0
    disk_free_gb: float = 0.0
    disk_used_pct: float = 0.0
    training_step: int | None = None
    loss: float | None = None
    anomaly: str | None = None
    timestamp: float = field(default_factory=time.time)


class HealthMonitor:
    """Real-time health monitoring with anomaly detection.

    Polls GPU and process status at regular intervals and invokes
    callbacks when anomalies are detected.
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        poll_interval: float = 5.0,
        hang_timeout: float = 300.0,
        work_dir: str = "",
    ):
        self.gpu_monitor = GPUMonitor(gpu_ids)
        self.poll_interval = poll_interval
        self.hang_timeout = hang_timeout
        self.work_dir = work_dir

        self._callbacks: list[Callable[[HealthStatus], None]] = []
        self._action_callback: Callable[[HealthStatus], None] | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_status: HealthStatus = HealthStatus()
        self._lock = threading.Lock()

        # State tracking for anomaly detection
        self._start_time: float = 0.0
        self._startup_grace: float = 30.0  # 30s grace period before anomaly detection
        self._last_log_mtime: float | None = None
        self._low_util_counter: int = 0
        self._process_check_fn: Callable[[], bool] | None = None
        self._process_kill_fn: Callable[[], bool] | None = None
        self._log_path: str = ""
        self._anomaly_counters: dict[str, int] = {}
        self._last_action_time: float = 0.0
        self._action_cooldown: float = 300.0  # 5 min between auto-actions

    def on_anomaly(self, callback: Callable[[HealthStatus], None]):
        """Register a callback for anomaly detection (notification only)."""
        self._callbacks.append(callback)

    def on_action(self, callback: Callable[[HealthStatus], None]):
        """Register a callback for automatic recovery actions.

        This callback receives health status and should take concrete action:
        kill hung processes, free disk space, etc.
        """
        self._action_callback = callback

    def set_process_checker(self, fn: Callable[[], bool]):
        """Set a function that returns True if the training process is alive."""
        self._process_check_fn = fn

    def set_process_killer(self, fn: Callable[[], bool]):
        """Set a function to kill the training process. Returns True on success."""
        self._process_kill_fn = fn

    def set_log_path(self, path: str):
        """Set the training log path for hang detection."""
        self._log_path = path

    def start(self):
        """Start the background monitoring thread."""
        if self._running:
            return
        self._start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="health-monitor")
        self._thread.start()

    def stop(self):
        """Stop the background monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def get_status(self) -> HealthStatus:
        """Get the latest health status snapshot."""
        with self._lock:
            return self._latest_status

    def _monitor_loop(self):
        """Main monitoring loop with auto-action triggering."""
        while self._running:
            try:
                status = self._check_all()
                with self._lock:
                    self._latest_status = status

                if status.anomaly:
                    # Notify all observers
                    for cb in self._callbacks:
                        try:
                            cb(status)
                        except Exception:
                            pass

                    # Trigger auto-recovery action (with cooldown)
                    self._maybe_trigger_action(status)
            except Exception:
                pass

            time.sleep(self.poll_interval)

    def _maybe_trigger_action(self, status: HealthStatus):
        """Trigger auto-recovery action if anomaly persists and cooldown elapsed."""
        anomaly_key = status.anomaly or ""
        self._anomaly_counters[anomaly_key] = self._anomaly_counters.get(anomaly_key, 0) + 1

        # Only act after anomaly has persisted for 2 consecutive checks (10s)
        if self._anomaly_counters[anomaly_key] < 2:
            return

        # Cooldown: don't spam actions
        now = time.time()
        if now - self._last_action_time < self._action_cooldown:
            return

        self._last_action_time = now

        # Execute action callback
        if self._action_callback:
            try:
                self._action_callback(status)
            except Exception:
                pass

        # Built-in auto-actions: kill hung processes
        if anomaly_key == "hang_detected" and self._process_kill_fn:
            try:
                killed = self._process_kill_fn()
                if killed:
                    status.anomaly = "process_killed_by_watchdog"
                    with self._lock:
                        self._latest_status = status
            except Exception:
                pass

    def _check_all(self) -> HealthStatus:
        """Run all health checks."""
        status = HealthStatus()

        # GPU check
        gpus = self.gpu_monitor.get_gpu_info()
        if gpus:
            status.gpu_available = True
            status.gpu_util = [g.utilization_gpu for g in gpus]
            status.gpu_mem_used_pct = [g.memory_used_pct for g in gpus]
            status.gpu_temp = [g.temperature for g in gpus]

        # Process check
        if self._process_check_fn:
            status.process_alive = self._process_check_fn()
        else:
            status.process_alive = True  # Assume alive if no checker

        # Log freshness check
        if self._log_path:
            mtime = get_file_mtime(self._log_path)
            if mtime is not None:
                if self._last_log_mtime is not None:
                    age = time.time() - mtime
                    status.last_log_age_seconds = age
                self._last_log_mtime = mtime

        # Disk space check
        if self.work_dir:
            try:
                usage = shutil.disk_usage(self.work_dir)
                status.disk_free_gb = usage.free / (1024 ** 3)
                status.disk_used_pct = (usage.used / usage.total) * 100
            except OSError:
                pass

        # Anomaly detection
        anomaly = self._detect_anomaly(status)
        status.anomaly = anomaly

        status.timestamp = time.time()
        return status

    def _detect_anomaly(self, status: HealthStatus) -> str | None:
        """Detect anomalies from health status."""
        # Startup grace: skip anomaly detection during warm-up
        if time.time() - self._start_time < self._startup_grace:
            return None

        # Process dead
        if not status.process_alive:
            self._anomaly_counters.clear()
            return "process_dead"

        # Hang detection
        if status.last_log_age_seconds > self.hang_timeout:
            return "hang_detected"

        # Disk space critical
        if status.disk_used_pct > 95:
            return "disk_full"

        # Disk space warning
        if status.disk_free_gb < 10 and status.disk_free_gb > 0:
            return "disk_low"

        # GPU OOM risk
        for i, mem in enumerate(status.gpu_mem_used_pct):
            if mem > 95:
                return f"oom_risk_gpu{i}"

        # GPU thermal risk
        for i, temp in enumerate(status.gpu_temp):
            if temp > 90:
                return f"thermal_risk_gpu{i}"

        # Low GPU utilization (sustained)
        if status.gpu_available and all(u < 5.0 for u in status.gpu_util):
            self._low_util_counter += 1
            if self._low_util_counter > 12:  # 60 seconds of low utilization
                return "low_gpu_utilization"
        else:
            self._low_util_counter = 0

        # No anomaly — reset counters
        self._anomaly_counters.clear()
        return None
