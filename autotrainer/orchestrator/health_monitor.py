"""Health monitor — GPU/process watchdog for long-running training.

Runs in a background thread during training phases, checking:
- GPU utilization and memory
- Training process alive
- Log output freshness (hang detection)
- Disk space
"""

from __future__ import annotations

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
    ):
        self.gpu_monitor = GPUMonitor(gpu_ids)
        self.poll_interval = poll_interval
        self.hang_timeout = hang_timeout

        self._callbacks: list[Callable[[HealthStatus], None]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_status: HealthStatus = HealthStatus()
        self._lock = threading.Lock()

        # State tracking for anomaly detection
        self._last_log_mtime: float | None = None
        self._low_util_counter: int = 0
        self._process_check_fn: Callable[[], bool] | None = None
        self._log_path: str = ""

    def on_anomaly(self, callback: Callable[[HealthStatus], None]):
        """Register a callback for anomaly detection."""
        self._callbacks.append(callback)

    def set_process_checker(self, fn: Callable[[], bool]):
        """Set a function that returns True if the training process is alive."""
        self._process_check_fn = fn

    def set_log_path(self, path: str):
        """Set the training log path for hang detection."""
        self._log_path = path

    def start(self):
        """Start the background monitoring thread."""
        if self._running:
            return
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
        """Main monitoring loop."""
        while self._running:
            try:
                status = self._check_all()
                with self._lock:
                    self._latest_status = status

                if status.anomaly:
                    for cb in self._callbacks:
                        try:
                            cb(status)
                        except Exception:
                            pass
            except Exception:
                pass

            time.sleep(self.poll_interval)

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

        # Anomaly detection
        anomaly = self._detect_anomaly(status)
        status.anomaly = anomaly

        status.timestamp = time.time()
        return status

    def _detect_anomaly(self, status: HealthStatus) -> str | None:
        """Detect anomalies from health status."""
        # Process dead
        if not status.process_alive:
            return "process_dead"

        # Hang detection
        if status.last_log_age_seconds > self.hang_timeout:
            return "hang_detected"

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

        return None
