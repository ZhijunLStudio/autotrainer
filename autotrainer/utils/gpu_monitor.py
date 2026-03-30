"""GPU monitoring via nvidia-smi."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    utilization_gpu: float
    utilization_memory: float
    temperature: float
    power_draw: float
    power_limit: float

    @property
    def memory_used_pct(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


class GPUMonitor:
    """Monitors NVIDIA GPU status via nvidia-smi."""

    def __init__(self, gpu_ids: list[int] | None = None):
        self.gpu_ids = gpu_ids
        self._available = self._check_nvidia_smi()

    @staticmethod
    def _check_nvidia_smi() -> bool:
        try:
            subprocess.run(["nvidia-smi", "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @property
    def available(self) -> bool:
        return self._available

    def get_gpu_info(self) -> list[GPUInfo]:
        """Query all GPUs (or specific gpu_ids) for current status."""
        if not self._available:
            return []

        query_fields = (
            "index,name,memory.total,memory.used,"
            "utilization.gpu,utilization.memory,"
            "temperature.gpu,power.draw,power.limit"
        )

        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []

            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 9:
                    continue

                gpu_id = int(parts[0])
                if self.gpu_ids and gpu_id not in self.gpu_ids:
                    continue

                gpus.append(
                    GPUInfo(
                        index=gpu_id,
                        name=parts[1],
                        memory_total_mb=float(parts[2]) if parts[2] != "[N/A]" else 0,
                        memory_used_mb=float(parts[3]) if parts[3] != "[N/A]" else 0,
                        utilization_gpu=float(parts[4]) if parts[4] != "[N/A]" else 0,
                        utilization_memory=float(parts[5]) if parts[5] != "[N/A]" else 0,
                        temperature=float(parts[6]) if parts[6] != "[N/A]" else 0,
                        power_draw=float(parts[7]) if parts[7] != "[N/A]" else 0,
                        power_limit=float(parts[8]) if parts[8] != "[N/A]" else 0,
                    )
                )
            return gpus
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return []

    def get_summary(self) -> dict:
        """Return a compact summary of GPU status."""
        gpus = self.get_gpu_info()
        if not gpus:
            return {"available": False, "gpu_count": 0}

        return {
            "available": True,
            "gpu_count": len(gpus),
            "total_memory_gb": sum(g.memory_total_mb for g in gpus) / 1024,
            "used_memory_gb": sum(g.memory_used_mb for g in gpus) / 1024,
            "avg_utilization": sum(g.utilization_gpu for g in gpus) / len(gpus),
            "max_temperature": max(g.temperature for g in gpus),
            "gpus": [
                {
                    "id": g.index,
                    "name": g.name,
                    "mem_used_pct": round(g.memory_used_pct, 1),
                    "util_gpu": g.utilization_gpu,
                    "temp": g.temperature,
                }
                for g in gpus
            ],
        }

    def detect_oom_risk(self, threshold_pct: float = 95.0) -> list[int]:
        """Return GPU indices that are at risk of OOM (memory > threshold)."""
        gpus = self.get_gpu_info()
        return [g.index for g in gpus if g.memory_used_pct > threshold_pct]

    def detect_thermal_risk(self, threshold_c: float = 85.0) -> list[int]:
        """Return GPU indices that are overheating."""
        gpus = self.get_gpu_info()
        return [g.index for g in gpus if g.temperature > threshold_c]

    def format_status_line(self) -> str:
        """Format a one-line GPU status for display."""
        gpus = self.get_gpu_info()
        if not gpus:
            return "GPU: N/A"

        parts = []
        for g in gpus:
            parts.append(f"GPU{g.index}: {g.utilization_gpu:.0f}%/{g.memory_used_pct:.0f}%mem {g.temperature:.0f}°C")
        return " | ".join(parts)
