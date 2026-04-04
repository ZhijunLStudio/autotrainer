"""Environment manager — checks and sets up the training environment.

Verifies GPU drivers, PaddlePaddle, PaddleFormers, and required packages.
Does NOT import PaddlePaddle at module level.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


@dataclass
class EnvStatus:
    """Current environment status."""

    python_version: str = ""
    paddle_version: str = ""
    paddleformers_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_memory_gb: list[float] = field(default_factory=list)
    packages: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    ready: bool = False


def _run_cmd(cmd: list[str], timeout: int = 30) -> str:
    """Run a command and return stdout, empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _get_version(package: str) -> str:
    """Get version of an installed package via importlib.metadata (same Python, no subprocess)."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


class EnvManager:
    """Environment verification and setup."""

    # Python binary for the PaddlePaddle conda environment
    _PADDLE_PYTHON = "/data/lizhijun/anaconda3/envs/paddle/bin/python3"

    # Required packages for PaddleFormers training
    REQUIRED_PACKAGES = [
        "paddlepaddle-gpu",
        "tqdm",
        "colorlog",
        "safetensors",
        "visualdl",
        "numpy",
    ]

    def __init__(self, paddleformers_root: str = ""):
        self.pf_root = paddleformers_root

    def _get_version_in_paddle_env(self, package: str) -> str:
        """Get version of a package installed in the paddle conda env."""
        import subprocess
        result = subprocess.run(
            [self._PADDLE_PYTHON, "-c",
             f"import importlib.metadata; print(importlib.metadata.version('{package}'))"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def check_environment(self) -> EnvStatus:
        """Run a comprehensive environment check."""
        status = EnvStatus()

        # Python version
        import sys

        status.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # GPU
        gpu_info = self._check_gpus()
        status.gpu_count = gpu_info.get("count", 0)
        status.gpu_names = gpu_info.get("names", [])
        status.gpu_memory_gb = gpu_info.get("memory_gb", [])
        if status.gpu_count == 0:
            status.issues.append("No NVIDIA GPUs detected")
            status.suggestions.append("Ensure nvidia-smi is available and drivers are installed")

        # CUDA
        cuda_out = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if cuda_out:
            status.cuda_version = cuda_out.splitlines()[0] if cuda_out.splitlines() else ""

        # PaddlePaddle — check in paddle conda env
        status.paddle_version = self._get_version_in_paddle_env("paddlepaddle-gpu") or self._get_version_in_paddle_env("paddlepaddle")
        if not status.paddle_version:
            status.issues.append("PaddlePaddle not installed")
            status.suggestions.append("pip install paddlepaddle-gpu")

        # PaddleFormers — check in paddle conda env
        status.paddleformers_version = self._get_version_in_paddle_env("paddleformers")
        if not status.paddleformers_version:
            # Fallback: check local source
            if self.pf_root and os.path.exists(os.path.join(self.pf_root, "setup.py")):
                status.paddleformers_version = "local"

        if not status.paddleformers_version:
            status.issues.append("PaddleFormers not installed")
            if self.pf_root:
                status.suggestions.append(f"cd {self.pf_root} && pip install -e .")
            else:
                status.suggestions.append("pip install paddleformers")

        # Required packages — check in paddle conda env
        for pkg in self.REQUIRED_PACKAGES:
            ver = self._get_version_in_paddle_env(pkg)
            status.packages[pkg] = ver
            if not ver:
                status.issues.append(f"Missing package: {pkg}")

        # Overall readiness
        status.ready = (
            status.gpu_count > 0
            and bool(status.paddle_version)
            and bool(status.paddleformers_version)
            and len(status.issues) == 0
        )

        return status

    def _check_gpus(self) -> dict:
        """Check GPU availability."""
        out = _run_cmd(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        )
        if not out:
            return {"count": 0, "names": [], "memory_gb": []}

        names = []
        memory = []
        for line in out.splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                names.append(parts[0].strip())
                try:
                    mem_mb = float(parts[1].strip().replace("MiB", "").replace("MB", "").strip())
                    memory.append(round(mem_mb / 1024, 1))
                except ValueError:
                    memory.append(0)

        return {"count": len(names), "names": names, "memory_gb": memory}

    def install_package(self, package: str, version: str | None = None):
        """Install a package via pip."""
        cmd = ["pip", "install"]
        if version:
            cmd.append(f"{package}=={version}")
        else:
            cmd.append(package)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            from autotrainer.errors import PackageInstallError

            raise PackageInstallError(f"Failed to install {package}: {result.stderr}")

    def setup_paddleformers(self, editable: bool = True):
        """Install PaddleFormers from local source."""
        if not self.pf_root:
            from autotrainer.errors import EnvironmentError

            raise EnvironmentError("PaddleFormers root not set")

        cmd = ["pip", "install", "-e", "."] if editable else ["pip", "install", "."]
        result = subprocess.run(cmd, cwd=self.pf_root, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            from autotrainer.errors import PackageInstallError

            raise PackageInstallError(f"Failed to install PaddleFormers: {result.stderr}")

    def get_package_diff(self, required: dict[str, str]) -> dict:
        """Compare installed packages against requirements.

        Returns dict with: needs_install, needs_upgrade, up_to_date
        """
        needs_install = {}
        needs_upgrade = {}
        up_to_date = {}

        for pkg, req_version in required.items():
            installed = _get_version(pkg)
            if not installed:
                needs_install[pkg] = req_version
            elif req_version and installed != req_version:
                needs_upgrade[pkg] = {"installed": installed, "required": req_version}
            else:
                up_to_date[pkg] = installed

        return {
            "needs_install": needs_install,
            "needs_upgrade": needs_upgrade,
            "up_to_date": up_to_date,
        }
