"""Global configuration, constants, and path detection.

Configuration is loaded in this priority order (later wins):
1. Defaults
2. Config file: ~/.autotrainer/config.yaml
3. Environment variables: AUTOTRAINER_*
4. CLI arguments (overrides)

Enhanced with pydantic validation (inspired by Claude Code's zod schemas).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config file location
_DEFAULT_CONFIG_PATH = os.path.expanduser("~/.autotrainer/config.yaml")


def _load_config_file(path: str = "") -> dict:
    """Load config from YAML file if it exists."""
    path = path or _DEFAULT_CONFIG_PATH
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError):
        return {}


def init_config_file(path: str = "") -> str:
    """Create a default config file at the given path. Returns the path."""
    path = path or _DEFAULT_CONFIG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    template = """# AutoTrainer Configuration
# Place at ~/.autotrainer/config.yaml

# Path to PaddleFormers repository root (leave empty for auto-detection)
paddleformers_root: ""

# Working directory for experiments
work_dir: "~/.autotrainer"

# LLM API (OpenAI-compatible) — used for skills (diagnose, plan, data-intel)
llm:
  base_url: ""        # e.g., "http://localhost:8000/v1"
  api_key: ""
  model: ""           # e.g., "qwen-72b"

# Data search sources
search:
  tavily_api_key: ""        # https://tavily.com — semantic web search
  modelscope_token: ""      # https://modelscope.cn — 魔搭，optional

# Context window settings
context:
  max_tokens: 128000

# Training defaults
training:
  ablation_subset_ratio: 0.05    # 5% of data for ablation
  ablation_max_steps: 1000       # Steps per ablation experiment
  full_epochs: 3

# Health monitor
health:
  poll_interval: 5.0             # Seconds between health checks
  hang_timeout: 300.0            # Seconds before declaring hang
"""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(template)

    return path


# ── Pydantic validation schema ──────────────────────────────────────────────


def _get_validator():
    """Lazily import/create the pydantic validator to avoid hard dependency at import time."""
    try:
        from pydantic import BaseModel, Field, field_validator

        class LLMConfig(BaseModel):
            base_url: str = ""
            api_key: str = ""
            model: str = ""

            @field_validator("base_url")
            @classmethod
            def validate_base_url(cls, v):
                if v and not v.startswith(("http://", "https://")):
                    raise ValueError(f"llm.base_url must start with http:// or https://, got: {v}")
                return v

        class TrainingConfig(BaseModel):
            ablation_subset_ratio: float = Field(default=0.05, ge=0.001, le=1.0)
            ablation_max_steps: int = Field(default=1000, ge=1)
            full_epochs: int = Field(default=3, ge=1)

        class HealthConfig(BaseModel):
            poll_interval: float = Field(default=5.0, ge=0.5)
            hang_timeout: float = Field(default=300.0, ge=10.0)

        class ContextConfig(BaseModel):
            max_tokens: int = Field(default=128000, ge=1000)

        class AutoTrainerConfigSchema(BaseModel):
            """Pydantic schema for config validation."""
            work_dir: str = ""
            paddleformers_root: str = ""
            skills_dir: str = ""
            llm: LLMConfig = Field(default_factory=LLMConfig)
            training: TrainingConfig = Field(default_factory=TrainingConfig)
            health: HealthConfig = Field(default_factory=HealthConfig)
            context: ContextConfig = Field(default_factory=ContextConfig)

        return AutoTrainerConfigSchema
    except ImportError:
        return None


def validate_config(raw_config: dict) -> list[str]:
    """Validate a raw config dict against the schema. Returns list of errors."""
    Schema = _get_validator()
    if Schema is None:
        # No pydantic, skip validation
        return []
    try:
        Schema(**raw_config)
        return []
    except Exception as e:
        return [str(e)]


# ── Main config class ───────────────────────────────────────────────────────


@dataclass
class AutoTrainerConfig:
    """Central configuration for the autotrainer system."""

    # Paths
    work_dir: str = ""
    paddleformers_root: str = ""
    skills_dir: str = ""

    # LLM API
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Context
    context_max_tokens: int = 128_000

    # Training defaults
    default_ablation_subset_ratio: float = 0.05
    default_ablation_max_steps: int = 1000
    default_full_epochs: int = 3

    # Health monitor
    health_poll_interval: float = 5.0
    health_hang_timeout: float = 300.0

    # Supported tasks
    supported_tasks: list[str] = field(default_factory=lambda: ["paddleocr-vl"])

    @staticmethod
    def detect_paddleformers_root() -> str:
        """Auto-detect PaddleFormers installation."""
        candidates = [
            os.environ.get("PADDLEFORMERS_ROOT"),
            os.path.expanduser("~/.local/lib/python3.10/site-packages/paddleformers"),
        ]
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "PaddleFormers"
            if candidate.is_dir() and (candidate / "setup.py").exists():
                return str(candidate)

        for c in candidates:
            if c and os.path.isdir(c):
                return c

        return ""

    @staticmethod
    def detect_model_path(model_id: str) -> str:
        """Resolve model path: check local cache first, fall back to HF ID."""
        local_path = Path(f"/data-ssd/lizhijun/models/{model_id}")
        if (local_path / "config.json").exists():
            return str(local_path)

        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        for suffix in (model_id, model_id.replace("/", "--")):
            candidate = hf_cache / suffix
            if (candidate / "config.json").exists():
                return str(candidate)

        return model_id

    @staticmethod
    def detect_gpu_count() -> int:
        """Detect number of available NVIDIA GPUs."""
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return len(result.stdout.strip().splitlines())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return 0

    @staticmethod
    def default_work_dir() -> str:
        """Default working directory for experiments."""
        base = os.environ.get("AUTOTRAINER_WORK_DIR", os.path.expanduser("~/.autotrainer"))
        os.makedirs(base, exist_ok=True)
        return base

    @classmethod
    def from_env(cls, config_file: str = "", **overrides) -> "AutoTrainerConfig":
        """Build config from config file + environment variables + overrides.

        Priority: defaults < config file < env vars < overrides
        """
        # Layer 1: defaults
        cfg = {
            "work_dir": cls.default_work_dir(),
            "paddleformers_root": cls.detect_paddleformers_root(),
            "skills_dir": os.path.join(os.path.dirname(__file__), "skills"),
            "llm_base_url": "",
            "llm_api_key": "",
            "llm_model": "",
            "context_max_tokens": 128_000,
            "default_ablation_subset_ratio": 0.05,
            "default_ablation_max_steps": 1000,
            "default_full_epochs": 3,
            "health_poll_interval": 5.0,
            "health_hang_timeout": 300.0,
        }

        # Layer 2: config file
        file_cfg = _load_config_file(config_file)
        if file_cfg:
            raw_work_dir = file_cfg.get("work_dir", cfg["work_dir"])
            cfg["work_dir"] = os.path.expanduser(raw_work_dir)
            cfg["paddleformers_root"] = file_cfg.get("paddleformers_root", cfg["paddleformers_root"])
            cfg["skills_dir"] = file_cfg.get("skills_dir", cfg["skills_dir"])

            llm = file_cfg.get("llm", {})
            cfg["llm_base_url"] = llm.get("base_url", cfg["llm_base_url"])
            cfg["llm_api_key"] = llm.get("api_key", cfg["llm_api_key"])
            cfg["llm_model"] = llm.get("model", cfg["llm_model"])

            ctx = file_cfg.get("context", {})
            cfg["context_max_tokens"] = ctx.get("max_tokens", cfg["context_max_tokens"])

            train = file_cfg.get("training", {})
            cfg["default_ablation_subset_ratio"] = train.get("ablation_subset_ratio", cfg["default_ablation_subset_ratio"])
            cfg["default_ablation_max_steps"] = train.get("ablation_max_steps", cfg["default_ablation_max_steps"])
            cfg["default_full_epochs"] = train.get("full_epochs", cfg["default_full_epochs"])

            health = file_cfg.get("health", {})
            cfg["health_poll_interval"] = health.get("poll_interval", cfg["health_poll_interval"])
            cfg["health_hang_timeout"] = health.get("hang_timeout", cfg["health_hang_timeout"])

            # Validate with pydantic if available
            errors = validate_config(file_cfg)
            if errors:
                for err in errors:
                    logger.warning("Config validation warning: %s", err)

        # Layer 3: environment variables
        env_map = {
            "work_dir": "AUTOTRAINER_WORK_DIR",
            "paddleformers_root": "PADDLEFORMERS_ROOT",
            "llm_base_url": "AUTOTRAINER_LLM_BASE_URL",
            "llm_api_key": "AUTOTRAINER_LLM_API_KEY",
            "llm_model": "AUTOTRAINER_LLM_MODEL",
        }
        for key, env_var in env_map.items():
            val = os.environ.get(env_var)
            if val:
                cfg[key] = val

        # Layer 4: explicit overrides
        cfg.update(overrides)

        return cls(**cfg)
