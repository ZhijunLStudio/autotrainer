"""PaddleFormers YAML config builder.

Generates config YAML files compatible with paddleformers-cli train.
Does NOT import PaddleFormers — all interaction is via subprocess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from autotrainer.utils.file_utils import atomic_write_text, safe_read_text


# Default PaddleOCR-VL task config template
_PADDLEOCR_VL_DEFAULTS = {
    "model": {
        "model_name_or_path": "PaddlePaddle/PaddleOCR-VL",
        "stage": "VL-SFT",
        "attention_implementation": "flash_attention_2",
        "use_lora": False,
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": ".*attn.*",
    },
    "data": {
        "dataset_type": "erniekit",
        "train_dataset_path": "",
        "eval_dataset_path": "",
        "max_seq_len": 8192,
        "template": "qwen2_vl",
        "packing": False,
    },
    "finetuning": {
        "output_dir": "",
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "bf16": True,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 3,
    },
}


class ConfigBuilder:
    """Builds PaddleFormers YAML config files from high-level parameters."""

    def __init__(self, templates_dir: str | None = None):
        self.templates_dir = templates_dir or str(Path(__file__).parent.parent / "configs" / "templates")
        self._templates_cache: dict[str, dict] = {}

    def load_template(self, name: str) -> dict:
        """Load a config template from the templates directory."""
        if name not in self._templates_cache:
            path = Path(self.templates_dir) / f"{name}.yaml"
            content = safe_read_text(str(path))
            if content:
                self._templates_cache[name] = yaml.safe_load(content) or {}
            else:
                self._templates_cache[name] = {}
        return self._templates_cache[name]

    def merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge two config dicts. Override values take precedence."""
        import copy

        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def build_sft_config(
        self,
        model_path: str,
        train_data_path: str,
        eval_data_path: str = "",
        data_type: str = "erniekit",
        stage: str = "VL-SFT",
        template: str = "qwen2_vl",
        output_dir: str = "",
        overrides: dict[str, Any] | None = None,
    ) -> dict:
        """Build a complete SFT config dict."""
        config = self.merge_configs(_PADDLEOCR_VL_DEFAULTS, {})
        config["model"]["model_name_or_path"] = model_path
        config["model"]["stage"] = stage
        config["data"]["dataset_type"] = data_type
        config["data"]["train_dataset_path"] = train_data_path
        if eval_data_path:
            config["data"]["eval_dataset_path"] = eval_data_path
        config["data"]["template"] = template
        if output_dir:
            config["finetuning"]["output_dir"] = output_dir

        if overrides:
            config = self.merge_configs(config, overrides)

        return config

    def build_paddleocr_vl_config(
        self,
        model_path: str,
        train_data: str,
        eval_data: str = "",
        freeze_vision: bool = True,
        freeze_aligner: bool = True,
        lora: bool = False,
        lora_rank: int = 8,
        overrides: dict[str, Any] | None = None,
    ) -> dict:
        """Build a PaddleOCR-VL specific config."""
        config = self.build_sft_config(
            model_path=model_path,
            train_data_path=train_data,
            eval_data_path=eval_data,
            stage="VL-SFT",
        )

        if lora:
            config["model"]["use_lora"] = True
            config["model"]["lora_rank"] = lora_rank
        else:
            config["model"]["use_lora"] = False

        if freeze_vision or freeze_aligner:
            freeze_parts = []
            if freeze_vision:
                freeze_parts.append("vision_model")
            if freeze_aligner:
                freeze_parts.append("aligner")
            config["finetuning"]["freeze_config"] = " ".join(freeze_parts)

        if overrides:
            config = self.merge_configs(config, overrides)

        return config

    def build_ablation_config(
        self,
        base: dict,
        factor_changes: dict[str, Any],
        subset_path: str = "",
        max_steps: int = 1000,
        output_dir: str = "",
    ) -> dict:
        """Build an ablation experiment config from a base + changes."""
        config = self.merge_configs(base, factor_changes)

        if subset_path:
            config["data"]["train_dataset_path"] = subset_path
        if output_dir:
            config["finetuning"]["output_dir"] = output_dir

        # Force shorter training for ablation
        config["finetuning"]["max_steps"] = max_steps
        config["finetuning"]["num_train_epochs"] = 1
        config["finetuning"]["save_steps"] = max_steps  # Only save at end
        config["finetuning"]["eval_steps"] = max(100, max_steps // 5)

        return config

    def to_yaml(self, config: dict, path: str):
        """Write config dict to a YAML file atomically."""
        content = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        atomic_write_text(path, content)

    def from_yaml(self, path: str) -> dict:
        """Read config from a YAML file."""
        content = safe_read_text(path)
        if not content:
            return {}
        return yaml.safe_load(content) or {}

    def config_diff(self, base: dict, new: dict) -> dict:
        """Compute the difference between two configs.

        Returns only the fields that changed in new vs base.
        """
        diff = {}
        for section in new:
            if section not in base:
                diff[section] = new[section]
                continue
            if isinstance(new[section], dict) and isinstance(base[section], dict):
                section_diff = {}
                for key in new[section]:
                    if key not in base[section] or base[section][key] != new[section][key]:
                        section_diff[key] = new[section][key]
                if section_diff:
                    diff[section] = section_diff
            elif base[section] != new[section]:
                diff[section] = new[section]
        return diff
