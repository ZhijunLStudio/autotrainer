"""Data Ratio Ablation skill handler.

Generates experiment specs for finding optimal multi-dataset mixing ratios
through leave-one-out ablation and ratio sweeps.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetInfo:
    """Info about a single dataset for ratio ablation."""

    name: str
    subset_path: str
    sample_count: int
    ds_type: str = "unknown"       # text_rich, chart, table, mixed


@dataclass
class RatioExperiment:
    """A single ratio experiment spec."""

    ratios: dict[str, float]       # {"ds1": 0.5, "ds2": 0.3, "ds3": 0.2}
    label: str                     # "baseline", "no-ds1", "sweep-50-30-20"
    description: str = ""


class DataRatioAblationHandler:
    """Handles data ratio ablation: leave-one-out + ratio sweep + scoring."""

    W_LOSS = 0.7
    W_STABILITY = 0.2
    W_SPEED = 0.1

    def generate_equal_ratios(self, n: int) -> list[float]:
        """Generate equal ratios that sum to 1.0."""
        base = 1.0 / n
        ratios = [base] * n
        ratios[-1] = 1.0 - sum(ratios[:-1])
        return ratios

    def generate_proportional_ratios(self, sample_counts: list[int]) -> list[float]:
        """Generate ratios proportional to sample counts."""
        total = sum(sample_counts)
        if total == 0:
            return self.generate_equal_ratios(len(sample_counts))
        ratios = [c / total for c in sample_counts]
        ratios[-1] = 1.0 - sum(ratios[:-1])
        return ratios

    def generate_leave_one_out(
        self,
        base_ratios: list[float],
        names: list[str],
    ) -> list[dict]:
        """Generate leave-one-out configs.

        For each dataset, remove it and renormalize the remaining ratios.
        Returns list of {"excluded": name, "ratios": [renormalized]}.
        """
        configs = []
        for i, name in enumerate(names):
            remaining = [r for j, r in enumerate(base_ratios) if j != i]
            total_remaining = sum(remaining)
            if total_remaining > 0:
                renormalized = [r / total_remaining for r in remaining]
                renormalized[-1] = 1.0 - sum(renormalized[:-1])
            else:
                renormalized = [1.0 / len(remaining)] * len(remaining)

            # Insert 0.0 at position i
            full_ratios = list(renormalized)
            full_ratios.insert(i, 0.0)

            configs.append({
                "excluded": name,
                "ratios": full_ratios,
            })
        return configs

    def generate_ratio_sweep(self, top_n: int = 2) -> list[list[float]]:
        """Generate ratio combinations to sweep for top N datasets.

        Returns list of ratio lists, each summing to 1.0.
        """
        if top_n == 2:
            combos = [
                [0.5, 0.5],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.8, 0.2],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.9, 0.1],
            ]
        elif top_n == 3:
            combos = [
                [0.33, 0.33, 0.34],
                [0.5, 0.3, 0.2],
                [0.6, 0.2, 0.2],
                [0.4, 0.4, 0.2],
                [0.3, 0.5, 0.2],
                [0.7, 0.15, 0.15],
                [0.4, 0.3, 0.3],
            ]
        else:
            equal = self.generate_equal_ratios(top_n)
            combos = [equal]
            for i in range(top_n):
                combo = [0.1] * top_n
                combo[i] = 1.0 - 0.1 * (top_n - 1)
                if combo[i] > 0:
                    combos.append(combo)

        # Validate all sum to 1.0
        for combo in combos:
            combo[-1] = 1.0 - sum(combo[:-1])

        return combos

    def generate_saturation_configs(self) -> list[float]:
        """Downsampling ratios for saturation curve."""
        return [0.25, 0.50, 0.75, 1.00]

    def compute_score(
        self,
        result: dict,
        baseline: dict,
    ) -> float:
        """Compute weighted composite score. Higher = better."""
        result_loss = result.get("eval_loss", float("inf"))
        baseline_loss = baseline.get("eval_loss", 1.0)
        if result_loss <= 0:
            result_loss = 1e-8
        loss_score = baseline_loss / result_loss

        result_speed = result.get("throughput", 0)
        baseline_speed = baseline.get("throughput", 1)
        speed_score = result_speed / baseline_speed if baseline_speed > 0 else 1.0

        stability_score = 1.0
        if result.get("has_nan", False):
            stability_score *= 0.5

        return (
            self.W_LOSS * loss_score
            + self.W_STABILITY * stability_score
            + self.W_SPEED * speed_score
        )

    def rank_datasets_by_importance(
        self,
        baseline_score: float,
        loo_scores: dict[str, float],
    ) -> list[tuple[str, float, str]]:
        """Rank datasets by importance based on leave-one-out results.

        Returns list of (name, delta, verdict) sorted by importance.
        """
        rankings = []
        for name, loo_score in loo_scores.items():
            delta = baseline_score - loo_score
            if delta > 0.05:
                verdict = "important"
            elif delta > -0.02:
                verdict = "neutral"
            else:
                verdict = "noise"
            rankings.append((name, delta, verdict))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def build_experiment_specs(
        self,
        datasets: list[DatasetInfo],
        base_config: dict,
        config_builder,
        subset_dir: str,
        ablation_max_steps: int = 1000,
    ) -> list:
        """Build the full list of experiment specs for ratio ablation.

        Returns list of ExperimentSpec dicts ready for scheduler.
        """
        from autotrainer.orchestrator.scheduler import ExperimentSpec

        specs = []
        names = [d.name for d in datasets]
        paths = [d.subset_path for d in datasets]

        # B1: Baseline (equal ratios)
        equal_ratios = self.generate_equal_ratios(len(datasets))
        baseline_config = config_builder.build_multi_dataset_config(
            model_path=base_config["model"]["model_name_or_path"],
            datasets=list(zip(paths, equal_ratios)),
            overrides={
                "finetuning": {
                    "learning_rate": base_config["finetuning"].get("learning_rate", 1e-4),
                    "per_device_train_batch_size": base_config["finetuning"].get("per_device_train_batch_size", 2),
                    "max_steps": ablation_max_steps,
                    "num_train_epochs": 1,
                    "do_eval": True,
                },
                "model": {
                    "use_lora": base_config["model"].get("use_lora", True),
                    "lora_rank": base_config["model"].get("lora_rank", 8),
                },
            },
        )
        specs.append(ExperimentSpec(
            id="ratio-b1-baseline",
            phase="ablation_ratio",
            config=baseline_config,
            config_diff={"ratios": dict(zip(names, equal_ratios))},
        ))

        # B2: Leave-one-out
        loo_configs = self.generate_leave_one_out(equal_ratios, names)
        for loo in loo_configs:
            excluded = loo["excluded"]
            remaining_names = [n for n in names if n != excluded]
            remaining_paths = [p for n, p in zip(names, paths) if n != excluded]
            remaining_ratios = [r for r in loo["ratios"] if r > 0]

            config = config_builder.build_multi_dataset_config(
                model_path=base_config["model"]["model_name_or_path"],
                datasets=list(zip(remaining_paths, remaining_ratios)),
                overrides={
                    "finetuning": {
                        "learning_rate": base_config["finetuning"].get("learning_rate", 1e-4),
                        "per_device_train_batch_size": base_config["finetuning"].get("per_device_train_batch_size", 2),
                        "max_steps": ablation_max_steps,
                        "num_train_epochs": 1,
                        "do_eval": True,
                    },
                    "model": {
                        "use_lora": base_config["model"].get("use_lora", True),
                        "lora_rank": base_config["model"].get("lora_rank", 8),
                    },
                },
            )
            ratio_dict = dict(zip(remaining_names, remaining_ratios))
            specs.append(ExperimentSpec(
                id=f"ratio-b2-no-{excluded}",
                phase="ablation_ratio",
                config=config,
                config_diff={"excluded": excluded, "ratios": ratio_dict},
            ))

        # B3: Ratio sweep (top 2 and top 3)
        for top_n in [2, 3]:
            if top_n > len(datasets):
                continue
            sweep_combos = self.generate_ratio_sweep(top_n)
            for i, combo in enumerate(sweep_combos):
                top_names = names[:top_n]
                top_paths = paths[:top_n]
                config = config_builder.build_multi_dataset_config(
                    model_path=base_config["model"]["model_name_or_path"],
                    datasets=list(zip(top_paths, combo)),
                    overrides={
                        "finetuning": {
                            "learning_rate": base_config["finetuning"].get("learning_rate", 1e-4),
                            "per_device_train_batch_size": base_config["finetuning"].get("per_device_train_batch_size", 2),
                            "max_steps": ablation_max_steps,
                            "num_train_epochs": 1,
                            "do_eval": True,
                        },
                        "model": {
                            "use_lora": base_config["model"].get("use_lora", True),
                            "lora_rank": base_config["model"].get("lora_rank", 8),
                        },
                    },
                )
                ratio_str = "-".join(f"{int(r*100)}" for r in combo)
                specs.append(ExperimentSpec(
                    id=f"ratio-b3-top{top_n}-{ratio_str}",
                    phase="ablation_ratio",
                    config=config,
                    config_diff={"ratios": dict(zip(top_names, combo))},
                ))

        return specs
