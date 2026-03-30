"""Plan Experiment skill handler — ablation planning and parameter tuning."""

from __future__ import annotations

import itertools
from typing import Any


class PlanExperimentHandler:
    """Handles experiment planning with ablation strategies."""

    def plan_ablation(
        self,
        base_config: dict,
        factors: dict[str, list],
        subset_ratio: float = 0.05,
        max_steps: int = 1000,
    ) -> dict:
        """Plan a set of ablation experiments.

        Generates single-factor experiments (one factor varied at a time).
        """
        experiments = []
        exp_num = 0

        for factor_name, values in factors.items():
            for value in values:
                exp_num += 1
                experiments.append(
                    {
                        "id": f"abl-{exp_num:03d}",
                        "factor": factor_name,
                        "value": value,
                        "config_changes": {factor_name: value},
                        "subset_ratio": subset_ratio,
                        "max_steps": max_steps,
                    }
                )

        return {
            "experiments": experiments,
            "total_runs": len(experiments),
            "subset_ratio": subset_ratio,
            "max_steps": max_steps,
        }

    def plan_next_experiment(
        self,
        history: list[dict],
        goal: str = "",
        current_best: dict | None = None,
    ) -> dict:
        """Based on experiment history, determine the next experiment to run.

        Analyzes trends and suggests the next step.
        """
        if not history:
            return {
                "analysis": "No history yet. Start with baseline.",
                "next_experiment": {
                    "name": "baseline",
                    "config_changes": {},
                    "reason": "Establish a baseline with default config",
                },
            }

        # Find the best experiment so far
        completed = [h for h in history if h.get("status") == "completed"]
        if not completed:
            return {
                "analysis": "No completed experiments yet.",
                "next_experiment": None,
                "reason": "Wait for current experiment to complete",
            }

        # Sort by eval loss (lower is better)
        best = min(
            completed,
            key=lambda x: x.get("result", {}).get("eval_loss", float("inf")),
        )

        # Analyze what has been tested
        tested_factors = set()
        for h in history:
            diff = h.get("config_diff", {})
            tested_factors.update(diff.keys())

        # Suggest untested factors
        all_factors = ["learning_rate", "lora_rank", "per_device_train_batch_size", "gradient_accumulation_steps"]
        untested = [f for f in all_factors if f not in tested_factors]

        if untested:
            return {
                "analysis": f"Best so far: {best.get('id')} with eval_loss={best.get('result', {}).get('eval_loss', 'N/A')}",
                "next_experiment": {
                    "name": f"test-{untested[0]}",
                    "base_from": best.get("id"),
                    "config_changes": {untested[0]: self._suggest_values(untested[0])},
                    "reason": f"Untested factor: {untested[0]}",
                },
            }

        # All factors tested — try multi-factor
        return {
            "analysis": f"All single factors tested. Best: {best.get('id')}",
            "next_experiment": {
                "name": "multi-factor-combo",
                "base_from": best.get("id"),
                "config_changes": self._best_combo_from_history(history),
                "reason": "Multi-factor combination based on best results",
            },
        }

    def generate_factor_grid(
        self,
        base_config: dict,
        factors: dict[str, list],
    ) -> list[dict]:
        """Generate all combinations of factor values (full grid).

        Use sparingly — single-factor ablation is usually more efficient.
        """
        keys = list(factors.keys())
        values = list(factors.values())

        configs = []
        for combo in itertools.product(*values):
            config = {}
            for key, value in zip(keys, combo):
                config[key] = value
            configs.append(config)

        return configs

    def rank_experiments(self, results: list[dict]) -> list[dict]:
        """Rank experiments by performance."""
        scored = []
        for r in results:
            result_data = r.get("result", {})
            eval_loss = result_data.get("eval_loss", float("inf"))
            if eval_loss is not None:
                scored.append({"id": r.get("id"), "eval_loss": eval_loss, "config_diff": r.get("config_diff", {})})

        scored.sort(key=lambda x: x["eval_loss"])
        return scored

    def _suggest_values(self, factor: str) -> list:
        """Suggest default values to test for a factor."""
        defaults = {
            "learning_rate": [1e-5, 3e-5, 1e-4],
            "lora_rank": [4, 8, 16],
            "per_device_train_batch_size": [1, 2, 4],
            "gradient_accumulation_steps": [2, 4, 8],
        }
        return defaults.get(factor, [1, 2, 4])

    def _best_combo_from_history(self, history: list[dict]) -> dict:
        """Extract the best value for each factor from history."""
        best_per_factor: dict[str, Any] = {}

        for h in history:
            if h.get("status") != "completed":
                continue
            result = h.get("result", {})
            eval_loss = result.get("eval_loss", float("inf"))
            for factor, value in h.get("config_diff", {}).items():
                if factor not in best_per_factor:
                    best_per_factor[factor] = value
                # Keep the one with lower eval_loss
                # (simplified — real logic would compare per-factor)
        return best_per_factor
