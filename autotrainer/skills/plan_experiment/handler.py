"""Plan Experiment skill handler — ablation planning and parameter tuning.

Provides both legacy rule-based planning and the new intelligent
iterative ablation agent with trend analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import itertools


# ─── New dataclasses for intelligent ablation ────────────────────────


@dataclass
class AblationFactorConfig:
    """Describes how to search over one hyperparameter."""

    dotted_key: str  # e.g. "finetuning.learning_rate"
    initial_values: list[float | int]  # first round values
    scale: str = "linear"  # "linear" | "log"
    discrete: bool = False  # True for int-only values (lora_rank, batch_size)
    min_value: float = 0.0
    max_value: float = float("inf")
    max_rounds: int = 3  # max refinement rounds (initial + 2)


@dataclass
class ExperimentRecord:
    """One ablation experiment and its result."""

    id: str
    factor: str  # dotted_key of the factor being varied
    value: float | int  # the value tested
    config_diff: dict  # {dotted_key: value}
    status: str  # "pending" | "completed" | "failed"
    final_loss: float | None = None
    eval_loss: float | None = None
    log_path: str = ""
    created_at: str = ""
    completed_at: str = ""


@dataclass
class FactorAnalysisResult:
    """Result of analyzing experiments for a single factor."""

    action: str  # "continue" | "done" | "expand"
    best_value: float | int  # best value found so far
    next_values: list[float | int] = field(default_factory=list)
    analysis: str = ""  # human-readable explanation
    round_number: int = 0
    best_loss: float = float("inf")


# ─── Legacy handler (kept for backward compat) ──────────────────────


class PlanExperimentHandler:
    """Handles experiment planning with intelligent iterative ablation."""

    # ── Intelligent ablation: analyze trend and suggest next values ──

    def analyze_and_suggest(
        self,
        factor_config: AblationFactorConfig,
        history: list[ExperimentRecord],
        round_number: int,
    ) -> FactorAnalysisResult:
        """Analyze loss trend for a factor and suggest next experiment values.

        Core algorithm:
        - Compute Pearson correlation between (transformed) param values and losses
        - |corr| < 0.3  → plateau → done
        - corr > 0.7     → loss increasing with param → done (pick best)
        - corr < -0.7    → loss decreasing → expand upward
        - otherwise       → curved/narrow → continue around minimum
        - Also checks convergence and round cap.
        """
        completed = [h for h in history if h.status == "completed" and h.final_loss is not None]
        valid = [h for h in completed if not (isinstance(h.final_loss, float) and (math.isnan(h.final_loss) or math.isinf(h.final_loss)))]

        # Edge: all failed or no valid data
        if not valid:
            default_val = factor_config.initial_values[len(factor_config.initial_values) // 2]
            return FactorAnalysisResult(
                action="done",
                best_value=default_val,
                analysis="All experiments failed or had NaN loss. Using default value.",
                round_number=round_number,
                best_loss=float("inf"),
            )

        # Edge: only 1 valid experiment — can't compute trend
        if len(valid) < 2:
            best = min(valid, key=lambda h: h.final_loss)
            return FactorAnalysisResult(
                action="done",
                best_value=best.value,
                analysis=f"Only 1 valid experiment. Best: {best.value} (loss={best.final_loss:.4f})",
                round_number=round_number,
                best_loss=best.final_loss,
            )

        # Build param_values and losses
        param_values = [h.value for h in valid]
        losses = [h.final_loss for h in valid]
        best_idx = losses.index(min(losses))
        best_value = param_values[best_idx]
        best_loss = losses[best_idx]

        # Round cap
        if round_number >= factor_config.max_rounds - 1:
            return FactorAnalysisResult(
                action="done",
                best_value=best_value,
                analysis=f"Max rounds ({factor_config.max_rounds}) reached. Best: {best_value} (loss={best_loss:.4f})",
                round_number=round_number,
                best_loss=best_loss,
            )

        # Convergence check: compare best loss across rounds
        # Only meaningful if we have completed at least 1 prior round
        if round_number >= 1 and len(valid) >= 4:
            # Split into rounds: first N/2 are prior rounds, rest are current
            # Simple heuristic: compare best loss of first half vs second half
            mid = len(valid) // 2
            first_half_best = min(h.final_loss for h in valid[:mid])
            second_half_best = min(h.final_loss for h in valid[mid:])
            if first_half_best > 0 and (first_half_best - second_half_best) / first_half_best < 0.05:
                return FactorAnalysisResult(
                    action="done",
                    best_value=best_value,
                    analysis=f"Converged: <5% improvement between rounds. Best: {best_value} (loss={best_loss:.4f})",
                    round_number=round_number,
                    best_loss=best_loss,
                )

        # Compute Pearson correlation
        corr = self._pearson_correlation(param_values, losses, factor_config.scale)

        # Trend interpretation
        if abs(corr) < 0.3:
            # Plateau
            return FactorAnalysisResult(
                action="done",
                best_value=best_value,
                analysis=f"Plateau detected (corr={corr:.2f}). Best: {best_value} (loss={best_loss:.4f})",
                round_number=round_number,
                best_loss=best_loss,
            )
        elif corr > 0.7:
            # Loss increasing with param — best is at the low end
            return FactorAnalysisResult(
                action="done",
                best_value=best_value,
                analysis=f"Loss increasing with param (corr={corr:.2f}). Best: {best_value} (loss={best_loss:.4f})",
                round_number=round_number,
                best_loss=best_loss,
            )
        elif corr < -0.7:
            # Loss decreasing with param — expand upward
            next_vals = self._generate_expanded_values(
                direction="up",
                current_range=(min(param_values), max(param_values)),
                scale=factor_config.scale,
                discrete=factor_config.discrete,
                min_value=factor_config.min_value,
                max_value=factor_config.max_value,
            )
            if not next_vals:
                return FactorAnalysisResult(
                    action="done",
                    best_value=best_value,
                    analysis=f"Loss decreasing but hit max boundary. Best: {best_value} (loss={best_loss:.4f})",
                    round_number=round_number,
                    best_loss=best_loss,
                )
            return FactorAnalysisResult(
                action="expand",
                best_value=best_value,
                next_values=next_vals,
                analysis=f"Loss decreasing (corr={corr:.2f}). Expanding range: {next_vals}",
                round_number=round_number,
                best_loss=best_loss,
            )
        else:
            # Curved / concave — narrow around best
            next_vals = self._generate_narrowed_values(
                best_value=best_value,
                param_values=sorted(param_values),
                scale=factor_config.scale,
                discrete=factor_config.discrete,
                min_value=factor_config.min_value,
                max_value=factor_config.max_value,
            )
            if not next_vals:
                return FactorAnalysisResult(
                    action="done",
                    best_value=best_value,
                    analysis=f"Cannot narrow further. Best: {best_value} (loss={best_loss:.4f})",
                    round_number=round_number,
                    best_loss=best_loss,
                )
            return FactorAnalysisResult(
                action="continue",
                best_value=best_value,
                next_values=next_vals,
                analysis=f"Curved trend (corr={corr:.2f}). Narrowing around {best_value}: {next_vals}",
                round_number=round_number,
                best_loss=best_loss,
            )

    # ── Value generation helpers ──

    @staticmethod
    def _generate_narrowed_values(
        best_value: float | int,
        param_values: list[float],
        scale: str,
        discrete: bool,
        min_value: float,
        max_value: float,
    ) -> list[float | int]:
        """Generate 3 values narrowed around the best point."""
        # Find neighbors of best_value in sorted list
        best_pos = param_values.index(best_value) if best_value in param_values else len(param_values) // 2
        left = param_values[best_pos - 1] if best_pos > 0 else min_value
        right = param_values[best_pos + 1] if best_pos < len(param_values) - 1 else max_value

        if scale == "log":
            if left <= 0:
                left = min_value
            log_left = math.log10(max(left, 1e-12))
            log_right = math.log10(max(right, 1e-12))
            log_best = math.log10(max(best_value, 1e-12))
            # Narrow: half the gap on each side
            log_mid_l = (log_left + log_best) / 2
            log_mid_r = (log_best + log_right) / 2
            vals = [10**log_mid_l, 10**log_best, 10**log_mid_r]
        else:
            mid_l = (left + best_value) / 2
            mid_r = (best_value + right) / 2
            vals = [mid_l, best_value, mid_r]

        # Clamp and deduplicate
        result = []
        seen = set()
        for v in vals:
            v = max(min_value, min(max_value, v))
            if discrete:
                v = int(round(v))
            v_key = round(v, 10)
            if v_key not in seen:
                seen.add(v_key)
                result.append(v)

        return sorted(result)

    @staticmethod
    def _generate_expanded_values(
        direction: str,
        current_range: tuple[float, float],
        scale: str,
        discrete: bool,
        min_value: float,
        max_value: float,
    ) -> list[float | int]:
        """Generate 3 values extending beyond the current range."""
        lo, hi = current_range

        if direction == "up":
            if scale == "log":
                log_hi = math.log10(max(hi, 1e-12))
                log_ext = log_hi + 0.5  # extend by ~3x
                new_hi = min(10**log_ext, max_value)
            else:
                span = hi - lo
                new_hi = min(hi + span, max_value)
            if new_hi <= hi:
                return []  # already at boundary
            if scale == "log":
                log_lo = math.log10(max(hi, 1e-12))
                log_hi2 = math.log10(max(new_hi, 1e-12))
                vals = [10 ** (log_lo + (log_hi2 - log_lo) * f) for f in [0.25, 0.5, 0.75]]
            else:
                vals = [hi + (new_hi - hi) * f for f in [0.25, 0.5, 0.75]]
        else:  # down
            if scale == "log":
                log_lo = math.log10(max(lo, 1e-12))
                log_ext = log_lo - 0.5
                new_lo = max(10**log_ext, min_value)
            else:
                span = hi - lo
                new_lo = max(lo - span, min_value)
            if new_lo >= lo:
                return []
            if scale == "log":
                log_lo2 = math.log10(max(new_lo, 1e-12))
                log_hi = math.log10(max(lo, 1e-12))
                vals = [10 ** (log_lo2 + (log_hi - log_lo2) * f) for f in [0.25, 0.5, 0.75]]
            else:
                vals = [new_lo + (lo - new_lo) * f for f in [0.25, 0.5, 0.75]]

        result = []
        seen = set()
        for v in vals:
            v = max(min_value, min(max_value, v))
            if discrete:
                v = int(round(v))
            v_key = round(v, 10)
            if v_key not in seen:
                seen.add(v_key)
                result.append(v)

        return sorted(result)

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float], scale: str = "linear") -> float:
        """Compute Pearson correlation. For log scale, transform x via log10."""
        n = len(x)
        if n < 2:
            return 0.0

        if scale == "log":
            x = [math.log10(max(v, 1e-12)) for v in x]

        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        if std_x < 1e-12 or std_y < 1e-12:
            return 0.0

        return cov / (std_x * std_y)

    # ── Legacy methods (backward compat) ──

    def plan_ablation(
        self,
        base_config: dict,
        factors: dict[str, list],
        subset_ratio: float = 0.05,
        max_steps: int = 1000,
    ) -> dict:
        """Plan a set of ablation experiments."""
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
        """Based on experiment history, determine the next experiment to run."""
        if not history:
            return {
                "analysis": "No history yet. Start with baseline.",
                "next_experiment": {
                    "name": "baseline",
                    "config_changes": {},
                    "reason": "Establish a baseline with default config",
                },
            }

        completed = [h for h in history if h.get("status") == "completed"]
        if not completed:
            return {
                "analysis": "No completed experiments yet.",
                "next_experiment": None,
                "reason": "Wait for current experiment to complete",
            }

        def _best_loss(h):
            r = h.get("result", {})
            return r.get("eval_loss") or r.get("final_loss") or float("inf")

        best = min(completed, key=_best_loss)

        tested_factors = set()
        for h in history:
            diff = h.get("config_diff", {})
            for key in diff.keys():
                tested_factors.add(key.split(".")[-1])

        all_factors = ["learning_rate", "lora_rank", "per_device_train_batch_size", "gradient_accumulation_steps"]
        untested = [f for f in all_factors if f not in tested_factors]

        if untested:
            return {
                "analysis": f"Best so far: {best.get('id')} with loss={_best_loss(best)}",
                "next_experiment": {
                    "name": f"test-{untested[0]}",
                    "base_from": best.get("id"),
                    "config_changes": {untested[0]: self._suggest_values(untested[0])},
                    "reason": f"Untested factor: {untested[0]}",
                },
            }

        return {
            "analysis": f"All single factors tested. Best: {best.get('id')} with loss={_best_loss(best)}",
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
        """Generate all combinations of factor values (full grid)."""
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
        """Rank experiments by performance (prefers eval_loss, fallback to final_loss)."""
        scored = []
        for r in results:
            result_data = r.get("result", {})
            eval_loss = result_data.get("eval_loss")
            final_loss = result_data.get("final_loss")
            best_loss = eval_loss if eval_loss is not None else final_loss
            if best_loss is not None:
                scored.append({"id": r.get("id"), "eval_loss": best_loss, "final_loss": final_loss, "config_diff": r.get("config_diff", {})})
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
        best_loss_per_factor: dict[str, float] = {}
        for h in history:
            if h.get("status") != "completed":
                continue
            result = h.get("result", {})
            loss = result.get("eval_loss") or result.get("final_loss") or float("inf")
            for factor, value in h.get("config_diff", {}).items():
                if factor not in best_loss_per_factor or loss < best_loss_per_factor[factor]:
                    best_per_factor[factor] = value
                    best_loss_per_factor[factor] = loss
        return best_per_factor
