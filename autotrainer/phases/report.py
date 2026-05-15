"""Phase 6: Report — generate final report with visualizations and summary."""

from __future__ import annotations

import os
from datetime import datetime

from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler
from autotrainer.core.store import PipelineStore


class ReportHandler(PhaseHandler):
    """Generate final training report with charts and summary."""

    def __init__(self, store: PipelineStore | None = None):
        self._store = store

    def execute(self, ctx: PipelineContext) -> PhaseResult:
        def notify(msg: str):
            ctx.notify("REPORT", msg)

        report_dir = os.path.join(ctx.work_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)

        experiments = []
        for abl in ctx.ablation_results:
            experiments.append(abl)
        if ctx.full_training_result:
            experiments.append(ctx.full_training_result)

        # Text summary
        report_lines = [
            "=" * 60,
            f"  AutoTrainer Report — {ctx.task}",
            f"  Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            f"Total experiments: {len(experiments)}",
        ]

        completed = [e for e in experiments if e.get("status") == "completed"]
        failed = [e for e in experiments if e.get("status") == "failed"]
        report_lines.append(f"Completed: {len(completed)}")
        report_lines.append(f"Failed: {len(failed)}")

        if completed:
            best = min(completed, key=lambda x: x.get("result", {}).get("eval_loss", float("inf")))
            report_lines.extend([
                "",
                f"Best experiment: {best.get('id')}",
                f"  Eval loss: {best.get('result', {}).get('eval_loss', 'N/A')}",
                f"  Config: {best.get('config_diff', {})}",
            ])

        report_text = "\n".join(report_lines)
        report_path = os.path.join(report_dir, "final_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        notify(f"Report saved to {report_path}")

        # Try full OCR report with charts
        try:
            from autotrainer.visualization.report_generator import generate_full_report

            training_histories = {}
            eval_dir = os.path.join(ctx.work_dir, "eval_results")
            if os.path.isdir(eval_dir):
                for fname in os.listdir(eval_dir):
                    if fname.endswith(".json"):
                        from autotrainer.utils.file_utils import safe_read_json
                        data = safe_read_json(os.path.join(eval_dir, fname)) or {}
                        exp_id = data.get("experiment_id", fname.replace(".json", ""))
                        if "history" in data:
                            training_histories[exp_id] = data["history"]

            full_report_path = generate_full_report(experiments, report_dir, training_histories)
            notify(f"Full OCR report generated: {full_report_path}")
        except Exception as e:
            notify(f"Full report generation failed ({e}), falling back to basic charts")
            self._generate_basic_charts(experiments, report_dir)

        return PhaseResult(Phase.REPORT, PhaseStatus.COMPLETED, f"Report saved to {report_path}",
                           data={"report_path": report_path})

    def _generate_basic_charts(self, experiments: list[dict], report_dir: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        completed = [e for e in experiments if e.get("status") == "completed" and e.get("result", {}).get("eval_loss")]
        if not completed:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        names = [e.get("id", "?")[:20] for e in completed]
        eval_losses = [e["result"]["eval_loss"] for e in completed]
        colors = ["green" if l == min(eval_losses) else "steelblue" for l in eval_losses]
        ax.barh(names, eval_losses, color=colors)
        ax.set_xlabel("Eval Loss")
        ax.set_title("Experiment Comparison")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(report_dir, "eval_loss_comparison.png"), dpi=100)
        plt.close(fig)
