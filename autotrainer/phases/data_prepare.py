"""Phase 1: Data Prepare — validate, profile, download, merge datasets."""

from __future__ import annotations

import json
import os

from autotrainer.core.interfaces import Phase, PhaseResult, PhaseStatus, PipelineContext, PhaseHandler
from autotrainer.managers.data_manager import DataManager
from autotrainer.managers.data_pipeline import DataPipeline


class DataPrepareHandler(PhaseHandler):
    """Validate, profile, and prepare training data.

    Two code paths:
      A) data_dir contains data_index.json → merge all completed datasets
      B) data_path is explicit JSONL/directory → validate + profile + split
    """

    def __init__(self):
        self._data_pipeline: DataPipeline | None = None
        self._data_mgr: DataManager | None = None

    def execute(self, ctx: PipelineContext) -> PhaseResult:
        data_dir = os.path.join(ctx.work_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        self._data_pipeline = DataPipeline(cache_dir=data_dir)
        self._data_mgr = DataManager(cache_dir=data_dir)

        ablation_data_dir = data_dir
        index_path = os.path.join(ctx.data_dir, "data_index.json") if ctx.data_dir else ""

        def notify(msg: str):
            ctx.notify("DATA_PREPARE", msg)

        # ── Path A: data_dir with data_index.json ──
        if index_path and os.path.exists(index_path):
            notify(f"Reading DataAgent output from {ctx.data_dir} ...")
            try:
                merge_result = self._data_pipeline.merge_from_index(
                    data_dir=ctx.data_dir,
                    output_dir=data_dir,
                )
            except (FileNotFoundError, RuntimeError) as e:
                return PhaseResult(Phase.DATA_PREPARE, PhaseStatus.FAILED, str(e))

            notify(
                f"Merged {len(merge_result['datasets'])} datasets: "
                f"train={merge_result['total_train']} rows, val={merge_result['total_val']} rows"
            )
            for ds in merge_result["datasets"]:
                notify(f"  {ds['name']}: train={ds['train_count']} val={ds['val_count']}")

            ctx.data_path = merge_result["train"]["path"]
            ctx.eval_data_path = merge_result["val"]["path"] if merge_result["val"]["count"] > 0 else ""

            # Profile merged train
            profile = self._data_mgr.profile_dataset(ctx.data_path)
            ctx.data_profile = profile.to_dict()

            # Multi-dataset subsets for ratio ablation
            self._create_per_dataset_subsets(ctx, ablation_data_dir, index_path)

        # ── Path B: explicit JSONL / directory ──
        else:
            if not ctx.data_path or not os.path.exists(ctx.data_path):
                return PhaseResult(
                    Phase.DATA_PREPARE,
                    PhaseStatus.FAILED,
                    "No training data found. Run `autotrainer data --path <dataset_dir>` first.",
                )

            # Validate
            validation = self._data_mgr.validate_dataset(ctx.data_path)
            if not validation["valid"]:
                error_msg = "\n".join(validation["errors"][:5])
                if not ctx.confirm(f"Data validation failed:\n{error_msg}\nContinue anyway?"):
                    return PhaseResult(Phase.DATA_PREPARE, PhaseStatus.FAILED, "Data validation failed, user aborted.")

            # Profile
            profile = self._data_mgr.profile_dataset(ctx.data_path)
            ctx.data_profile = profile.to_dict()

            # Split if no eval data
            if not ctx.eval_data_path:
                split_result = self._data_mgr.split_dataset(ctx.data_path, train_ratio=0.9, val_ratio=0.05)
                ctx.data_path = split_result["train"]["path"]
                ctx.eval_data_path = split_result["val"]["path"]
                notify(
                    f"Split: train={split_result['train']['count']}, "
                    f"val={split_result['val']['count']}, test={split_result['test']['count']}"
                )

        # ── Ablation subset (5%) ──
        ablation_subset_path = os.path.join(ablation_data_dir, "subset_5pct.jsonl")
        subset_info = self._data_mgr.create_subset(ctx.data_path, ablation_subset_path, ratio=0.05)
        ctx.ablation_config = {"subset_path": ablation_subset_path, "subset_info": subset_info}

        notify(
            f"Data ready: {ctx.data_profile.get('num_samples', '?')} samples, "
            f"format={ctx.data_profile.get('format', '?')}"
        )
        return PhaseResult(Phase.DATA_PREPARE, PhaseStatus.COMPLETED, "Data preparation complete.")

    def _create_per_dataset_subsets(self, ctx: PipelineContext, ablation_data_dir: str, index_path: str):
        """Create per-dataset 5% subsets for ratio ablation (multi-dataset only)."""
        with open(index_path, "r") as f:
            index_data = json.load(f)
        completed_datasets = [d for d in index_data.get("datasets", []) if d.get("status") == "completed"]
        if len(completed_datasets) <= 1:
            return

        if self._data_mgr is None:
            return

        ctx.multi_dataset_info = []
        for ds in completed_datasets:
            ds_name = ds.get("dataset_name", "unknown")
            train_path = ds.get("split", {}).get("train", {}).get("path", "")
            if train_path and os.path.exists(train_path):
                subset_path = os.path.join(ablation_data_dir, f"subset_5pct_{ds_name}.jsonl")
                info = self._data_mgr.create_subset(train_path, subset_path, ratio=0.05)
                ctx.multi_dataset_info.append({
                    "name": ds_name,
                    "subset_path": subset_path,
                    "sample_count": info.get("subset", 0),
                    "total_count": info.get("total", 0),
                })
