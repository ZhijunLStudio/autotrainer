"""Data manager — dataset search, download, validation, profiling, and conversion."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetProfile:
    """Statistical profile of a dataset."""

    path: str = ""
    format: str = ""
    num_samples: int = 0
    has_images: bool = False
    image_count: int = 0
    text_lengths: dict = field(default_factory=dict)  # min, max, avg, p95, p50
    field_coverage: dict = field(default_factory=dict)
    sample_preview: list[dict] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    total_size_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "num_samples": self.num_samples,
            "has_images": self.has_images,
            "image_count": self.image_count,
            "text_lengths": self.text_lengths,
            "field_coverage": self.field_coverage,
            "sample_preview": self.sample_preview,
            "quality_flags": self.quality_flags,
            "total_size_mb": round(self.total_size_mb, 2),
        }


class DataManager:
    """Dataset search, download, validation, and conversion."""

    def __init__(self, cache_dir: str, paddleformers_root: str = ""):
        self.cache_dir = cache_dir
        self.pf_root = paddleformers_root
        os.makedirs(cache_dir, exist_ok=True)

    def validate_dataset(self, path: str) -> dict:
        """Validate a dataset file. Returns validation result."""
        from autotrainer.pf_integration.dataset_validator import DatasetValidator

        validator = DatasetValidator()
        result = validator.validate_file(path)
        return {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "format": result.format_detected,
            "samples_checked": result.samples_checked,
        }

    def profile_dataset(self, path: str) -> DatasetProfile:
        """Generate a statistical profile of a dataset."""
        from autotrainer.pf_integration.dataset_validator import DatasetValidator

        p = Path(path)
        profile = DatasetProfile(
            path=path,
            total_size_mb=p.stat().st_size / (1024 * 1024) if p.exists() else 0,
        )

        if not p.exists():
            return profile

        validator = DatasetValidator()
        result = validator.validate_file(path)
        profile.format = result.format_detected

        # Scan all lines for statistics
        text_lens = []
        field_counts: dict[str, int] = {}
        image_count = 0
        empty_count = 0
        line_count = 0

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    empty_count += 1
                    continue

                line_count += 1

                # Count fields
                for key in data:
                    field_counts[key] = field_counts.get(key, 0) + 1

                # Text lengths
                if profile.format == "erniekit":
                    text_info = data.get("text_info", [])
                    if isinstance(text_info, list):
                        total_len = sum(len(str(t.get("text", ""))) for t in text_info if isinstance(t, dict))
                        text_lens.append(total_len)
                    images = data.get("image_info", [])
                    if isinstance(images, list):
                        image_count += len(images)
                        if images:
                            profile.has_images = True
                elif profile.format == "messages":
                    messages = data.get("messages", [])
                    if isinstance(messages, list):
                        total_len = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
                        text_lens.append(total_len)
                    images = data.get("images", [])
                    if images:
                        profile.has_images = True
                        image_count += len(images) if isinstance(images, list) else 1

        profile.num_samples = line_count
        profile.image_count = image_count

        # Field coverage
        if line_count > 0:
            profile.field_coverage = {k: round(v / line_count, 2) for k, v in field_counts.items()}

        # Text statistics
        if text_lens:
            sorted_lens = sorted(text_lens)
            n = len(sorted_lens)
            profile.text_lengths = {
                "min": sorted_lens[0],
                "max": sorted_lens[-1],
                "avg": round(sum(sorted_lens) / n),
                "p50": sorted_lens[n // 2],
                "p95": sorted_lens[int(n * 0.95)] if n > 20 else sorted_lens[-1],
            }

        # Quality flags
        if empty_count > 0:
            pct = round(empty_count / (line_count + empty_count) * 100, 1)
            profile.quality_flags.append(f"{pct}% lines have JSON errors")

        # Sample preview
        samples = validator.sample_and_preview(path, n=3)
        profile.sample_preview = samples

        return profile

    def create_subset(self, src: str, dst: str, ratio: float, seed: int = 42):
        """Create a random subset of a dataset."""
        random.seed(seed)

        lines = []
        with open(src, "r") as f:
            lines = f.readlines()

        subset_size = max(1, int(len(lines) * ratio))
        sampled = random.sample(lines, subset_size)

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w") as f:
            f.writelines(sampled)

        return {"total": len(lines), "subset": len(sampled), "ratio": ratio, "path": dst}

    def search_hf(self, query: str, task: str = "", limit: int = 10) -> list[dict]:
        """Search HuggingFace Hub for datasets."""
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            filters = {}
            if task:
                filters["task_categories"] = task

            datasets = api.list_datasets(search=query, limit=limit, **filters)
            return [
                {
                    "id": d.id,
                    "downloads": d.downloads or 0,
                    "likes": d.likes or 0,
                    "source": "huggingface",
                    "tags": list(d.tags or []),
                }
                for d in datasets
            ]
        except Exception:
            return []

    def search_tavily(self, query: str, api_key: str = "") -> list[dict]:
        """Search using Tavily for broader dataset discovery.

        Requires: pip install autotrainer[search]
        Also needs TAVILY_API_KEY in config or environment.
        """
        if not api_key:
            return []

        try:
            from tavily import TavilyClient
        except ImportError:
            return [{"source": "tavily", "error": "tavily-python not installed. Run: pip install autotrainer[search]"}]

        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(query, max_results=5)
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:200],
                    "source": "tavily",
                }
                for r in response.get("results", [])
            ]
        except Exception:
            return []

    def download_from_hf(self, repo_id: str, filename: str | None = None) -> str:
        """Download a dataset from HuggingFace Hub.

        Returns the local path to the downloaded file.
        """
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
            )
            return path
        except Exception as e:
            from autotrainer.errors import DatasetDownloadError

            raise DatasetDownloadError(f"Failed to download {repo_id}: {e}")

    def convert_format(self, src: str, dst: str, src_fmt: str, dst_fmt: str) -> dict:
        """Convert dataset format."""
        from autotrainer.pf_integration.format_converter import convert

        return convert(src, dst, src_fmt, dst_fmt, base_dir=os.path.dirname(src))

    def split_dataset(self, path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> dict:
        """Split a dataset into train/val/test."""
        random.seed(seed)

        with open(path, "r") as f:
            lines = f.readlines()

        random.shuffle(lines)
        n = len(lines)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        base_dir = os.path.dirname(path)
        base_name = Path(path).stem

        train_path = os.path.join(base_dir, f"{base_name}_train.jsonl")
        val_path = os.path.join(base_dir, f"{base_name}_val.jsonl")
        test_path = os.path.join(base_dir, f"{base_name}_test.jsonl")

        for p, data in [
            (train_path, lines[:train_end]),
            (val_path, lines[train_end:val_end]),
            (test_path, lines[val_end:]),
        ]:
            with open(p, "w") as f:
                f.writelines(data)

        return {
            "train": {"path": train_path, "count": len(lines[:train_end])},
            "val": {"path": val_path, "count": len(lines[train_end:val_end])},
            "test": {"path": test_path, "count": len(lines[val_end:])},
            "total": n,
        }
