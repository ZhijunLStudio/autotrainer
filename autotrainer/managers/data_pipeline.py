"""Data pipeline — complete search → select → download → clean → convert → profile workflow."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

from autotrainer.utils.file_utils import safe_read_json


@dataclass
class DatasetInfo:
    """A discovered dataset candidate."""

    name: str = ""
    repo_id: str = ""
    source: str = ""  # huggingface / tavily / opendatalab / manual
    url: str = ""
    description: str = ""
    downloads: int = 0
    size_hint: str = ""
    tags: list[str] = field(default_factory=list)

    def display(self, idx: int) -> str:
        """Format for display in list."""
        src = self.source[:10]
        dl = f"{self.downloads:,}" if self.downloads else "-"
        desc = self.description[:50] if self.description else ""
        return f"  [{idx}] {self.name:<45} {src:<12} downloads={dl:<10} {desc}"


@dataclass
class PipelineResult:
    """Result of running the full data pipeline on one dataset."""

    dataset_name: str = ""
    download_path: str = ""
    cleaned_path: str = ""
    converted_path: str = ""
    profile: dict = field(default_factory=dict)
    dedup_stats: dict = field(default_factory=dict)
    split_result: dict = field(default_factory=dict)
    status: str = "completed"
    errors: list[str] = field(default_factory=list)


class DataPipeline:
    """Full data pipeline: search → select → download → clean → convert → profile → split.

    One class that handles the entire data workflow end-to-end.
    """

    def __init__(self, cache_dir: str, paddleformers_root: str = ""):
        self.cache_dir = cache_dir
        self.pf_root = paddleformers_root
        os.makedirs(cache_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # Step 1: Search (multi-source)
    # ══════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        hf_limit: int = 20,
        tavily_limit: int = 10,
        tavily_key: str = "",
    ) -> list[DatasetInfo]:
        """Search across multiple sources and return unified candidate list."""
        candidates = []

        # Source 1: HuggingFace Hub
        hf_results = self._search_hf(query, limit=hf_limit)
        candidates.extend(hf_results)

        # Source 2: Tavily
        if tavily_key:
            tavily_results = self._search_tavily(query, api_key=tavily_key, limit=tavily_limit)
            candidates.extend(tavily_results)

        # Deduplicate by repo_id
        seen = set()
        unique = []
        for c in candidates:
            key = c.repo_id or c.url or c.name
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def _search_hf(self, query: str, limit: int = 20) -> list[DatasetInfo]:
        """Search HuggingFace Hub."""
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            datasets = api.list_datasets(search=query, limit=limit)
            return [
                DatasetInfo(
                    name=d.id.split("/")[-1] if "/" in d.id else d.id,
                    repo_id=d.id,
                    source="huggingface",
                    url=f"https://huggingface.co/datasets/{d.id}",
                    downloads=d.downloads or 0,
                    tags=list(d.tags or []),
                )
                for d in datasets
            ]
        except Exception:
            return []

    def _search_tavily(self, query: str, api_key: str, limit: int = 10) -> list[DatasetInfo]:
        """Search Tavily and extract HF dataset references."""
        try:
            from tavily import TavilyClient
        except ImportError:
            return []

        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(query + " huggingface dataset", max_results=limit)

            results = []
            for r in response.get("results", []):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("content", "")[:200]

                # Try to extract HF repo_id from URL
                repo_id = ""
                hf_match = re.search(r"huggingface\.co/datasets/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)", url)
                if hf_match:
                    repo_id = hf_match.group(1)

                results.append(
                    DatasetInfo(
                        name=title[:60],
                        repo_id=repo_id,
                        source="huggingface" if repo_id else "tavily",
                        url=url,
                        description=snippet,
                    )
                )
            return results
        except Exception:
            return []

    # ══════════════════════════════════════════════════════════
    # Step 2: Download
    # ══════════════════════════════════════════════════════════

    def download(self, dataset: DatasetInfo) -> str:
        """Download a dataset. Returns path to the downloaded file(s)."""
        if dataset.source == "huggingface" and dataset.repo_id:
            return self._download_hf(dataset.repo_id)
        elif dataset.url:
            click.echo(f"  Manual download required: {dataset.url}")
            return ""
        return ""

    def _download_hf(self, repo_id: str) -> str:
        """Download from HuggingFace Hub."""
        from huggingface_hub import snapshot_download

        download_dir = os.path.join(self.cache_dir, "downloads", repo_id.replace("/", "_"))
        click.echo(f"  Downloading {repo_id} to {download_dir}...")

        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=download_dir,
            )
            click.echo(f"  Downloaded to: {path}")
            return path
        except Exception as e:
            click.echo(f"  Download failed: {e}")
            return ""

    # ══════════════════════════════════════════════════════════
    # Step 3: Find data files in download directory
    # ══════════════════════════════════════════════════════════

    def find_data_files(self, download_path: str) -> list[str]:
        """Find all data files (JSONL/JSON/CSV) in a download directory."""
        data_files = []
        extensions = {".jsonl", ".json", ".csv", ".tsv", ".parquet"}

        p = Path(download_path)
        if p.is_file() and p.suffix in extensions:
            return [str(p)]

        if p.is_dir():
            for root, dirs, files in os.walk(p):
                for f in files:
                    if Path(f).suffix in extensions:
                        data_files.append(os.path.join(root, f))

        return sorted(data_files)

    # ══════════════════════════════════════════════════════════
    # Step 4: Clean (dedup, remove bad rows, normalize)
    # ══════════════════════════════════════════════════════════

    def clean(self, src: str, dst: str) -> dict:
        """Clean a data file: remove duplicates, bad rows, normalize.

        Returns cleaning stats.
        """
        stats = {
            "input_lines": 0,
            "valid_lines": 0,
            "json_errors": 0,
            "duplicates": 0,
            "empty_content": 0,
            "output_lines": 0,
        }

        seen_hashes = set()
        cleaned_lines = []

        with open(src, "r") as fin:
            for line in fin:
                stats["input_lines"] += 1
                line = line.strip()
                if not line:
                    continue

                # JSON validation
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    stats["json_errors"] += 1
                    continue

                # Dedup by content hash
                content_hash = hashlib.md5(line.encode()).hexdigest()
                if content_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(content_hash)

                # Check for empty content
                if self._is_empty_sample(data):
                    stats["empty_content"] += 1
                    continue

                # Normalize
                normalized = self._normalize_sample(data)
                cleaned_lines.append(json.dumps(normalized, ensure_ascii=False) + "\n")
                stats["valid_lines"] += 1

        # Write cleaned file
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w") as fout:
            fout.writelines(cleaned_lines)

        stats["output_lines"] = len(cleaned_lines)
        return stats

    def _is_empty_sample(self, data: dict) -> bool:
        """Check if a sample has no meaningful content."""
        # erniekit format
        if "text_info" in data:
            text_info = data.get("text_info", [])
            if isinstance(text_info, list):
                has_text = any(
                    isinstance(t, dict) and str(t.get("text", "")).strip() for t in text_info
                )
                return not has_text
        # messages format
        if "messages" in data:
            messages = data.get("messages", [])
            if isinstance(messages, list):
                has_content = any(
                    isinstance(m, dict) and str(m.get("content", "")).strip() for m in messages
                )
                return not has_content
        return False

    def _normalize_sample(self, data: dict) -> dict:
        """Normalize a sample (strip whitespace, etc.)."""
        # Normalize text_info
        if "text_info" in data and isinstance(data["text_info"], list):
            for item in data["text_info"]:
                if isinstance(item, dict) and "text" in item:
                    item["text"] = str(item["text"]).strip()

        # Normalize messages
        if "messages" in data and isinstance(data["messages"], list):
            for msg in data["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    msg["content"] = str(msg["content"]).strip()

        return data

    # ══════════════════════════════════════════════════════════
    # Step 5: Convert format
    # ══════════════════════════════════════════════════════════

    def detect_format(self, path: str) -> str:
        """Auto-detect the format of a data file."""
        from autotrainer.pf_integration.dataset_validator import DatasetValidator

        validator = DatasetValidator()
        result = validator.validate_file(path)
        return result.format_detected or "unknown"

    def convert(self, src: str, dst: str, src_fmt: str, dst_fmt: str) -> dict:
        """Convert data format."""
        from autotrainer.pf_integration.format_converter import convert

        return convert(src, dst, src_fmt, dst_fmt, base_dir=os.path.dirname(src))

    # ══════════════════════════════════════════════════════════
    # Step 6: Profile
    # ══════════════════════════════════════════════════════════

    def profile(self, path: str) -> dict:
        """Generate a statistical profile of a dataset."""
        p = Path(path)
        if not p.exists():
            return {"error": "file not found"}

        profile = {
            "path": path,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "format": self.detect_format(path),
        }

        # Count lines and stats
        text_lens = []
        field_counts: dict[str, int] = {}
        json_errors = 0
        line_count = 0
        image_count = 0

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    json_errors += 1
                    continue

                line_count += 1
                for key in data:
                    field_counts[key] = field_counts.get(key, 0) + 1

                # Text lengths
                if "text_info" in data:
                    tl = data.get("text_info", [])
                    if isinstance(tl, list):
                        total = sum(len(str(t.get("text", ""))) for t in tl if isinstance(t, dict))
                        text_lens.append(total)
                    imgs = data.get("image_info", [])
                    if isinstance(imgs, list):
                        image_count += len(imgs)
                elif "messages" in data:
                    msgs = data.get("messages", [])
                    if isinstance(msgs, list):
                        total = sum(len(str(m.get("content", ""))) for m in msgs if isinstance(m, dict))
                        text_lens.append(total)
                    imgs = data.get("images", [])
                    if isinstance(imgs, list):
                        image_count += len(imgs)

        profile["num_samples"] = line_count
        profile["json_errors"] = json_errors
        profile["image_count"] = image_count
        profile["has_images"] = image_count > 0
        profile["field_coverage"] = {k: round(v / line_count, 2) for k, v in field_counts.items()} if line_count else {}

        if text_lens:
            s = sorted(text_lens)
            n = len(s)
            profile["text_lengths"] = {
                "min": s[0],
                "max": s[-1],
                "avg": round(sum(s) / n),
                "p50": s[n // 2],
                "p95": s[int(n * 0.95)] if n > 20 else s[-1],
            }

        # Sample preview
        samples = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        profile["sample_preview"] = samples

        return profile

    # ══════════════════════════════════════════════════════════
    # Step 7: Split
    # ══════════════════════════════════════════════════════════

    def split(self, path: str, train_ratio: float = 0.9, val_ratio: float = 0.05, seed: int = 42) -> dict:
        """Split dataset into train/val/test."""
        random.seed(seed)

        with open(path, "r") as f:
            lines = f.readlines()

        random.shuffle(lines)
        n = len(lines)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        base_dir = os.path.dirname(path)
        base_name = Path(path).stem

        parts = {}
        for name, start, end in [
            ("train", 0, train_end),
            ("val", train_end, val_end),
            ("test", val_end, n),
        ]:
            part_path = os.path.join(base_dir, f"{base_name}_{name}.jsonl")
            with open(part_path, "w") as f:
                f.writelines(lines[start:end])
            parts[name] = {"path": part_path, "count": end - start}

        parts["total"] = n
        return parts

    # ══════════════════════════════════════════════════════════
    # Full pipeline: search → select → download → clean → convert → profile → split
    # ══════════════════════════════════════════════════════════

    def run_full_pipeline(
        self,
        query: str,
        task: str = "paddleocr-vl",
        target_format: str = "erniekit",
        tavily_key: str = "",
        output_dir: str = "",
    ) -> list[PipelineResult]:
        """Run the complete data pipeline for selected datasets.

        1. Search
        2. User selects
        3. Download each
        4. Find data files
        5. Clean (dedup, bad rows)
        6. Convert to target format
        7. Profile
        8. Split (train/val/test)
        """
        if not output_dir:
            output_dir = os.path.join(self.cache_dir, "processed", task)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Search
        click.echo(f"\n[Step 1] Searching: {query}")
        candidates = self.search(query, tavily_key=tavily_key)

        if not candidates:
            click.echo("  No datasets found.")
            return []

        # Step 2: Display and select
        click.echo(f"\n[Step 2] Select datasets to download:")
        click.echo(f"  {'#':<5} {'Name':<46} {'Source':<13} {'Info'}")
        click.echo(f"  {'-' * 90}")
        for i, c in enumerate(candidates, 1):
            click.echo(c.display(i))

        click.echo(f"\n  Enter numbers to download (e.g., 1,3,5), or 'all' for all, 'q' to quit:")
        selection = click.prompt("  Select", default="1")

        if selection.lower() == "q":
            return []

        if selection.lower() == "all":
            selected_indices = list(range(len(candidates)))
        else:
            selected_indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]

        selected = [candidates[i] for i in selected_indices if 0 <= i < len(candidates)]
        if not selected:
            click.echo("  No valid selection.")
            return []

        results = []

        for ds in selected:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Processing: {ds.name} ({ds.repo_id or ds.url})")
            click.echo(f"{'=' * 60}")

            result = PipelineResult(dataset_name=ds.name)

            # Step 3: Download
            click.echo(f"\n[Step 3] Downloading...")
            download_path = self.download(ds)
            if not download_path:
                result.status = "download_failed"
                result.errors.append("Download failed")
                results.append(result)
                continue

            # Step 4: Find data files
            click.echo(f"\n[Step 4] Finding data files...")
            data_files = self.find_data_files(download_path)
            click.echo(f"  Found {len(data_files)} data files")
            if not data_files:
                result.status = "no_data_files"
                result.errors.append("No data files found in download")
                results.append(result)
                continue

            # Process each data file
            ds_output_dir = os.path.join(output_dir, ds.name.replace("/", "_"))
            os.makedirs(ds_output_dir, exist_ok=True)

            for data_file in data_files:
                fname = Path(data_file).name
                click.echo(f"\n  Processing: {fname}")

                # Step 5: Clean
                click.echo(f"  [Step 5] Cleaning (dedup, remove bad rows)...")
                cleaned_path = os.path.join(ds_output_dir, f"cleaned_{fname}")
                clean_stats = self.clean(data_file, cleaned_path)
                click.echo(
                    f"    Input: {clean_stats['input_lines']}, "
                    f"JSON errors: {clean_stats['json_errors']}, "
                    f"Duplicates: {clean_stats['duplicates']}, "
                    f"Empty: {clean_stats['empty_content']}, "
                    f"Output: {clean_stats['output_lines']}"
                )
                result.cleaned_path = cleaned_path
                result.dedup_stats = clean_stats

                # Step 6: Convert if needed
                click.echo(f"  [Step 6] Detecting format...")
                src_fmt = self.detect_format(cleaned_path)
                click.echo(f"    Detected: {src_fmt}")

                if src_fmt != target_format and src_fmt != "unknown":
                    click.echo(f"    Converting {src_fmt} → {target_format}...")
                    converted_path = os.path.join(ds_output_dir, f"converted_{fname}")
                    conv_stats = self.convert(cleaned_path, converted_path, src_fmt, target_format)
                    click.echo(f"    Converted: {conv_stats.get('converted', 0)} samples, skipped: {conv_stats.get('skipped', 0)}")
                    result.converted_path = converted_path
                    final_path = converted_path
                else:
                    final_path = cleaned_path

                # Step 7: Profile
                click.echo(f"  [Step 7] Profiling...")
                prof = self.profile(final_path)
                result.profile = prof
                click.echo(f"    Samples: {prof.get('num_samples', 0)}")
                click.echo(f"    Size: {prof.get('size_mb', 0)} MB")
                click.echo(f"    Has images: {prof.get('has_images', False)}")
                tl = prof.get("text_lengths", {})
                if tl:
                    click.echo(f"    Text lengths: min={tl.get('min', 0)}, avg={tl.get('avg', 0)}, max={tl.get('max', 0)}")

                # Step 8: Split
                click.echo(f"  [Step 8] Splitting train/val/test...")
                split_result = self.split(final_path)
                result.split_result = split_result
                click.echo(
                    f"    train={split_result.get('train', {}).get('count', 0)}, "
                    f"val={split_result.get('val', {}).get('count', 0)}, "
                    f"test={split_result.get('test', {}).get('count', 0)}"
                )

                result.download_path = final_path
                result.status = "completed"

            results.append(result)

        # Save pipeline summary
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        summary = {
            "query": query,
            "task": task,
            "target_format": target_format,
            "results": [
                {
                    "dataset": r.dataset_name,
                    "status": r.status,
                    "num_samples": r.profile.get("num_samples", 0),
                    "cleaned_path": r.cleaned_path,
                    "converted_path": r.converted_path,
                    "errors": r.errors,
                }
                for r in results
            ],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        click.echo(f"\n  Pipeline summary saved to: {summary_path}")

        return results
