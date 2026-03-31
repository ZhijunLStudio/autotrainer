"""Data pipeline — complete search → select → download → clean → convert → profile workflow."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import click


@dataclass
class DatasetInfo:
    """A discovered dataset candidate."""

    name: str = ""
    repo_id: str = ""
    source: str = ""  # huggingface / modelscope / kaggle / opendatalab / pwc / tavily
    url: str = ""
    description: str = ""
    downloads: int = 0
    size_hint: str = ""
    tags: list[str] = field(default_factory=list)
    download_method: str = "hf"  # hf / modelscope / kaggle / manual

    def display(self, idx: int) -> str:
        src_abbr = {
            "huggingface": "HF",
            "modelscope": "ModelScope",
            "kaggle": "Kaggle",
            "opendatalab": "OpenDataLab",
            "pwc": "PapersWithCode",
            "tavily": "Web",
        }.get(self.source, self.source[:10])
        dl = f"{self.downloads:,}" if self.downloads > 0 else "-"
        desc = self.description[:55] if self.description else ""
        return f"  [{idx:>2}] {self.name:<42} [{src_abbr:<12}] dl={dl:<10} {desc}"


@dataclass
class PipelineResult:
    """Result of running the full data pipeline on one dataset."""

    dataset_name: str = ""
    download_path: str = ""
    cleaned_path: str = ""
    converted_path: str = ""
    profile: dict = field(default_factory=dict)
    clean_stats: dict = field(default_factory=dict)
    split_result: dict = field(default_factory=dict)
    status: str = "completed"
    errors: list[str] = field(default_factory=list)


class DataPipeline:
    """Full data pipeline: search → interactive select → download → clean → convert → profile → split."""

    def __init__(self, cache_dir: str, paddleformers_root: str = ""):
        self.cache_dir = cache_dir
        self.pf_root = paddleformers_root
        os.makedirs(cache_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # Step 1: Multi-source search
    # ══════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        tavily_key: str = "",
        modelscope_token: str = "",
        kaggle_configured: bool = False,
    ) -> list[DatasetInfo]:
        """Search across all configured sources, deduplicate, return combined list."""
        all_results: list[DatasetInfo] = []

        sources = [
            ("HuggingFace", lambda: self._search_hf(query, limit=20)),
            ("ModelScope",  lambda: self._search_modelscope(query, limit=15, token=modelscope_token)),
            ("PapersWithCode", lambda: self._search_pwc(query, limit=10)),
        ]
        if kaggle_configured:
            sources.append(("Kaggle", lambda: self._search_kaggle(query, limit=10)))
        if tavily_key:
            sources.append(("Tavily", lambda: self._search_tavily(query, api_key=tavily_key, limit=10)))

        for src_name, search_fn in sources:
            click.echo(f"  Searching {src_name}...", nl=False)
            try:
                results = search_fn()
                click.echo(f" {len(results)} results")
                all_results.extend(results)
            except Exception as e:
                click.echo(f" failed ({e})")

        # Dedup by repo_id → url → name
        seen: set[str] = set()
        unique: list[DatasetInfo] = []
        for c in all_results:
            key = c.repo_id or c.url or c.name
            if key and key not in seen:
                seen.add(key)
                unique.append(c)

        # Sort: HF first (most reliable download), then others
        order = {"huggingface": 0, "modelscope": 1, "pwc": 2, "kaggle": 3, "tavily": 4}
        unique.sort(key=lambda x: (order.get(x.source, 9), -(x.downloads or 0)))

        return unique

    def _search_hf(self, query: str, limit: int = 20) -> list[DatasetInfo]:
        """Search HuggingFace Hub."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            datasets = list(api.list_datasets(search=query, limit=limit))
            return [
                DatasetInfo(
                    name=d.id,
                    repo_id=d.id,
                    source="huggingface",
                    url=f"https://huggingface.co/datasets/{d.id}",
                    downloads=d.downloads or 0,
                    tags=list(d.tags or []),
                    download_method="hf",
                )
                for d in datasets
            ]
        except Exception:
            return []

    def _search_modelscope(self, query: str, limit: int = 15, token: str = "") -> list[DatasetInfo]:
        """Search ModelScope (魔搭)."""
        try:
            from modelscope.hub.api import HubApi
            api = HubApi()
            datasets = api.list_datasets(query=query, page_number=1, page_size=limit)
            results = []
            for d in (datasets or []):
                did = getattr(d, "Id", "") or getattr(d, "id", "") or ""
                name = getattr(d, "Name", "") or getattr(d, "name", did) or did
                dl = getattr(d, "Downloads", 0) or 0
                results.append(DatasetInfo(
                    name=name,
                    repo_id=did,
                    source="modelscope",
                    url=f"https://modelscope.cn/datasets/{did}",
                    downloads=int(dl),
                    download_method="modelscope",
                ))
            return results
        except Exception:
            return []

    def _search_pwc(self, query: str, limit: int = 10) -> list[DatasetInfo]:
        """Search Papers With Code datasets API (no auth needed)."""
        try:
            import urllib.request, urllib.parse
            q = urllib.parse.quote(query)
            url = f"https://paperswithcode.com/api/v1/datasets/?q={q}&page_size={limit}"
            req = urllib.request.Request(url, headers={"User-Agent": "autotrainer/0.1"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            results = []
            for d in data.get("results", []):
                results.append(DatasetInfo(
                    name=d.get("name", ""),
                    repo_id="",
                    source="pwc",
                    url=d.get("url", ""),
                    description=d.get("description", "")[:120],
                    download_method="manual",
                ))
            return results
        except Exception:
            return []

    def _search_kaggle(self, query: str, limit: int = 10) -> list[DatasetInfo]:
        """Search Kaggle datasets (requires kaggle CLI configured)."""
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "list", "-s", query, "--csv"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                return []
            lines = result.stdout.strip().splitlines()
            results = []
            for line in lines[1:limit + 1]:  # skip header
                parts = line.split(",")
                if len(parts) >= 2:
                    ref = parts[0].strip()
                    title = parts[1].strip()
                    results.append(DatasetInfo(
                        name=title or ref,
                        repo_id=ref,
                        source="kaggle",
                        url=f"https://www.kaggle.com/datasets/{ref}",
                        download_method="kaggle",
                    ))
            return results
        except Exception:
            return []

    def _search_tavily(self, query: str, api_key: str, limit: int = 10) -> list[DatasetInfo]:
        """Search Tavily and parse HF/ModelScope references out of results."""
        try:
            from tavily import TavilyClient
        except ImportError:
            return []
        try:
            client = TavilyClient(api_key=api_key)
            resp = client.search(query + " dataset download", max_results=limit)
            results = []
            for r in resp.get("results", []):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("content", "")[:120]

                # Try to promote to HF if URL contains a dataset ref
                hf_match = re.search(r"huggingface\.co/datasets/([\w-]+/[\w.-]+)", url)
                ms_match = re.search(r"modelscope\.cn/datasets/([\w-]+/[\w.-]+)", url)

                if hf_match:
                    repo_id = hf_match.group(1)
                    results.append(DatasetInfo(
                        name=repo_id,
                        repo_id=repo_id,
                        source="huggingface",
                        url=url,
                        description=snippet,
                        download_method="hf",
                    ))
                elif ms_match:
                    repo_id = ms_match.group(1)
                    results.append(DatasetInfo(
                        name=repo_id,
                        repo_id=repo_id,
                        source="modelscope",
                        url=url,
                        description=snippet,
                        download_method="modelscope",
                    ))
                else:
                    results.append(DatasetInfo(
                        name=title[:60],
                        source="tavily",
                        url=url,
                        description=snippet,
                        download_method="manual",
                    ))
            return results
        except Exception:
            return []

    # ══════════════════════════════════════════════════════════
    # Step 2: Interactive selection (in-terminal)
    # ══════════════════════════════════════════════════════════

    def interactive_select(self, candidates: list[DatasetInfo]) -> list[DatasetInfo]:
        """Show candidate list and prompt user to select which to download.

        Returns the selected subset.
        """
        if not candidates:
            return []

        click.echo(f"\n  Found {len(candidates)} datasets. Enter numbers to download:")
        click.echo(f"  {'':4} {'Name':<43} {'Source':<14} {'Downloads':<12} {'Description'}")
        click.echo(f"  {'-' * 100}")
        for i, c in enumerate(candidates, 1):
            click.echo(c.display(i))

        click.echo()
        click.echo("  Enter numbers (e.g. 1,3,5), range (e.g. 1-5), 'all', or Enter to skip download:")
        raw = click.prompt("  Select", default="", show_default=False)

        raw = raw.strip()
        if not raw or raw.lower() in ("q", "quit", "skip"):
            return []
        if raw.lower() == "all":
            return candidates

        indices: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                lo, _, hi = part.partition("-")
                try:
                    for n in range(int(lo), int(hi) + 1):
                        indices.add(n - 1)
                except ValueError:
                    pass
            elif part.isdigit():
                indices.add(int(part) - 1)

        selected = [candidates[i] for i in sorted(indices) if 0 <= i < len(candidates)]
        if not selected:
            click.echo("  No valid selection, skipping download.")
        return selected

    # ══════════════════════════════════════════════════════════
    # Step 3: Download
    # ══════════════════════════════════════════════════════════

    def download(self, dataset: DatasetInfo) -> str:
        """Download a dataset. Returns local path (empty if failed)."""
        method = dataset.download_method

        if method == "hf" and dataset.repo_id:
            return self._download_hf(dataset.repo_id)
        elif method == "modelscope" and dataset.repo_id:
            return self._download_modelscope(dataset.repo_id)
        elif method == "kaggle" and dataset.repo_id:
            return self._download_kaggle(dataset.repo_id)
        else:
            click.echo(f"  Manual download required: {dataset.url}")
            path = click.prompt(
                "  Paste the local path after downloading (or Enter to skip)",
                default="", show_default=False
            )
            return path.strip() if path.strip() else ""

    def _download_hf(self, repo_id: str) -> str:
        from huggingface_hub import snapshot_download
        dst = os.path.join(self.cache_dir, "downloads", repo_id.replace("/", "__"))
        click.echo(f"  Downloading {repo_id} from HuggingFace...")
        try:
            path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dst)
            click.echo(f"  Saved to: {path}")
            return path
        except Exception as e:
            click.echo(f"  [FAIL] {e}")
            # Offer manual fallback
            manual = click.prompt(
                "  Download failed. Paste local path if you downloaded manually (or Enter to skip)",
                default="", show_default=False
            )
            return manual.strip() if manual.strip() else ""

    def _download_modelscope(self, repo_id: str) -> str:
        dst = os.path.join(self.cache_dir, "downloads", repo_id.replace("/", "__"))
        click.echo(f"  Downloading {repo_id} from ModelScope...")
        try:
            from modelscope.msdatasets import MsDataset
            ds = MsDataset.load(repo_id, cache_dir=dst)
            click.echo(f"  Saved to: {dst}")
            return dst
        except Exception as e:
            click.echo(f"  [FAIL] {e}")
            manual = click.prompt(
                "  Paste local path if downloaded manually (or Enter to skip)",
                default="", show_default=False
            )
            return manual.strip() if manual.strip() else ""

    def _download_kaggle(self, ref: str) -> str:
        dst = os.path.join(self.cache_dir, "downloads", ref.replace("/", "__"))
        os.makedirs(dst, exist_ok=True)
        click.echo(f"  Downloading {ref} from Kaggle...")
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "download", ref, "--unzip", "-p", dst],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                click.echo(f"  Saved to: {dst}")
                return dst
            else:
                click.echo(f"  [FAIL] {result.stderr[:200]}")
        except Exception as e:
            click.echo(f"  [FAIL] {e}")

        manual = click.prompt(
            "  Paste local path if downloaded manually (or Enter to skip)",
            default="", show_default=False
        )
        return manual.strip() if manual.strip() else ""

    # ══════════════════════════════════════════════════════════
    # Step 4: Find data files in downloaded directory
    # ══════════════════════════════════════════════════════════

    def find_data_files(self, path: str) -> list[str]:
        """Recursively find all data files (JSONL/JSON/CSV/TSV/Parquet)."""
        extensions = {".jsonl", ".json", ".csv", ".tsv", ".parquet"}
        p = Path(path)
        if p.is_file() and p.suffix.lower() in extensions:
            return [str(p)]
        files = []
        for root, _, filenames in os.walk(p):
            for fn in filenames:
                if Path(fn).suffix.lower() in extensions:
                    files.append(os.path.join(root, fn))
        return sorted(files)

    # ══════════════════════════════════════════════════════════
    # Step 5: Clean
    # ══════════════════════════════════════════════════════════

    def clean(self, src: str, dst: str) -> dict:
        """Remove bad JSON, duplicates (by MD5), empty content, normalize whitespace."""
        stats = {
            "input_lines": 0, "valid_lines": 0,
            "json_errors": 0, "duplicates": 0,
            "empty_content": 0, "output_lines": 0,
        }
        seen_hashes: set[str] = set()
        cleaned: list[str] = []

        with open(src, "r", errors="replace") as fin:
            for line in fin:
                stats["input_lines"] += 1
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    stats["json_errors"] += 1
                    continue
                h = hashlib.md5(stripped.encode()).hexdigest()
                if h in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(h)
                if self._is_empty(data):
                    stats["empty_content"] += 1
                    continue
                normalized = self._normalize(data)
                cleaned.append(json.dumps(normalized, ensure_ascii=False) + "\n")
                stats["valid_lines"] += 1

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w") as fout:
            fout.writelines(cleaned)

        stats["output_lines"] = len(cleaned)
        return stats

    def _is_empty(self, data: dict) -> bool:
        if "text_info" in data:
            return not any(
                isinstance(t, dict) and str(t.get("text", "")).strip()
                for t in (data.get("text_info") or [])
            )
        if "messages" in data:
            return not any(
                isinstance(m, dict) and str(m.get("content", "")).strip()
                for m in (data.get("messages") or [])
            )
        return False

    def _normalize(self, data: dict) -> dict:
        if "text_info" in data and isinstance(data["text_info"], list):
            for item in data["text_info"]:
                if isinstance(item, dict) and "text" in item:
                    item["text"] = str(item["text"]).strip()
        if "messages" in data and isinstance(data["messages"], list):
            for msg in data["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    msg["content"] = str(msg["content"]).strip()
        return data

    # ══════════════════════════════════════════════════════════
    # Step 6: Detect format and convert
    # ══════════════════════════════════════════════════════════

    def detect_format(self, path: str) -> str:
        from autotrainer.pf_integration.dataset_validator import DatasetValidator
        return DatasetValidator().validate_file(path).format_detected or "unknown"

    def convert(self, src: str, dst: str, src_fmt: str, dst_fmt: str) -> dict:
        from autotrainer.pf_integration.format_converter import convert
        return convert(src, dst, src_fmt, dst_fmt, base_dir=os.path.dirname(src))

    # ══════════════════════════════════════════════════════════
    # Step 7: Profile
    # ══════════════════════════════════════════════════════════

    def profile(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {"error": "file not found"}

        fmt = self.detect_format(path)
        text_lens: list[int] = []
        field_counts: dict[str, int] = {}
        json_errors = 0
        line_count = 0
        image_count = 0

        with open(path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    data = json.loads(s)
                except json.JSONDecodeError:
                    json_errors += 1
                    continue

                line_count += 1
                for key in data:
                    field_counts[key] = field_counts.get(key, 0) + 1

                if "text_info" in data:
                    tl = data.get("text_info", [])
                    text_lens.append(sum(len(str(t.get("text", ""))) for t in (tl or []) if isinstance(t, dict)))
                    imgs = data.get("image_info", [])
                    image_count += len(imgs) if isinstance(imgs, list) else 0
                elif "messages" in data:
                    msgs = data.get("messages", [])
                    text_lens.append(sum(len(str(m.get("content", ""))) for m in (msgs or []) if isinstance(m, dict)))
                    imgs = data.get("images", [])
                    image_count += len(imgs) if isinstance(imgs, list) else (1 if imgs else 0)

        result: dict = {
            "path": path,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            "format": fmt,
            "num_samples": line_count,
            "json_errors": json_errors,
            "image_count": image_count,
            "has_images": image_count > 0,
            "field_coverage": {k: round(v / line_count, 2) for k, v in field_counts.items()} if line_count else {},
        }
        if text_lens:
            s = sorted(text_lens)
            n = len(s)
            result["text_lengths"] = {
                "min": s[0], "max": s[-1], "avg": round(sum(s) / n),
                "p50": s[n // 2], "p95": s[int(n * 0.95)] if n > 20 else s[-1],
            }
        # sample preview
        samples: list[dict] = []
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        result["sample_preview"] = samples
        return result

    # ══════════════════════════════════════════════════════════
    # Step 8: Split
    # ══════════════════════════════════════════════════════════

    def split(self, path: str, train_ratio: float = 0.9, val_ratio: float = 0.05, seed: int = 42) -> dict:
        random.seed(seed)
        with open(path, "r") as f:
            lines = f.readlines()
        random.shuffle(lines)
        n = len(lines)
        t = int(n * train_ratio)
        v = t + int(n * val_ratio)
        base = Path(path)
        parts = {}
        for name, start, end in [("train", 0, t), ("val", t, v), ("test", v, n)]:
            part_path = str(base.parent / f"{base.stem}_{name}.jsonl")
            with open(part_path, "w") as f:
                f.writelines(lines[start:end])
            parts[name] = {"path": part_path, "count": end - start}
        parts["total"] = n
        return parts

    # ══════════════════════════════════════════════════════════
    # Full pipeline
    # ══════════════════════════════════════════════════════════

    def run_full_pipeline(
        self,
        query: str,
        task: str = "paddleocr-vl",
        target_format: str = "erniekit",
        tavily_key: str = "",
        modelscope_token: str = "",
        output_dir: str = "",
    ) -> list[PipelineResult]:
        """Search → interactive select → download → clean → convert → profile → split."""
        if not output_dir:
            output_dir = os.path.join(self.cache_dir, "processed", task)
        os.makedirs(output_dir, exist_ok=True)

        # 1. Search
        click.echo(f"\n[1/7] Searching: {query}")
        kaggle_ok = subprocess.run(["kaggle", "--version"], capture_output=True).returncode == 0
        candidates = self.search(query, tavily_key=tavily_key, modelscope_token=modelscope_token, kaggle_configured=kaggle_ok)

        if not candidates:
            click.echo("  No datasets found.")
            return []

        # 2. Interactive selection
        click.echo(f"\n[2/7] Select datasets to download:")
        selected = self.interactive_select(candidates)
        if not selected:
            click.echo("  No selection, exiting.")
            return []

        results: list[PipelineResult] = []

        for ds in selected:
            click.echo(f"\n{'═' * 65}")
            click.echo(f"  {ds.name}  [{ds.source}]  {ds.url}")
            click.echo(f"{'═' * 65}")

            res = PipelineResult(dataset_name=ds.name)

            # 3. Download
            click.echo(f"\n[3/7] Downloading...")
            dl_path = self.download(ds)
            if not dl_path:
                res.status = "download_failed"
                res.errors.append("Download failed or skipped")
                results.append(res)
                continue

            # 4. Find data files
            click.echo(f"\n[4/7] Finding data files in {dl_path}...")
            data_files = self.find_data_files(dl_path)
            click.echo(f"  Found {len(data_files)} file(s): {[os.path.basename(f) for f in data_files]}")
            if not data_files:
                res.status = "no_data_files"
                res.errors.append("No data files found")
                results.append(res)
                continue

            ds_dir = os.path.join(output_dir, re.sub(r"[/\\]", "__", ds.name))
            os.makedirs(ds_dir, exist_ok=True)

            for data_file in data_files:
                fname = Path(data_file).name
                click.echo(f"\n  File: {fname}")

                # 5. Clean
                click.echo(f"  [5/7] Cleaning...")
                cleaned_path = os.path.join(ds_dir, f"cleaned_{fname}")
                stats = self.clean(data_file, cleaned_path)
                click.echo(
                    f"    {stats['input_lines']} in → "
                    f"dupes={stats['duplicates']}, "
                    f"bad_json={stats['json_errors']}, "
                    f"empty={stats['empty_content']} → "
                    f"{stats['output_lines']} out"
                )
                res.cleaned_path = cleaned_path
                res.clean_stats = stats

                if stats["output_lines"] == 0:
                    click.echo("    [WARN] No usable lines after cleaning, skipping.")
                    continue

                # 6. Convert if needed
                click.echo(f"  [6/7] Format conversion...")
                src_fmt = self.detect_format(cleaned_path)
                click.echo(f"    Detected: {src_fmt or 'unknown'}")
                if src_fmt and src_fmt != "unknown" and src_fmt != target_format:
                    converted_path = os.path.join(ds_dir, f"converted_{fname}")
                    conv = self.convert(cleaned_path, converted_path, src_fmt, target_format)
                    click.echo(f"    {src_fmt} → {target_format}: {conv.get('converted', 0)} converted, {conv.get('skipped', 0)} skipped")
                    res.converted_path = converted_path
                    final = converted_path
                else:
                    click.echo(f"    Already {target_format or src_fmt}, no conversion needed")
                    final = cleaned_path

                # 7. Profile
                click.echo(f"  [7/7] Profiling...")
                prof = self.profile(final)
                res.profile = prof
                tl = prof.get("text_lengths", {})
                click.echo(f"    {prof.get('num_samples', 0)} samples, {prof.get('size_mb', 0)} MB, images={prof.get('image_count', 0)}")
                if tl:
                    click.echo(f"    text: min={tl['min']}, avg={tl['avg']}, p95={tl['p95']}, max={tl['max']}")
                if prof.get("sample_preview"):
                    click.echo(f"    Preview of first sample:")
                    sample = prof["sample_preview"][0]
                    click.echo(f"      {json.dumps(sample, ensure_ascii=False)[:200]}")

                # 8. Split
                click.echo(f"\n  Splitting train/val/test (90/5/5)...")
                split_r = self.split(final)
                res.split_result = split_r
                click.echo(
                    f"    train={split_r['train']['count']}, "
                    f"val={split_r['val']['count']}, "
                    f"test={split_r['test']['count']}"
                )
                click.echo(f"    train → {split_r['train']['path']}")
                res.download_path = final
                res.status = "completed"

            results.append(res)

        # Summary
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "query": query, "task": task, "target_format": target_format,
                    "results": [
                        {
                            "dataset": r.dataset_name, "status": r.status,
                            "samples": r.profile.get("num_samples", 0),
                            "train_path": r.split_result.get("train", {}).get("path", ""),
                            "errors": r.errors,
                        }
                        for r in results
                    ],
                },
                f, indent=2, ensure_ascii=False,
            )
        click.echo(f"\n  Summary → {summary_path}")
        return results
