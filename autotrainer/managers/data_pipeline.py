"""Data pipeline — search → select → download → convert → clean → profile → split.

Search strategy: Tavily is the universal primary search engine (handles any language).
HuggingFace API is used to enrich results where a repo_id is identified.
ModelScope / Kaggle are direct API searches as secondary sources.
"""

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
    source: str = ""          # huggingface / modelscope / kaggle / web
    url: str = ""
    description: str = ""
    downloads: int = 0
    tags: list[str] = field(default_factory=list)
    download_method: str = "manual"   # hf / modelscope / kaggle / manual


@dataclass
class PipelineResult:
    """Result of running the full data pipeline on one dataset."""

    dataset_name: str = ""
    raw_path: str = ""
    cleaned_path: str = ""
    converted_path: str = ""
    profile: dict = field(default_factory=dict)
    clean_stats: dict = field(default_factory=dict)
    split_result: dict = field(default_factory=dict)
    status: str = "completed"
    errors: list[str] = field(default_factory=list)


class DataPipeline:
    """Full data pipeline: search → interactive select → download → convert raw format
    → clean (dedup + bad rows) → profile → split train/val/test.
    """

    def __init__(self, cache_dir: str, paddleformers_root: str = ""):
        self.cache_dir = cache_dir
        self.pf_root = paddleformers_root
        os.makedirs(cache_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # Step 1: Search
    # ══════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        tavily_key: str = "",
        modelscope_token: str = "",
    ) -> list[DatasetInfo]:
        """Search for datasets. Tavily is primary (any language), HF/MS/Kaggle secondary."""
        all_results: list[DatasetInfo] = []

        # Primary: Tavily (handles Chinese, English, everything)
        if tavily_key:
            click.echo("  Tavily (primary)...", nl=False)
            tv = self._search_tavily(query, api_key=tavily_key, limit=15)
            click.echo(f" {len(tv)} results")
            all_results.extend(tv)

            # Also search in English for broader coverage
            en_query = self._to_english_hint(query)
            if en_query != query:
                click.echo(f"  Tavily (English fallback: {en_query})...", nl=False)
                tv2 = self._search_tavily(en_query, api_key=tavily_key, limit=10)
                click.echo(f" {len(tv2)} results")
                all_results.extend(tv2)
        else:
            click.echo("  [Tavily skipped] Add tavily_api_key to ~/.autotrainer/config.yaml for best results")
            # Fallback: HF direct search
            click.echo("  HuggingFace (fallback)...", nl=False)
            hf = self._search_hf(query, limit=20)
            click.echo(f" {len(hf)} results")
            all_results.extend(hf)

        # Secondary: ModelScope
        click.echo("  ModelScope...", nl=False)
        ms = self._search_modelscope(query, limit=10)
        click.echo(f" {len(ms)} results")
        all_results.extend(ms)

        # Secondary: Kaggle
        click.echo("  Kaggle...", nl=False)
        kg = self._search_kaggle(query, limit=10)
        click.echo(f" {len(kg)} results")
        all_results.extend(kg)

        # Deduplicate by repo_id → url → name
        seen: set[str] = set()
        unique: list[DatasetInfo] = []
        for c in all_results:
            key = c.repo_id or c.url or c.name
            if key and key not in seen:
                seen.add(key)
                unique.append(c)

        # Sort: known-downloadable sources first, then by downloads desc
        prio = {"huggingface": 0, "modelscope": 1, "kaggle": 2, "web": 3}
        unique.sort(key=lambda x: (prio.get(x.source, 9), -(x.downloads or 0)))

        return unique

    def _to_english_hint(self, query: str) -> str:
        """Very simple heuristic: if query has CJK chars, add an English equivalent hint."""
        if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", query):
            # For well-known terms, map common ones
            mapping = {
                "阿拉伯": "Arabic", "中文": "Chinese", "日语": "Japanese",
                "手写": "handwritten", "场景文字": "scene text", "数据集": "dataset",
                "识别": "recognition", "检测": "detection", "OCR": "OCR",
            }
            en = query
            for zh, eng in mapping.items():
                en = en.replace(zh, eng)
            return en if en != query else query
        return query

    def _search_tavily(self, query: str, api_key: str, limit: int = 15) -> list[DatasetInfo]:
        """Tavily web search — extract HF/MS dataset refs from results."""
        try:
            from tavily import TavilyClient
        except ImportError:
            return []

        try:
            client = TavilyClient(api_key=api_key)
            resp = client.search(
                f"{query} dataset download",
                max_results=limit,
                search_depth="advanced",
                include_domains=["huggingface.co", "modelscope.cn", "kaggle.com",
                                 "opendatalab.com", "paperswithcode.com", "github.com"],
            )
        except Exception:
            try:
                client = TavilyClient(api_key=api_key)
                resp = client.search(f"{query} dataset", max_results=limit)
            except Exception:
                return []

        results = []
        for r in resp.get("results", []):
            title = r.get("title", "").strip()
            url = r.get("url", "").strip()
            # Use full content, truncate only at 300 chars for description
            content = r.get("content", "").strip()
            desc = content[:300] if content else ""

            # Detect source and extract repo_id
            hf_match = re.search(r"huggingface\.co/datasets/([\w.-]+/[\w.-]+)", url)
            ms_match = re.search(r"modelscope\.cn/datasets/([\w.-]+/[\w.-]+)", url)
            kg_match = re.search(r"kaggle\.com/datasets/([\w-]+/[\w-]+)", url)

            if hf_match:
                repo_id = hf_match.group(1)
                results.append(DatasetInfo(
                    name=repo_id,
                    repo_id=repo_id,
                    source="huggingface",
                    url=url,
                    description=desc,
                    download_method="hf",
                ))
            elif ms_match:
                repo_id = ms_match.group(1)
                results.append(DatasetInfo(
                    name=repo_id,
                    repo_id=repo_id,
                    source="modelscope",
                    url=url,
                    description=desc,
                    download_method="modelscope",
                ))
            elif kg_match:
                repo_id = kg_match.group(1)
                results.append(DatasetInfo(
                    name=title or repo_id,
                    repo_id=repo_id,
                    source="kaggle",
                    url=url,
                    description=desc,
                    download_method="kaggle",
                ))
            else:
                results.append(DatasetInfo(
                    name=title[:80],
                    source="web",
                    url=url,
                    description=desc,
                    download_method="manual",
                ))

        return results

    def _search_hf(self, query: str, limit: int = 20) -> list[DatasetInfo]:
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

    def _search_modelscope(self, query: str, limit: int = 10) -> list[DatasetInfo]:
        try:
            from modelscope.hub.api import HubApi
            api = HubApi()
            datasets = api.list_datasets(query=query, page_number=1, page_size=limit) or []
            results = []
            for d in datasets:
                did = getattr(d, "Id", "") or getattr(d, "id", "") or ""
                name = getattr(d, "Name", "") or getattr(d, "name", "") or did
                dl = int(getattr(d, "Downloads", 0) or 0)
                results.append(DatasetInfo(
                    name=name or did,
                    repo_id=did,
                    source="modelscope",
                    url=f"https://modelscope.cn/datasets/{did}",
                    downloads=dl,
                    download_method="modelscope",
                ))
            return results
        except Exception:
            return []

    def _search_kaggle(self, query: str, limit: int = 10) -> list[DatasetInfo]:
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "list", "-s", query, "--csv"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return []
            lines = result.stdout.strip().splitlines()
            out = []
            for line in lines[1:limit + 1]:
                parts = line.split(",")
                if len(parts) >= 2:
                    ref = parts[0].strip()
                    title = parts[1].strip()
                    out.append(DatasetInfo(
                        name=title or ref,
                        repo_id=ref,
                        source="kaggle",
                        url=f"https://www.kaggle.com/datasets/{ref}",
                        download_method="kaggle",
                    ))
            return out
        except Exception:
            return []

    # ══════════════════════════════════════════════════════════
    # Step 2: Interactive display + selection
    # ══════════════════════════════════════════════════════════

    def interactive_select(self, candidates: list[DatasetInfo]) -> list[DatasetInfo]:
        """Display candidates with wrapped descriptions, prompt for selection."""
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            expand=False,
            show_lines=True,
        )

        source_colors = {
            "huggingface": "green",
            "modelscope": "yellow",
            "kaggle": "blue",
            "web": "dim",
        }

        table.add_column("#", style="bold", width=4, no_wrap=True)
        table.add_column("Name / ID", width=38, no_wrap=False)
        table.add_column("Source", width=12, no_wrap=True)
        table.add_column("DL", width=8, justify="right", no_wrap=True)
        table.add_column("Description", width=60, no_wrap=False)

        for i, c in enumerate(candidates, 1):
            color = source_colors.get(c.source, "white")
            dl = f"{c.downloads:,}" if c.downloads > 0 else "-"
            # Wrap description at 60 chars
            desc = c.description.replace("\n", " ").strip()
            table.add_row(
                str(i),
                f"[{color}]{c.name}[/{color}]",
                f"[{color}]{c.source}[/{color}]",
                dl,
                desc,
            )

        console.print(f"\n  Found [bold]{len(candidates)}[/bold] datasets:\n")
        console.print(table)
        console.print()
        console.print("  [bold]Enter numbers to download:[/bold]")
        console.print("  Examples:  [cyan]1,3,5[/cyan]   [cyan]1-5[/cyan]   [cyan]all[/cyan]   or [dim]Enter[/dim] to skip")

        raw = click.prompt("  Select", default="", show_default=False).strip()

        if not raw or raw.lower() in ("q", "quit", "skip", ""):
            return []
        if raw.lower() == "all":
            return candidates

        indices: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if "-" in part and not part.startswith("-"):
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
            click.echo("  No valid selection.")
        return selected

    # ══════════════════════════════════════════════════════════
    # Step 3: Download
    # ══════════════════════════════════════════════════════════

    def download(self, dataset: DatasetInfo) -> str:
        """Download a dataset. Returns local path or empty string if failed/skipped."""
        method = dataset.download_method

        if method == "hf" and dataset.repo_id:
            return self._download_hf(dataset.repo_id)
        elif method == "modelscope" and dataset.repo_id:
            return self._download_modelscope(dataset.repo_id)
        elif method == "kaggle" and dataset.repo_id:
            return self._download_kaggle(dataset.repo_id)
        else:
            click.echo(f"  URL: {dataset.url}")
            click.echo("  This dataset requires manual download.")
            path = click.prompt(
                "  Paste local path after downloading (or Enter to skip)",
                default="", show_default=False,
            ).strip()
            return path if path and os.path.exists(path) else ""

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
            path = click.prompt("  Manual path (or Enter to skip)", default="", show_default=False).strip()
            return path if path and os.path.exists(path) else ""

    def _download_modelscope(self, repo_id: str) -> str:
        dst = os.path.join(self.cache_dir, "downloads", repo_id.replace("/", "__"))
        click.echo(f"  Downloading {repo_id} from ModelScope...")
        try:
            from modelscope.msdatasets import MsDataset
            MsDataset.load(repo_id, cache_dir=dst)
            click.echo(f"  Saved to: {dst}")
            return dst
        except Exception as e:
            click.echo(f"  [FAIL] {e}")
            path = click.prompt("  Manual path (or Enter to skip)", default="", show_default=False).strip()
            return path if path and os.path.exists(path) else ""

    def _download_kaggle(self, ref: str) -> str:
        dst = os.path.join(self.cache_dir, "downloads", ref.replace("/", "__"))
        os.makedirs(dst, exist_ok=True)
        click.echo(f"  Downloading {ref} from Kaggle...")
        try:
            r = subprocess.run(
                ["kaggle", "datasets", "download", ref, "--unzip", "-p", dst],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode == 0:
                click.echo(f"  Saved to: {dst}")
                return dst
            click.echo(f"  [FAIL] {r.stderr[:200]}")
        except Exception as e:
            click.echo(f"  [FAIL] {e}")
        path = click.prompt("  Manual path (or Enter to skip)", default="", show_default=False).strip()
        return path if path and os.path.exists(path) else ""

    # ══════════════════════════════════════════════════════════
    # Step 4: Find + convert raw format files to JSONL
    # ══════════════════════════════════════════════════════════

    def find_data_files(self, path: str) -> list[str]:
        """Find all supported data files under a path."""
        extensions = {".jsonl", ".json", ".csv", ".tsv", ".parquet"}
        p = Path(path)
        if p.is_file() and p.suffix.lower() in extensions:
            return [str(p)]
        files = []
        for root, _, fnames in os.walk(p):
            for fn in fnames:
                fp = Path(fn)
                if fp.suffix.lower() in extensions and not fn.startswith("."):
                    files.append(os.path.join(root, fn))
        return sorted(files)

    def to_jsonl(self, src: str, dst: str) -> int:
        """Convert any supported format (parquet/csv/tsv/json) to JSONL.

        Returns number of rows written.
        """
        suffix = Path(src).suffix.lower()
        rows_written = 0

        try:
            import pandas as pd
        except ImportError:
            click.echo("  [WARN] pandas not installed. Run: pip install pandas pyarrow")
            # Fallback: copy as-is if already JSONL
            if suffix == ".jsonl":
                import shutil
                shutil.copy(src, dst)
                with open(dst) as f:
                    return sum(1 for _ in f)
            return 0

        try:
            if suffix == ".parquet":
                df = pd.read_parquet(src)
            elif suffix == ".csv":
                df = pd.read_csv(src, encoding="utf-8", on_bad_lines="skip")
            elif suffix == ".tsv":
                df = pd.read_csv(src, sep="\t", encoding="utf-8", on_bad_lines="skip")
            elif suffix == ".json":
                # Could be a list or dict
                with open(src, "r", errors="replace") as f:
                    content = json.load(f)
                if isinstance(content, list):
                    df = pd.DataFrame(content)
                elif isinstance(content, dict):
                    # Could be {"data": [...]} or similar
                    for k, v in content.items():
                        if isinstance(v, list) and len(v) > 0:
                            df = pd.DataFrame(v)
                            break
                    else:
                        df = pd.DataFrame([content])
                else:
                    return 0
            elif suffix == ".jsonl":
                # Already JSONL — just copy
                import shutil
                shutil.copy(src, dst)
                with open(dst) as f:
                    return sum(1 for line in f if line.strip())
            else:
                return 0

            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fout:
                for _, row in df.iterrows():
                    # Convert numpy types to native Python
                    record = {}
                    for k, v in row.items():
                        try:
                            if hasattr(v, "item"):
                                v = v.item()
                            elif hasattr(v, "tolist"):
                                v = v.tolist()
                        except Exception:
                            v = str(v)
                        record[str(k)] = v
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows_written += 1

        except Exception as e:
            click.echo(f"  [WARN] Could not convert {Path(src).name}: {e}")
            return 0

        return rows_written

    # ══════════════════════════════════════════════════════════
    # Step 5: Clean JSONL
    # ══════════════════════════════════════════════════════════

    def clean(self, src: str, dst: str) -> dict:
        """Dedup by MD5, remove bad JSON rows, remove empty-content rows, normalize."""
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

                # Must be a dict
                if not isinstance(data, dict):
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
        # erniekit
        if "text_info" in data:
            ti = data["text_info"]
            if not isinstance(ti, list):
                return False
            return not any(isinstance(t, dict) and str(t.get("text", "")).strip() for t in ti)
        # messages
        if "messages" in data:
            msgs = data["messages"]
            if not isinstance(msgs, list):
                return False
            return not any(isinstance(m, dict) and str(m.get("content", "")).strip() for m in msgs)
        # Generic: check if all string values are empty
        all_values = list(data.values())
        text_values = [str(v).strip() for v in all_values if isinstance(v, (str, int, float))]
        return all(not v for v in text_values)

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
    # Step 6: Format detection + conversion
    # ══════════════════════════════════════════════════════════

    def detect_format(self, path: str) -> str:
        from autotrainer.pf_integration.dataset_validator import DatasetValidator
        return DatasetValidator().validate_file(path).format_detected or "unknown"

    def convert_format(self, src: str, dst: str, src_fmt: str, dst_fmt: str) -> dict:
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
                if not isinstance(data, dict):
                    continue

                line_count += 1
                for key in data:
                    field_counts[key] = field_counts.get(key, 0) + 1

                if "text_info" in data:
                    tl = data.get("text_info") or []
                    if isinstance(tl, list):
                        text_lens.append(sum(len(str(t.get("text", ""))) for t in tl if isinstance(t, dict)))
                    imgs = data.get("image_info") or []
                    image_count += len(imgs) if isinstance(imgs, list) else 0
                elif "messages" in data:
                    msgs = data.get("messages") or []
                    if isinstance(msgs, list):
                        text_lens.append(sum(len(str(m.get("content", ""))) for m in msgs if isinstance(m, dict)))
                    imgs = data.get("images") or []
                    image_count += len(imgs) if isinstance(imgs, list) else (1 if imgs else 0)

        result: dict = {
            "path": path,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            "format": fmt,
            "num_samples": line_count,
            "json_errors": json_errors,
            "image_count": image_count,
            "has_images": image_count > 0,
            "field_coverage": (
                {k: round(v / line_count, 2) for k, v in field_counts.items()}
                if line_count else {}
            ),
        }
        if text_lens:
            s = sorted(text_lens)
            n = len(s)
            result["text_lengths"] = {
                "min": s[0], "max": s[-1], "avg": round(sum(s) / n),
                "p50": s[n // 2], "p95": s[int(n * 0.95)] if n > 20 else s[-1],
            }
        samples: list[dict] = []
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    samples.append(json.loads(line.strip()))
                except Exception:
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
        parts: dict = {}
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
        """Full pipeline: search → select → download → to_jsonl → clean → convert → profile → split."""
        if not output_dir:
            output_dir = os.path.join(self.cache_dir, "processed", task)
        os.makedirs(output_dir, exist_ok=True)

        # 1. Search
        click.echo(f"\n[1/8] Searching: {query}")
        candidates = self.search(query, tavily_key=tavily_key, modelscope_token=modelscope_token)
        if not candidates:
            click.echo("  No datasets found.")
            return []

        # 2. Interactive select
        click.echo(f"\n[2/8] Select datasets to download:")
        selected = self.interactive_select(candidates)
        if not selected:
            click.echo("  No selection, exiting.")
            return []

        results: list[PipelineResult] = []

        for ds in selected:
            from rich.console import Console
            Console().print(f"\n[bold]{'═' * 65}[/bold]")
            Console().print(f"  [bold cyan]{ds.name}[/bold cyan]  [{ds.source}]")
            Console().print(f"  {ds.url}")
            Console().print(f"[bold]{'═' * 65}[/bold]")

            res = PipelineResult(dataset_name=ds.name)
            ds_dir = os.path.join(output_dir, re.sub(r"[^\w-]", "_", ds.name)[:60])
            os.makedirs(ds_dir, exist_ok=True)

            # 3. Download
            click.echo(f"\n[3/8] Downloading...")
            dl_path = self.download(ds)
            if not dl_path:
                res.status = "download_failed"
                res.errors.append("Download failed or skipped")
                results.append(res)
                continue

            # 4. Find data files
            click.echo(f"\n[4/8] Finding data files...")
            data_files = self.find_data_files(dl_path)
            names = [Path(f).name for f in data_files]
            click.echo(f"  {len(data_files)} file(s): {names}")
            if not data_files:
                res.status = "no_data_files"
                res.errors.append("No data files found")
                results.append(res)
                continue

            for data_file in data_files:
                fname = Path(data_file).name
                suffix = Path(data_file).suffix.lower()
                click.echo(f"\n  ── {fname} ──")

                # 5. Convert raw format → JSONL (handles parquet/csv/json)
                jsonl_path = os.path.join(ds_dir, f"raw_{Path(fname).stem}.jsonl")
                if suffix != ".jsonl":
                    click.echo(f"  [5/8] Converting {suffix} → JSONL...")
                    n_rows = self.to_jsonl(data_file, jsonl_path)
                    click.echo(f"    {n_rows} rows converted")
                    if n_rows == 0:
                        continue
                else:
                    import shutil
                    shutil.copy(data_file, jsonl_path)
                    click.echo(f"  [5/8] Already JSONL, copied")

                res.raw_path = jsonl_path

                # 6. Clean
                click.echo(f"  [6/8] Cleaning (dedup, bad rows)...")
                cleaned_path = os.path.join(ds_dir, f"cleaned_{Path(fname).stem}.jsonl")
                stats = self.clean(jsonl_path, cleaned_path)
                click.echo(
                    f"    {stats['input_lines']} in → "
                    f"dupes={stats['duplicates']}, "
                    f"bad_json={stats['json_errors']}, "
                    f"empty={stats['empty_content']} → "
                    f"[bold]{stats['output_lines']} out[/bold]"
                )
                res.clean_stats = stats
                res.cleaned_path = cleaned_path

                if stats["output_lines"] == 0:
                    click.echo("    [WARN] No usable rows after cleaning, skipping.")
                    continue

                # 7. Format conversion (e.g. messages → erniekit)
                click.echo(f"  [7/8] Format check...")
                src_fmt = self.detect_format(cleaned_path)
                click.echo(f"    Detected: {src_fmt or 'unknown'}")
                if src_fmt and src_fmt not in ("unknown", target_format):
                    converted_path = os.path.join(ds_dir, f"converted_{Path(fname).stem}.jsonl")
                    conv = self.convert_format(cleaned_path, converted_path, src_fmt, target_format)
                    click.echo(f"    {src_fmt} → {target_format}: {conv.get('converted', 0)} ok, {conv.get('skipped', 0)} skipped")
                    res.converted_path = converted_path
                    final = converted_path
                else:
                    click.echo(f"    No conversion needed ({src_fmt or 'keep as-is'})")
                    final = cleaned_path

                # 8. Profile
                click.echo(f"  [8/8] Profiling...")
                prof = self.profile(final)
                res.profile = prof
                tl = prof.get("text_lengths", {})
                click.echo(f"    samples={prof.get('num_samples', 0)}, size={prof.get('size_mb', 0)}MB, images={prof.get('image_count', 0)}")
                if tl:
                    click.echo(f"    text len: min={tl['min']} avg={tl['avg']} p95={tl['p95']} max={tl['max']}")
                if prof.get("sample_preview"):
                    raw_sample = json.dumps(prof["sample_preview"][0], ensure_ascii=False)
                    click.echo(f"    preview: {raw_sample[:200]}")

                # Split
                click.echo(f"\n  Splitting 90/5/5...")
                split_r = self.split(final)
                res.split_result = split_r
                click.echo(f"    train={split_r['train']['count']} → {split_r['train']['path']}")
                click.echo(f"    val  ={split_r['val']['count']}   val path available")
                click.echo(f"    test ={split_r['test']['count']}  test path available")
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
