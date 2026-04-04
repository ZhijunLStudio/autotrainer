"""Data pipeline utilities — clean, profile, split, to_jsonl.

These are format-independent operations on JSONL files.
The search and download functionality has been removed.
Format conversion is now handled by DataAgent (LLM-generated scripts).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetProfile:
    path: str = ""
    format: str = ""
    num_samples: int = 0
    has_images: bool = False
    image_count: int = 0
    text_lengths: dict = field(default_factory=dict)
    field_coverage: dict = field(default_factory=dict)
    sample_preview: list = field(default_factory=list)
    quality_flags: list = field(default_factory=list)
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


class DataPipeline:
    """Standard data utilities: to_jsonl, clean, profile, split.

    These operate on JSONL and are reused by DataAgent after
    LLM-generated conversion scripts produce raw JSONL.
    """

    def __init__(self, cache_dir: str = "", paddleformers_root: str = ""):
        self.cache_dir = cache_dir
        self.pf_root = paddleformers_root
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Convert any tabular format to JSONL via pandas
    # ──────────────────────────────────────────────────────────

    def to_jsonl(self, src: str, dst: str) -> int:
        """Convert parquet/csv/tsv/json → JSONL. Returns row count."""
        import shutil
        suffix = Path(src).suffix.lower()
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

        if suffix == ".jsonl":
            shutil.copy(src, dst)
            with open(dst) as f:
                return sum(1 for line in f if line.strip())

        try:
            import pandas as pd
        except ImportError:
            return 0

        try:
            if suffix == ".parquet":
                df = pd.read_parquet(src)
            elif suffix == ".csv":
                df = pd.read_csv(src, encoding="utf-8", on_bad_lines="skip")
            elif suffix == ".tsv":
                df = pd.read_csv(src, sep="\t", encoding="utf-8", on_bad_lines="skip")
            elif suffix == ".json":
                with open(src, "r", errors="replace") as f:
                    content = json.load(f)
                if isinstance(content, list):
                    df = pd.DataFrame(content)
                elif isinstance(content, dict):
                    for k, v in content.items():
                        if isinstance(v, list):
                            df = pd.DataFrame(v)
                            break
                    else:
                        df = pd.DataFrame([content])
                else:
                    return 0
            else:
                return 0

            count = 0
            with open(dst, "w", encoding="utf-8") as fout:
                for _, row in df.iterrows():
                    record = {}
                    for k, v in row.items():
                        try:
                            v = v.item() if hasattr(v, "item") else (v.tolist() if hasattr(v, "tolist") else v)
                        except Exception:
                            v = str(v)
                        record[str(k)] = v
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            return count
        except Exception:
            return 0

    # ──────────────────────────────────────────────────────────
    # Clean
    # ──────────────────────────────────────────────────────────

    def clean(self, src: str, dst: str) -> dict:
        """Deduplicate by MD5, remove bad JSON, empty content, normalize whitespace."""
        stats = {
            "input_lines": 0, "valid_lines": 0,
            "json_errors": 0, "duplicates": 0,
            "empty_content": 0, "output_lines": 0,
        }
        seen: set[str] = set()
        cleaned: list[str] = []

        with open(src, "r", errors="replace") as fin:
            for line in fin:
                stats["input_lines"] += 1
                s = line.strip()
                if not s:
                    continue
                try:
                    data = json.loads(s)
                except json.JSONDecodeError:
                    stats["json_errors"] += 1
                    continue
                if not isinstance(data, dict):
                    stats["json_errors"] += 1
                    continue

                h = hashlib.md5(s.encode()).hexdigest()
                if h in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(h)

                if self._is_empty(data):
                    stats["empty_content"] += 1
                    continue

                cleaned.append(json.dumps(self._normalize(data), ensure_ascii=False) + "\n")
                stats["valid_lines"] += 1

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w") as fout:
            fout.writelines(cleaned)
        stats["output_lines"] = len(cleaned)
        return stats

    def _is_empty(self, data: dict) -> bool:
        if "text_info" in data:
            ti = data["text_info"]
            return isinstance(ti, list) and not any(
                isinstance(t, dict) and str(t.get("text", "")).strip() for t in ti
            )
        if "messages" in data:
            ms = data["messages"]
            return isinstance(ms, list) and not any(
                isinstance(m, dict) and str(m.get("content", "")).strip() for m in ms
            )
        return all(not str(v).strip() for v in data.values() if isinstance(v, (str, int, float)))

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

    # ──────────────────────────────────────────────────────────
    # Profile
    # ──────────────────────────────────────────────────────────

    def detect_format(self, path: str) -> str:
        from autotrainer.pf_integration.dataset_validator import DatasetValidator
        return DatasetValidator().validate_file(path).format_detected or "unknown"

    def profile(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {"error": "file not found"}

        fmt = self.detect_format(path)
        text_lens: list[int] = []
        field_counts: dict[str, int] = {}
        json_errors = line_count = image_count = 0

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
                    ms = data.get("messages") or []
                    if isinstance(ms, list):
                        text_lens.append(sum(len(str(m.get("content", ""))) for m in ms if isinstance(m, dict)))
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
                {k: round(v / line_count, 2) for k, v in field_counts.items()} if line_count else {}
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

    # ──────────────────────────────────────────────────────────
    # Split
    # ──────────────────────────────────────────────────────────

    def split(
        self,
        path: str,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> dict:
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
            pp = str(base.parent / f"{base.stem}_{name}.jsonl")
            with open(pp, "w") as f:
                f.writelines(lines[start:end])
            parts[name] = {"path": pp, "count": end - start}
        parts["total"] = n
        return parts

    # ──────────────────────────────────────────────────────────
    # Merge DataAgent output into ready-to-train files
    # ──────────────────────────────────────────────────────────

    def merge_from_index(
        self,
        data_dir: str,
        output_dir: str | None = None,
    ) -> dict:
        """Read data_index.json produced by DataAgent and merge all completed
        datasets' train/val JSONL files into single merged files.

        Returns:
            {
              "train": {"path": "...", "count": N},
              "val":   {"path": "...", "count": N},
              "datasets": [{"name": ..., "train_count": ..., "val_count": ...}, ...],
              "total_train": N,
              "total_val": N,
            }
        """
        index_path = os.path.join(data_dir, "data_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"data_index.json not found in {data_dir}. "
                "Run `autotrainer data --path <dataset> --output-dir <dir>` first."
            )

        with open(index_path, "r", errors="replace") as f:
            index = json.load(f)

        datasets = [d for d in index.get("datasets", []) if d.get("status") == "completed"]
        if not datasets:
            raise RuntimeError(
                f"No completed datasets found in {index_path}. "
                "Check that `autotrainer data` ran successfully."
            )

        out_dir = output_dir or data_dir
        os.makedirs(out_dir, exist_ok=True)
        merged_train = os.path.join(out_dir, "merged_train.jsonl")
        merged_val = os.path.join(out_dir, "merged_val.jsonl")

        train_count = 0
        val_count = 0
        dataset_stats = []

        def _rewrite_image_paths(line: str, src_dir: str, out_dir: str) -> str:
            """Rewrite relative ./images/ paths to absolute paths in a JSONL line."""
            if "./images/" not in line:
                return line
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return line
            img_info = data.get("image_info")
            if isinstance(img_info, list):
                for img in img_info:
                    if isinstance(img, dict):
                        url = img.get("image_url", "")
                        if url.startswith("./images/"):
                            img["image_url"] = os.path.join(src_dir, url[2:])
                        elif url.startswith("./"):
                            img["image_url"] = os.path.join(src_dir, url[2:])
            return json.dumps(data, ensure_ascii=False) + "\n"

        with open(merged_train, "w", encoding="utf-8") as f_train, \
             open(merged_val, "w", encoding="utf-8") as f_val:

            for ds in datasets:
                ds_train_count = 0
                ds_val_count = 0

                train_path = ds.get("split", {}).get("train", {}).get("path", "")
                val_path = ds.get("split", {}).get("val", {}).get("path", "")

                if train_path and os.path.exists(train_path):
                    src_dir = os.path.dirname(train_path)

                    # Symlink images directory into the merged output dir
                    # (backup for code that resolves relative paths)
                    source_images_dir = os.path.join(src_dir, "images")
                    if os.path.isdir(source_images_dir) and not os.path.isdir(os.path.join(out_dir, "images")):
                        try:
                            os.symlink(source_images_dir, os.path.join(out_dir, "images"), target_is_directory=True)
                        except OSError:
                            pass

                    with open(train_path, "r", errors="replace") as src:
                        for line in src:
                            if line.strip():
                                line = _rewrite_image_paths(line, src_dir, out_dir)
                                f_train.write(line if line.endswith("\n") else line + "\n")
                                ds_train_count += 1
                    train_count += ds_train_count

                if val_path and os.path.exists(val_path):
                    with open(val_path, "r", errors="replace") as src:
                        for line in src:
                            if line.strip():
                                f_val.write(line if line.endswith("\n") else line + "\n")
                                ds_val_count += 1
                    val_count += ds_val_count

                dataset_stats.append({
                    "name": ds.get("dataset_name", ""),
                    "train_count": ds_train_count,
                    "val_count": ds_val_count,
                    "source": ds.get("source_path", ""),
                })

        # Remove empty val file if nothing was written
        if val_count == 0 and os.path.exists(merged_val):
            os.unlink(merged_val)
            merged_val = ""

        return {
            "train": {"path": merged_train, "count": train_count},
            "val": {"path": merged_val, "count": val_count},
            "datasets": dataset_stats,
            "total_train": train_count,
            "total_val": val_count,
        }

    # ──────────────────────────────────────────────────────────
    # Subset creation (for ablation)
    # ──────────────────────────────────────────────────────────

    def create_subset(self, src: str, dst: str, ratio: float, seed: int = 42) -> dict:
        """Create a random subset of a JSONL file.

        If the source file contains relative image paths, they are rewritten
        to absolute paths (relative to the source file's directory) so the
        subset can be used from any working directory.
        """
        random.seed(seed)
        src_dir = os.path.dirname(src)
        with open(src, "r") as f:
            lines = f.readlines()
        subset_size = max(1, int(len(lines) * ratio))
        sampled = random.sample(lines, subset_size)

        # Rewrite relative image paths to absolute
        rewritten = []
        for line in sampled:
            line = line.strip()
            if not line:
                continue
            if "./images/" in line or "./" in line:
                try:
                    data = json.loads(line)
                    img_info = data.get("image_info")
                    if isinstance(img_info, list):
                        for img in img_info:
                            if isinstance(img, dict):
                                url = img.get("image_url", "")
                                if url.startswith("./"):
                                    img["image_url"] = os.path.join(src_dir, url[2:])
                    line = json.dumps(data, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            rewritten.append(line + "\n")

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w") as f:
            f.writelines(rewritten)
        return {"total": len(lines), "subset": len(rewritten), "ratio": ratio, "path": dst}
