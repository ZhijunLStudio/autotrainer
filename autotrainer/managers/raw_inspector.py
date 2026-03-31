"""Raw data inspector — samples any format to produce structured metadata for LLM.

Supports: JSONL, JSON, CSV, TSV, Parquet, XML, plain text, directories, ZIP/tar.
Never loads entire file into memory — samples efficiently.
"""

from __future__ import annotations

import json
import os
import zipfile
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InspectionResult:
    """Structured result of inspecting a raw data source."""

    path: str = ""
    format_hint: str = ""          # jsonl / json / csv / parquet / xml / text / directory / zip
    file_count: int = 0
    total_size_mb: float = 0.0
    estimated_rows: int = 0

    # Raw samples for LLM to inspect
    samples: list[Any] = field(default_factory=list)   # up to 20 raw rows

    # File/directory structure
    structure: list[str] = field(default_factory=list)  # top-level entries

    # Inferred schema from samples
    schema: dict = field(default_factory=dict)          # field -> type/example

    # Any issues found during inspection
    issues: list[str] = field(default_factory=list)

    def to_llm_summary(self) -> str:
        """Produce a compact text summary suitable for injection into LLM context."""
        lines = [
            f"Path: {self.path}",
            f"Format: {self.format_hint}",
            f"Files: {self.file_count}, Size: {self.total_size_mb:.1f}MB, Estimated rows: {self.estimated_rows}",
        ]

        if self.structure:
            lines.append(f"Structure (top-level):\n  " + "\n  ".join(self.structure[:20]))

        if self.schema:
            schema_str = json.dumps(self.schema, ensure_ascii=False, indent=2)
            lines.append(f"Schema:\n{schema_str}")

        if self.samples:
            lines.append(f"Samples ({len(self.samples)} rows):")
            for i, s in enumerate(self.samples[:5]):
                row_str = json.dumps(s, ensure_ascii=False)
                if len(row_str) > 400:
                    row_str = row_str[:400] + "..."
                lines.append(f"  [{i}] {row_str}")

        if self.issues:
            lines.append(f"Issues: " + "; ".join(self.issues))

        return "\n".join(lines)


class RawInspector:
    """Inspect raw data files of any format and produce samples + schema."""

    MAX_SAMPLES = 20
    MAX_SAMPLE_BYTES = 500        # truncate individual sample display

    def inspect(self, path: str) -> InspectionResult:
        """Auto-detect format and inspect a file or directory."""
        p = Path(path)
        result = InspectionResult(path=path)

        if not p.exists():
            result.issues.append(f"Path does not exist: {path}")
            return result

        # Directory
        if p.is_dir():
            return self._inspect_directory(p, result)

        # Single file — detect by extension + content sniff
        suffix = p.suffix.lower()
        size_mb = p.stat().st_size / 1024 / 1024
        result.total_size_mb = round(size_mb, 2)
        result.file_count = 1

        if suffix == ".parquet":
            return self._inspect_parquet(p, result)
        elif suffix in (".csv", ".tsv"):
            return self._inspect_csv(p, result, sep="\t" if suffix == ".tsv" else ",")
        elif suffix == ".xml":
            return self._inspect_xml(p, result)
        elif suffix in (".zip",):
            return self._inspect_zip(p, result)
        elif suffix in (".tar", ".gz", ".tgz"):
            return self._inspect_tar(p, result)
        elif suffix == ".jsonl":
            return self._inspect_jsonl(p, result)
        elif suffix == ".json":
            return self._inspect_json(p, result)
        else:
            # Try JSONL first (sniff first line)
            return self._inspect_text_sniff(p, result)

    # ──────────────────────────────────────────────────────────
    # Format-specific inspectors
    # ──────────────────────────────────────────────────────────

    # 只认这些后缀为真实数据文件，排除 log/readme/script 等
    DATA_EXTENSIONS = {".jsonl", ".json", ".csv", ".tsv", ".parquet", ".xml"}
    # 名字里含这些关键词的文件跳过（下载日志、readme、脚本等）
    SKIP_NAME_PATTERNS = {"log", "readme", "summary", "license", "download", ".sh", ".md", ".py", ".txt"}

    def _is_data_file(self, p: Path) -> bool:
        """Return True only if this file looks like actual training data."""
        suffix = p.suffix.lower()
        name_lower = p.name.lower()
        if suffix not in self.DATA_EXTENSIONS:
            return False
        # Skip obvious non-data files by name
        return not any(pat in name_lower for pat in self.SKIP_NAME_PATTERNS)

    def _inspect_directory(self, p: Path, r: InspectionResult) -> InspectionResult:
        r.format_hint = "directory"
        entries = sorted(p.iterdir())
        r.structure = [f.name for f in entries]
        r.file_count = sum(1 for e in entries if e.is_file())

        # Total size (data files only)
        total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        r.total_size_mb = round(total_bytes / 1024 / 1024, 2)

        # Find representative DATA files (exclude logs, readmes, scripts)
        data_files = [
            e for e in p.rglob("*")
            if e.is_file() and self._is_data_file(e)
        ]

        if data_files:
            # Inspect the first data file for samples
            sub = self.inspect(str(data_files[0]))
            r.samples = sub.samples
            r.schema = sub.schema
            r.format_hint = f"directory ({sub.format_hint} inside)"
            r.estimated_rows = sub.estimated_rows
            if len(data_files) > 1:
                r.structure.append(f"... {len(data_files)} data files total")
        else:
            # Look for image files
            img_files = [
                e for e in p.rglob("*")
                if e.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            ]
            if img_files:
                r.format_hint = "directory (images)"
                r.file_count = len(img_files)
                r.samples = [str(f.name) for f in img_files[:10]]

        return r

    def _inspect_jsonl(self, p: Path, r: InspectionResult) -> InspectionResult:
        r.format_hint = "jsonl"
        samples = []
        line_count = 0
        errors = 0

        with open(p, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                if len(samples) < self.MAX_SAMPLES:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        errors += 1

        r.samples = samples
        r.estimated_rows = line_count
        if samples:
            r.schema = self._infer_schema(samples)
        if errors:
            r.issues.append(f"{errors} JSON parse errors in first {self.MAX_SAMPLES} rows")
        return r

    def _inspect_json(self, p: Path, r: InspectionResult) -> InspectionResult:
        r.format_hint = "json"
        try:
            # Read first 200KB only
            with open(p, "r", errors="replace") as f:
                raw = f.read(200_000)

            data = json.loads(raw)

            if isinstance(data, list):
                r.estimated_rows = len(data)
                r.samples = data[:self.MAX_SAMPLES]
            elif isinstance(data, dict):
                # Try common envelope patterns: {"data": [...], "images": [...], "annotations": [...]}
                for key in ("data", "samples", "items", "records", "images", "annotations", "train", "test"):
                    if key in data and isinstance(data[key], list):
                        r.estimated_rows = len(data[key])
                        r.samples = data[key][:self.MAX_SAMPLES]
                        r.schema = {"root_key": key}
                        break
                else:
                    r.samples = [data]
                    r.estimated_rows = 1

            if r.samples:
                r.schema.update(self._infer_schema(
                    r.samples if isinstance(r.samples[0], dict) else [{"value": s} for s in r.samples]
                ))
        except Exception as e:
            r.issues.append(f"JSON parse error: {e}")
        return r

    def _inspect_csv(self, p: Path, r: InspectionResult, sep: str = ",") -> InspectionResult:
        r.format_hint = "tsv" if sep == "\t" else "csv"
        try:
            import pandas as pd
            df = pd.read_csv(p, sep=sep, nrows=self.MAX_SAMPLES, encoding="utf-8", on_bad_lines="skip")
            r.samples = df.to_dict(orient="records")
            r.schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # Estimate total rows
            with open(p, "rb") as f:
                total_bytes = p.stat().st_size
                sample_bytes = f.read(4096)
            avg_row_bytes = max(1, len(sample_bytes) / max(1, sample_bytes.count(b"\n")))
            r.estimated_rows = int(total_bytes / avg_row_bytes)
        except ImportError:
            r.issues.append("pandas not installed, CSV preview unavailable")
        except Exception as e:
            r.issues.append(f"CSV read error: {e}")
        return r

    def _inspect_parquet(self, p: Path, r: InspectionResult) -> InspectionResult:
        r.format_hint = "parquet"
        try:
            import pandas as pd
            df = pd.read_parquet(p)
            r.estimated_rows = len(df)
            r.samples = df.head(self.MAX_SAMPLES).to_dict(orient="records")
            r.schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        except ImportError:
            r.issues.append("pandas/pyarrow not installed, parquet preview unavailable")
        except Exception as e:
            r.issues.append(f"Parquet read error: {e}")
        return r

    def _inspect_xml(self, p: Path, r: InspectionResult) -> InspectionResult:
        r.format_hint = "xml"
        try:
            import xml.etree.ElementTree as ET
            # Read first 100KB to avoid huge files
            with open(p, "r", errors="replace") as f:
                raw = f.read(100_000)

            root = ET.fromstring(raw + ("" if raw.endswith(">") else ">"))

            # Collect structure
            def _describe(elem, depth=0) -> list[str]:
                if depth > 3:
                    return []
                lines = [f"{'  ' * depth}<{elem.tag} {dict(elem.attrib)}>"]
                for child in list(elem)[:3]:
                    lines.extend(_describe(child, depth + 1))
                return lines

            r.structure = _describe(root)

            # Collect leaf samples
            def _collect(elem, samples: list, max_s: int = 20):
                if len(samples) >= max_s:
                    return
                if not list(elem):  # leaf
                    samples.append({
                        "tag": elem.tag,
                        "attrib": dict(elem.attrib),
                        "text": (elem.text or "").strip()[:200],
                    })
                for child in elem:
                    _collect(child, samples, max_s)

            leaf_samples: list = []
            _collect(root, leaf_samples)
            r.samples = leaf_samples[:self.MAX_SAMPLES]

            # Estimate rows by common patterns
            repeating_tags = {}
            for child in root:
                repeating_tags[child.tag] = repeating_tags.get(child.tag, 0) + 1
            if repeating_tags:
                most_common = max(repeating_tags, key=lambda k: repeating_tags[k])
                r.estimated_rows = repeating_tags[most_common]
                r.schema = {"repeating_element": most_common, "sample_attribs": list((root.find(most_common) or root).attrib.keys())}

        except Exception as e:
            # Fallback: show raw text
            r.issues.append(f"XML parse partial: {e}")
            with open(p, "r", errors="replace") as f:
                r.samples = [{"raw_text": f.read(2000)}]
        return r

    def _inspect_zip(self, p: Path, r: InspectionResult) -> InspectionResult:
        r.format_hint = "zip"
        try:
            with zipfile.ZipFile(p, "r") as z:
                names = z.namelist()
                r.structure = names[:50]
                r.file_count = len(names)

                # Find first data file inside
                for name in names:
                    ext = Path(name).suffix.lower()
                    if ext in {".jsonl", ".json", ".csv", ".tsv", ".parquet", ".xml"}:
                        with z.open(name) as f:
                            content = f.read(50_000).decode("utf-8", errors="replace")
                        # Write to temp and inspect
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w") as tmp:
                            tmp.write(content)
                            tmp_path = tmp.name
                        sub = self.inspect(tmp_path)
                        os.unlink(tmp_path)
                        r.samples = sub.samples
                        r.schema = sub.schema
                        r.format_hint = f"zip ({sub.format_hint} inside)"
                        break
        except Exception as e:
            r.issues.append(f"ZIP read error: {e}")
        return r

    def _inspect_tar(self, p: Path, r: InspectionResult) -> InspectionResult:
        r.format_hint = "tar"
        try:
            with tarfile.open(p, "r:*") as t:
                members = t.getmembers()
                r.structure = [m.name for m in members[:50]]
                r.file_count = len(members)
                for m in members:
                    ext = Path(m.name).suffix.lower()
                    if ext in {".jsonl", ".json", ".csv", ".tsv"}:
                        f = t.extractfile(m)
                        if f:
                            content = f.read(50_000).decode("utf-8", errors="replace")
                            r.samples = [{"raw_text": content[:2000]}]
                            r.format_hint = f"tar ({ext} inside)"
                            break
        except Exception as e:
            r.issues.append(f"TAR read error: {e}")
        return r

    def _inspect_text_sniff(self, p: Path, r: InspectionResult) -> InspectionResult:
        """Try to determine format by reading first few lines."""
        try:
            with open(p, "r", errors="replace") as f:
                lines = [f.readline() for _ in range(5)]
            first_non_empty = next((l.strip() for l in lines if l.strip()), "")

            if first_non_empty.startswith("{"):
                # Likely JSONL
                return self._inspect_jsonl(p, r)
            elif first_non_empty.startswith("["):
                # Likely JSON array
                return self._inspect_json(p, r)
            elif "," in first_non_empty:
                return self._inspect_csv(p, r, sep=",")
            elif "\t" in first_non_empty:
                return self._inspect_csv(p, r, sep="\t")
            else:
                r.format_hint = "text"
                with open(p, "r", errors="replace") as f:
                    r.samples = [{"line": l.rstrip()} for l in f.readlines()[:self.MAX_SAMPLES]]
                r.estimated_rows = sum(1 for _ in open(p, errors="replace"))
        except Exception as e:
            r.issues.append(f"Sniff error: {e}")
        return r

    # ──────────────────────────────────────────────────────────
    # Schema inference
    # ──────────────────────────────────────────────────────────

    def _infer_schema(self, samples: list[dict]) -> dict:
        """Infer field names, types, and example values from samples."""
        schema: dict = {}
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            for k, v in sample.items():
                if k not in schema:
                    type_name = type(v).__name__
                    example = v
                    if isinstance(v, str) and len(v) > 80:
                        example = v[:80] + "..."
                    elif isinstance(v, list):
                        type_name = f"list[{type(v[0]).__name__ if v else 'any'}]"
                        example = v[:2]
                    schema[k] = {"type": type_name, "example": example}
        return schema
