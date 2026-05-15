"""Dataset validation for PaddleFormers expected formats.

Validates JSONL data against erniekit and messages formats
without importing PaddleFormers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of a dataset validation check."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    samples_checked: int = 0
    format_detected: str = ""


class DatasetValidator:
    """Validates JSONL data against PaddleFormers expected formats."""

    # Required fields per format
    ERNIEKIT_REQUIRED = {"image_info", "text_info"}
    MESSAGES_REQUIRED = {"messages"}

    def validate_file(self, path: str) -> ValidationResult:
        """Auto-detect format and validate a JSONL file."""
        p = Path(path)
        if not p.exists():
            return ValidationResult(valid=False, errors=[f"File not found: {path}"])

        errors = []
        warnings = []

        # Read first few lines to detect format
        lines = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                lines.append(line.strip())

        if not lines:
            return ValidationResult(valid=False, errors=["File is empty"], warnings=[])

        # Detect format
        fmt = self._detect_format(lines)
        if not fmt:
            return ValidationResult(valid=False, errors=["Cannot detect data format"], warnings=[])

        if fmt == "erniekit":
            return self._validate_erniekit(path)
        elif fmt == "messages":
            return self._validate_messages(path)
        else:
            return ValidationResult(valid=False, errors=[f"Unknown format: {fmt}"], warnings=[])

    def _detect_format(self, lines: list[str]) -> str | None:
        """Detect data format from sample lines."""
        format_votes = {"erniekit": 0, "messages": 0}

        for line in lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if self.ERNIEKIT_REQUIRED.issubset(data.keys()):
                format_votes["erniekit"] += 1
            elif self.MESSAGES_REQUIRED.issubset(data.keys()):
                format_votes["messages"] += 1

        if format_votes["erniekit"] > 0:
            return "erniekit"
        elif format_votes["messages"] > 0:
            return "messages"
        return None

    def _validate_erniekit(self, path: str) -> ValidationResult:
        """Validate erniekit JSONL format."""
        errors = []
        warnings = []
        line_count = 0

        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                line_count += 1

                if line_count > 10_000 and line_count % 1000 == 0:
                    continue  # Skip detailed checks after 10K lines

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON: {e}")
                    if len(errors) >= 10:
                        break
                    continue

                # Check required fields
                for field in self.ERNIEKIT_REQUIRED:
                    if field not in data:
                        errors.append(f"Line {i}: Missing required field '{field}'")
                        break

                # Check image_info structure
                if "image_info" in data:
                    if not isinstance(data["image_info"], list):
                        errors.append(f"Line {i}: 'image_info' must be a list")

                # Check text_info structure
                if "text_info" in data:
                    if not isinstance(data["text_info"], list):
                        errors.append(f"Line {i}: 'text_info' must be a list")
                    elif len(data["text_info"]) == 0:
                        warnings.append(f"Line {i}: 'text_info' is empty")

                if len(errors) >= 10:
                    break

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            samples_checked=min(line_count, 10_000),
            format_detected="erniekit",
        )

    def _validate_messages(self, path: str) -> ValidationResult:
        """Validate messages JSONL format."""
        errors = []
        warnings = []
        line_count = 0

        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                line_count += 1

                if line_count > 10_000 and line_count % 1000 == 0:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON: {e}")
                    if len(errors) >= 10:
                        break
                    continue

                if "messages" not in data:
                    errors.append(f"Line {i}: Missing 'messages' field")
                    continue

                msgs = data["messages"]
                if not isinstance(msgs, list):
                    errors.append(f"Line {i}: 'messages' must be a list")
                    continue

                for j, msg in enumerate(msgs):
                    if not isinstance(msg, dict):
                        errors.append(f"Line {i}: message[{j}] must be a dict")
                        break
                    if "role" not in msg:
                        errors.append(f"Line {i}: message[{j}] missing 'role'")
                    if "content" not in msg:
                        errors.append(f"Line {i}: message[{j}] missing 'content'")

                if len(errors) >= 10:
                    break

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            samples_checked=min(line_count, 10_000),
            format_detected="messages",
        )

    def check_image_references(self, path: str, base_dir: str) -> list[str]:
        """Check that image paths referenced in the data exist on disk."""
        missing = []
        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                if i > 10_000:
                    break
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                images = data.get("image_info", [])
                if isinstance(images, list):
                    for img in images:
                        if isinstance(img, dict):
                            img_path = img.get("image_url", "")
                            if img_path and img_path.startswith("./"):
                                full_path = str(Path(base_dir) / img_path)
                                if not Path(full_path).exists():
                                    missing.append(f"Line {i}: {img_path}")

                if len(missing) >= 20:
                    break

        return missing

    def sample_and_preview(self, path: str, n: int = 3) -> list[dict]:
        """Read N random samples from the file for preview."""
        import random

        # Count lines first
        total_lines = 0
        with open(path, "r") as f:
            for _ in f:
                total_lines += 1

        if total_lines == 0:
            return []

        # Pick random line indices
        sample_count = min(n, total_lines)
        indices = sorted(random.sample(range(total_lines), sample_count))

        # Read samples
        samples = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if indices and i == indices[0]:
                    indices.pop(0)
                    try:
                        samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
                if not indices:
                    break

        return samples
