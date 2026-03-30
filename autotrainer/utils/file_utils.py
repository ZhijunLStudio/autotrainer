"""Safe file operations with atomic writes and file locking."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import filelock


def atomic_write_json(path: str, data: dict, ensure_ascii: bool = False):
    """Write JSON file atomically using temp file + rename.

    Uses filelock to prevent concurrent writes.
    """
    lock_path = path + ".lock"
    with filelock.FileLock(lock_path, timeout=10):
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)
                f.write("\n")
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


def atomic_write_text(path: str, content: str):
    """Write text file atomically using temp file + rename."""
    lock_path = path + ".lock"
    with filelock.FileLock(lock_path, timeout=10):
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


def safe_read_json(path: str, default: dict | None = None) -> dict | None:
    """Read JSON file, returning default on any error."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def safe_read_text(path: str, default: str = "") -> str:
    """Read text file, returning default on any error."""
    try:
        with open(path, "r") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return default


def ensure_dir(path: str) -> str:
    """Ensure directory exists, return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def read_file_lines(path: str, offset: int = 0, limit: int | None = None) -> list[str]:
    """Read specific lines from a file by offset and limit.

    Useful for reading fragments of large log files without loading
    the entire file into memory.
    """
    lines = []
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if limit is not None and len(lines) >= limit:
                    break
                lines.append(line.rstrip("\n"))
    except (FileNotFoundError, OSError):
        pass
    return lines


def count_file_lines(path: str) -> int:
    """Count lines in a file efficiently."""
    try:
        with open(path, "r") as f:
            return sum(1 for _ in f)
    except (FileNotFoundError, OSError):
        return 0


def get_file_mtime(path: str) -> float | None:
    """Get file modification time, None if not found."""
    try:
        return os.path.getmtime(path)
    except (FileNotFoundError, OSError):
        return None


def list_dir_contents(path: str) -> list[str]:
    """List directory contents, empty list if not found."""
    try:
        return os.listdir(path)
    except (FileNotFoundError, OSError):
        return []
