"""Dataset format conversion between different data formats.

Supports conversion to/from PaddleFormers erniekit format (messages JSONL).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotrainer.utils.file_utils import atomic_write_text


# Registry of format converters: (source_format, target_format) -> converter_function
_CONVERTERS: dict[tuple[str, str], Any] = {}


def register_converter(src_fmt: str, dst_fmt: str):
    """Decorator to register a format converter."""

    def decorator(func):
        _CONVERTERS[(src_fmt, dst_fmt)] = func
        return func

    return decorator


def get_available_conversions() -> list[tuple[str, str]]:
    """List all available format conversions."""
    return list(_CONVERTERS.keys())


def convert(
    src_path: str,
    dst_path: str,
    src_fmt: str,
    dst_fmt: str,
    base_dir: str = "",
) -> dict:
    """Convert a dataset from one format to another.

    Returns conversion stats.
    """
    key = (src_fmt, dst_fmt)
    if key not in _CONVERTERS:
        available = get_available_conversions()
        raise ValueError(f"No converter for {src_fmt} -> {dst_fmt}. Available: {available}")

    return _CONVERTERS[key](src_path, dst_path, base_dir)


# --- Built-in converters ---


@register_converter("messages", "erniekit")
def messages_to_erniekit(src: str, dst: str, base_dir: str = "") -> dict:
    """Convert messages JSONL to erniekit format.

    Messages format:
        {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}], "images": [...]}

    Erniekit format:
        {"image_info": [...], "text_info": [{"text":"...", "tag":"mask"}, ...]}
    """
    converted = 0
    skipped = 0

    with open(src, "r") as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            messages = data.get("messages", [])
            images = data.get("images", [])

            image_info = []
            for i, img in enumerate(images):
                if isinstance(img, str):
                    image_info.append({"image_url": img, "matched_text_index": 0})
                elif isinstance(img, dict):
                    image_info.append(img)

            text_info = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                tag = "mask" if role == "user" else "no_mask"
                text_info.append({"text": content, "tag": tag})

            out = {"image_info": image_info, "text_info": text_info}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            converted += 1

    return {"converted": converted, "skipped": skipped, "format": "erniekit"}


@register_converter("erniekit", "messages")
def erniekit_to_messages(src: str, dst: str, base_dir: str = "") -> dict:
    """Convert erniekit format to messages JSONL."""
    converted = 0
    skipped = 0

    with open(src, "r") as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            text_info = data.get("text_info", [])
            images = []
            for img in data.get("image_info", []):
                url = img.get("image_url", "")
                if url:
                    images.append(url)

            messages = []
            for item in text_info:
                text = item.get("text", "")
                tag = item.get("tag", "no_mask")
                role = "user" if tag == "mask" else "assistant"
                messages.append({"role": role, "content": text})

            out = {"messages": messages}
            if images:
                out["images"] = images

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            converted += 1

    return {"converted": converted, "skipped": skipped, "format": "messages"}


@register_converter("csv_qa", "erniekit")
def csv_qa_to_erniekit(src: str, dst: str, base_dir: str = "") -> dict:
    """Convert CSV (question, answer) to erniekit format.

    Expects CSV with columns: question, answer [, image_path]
    """
    import csv

    converted = 0
    skipped = 0

    with open(src, "r") as fin:
        reader = csv.DictReader(fin)
        with open(dst, "w") as fout:
            for row in reader:
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()
                image_path = row.get("image_path", "").strip()

                if not question or not answer:
                    skipped += 1
                    continue

                image_info = []
                if image_path:
                    image_info.append({"image_url": image_path, "matched_text_index": 0})

                text_info = [
                    {"text": question, "tag": "mask"},
                    {"text": answer, "tag": "no_mask"},
                ]

                out = {"image_info": image_info, "text_info": text_info}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                converted += 1

    return {"converted": converted, "skipped": skipped, "format": "erniekit"}
