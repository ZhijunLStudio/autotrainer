"""Visualize training annotations overlaid on images.

Reads erniekit JSONL format, resolves image paths, and draws
annotation text directly onto the image for verification.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_samples(jsonl_path: str, n: int = 5) -> list[dict]:
    """Load first N samples from a JSONL file."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def resolve_image_path(jsonl_dir: str, image_url: str) -> str:
    """Resolve relative image path to absolute."""
    if image_url.startswith("./"):
        return str(Path(jsonl_dir) / image_url[2:])
    elif not Path(image_url).is_absolute():
        return str(Path(jsonl_dir) / image_url)
    return image_url


def wrap_text(text: str, max_chars: int = 60) -> list[str]:
    """Simple text wrapping."""
    lines = []
    while len(text) > max_chars:
        # Try to break at space
        split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        lines.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        lines.append(text)
    return lines


def draw_annotation(img: Image.Image, sample: dict) -> Image.Image:
    """Draw annotation overlay on image.

    For small images, resize to a minimum height so text is readable.
    """
    w, h = img.size

    # Scale up small images for readability
    min_height = 400
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        img = img.resize((new_w, min_height), Image.LANCZOS)
        w, h = img.size

    draw = ImageDraw.Draw(img)

    # Try to use a font that supports Arabic
    font_size = max(16, min(24, w // 30))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    y_offset = 10
    padding = 8

    # Draw each text_info entry
    for idx, text_item in enumerate(sample.get("text_info", [])):
        text = text_item.get("text", "")
        tag = text_item.get("tag", "unknown")

        # Clean up markdown code blocks
        if text.startswith("```html"):
            text = text.replace("```html\n", "").replace("```", "").strip()
        if text.startswith("```"):
            text = text.replace("```", "").strip()

        # Truncate very long text for visualization
        if len(text) > 300:
            text = text[:300] + "..."

        # Color based on tag
        if tag == "mask":
            bg_color = (52, 119, 235, 200)   # blue for question/mask
            text_color = "white"
            label = "INPUT (mask)"
        elif tag == "no_mask":
            bg_color = (46, 139, 87, 200)    # green for answer
            text_color = "white"
            label = "OUTPUT (no_mask)"
        else:
            bg_color = (128, 128, 128, 200)
            text_color = "white"
            label = tag

        # Wrap text
        max_chars = max(30, w // (font_size // 2))
        wrapped = wrap_text(text, max_chars)

        # Calculate box height
        line_height = font_size + 6
        box_height = line_height * len(wrapped) + padding * 2 + line_height  # +1 line for label

        # Draw semi-transparent background
        overlay = Image.new("RGBA", (w - 20, box_height), bg_color)
        img_rgba = img.convert("RGBA")
        img_rgba.paste(overlay, (10, y_offset), overlay)
        img = img_rgba.convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw label
        draw.text((15, y_offset + 2), label, fill=(255, 255, 100), font=font_bold)

        # Draw text lines
        for line_idx, line in enumerate(wrapped):
            draw.text(
                (15, y_offset + padding + (line_idx + 1) * line_height),
                line,
                fill=text_color,
                font=font,
            )

        y_offset += box_height + 8

    # Draw image info at bottom
    img_info = sample.get("image_info", [])
    if img_info:
        info_text = f"Image: {img_info[0].get('image_url', '?')}"
        draw.text((10, h - 25), info_text, fill=(200, 200, 200), font=font)

    return img


def visualize_dataset(jsonl_path: str, output_dir: str, max_samples: int = 5):
    """Visualize annotations for a dataset."""
    jsonl_dir = str(Path(jsonl_path).parent)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = load_samples(jsonl_path, max_samples)
    if not samples:
        print(f"No samples found in {jsonl_path}")
        return

    print(f"Loaded {len(samples)} samples from {jsonl_path}")

    for i, sample in enumerate(samples):
        img_info = sample.get("image_info", [])
        if not img_info:
            print(f"  Sample {i}: no image_info, skipping")
            continue

        image_url = img_info[0].get("image_url", "")
        image_path = resolve_image_path(jsonl_dir, image_url)

        if not Path(image_path).exists():
            print(f"  Sample {i}: image not found: {image_path}")
            continue

        # Load and annotate
        img = Image.open(image_path).convert("RGB")
        annotated = draw_annotation(img, sample)

        # Save
        out_file = output_path / f"sample_{i:02d}.png"
        annotated.save(str(out_file))

        # Print summary
        texts = sample.get("text_info", [])
        print(f"  Sample {i}: {Path(image_path).name}")
        for t in texts:
            preview = t["text"][:80].replace("\n", " ")
            print(f"    [{t['tag']}] {preview}...")

    print(f"\nSaved to {output_path}/")


if __name__ == "__main__":
    data_dir = "/data-ssd/lizhijun/papers-data/data/arabic_ocr_final"

    # Visualize multiple representative datasets
    datasets_to_show = [
        "NAMAA-Space__Qari-0.1-eval",                                       # Arabic news text
        "Melaraby__EvArEST-dataset-for-Arabic-scene-text-recognition",       # Scene text (mixed lang)
        "Omar-youssef__arabic-ocr-dataset",                                  # HTML OCR
    ]

    for dataset_name in datasets_to_show:
        dataset_path = Path(data_dir) / dataset_name
        if not dataset_path.exists():
            print(f"SKIP: {dataset_name} not found")
            continue

        train_files = sorted(dataset_path.glob("cleaned_*train*.jsonl"))
        if not train_files:
            print(f"SKIP: {dataset_name} no train file")
            continue

        jsonl_file = str(train_files[0])
        output_dir = f"/tmp/annotation_viz/{dataset_name}"
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        visualize_dataset(jsonl_file, output_dir, max_samples=3)
