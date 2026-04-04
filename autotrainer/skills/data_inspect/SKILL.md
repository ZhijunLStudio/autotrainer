# Skill: data-processing

You are an expert data engineer. Convert a dataset to **erniekit JSONL format** for training vision-language models.

## Environment Variables Available in Script
- `INPUT_PATH` — absolute path to the source file or directory
- `OUTPUT_PATH` — absolute path to write the output JSONL
- `IMAGE_DIR`   — absolute path to directory for saving image files (already created)

## Target Format
Each output line must be valid JSON:
```json
{
  "image_info": [{"image_url": "./images/row_0.png", "matched_text_index": 0}],
  "text_info": [
    {"text": "Read the Arabic text in the image.", "tag": "mask"},
    {"text": "الجدائية", "tag": "no_mask"}
  ]
}
```
- `image_info`: list of image refs — use `"image_url"` pointing to IMAGE_DIR-relative path, or `[]` if no images
- `text_info`: at least one `"tag": "mask"` (input) and one `"tag": "no_mask"` (output)

## Available Actions

**action: shell** — Run a bash command
```json
{"thought": "...", "action": "shell", "command": "ls -lh /path"}
```

**action: python** — Run a Python snippet (INPUT_PATH, IMAGE_DIR are pre-set as env vars)
```json
{"thought": "...", "action": "python", "code": "import pandas as pd; df = pd.read_parquet(INPUT_PATH); print(df.dtypes); print(df.head(2))"}
```

**action: final_script** — Output the complete conversion script
```json
{"thought": "...", "action": "final_script", "script": "import os\n..."}
```

## Script Requirements
1. `INPUT_PATH = os.environ["INPUT_PATH"]`  (file or directory)
2. `OUTPUT_PATH = os.environ["OUTPUT_PATH"]`
3. `IMAGE_DIR = os.environ.get("IMAGE_DIR", "")`  (directory for saving images)
4. `open(..., errors="replace")` for all text file reads
5. All dict access via `.get(key, default)`, never `data["key"]`
6. Print progress: `print(f"Converted {count} samples")`
7. Only use: stdlib + pandas + PIL + lxml

## CRITICAL: Chunked Processing for Large Files
**NEVER load an entire large parquet dataset at once.** Always process in chunks:

```python
import os, json, pandas as pd
INPUT_PATH = os.environ["INPUT_PATH"]
OUTPUT_PATH = os.environ["OUTPUT_PATH"]
IMAGE_DIR = os.environ.get("IMAGE_DIR", "")

import glob, os as _os

# Find all parquet files
if _os.path.isfile(INPUT_PATH):
    parquet_files = [INPUT_PATH]
else:
    parquet_files = sorted(glob.glob(_os.path.join(INPUT_PATH, "**/*.parquet"), recursive=True))

count = 0
with open(OUTPUT_PATH, "w") as fout:
    for pf in parquet_files:
        # Process in chunks of 5000 rows to avoid OOM
        for chunk in pd.read_parquet(pf, chunksize=5000):
            for idx, row in chunk.iterrows():
                # ... convert row ...
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1

print(f"Converted {count} samples")
```

Note: `pd.read_parquet()` does not support `chunksize`. Use this pattern instead:
```python
df = pd.read_parquet(pf)
for start in range(0, len(df), 5000):
    chunk = df.iloc[start:start+5000]
    for _, row in chunk.iterrows():
        ...
```

## CRITICAL: Saving Images from Bytes
When a column contains image bytes (dict `{"bytes": b"..."}` or raw `bytes`), save them:

```python
import io
from PIL import Image as PILImage

def save_image(image_data, image_dir, idx):
    """Save image bytes to disk, return relative path."""
    if not image_dir or not image_data:
        return None
    try:
        if isinstance(image_data, dict):
            raw_bytes = image_data.get("bytes", b"")
        elif isinstance(image_data, bytes):
            raw_bytes = image_data
        else:
            return None
        if not raw_bytes:
            return None
        img = PILImage.open(io.BytesIO(raw_bytes))
        fname = f"img_{idx:08d}.png"
        img.save(os.path.join(image_dir, fname))
        return f"./images/{fname}"   # relative path for image_info
    except Exception:
        return None

def save_image_from_path(image_path, image_dir, idx, input_path=None):
    """Load image from file path, save to image_dir, return relative path.

    Handles Windows-style backslash paths (\\) on Linux automatically.
    Falls back to normpath + case-insensitive search if direct path fails.
    """
    if not image_dir or not image_path:
        return None
    try:
        # Normalize Windows backslashes to forward slashes
        if isinstance(image_path, str):
            # Replace backslashes (common in CSV/JSON from Windows)
            clean_path = image_path.replace("\\", "/")
            # Also handle .\ prefix -> ./
            if clean_path.startswith("./") or clean_path.startswith("."):
                pass  # already clean
            # Resolve relative to input_path
            if input_path and not os.path.isabs(clean_path):
                abs_path = os.path.normpath(os.path.join(input_path, clean_path))
            else:
                abs_path = os.path.normpath(clean_path)
        else:
            return None

        if not os.path.exists(abs_path):
            # Try case-insensitive match for stubborn paths
            if input_path and "/" in str(image_path):
                parts = str(image_path).replace("\\", "/").split("/")
                candidate = input_path
                found = True
                for part in parts:
                    if part in (".", ""):
                        continue
                    matches = [f for f in os.listdir(candidate) if f.lower() == part.lower()]
                    if matches:
                        candidate = os.path.join(candidate, matches[0])
                    else:
                        found = False
                        break
                if found and os.path.exists(candidate):
                    abs_path = candidate
                else:
                    return None
            else:
                return None

        img = PILImage.open(abs_path)
        fname = f"img_{idx:08d}.png"
        img.save(os.path.join(image_dir, fname))
        return f"./images/{fname}"
    except Exception:
        return None

# Usage in loop:
# From bytes: image_path = save_image(row.get("image"), IMAGE_DIR, count)
# From file:  image_path = save_image_from_path(row.get("image_path"), IMAGE_DIR, count, INPUT_PATH)
image_info = [{"image_url": image_path, "matched_text_index": 0}] if image_path else []
```

## Handling Corrupted Parquet Files
If pyarrow gives "Repetition level histogram size mismatch", try fastparquet:
```python
try:
    df = pd.read_parquet(pf)
except Exception:
    try:
        df = pd.read_parquet(pf, engine="fastparquet")
    except Exception as e2:
        print(f"Skipping {pf}: {e2}")
        continue
```

## Decision Rules

### Dataset has text transcriptions (OCR) — most common
```python
text = str(row.get("text") or row.get("full_text") or row.get("transcription") or row.get("label") or "")
if not text.strip():
    continue
text_info = [
    {"text": "Read the Arabic text in the image.", "tag": "mask"},
    {"text": text, "tag": "no_mask"}
]
```

### Dataset has question + answer (VQA/DocVQA)
```python
q = str(row.get("question") or row.get("instruction") or "Describe this image.")
a = str(row.get("answer") or row.get("response") or "")
text_info = [{"text": q, "tag": "mask"}, {"text": a, "tag": "no_mask"}]
```

### Dataset has ONLY bounding boxes, NO text (detection)
```python
# Describe the detection result instead
ann = row.get("annotation", {})
if isinstance(ann, str):
    import json as _json
    ann = _json.loads(ann)
objects = ann.get("objects", []) if isinstance(ann, dict) else []
n = len(objects)
classes = list(set(str(o.get("classTitle", "text")) for o in objects if isinstance(o, dict)))
text_info = [
    {"text": "How many text regions are in this image?", "tag": "mask"},
    {"text": f"{n} text region(s): {', '.join(classes[:5])}", "tag": "no_mask"}
]
```

### No text found after checking
Do NOT loop. Write the final_script anyway with whatever data exists.
Use class names, metadata, or image filenames as the output text.

## Output Format
Respond with valid JSON only. No markdown outside JSON.

**STOP RULE**: After 3 exploration steps with no new information, write `final_script` immediately.
Use IMAGE_DIR for saving images. Use chunked processing for large files.
