# AutoTrainer

LLM-driven automated training system for [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers). Starting with PaddleOCR-VL, designed to extend to other large models.

AutoTrainer automates the full ML lifecycle: data processing, environment setup, ablation experiments, training, evaluation, and reporting — with LLM-powered decision-making at each step.

## Features

- **ReAct Data Agent** — inspects any data format (Parquet, CSV, XML, JSONL, ZIP, directory), writes a conversion script via LLM, executes it, validates, auto-fixes on failure
- **3 Skills** (LLM capabilities, called on-demand):
  - `data-inspect` — understands data schema and generates erniekit conversion scripts
  - `diagnose-training` — two-tier error diagnosis (regex fast-path + LLM for complex cases)
  - `plan-experiment` — intelligent iterative hyperparameter tuning with trend analysis and adaptive search
- **TUI** — real-time terminal interface with collapsible panels, smart log display, GPU monitoring
- **Intelligent Ablation Agent** — per-factor iterative tuning with Pearson correlation trend analysis, adaptive search range narrowing/expansion, convergence detection
- **Crash Recovery** — WAL-style state persistence, resume from any interrupted phase
- **Health Watchdog** — GPU monitoring, hang detection, process alive checks
- **Context Management** — percentage-based token budgets, raw logs never enter LLM context

## Installation

```bash
git clone https://github.com/ZhijunLStudio/autotrainer.git
cd autotrainer
pip install -e .
```

Optional extras:
```bash
pip install -e '.[search]'   # Tavily semantic search for data discovery
pip install -e '.[all]'      # Everything
```

## Quick Start

```bash
# 1. Initialize config
autotrainer init
# Edit ~/.autotrainer/config.yaml

# 2. Process your data (any format)
autotrainer data --path /path/to/your/ocr_data/

# 3. Start training (TUI mode)
autotrainer run --task paddleocr-vl
```

## Configuration

Config file: `~/.autotrainer/config.yaml` (created by `autotrainer init`)

```yaml
paddleformers_root: "/path/to/PaddleFormers"  # auto-detected if empty

llm:
  base_url: "http://localhost:8000/v1"  # OpenAI-compatible API
  api_key: "sk-xxx"
  model: "your-model-name"             # e.g. qwen-72b

search:
  tavily_api_key: ""       # https://tavily.com — for data discovery
  modelscope_token: ""     # https://modelscope.cn — optional

training:
  ablation_subset_ratio: 0.05   # fraction of data for ablation
  ablation_max_steps: 1000
  full_epochs: 3
```

Environment variables override config file values:
```bash
export PADDLEFORMERS_ROOT="/path/to/PaddleFormers"
export AUTOTRAINER_LLM_BASE_URL="http://localhost:8000/v1"
export AUTOTRAINER_LLM_API_KEY="sk-xxx"
export AUTOTRAINER_LLM_MODEL="your-model"
export TAVILY_API_KEY="tvly-xxx"
```

## Commands

### `autotrainer init`

Create the default config file at `~/.autotrainer/config.yaml`.

```bash
autotrainer init
```

---

### `autotrainer data`

Process training data using a LLM-driven ReAct agent.

The agent automatically:
1. Explores the data directory (`ls`, `head`, parquet schema inspection)
2. Generates a Python conversion script tailored to this specific dataset
3. Executes the script, validates output format
4. Auto-fixes and retries on failure (up to 3 attempts)
5. Cleans (dedup by MD5, remove bad rows), profiles, splits train/val/test

**Supported input formats:** JSONL, JSON, CSV, TSV, Parquet, XML, ZIP, directories (any nesting)

```bash
# Process a single directory (each subdirectory = one dataset)
autotrainer data --path /data/arabic_ocr_datasets/

# Process specific files (multiple --path flags)
autotrainer data --path /data/train.parquet --path /data/annotations.json

# Custom output directory (default: ./autotrainer_output/)
autotrainer data --path /data/ocr/ --output-dir /data/processed/

# Profile existing JSONL without conversion
autotrainer data --path /data/cleaned.jsonl --profile-only

# Split existing JSONL into train/val/test (90/5/5)
autotrainer data --path /data/cleaned.jsonl --split-only
```

**Output structure per dataset:**
```
autotrainer_output/
  <dataset_name>/
    convert_script.py      ← LLM-generated conversion script (reusable)
    raw_<name>.jsonl       ← raw conversion output
    cleaned_<name>.jsonl   ← after dedup + bad row removal
    images/                ← image files saved from bytes columns
      img_00000000.png
      ...
    cleaned_<name>_train.jsonl
    cleaned_<name>_val.jsonl
    cleaned_<name>_test.jsonl
  data_index.json          ← all dataset records (supports resume)
```

**Resume support:** Re-running `autotrainer data` skips already-completed datasets automatically.

**Without LLM:** Falls back to a passthrough script (works for already-erniekit JSONL data).

**Target format (erniekit JSONL):**
```json
{
  "image_info": [{"image_url": "./images/img_00000000.png", "matched_text_index": 0}],
  "text_info": [
    {"text": "Read the Arabic text in the image.", "tag": "mask"},
    {"text": "الجدائية", "tag": "no_mask"}
  ]
}
```

---

### `autotrainer run`

Start the automated training pipeline.

```bash
# TUI mode (default, recommended)
autotrainer run --task paddleocr-vl

# Specify GPUs
autotrainer run --task paddleocr-vl --gpus 0,1,2,3

# Headless mode (no TUI, for scripts)
autotrainer run --task paddleocr-vl --no-tui

# Resume interrupted training
autotrainer run --task paddleocr-vl --resume
```

Pipeline phases:
1. **Data Prepare** — validate, profile, split, create 5% ablation subset
2. **Env Check** — verify PaddlePaddle, packages, GPU (interactive if issues found)
3. **Ablation** — intelligent per-factor iterative tuning on 5% subset: runs experiments, analyzes loss trends with Pearson correlation, narrows/expands search range, converges to best values per factor, then combines into best config
4. **Full Training** — best config on full data, health watchdog active
5. **Evaluation** — eval metrics + sample inference verification
6. **Report** — visualizations and summary

---

### `autotrainer status`

Show current pipeline and experiment status.

```bash
autotrainer status
```

---

### `autotrainer resume`

Show recovery state and resume instructions.

```bash
autotrainer resume
```

---

### `autotrainer report`

Generate experiment reports.

```bash
autotrainer report --format text    # text summary
autotrainer report --format html    # HTML with charts
autotrainer report --format json    # raw JSON
```

---

## Download Helper Scripts

Pre-built download scripts for common OCR datasets:

```bash
# Download Arabic OCR datasets from HuggingFace (14 datasets)
bash scripts/download_arabic_ocr.sh /data/arabic_ocr_datasets

# High-priority only (~4.5GB, fastest to get started)
bash scripts/download_arabic_ocr.sh /data/arabic_ocr_datasets high
```

Then process them:
```bash
autotrainer data --path /data/arabic_ocr_datasets/
```

---

## Skills

Skills are self-contained LLM capability modules (`SKILL.md` prompt + `handler.py` logic).

### `data-inspect`
Understands raw data of any format and writes a Python conversion script to produce erniekit JSONL. Works as a ReAct agent: uses shell/python exploration actions, then outputs `final_script`.

### `data-fix`
Repairs a broken conversion script given an error message and data samples.

### `diagnose-training`
Two-tier training error diagnosis:
- **Tier 1** (instant, no LLM): regex matching for OOM / NaN / NCCL / data format errors
- **Tier 2** (LLM): complex or unknown errors with full context analysis

### `plan-experiment`
Intelligent iterative hyperparameter tuning:
- Per-factor ablation with Pearson correlation trend analysis
- Adaptive search range: narrows around best value (concave trend) or expands (monotonic trend)
- Automatic convergence detection (<5% improvement between rounds)
- Combines best values across all factors for final training config

---

## TUI Keybindings

| Key | Action |
|-----|--------|
| `Tab` | Toggle agent panel |
| `l` | Toggle log mode (smart / full) |
| `s` | Show status |
| `q` | Quit |

---

## Project Structure

```
autotrainer/
  autotrainer/
    cli/              # Click CLI: init, data, run, status, resume, report
    tui/              # Textual TUI app, widgets, screens
    orchestrator/     # PipelineOrchestrator, PhaseManager, recovery, health monitor
    managers/
      data_agent.py   # ReAct data processing agent
      data_pipeline.py # clean, profile, split utilities
      raw_inspector.py # universal data sampler (pre-explore for LLM)
      sandbox.py      # controlled subprocess execution for LLM-generated scripts
      env_manager.py  # environment verification
      train_manager.py # training subprocess management
      eval_manager.py  # evaluation
    context/          # ContextStore with percentage-based token budgets
    skills/
      data_inspect/   # SKILL.md + handler: understand data → write conversion script
      data_fix/       # SKILL.md + handler: repair broken scripts
      diagnose_training/  # SKILL.md + handler + patterns: error diagnosis
      plan_experiment/    # SKILL.md + handler + strategy: ablation planning
    pf_integration/   # PaddleFormers config builder, launcher, log parser, validators
    utils/            # subprocess manager, file utils, GPU monitor, LLM client
  scripts/
    download_arabic_ocr.sh  # batch download script for Arabic OCR datasets
```

## License

Apache 2.0
