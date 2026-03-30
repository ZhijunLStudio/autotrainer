# AutoTrainer

LLM-driven automated training system for [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers). Starting with PaddleOCR-VL, designed to extend to other large models.

AutoTrainer automates the full training lifecycle: environment setup, data search/download/format conversion, ablation experiments, hyperparameter tuning, training, evaluation, and reporting — with LLM-powered decision-making at each step.

## Features

- **3 Skills** (LLM capabilities, called on-demand):
  - `diagnose-training` — analyzes training errors (OOM, NaN, NCCL, etc.) and suggests fixes
  - `plan-experiment` — plans ablation experiments and next-step parameter tuning
  - `data-intel` — searches, validates, profiles, and converts training datasets
- **TUI** — real-time terminal interface with collapsible panels, smart log display, GPU monitoring
- **Ablation Engine** — single-factor → multi-factor → full training, all on small subsets first
- **Crash Recovery** — WAL-style state persistence, can resume from any interrupted phase
- **Health Watchdog** — background monitoring of GPU, process, log freshness, hang detection
- **Context Management** — percentage-based token budgets, raw logs never enter LLM context

## Installation

```bash
git clone https://github.com/ZhijunLStudio/autotrainer.git
cd autotrainer

# Basic install (HF search + validate + train)
pip install -e .

# With Tavily semantic search
pip install -e '.[search]'

# With OpenDataLab data source
pip install -e '.[data]'

# Everything
pip install -e '.[all]'
```

## Quick Start

```bash
# 1. Initialize config
autotrainer init
# Edit ~/.autotrainer/config.yaml to set your paths and API keys

# 2. Prepare data (validate + profile)
autotrainer data --mode fixed --task paddleocr-vl --data-path /path/to/data.jsonl

# 3. Run the full pipeline (TUI mode)
autotrainer run --task paddleocr-vl

# 4. Or run headless
autotrainer run --task paddleocr-vl --no-tui
```

## Configuration

Config file at `~/.autotrainer/config.yaml`:

```yaml
paddleformers_root: "/path/to/PaddleFormers"

llm:
  base_url: "http://localhost:8000/v1"  # OpenAI-compatible API
  api_key: "sk-xxx"
  model: "your-model-name"

tavily_api_key: ""  # Optional, for broader data search

training:
  ablation_subset_ratio: 0.05
  ablation_max_steps: 1000
  full_epochs: 3
```

Environment variables also work (config file < env vars < CLI args):

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

### `autotrainer data`

Manage training datasets in 3 modes.

**Mode 1: `fixed`** — Validate, profile, and prepare existing data.

```bash
autotrainer data --mode fixed --task paddleocr-vl --data-path /data/train.jsonl

# With output (saves data_profile.json)
autotrainer data --mode fixed --task paddleocr-vl --data-path /data/train.jsonl --output-dir /data/profiles
```

Output includes: format validation, sample count, text length distribution, quality flags, sample preview.

**Mode 2: `expand`** — Find additional datasets to complement your existing data.

```bash
autotrainer data --mode expand --task paddleocr-vl --data-path /data/existing.jsonl
```

**Mode 3: `discover`** — Search for datasets from scratch.

```bash
# Default search (auto-generates query from task)
autotrainer data --mode discover --task paddleocr-vl

# Custom search query
autotrainer data --mode discover --task paddleocr-vl --query "Chinese handwritten OCR recognition dataset"
```

Searches HuggingFace Hub (always) + Tavily (if API key configured). Results show ID, source, download count.

After finding a dataset, download it and validate:

```bash
huggingface-cli download <repo_id>
autotrainer data --mode fixed --task paddleocr-vl --data-path /path/to/downloaded.jsonl
```

### `autotrainer run`

Start the automated training pipeline.

```bash
# TUI mode (default)
autotrainer run --task paddleocr-vl

# Specify GPUs
autotrainer run --task paddleocr-vl --gpus 0,1,2,3

# Headless mode (no TUI)
autotrainer run --task paddleocr-vl --no-tui

# Resume interrupted training
autotrainer run --task paddleocr-vl --resume
```

The pipeline runs through these phases:
1. **Data Prepare** — validate, profile, split train/val/test, create 5% ablation subset
2. **Env Check** — verify PaddlePaddle, packages, GPU (interactive if issues found)
3. **Ablation** — single-factor experiments on 5% subset, rank results
4. **Full Training** — best config on full data, with health watchdog
5. **Evaluation** — eval metrics + sample inference verification
6. **Report** — generate charts and summary

### `autotrainer status`

Show current pipeline and experiment status.

```bash
autotrainer status
```

### `autotrainer resume`

View recovery state and instructions for resuming.

```bash
autotrainer resume
```

### `autotrainer report`

Generate experiment reports.

```bash
# Text summary
autotrainer report --format text

# HTML with charts
autotrainer report --format html

# JSON
autotrainer report --format json
```

## Skills

Skills are self-contained LLM capabilities invoked during the pipeline. Each skill has a `SKILL.md` (prompt) and a `handler.py` (Python logic).

### `diagnose-training`

Two-tier error diagnosis:
- **Tier 1** (instant): Regex pattern matching for common errors (OOM, NaN, NCCL, data format, checkpoint)
- **Tier 2** (LLM): For complex/unknown errors, sends structured context to LLM for analysis

### `plan-experiment`

Experiment planning and ablation strategy:
- Plans single-factor ablation grid (which factors to test, which values)
- Ranks experiment results and suggests next experiments
- Identifies optimal config combinations

### `data-intel`

Dataset intelligence in 3 modes (fixed/expand/discover):
- Validates JSONL format against PaddleFormers schemas (erniekit, messages)
- Generates statistical profiles (sample counts, length distributions, quality)
- Searches across HuggingFace Hub + Tavily for datasets

## TUI Keybindings

| Key | Action |
|-----|--------|
| `Tab` | Toggle agent panel |
| `l` | Toggle log mode (smart/full) |
| `s` | Show status |
| `q` | Quit |

## Project Structure

```
autotrainer/
  autotrainer/
    cli/              # Click CLI commands
    tui/              # Textual TUI app, widgets, screens
    orchestrator/     # Pipeline orchestrator, state machine, recovery, health monitor
    managers/         # env, data, train, eval managers
    context/          # Context store with percentage-based token budgets
    skills/           # 3 skills: diagnose-training, plan-experiment, data-intel
    pf_integration/   # PaddleFormers config builder, launcher, log parser, validators
    utils/            # subprocess manager, file utils, GPU monitor, LLM client
```

## License

Apache 2.0
