# AutoTrainer · 自动训练器

<p align="center">
  <a href="README.md">English</a> · <a href="README-zh.md">中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/tests-74%20passed-brightgreen" alt="Tests">
</p>

<p align="center">
  <b>LLM-driven automated training system for large models.</b><br>
  <sub>Drop your data. Walk away. Come back to a trained model.</sub>
</p>

---

## Quick Start

```bash
# Install
pip install -e . && autotrainer init

# Configure LLM API
export AUTOTRAINER_LLM_BASE_URL="http://localhost:8000/v1"
export AUTOTRAINER_LLM_API_KEY="sk-xxx"
export AUTOTRAINER_LLM_MODEL="qwen-72b"

# Train — any data format, one command
autotrainer train --data-path ./your_data.jsonl --gpus 0,1,2,3,4,5,6
```

Supports JSONL, CSV, Parquet, XML, ZIP, and directories. The LLM auto-detects and converts your data.

---

## How It Works

```
  Raw Data ────► Env Check ────► Ablation ────► Full Training ────► Report
  (any format)    (GPU/pkg)       (5% subset)    (full dataset)     (charts)
                                       │
                                  LLM-driven:
                                  · Hyperparameter tuning
                                  · Error diagnosis
                                  · Adaptive search
```

The LLM handles every engineering decision: data conversion, hyperparameter search, error diagnosis, and result analysis. Crash recovery resumes from any interrupted phase. Health watchdog monitors GPU, detects hangs, and warns on OOM risk.

---

## Installation

```bash
git clone https://github.com/ZhijunLStudio/autotrainer.git
cd autotrainer
pip install -e .           # standard
pip install -e '.[all]'    # everything (search + data plugins)
```

Requires Python 3.10+, PaddlePaddle 3.3+, PaddleFormers.

---

## Configuration

Minimal config — just set the LLM API:

```yaml
# ~/.autotrainer/config.yaml
llm:
  base_url: "http://localhost:8000/v1"
  api_key: "sk-xxx"
  model: "qwen-72b"
```

Or use environment variables:

```bash
export AUTOTRAINER_LLM_BASE_URL="http://localhost:8000/v1"
export AUTOTRAINER_LLM_API_KEY="sk-xxx"
export AUTOTRAINER_LLM_MODEL="qwen-72b"
```

Everything else (PaddleFormers path, GPU count, packages) is auto-detected.

---

## Commands

| Command | Purpose |
|---------|---------|
| `autotrainer init` | Create config file `~/.autotrainer/config.yaml` |
| `autotrainer data` | LLM-driven data conversion (any format) |
| `autotrainer train` | Launch training pipeline |
| `autotrainer status` | Show pipeline and experiment status |
| `autotrainer resume` | Resume from last checkpoint |
| `autotrainer report` | Generate reports (text / html / json) |

### `autotrainer data` — Smart Data Processing

A ReAct agent that inspects your data, writes a conversion script, runs it, and auto-fixes on failure.

```bash
autotrainer data --path /data/ocr_datasets/ --output-dir ./output
autotrainer data --path /data/a.parquet --path /data/b.csv
autotrainer data --path /data/cleaned.jsonl --profile-only
autotrainer data --path /data/cleaned.jsonl --split-only
```

Supported inputs: JSONL, JSON, CSV, TSV, Parquet, XML, ZIP, directories (any nesting).

### `autotrainer train` — One-Click Training

```bash
# TUI mode
autotrainer train --data-path ./train.jsonl --gpus 0,1,2,3

# Skip ablation, go straight to full training
autotrainer train --data-path ./train.jsonl --gpus all --skip-ablation

# From DataAgent output
autotrainer train --data-dir ./autotrainer_output --gpus 0,1,2,3

# Headless (scripts / background)
autotrainer train --data-path ./train.jsonl --no-tui
```

## Pipeline Phases

```
0. Task Confirm     Confirm goal, GPUs, config
1. Data Prepare     Validate, profile, split data, create 5% ablation subset
2. Env Check        Verify PaddlePaddle, packages, GPU
3. Ablation         Per-factor tuning on 5% subset with Pearson correlation
4. Full Training    Best config on full data, health watchdog active
5. Evaluation       Metrics + inference sample verification
6. Report           Charts, comparisons, summary
```

## TUI

```
┌─ AutoTrainer ──────────────────────────────────────────────────┐
│  Phase: ABLATION  │  Task: paddleocr-vl  │  GPUs: 0,1,2,3    │
├──────────────────────────┬─────────────────────────────────────┤
│  Experiment Queue        │  Agent Panel                        │
│  ablation_hyperparams    │  Best LR this round: 3e-4           │
│  5/5 completed           │  Loss trend: concave                │
│                          │  Search narrowed: [1e-4, 5e-4]      │
│  Running: abl-lr=3e-4    │                                     │
├──────────────────────────┴─────────────────────────────────────┤
│  Log: [step 100] loss=2.345  lr=3.00e-04  tokens=8192          │
└─────────────────────────────────────────────────────────────────┘
```

| Key | Action |
|-----|--------|
| `Tab` | Toggle Agent panel |
| `l` | Toggle log mode |
| `s` | Show status |
| `q` | Quit |

## Features

**LLM-Driven Decisions**: Data conversion, hyperparameter search, and error diagnosis are all handled by the LLM. No manual scripting needed.

**Any Data Format**: Drop in JSONL, CSV, Parquet, XML, ZIP, or nested directories. The ReAct agent inspects, converts, and validates automatically.

**Intelligent Ablation**: Single-factor iterative tuning with Pearson correlation trend analysis, adaptive search range narrowing/expansion, and automatic convergence detection.

**Crash Recovery**: SQLite-backed WAL persistence. Interrupt at any phase — resume from exactly where you left off.

**Health Watchdog**: GPU memory monitoring, training hang detection, OOM early warning.

**Plug-in Tasks**: Add new models by writing a `manifest.yaml`. No code changes needed.

## Project Structure

```
autotrainer/
├── autotrainer/
│   ├── core/              # Infrastructure — Store (SQLite), Registry, Interfaces
│   ├── phases/            # Pipeline — 7 independent PhaseHandlers
│   ├── services/          # Shared services — ExperimentService, CheckpointService
│   ├── tasks/             # Task plugins — manifest-based registration
│   ├── skills/            # LLM capabilities — data-inspect, diagnose, plan-experiment
│   ├── orchestrator/      # Scheduler + health monitor
│   ├── managers/          # Train, data, env, eval managers
│   ├── tui/               # Textual terminal UI
│   ├── cli/               # Click CLI
│   └── pf_integration/    # PaddleFormers integration
├── scripts/               # Helper scripts
└── tests/                 # 74 tests
```

## License

Apache 2.0
