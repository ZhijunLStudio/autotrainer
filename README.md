# AutoTrainer · 自动训练器

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/tests-74%20passed-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/GPU-A800%2080GB-orange?style=flat-square" alt="GPU">
  <img src="https://img.shields.io/badge/PaddlePaddle-3.3%2B-cyan?style=flat-square" alt="PaddlePaddle">
</p>

<p align="center">
  <b>LLM 驱动的全自动大模型训练系统</b><br>
  <sub>LLM-Driven Automated Training System for Large Models</sub>
</p>

<p align="center">
  把你的数据丢进去，睡一觉，模型就训好了。<br>
  <sub>Drop your data in. Go to sleep. Wake up to a trained model.</sub>
</p>

---

<p align="center">
  <a href="#一键启动-quick-start">Quick Start</a> ·
  <a href="#工作原理-how-it-works">How It Works</a> ·
  <a href="#安装-installation">Install</a> ·
  <a href="#命令-commands">Commands</a> ·
  <a href="#项目结构-project-structure">Structure</a>
</p>

---

## 一键启动 · Quick Start

```bash
# 1. 安装 (一行搞定)
pip install -e . && autotrainer init

# 2. 配置 LLM API (编辑 ~/.autotrainer/config.yaml，填写你的 API 地址)
# 或者用环境变量:
export AUTOTRAINER_LLM_BASE_URL="http://localhost:8000/v1"
export AUTOTRAINER_LLM_API_KEY="sk-xxx"
export AUTOTRAINER_LLM_MODEL="qwen-72b"

# 3. 开始训练 (一行搞定，支持任意格式数据)
autotrainer train --data-path ./your_data.jsonl --gpus 0,1,2,3,4,5,6
```

> **就这么简单。** 支持 JSONL / CSV / Parquet / XML / ZIP / 目录，LLM 自动识别并转换格式。

<sub>That's it. Any data format — JSONL, CSV, Parquet, XML, ZIP, directories — the LLM auto-detects and converts.</sub>

---

## 工作原理 · How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      AutoTrainer Pipeline                    │
│                                                             │
│  📦 数据         🔍 环境        🧪 消融        🚀 全量训练   │
│  Data ──────► Env Check ──► Ablation ──► Full Training ──┐  │
│  Prepare        (GPU/包)      (5%数据)       (全量数据)    │  │
│                                                          │  │
│               📊 报告  ◄────── 📈 评估  ◄─────────────────┘  │
│               Report           Eval                         │
│                                                             │
│  🧠 LLM 驱动每一步决策:                                      │
│  · 数据格式转换 (ReAct Agent)          · 超参搜索 (自适应)    │
│  · 训练错误诊断 (两阶段: 正则 + LLM)   · 报告生成 (图表+摘要)  │
│                                                             │
│  💾 崩溃恢复: 任何阶段中断都能自动续跑                        │
│  🩺 健康看护: GPU 监控 / 假死检测 / OOM 预警                 │
└─────────────────────────────────────────────────────────────┘
```

**核心理念**：只需要提供数据和 GPU，LLM 接管所有工程决策 — 数据转换、超参搜索、错误诊断、结果分析。

<sub>**Core idea**: Provide data + GPUs. The LLM handles every engineering decision — data conversion, hyperparameter search, error diagnosis, result analysis.</sub>

---

## 安装 · Installation

```bash
# 标准安装
git clone https://github.com/ZhijunLStudio/autotrainer.git
cd autotrainer
pip install -e .

# 可选扩展
pip install -e '.[search]'   # Tavily 语义搜索，发现新数据集
pip install -e '.[all]'      # 全部功能
```

<sub>Requires: Python 3.10+, PaddlePaddle 3.3+, PaddleFormers. The pipeline auto-detects your environment.</sub>

---

## 配置 · Configuration

**最小配置** — 只需要填 LLM API：

```yaml
# ~/.autotrainer/config.yaml
llm:
  base_url: "http://localhost:8000/v1"   # OpenAI 兼容 API
  api_key: "sk-xxx"
  model: "qwen-72b"                      # 任意兼容模型
```

其他全部自动检测（PaddleFormers 路径、GPU 数量、环境依赖）。也可以用环境变量：

```bash
export AUTOTRAINER_LLM_BASE_URL="http://localhost:8000/v1"
export AUTOTRAINER_LLM_API_KEY="sk-xxx"
export AUTOTRAINER_LLM_MODEL="qwen-72b"
```

<sub>Everything else auto-detects: PaddleFormers path, GPU count, packages.</sub>

---

## 命令 · Commands

| 命令 | 功能 | 示例 |
|------|------|------|
| `autotrainer init` | 创建配置文件 | `autotrainer init` |
| `autotrainer data` | LLM 驱动数据转换 | `autotrainer data --path ./ocr_data/` |
| `autotrainer train` | 启动训练流水线 | `autotrainer train --data-path ./train.jsonl --gpus 0,1,2,3` |
| `autotrainer status` | 查看当前状态 | `autotrainer status` |
| `autotrainer resume` | 从中断处恢复 | `autotrainer resume` |
| `autotrainer report` | 生成报告 | `autotrainer report --format html` |

### `autotrainer data` — 智能数据处理

LLM 驱动的 ReAct Agent 自动处理任意格式数据：

```
  📂 原始数据 (任意格式)
       │
       ▼
  🔍 LLM 探索 (ls / head / schema 检测)
       │
       ▼
  ✍️ 生成转换脚本 (Python, 针对此数据集定制)
       │
       ▼
  ✅ 执行 + 验证 (失败自动重试 ≤3次)
       │
       ▼
  🧹 清洗 (MD5去重, 坏行移除)
       │
       ▼
  ✂️ 分割 (train/val/test = 90/5/5)
       │
       ▼
  📦 输出 → autotrainer_output/
```

```bash
# 处理整个目录 (每个子目录 = 一个数据集)
autotrainer data --path /data/arabic_ocr_datasets/ --output-dir ./output

# 处理多个文件
autotrainer data --path /data/a.parquet --path /data/b.csv

# 只做分析，不转换
autotrainer data --path /data/cleaned.jsonl --profile-only

# 用自定义脚本 (手动修复后重新跑)
autotrainer data --path /data/dataset/ --script ./output/dataset/convert_script.py
```

<sub>Supported input: JSONL, JSON, CSV, TSV, Parquet, XML, ZIP, directories (any nesting depth).</sub>

### `autotrainer train` — 一键训练

```bash
# TUI 模式 (推荐，实时面板)
autotrainer train --data-path ./train.jsonl --gpus 0,1,2,3

# 跳过消融直接全量训练
autotrainer train --data-path ./train.jsonl --gpus 0,1,2,3,4,5,6 --skip-ablation

# 从 DataAgent 输出目录启动
autotrainer train --data-dir ./autotrainer_output --gpus all

# 无头模式 (适合脚本 / 后台运行)
autotrainer train --data-path ./train.jsonl --no-tui
```

### TUI 界面 · Terminal UI

```
┌─ AutoTrainer ───────────────────────────────────────────────┐
│  Phase: ABLATION  │  Task: paddleocr-vl  │  GPUs: 0,1,2,3  │
├──────────────────────────┬───────────────────────────────────┤
│                          │                                   │
│   🧪 Experiment Queue    │   🤖 Agent Panel                  │
│   ────────────────────  │   ─────────────────────────────  │
│   ablation_hyperparams  │   > 本轮最佳学习率: 3e-4            │
│   ✅ 5/5 completed      │   > loss 下降趋势: concave         │
│                          │   > 搜索范围收窄: [1e-4, 5e-4]     │
│   ablation_ratio        │                                   │
│   ⏳ 2/4 completed      │                                   │
│                          │                                   │
│   ▶️ Running: abl-lr=3e-4│                                   │
├──────────────────────────┴───────────────────────────────────┤
│  📋 Log Panel                                                │
│  ────────────────────────────────────────────────────────    │
│  [step 100] loss=2.345  lr=3.00e-04  tokens=8192            │
│  [step 200] loss=1.876  lr=3.00e-04  tokens=16384           │
└─────────────────────────────────────────────────────────────┘
```

| 快捷键 | 功能 |
|--------|------|
| `Tab` | 切换 Agent 面板 |
| `l` | 切换日志模式 (智能 / 完整) |
| `s` | 显示当前状态 |
| `q` | 退出 |

---

## 特性 · Features

| 特性 | 说明 |
|------|------|
| 🧠 **LLM 驱动** | 每一步决策都由 LLM 做出：数据转换、超参搜索、错误诊断 |
| 📦 **任意格式数据** | ReAct Agent 自动识别并转换 JSONL/CSV/Parquet/XML/ZIP/目录 |
| 🧪 **智能消融** | 单因子逐轮调优 + Pearson 相关性 + 自适应搜索 + 收敛检测 |
| 🩺 **健康看护** | GPU 监控 / 训练假死检测 / OOM 预警 |
| 💾 **崩溃恢复** | SQLite 存储所有状态，中断后从断点继续 |
| 🎨 **TUI 终端界面** | 实时面板：实验队列 / Agent 决策 / 训练日志 / GPU 状态 |
| 🔌 **插件扩展** | Task manifest 一行声明新模型/新任务 |
| 📊 **自动报告** | 训练结束自动生成图表和摘要 |

<details>
<summary>🔍 展开查看所有特性详情</summary>

### LLM Skills (按需调用)
- **data-inspect** — 理解数据 schema，生成 erniekit 转换脚本
- **data-fix** — 出错时自动修复转换脚本
- **diagnose-training** — 两阶段错误诊断 (正则快速匹配 + LLM 深度分析)
- **plan-experiment** — 智能超参搜索：单因子消融 → Pearson 趋势分析 → 自适应搜索范围

### 崩溃恢复
- SQLite WAL 模式，事务保证写入安全
- 任何 phase 中断，重跑自动跳过已完成步骤
- 实验级别粒度：已完成实验不重复跑

### 上下文管理
- 百分比 token 预算约束
- 原始日志不入 LLM 上下文
- 智能摘要替代原始输出

</details>

---

## 项目结构 · Project Structure

```
autotrainer/
├── autotrainer/
│   ├── core/              # 🏗️ 基础设施 — Store(SQLite), Registry(插件), Interfaces
│   ├── phases/            # ⚙️ 流水线 — 7个独立 PhaseHandler
│   │   ├── data_prepare.py    # 数据准备
│   │   ├── env_check.py       # 环境检测
│   │   ├── ablation.py        # 消融实验
│   │   ├── full_training.py   # 全量训练
│   │   ├── evaluation.py      # 评估验证
│   │   └── report.py          # 报告生成
│   ├── services/          # 🔧 共享服务 — ExperimentService, CheckpointService
│   ├── tasks/             # 📋 任务插件 — paddleocr-vl + manifest 注册
│   ├── skills/            # 🧠 LLM 能力 — data-inspect, diagnose, plan-experiment
│   ├── orchestrator/      # 🎯 调度器 — 薄调度层 + 健康监控
│   ├── managers/          # 🛠️ 管理器 — train, data, env, eval
│   ├── tui/               # 🎨 终端界面 — Textual TUI
│   ├── cli/               # ⌨️ 命令行 — Click CLI
│   └── pf_integration/    # 🔗 PaddleFormers 集成 — launcher, config, log parser
├── scripts/               # 📜 辅助脚本
└── tests/                 # ✅ 74个测试
```

---

## 许可证 · License

Apache 2.0 © 2025
