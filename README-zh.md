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
  <b>LLM 驱动的全自动大模型训练系统</b><br>
  <sub>数据丢进去，睡一觉，模型就训好了。</sub>
</p>

---

## 快速开始

```bash
# 安装
pip install -e . && autotrainer init

# 配置 LLM API
export AUTOTRAINER_LLM_BASE_URL="http://localhost:8000/v1"
export AUTOTRAINER_LLM_API_KEY="sk-xxx"
export AUTOTRAINER_LLM_MODEL="qwen-72b"

# 开始训练 — 支持任意格式数据，一行命令
autotrainer train --data-path ./your_data.jsonl --gpus 0,1,2,3,4,5,6
```

支持 JSONL、CSV、Parquet、XML、ZIP、目录等格式，LLM 自动识别并转换。

---

## 工作原理

```
  原始数据 ────► 环境检测 ────► 消融实验 ────► 全量训练 ────► 报告生成
  (任意格式)      (GPU/依赖)     (5%子集)       (全量数据)     (图表+摘要)
                                     │
                                LLM 驱动决策:
                                · 超参数搜索
                                · 错误诊断
                                · 自适应调优
```

LLM 接管所有工程决策：数据格式转换、超参数搜索、训练错误诊断、结果分析。崩溃后可从任意中断阶段恢复运行。健康看护实时监控 GPU、检测假死、预警 OOM。

---

## 安装

```bash
git clone https://github.com/ZhijunLStudio/autotrainer.git
cd autotrainer
pip install -e .           # 标准安装
pip install -e '.[all]'    # 全部功能（含搜索和数据插件）
```

依赖：Python 3.10+、PaddlePaddle 3.3+、PaddleFormers。

---

## 配置

最小配置 — 只需设置 LLM API：

```yaml
# ~/.autotrainer/config.yaml
llm:
  base_url: "http://localhost:8000/v1"
  api_key: "sk-xxx"
  model: "qwen-72b"
```

或使用环境变量：

```bash
export AUTOTRAINER_LLM_BASE_URL="http://localhost:8000/v1"
export AUTOTRAINER_LLM_API_KEY="sk-xxx"
export AUTOTRAINER_LLM_MODEL="qwen-72b"
```

其他配置（PaddleFormers 路径、GPU 数量、依赖包）自动检测。

---

## 命令

| 命令 | 用途 |
|------|------|
| `autotrainer init` | 创建配置文件 `~/.autotrainer/config.yaml` |
| `autotrainer data` | LLM 驱动的智能数据转换（支持任意格式） |
| `autotrainer train` | 启动训练流水线 |
| `autotrainer status` | 查看流水线和实验状态 |
| `autotrainer resume` | 从上次中断处恢复 |
| `autotrainer report` | 生成报告（text / html / json） |

### `autotrainer data` — 智能数据处理

ReAct Agent 自动探查数据、生成转换脚本、执行并验证，失败自动重试。

```bash
autotrainer data --path /data/ocr_datasets/ --output-dir ./output
autotrainer data --path /data/a.parquet --path /data/b.csv
autotrainer data --path /data/cleaned.jsonl --profile-only
autotrainer data --path /data/cleaned.jsonl --split-only
```

支持输入格式：JSONL、JSON、CSV、TSV、Parquet、XML、ZIP、目录（任意嵌套）。

### `autotrainer train` — 一键训练

```bash
# TUI 模式
autotrainer train --data-path ./train.jsonl --gpus 0,1,2,3

# 跳过消融，直接全量训练
autotrainer train --data-path ./train.jsonl --gpus all --skip-ablation

# 从 DataAgent 输出目录启动
autotrainer train --data-dir ./autotrainer_output --gpus 0,1,2,3

# 无头模式（脚本/后台运行）
autotrainer train --data-path ./train.jsonl --no-tui
```

## 流水线阶段

```
0. 任务确认     确认目标、GPU、配置
1. 数据准备     验证、分析、切分数据，生成 5% 消融子集
2. 环境检测     验证 PaddlePaddle、依赖包、GPU
3. 消融实验     在 5% 子集上单因子迭代调优（Pearson 相关性分析）
4. 全量训练     最优配置在完整数据上训练（健康看护激活）
5. 评估验证     指标计算 + 推理抽样验证
6. 报告生成     图表、对比、摘要
```

## TUI 界面

```
┌─ AutoTrainer ──────────────────────────────────────────────────┐
│  Phase: ABLATION  │  Task: paddleocr-vl  │  GPUs: 0,1,2,3    │
├──────────────────────────┬─────────────────────────────────────┤
│  实验队列                │  Agent 面板                         │
│  ablation_hyperparams    │  本轮最佳学习率: 3e-4                │
│  5/5 已完成              │  损失趋势: 凹形                      │
│                          │  搜索范围收窄: [1e-4, 5e-4]          │
│  运行中: abl-lr=3e-4     │                                     │
├──────────────────────────┴─────────────────────────────────────┤
│  日志: [step 100] loss=2.345  lr=3.00e-04  tokens=8192         │
└─────────────────────────────────────────────────────────────────┘
```

| 快捷键 | 功能 |
|--------|------|
| `Tab` | 切换 Agent 面板 |
| `l` | 切换日志模式 |
| `s` | 显示状态 |
| `q` | 退出 |

## 特性

**LLM 驱动决策**：数据转换、超参数搜索、错误诊断全部由 LLM 完成，无需人工编写脚本。

**任意数据格式**：支持 JSONL、CSV、Parquet、XML、ZIP 及嵌套目录，ReAct Agent 自动探查、转换、验证。

**智能消融实验**：单因子逐轮调优，Pearson 相关性趋势分析，搜索范围自适应收窄/扩大，自动收敛检测。

**崩溃恢复**：基于 SQLite 的 WAL 持久化，任意阶段中断后可从断点恢复运行。

**健康看护**：GPU 内存监控、训练假死检测、OOM 预警。

**插件化任务**：通过 `manifest.yaml` 声明即可添加新模型，无需修改代码。

## 项目结构

```
autotrainer/
├── autotrainer/
│   ├── core/              # 基础设施 — Store (SQLite)、Registry、Interfaces
│   ├── phases/            # 流水线 — 7 个独立 PhaseHandler
│   ├── services/          # 共享服务 — ExperimentService、CheckpointService
│   ├── tasks/             # 任务插件 — manifest 注册
│   ├── skills/            # LLM 能力 — data-inspect、diagnose、plan-experiment
│   ├── orchestrator/      # 调度器 + 健康监控
│   ├── managers/          # 训练、数据、环境、评估管理器
│   ├── tui/               # Textual 终端界面
│   ├── cli/               # Click 命令行
│   └── pf_integration/    # PaddleFormers 集成
├── scripts/               # 辅助脚本
└── tests/                 # 74 个测试
```

## 许可证

Apache 2.0
