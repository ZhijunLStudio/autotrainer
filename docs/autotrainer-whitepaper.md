# AutoTrainer 白皮书

> 一个 LLM 驱动的大模型训练自动化系统。扔下数据，离开，回来看训练好的模型。

---

## 一、AutoTrainer 是什么，不是什么

### AutoTrainer vs OpenCode / Claude Code

很多人第一反应："这不就是个用了 LLM 的训练脚本吗？和 Claude Code / OpenCode 有什么本质区别？"

答案是：**它们解决的是完全不同层面的问题**。

| 维度 | AutoTrainer | OpenCode / Claude Code |
|------|-------------|----------------------|
| **解决的问题** | 大模型训练的端到端自动化 | 通用编程辅助 |
| **LLM 的角色** | 训练决策引擎（调参、诊断、数据分析） | 代码生成/编辑引擎 |
| **核心用户** | ML 工程师，做模型微调 | 所有开发者 |
| **产出物** | 训练好的模型 + 评估报告 | 代码修改、文件编辑 |
| **运行时长** | 数小时到数天 | 秒到分钟级 |
| **核心能力** | 数据探索、超参搜索、错误自愈、崩溃恢复 | 代码阅读、编辑、重构、生成 |

### 类比

如果把 AutoTrainer 比作"自动挡汽车"，那 OpenCode/Claude Code 就是"通用工具箱"。

工具箱可以做很多事，但不会自动帮你开车。AutoTrainer 做的是：你扔数据上去，它自己决定怎么训练、用什么参数、出错了怎么修、最后给你一个训练好的模型。

### 不可替代的独有能力

1. **ReAct 数据代理** — 自动探索未知格式数据集，生成转换脚本，自动修复失败
2. **消融实验自动化** — 基于单因子调优 + Pearson 相关性趋势分析的 LLM 驱动超参搜索
3. **训练生命周期管理** — SQLite WAL 持久化的状态机，保证从任意中断点恢复
4. **GPU 健康监控** — GPU 内存、温度、卡死检测、OOM 预警、磁盘空间监控
5. **深度框架集成** — 直接对接 PaddleFormers/PaddlePaddle 的训练管线

### 对标的竞品

AutoTrainer 对标的是 **HuggingFace AutoTrain**、**Google Vertex AI AutoML**，而不是 OpenCode 或 Claude Code。

---

## 二、Skills 体系 — ML 训练专用的 LLM 能力封装

AutoTrainer 内置了一套独立的 skills 体系（6 个 skills），结构上仿照 Claude Code 的 skill 系统：

```
skills/
├── data_inspect/      # ReAct agent 探索未知格式数据，生成转换脚本
├── data_fix/          # 收到错误后自动修复转换脚本
├── data_intel/        # 从 HF Hub/OpenDataLab 发现和评估训练数据集
├── data_ratio_ablation/ # 多数据集混合比例优化
├── diagnose_training/ # 训练 OOM/NaN/NCCL 等错误的诊断和修复
└── plan_experiment/   # Pearson 相关性分析，指导超参搜索方向
```

每个 skill 都是 `SKILL.md` (YAML frontmatter + prompt) + `handler.py` (执行逻辑) 的组合。

**与 Claude Code Skills 的关键差异：**

| 维度 | AutoTrainer Skills | Claude Code Skills |
|------|-------------------|-------------------|
| 解决问题 | ML 训练流水线各环节 | 通用编程任务 |
| 运行方式 | 由 PipelineOrchestrator 按阶段调用 | 用户手动触发 |
| 分级策略 | Tier 1 快速模式匹配 + Tier 2 LLM | 直接用 LLM |
| 上下文注入 | 训练 metrics、config、GPU 状态 | 代码库结构 |

---

## 三、长程任务执行分析

### 现有的保障机制（4 层）

**第一层：崩溃恢复**
- SQLite + WAL 日志模式 (`PRAGMA journal_mode=WAL`)
- 每个 pipeline 阶段执行前写状态，恢复时自动跳过已完成阶段
- `pipeline_runs` → `phase_events` → `experiments` → `checkpoints` 四表联动

**第二层：健康监控**
- 后台线程每 5 秒轮询：进程存活、日志新鲜度、GPU 利用率/内存/温度
- 异常检测：进程死亡、训练卡死(>300s 无输出)、OOM风险(>95% 显存)、过热(>90°C)、GPU空转(>60s)
- **新增**：自动杀进程恢复 (5min冷却期)，磁盘空间监控(>95% 或 <10GB 告警)

**第三层：上下文窗口管理**
- 模仿 Claude Code 的 tokenBudget：超过 85% 自动 compact，超过 95% hard stop
- diminishing returns 检测：连续 3 轮 <500 tokens 新内容触发压缩

**第四层：沙箱 + 超时**
- 数据转换脚本：自适应超时 (300s/GB)，快速验证 (100行先测)
- ReAct agent：循环检测 (连续3次相同 thought 强制结束)
- 危险命令拦截 (rm -rf, mkfs 等)

### 已修复的 5 个长程问题

| 问题 | 修复方案 | 文件 |
|------|---------|------|
| 训练 hang 不会被自动处理 | 新增 `on_action` 回调 + `process_kill_fn`，hang 后自动杀进程 | `health_monitor.py` |
| LLM API 无同步重试 | 新增 `complete_sync`/`complete_json_sync`，指数退避 + 429/529 分类 | `llm_client.py` |
| 无步骤级 checkpoint 续训 | 自动扫描 `checkpoint-N` 目录，从最大 step 续训 | `full_training.py` |
| 异常无外部通知 | JSON 格式 `notifications.log`，五级分类 (INFO/WARN/ERROR/ACTION/CRITICAL) | `full_training.py` |
| 无磁盘空间监控 | `shutil.disk_usage` 检测，>95% 或 <10GB 触发异常 | `health_monitor.py` |

### 仍存在的限制

1. **单机假设** — 无多机/分布式支持，机器宕机需人工重启
2. **LLM API 单点故障** — API 挂了无降级路径（但已有重试机制缓解）
3. **阶段级恢复为主** — 最小恢复粒度是 Phase，不是 step（但 full_training 已支持 checkpoint 续训）

### 大型数据集运行预期

| 阶段 | 耗时估算 | 风险等级 |
|------|---------|---------|
| Data Prepare | 小时级 | 低 — 有沙箱保护 |
| Ablation | 数小时 | 低 — 每个实验独立，失败不影响其他 |
| Full Training | 数天到数周 | 中 — 有 health monitor + auto-kill + resume |
| Evaluation + Report | 分钟级 | 极低 |

结论：**best-effort 持久化 + 被动监控 + 主动自愈（新增）**。机子不宕、LLM API 不挂，可以跑数周。一旦出问题，保证不丢进度，大部分异常能自动恢复。

---

## 四、全面消融实验结果

### 实验设计

- **方法**: 单因子迭代消融 (Single-Factor Iterative Ablation)
- **硬件**: 4× NVIDIA A800 80GB (GPU 4,5,6,7)
- **模型**: PaddlePaddle/PaddleOCR-VL
- **数据**: 1000 张阿拉伯语 OCR 文本行图片 (900 train / 100 val)
- **消融规模**: 100 train / 50 val，每实验 1 epoch

### 因子 1：学习率扫描 (6 个值)

| LR | Loss | 排名 |
|----|------|------|
| 1e-6 | 0.8435 | 4 |
| 3e-6 | 0.8242 | 3 |
| 5e-6 | 0.7737 | 2 |
| **1e-5** | **0.7343** | **1 ✅** |
| 3e-5 | 1.0112 | 5 |
| 5e-5 | 1.8296 | 6 |

### 因子 2：Batch Size 扫描 (3 个值)

| BS | Loss | 排名 |
|----|------|------|
| 1 | 0.8720 | 3 |
| 2 | 0.7364 | 2 |
| **4** | **0.6198** | **1 ✅** |

### 因子 3：Warmup Ratio (3 个值)

| Warmup | Loss | 排名 |
|--------|------|------|
| **0.01** | **0.6191** | **1 ✅** |
| 0.05 | 0.6203 | 2 |
| 0.15 | 0.6197 | 3 |

### 因子 4：Weight Decay (3 个值)

| WD | Loss | 排名 |
|----|------|------|
| 0.01 | 0.6195 | 2 |
| 0.05 | 0.6202 | 3 |
| **0.1** | **0.6195** | **1 ✅** |

### 因子 5：Sharding (2 个值)

| Sharding | Loss | 排名 |
|----------|------|------|
| Stage1 | 0.6199 | 2 |
| **Stage2** | **0.6197** | **1 ✅** |

### 最终结果

**最优配置**: LR=1e-5, BS=4, warmup=0.01, weight_decay=0.1, sharding=stage2

| 指标 | 消融阶段 | 全量训练 |
|------|---------|---------|
| Train Loss | 0.6197 | **0.3392** |
| 吞吐 | — | 51.2 samples/sec |
| 训练时间 | ~70s/exp | 54s |

消融迭代轨迹：**0.84 → 0.73 → 0.62 → 0.619 → 0.339** (改善 60%)

---

## 五、Pipeline 架构

```
Raw Data ──► Data Prepare ──► Env Check ──► Ablation ──► Full Training ──► Eval ──► Report
(any format)   (ReAct agent)   (GPU/pkg)    (5% subset)   (full data)    (metrics)  (charts)
                                                  │
                                           LLM-driven:
                                           · Hyperparameter tuning
                                           · Error diagnosis
                                           · Adaptive search
```

7 个 Phase 由 `PipelineOrchestratorV2` 调度，每个 Phase 都有独立的 `PhaseHandler`，状态持久化到 SQLite。
