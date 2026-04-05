# OCR 训练指标与可视化系统设计

> 日期：2026-04-05
> 状态：待审批

## 1. 背景

当前 autotrainer 的训练指标体系过于简单——训练集仅追踪 loss/lr/throughput/memory，验证集仅有 eval_loss/eval_accuracy/eval_perplexity，可视化也只有一张 eval_loss 柱状图。OCR 领域有大量可量化指标，需要建立完整的指标体系和多维可视化方案，以支持多轮训练的成果对比和分析。

## 2. 指标体系

### 2.1 训练指标（从日志解析）

| 指标 | 说明 | 来源 |
|------|------|------|
| `loss` | 训练 loss | 已有 |
| `learning_rate` | 学习率 | 已有 |
| `throughput` | 吞吐量 (tokens/s) | 已有 |
| `gpu_memory_mb` | GPU 显存占用 | 已有 |
| `grad_norm` | 梯度范数，监控训练稳定性 | **新增** |

### 2.2 验证集核心指标（全部新增）

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **NED** | Normalized Edit Distance | `levenshtein(pred, ref) / max(len(pred), len(ref))`，PaddleOCR-VL 官方核心指标 |
| **CER** | Character Error Rate | `levenshtein(pred_chars, ref_chars) / len(ref_chars)` |
| **WER** | Word Error Rate | `levenshtein(pred_words, ref_words) / len(ref_words)` |
| **Exact Match** | 完全匹配率 | `count(pred == ref) / total` |
| **TEDS** | Tree Edit Distance Similarity | 表格结构识别指标，基于树编辑距离 |
| **Layout mAP** | 版面分析 mean Average Precision | 检测框 IoU + 类别匹配 |

### 2.3 效率指标（衍生计算）

| 指标 | 说明 |
|------|------|
| `best_step` | 达到最佳 eval loss 的训练步数 |
| `loss_reduction_rate` | loss 下降速率 `(init_loss - final_loss) / steps` |
| `overfitting_gap` | `train_loss - eval_loss` 差值 |
| `training_time_per_step` | 每步平均耗时 |

## 3. 可视化方案

### 3.1 雷达图 — 多维能力画像

- 将 NED、CER、WER、Exact Match、TEDS、Layout mAP 归一化到 0-1 范围
- 单次实验 = 一条封闭折线，多轮训练 = 多条线叠加对比
- 直观展示模型在各维度的能力分布

### 3.2 柱状图 — 实验间横向对比

- 每个指标一组柱状图，X 轴 = 实验 ID，Y 轴 = 指标值
- 分组柱状图：多个关键指标放在同一图中对比
- 标注最佳值（高亮/加粗）

### 3.3 折线图 — 训练过程趋势

- loss 曲线（train + eval 双线）随 step 变化
- NED 等验证指标随 eval step 变化
- 双轴图：loss + learning_rate 同图展示
- 用阴影区域标注过拟合区间（train_loss < eval_loss 且 gap 持续增大）

### 3.4 热力图 — 指标×实验矩阵

- 行 = 指标，列 = 实验
- 颜色深浅表示指标优劣
- 一眼看出哪个实验在哪些维度强/弱

### 3.5 输出格式

| 类型 | 库 | 格式 | 用途 |
|------|------|------|------|
| 静态图 | matplotlib | PNG | 快速查看、嵌入文档 |
| 交互式 | plotly | HTML | 浏览器中缩放、悬停查看数值 |

## 4. 架构设计

### 4.1 新增模块

```
autotrainer/
  metrics/
    __init__.py
    ocr_metrics.py        # NED, CER, WER, Exact Match, TEDS, Layout mAP
    efficiency.py          # 收敛速度、过拟合差距等衍生指标
    compute.py             # 统一入口 compute_all_metrics() -> dict
  visualization/
    __init__.py
    charts.py              # matplotlib 静态图
    interactive.py         # plotly 交互式 HTML
    report_generator.py    # 组装完整 HTML 报告
```

### 4.2 数据流

```
训练完成
  → eval_manager 调用 metrics/compute.py: compute_all_metrics()
  → 结果写入 eval_results/<exp_id>.json（扩展格式）
  → pipeline report 阶段读取所有 eval_results
  → visualization/report_generator.py 生成报告
  → 输出: reports/ 目录下的 PNG + HTML
```

### 4.3 扩展的 eval_results 格式

```json
{
  "experiment_id": "full_train_001",
  "timestamp": "2026-04-05T10:00:00",
  "metrics": {
    "text_recognition": {
      "ned": 0.95,
      "cer": 0.03,
      "wer": 0.05,
      "exact_match": 0.82
    },
    "table_recognition": {
      "teds": 0.88
    },
    "layout_analysis": {
      "map": 0.91
    },
    "training": {
      "eval_loss": 0.42,
      "train_loss": 0.38,
      "best_step": 5000,
      "overfitting_gap": 0.04
    }
  },
  "history": {
    "steps": [100, 200, 300],
    "train_loss": [2.1, 1.5, 0.9],
    "eval_loss": [2.0, 1.4, 0.85],
    "ned": [0.6, 0.75, 0.88]
  }
}
```

### 4.4 HTML 报告结构

1. **概览卡片：** 实验 ID、状态、最佳指标一览
2. **雷达图区：** 多实验能力画像对比
3. **柱状图区：** 各指标横向对比（PNG 内嵌 + 交互式链接）
4. **折线图区：** 训练曲线（loss、NED、lr）
5. **热力图区：** 指标×实验矩阵
6. **详情表格：** 所有指标精确数值，支持排序

## 5. 与现有代码的集成点

| 现有文件 | 改动 |
|----------|------|
| `managers/eval_manager.py` | 调用 `compute_all_metrics()` 替代原有简单评估 |
| `pf_integration/log_parser.py` | 新增 `grad_norm` 解析 |
| `cli/report_cmd.py` | 调用 `visualization/report_generator.py` 替代原有简单报告 |
| `orchestrator/pipeline.py` | report 阶段传入扩展数据 |
| `managers/train_manager.py` | 收集 training history（每步 loss/lr）用于折线图 |

## 6. 依赖

新增 Python 包依赖：
- `python-Levenshtein` — 编辑距离计算
- `plotly` — 交互式图表
- `jinja2` — HTML 模板渲染（可选，也可直接用 plotly 内置）

已有依赖（无需新增）：
- `matplotlib` — 静态图表
- `numpy` — 数值计算

## 7. 实施顺序

1. `metrics/ocr_metrics.py` — 核心指标计算函数
2. `metrics/efficiency.py` — 衍生效率指标
3. `metrics/compute.py` — 统一入口
4. 修改 `eval_manager.py` — 集成新指标
5. 修改 `log_parser.py` — 新增 grad_norm
6. `visualization/charts.py` — 静态图
7. `visualization/interactive.py` — 交互式图
8. `visualization/report_generator.py` — 完整报告
9. 修改 `report_cmd.py` — 集成新报告
10. 修改 `pipeline.py` — 串联全流程
