# 学生体质健康 RAG 研究系统

> 16 模型表格基准 · 跨年级早期退化预警 · RAG–情境 Bandit 个性化推荐

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: CHIL 2027](https://img.shields.io/badge/Paper-CHIL%202027-blue.svg)](paper/CHIL2027_paper_outline.md)
[![Data: Aggregated Only](https://img.shields.io/badge/Data-Aggregated%20Only-green.svg)](#数据说明)

## 项目概述

本系统基于大学生体质健康测试纵向数据，构建"分类 → 预警 → 干预推荐"三模块研究管线：

| 模块 | 名称 | 任务 |
|------|------|------|
| **M1** | 表格模型基准 | 16 种现代表格模型在九类健康状态分类（HI9）上的对比基准 |
| **M2** | 早期退化预警 | 基于早期观测跨年级预测未来体质退化 |
| **M3** | RAG–Bandit 推荐 | 基于健康知识库的 TF-IDF 检索 + 情境 Bandit 个性化干预选择 |

此外，**HealthAgent** 是基于 LangGraph 的多智能体演示系统，以 SFT/DPO 微调的 Qwen2.5-7B-Instruct 为核心，提供健康评估与运动处方的交互式网页界面。

## 核心结果

### M1 — 16 模型基准（HI9 分类）

| 模型族 | 最佳模型 | Accuracy | Macro-F1 |
|--------|---------|----------|----------|
| 表格基础模型 / 预训练 | TabM | 0.859 | — |
| | TabPFN-2.5 | 0.856 | — |
| | ModernNCA | 0.855 | — |
| 对比学习 | SupCon | 0.845 | 0.803 |
| 经典模型 | RandomForest | 0.630 | — |

**关键发现：** 趋势/斜率特征主导性能——7 维原始特征 ≈ 49% → 14 维原始+斜率 ≈ 82% → 57 维 HI 管线 ≈ 85%。显式时序模型（GRU/LSTM/CNN）在 4–8 点序列上无额外收益。

### M2 — 早期退化预警

| 任务 | Accuracy | 备注 |
|------|----------|------|
| 二元退化（样本内） | ~82% | 可行 |
| 二元退化（跨年级） | ~84% | 无泄漏划分 |
| 九类细粒度（跨年级） | ~53% | 困难 |
| 观测窗口消融：g1→g4 | ~44% | |
| 观测窗口消融：g12→g4 | ~46% | |
| 观测窗口消融：g123→g4 | ~54% | 单调提升 |

### M3 — RAG–Bandit 推荐

- 知识库：2,674 块（运动处方、膳食营养、体测标准、健康政策）
- 4 种干预策略：运动 / 膳食 / 综合 / 维持
- **8/9 学生群体收敛**；高退化小群体（93 人，0.3%）不收敛
- Agentic RL 变体累积奖励 2.12（对比规则基线 1.35、随机 1.27）

## 仓库结构

```
health-rag-student-fitness/
├── README.md                  # 本文件
├── LICENSE                    # MIT（代码）/ CC BY-NC（论文）
├── CITATION.cff               # 引用元数据
├── requirements.txt           # Python 依赖
├── .gitignore
├── code/                      # 全部实验脚本 + HealthAgent 应用
│   ├── 01_*.py – 30_*.py     # 编号实验管线
│   ├── health_agent/          # LangGraph 多智能体 Demo（Flask + HTML）
│   ├── m3_agentic_rl/         # M3 Agentic RL 模块
│   └── run_with_retry.bat    # Windows 训练辅助脚本
├── paper/                     # 论文大纲、审计报告、专家评估
│   ├── CHIL2027_paper_outline.md
│   ├── HI九类_代码审计与实验报告.md
│   ├── HI九类_10方法审计与重跑报告.md
│   ├── working_notes/         # 内部方法学讨论
│   └── HI12_supervised_profiling/
├── results/                   # 聚合实验结果（无学生级数据）
│   ├── hi_vs_ml/              # M1 模型对比
│   ├── m2_warning/            # M2 早期预警结果
│   ├── m3_bandit/             # M3 Bandit 对比
│   ├── m3_agentic_rl/         # M3 Agentic RL 结果
│   └── ...
├── knowledge_base/            # 公开来源健康知识库（453 块）
│   ├── chunks/                # 分类 Markdown 块 + INDEX.json
│   └── README.md              # 来源文档说明
├── scripts/                   # 工具脚本（chunk 构建、ModelScope 上传）
└── assets/                    # 图表
```

## 快速开始

### 安装

```bash
git clone https://github.com/LLLLLZ-529/health-rag-student-fitness.git
cd health-rag-student-fitness
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 运行 HealthAgent Demo

```bash
# 先从 ModelScope 下载模型权重（见"模型权重"一节）
# 然后：
cd code/health_agent
python server.py
# 浏览器打开 http://localhost:5000
```

### 命令行使用

```bash
cd code/health_agent
python run.py \
  --query "这个学生该怎么练" \
  --bmi 22 --vc 2800 --str 15 --endurance 270 \
  --speed 9.0 --jump 170 --flex 10
```

## 数据说明

> **重要：** 本仓库**仅包含聚合统计**。原始学生级健康数据（含 `student_id`、性别、民族、个人体测明细）因涉及未成年学生隐私**不予公开**。

原始数据集为 **36,059 名本科生**连续 4 年体质健康测试数据（BMI、肺活量、50m 跑、立定跳远、坐位体前屈、耐力跑、力量，每年 7 项）。

如需用自己的数据复现实验：
1. 按 `code/01_hi_vs_ml.py` 中的字段说明准备宽表 CSV
2. 通过命令行参数或环境变量指定数据路径
3. HI（健康指数）计算与 HI9 标签构建均实现在实验脚本中

## 模型权重

LoRA 适配器权重（基于 Qwen2.5-7B-Instruct 的 SFT 与 DPO 微调）托管在 **ModelScope**：

- **SFT 适配器：** `models/sft_best_model/`（约 165 MB）
- **DPO 适配器：** `models/dpo_model/`（约 165 MB）

下载并放入项目根目录 `models/`：

```bash
# 安装 ModelScope CLI
pip install modelscope

# 下载 SFT 适配器
modelscope download --model anne118/health-rag-sft-7b --local_dir models/sft_best_model

# 下载 DPO 适配器
modelscope download --model anne118/health-rag-dpo-7b --local_dir models/dpo_model
```

基础模型（Qwen2.5-7B-Instruct）首次运行时将自动从 ModelScope 下载。

## 知识库

本仓库包含来自公开政府与 WHO 出版物的 **453 块**知识：

| 类别 | 来源 | 数量 |
|------|------|------|
| 体测标准 | 国民体质测定标准手册、大学生国家体质健康测试评分标准 | 16 |
| 健康政策 | "健康中国2030"规划纲要 | 35 |
| 膳食营养 | 成人肥胖食养指南(2024)、中国居民平衡膳食宝塔、主要食物营养成分表 | 142 |
| 运动处方 | WHO身体活动指南(3部)、全民健身指南(国家体育总局)、全年龄段科学健身指南、慢性疾病运动风险防控指南(征求意见稿)、BROG量表 | 260 |

完整 2,674 块知识库（含 Z-Library 盗版书与 ACSM 版权教材）**不在本仓库**。重建完整知识库的方法见 `knowledge_base/README.md`。

## 复现实验

`code/` 中实验按编号顺序执行：

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 数据与 SFT 数据集 | `01_build_instruction_dataset.py`, `01b_augment_instruction_dataset.py` | 构建指令微调数据集 |
| SFT 训练 | `02_run_sft.py`, `02_run_sft_1.5b.py`, `02_run_sft_7b.py` | LoRA 监督微调 |
| DPO 数据集与训练 | `03_build_dpo_dataset.py`, `03b_`, `03c_`, `04_run_dpo*.py` | DPO 偏好对齐 |
| M1 基准 | `01_hi_vs_ml.py`, `05_run_tabm.py`, `06_raw_vs_hi_features.py`, `08_run_tabpfn25.py`, `10_run_tabiclv2.py`, `11_model_efficiency.py` | 16 模型对比 |
| M2 早期预警 | `07_m2_early_prediction.py`, `09_cross_grade_generalization.py`, `12_temporal_model_comparison.py`, `13-14_m2_decline*.py`, `15_mtg_kill_test.py`, `16_cross_grade_v2.py`, `20-26_m2_warning*.py`, `30_m2_ablation.py` | 早期退化预测 |
| M3 RAG 与 Bandit | `19_m3_rag_recommendation.py`, `21_m3_bandit_comparison.py`, `23_m3_neural_bandit_comparison.py`, `24_m3_rag_bandit_pipeline.py`, `27_m3_agentic_rl.py` | 推荐管线 |
| 评估 | `05_evaluate_models.py`, `05_evaluate_models_1.5b.py` | 模型生成质量评估 |
| 漏斗分析 | `28_funnel_m1_to_m2.py`, `29_funnel_m2_to_m3.py` | 端到端管线分析 |

每个脚本与论文图表的具体对应关系见 `code/README.md`。

## 引用

如使用本仓库代码或结果，请引用：

```bibtex
@misc{healthrag2026,
  title={Beyond Decision Trees: A 16-Model Benchmark of Modern Tabular Models and an Integrated Classification–Early-Warning–Recommendation Pipeline for Student Physical Health},
  author={Project Contributors},
  year={2026},
  note={Submitted to CHIL 2027}
}
```

## 许可证

- **代码：** MIT License
- **论文与文档（`paper/`）：** CC BY-NC 4.0
- **知识库：** 原始公开政府 / WHO 出版物——各来源的许可条款以原文为准

## 致谢

感谢合作高校提供的学生体质健康测试数据，以及为 HI 趋势阈值校准提供盲评的体育医学与体育教育专家。
