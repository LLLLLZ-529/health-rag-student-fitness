# Student Physical Health RAG

> 16-Model Tabular Benchmark, Early Degradation Warning, and RAG–Bandit Personalized Recommendation for Student Physical Fitness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: CHIL 2027](https://img.shields.io/badge/Paper-CHIL%202027-blue.svg)](paper/CHIL2027_paper_outline.md)
[![Data: Aggregated Only](https://img.shields.io/badge/Data-Aggregated%20Only-green.svg)](#data)

## Overview

This project builds a three-module system for monitoring and intervening on student physical health using longitudinal physical-fitness test data:

| Module | Name | Task |
|--------|------|------|
| **M1** | Tabular Model Benchmark | 16 modern tabular models on 9-class health-state classification (HI9) |
| **M2** | Early Degradation Warning | Cross-grade prediction of future health decline from early observations |
| **M3** | RAG–Bandit Recommendation | TF-IDF retrieval over a health knowledge base + contextual bandit for personalized intervention selection |

Additionally, **HealthAgent** is a LangGraph-based multi-agent demo that wraps the pipeline with an SFT/DPO-fine-tuned Qwen2.5-7B-Instruct language model, providing an interactive web interface for health assessment and exercise recommendation.

## Key Results

### M1 — 16-Model Benchmark (HI9 Classification)

| Model Family | Best Model | Accuracy | Macro-F1 |
|-------------|-----------|----------|----------|
| Tabular foundation / pretrained | TabM | 0.859 | — |
| | TabPFN-2.5 | 0.856 | — |
| | ModernNCA | 0.855 | — |
| Contrastive | SupCon | 0.845 | 0.803 |
| Classical | RandomForest | 0.630 | — |

**Key finding:** Trend/slope features dominate performance — 7-D raw ≈ 49% → 14-D raw+slope ≈ 82% → 57-D HI pipeline ≈ 85%. Explicit temporal models (GRU/LSTM/CNN) add no gain on 4–8 point sequences.

### M2 — Early Degradation Warning

| Task | Accuracy | Notes |
|------|----------|-------|
| Binary decline (in-sample) | ~82% | Feasible |
| Binary decline (cross-grade) | ~84% | No leakage split |
| 9-class fine-grained (cross-grade) | ~53% | Hard |
| Horizon ablation: g1→g4 | ~44% | |
| Horizon ablation: g12→g4 | ~46% | |
| Horizon ablation: g123→g4 | ~54% | Monotonic improvement |

### M3 — RAG–Bandit Recommendation

- Knowledge base: 2,674 chunks (exercise prescription, diet/nutrition, fitness standards, health policy)
- 4 intervention strategies: exercise / diet / combined / maintain
- **8/9 student groups converge**; the small high-level-declining stratum (93 students, 0.3%) does not
- Agentic RL variant achieves cumulative reward 2.12 (vs. 1.35 rule-based, 1.27 random)

## Repository Structure

```
health-rag-student-fitness/
├── README.md                  # This file
├── LICENSE                    # MIT (code) / CC BY-NC (paper)
├── CITATION.cff               # Citation metadata
├── requirements.txt           # Python dependencies
├── .gitignore
├── code/                      # All experiment scripts + HealthAgent app
│   ├── 01_*.py – 30_*.py     # Numbered experiment pipeline
│   ├── health_agent/          # LangGraph multi-agent demo (Flask + HTML)
│   ├── m3_agentic_rl/        # Agentic RL module for M3
│   └── run_with_retry.bat    # Windows training helper
├── paper/                     # Paper outline, audit reports, expert evaluation
│   ├── CHIL2027_paper_outline.md
│   ├── HI九类_代码审计与实验报告.md
│   ├── HI九类_10方法审计与重跑报告.md
│   ├── working_notes/         # Internal methodological discussions
│   └── HI12_supervised_profiling/
├── results/                   # Aggregated experiment results (no student-level data)
│   ├── hi_vs_ml/              # M1 model comparison
│   ├── m2_warning/            # M2 early warning results
│   ├── m3_bandit/             # M3 bandit comparison
│   ├── m3_agentic_rl/         # M3 agentic RL results
│   └── ...
├── knowledge_base/            # Public-source health knowledge chunks (~453)
│   ├── chunks/                # Categorized markdown chunks + INDEX.json
│   └── README.md              # Source documentation
├── scripts/                   # Utility scripts (chunk builder, ModelScope upload)
└── assets/                    # Figures and diagrams
```

## Quick Start

### Installation

```bash
git clone https://github.com/LLLLLZ-529/health-rag-student-fitness.git
cd health-rag-student-fitness
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the HealthAgent Demo

```bash
# Download model weights from ModelScope (see Model Weights section)
# Then:
cd code/health_agent
python server.py
# Open http://localhost:5000 in your browser
```

### CLI Usage

```bash
cd code/health_agent
python run.py \
  --query "这个学生该怎么练" \
  --bmi 22 --vc 2800 --str 15 --endurance 270 \
  --speed 9.0 --jump 170 --flex 10
```

## Data

> **Important:** This repository contains **only aggregated statistics**. Raw student-level health data (including `student_id`, gender, ethnicity, and individual test measurements) is **not included** due to privacy concerns for minor students.

The original dataset comprises **36,059 undergraduate students** with 4 years of physical-fitness test data (BMI, vital capacity, 50m sprint, standing long jump, sit-and-reach, endurance run, strength), across 7 test items per year.

To reproduce experiments with your own data:
1. Prepare a wide-format CSV with columns matching the expected schema (see `code/01_hi_vs_ml.py` for reference)
2. Set the data path in the experiment scripts via command-line arguments or environment variables
3. The HI (Health Index) computation and HI9 label construction are implemented in the experiment scripts

## Model Weights

LoRA adapter weights (SFT and DPO fine-tuned on Qwen2.5-7B-Instruct) are hosted on **ModelScope**:

- **SFT adapter:** `models/sft_best_model/` (~165 MB)
- **DPO adapter:** `models/dpo_model/` (~165 MB)

Download and place them under `models/` in the project root:

```bash
# Install ModelScope CLI
pip install modelscope

# Download SFT adapter
modelscope download --model anne118/health-rag-sft-7b --local_dir models/sft_best_model

# Download DPO adapter
modelscope download --model anne118/health-rag-dpo-7b --local_dir models/dpo_model
```

The base model (Qwen2.5-7B-Instruct) will be automatically downloaded from ModelScope on first run.

## Knowledge Base

The repository includes **453 chunks** from public government and WHO publications:

| Category | Sources | Count |
|----------|---------|-------|
| 体测标准 (Fitness Standards) | 国民体质测定标准手册, 大学生国家体质健康测试评分标准 | 16 |
| 健康政策 (Health Policy) | "健康中国2030"规划纲要 | 35 |
| 膳食营养 (Diet & Nutrition) | 成人肥胖食养指南(2024), 中国居民平衡膳食宝塔, 主要食物营养成分表 | 142 |
| 运动处方 (Exercise Prescription) | WHO身体活动指南(3部), 全民健身指南(国家体育总局), 全年龄段科学健身指南, 慢性疾病运动风险防控指南(征求意见稿), BROG量表 | 260 |

The full 2,674-chunk knowledge base (including copyrighted books from Z-Library and ACSM) is **not included** in this repository. See `knowledge_base/README.md` for details on rebuilding the full knowledge base.

## Reproducing Experiments

Experiments are numbered sequentially in `code/`:

| Stage | Scripts | Description |
|-------|---------|-------------|
| Data & SFT dataset | `01_build_instruction_dataset.py`, `01b_augment_instruction_dataset.py` | Build instruction tuning dataset |
| SFT training | `02_run_sft.py`, `02_run_sft_1.5b.py`, `02_run_sft_7b.py` | Supervised fine-tuning with LoRA |
| DPO dataset & training | `03_build_dpo_dataset.py`, `03b_`, `03c_`, `04_run_dpo*.py` | DPO preference alignment |
| M1 benchmark | `01_hi_vs_ml.py`, `05_run_tabm.py`, `06_raw_vs_hi_features.py`, `08_run_tabpfn25.py`, `10_run_tabiclv2.py`, `11_model_efficiency.py` | 16-model comparison |
| M2 early warning | `07_m2_early_prediction.py`, `09_cross_grade_generalization.py`, `12_temporal_model_comparison.py`, `13-14_m2_decline*.py`, `15_mtg_kill_test.py`, `16_cross_grade_v2.py`, `20-26_m2_warning*.py`, `30_m2_ablation.py` | Early degradation prediction |
| M3 RAG & bandit | `19_m3_rag_recommendation.py`, `21_m3_bandit_comparison.py`, `23_m3_neural_bandit_comparison.py`, `24_m3_rag_bandit_pipeline.py`, `27_m3_agentic_rl.py` | Recommendation pipeline |
| Evaluation | `05_evaluate_models.py`, `05_evaluate_models_1.5b.py` | Model generation quality evaluation |
| Funnel analysis | `28_funnel_m1_to_m2.py`, `29_funnel_m2_to_m3.py` | End-to-end pipeline analysis |

See `code/README.md` for a detailed mapping of each script to its corresponding paper table/figure.

## Citation

If you use this code or results in your research, please cite:

```bibtex
@misc{healthrag2026,
  title={Beyond Decision Trees: A 16-Model Benchmark of Modern Tabular Models and an Integrated Classification–Early-Warning–Recommendation Pipeline for Student Physical Health},
  author={Project Contributors},
  year={2026},
  note={Submitted to CHIL 2027}
}
```

## License

- **Code:** MIT License
- **Paper & documentation (`paper/`):** CC BY-NC 4.0
- **Knowledge base chunks:** Original public government / WHO publications — refer to each source for license terms

## Acknowledgments

This work was supported by the student physical-fitness testing data from a partner university. We thank the sports medicine and physical education experts who provided blind evaluation of the HI trend threshold calibration.
