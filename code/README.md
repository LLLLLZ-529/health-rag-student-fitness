# Code Index

This directory contains all experiment scripts and the HealthAgent demo application.

## Experiment Pipeline

Scripts are numbered by execution order. Each script corresponds to specific experiments in the paper.

### Data Preparation & Instruction Dataset

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `01_build_instruction_dataset.py` | Build SFT instruction dataset from health test data | §3.2 |
| `01b_augment_instruction_dataset.py` | Augment instruction dataset with diversity strategies | §3.2 |

### SFT Training

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `02_run_sft.py` | SFT training (default model size) | §3.3 |
| `02_run_sft_1.5b.py` | SFT on Qwen2.5-1.5B-Instruct | §3.3, Table 6 |
| `02_run_sft_7b.py` | SFT on Qwen2.5-7B-Instruct | §3.3 |
| `run_with_retry.bat` | Windows helper for automated training retries | — |

### DPO Training

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `03_build_dpo_dataset.py` | Build DPO preference pairs | §3.4 |
| `03b_verify_dpo_dataset.py` | Verify DPO dataset quality | — |
| `03c_dpo_quality_analysis.py` | Analyze DPO preference pair quality | — |
| `04_run_dpo.py` | DPO training (default) | §3.4 |
| `04_run_dpo_1.5b.py` | DPO on Qwen2.5-1.5B-Instruct | §3.4, Table 7 |
| `04_run_dpo_7b.py` | DPO on Qwen2.5-7B-Instruct | §3.4 |

### M1 — Tabular Model Benchmark

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `01_hi_vs_ml.py` | HI pipeline vs classical ML (RF, XGBoost, etc.) | §4.1, Table 1 |
| `05_run_tabm.py` | TabM model training and evaluation | §4.1, Table 1 |
| `06_raw_vs_hi_features.py` | Feature ablation: raw vs slope vs HI features | §4.2, Table 2 |
| `08_run_tabpfn25.py` | TabPFN-2.5 evaluation | §4.1, Table 1 |
| `10_run_tabiclv2.py` | TabICLv2 evaluation | §4.1, Table 1 |
| `11_model_efficiency.py` | Model efficiency comparison (inference time, memory) | §4.3, Table 3 |

### M2 — Early Degradation Warning

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `07_m2_early_prediction.py` | Initial M2 early prediction experiment | §5.1 |
| `09_cross_grade_generalization.py` | Cross-grade generalization (no leakage split) | §5.2, Table 4 |
| `12_temporal_model_comparison.py` | Temporal model comparison (GRU, LSTM, CNN) | §5.3 |
| `13_m2_decline_early_prediction.py` | Binary decline early prediction | §5.1 |
| `14_m2_decline_cross_grade.py` | Binary decline cross-grade | §5.2 |
| `15_mtg_kill_test.py` | Multi-task gradient (MTG) kill test | §5.3 |
| `16_cross_grade_v2.py` | Cross-grade v2 (improved split) | §5.2 |
| `20_m2_warning_decline.py` | M2 warning: decline prediction | §5.1 |
| `21_m2_warning_9class.py` | M2 warning: 9-class prediction | §5.1 |
| `22_m2_warning_cross_grade.py` | M2 warning: cross-grade | §5.2 |
| `23_m2_warning_horizon.py` | M2 warning: observation horizon ablation | §5.4, Table 5 |
| `24_m2_warning_cross_grade_9class.py` | M2 warning: cross-grade 9-class | §5.2 |
| `25_m2_warning_cross_grade_decline.py` | M2 warning: cross-grade decline | §5.2 |
| `26_m2_warning_horizon_ablation.py` | M2 warning: detailed horizon ablation | §5.4 |
| `30_m2_ablation.py` | M2 comprehensive ablation | §5.4 |

### M3 — RAG & Bandit Recommendation

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `19_m3_rag_recommendation.py` | M3 RAG recommendation baseline | §6.1 |
| `21_m3_bandit_comparison.py` | Bandit algorithm comparison (ε-greedy, UCB, LinUCB, Thompson) | §6.2, Table 8 |
| `23_m3_neural_bandit_comparison.py` | Neural bandit comparison | §6.2 |
| `24_m3_rag_bandit_pipeline.py` | End-to-end RAG + bandit pipeline | §6.3 |
| `27_m3_agentic_rl.py` | Agentic RL for M3 | §6.4, Table 9 |

### Model Evaluation

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `05_evaluate_models.py` | Evaluate SFT/DPO model generation quality | §3.5, Table 6–7 |
| `05_evaluate_models_1.5b.py` | Evaluate 1.5B model variants | §3.5 |

### End-to-End Funnel Analysis

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `28_funnel_m1_to_m2.py` | Funnel: M1 classification → M2 warning | §7 |
| `29_funnel_m2_to_m3.py` | Funnel: M2 warning → M3 recommendation | §7 |

## HealthAgent Demo

Located in `health_agent/`:

| File | Description |
|------|-------------|
| `server.py` | Flask server for the web demo |
| `run.py` | CLI entry point |
| `config.py` | Configuration (paths, model settings) |
| `agents.py` | LangGraph agent definitions |
| `graph.py` | LangGraph state graph construction |
| `state.py` | Shared state definitions |
| `tools.py` | Agent tools (RAG retrieval, bandit, etc.) |
| `prompts.py` | System prompts for each agent |
| `demo.html` | Web interface (open via `server.py`) |

## M3 Agentic RL Module

Located in `m3_agentic_rl/`:

| File | Description |
|------|-------------|
| `environment.py` | RL environment for intervention selection |
| `agent.py` | Agentic RL agent implementation |
| `train.py` | Training script |
| `evaluate.py` | Evaluation script |

## Configuration

All scripts accept data paths and output directories via command-line arguments. The `health_agent/config.py` uses environment variables for model paths:

```bash
export SFT_MODEL_PATH=/path/to/sft_best_model
export DPO_MODEL_PATH=/path/to/dpo_model
export BASE_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```
