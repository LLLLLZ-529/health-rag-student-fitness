# Experiment Results

This directory contains **aggregated** experiment results. No student-level data (no `student_id`, no individual records) is included.

## Directory Map

### M1 — Tabular Model Benchmark

| Directory | Contents |
|-----------|----------|
| `hi_vs_ml/` | HI pipeline vs classical ML comparison (accuracy, F1, confusion matrices) |
| `tabm_results/` | TabM model results |
| `tabpfn25_results/` | TabPFN-2.5 results |
| `tabiclv2_results/` | TabICLv2 results |
| `raw_vs_hi_features/` | Feature ablation results (raw vs slope vs HI) |
| `model_efficiency/` | Inference time and memory comparison (`efficiency_results.csv`) |

### M2 — Early Degradation Warning

| Directory | Contents |
|-----------|----------|
| `m2_warning/` | Main M2 warning results (binary decline, 9-class) |
| `m2_cross_grade/` | Cross-grade generalization results |
| `m2_horizon_ablation/` | Observation horizon ablation (g1, g12, g123 → g4) |
| `m2_temporal_models/` | Temporal model comparison (GRU, LSTM, CNN) |
| `m2_ablation/` | Comprehensive M2 ablation study |

### M3 — RAG & Bandit Recommendation

| Directory | Contents |
|-----------|----------|
| `m3_bandit/` | Bandit algorithm comparison (ε-greedy, UCB, LinUCB, Thompson sampling) |
| `m3_neural_bandit/` | Neural bandit results |
| `m3_rag_bandit/` | End-to-end RAG + bandit pipeline results |
| `m3_agentic_rl/` | Agentic RL results (cumulative reward, convergence curves) |

### End-to-End Funnel

| Directory | Contents |
|-----------|----------|
| `funnel_analysis/` | M1→M2→M3 funnel analysis results |

### Model Training & Evaluation

| Directory | Contents |
|-----------|----------|
| `sft_results/` | SFT training logs and metrics |
| `dpo_results/` | DPO training logs and metrics |
| `model_evaluation/` | Generation quality evaluation (BLEU, ROUGE, human eval) |

## Result File Formats

- **CSV**: Aggregated metrics (accuracy, F1, per-class results)
- **JSON**: Detailed experiment configurations and per-run results
- **PNG/SVG**: Plots and figures (confusion matrices, learning curves, convergence)
- **MD**: Summary reports and analysis notes

## Reproducing Results

Each results directory corresponds to one or more scripts in `code/`. See `code/README.md` for the mapping. To reproduce:

1. Prepare your own dataset (see main README for data schema)
2. Run the corresponding experiment script
3. Results will be written to the matching directory under `results/`

## Privacy Note

All results in this directory are aggregated at the model/experiment level. Per-student predictions (`test_predictions.csv`, `high_risk_students.csv`, etc.) have been **excluded**. If you need per-student predictions for your own analysis, run the experiment scripts with your own data.
