# HI vs ML 对比摘要（v2 — 原始规则复现）

## HI 规则自洽性
- 复现规则：latest total_score [<60, 60-80, ≥80] × HI_slope [<-0.1, ±0.1, >0.1]
- HI 切分 vs `rule_class` (gold): **acc=1.0000**, kappa=1.0000, macro_f1=1.0000
- v1 简化版（HI 平均分 3 分位 + slope sign）仅 28.4%，v2 已修复

## 10 模型对比（有逐样本预测的模型）
| 模型 | ML vs Gold Acc | ML vs Gold F1 | ML vs HI Acc | ML vs HI F1 | 判定 |
|---|---|---|---|---|---|
| TabM | 0.8519 | 0.8060 | 0.8519 | 0.8060 | MISS |
| ModernNCA | 0.8474 | 0.8073 | 0.8474 | 0.8073 | MISS |
| TabPFN_v2 | 0.8354 | 0.7753 | 0.8354 | 0.7753 | MISS |
| SAINT | 0.8292 | 0.7606 | 0.8292 | 0.7606 | MISS |
| TabNet | 0.8249 | 0.7506 | 0.8249 | 0.7506 | MISS |
| FT_Transformer | 0.8197 | 0.7341 | 0.8197 | 0.7341 | MISS |
| LightGBM | 0.7837 | 0.6803 | 0.7837 | 0.6803 | MISS |
| GRANDE | 0.7627 | 0.7057 | 0.7627 | 0.7057 | MISS |
| XGBoost | 0.7626 | 0.6799 | 0.7626 | 0.6799 | MISS |
| ResNet1D | 0.7478 | 0.6655 | 0.7478 | 0.6655 | MISS |

## 解读
- `ML vs Gold`：模型预测 vs 9 类金标准（rule_class）
- `ML vs HI`：模型预测 vs 复现的 HI 规则（v2 自洽性 100%，即等于 gold）
- 由于 HI 规则与 gold 完全一致，ML vs HI ≈ ML vs Gold
- 两者差异反映模型在 HI 规则边界样本上的表现
- SupCon/RBF_SVM/Logistic/RF 仅有汇总 status.json，无逐样本预测，未纳入本表