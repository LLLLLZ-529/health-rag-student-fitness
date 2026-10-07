# Paired Bootstrap 显著性检验

- 公共 test 样本: 3931
- Bootstrap 次数: B=1000
- 显著性水平: α=0.05

## 每模型 Accuracy + 95% CI

| 模型 | Acc | 2.5% | 97.5% |
|---|---|---|---|
| TabPFN_v2 | 0.8354 | 0.8232 | 0.8466 |
| SAINT | 0.8230 | 0.8112 | 0.8344 |
| TabNet | 0.8215 | 0.8092 | 0.8339 |
| FT_Transformer | 0.8067 | 0.7950 | 0.8186 |
| LightGBM | 0.7750 | 0.7627 | 0.7878 |
| GRANDE | 0.7631 | 0.7504 | 0.7756 |
| XGBoost | 0.7507 | 0.7365 | 0.7639 |
| ResNet1D | 0.7452 | 0.7314 | 0.7583 |

## 显著成对差异 (p < 0.05)

| 优者 | 劣者 | p-value |
|---|---|---|
| FT_Transformer | GRANDE | 0.0000 |
| FT_Transformer | LightGBM | 0.0000 |
| FT_Transformer | ResNet1D | 0.0000 |
| SAINT | FT_Transformer | 0.0000 |
| TabPFN_v2 | FT_Transformer | 0.0000 |
| FT_Transformer | XGBoost | 0.0000 |
| SAINT | GRANDE | 0.0000 |
| TabNet | GRANDE | 0.0000 |
| TabPFN_v2 | GRANDE | 0.0000 |
| LightGBM | ResNet1D | 0.0000 |
| SAINT | LightGBM | 0.0000 |
| TabNet | LightGBM | 0.0000 |
| TabPFN_v2 | LightGBM | 0.0000 |
| LightGBM | XGBoost | 0.0000 |
| SAINT | ResNet1D | 0.0000 |
| TabNet | ResNet1D | 0.0000 |
| TabPFN_v2 | ResNet1D | 0.0000 |
| SAINT | XGBoost | 0.0000 |
| TabNet | XGBoost | 0.0000 |
| TabPFN_v2 | XGBoost | 0.0000 |
| GRANDE | ResNet1D | 0.0120 |
| TabPFN_v2 | TabNet | 0.0260 |
| TabPFN_v2 | SAINT | 0.0400 |
| TabNet | FT_Transformer | 0.0480 |

## 解读
- 显著对越多，说明模型间真实差异越大
- 若 SupCon / TabPFN_v2 与多数 GBDT 显著不同，则 DL 优势成立
- 若仅与 LR / RF 显著，但 GBDT 之间不显著，说明 GBDT 已接近上限