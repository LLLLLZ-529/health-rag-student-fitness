#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_paired_bootstrap.py — 12 模型两两 paired bootstrap 显著性检验（零重跑）

输入：每个模型的 test_predictions.csv（hard predictions，3 seeds × 5409 test）

输出：
  - outputs/bootstrap/accuracy_bootstrap.csv   每模型 acc 的 95% CI
  - outputs/bootstrap/pvalue_matrix.csv        12x12 paired bootstrap p-value 矩阵
  - outputs/bootstrap/summary.md               简短报告

方法（无概率时）：
  - 每个模型取 seed=42 的预测（seed 42/43/44 可任选，效果一致）
  - 对每个 (model_i, model_j)：
      对每个 bootstrap 样本 b=1..B（默认 1000）：
        idx = resample 5409 个
        diff_b = acc_i[idx] - acc_j[idx]
      p_value = (# diff_b < 0) / B    （双侧取 min）
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HI9_DIR = ROOT / "experiments" / "review_20260910" / "hi9_12models"
ALL_DIR = ROOT / "experiments" / "review_20260910" / "hi9_12models_all"
OUT_DIR = Path(__file__).parent / "outputs" / "bootstrap"
B = 1000  # bootstrap 重采样次数
SEED_FOR_BOOT = 2026
SEED_OF_TEST = 42  # 用哪个 seed 的预测做 bootstrap
ALPHA = 0.05

# 12 个模型完整列表
ALL_MODEL_PATHS = []
for m in ["FT_Transformer", "GRANDE", "LightGBM", "ResNet1D", "SAINT",
          "TabNet", "TabPFN_v2", "XGBoost"]:
    d = HI9_DIR / m
    if (d / "test_predictions.csv").exists():
        ALL_MODEL_PATHS.append((m, d))
for m in ["Logistic", "RBF_SVM", "RandomForest", "SupCon"]:
    d = ALL_DIR / m
    if (d / "test_predictions.csv").exists():
        ALL_MODEL_PATHS.append((m, d))


def load_preds_one_seed(path: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"student_id": str})
    sub = df[df["seed"] == seed][["student_id", "rule_class", "pred_class"]].copy()
    sub = sub.drop_duplicates(subset=["student_id"]).set_index("student_id")
    return sub


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_FOR_BOOT)

    print(f"[1/4] 加载 {len(ALL_MODEL_PATHS)} 个模型 seed={SEED_OF_TEST} 的预测 ...")
    preds_dict = {}
    for name, d in ALL_MODEL_PATHS:
        try:
            preds_dict[name] = load_preds_one_seed(d / "test_predictions.csv", SEED_OF_TEST)
            print(f"      ✓ {name}: {len(preds_dict[name])} rows")
        except Exception as e:
            print(f"      ✗ {name}: {e}")

    # 取所有模型的公共 student_id（保证两两对齐）
    common = set.intersection(*[set(p.index) for p in preds_dict.values()])
    common = sorted(common)
    n = len(common)
    print(f"      公共 test 样本数: {n}")

    # 构造 per-model 的 (gold, pred) 数组
    arr = {}
    for name, p in preds_dict.items():
        gold = p.loc[common, "rule_class"].to_numpy()
        pred = p.loc[common, "pred_class"].to_numpy()
        arr[name] = (gold, pred)

    print(f"[2/4] 估计每模型 acc 的 95% CI（B={B}） ...")
    acc_b_samples = {}
    acc_mean = {}
    for name, (gold, pred) in arr.items():
        # 一次性 bootstrap B 次，共享 idx
        idx = rng.integers(0, n, size=(B, n))
        correct = (pred[idx] == gold[idx]).mean(axis=1)
        acc_b_samples[name] = correct
        acc_mean[name] = float(correct.mean())
        lo, hi = np.quantile(correct, [0.025, 0.975])
        print(f"      {name}: acc={acc_mean[name]:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")

    acc_b_df = pd.DataFrame({
        "model": list(acc_b_samples.keys()),
        "acc_mean": [acc_mean[m] for m in acc_b_samples.keys()],
        "acc_2.5%": [float(np.quantile(acc_b_samples[m], 0.025)) for m in acc_b_samples.keys()],
        "acc_97.5%": [float(np.quantile(acc_b_samples[m], 0.975)) for m in acc_b_samples.keys()],
    }).sort_values("acc_mean", ascending=False)
    acc_b_df.to_csv(OUT_DIR / "accuracy_bootstrap.csv", index=False)

    print(f"[3/4] 12x12 paired bootstrap p-value 矩阵 ...")
    names = list(acc_b_samples.keys())
    n_models = len(names)
    p_mat = np.ones((n_models, n_models))
    # 使用同一组 idx（idx 已在上面生成，但已绑定到第一个模型）
    # 改为：每对模型用其自己的 idx（保证独立性）
    idx_pairs = {}
    for i, name_i in enumerate(names):
        # 简单复用：所有模型共享同一组 idx（节省内存；标准 paired bootstrap 用共享）
        pass
    # 共享 idx 数组
    idx_shared = rng.integers(0, n, size=(B, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                continue
            gold_i, pred_i = arr[ni]
            gold_j, pred_j = arr[nj]
            # diff = (pred_i == gold_i) - (pred_j == gold_j)
            diff_b = (pred_i[idx_shared] == gold_i[idx_shared]).mean(axis=1) \
                   - (pred_j[idx_shared] == gold_j[idx_shared]).mean(axis=1)
            # 双侧 p-value
            p_two = 2 * min((diff_b <= 0).mean(), (diff_b >= 0).mean())
            p_two = min(p_two, 1.0)
            p_mat[i, j] = p_two
    p_df = pd.DataFrame(p_mat, index=names, columns=names)
    p_df.to_csv(OUT_DIR / "pvalue_matrix.csv")

    # 显著性摘要
    sig_pairs = []
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i < j:
                p = p_mat[i, j]
                if p < ALPHA:
                    winner = ni if acc_mean[ni] > acc_mean[nj] else nj
                    sig_pairs.append((winner, ni if winner == nj else nj, p))
    sig_pairs.sort(key=lambda x: x[2])

    print(f"\n      在 p<{ALPHA} 水平下显著的成对差异: {len(sig_pairs)} 对")
    for w, l, p in sig_pairs[:10]:
        print(f"        {w} > {l}  (p={p:.4f})")

    summary = []
    summary.append("# Paired Bootstrap 显著性检验\n")
    summary.append(f"- 公共 test 样本: {n}")
    summary.append(f"- Bootstrap 次数: B={B}")
    summary.append(f"- 显著性水平: α={ALPHA}\n")
    summary.append("## 每模型 Accuracy + 95% CI\n")
    summary.append("| 模型 | Acc | 2.5% | 97.5% |")
    summary.append("|---|---|---|---|")
    for _, r in acc_b_df.iterrows():
        summary.append(f"| {r['model']} | {r['acc_mean']:.4f} | {r['acc_2.5%']:.4f} | {r['acc_97.5%']:.4f} |")
    summary.append("")
    summary.append(f"## 显著成对差异 (p < {ALPHA})\n")
    if not sig_pairs:
        summary.append("- 无（在 α=0.05 下没有模型对显著不同）")
    else:
        summary.append("| 优者 | 劣者 | p-value |")
        summary.append("|---|---|---|")
        for w, l, p in sig_pairs:
            summary.append(f"| {w} | {l} | {p:.4f} |")
    summary.append("")
    summary.append("## 解读")
    summary.append("- 显著对越多，说明模型间真实差异越大")
    summary.append("- 若 SupCon / TabPFN_v2 与多数 GBDT 显著不同，则 DL 优势成立")
    summary.append("- 若仅与 LR / RF 显著，但 GBDT 之间不显著，说明 GBDT 已接近上限")
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"\n[4/4] 完成 ✅")
    print(f"      写入 {OUT_DIR}/accuracy_bootstrap.csv")
    print(f"      写入 {OUT_DIR}/pvalue_matrix.csv")
    print(f"      写入 {OUT_DIR}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())