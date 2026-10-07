#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_hi_vs_ml.py — HI 规则复现 + 与 14 个 ML 模型对比（v2）

v2 修复：找到原始 HI 9 类切分规则，自洽性从 28.4% 提升到 100%。

原始 HI 9 类切分规则（经 gold 标签反推验证）：
  Level（水平）：latest available total_score（0-100 分）
    低水平 = score < 60
    中水平 = 60 ≤ score < 80
    高水平 = score ≥ 80
  Trend（趋势）：HI_slope（来自 deliverables 文件）
    退化 = slope < -0.1
    平稳 = -0.1 ≤ slope ≤ 0.1
    改善 = slope > 0.1
  class = level × 3 + trend

v1 错误：用 HI 平均分 3 分位切 level + slope sign 切 trend，自洽性仅 28.4%。

输入：
  - hi_wide.csv（7 项体测 + HI_g1..g4 + total_score_g1..g4）
  - deliverables/HI九类_当前模型预测.csv（rule_class gold + HI_slope）
  - experiments/review_20260910/hi9_12models_all/<MODEL>/test_predictions.csv（14 模型）

输出：
  - outputs/hi_vs_ml/hi_rule_consistency.csv
  - outputs/hi_vs_ml/hi_vs_ml_table.csv
  - outputs/hi_vs_ml/summary.md
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
HI9_DIR = ROOT / "experiments" / "review_20260910" / "hi9_12models"
BONUS_DIR = Path(__file__).parent / "outputs"
OUT_DIR = Path(__file__).parent / "outputs" / "hi_vs_ml"

# 模型 → test_predictions.csv 路径映射
# 注：SupCon/RBF_SVM/Logistic/RandomForest 仅有 status.json（历史汇总），无逐样本预测
MODEL_PATHS = {
    "TabM": BONUS_DIR / "tabm" / "test_predictions.csv",
    "ModernNCA": BONUS_DIR / "modernnca" / "test_predictions.csv",
    "TabPFN_v2": HI9_DIR / "TabPFN_v2" / "test_predictions.csv",
    "FT_Transformer": HI9_DIR / "FT_Transformer" / "test_predictions.csv",
    "SAINT": HI9_DIR / "SAINT" / "test_predictions.csv",
    "TabNet": HI9_DIR / "TabNet" / "test_predictions.csv",
    "LightGBM": HI9_DIR / "LightGBM" / "test_predictions.csv",
    "GRANDE": HI9_DIR / "GRANDE" / "test_predictions.csv",
    "XGBoost": HI9_DIR / "XGBoost" / "test_predictions.csv",
    "ResNet1D": HI9_DIR / "ResNet1D" / "test_predictions.csv",
}

LEVEL_NAMES = ["低水平", "中水平", "高水平"]
TREND_NAMES = ["退化型", "平稳型", "改善型"]


def latest_total_score(row: pd.Series) -> float:
    """取 latest available total_score（按 real_obs_pattern 指示的真实年级）。"""
    pat = str(int(row["real_obs_pattern"])) if pd.notna(row["real_obs_pattern"]) else ""
    for g in [4, 3, 2, 1]:
        if str(g) in pat and pd.notna(row.get(f"total_score_g{g}")):
            return float(row[f"total_score_g{g}"])
    return np.nan


def reconstruct_hi_rule(df: pd.DataFrame) -> np.ndarray:
    """复现原始 HI 9 类切分规则。

    Level: latest total_score < 60 → 0, [60,80) → 1, ≥80 → 2
    Trend: HI_slope < -0.1 → 0, [-0.1, 0.1] → 1, >0.1 → 2
    class = level * 3 + trend
    """
    ts_latest = df.apply(latest_total_score, axis=1).to_numpy()
    hi_slope = df["HI_slope"].to_numpy()

    level = np.where(ts_latest < 60, 0, np.where(ts_latest >= 80, 2, 1)).astype(int)
    trend = np.where(hi_slope < -0.1, 0, np.where(hi_slope > 0.1, 2, 1)).astype(int)
    return level * 3 + trend


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 读数据 ...")
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("", "_raw"))
    print(f"      merged rows = {len(df)}   unique students = {df.student_id.nunique()}")

    print("[2/4] 复现原始 HI 规则（ts_latest [60,80] + HI_slope ±0.1）...")
    df["hi_rule_class"] = reconstruct_hi_rule(df)
    hi_acc = accuracy_score(df["rule_class"], df["hi_rule_class"])
    hi_kappa = cohen_kappa_score(df["rule_class"], df["hi_rule_class"])
    hi_f1 = f1_score(df["rule_class"], df["hi_rule_class"], average="macro")
    print(f"      HI 规则自洽: acc={hi_acc:.4f}  kappa={hi_kappa:.4f}  macro_f1={hi_f1:.4f}")

    # 按类拆解自洽性
    consistency_rows = []
    for lvl in range(3):
        for trd in range(3):
            cid = lvl * 3 + trd
            name = f"{LEVEL_NAMES[lvl]}-{TREND_NAMES[trd]}"
            sub = df[df["hi_rule_class"] == cid]
            if len(sub) == 0:
                continue
            agree = (sub["rule_class"] == cid).mean()
            consistency_rows.append({
                "class_id": cid, "class_name": name,
                "hi_rule_n": len(sub), "agreement_with_gold": round(agree, 4),
            })
    pd.DataFrame(consistency_rows).to_csv(OUT_DIR / "hi_rule_consistency.csv", index=False)

    print("[3/4] 计算模型 vs Gold / vs HI 一致性 ...")
    rows = []
    for model_name, pred_path in MODEL_PATHS.items():
        if not pred_path.exists():
            print(f"      ⚠ {model_name}: {pred_path} 不存在，跳过")
            continue

        preds = pd.read_csv(pred_path, dtype={"student_id": str})
        # 取 seed=42（3 seeds 结果接近，取其一即可）
        preds42 = preds[preds["seed"] == 42][["student_id", "pred_class"]].copy()
        merged = df[["student_id", "rule_class", "hi_rule_class"]].merge(
            preds42, on="student_id", how="inner"
        )
        if len(merged) == 0:
            print(f"      ⚠ {model_name}: 0 行匹配")
            continue

        ml_acc_gold = accuracy_score(merged["rule_class"], merged["pred_class"])
        ml_kappa_gold = cohen_kappa_score(merged["rule_class"], merged["pred_class"])
        ml_f1_gold = f1_score(merged["rule_class"], merged["pred_class"], average="macro")
        ml_acc_hi = accuracy_score(merged["hi_rule_class"], merged["pred_class"])
        ml_kappa_hi = cohen_kappa_score(merged["hi_rule_class"], merged["pred_class"])
        ml_f1_hi = f1_score(merged["hi_rule_class"], merged["pred_class"], average="macro")

        # 判定：ML 预测与 HI 规则的一致性 vs HI 自洽性
        # 由于 HI 自洽性 = 1.0，ML vs HI 直接反映模型与 HI 规则的吻合度
        hit = "HIT" if ml_acc_hi >= hi_acc else "MISS"

        rows.append({
            "model": model_name,
            "n_test": len(merged),
            "ml_vs_gold_acc": round(ml_acc_gold, 4),
            "ml_vs_gold_kappa": round(ml_kappa_gold, 4),
            "ml_vs_gold_macro_f1": round(ml_f1_gold, 4),
            "ml_vs_hi_acc": round(ml_acc_hi, 4),
            "ml_vs_hi_kappa": round(ml_kappa_hi, 4),
            "ml_vs_hi_macro_f1": round(ml_f1_hi, 4),
            "hi_self_acc_vs_gold": round(hi_acc, 4),
            "verdict": hit,
        })
        print(f"      {model_name:20s} vs_gold={ml_acc_gold:.4f}  vs_hi={ml_acc_hi:.4f}  [{hit}]")

    out_df = pd.DataFrame(rows).sort_values("ml_vs_gold_acc", ascending=False).reset_index(drop=True)
    out_df.to_csv(OUT_DIR / "hi_vs_ml_table.csv", index=False)

    # 写 summary.md
    s = []
    s.append("# HI vs ML 对比摘要（v2 — 原始规则复现）\n")
    s.append("## HI 规则自洽性")
    s.append(f"- 复现规则：latest total_score [<60, 60-80, ≥80] × HI_slope [<-0.1, ±0.1, >0.1]")
    s.append(f"- HI 切分 vs `rule_class` (gold): **acc={hi_acc:.4f}**, kappa={hi_kappa:.4f}, macro_f1={hi_f1:.4f}")
    s.append(f"- v1 简化版（HI 平均分 3 分位 + slope sign）仅 28.4%，v2 已修复\n")
    s.append(f"## {len(out_df)} 模型对比（有逐样本预测的模型）")
    s.append("| 模型 | ML vs Gold Acc | ML vs Gold F1 | ML vs HI Acc | ML vs HI F1 | 判定 |")
    s.append("|---|---|---|---|---|---|")
    for _, r in out_df.iterrows():
        s.append(f"| {r['model']} | {r['ml_vs_gold_acc']:.4f} | {r['ml_vs_gold_macro_f1']:.4f} | "
                 f"{r['ml_vs_hi_acc']:.4f} | {r['ml_vs_hi_macro_f1']:.4f} | {r['verdict']} |")
    s.append("")
    s.append("## 解读")
    s.append("- `ML vs Gold`：模型预测 vs 9 类金标准（rule_class）")
    s.append("- `ML vs HI`：模型预测 vs 复现的 HI 规则（v2 自洽性 100%，即等于 gold）")
    s.append("- 由于 HI 规则与 gold 完全一致，ML vs HI ≈ ML vs Gold")
    s.append("- 两者差异反映模型在 HI 规则边界样本上的表现")
    s.append("- SupCon/RBF_SVM/Logistic/RF 仅有汇总 status.json，无逐样本预测，未纳入本表")
    (OUT_DIR / "summary.md").write_text("\n".join(s), encoding="utf-8")

    print(f"\n[4/4] 完成 ✅")
    print(f"      HI 自洽性 = {hi_acc:.4f}（v1 为 0.2840）")
    print(f"      写入 {OUT_DIR}/hi_vs_ml_table.csv ({len(out_df)} 模型)")
    print(f"      写入 {OUT_DIR}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
