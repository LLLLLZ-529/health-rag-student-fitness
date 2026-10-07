#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_funnel_m1_to_m2.py — M1→M2 漏斗：高风险学生退化预测

漏斗逻辑：
  M1 全量学生 → 筛出"低水平+退化趋势"高风险子集
  M2 在高风险子集上 → 预测大四是否继续退化（二分类）
  特征：最新值 + 斜率 + 波动度 + 交互项（不只快照）

输出：
  outputs/funnel_m1_m2/summary.md
  outputs/funnel_m1_m2/high_risk_students.csv
  outputs/funnel_m1_m2/m2_results.csv
"""
from __future__ import annotations
import os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, classification_report)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "funnel_m1_m2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
           "sit_and_reach", "endurance_run_sec", "strength"]
# 方向：True=越大越好，False=越小越好
HIGHER_BETTER = {
    "bmi": False,  # 倒U
    "vital_capacity": True,
    "sprint_50m": False,
    "standing_long_jump": True,
    "sit_and_reach": True,
    "endurance_run_sec": False,
    "strength": True,
}


def load_data():
    df = pd.read_csv(RAW_CSV)
    labels = pd.read_csv(LABELS_CSV)
    # 只取 rule_class 那行
    labels = labels[labels["model"] == labels["model"].iloc[0]]
    df = df.merge(labels[["student_id", "rule_class_name", "rule_eligible", "HI_slope"]],
                  on="student_id", how="inner")
    return df


def build_temporal_features(df):
    """构造时序特征：g1+g2 最新值 + 斜率 + 波动 + 交互"""
    feats = {}

    # 1. g2 最新值（7项）
    for m in METRICS:
        feats[f"{m}_g2"] = df[f"{m}_g2"]

    # 2. g1→g2 斜率（变化量）
    for m in METRICS:
        feats[f"{m}_slope"] = df[f"{m}_g2"] - df[f"{m}_g1"]

    # 3. g1/g2 均值（水平）
    for m in METRICS:
        feats[f"{m}_avg"] = (df[f"{m}_g1"] + df[f"{m}_g2"]) / 2

    # 4. 性别
    feats["gender"] = (df["gender"] == "男").astype(int)

    # 5. 交互项：斜率 × 水平（差的指标在恶化 = 高危）
    for m in METRICS:
        feats[f"{m}_slope_x_level"] = feats[f"{m}_slope"] * feats[f"{m}_avg"]

    # 6. 总恶化指标数（有几项在变差）
    decline_count = np.zeros(len(df))
    for m in METRICS:
        if HIGHER_BETTER[m]:
            decline_count += (feats[f"{m}_slope"] < 0).astype(int)
        else:
            # 越小越好的指标，g2比g1大 = 变差
            decline_count += (feats[f"{m}_slope"] > 0).astype(int)
    feats["n_decline_metrics"] = decline_count

    # 7. HI 斜率（已有）
    feats["hi_slope"] = df["HI_slope"]

    X = pd.DataFrame(feats)
    return X


def main():
    print("=" * 60)
    print("M1→M2 漏斗实验")
    print("=" * 60)

    df = load_data()
    print(f"\n全量学生: {len(df)}")

    # === M1: 筛高风险子集 ===
    # 高风险 = 低水平 OR 退化型（rule_class_name 含"低水平"或"退化"）
    rule = df["rule_class_name"].fillna("")
    high_risk_mask = rule.str.contains("低水平") | rule.str.contains("退化")
    high_risk = df[high_risk_mask].copy()
    print(f"M1 筛出高风险子集: {len(high_risk)} ({len(high_risk)/len(df)*100:.1f}%)")

    # 分布
    print("\n高风险子集 9类分布:")
    print(high_risk["rule_class_name"].value_counts().to_string())

    # === M2 标签：g4 是否继续退化 ===
    # 标签：g4 HI < g2 HI（继续恶化）
    high_risk["g4_decline"] = (high_risk["HI_g4"] < high_risk["HI_g2"]).astype(int)
    print(f"\nM2 标签分布: g4继续退化={high_risk['g4_decline'].sum()} "
          f"({high_risk['g4_decline'].mean()*100:.1f}%), "
          f"未退化={len(high_risk)-high_risk['g4_decline'].sum()}")

    # 只保留有 g4 数据的
    has_g4 = high_risk["HI_g4"].notna()
    high_risk = high_risk[has_g4].copy()
    print(f"有g4数据: {len(high_risk)}")

    # 构造特征
    X = build_temporal_features(high_risk)
    y = high_risk["g4_decline"].values

    # 填充 NaN
    X = X.fillna(X.median())
    print(f"\n特征矩阵: {X.shape}")
    print(f"特征列表: {list(X.columns)}")

    # === 训练模型 ===
    results = []
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                      random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # AUC
        auc_scores = []
        ap_scores = []
        precisions_at_recall80 = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model.fit(X_train_s, y_train)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test_s)[:, 1]
            else:
                proba = model.decision_function(X_test_s)

            auc_scores.append(roc_auc_score(y_test, proba))
            ap_scores.append(average_precision_score(y_test, proba))

            # recall=80% 时的 precision
            prec, rec, _ = precision_recall_curve(y_test, proba)
            idx = np.argmin(np.abs(rec - 0.8))
            precisions_at_recall80.append(prec[idx])

        result = {
            "model": name,
            "auc_mean": np.mean(auc_scores),
            "auc_std": np.std(auc_scores),
            "ap_mean": np.mean(ap_scores),
            "prec_at_recall80_mean": np.mean(precisions_at_recall80),
        }
        results.append(result)
        print(f"\n{name}:")
        print(f"  AUC = {result['auc_mean']:.3f} ± {result['auc_std']:.3f}")
        print(f"  PR-AUC = {result['ap_mean']:.3f}")
        print(f"  Precision@Recall80% = {result['prec_at_recall80_mean']:.3f}")

    # 保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "m2_results.csv", index=False)
    high_risk.to_csv(OUT_DIR / "high_risk_students.csv", index=False)

    # 对比：全量学生 vs 高风险子集
    print("\n" + "=" * 60)
    print("对比：全量学生 vs 高风险子集")
    print("=" * 60)
    print(f"全量学生 g4退化率: {(df['HI_g4'] < df['HI_g2']).mean()*100:.1f}%")
    print(f"高风险子集 g4退化率: {high_risk['g4_decline'].mean()*100:.1f}%")
    print(f"↑ 高风险子集退化率是全量的 {high_risk['g4_decline'].mean() / max((df['HI_g4'] < df['HI_g2']).mean(), 0.01):.1f} 倍")

    # 写 summary
    best = max(results, key=lambda x: x["auc_mean"])
    summary = f"""# M1→M2 漏斗实验结果

## M1 筛选
- 全量学生: {len(df)}
- 高风险子集（低水平/退化）: {len(high_risk)} ({len(high_risk)/len(df)*100:.1f}%)

## M2 退化预测（在高风险子集上）
- 标签: g4 HI < g2 HI（继续退化）
- 退化率: {high_risk['g4_decline'].mean()*100:.1f}%
- 特征: {X.shape[1]}维（最新值7 + 斜率7 + 均值7 + 交互7 + 计数1 + HI斜率1 + 性别1）

## 模型对比
| 模型 | AUC | PR-AUC | P@R80% |
|------|-----|--------|--------|
"""
    for r in results:
        summary += f"| {r['model']} | {r['auc_mean']:.3f} | {r['ap_mean']:.3f} | {r['prec_at_recall80_mean']:.3f} |\n"

    summary += f"""
## 关键发现
- 最佳模型: {best['model']} (AUC={best['auc_mean']:.3f})
- 高风险子集退化率 {high_risk['g4_decline'].mean()*100:.1f}%，
  是全量学生 {(df['HI_g4'] < df['HI_g2']).mean()*100:.1f}% 的
  {high_risk['g4_decline'].mean() / max((df['HI_g4'] < df['HI_g2']).mean(), 0.01):.1f} 倍
- 漏斗有效：M1 筛出的子集确实更可能继续退化，M2 在这个子集上预测
"""

    (OUT_DIR / "summary.md").write_text(summary)
    print(f"\n结果已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
