#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30_m2_ablation.py — M2 消融实验：验证 AUC=0.94 是不是惯性

实验：
  A. 完整特征（含HI_slope + 所有斜率）→ 基准
  B. 删 HI_slope（只删这个）
  C. 删所有斜率特征（只用g2最新值）
  D. Naive baseline：全部预测"退化"（因为子集退化率66%）
  E. 删斜率+删交互项（最简模型）
"""
from __future__ import annotations
import os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "m2_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
           "sit_and_reach", "endurance_run_sec", "strength"]
HIGHER_BETTER = {"bmi": False, "vital_capacity": True, "sprint_50m": False,
                 "standing_long_jump": True, "sit_and_reach": True,
                 "endurance_run_sec": False, "strength": True}


def load_data():
    df = pd.read_csv(RAW_CSV)
    labels = pd.read_csv(LABELS_CSV)
    labels = labels[labels["model"] == labels["model"].iloc[0]]
    df = df.merge(labels[["student_id", "rule_class_name", "HI_slope"]], on="student_id")
    rule = df["rule_class_name"].fillna("")
    hr = df[rule.str.contains("低水平") | rule.str.contains("退化")].copy()
    hr = hr[hr["HI_g4"].notna()].copy()
    hr["g4_decline"] = (hr["HI_g4"] < hr["HI_g2"]).astype(int)
    return hr


def build_features(df, use_slope=True, use_hislope=True, use_interaction=True):
    feats = {}
    for m in METRICS:
        feats[f"{m}_g2"] = df[f"{m}_g2"]
    if use_slope:
        for m in METRICS:
            feats[f"{m}_slope"] = df[f"{m}_g2"] - df[f"{m}_g1"]
        for m in METRICS:
            feats[f"{m}_avg"] = (df[f"{m}_g1"] + df[f"{m}_g2"]) / 2
    feats["gender"] = (df["gender"] == "男").astype(int)
    if use_interaction and use_slope:
        for m in METRICS:
            feats[f"{m}_sx"] = feats[f"{m}_slope"] * feats[f"{m}_avg"]
        decline_count = np.zeros(len(df))
        for m in METRICS:
            if HIGHER_BETTER[m]:
                decline_count += (feats[f"{m}_slope"] < 0).astype(int)
            else:
                decline_count += (feats[f"{m}_slope"] > 0).astype(int)
        feats["n_decline"] = decline_count
    if use_hislope:
        feats["hi_slope"] = df["HI_slope"]
    return pd.DataFrame(feats).fillna(pd.DataFrame(feats).median())


def eval_model(X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, aps, p80 = [], [], []

    for tr, te in cv.split(X, y):
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
        aps.append(average_precision_score(y[te], proba))
        prec, rec, _ = precision_recall_curve(y[te], proba)
        idx = np.argmin(np.abs(rec - 0.8))
        p80.append(prec[idx])

    acc = (model.predict(X) == y).mean()
    return {
        "name": name,
        "n_features": X.shape[1],
        "auc": np.mean(aucs),
        "auc_std": np.std(aucs),
        "ap": np.mean(aps),
        "p_at_r80": np.mean(p80),
        "train_acc": acc,
    }


def main():
    print("=" * 60)
    print("M2 消融实验：AUC=0.94 是真本事还是惯性？")
    print("=" * 60)

    df = load_data()
    y = df["g4_decline"].values
    base_rate = y.mean()
    print(f"\n高风险学生: {len(df)}, 退化率: {base_rate*100:.1f}%")

    results = []

    # A. 完整特征
    X = build_features(df)
    r = eval_model(X, y, "A.完整(含HI_slope+斜率+交互)")
    results.append(r)
    print(f"\n{r['name']}: {r['n_features']}维, AUC={r['auc']:.3f}±{r['auc_std']:.3f}")

    # B. 删 HI_slope
    X = build_features(df, use_hislope=False)
    r = eval_model(X, y, "B.删HI_slope")
    results.append(r)
    print(f"{r['name']}: {r['n_features']}维, AUC={r['auc']:.3f}±{r['auc_std']:.3f}")

    # C. 删所有斜率
    X = build_features(df, use_slope=False)
    r = eval_model(X, y, "C.只用g2最新值(无斜率)")
    results.append(r)
    print(f"{r['name']}: {r['n_features']}维, AUC={r['auc']:.3f}±{r['auc_std']:.3f}")

    # D. 最简：只有g2 + gender
    X = build_features(df, use_slope=False, use_interaction=False)
    r = eval_model(X, y, "D.最简(7指标+gender)")
    results.append(r)
    print(f"{r['name']}: {r['n_features']}维, AUC={r['auc']:.3f}±{r['auc_std']:.3f}")

    # E. Naive baseline：全部预测"退化"
    naive_pred = np.ones(len(y))
    naive_acc = (naive_pred == y).mean()
    print(f"\nE.Naive baseline(全预测退化): acc={naive_acc:.3f} (退化率{base_rate*100:.1f}%)")
    results.append({
        "name": "E.Naive(全预测退化)",
        "n_features": 0,
        "auc": 0.5,
        "auc_std": 0,
        "ap": base_rate,
        "p_at_r80": base_rate,
        "train_acc": naive_acc,
    })

    # 保存
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / "ablation_results.csv", index=False)

    summary = f"""# M2 消融实验结果

## 数据
- 高风险学生: {len(df)}
- 退化率: {base_rate*100:.1f}%

## 消融对比
| 实验 | 特征数 | AUC | PR-AUC | P@R80% |
|------|--------|-----|--------|--------|
"""
    for r in results:
        summary += f"| {r['name']} | {r['n_features']} | {r['auc']:.3f} | {r['ap']:.3f} | {r['p_at_r80']:.3f} |\n"

    a_auc = results[0]["auc"]
    c_auc = results[2]["auc"]
    summary += f"""
## 结论
- 完整模型 AUC={a_auc:.3f}
- 删斜率后 AUC={c_auc:.3f}
- 差值 = {a_auc - c_auc:.3f}（这部分是"惯性"贡献的）
- 超过 naive baseline({base_rate:.3f}) 的部分才是真正的预测力
"""
    (OUT_DIR / "summary.md").write_text(summary)
    print(f"\n保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
