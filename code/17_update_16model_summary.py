#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_update_16model_summary.py — 更新16模型汇总表

补算 TabPFN-2.5 和 TabICLv2 的聚类指标（JC/FMI/RI/DBI/DI），
加入 hi9_12models_all/_analysis/summary.csv，形成完整16模型对比。

输入：
  - experiments/review_20260910/hi9_12models_all/_analysis/summary.csv（14模型）
  - outputs/tabpfn25/test_predictions.csv + TabPFN25_status.json
  - outputs/tabiclv2/test_predictions.csv + TabICLv2_status.json
  - 标准57维特征 + 21维几何特征（用于计算DBI/DI）

输出：
  - experiments/review_20260910/hi9_12models_all/_analysis/summary_16models.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"
SUMMARY_CSV = ROOT / "experiments" / "review_20260910" / "hi9_12models_all" / "_analysis" / "summary.csv"
OUT_CSV = ROOT / "experiments" / "review_20260910" / "hi9_12models_all" / "_analysis" / "summary_16models.csv"

NEW_MODELS = [
    {
        "name": "TabPFN-2.5",
        "pred_csv": ROOT / "experiments" / "review_20260910" / "hi9_12models_bonus" / "outputs" / "tabpfn25" / "test_predictions.csv",
        "status_json": ROOT / "experiments" / "review_20260910" / "hi9_12models_bonus" / "outputs" / "tabpfn25" / "TabPFN25_status.json",
    },
    {
        "name": "TabICLv2",
        "pred_csv": ROOT / "experiments" / "review_20260910" / "hi9_12models_bonus" / "outputs" / "tabiclv2" / "test_predictions.csv",
        "status_json": ROOT / "experiments" / "review_20260910" / "hi9_12models_bonus" / "outputs" / "tabiclv2" / "TabICLv2_status.json",
    },
]


def main():
    sys.path.insert(0, str(HI9BENCH_DIR))
    from hi9bench.data import load_bundle_legacy_unverified
    from hi9bench.metrics import evaluate_all

    print("[1/4] 加载标准57维特征 + 21维几何特征 ...")
    bundle = load_bundle_legacy_unverified(RAW_CSV, LABELS_CSV, split_seed=91)
    y_test = bundle.y_test
    X_geom_test = bundle.X_geom_test
    ids_test = bundle.ids_test
    print(f"  test 集: {len(y_test)} 样本, 几何特征 {X_geom_test.shape}")

    # 建立 student_id -> test 索引的映射
    id_to_idx = {str(sid): i for i, sid in enumerate(ids_test)}

    print("\n[2/4] 读取现有14模型汇总表 ...")
    summary = pd.read_csv(SUMMARY_CSV)
    print(f"  现有模型: {len(summary)} 个")

    print("\n[3/4] 补算 TabPFN-2.5 和 TabICLv2 的聚类指标 ...")
    new_rows = []

    for model_info in NEW_MODELS:
        name = model_info["name"]
        print(f"\n  === {name} ===")

        # 读取 status.json 获取 accuracy/f1
        with open(model_info["status_json"]) as f:
            status = json.load(f)
        acc_mean = status["accuracy_mean"]
        acc_sd = status["accuracy_sd"]
        f1_mean = status["macro_f1_mean"]
        f1_sd = status["macro_f1_sd"]
        seeds = ",".join(str(s["seed"]) for s in status["per_seed"])

        # 读取 test_predictions.csv，按 seed 计算聚类指标
        pred_df = pd.read_csv(model_info["pred_csv"], dtype={"student_id": str})

        jc_list, fmi_list, ri_list, dbi_list, di_list = [], [], [], [], []

        for seed in [42, 43, 44]:
            seed_pred = pred_df[pred_df["seed"] == seed].copy()
            # 映射到标准 test 集顺序
            seed_pred["test_idx"] = seed_pred["student_id"].map(id_to_idx)
            seed_pred = seed_pred.dropna(subset=["test_idx"])
            seed_pred = seed_pred.sort_values("test_idx")
            pred_labels = seed_pred["pred_class"].to_numpy(dtype=np.int64)
            test_indices = seed_pred["test_idx"].to_numpy(dtype=np.int64)

            y_true = y_test[test_indices]
            X_geom = X_geom_test[test_indices]

            metrics = evaluate_all(y_true, pred_labels, X_geometry=X_geom, compute_geometry=True)
            jc_list.append(metrics["JC"])
            fmi_list.append(metrics["FMI"])
            ri_list.append(metrics["RI"])
            dbi_list.append(metrics["DBI"])
            di_list.append(metrics["DI"])
            print(f"    seed={seed}: JC={metrics['JC']:.4f} FMI={metrics['FMI']:.4f} RI={metrics['RI']:.4f} "
                  f"DBI={metrics['DBI']:.4f} DI={metrics['DI']:.4f} acc={metrics['accuracy']:.4f}")

        row = {
            "model": name,
            "status": "complete",
            "train": 25241,
            "val": 5409,
            "test": 5409,
            "seeds": seeds,
            "JC": np.mean(jc_list),
            "JC_sd": np.std(jc_list, ddof=0),
            "FMI": np.mean(fmi_list),
            "FMI_sd": np.std(fmi_list, ddof=0),
            "RI": np.mean(ri_list),
            "RI_sd": np.std(ri_list, ddof=0),
            "DBI": np.mean(dbi_list),
            "DBI_sd": np.std(dbi_list, ddof=0),
            "DI": np.mean(di_list),
            "DI_sd": np.std(di_list, ddof=0),
            "accuracy": acc_mean,
            "accuracy_sd": acc_sd,
            "macro_f1": f1_mean,
            "macro_f1_sd": f1_sd,
        }
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)

    print("\n[4/4] 合并并保存16模型汇总表 ...")
    summary_16 = pd.concat([summary, new_df], ignore_index=True)
    summary_16 = summary_16.sort_values("accuracy", ascending=False).reset_index(drop=True)
    summary_16.to_csv(OUT_CSV, index=False)

    print(f"\n完成 ✅ 16模型汇总表已保存:")
    print(f"  {OUT_CSV}")
    print(f"\n排名（按 accuracy）：")
    for i, row in summary_16.iterrows():
        print(f"  {i+1:2d}. {row['model']:15s} acc={row['accuracy']:.4f}±{row['accuracy_sd']:.4f}  "
              f"f1={row['macro_f1']:.4f}  JC={row['JC']:.4f}  DBI={row['DBI']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
