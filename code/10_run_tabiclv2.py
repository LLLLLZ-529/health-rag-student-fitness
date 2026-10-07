#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10_run_tabiclv2.py — TabICLv2 在 HI9 9 类分类上的实验

TabICLv2（INRIA, 2026.02）是最新的表格基础模型，
在 TabArena/TALENT 上超过调过参的 TabPFN-2.5。
开源：https://github.com/soda-inria/tabicl
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"
OUT_DIR = Path(__file__).parent / "outputs" / "tabiclv2"
SEEDS = [42, 43, 44]
N_ESTIMATORS = 4  # TabICLv2 ensemble 成员数

def load_standard_data():
    sys.path.insert(0, str(HI9BENCH_DIR))
    from hi9bench.data import load_bundle_legacy_unverified
    return load_bundle_legacy_unverified(str(RAW_CSV), str(LABELS_CSV), split_seed=91)

def run_tabicl(bundle, seed=42, max_ctx=10000):
    from tabicl import TabICLClassifier
    X_ctx_full = np.vstack([bundle.X_train, bundle.X_val])
    y_ctx_full = np.concatenate([bundle.y_train, bundle.y_val])
    X_test, y_test = bundle.X_test, bundle.y_test
    # 子采样 context 到 max_ctx（MPS 显存限制）
    if len(X_ctx_full) > max_ctx:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_ctx_full), max_ctx, replace=False)
        X_ctx, y_ctx = X_ctx_full[idx], y_ctx_full[idx]
    else:
        X_ctx, y_ctx = X_ctx_full, y_ctx_full
    print(f"      context: {X_ctx.shape} (subsampled from {X_ctx_full.shape}), test: {X_test.shape}")
    t0 = time.time()
    clf = TabICLClassifier(
        n_estimators=N_ESTIMATORS,
        device="mps",
        batch_size=32,
        random_state=seed,
        verbose=False,
    )
    clf.fit(X_ctx, y_ctx)
    fit_time = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    pred_time = time.time() - t0
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    return {"seed": seed, "accuracy": round(float(acc),4), "macro_f1": round(float(f1),4),
            "fit_time_s": round(fit_time,1), "pred_time_s": round(pred_time,1),
            "ctx_size": len(X_ctx)}, y_pred, y_proba

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] 加载标准 57 维特征 ...")
    bundle = load_standard_data()
    print(f"      train={bundle.X_train.shape}, val={bundle.X_val.shape}, test={bundle.X_test.shape}")
    print("[2/3] 跑 TabICLv2 (3 seeds) ...")
    all_results, all_preds = [], []
    for seed in SEEDS:
        print(f"    seed {seed}:")
        t0 = time.time()
        res, y_pred, y_proba = run_tabicl(bundle, seed=seed)
        dt = time.time() - t0
        print(f"      acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}  ({dt:.1f}s)")
        all_results.append(res)
        all_preds.append((seed, bundle.ids_test, y_pred, y_proba))
    accs = [r["accuracy"] for r in all_results]
    f1s = [r["macro_f1"] for r in all_results]
    summary = {"model": "TabICLv2", "n_seeds": len(SEEDS),
               "accuracy_mean": round(float(np.mean(accs)),4), "accuracy_sd": round(float(np.std(accs)),4),
               "macro_f1_mean": round(float(np.mean(f1s)),4), "macro_f1_sd": round(float(np.std(f1s)),4),
               "per_seed": all_results, "n_estimators": N_ESTIMATORS,
               "features": "57-D (28 z-score + 28 mask + gender)"}
    print(f"\n[3/3] 保存结果 ...")
    with open(OUT_DIR / "TabICLv2_status.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    pred_rows = []
    for seed, ids, y_pred, y_proba in all_preds:
        for i, (sid, pred) in enumerate(zip(ids, y_pred)):
            row = {"seed": seed, "student_id": str(sid), "pred_class": int(pred)}
            for c in range(y_proba.shape[1]):
                row[f"prob_{c}"] = round(float(y_proba[i,c]), 6)
            pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    print(f"\n完成 ✅")
    print(f"  accuracy = {summary['accuracy_mean']:.4f} ± {summary['accuracy_sd']:.4f}")
    print(f"  macro_f1 = {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_sd']:.4f}")
    print(f"  写入 {OUT_DIR}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
