#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_m2_warning_final_backfill.py — 最终补跑 M2 剩余 3 个模型

修复：TabM（正确参数d_in/d_out）、GRANDE（fit传Xva/yva）、TabPFN-2.5（ignore_pretraining_limits）
"""
from __future__ import annotations
import os, sys, time, warnings, importlib.util
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "m2_warning"
PRED_DIR = OUT_DIR / "predictions"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"
TABM_SCRIPT = Path(__file__).parent / "05_run_tabm.py"

sys.path.insert(0, str(HI9BENCH_DIR))

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
METRICS = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
           "sit_and_reach", "endurance_run_sec", "strength"]
GRADES = [1, 2]
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3


def load_data():
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("", "_raw"))
    def has_grades(row):
        pat = str(int(row["real_obs_pattern"])) if pd.notna(row["real_obs_pattern"]) else ""
        return all(str(g) in pat for g in [1, 2, 4])
    df = df[df.apply(has_grades, axis=1)].copy()
    df["is_decline"] = df["rule_class"].isin([0, 3, 6]).astype(int)
    print(f"  样本: {len(df)}, 退化率: {df['is_decline'].mean():.4f}")
    return df


def build_features(df):
    cols = []
    for g in GRADES:
        for m in METRICS:
            cols.append(f"{m}_g{g}")
    X = df[cols].to_numpy(dtype=np.float32)
    gender = (df["gender"] == "男").to_numpy(dtype=np.float32).reshape(-1, 1)
    X = np.column_stack([X, gender])
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X


def split_data(X, y, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.7 * len(idx))
    n_val = int(0.15 * len(idx))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
        "ap": average_precision_score(y_true, y_prob),
    }


def train_torch(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    counts = np.bincount(y_train, minlength=2)
    weights = torch.tensor([len(y_train) / (2 * c) for c in counts], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    ytr = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    yva = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
    best_val_loss, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            loss = criterion(model(Xtr[idx]), ytr[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva), yva).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def get_tabm_class():
    spec = importlib.util.spec_from_file_location("tabm_script", TABM_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TabM


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/2] 加载数据 ...")
    df = load_data()
    X = build_features(df)
    y = df["is_decline"].to_numpy()
    in_dim = X.shape[1]
    TabM = get_tabm_class()

    models_to_run = ["TabM", "GRANDE", "TabPFN_2.5"]
    all_results = []

    print(f"\n[2/2] 补跑 {len(models_to_run)} 个模型 × {len(SEEDS)} seeds ...")

    for model_name in models_to_run:
        print(f"\n  === {model_name} ===")
        for seed in SEEDS:
            t0 = time.time()
            train_idx, val_idx, test_idx = split_data(X, y, seed)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            try:
                if model_name == "TabM":
                    torch.manual_seed(seed)
                    model = TabM(d_in=in_dim, d_out=2, k=16, n_blocks=2, d_block=128, dropout=0.2)
                    model = train_torch(model, X_train, y_train, X_val, y_val)
                    model.eval()
                    with torch.no_grad():
                        logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
                        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

                elif model_name == "GRANDE":
                    from hi9bench.models.grande import GRANDEModel
                    config = {"n_classes": 2, "n_estimators": 200, "max_depth": 6}
                    model = GRANDEModel(config, seed=seed)
                    model.fit(X_train, y_train, X_val, y_val)
                    prob = model.predict_proba(X_test)[:, 1]

                elif model_name == "TabPFN_2.5":
                    from tabpfn import TabPFNClassifier
                    # 子采样到5000以内避免CPU限制
                    rng = np.random.RandomState(seed)
                    if len(X_train) > 4000:
                        idx = rng.choice(len(X_train), 4000, replace=False)
                        Xtr, ytr = X_train[idx], y_train[idx]
                    else:
                        Xtr, ytr = X_train, y_train
                    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
                    clf.fit(Xtr, ytr)
                    prob = clf.predict_proba(X_test)[:, 1]

                pred_df = pd.DataFrame({
                    "student_id": df.iloc[test_idx]["student_id"].values,
                    "y_true": y_test, "y_prob": prob,
                })
                pred_df.to_csv(PRED_DIR / f"{model_name}_seed{seed}.csv", index=False)

                metrics = compute_metrics(y_test, prob)
                metrics["model"] = model_name
                metrics["seed"] = seed
                metrics["seconds"] = round(time.time() - t0, 1)
                all_results.append(metrics)

                print(f"    seed={seed}: acc={metrics['accuracy']:.4f}  "
                      f"recall={metrics['recall']:.4f}  precision={metrics['precision']:.4f}  "
                      f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  "
                      f"({metrics['seconds']}s)")

            except Exception as e:
                print(f"    seed={seed}: 失败 - {e}")
                import traceback; traceback.print_exc()

    if all_results:
        new_df = pd.DataFrame(all_results)
        new_df.to_csv(OUT_DIR / "m2_warning_final_backfill.csv", index=False)

        # 合并所有结果
        old_path = OUT_DIR / "m2_warning_results.csv"
        backfill_path = OUT_DIR / "m2_warning_backfill_results.csv"
        dfs = [pd.read_csv(old_path)]
        if backfill_path.exists():
            dfs.append(pd.read_csv(backfill_path))
        dfs.append(new_df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(OUT_DIR / "m2_warning_results_FINAL.csv", index=False)

        summary = combined.groupby("model").agg(
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"), precision_std=("precision", "std"),
            recall_mean=("recall", "mean"), recall_std=("recall", "std"),
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
            specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
            auc_mean=("auc", "mean"), auc_std=("auc", "std"),
            ap_mean=("ap", "mean"), ap_std=("ap", "std"),
        ).reset_index().sort_values("recall_mean", ascending=False)
        summary.to_csv(OUT_DIR / "m2_warning_summary_FINAL.csv", index=False)

        print(f"\n完成 ✅ 完整 {len(summary)} 模型按 Recall 排名：")
        for _, row in summary.iterrows():
            print(f"  {row['model']:20s}  recall={row['recall_mean']:.4f}±{row['recall_std']:.4f}  "
                  f"precision={row['precision_mean']:.4f}  F1={row['f1_mean']:.4f}  "
                  f"AUC={row['auc_mean']:.4f}  AP={row['ap_mean']:.4f}")
        print(f"\n  写入 {OUT_DIR / 'm2_warning_results_FINAL.csv'}")
        print(f"  写入 {OUT_DIR / 'm2_warning_summary_FINAL.csv'}")


if __name__ == "__main__":
    sys.exit(main())
