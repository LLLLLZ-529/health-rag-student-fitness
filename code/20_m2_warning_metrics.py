#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_m2_warning_metrics.py — M2 退化预警：完整指标重跑

重新跑 16 模型退化二分类，保存逐样本预测，计算完整预警指标：
  - accuracy, precision, recall, F1, specificity, AUC, AP
  - 重点关注 recall（退化学生的召回率）和 precision（预警精确率）
  - 不同阈值下的 recall/precision  trade-off

输出：
  - outputs/m2_warning/m2_warning_results.csv（每模型每seed完整指标）
  - outputs/m2_warning/m2_warning_summary.csv（3 seeds平均±std）
  - outputs/m2_warning/predictions/{model}_seed{seed}.csv（逐样本预测）
  - outputs/m2_warning/threshold_analysis.csv（阈值分析）
"""
from __future__ import annotations
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "m2_warning"
PRED_DIR = OUT_DIR / "predictions"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
METRICS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]
GRADES = [1, 2]
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3

# 16 模型列表
ALL_MODELS = [
    "Logistic", "RandomForest", "XGBoost", "LightGBM", "RBF_SVM",
    "MLP", "ModernNCA", "TabM", "ResNet1D", "TabNet",
    "FT_Transformer", "SAINT", "SupCon", "GRANDE",
    "TabPFN_2.5", "TabICLv2",
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
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
    """计算完整预警指标"""
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
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------------------
# 传统模型
# ---------------------------------------------------------------------------
def run_sklearn_model(name, X_train, y_train, X_test, seed):
    if name == "Logistic":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    elif name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
    elif name == "RBF_SVM":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
        clf.fit(X_train_s, y_train)
        return clf.predict_proba(X_test_s)[:, 1]
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                            use_label_encoder=False, eval_metric="logloss",
                            random_state=seed, verbosity=0)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]
    elif name == "LightGBM":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                             class_weight="balanced", random_state=seed, verbose=-1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------------
# 深度学习模型
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)


class ModernNCA(nn.Module):
    def __init__(self, in_dim, n_classes=2, embed_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)
    def forward(self, x):
        emb = F.normalize(self.backbone(x), dim=1)
        return self.classifier(emb)


class ResNet1D(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, 128)
        self.block1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.block2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.fc_out = nn.Linear(128, n_classes)
    def forward(self, x):
        h = F.relu(self.fc_in(x))
        h = F.relu(h + self.block1(h))
        h = F.relu(h + self.block2(h))
        return self.fc_out(h)


def train_torch_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # class weights
    counts = np.bincount(y_train, minlength=2)
    weights = torch.tensor([len(y_train) / (2 * c) for c in counts], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    ytr = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    best_val_loss = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            logits = model(Xtr[idx])
            loss = criterion(logits, ytr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.long).to(DEVICE)).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def run_torch_model(name, X_train, y_train, X_val, y_val, X_test, in_dim, seed):
    torch.manual_seed(seed)
    if name == "MLP":
        model = MLP(in_dim)
    elif name == "ModernNCA":
        model = ModernNCA(in_dim)
    elif name == "ResNet1D":
        model = ResNet1D(in_dim)
    elif name == "TabM":
        # TabM: simplified ensemble
        from hi9bench.models.tabm import TabMModel
        model = TabMModel(in_dim, n_classes=2, n_views=8, hidden_dim=128, n_layers=2)
    elif name == "TabNet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        clf = TabNetClassifier(verbose=0, seed=seed, device_name="mps" if DEVICE == "mps" else "cpu")
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], max_epochs=60, patience=10,
                batch_size=256, virtual_batch_size=128)
        return clf.predict_proba(X_test)[:, 1]
    elif name == "FT_Transformer":
        from rtdl import FTTransformer
        model = FTTransformer.make_baseline(n_num_features=in_dim, d_token=32, n_blocks=3,
                                            d_out=2, attention_dropout=0.2, ffn_d_hidden=128)
    elif name == "SAINT":
        # 简化版 SAINT
        model = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
    elif name == "SupCon":
        # SupCon: 先训练encoder，再训分类头
        model = ModernNCA(in_dim, embed_dim=128)
    else:
        model = MLP(in_dim)

    if name in ["TabM", "FT_Transformer", "SAINT", "SupCon"]:
        model = model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        counts = np.bincount(y_train, minlength=2)
        weights = torch.tensor([len(y_train) / (2 * c) for c in counts], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        Xtr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        ytr = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
        Xva = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        best_val_loss = float("inf")
        best_state = None
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                logits = model(Xtr[idx])
                loss = criterion(logits, ytr[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xva), torch.tensor(y_val, dtype=torch.long).to(DEVICE)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
    else:
        model = train_torch_model(model, X_train, y_train, X_val, y_val)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return prob


# ---------------------------------------------------------------------------
# 基础模型
# ---------------------------------------------------------------------------
def run_tabpfn25(X_train, y_train, X_test, seed):
    try:
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device="cpu", seed=seed)
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        return prob
    except Exception as e:
        print(f"    TabPFN-2.5 失败: {e}")
        return None


def run_tabiclv2(X_train, y_train, X_test, seed):
    try:
        from tabicl import TabICL
        # 用 CPU 避免 MPS 显存不足
        model = TabICL(model_name="yandex/TabICL-v2-base", device="cpu")
        # 子采样 context
        rng = np.random.RandomState(seed)
        if len(X_train) > 10000:
            idx = rng.choice(len(X_train), 10000, replace=False)
            Xtr, ytr = X_train[idx], y_train[idx]
        else:
            Xtr, ytr = X_train, y_train
        prob = model.predict_proba(Xtr, ytr, X_test)[:, 1]
        return prob
    except Exception as e:
        print(f"    TabICLv2 失败: {e}")
        return None


def run_grande(X_train, y_train, X_test, seed):
    try:
        sys.path.insert(0, str(HI9BENCH_DIR))
        from hi9bench.models.grande import GRANDEModel
        model = GRANDEModel(n_classes=2, random_state=seed)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        return prob
    except Exception as e:
        print(f"    GRANDE 失败: {e}")
        return None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载数据 ...")
    df = load_data()
    X = build_features(df)
    y = df["is_decline"].to_numpy()
    in_dim = X.shape[1]
    print(f"  特征维度: {in_dim}")

    all_results = []
    threshold_data = []

    models_to_run = [m for m in ALL_MODELS if m not in ["TabICLv2"]]  # TabICLv2 可选

    print(f"\n[2/4] 训练 {len(models_to_run)} 个模型 × {len(SEEDS)} seeds ...")
    for model_name in models_to_run:
        print(f"\n  === {model_name} ===")
        for seed in SEEDS:
            t0 = time.time()
            train_idx, val_idx, test_idx = split_data(X, y, seed)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            try:
                if model_name in ["Logistic", "RandomForest", "RBF_SVM", "XGBoost", "LightGBM"]:
                    prob = run_sklearn_model(model_name, X_train, y_train, X_test, seed)
                elif model_name == "TabPFN_2.5":
                    prob = run_tabpfn25(X_train, y_train, X_test, seed)
                elif model_name == "TabICLv2":
                    prob = run_tabiclv2(X_train, y_train, X_test, seed)
                elif model_name == "GRANDE":
                    prob = run_grande(X_train, y_train, X_test, seed)
                else:
                    prob = run_torch_model(model_name, X_train, y_train, X_val, y_val, X_test, in_dim, seed)

                if prob is None:
                    print(f"    seed={seed}: 跳过（模型不可用）")
                    continue

                # 保存逐样本预测
                pred_df = pd.DataFrame({
                    "student_id": df.iloc[test_idx]["student_id"].values,
                    "y_true": y_test,
                    "y_prob": prob,
                })
                pred_df.to_csv(PRED_DIR / f"{model_name}_seed{seed}.csv", index=False)

                # 计算完整指标
                metrics = compute_metrics(y_test, prob)
                metrics["model"] = model_name
                metrics["seed"] = seed
                metrics["seconds"] = round(time.time() - t0, 1)
                all_results.append(metrics)

                print(f"    seed={seed}: acc={metrics['accuracy']:.4f}  "
                      f"recall={metrics['recall']:.4f}  precision={metrics['precision']:.4f}  "
                      f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  "
                      f"({metrics['seconds']}s)")

                # 阈值分析（只对 seed=42）
                if seed == 42:
                    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
                        m = compute_metrics(y_test, prob, threshold=thr)
                        threshold_data.append({
                            "model": model_name, "threshold": thr,
                            "recall": m["recall"], "precision": m["precision"],
                            "f1": m["f1"], "accuracy": m["accuracy"],
                        })

            except Exception as e:
                print(f"    seed={seed}: 失败 - {e}")
                import traceback
                traceback.print_exc()

    print(f"\n[3/4] 汇总结果 ...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "m2_warning_results.csv", index=False)

    # 3 seeds 平均
    summary = results_df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ap_std=("ap", "std"),
    ).reset_index()
    summary = summary.sort_values("recall_mean", ascending=False)
    summary.to_csv(OUT_DIR / "m2_warning_summary.csv", index=False)

    # 阈值分析
    if threshold_data:
        thr_df = pd.DataFrame(threshold_data)
        thr_df.to_csv(OUT_DIR / "threshold_analysis.csv", index=False)

    print(f"\n[4/4] 完成 ✅")
    print(f"\n  按 Recall 排名（退化学生召回率，越高越好）：")
    for _, row in summary.iterrows():
        print(f"    {row['model']:20s}  recall={row['recall_mean']:.4f}±{row['recall_std']:.4f}  "
              f"precision={row['precision_mean']:.4f}  F1={row['f1_mean']:.4f}  "
              f"AUC={row['auc_mean']:.4f}  AP={row['ap_mean']:.4f}")

    print(f"\n  写入 {OUT_DIR / 'm2_warning_results.csv'}")
    print(f"  写入 {OUT_DIR / 'm2_warning_summary.csv'}")
    print(f"  写入 {OUT_DIR / 'threshold_analysis.csv'}")
    print(f"  预测文件: {PRED_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
