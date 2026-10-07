#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_m2_decline_multimodel.py — M2-3 退化预警二分类：多模型对比

用大一大二数据预测大四是否退化（二分类），在多个模型上验证，与 M1 的 16 模型基准衔接。

模型：
  传统：Logistic, RandomForest, XGBoost, LightGBM
  深度学习：MLP, ModernNCA(15-D输入), TabM(15-D输入), ResNet1D
  基础模型：TabPFN-2.5

输出：
  - outputs/m2_decline_multimodel/m2_decline_results.csv
  - outputs/m2_decline_multimodel/summary.md
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "m2_decline_multimodel"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
METRICS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]
GRADES = [1, 2]  # 用大一大二预测
N_CLASSES = 2
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data():
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("", "_raw"))

    # 筛选有 g1+g2+g4 的学生
    def has_grades(row):
        pat = str(int(row["real_obs_pattern"])) if pd.notna(row["real_obs_pattern"]) else ""
        return all(str(g) in pat for g in [1, 2, 4])
    df = df[df.apply(has_grades, axis=1)].copy()

    # 退化二分类标签：trend=0 (class 0,3,6) = 退化
    df["is_decline"] = df["rule_class"].isin([0, 3, 6]).astype(int)
    print(f"  有 g1+g2+g4 的学生: {len(df)}, 退化率: {df['is_decline'].mean():.4f}")
    return df


def build_features(df):
    """g1+g2 的 14 维 raw + gender = 15 维。"""
    cols = []
    for g in GRADES:
        for m in METRICS:
            cols.append(f"{m}_g{g}")
    X = df[cols].to_numpy(dtype=np.float32)
    gender = (df["gender"] == "男").to_numpy(dtype=np.float32).reshape(-1, 1)
    X = np.column_stack([X, gender])
    # NaN 填充
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


# ---------------------------------------------------------------------------
# 传统模型
# ---------------------------------------------------------------------------
def run_sklearn_model(model_cls, X, y, seed, **kwargs):
    train_idx, val_idx, test_idx = split_data(X, y, seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    model = model_cls(random_state=seed, **kwargs)
    t0 = time.time()
    model.fit(X_train, y[train_idx])
    fit_time = time.time() - t0

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    acc = accuracy_score(y[test_idx], y_pred)
    f1 = f1_score(y[test_idx], y_pred, average="macro")
    ap = average_precision_score(y[test_idx], y_proba)
    return acc, f1, ap, fit_time


# ---------------------------------------------------------------------------
# 深度学习模型
# ---------------------------------------------------------------------------
class MLPBin(nn.Module):
    def __init__(self, n_features=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class ModernNCABin(nn.Module):
    """ModernNCA 简化版：MLP backbone + L2 normalized embedding + 线性分类头"""
    def __init__(self, n_features=15, embed_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, N_CLASSES)

    def forward(self, x):
        emb = self.backbone(x)
        emb = F.normalize(emb, p=2, dim=1)
        return self.classifier(emb)


class ResNet1DBin(nn.Module):
    """1D-ResNet：把 15 维特征当作 1D 序列"""
    def __init__(self, n_features=15):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 15)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class EnsembleView(nn.Module):
    """TabM 的 EnsembleView：生成 k 个扰动视图"""
    def __init__(self, n_features, k=32, noise_std=0.1):
        super().__init__()
        self.k = k
        self.noise_std = noise_std
        self.weights = nn.Parameter(torch.randn(k, n_features) * 0.02)

    def forward(self, x):
        # x: (B, D) → (B, k, D)
        B = x.size(0)
        x_exp = x.unsqueeze(1).expand(B, self.k, -1)
        noise = torch.randn_like(x_exp) * self.noise_std
        mask = torch.sigmoid(self.weights.unsqueeze(0).expand(B, -1, -1))
        return x_exp * mask + noise * (1 - mask)


class LinearBatchEnsemble(nn.Module):
    def __init__(self, in_dim, out_dim, k=32):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.randn(k, in_dim, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(k, out_dim))

    def forward(self, x):
        # x: (B, k, in) → (B, k, out)
        return torch.einsum('bki,kio->bko', x, self.weight) + self.bias.unsqueeze(0)


class SharedBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        B, k, D = x.shape
        x = x.reshape(B * k, D)
        x = self.bn(x)
        return x.reshape(B, k, D)


class TabMBin(nn.Module):
    """TabM 简化版：EnsembleView(k=32) → 3层 LinearBatchEnsemble → 投票"""
    def __init__(self, n_features=15, k=32, hidden=128):
        super().__init__()
        self.k = k
        self.view = EnsembleView(n_features, k=k)
        self.layers = nn.Sequential(
            LinearBatchEnsemble(n_features, hidden, k),
            SharedBatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            LinearBatchEnsemble(hidden, hidden, k),
            SharedBatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            LinearBatchEnsemble(hidden, hidden, k),
            SharedBatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.head = LinearBatchEnsemble(hidden, N_CLASSES, k)

    def forward(self, x):
        x = self.view(x)  # (B, k, D)
        x = self.layers(x)
        logits = self.head(x)  # (B, k, 2)
        return logits.mean(dim=1)  # 投票平均


def train_torch_model(model_cls, X, y, seed, n_features=15, epochs=EPOCHS):
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = split_data(X, y, seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = model_cls(n_features=n_features).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # class weights（退化类是少数类）
    n_pos = y[train_idx].sum()
    n_neg = len(y[train_idx]) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        rng = np.random.RandomState(seed + epoch)
        perm = rng.permutation(len(train_idx))
        for i in range(0, len(perm), BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            xb = X_train_t[batch_idx].to(DEVICE)
            yb = y_train_t[batch_idx].to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weight)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(DEVICE))
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y[val_idx], val_pred)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(DEVICE))
        test_proba = F.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_pred = test_logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y[test_idx], test_pred)
    f1 = f1_score(y[test_idx], test_pred, average="macro")
    ap = average_precision_score(y[test_idx], test_proba)
    return acc, f1, ap


# ---------------------------------------------------------------------------
# TabPFN-2.5
# ---------------------------------------------------------------------------
def run_tabpfn25(X, y, seed):
    from tabpfn import TabPFNClassifier
    train_idx, val_idx, test_idx = split_data(X, y, seed)
    X_ctx = np.vstack([X[train_idx], X[val_idx]])
    y_ctx = np.concatenate([y[train_idx], y[val_idx]])

    t0 = time.time()
    clf = TabPFNClassifier(device=DEVICE, random_state=seed)
    clf.fit(X_ctx, y_ctx)
    y_proba = clf.predict_proba(X[test_idx])
    fit_time = time.time() - t0

    y_pred = y_proba.argmax(axis=1)
    acc = accuracy_score(y[test_idx], y_pred)
    f1 = f1_score(y[test_idx], y_pred, average="macro")
    ap = average_precision_score(y[test_idx], y_proba[:, 1])
    return acc, f1, ap, fit_time


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备 = {DEVICE}")
    print("[1/3] 加载数据 ...")
    df = load_data()
    X = build_features(df)
    y = df["is_decline"].to_numpy(dtype=np.int64)
    print(f"  特征维度: {X.shape[1]}, 样本数: {len(X)}")

    print(f"\n[2/3] 跑多模型退化二分类 × {len(SEEDS)} seeds ...")
    all_results = []

    # --- 传统模型 ---
    sklearn_models = [
        ("Logistic", LogisticRegression, {"max_iter": 1000, "class_weight": "balanced"}),
        ("RandomForest", RandomForestClassifier, {"n_estimators": 200, "class_weight": "balanced", "n_jobs": -1}),
        ("XGBoost", None, {}),  # 特殊处理
        ("LightGBM", None, {}),  # 特殊处理
    ]

    for name, model_cls, kwargs in sklearn_models:
        if name == "XGBoost":
            from xgboost import XGBClassifier
            model_cls = XGBClassifier
            kwargs = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
                      "use_label_encoder": False, "eval_metric": "logloss",
                      "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1)}
        elif name == "LightGBM":
            from lightgbm import LGBMClassifier
            model_cls = LGBMClassifier
            kwargs = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
                      "class_weight": "balanced", "verbose": -1}

        for seed in SEEDS:
            t0 = time.time()
            acc, f1, ap, fit_time = run_sklearn_model(model_cls, X, y, seed, **kwargs)
            elapsed = time.time() - t0
            print(f"  {name:15s} seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
            all_results.append({"model": name, "seed": seed, "accuracy": round(acc, 4),
                                "macro_f1": round(f1, 4), "avg_precision": round(ap, 4),
                                "seconds": round(elapsed, 1)})

    # --- 深度学习模型 ---
    torch_models = [
        ("MLP", MLPBin),
        ("ModernNCA", ModernNCABin),
        ("ResNet1D", ResNet1DBin),
        ("TabM", TabMBin),
    ]

    for name, model_cls in torch_models:
        for seed in SEEDS:
            t0 = time.time()
            acc, f1, ap = train_torch_model(model_cls, X, y, seed)
            elapsed = time.time() - t0
            print(f"  {name:15s} seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
            all_results.append({"model": name, "seed": seed, "accuracy": round(acc, 4),
                                "macro_f1": round(f1, 4), "avg_precision": round(ap, 4),
                                "seconds": round(elapsed, 1)})

    # --- TabPFN-2.5 ---
    for seed in SEEDS:
        try:
            t0 = time.time()
            acc, f1, ap, fit_time = run_tabpfn25(X, y, seed)
            elapsed = time.time() - t0
            print(f"  {'TabPFN-2.5':15s} seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
            all_results.append({"model": "TabPFN-2.5", "seed": seed, "accuracy": round(acc, 4),
                                "macro_f1": round(f1, 4), "avg_precision": round(ap, 4),
                                "seconds": round(elapsed, 1)})
        except Exception as e:
            print(f"  TabPFN-2.5 seed={seed}: 失败 - {e}")

    # --- 保存 ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "m2_decline_results.csv", index=False)

    summary = results_df.groupby("model").agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
        ap_mean=("avg_precision", "mean"),
        ap_std=("avg_precision", "std"),
    ).round(4).sort_values("acc_mean", ascending=False)

    print(f"\n[3/3] 汇总（按 accuracy 排序）：")
    print(summary.to_string())

    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("# M2-3 退化预警二分类：多模型对比\n\n")
        f.write(f"任务：用 g1+g2 体测数据（{X.shape[1]}-D）预测大四是否退化（trend=0）\n")
        f.write(f"样本：{len(X)} 人（有 g1+g2+g4 完整数据），退化率 {y.mean():.4f}\n\n")
        f.write("## 结果\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n## 与 M1 衔接\n\n")
        f.write("M1 用全部 4 年数据（57-D）做 9 类静态分类，TabM 准确率 85.9%。\n")
        f.write("M2-3 用大一大二数据（15-D）做退化二分类，验证多个模型在早期预警任务上的表现。\n")
        f.write("最佳模型可作为 M3 RAG 系统的预警引擎。\n")

    print(f"\n写入 {OUT_DIR / 'm2_decline_results.csv'}")
    print(f"写入 {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    sys.exit(main())
