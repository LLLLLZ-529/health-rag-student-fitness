#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_temporal_model_comparison.py — 时序模型对比：MLP(摊平) vs LSTM vs GRU vs 1D-CNN

科学问题：在 4 个时间点的体测数据上，显式时序建模比简单摊平更好吗？

实验设计：
  - 筛选有 g1+g2+g3+g4 完整数据的学生
  - 特征：(4, 7) 序列（4 年级 × 7 项体测），标准化后输入
  - 标签：9 类 rule_class（基于 g4 total_score）
  - 模型：MLP(摊平28维+gender) / LSTM / GRU / 1D-CNN
  - 3 seeds，对比 accuracy / macro_f1

输出：
  - outputs/temporal_comparison/temporal_results.csv
  - outputs/temporal_comparison/summary.md
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "temporal_comparison"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
METRICS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]
GRADES = [1, 2, 3, 4]
N_CLASSES = 9
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(RAW_CSV)
    labels = pd.read_csv(LABELS_CSV)
    df = df.merge(labels[["student_id", "rule_class", "rule_eligible"]], on="student_id", how="inner")
    df = df[df["rule_eligible"] == True].copy()

    # 筛选有全部 4 年级数据的学生
    def has_all_grades(row):
        pat = str(int(row["real_obs_pattern"])) if pd.notna(row["real_obs_pattern"]) else ""
        return all(str(g) in pat for g in GRADES)
    df = df[df.apply(has_all_grades, axis=1)].copy()
    print(f"  有 g1-g4 完整数据的学生: {len(df)}")
    return df


def build_sequence_data(df, seed=42):
    """构造 (N, 4, 7) 序列特征和 9 类标签，按学生随机划分 train/val/test。"""
    # 构造序列 (N, 4, 7)
    seq = np.zeros((len(df), len(GRADES), len(METRICS)), dtype=np.float32)
    for i, g in enumerate(GRADES):
        for j, m in enumerate(METRICS):
            seq[:, i, j] = df[f"{m}_g{g}"].to_numpy(dtype=np.float32)

    # NaN 填充（按特征维度，用训练集中位数）—— 必须在划分后做
    # 先记录 NaN mask，划分后用训练集中位数填充
    nan_mask = np.isnan(seq)

    # gender
    gender = (df["gender"] == "男").to_numpy(dtype=np.float32).reshape(-1, 1)

    # 标签
    y = df["rule_class"].to_numpy(dtype=np.int64)

    # 按学生随机划分
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_train = int(0.7 * len(idx))
    n_val = int(0.15 * len(idx))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # 用训练集中位数填充 NaN
    train_seq = seq[train_idx]
    for j in range(len(METRICS)):
        col_vals = train_seq[:, :, j].reshape(-1)
        col_vals = col_vals[~np.isnan(col_vals)]
        median_val = np.median(col_vals) if len(col_vals) > 0 else 0.0
        seq[:, :, j] = np.nan_to_num(seq[:, :, j], nan=median_val)

    # 用训练集统计量做标准化（按特征维度）
    train_seq = seq[train_idx]
    mean = train_seq.reshape(-1, len(METRICS)).mean(axis=0)
    std = train_seq.reshape(-1, len(METRICS)).std(axis=0) + 1e-8
    seq_norm = (seq - mean) / std

    return {
        "X_seq": seq_norm, "X_gender": gender, "y": y,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------
class MLPFlat(nn.Module):
    """摊平 MLP：28 维 + gender → 256→256→128→9"""
    def __init__(self, n_features=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x_seq, x_gender):
        x_flat = x_seq.reshape(x_seq.size(0), -1)
        x = torch.cat([x_flat, x_gender], dim=1)
        return self.net(x)


class LSTMClassifier(nn.Module):
    """LSTM：input(4,7) → hidden=128, 2层 → 最后时间步 + gender → 9"""
    def __init__(self, input_size=7, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x_seq, x_gender):
        out, (h_n, _) = self.lstm(x_seq)
        last = out[:, -1, :]
        x = torch.cat([last, x_gender], dim=1)
        return self.head(x)


class GRUClassifier(nn.Module):
    """GRU：input(4,7) → hidden=128, 2层 → 最后时间步 + gender → 9"""
    def __init__(self, input_size=7, hidden_size=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x_seq, x_gender):
        out, h_n = self.gru(x_seq)
        last = out[:, -1, :]
        x = torch.cat([last, x_gender], dim=1)
        return self.head(x)


class CNN1DClassifier(nn.Module):
    """1D-CNN：input(7,4) [channels=7, seq_len=4] → Conv1d×2 → 全局池化 + gender → 9"""
    def __init__(self, n_channels=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=2, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 1, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x_seq, x_gender):
        # x_seq: (B, 4, 7) → transpose to (B, 7, 4) for Conv1d
        x = x_seq.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        x = torch.cat([x, x_gender], dim=1)
        return self.head(x)


MODELS = {
    "MLP_flat": MLPFlat,
    "LSTM": LSTMClassifier,
    "GRU": GRUClassifier,
    "CNN1D": CNN1DClassifier,
}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_model(model_cls, data, seed=42):
    torch.manual_seed(seed)
    model = model_cls().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    X_seq = torch.tensor(data["X_seq"], dtype=torch.float32)
    X_gender = torch.tensor(data["X_gender"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    # class weights（sqrt-inverse，处理极度不平衡）
    y_train_np = y[train_idx].numpy()
    counts = np.bincount(y_train_np, minlength=N_CLASSES).astype(np.float32)
    weights = 1.0 / np.sqrt(counts + 1)
    weights = weights / weights.sum() * N_CLASSES
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    best_val_acc = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.RandomState(seed + epoch)
        perm = rng.permutation(len(train_idx))
        for i in range(0, len(perm), BATCH_SIZE):
            batch_idx = train_idx[perm[i:i + BATCH_SIZE]]
            xb = X_seq[batch_idx].to(DEVICE)
            gb = X_gender[batch_idx].to(DEVICE)
            yb = y[batch_idx].to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, gb)
            loss = nn.functional.cross_entropy(logits, yb, weight=class_weights)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(X_seq[val_idx].to(DEVICE), X_gender[val_idx].to(DEVICE))
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y[val_idx].numpy(), val_pred)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 加载最佳模型，测试
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_seq[test_idx].to(DEVICE), X_gender[test_idx].to(DEVICE))
        test_pred = test_logits.argmax(dim=1).cpu().numpy()
    test_acc = accuracy_score(y[test_idx].numpy(), test_pred)
    test_f1 = f1_score(y[test_idx].numpy(), test_pred, average="macro")
    return test_acc, test_f1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] 加载数据 ...")
    df = load_data()

    print(f"[2/3] 训练 4 个时序模型 × {len(SEEDS)} seeds ...")
    all_results = []
    for model_name, model_cls in MODELS.items():
        for seed in SEEDS:
            data = build_sequence_data(df, seed=seed)
            t0 = time.time()
            acc, f1 = train_model(model_cls, data, seed=seed)
            elapsed = time.time() - t0
            print(f"  {model_name:10s} seed={seed}: acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)")
            all_results.append({
                "model": model_name, "seed": seed,
                "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                "seconds": round(elapsed, 1),
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "temporal_results.csv", index=False)

    # 汇总
    summary = results_df.groupby("model").agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
    ).round(4).sort_values("acc_mean", ascending=False)

    print(f"\n[3/3] 汇总：")
    print(summary.to_string())

    # 写 summary.md
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("# 时序模型对比：MLP(摊平) vs LSTM vs GRU vs 1D-CNN\n\n")
        f.write(f"数据：有 g1-g4 完整数据的学生（{len(df)} 人），特征 (4,7) 序列，标签 9 类 rule_class\n\n")
        f.write("## 结果\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n## 结论\n\n")
        best = summary.index[0]
        mlp_acc = summary.loc["MLP_flat", "acc_mean"]
        best_acc = summary.iloc[0]["acc_mean"]
        diff = best_acc - mlp_acc
        if abs(diff) < 0.01:
            f.write(f"显式时序建模（LSTM/GRU/CNN1D）与摊平 MLP 差距 <1%，说明在 4 个时间点的短序列上，时序建模没有明显增益。摊平特征已足够捕捉时序信息。\n")
        else:
            f.write(f"最佳模型 {best}（{best_acc:.4f}）比 MLP 摊平（{mlp_acc:.4f}）高 {diff:.4f}，说明显式时序建模有增量贡献。\n")

    print(f"\n写入 {OUT_DIR / 'temporal_results.csv'}")
    print(f"写入 {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    sys.exit(main())
