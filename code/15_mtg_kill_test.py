#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_mtg_kill_test.py — MTG-Net 可行性验证（Kill Test）

科学问题：在 4 个时间点的体测数据上，显式构造的趋势特征（delta / slope / acceleration）
          能否比简单摊平原始值带来分类增益？

实验设计（逐步消融）：
  A. MLP_flat        : 7指标×4年级 raw 摊平 (28维) + gender = 29维  [基线]
  B. MLP_flat_delta  : A + 相邻年差值 (g2-g1, g3-g2, g4-g3) ×7 = 21维 → 50维
  C. MLP_flat+delta+slope : B + 全局OLS斜率 ×7 = 7维 → 57维
  D. MLP_full        : C + 加速度 (delta-of-delta) ×7×2 = 14维 → 71维

数据：有 g1-g4 完整数据的学生，标签 9 类 rule_class
评估：3 seeds，accuracy / macro_f1，逐步对比验证显式趋势特征的增量贡献

输出：
  - outputs/mtg_kill_test/mtg_results.csv
  - outputs/mtg_kill_test/summary.md
"""
from __future__ import annotations
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "mtg_kill_test"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
METRICS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]
GRADES = [1, 2, 3, 4]
N_CLASSES = 9
EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-3


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(RAW_CSV)
    labels = pd.read_csv(LABELS_CSV)
    df = df.merge(labels[["student_id", "rule_class", "rule_eligible"]],
                  on="student_id", how="inner")
    df = df[df["rule_eligible"] == True].copy()

    def has_all_grades(row):
        pat = str(int(row["real_obs_pattern"])) if pd.notna(row["real_obs_pattern"]) else ""
        return all(str(g) in pat for g in GRADES)
    df = df[df.apply(has_all_grades, axis=1)].copy()
    print(f"  有 g1-g4 完整数据的学生: {len(df)}")
    return df


def build_trend_features(df, feature_set="full"):
    """
    构造特征矩阵。feature_set:
      'flat'          : raw 摊平 (28) + gender (1) = 29
      'flat_delta'    : + 相邻年差值 (21) = 50
      'flat_delta_slope' : + 全局斜率 (7) = 57
      'full'          : + 加速度 (14) = 71
    """
    n = len(df)
    # 原始值矩阵 (n, 4, 7)
    raw = np.zeros((n, 4, 7), dtype=np.float32)
    for i, g in enumerate(GRADES):
        for j, m in enumerate(METRICS):
            raw[:, i, j] = pd.to_numeric(df[f"{m}_g{g}"], errors="coerce").to_numpy(float)

    # 用列中位数填充 NaN（在这个完整数据子集里 NaN 应该很少）
    for j in range(7):
        col = raw[:, :, j]
        med = np.nanmedian(col)
        raw[:, :, j] = np.nan_to_num(col, nan=med)

    features = {}

    # A. 原始值摊平 (n, 28)
    flat = raw.reshape(n, -1)
    features["flat"] = flat

    # B. 相邻年差值 (n, 21): g2-g1, g3-g2, g4-g3
    deltas = np.concatenate([raw[:, i+1, :] - raw[:, i, :] for i in range(3)], axis=1)
    features["delta"] = deltas

    # C. 全局 OLS 斜率 (n, 7): slope over 4 time points
    t = np.arange(1, 5, dtype=np.float32)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    slopes = np.zeros((n, 7), dtype=np.float32)
    for j in range(7):
        y_mean = raw[:, :, j].mean(axis=1, keepdims=True)
        num = ((raw[:, :, j] - y_mean) * (t - t_mean)).sum(axis=1)
        slopes[:, j] = num / t_var
    features["slope"] = slopes

    # D. 加速度 delta-of-delta (n, 14): (g3-g2)-(g2-g1), (g4-g3)-(g3-g2)
    accel = np.concatenate([
        deltas[:, 7:14] - deltas[:, 0:7],   # (g3-g2)-(g2-g1)
        deltas[:, 14:21] - deltas[:, 7:14],  # (g4-g3)-(g3-g2)
    ], axis=1)
    features["accel"] = accel

    # gender
    gender = (df["gender"] == "男").to_numpy(dtype=np.float32).reshape(-1, 1)

    # 组合
    if feature_set == "flat":
        X = np.column_stack([flat, gender])
    elif feature_set == "flat_delta":
        X = np.column_stack([flat, deltas, gender])
    elif feature_set == "flat_delta_slope":
        X = np.column_stack([flat, deltas, slopes, gender])
    elif feature_set == "full":
        X = np.column_stack([flat, deltas, slopes, accel, gender])
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    y = df["rule_class"].to_numpy(dtype=np.int64)
    return X.astype(np.float32), y


def split_data(n, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def train_model(X, y, train_idx, val_idx, test_idx, seed, in_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 标准化（用训练集统计量）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    model = MLP(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)

    Xtr = torch.from_numpy(X_train).to(DEVICE)
    ytr = torch.from_numpy(y[train_idx]).to(DEVICE)
    Xva = torch.from_numpy(X_val).to(DEVICE)
    yva = torch.from_numpy(y[val_idx]).to(DEVICE)
    Xte = torch.from_numpy(X_test).to(DEVICE)

    # class weights (sqrt-inverse)
    counts = torch.bincount(ytr, minlength=N_CLASSES).float()
    cw = (1.0 / counts.clamp(min=1)).sqrt()
    cw = cw / cw.sum() * N_CLASSES
    cw = cw.to(DEVICE)

    best_val_acc = 0.0
    best_state = None
    n = len(Xtr)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, BATCH_SIZE):
            e = min(s + BATCH_SIZE, n)
            idx = perm[s:e]
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(Xva).argmax(1).cpu().numpy()
            va_acc = accuracy_score(y[val_idx], va_pred)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(Xte).argmax(1).cpu().numpy()
    te_acc = accuracy_score(y[test_idx], te_pred)
    te_f1 = f1_score(y[test_idx], te_pred, average="macro")
    return te_acc, te_f1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
FEATURE_SETS = [
    ("flat", "A. Raw flat (28+1=29D)"),
    ("flat_delta", "B. Flat + Δ (28+21+1=50D)"),
    ("flat_delta_slope", "C. Flat + Δ + slope (28+21+7+1=57D)"),
    ("full", "D. Full: Flat + Δ + slope + accel (28+21+7+14+1=71D)"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] 加载数据 ...")
    df = load_data()

    print(f"\n[2/3] 训练 4 种特征集 × {len(SEEDS)} seeds ...")
    all_results = []

    for feat_key, feat_name in FEATURE_SETS:
        X, y = build_trend_features(df, feature_set=feat_key)
        in_dim = X.shape[1]
        print(f"\n  === {feat_name} (in_dim={in_dim}) ===")
        for seed in SEEDS:
            tr, va, te = split_data(len(df), seed=seed)
            t0 = time.time()
            acc, f1 = train_model(X, y, tr, va, te, seed, in_dim)
            elapsed = time.time() - t0
            print(f"    seed={seed}: acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)")
            all_results.append({
                "feature_set": feat_key,
                "feature_name": feat_name,
                "in_dim": in_dim,
                "seed": seed,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "seconds": round(elapsed, 1),
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "mtg_results.csv", index=False)

    # 汇总
    summary = results_df.groupby(["feature_set", "feature_name", "in_dim"]).agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
    ).round(4).reset_index().sort_values("acc_mean", ascending=False)

    print(f"\n[3/3] 汇总：")
    print(summary[["feature_name", "in_dim", "acc_mean", "acc_std", "f1_mean", "f1_std"]].to_string(index=False))

    # 写 summary.md
    flat_acc = summary[summary["feature_set"] == "flat"]["acc_mean"].values[0]
    flat_f1 = summary[summary["feature_set"] == "flat"]["f1_mean"].values[0]
    best_row = summary.iloc[0]
    best_acc = best_row["acc_mean"]
    best_name = best_row["feature_name"]
    gain = best_acc - flat_acc

    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("# MTG-Net Kill Test：显式趋势特征是否有用？\n\n")
        f.write(f"数据：有 g1-g4 完整数据的学生（{len(df)} 人），标签 9 类 rule_class\n\n")
        f.write("## 结果\n\n")
        f.write(summary[["feature_name", "in_dim", "acc_mean", "acc_std", "f1_mean", "f1_std"]].to_markdown(index=False))
        f.write("\n\n## 逐步增益分析\n\n")
        f.write(f"| 对比 | Acc 增益 | F1 增益 |\n")
        f.write(f"|---|---|---|\n")
        prev_acc, prev_f1 = flat_acc, flat_f1
        prev_name = "A. Raw flat"
        for _, row in summary.sort_values("in_dim").iterrows():
            if row["feature_set"] == "flat":
                continue
            f.write(f"{prev_name} → {row['feature_name']} | +{row['acc_mean'] - prev_acc:.4f} | +{row['f1_mean'] - prev_f1:.4f} |\n")
            prev_acc, prev_f1 = row["acc_mean"], row["f1_mean"]
            prev_name = row["feature_name"]
        f.write(f"\n## 结论\n\n")
        if gain > 0.01:
            f.write(f"**显式趋势特征有用**。最佳特征集 {best_name}（acc={best_acc:.4f}）比基线 Raw flat（acc={flat_acc:.4f}）高 {gain:.4f}（{gain*100:.1f}个百分点）。")
            f.write(f" 这验证了 MTG-Net 的核心假设——在短时序体测数据上，显式构造的 delta/slope/acceleration 特征能带来分类增益，值得进一步发展为多尺度趋势门控模块。\n")
        elif gain > 0.003:
            f.write(f"**显式趋势特征有微弱增益**。最佳 {best_name}（acc={best_acc:.4f}）比基线高 {gain:.4f}，增益较小但稳定。MTG-Net 方向可以探索但预期提升有限。\n")
        else:
            f.write(f"**显式趋势特征无明显增益**。最佳 {best_name}（acc={best_acc:.4f}）与基线 Raw flat（acc={flat_acc:.4f}）差距 <0.3%。")
            f.write(f" 这说明在 4 个时间点的短序列上，MLP 已经能从摊平特征中学到趋势信息，显式构造趋势特征的增量贡献有限。MTG-Net 方向可能不成立，建议放弃方法创新，转向基准+应用论文。\n")
        f.write(f"\n基线 F1={flat_f1:.4f}，最佳 F1={best_row['f1_mean']:.4f}\n")

    print(f"\n写入 {OUT_DIR / 'mtg_results.csv'}")
    print(f"写入 {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    sys.exit(main())
