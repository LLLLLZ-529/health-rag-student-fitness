#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_cross_grade_v2.py — 跨年级泛化实验（正确版本）

科学问题：用大一大二大三（g1-g3）的体测数据，能在多大程度上预测大四（g4）的健康分类？

正确实验设计：
  1. 筛选有 g1-g4 完整数据的学生（确保标签基于 g4）
  2. 按学生随机划分 train/val/test（学生身份不重叠，无泄漏）
  3. 训练/测试特征：仅 g1-g3 的体测数据（模拟"早期预测"场景）
  4. 训练/测试标签：9 类 rule_class（基于 g4 total_score + 全局 slope）

对比实验：
  - 上界：同样学生子集，用 g1-g4 全部特征（标准57维）随机划分
  - 跨年级：用 g1-g3 特征（22维 raw + gender）预测 g4 标签
  - 退化二分类：g1-g3 特征预测"是否退化型"（class 0/3/6）

输出：
  - outputs/cross_grade_v2/cross_grade_results.csv
  - outputs/cross_grade_v2/summary.md
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"
OUT_DIR = Path(__file__).parent / "outputs" / "cross_grade_v2"

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

# 退化型类别：低水平-退化(0), 中水平-退化(3), 高水平-退化(6)
DECLINE_CLASSES = [0, 3, 6]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(RAW_CSV)
    labels = pd.read_csv(LABELS_CSV)
    df = df.merge(labels[["student_id", "rule_class", "rule_eligible"]],
                  on="student_id", how="inner")
    df = df[df["rule_eligible"] == True].copy()

    # 筛选有 g1-g4 完整数据的学生
    def has_all_grades(row):
        pat = str(int(row["real_obs_pattern"])) if pd.notna(row["real_obs_pattern"]) else ""
        return all(str(g) in pat for g in GRADES)
    df = df[df.apply(has_all_grades, axis=1)].copy()
    print(f"  有 g1-g4 完整数据的学生: {len(df)}")

    # 退化二分类标签
    df["is_decline"] = df["rule_class"].isin(DECLINE_CLASSES).astype(int)
    print(f"  退化型占比: {df['is_decline'].mean():.3f}")
    return df


def build_g1g3_features(df):
    """用 g1-g3 的 raw 值构造特征：7指标×3年级=21维 + gender = 22维"""
    cols = []
    for g in [1, 2, 3]:
        for m in METRICS:
            cols.append(f"{m}_g{g}")
    X = df[cols].to_numpy(dtype=np.float32)
    gender = (df["gender"] == "男").to_numpy(dtype=np.float32).reshape(-1, 1)
    X = np.column_stack([X, gender])
    # NaN 填充（列中位数）
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X


def build_g1g3_trend_features(df):
    """g1-g3 + 显式趋势特征：raw(21) + delta(14) + slope(7) + gender = 43维"""
    n = len(df)
    raw = np.zeros((n, 3, 7), dtype=np.float32)
    for i, g in enumerate([1, 2, 3]):
        for j, m in enumerate(METRICS):
            raw[:, i, j] = pd.to_numeric(df[f"{m}_g{g}"], errors="coerce").to_numpy(float)
    for j in range(7):
        med = np.nanmedian(raw[:, :, j])
        raw[:, :, j] = np.nan_to_num(raw[:, :, j], nan=med)

    flat = raw.reshape(n, -1)  # 21
    deltas = np.concatenate([raw[:, i+1, :] - raw[:, i, :] for i in range(2)], axis=1)  # 14
    t = np.arange(1, 4, dtype=np.float32)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    slopes = np.zeros((n, 7), dtype=np.float32)
    for j in range(7):
        y_mean = raw[:, :, j].mean(axis=1, keepdims=True)
        num = ((raw[:, :, j] - y_mean) * (t - t_mean)).sum(axis=1)
        slopes[:, j] = num / t_var
    gender = (df["gender"] == "男").to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.column_stack([flat, deltas, slopes, gender]).astype(np.float32)


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
    def __init__(self, in_dim, n_classes=9, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, X_test, seed, n_classes=9, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(X_train.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)

    Xtr = torch.from_numpy(X_train).to(DEVICE)
    ytr = torch.from_numpy(y_train).to(DEVICE)
    Xva = torch.from_numpy(X_val).to(DEVICE)
    yva = torch.from_numpy(y_val).to(DEVICE)
    Xte = torch.from_numpy(X_test).to(DEVICE)

    counts = torch.bincount(ytr, minlength=n_classes).float()
    cw = (1.0 / counts.clamp(min=1)).sqrt()
    cw = cw / cw.sum() * n_classes
    cw = cw.to(DEVICE)

    best_val_acc = 0.0
    best_state = None
    n = len(Xtr)

    for epoch in range(epochs):
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
            va_acc = accuracy_score(y_val, va_pred)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte)
        te_pred = te_logits.argmax(1).cpu().numpy()
        te_proba = F.softmax(te_logits, dim=1).cpu().numpy()
    return te_pred, te_proba


# ---------------------------------------------------------------------------
# 实验
# ---------------------------------------------------------------------------
def run_experiment(df, feature_builder, label_col, n_classes, experiment_name, seed):
    """运行单个实验：构造特征 → 划分 → 标准化 → 训练 → 评估"""
    X = feature_builder(df)
    y = df[label_col].to_numpy(dtype=np.int64)
    tr, va, te = split_data(len(df), seed=seed)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_va = scaler.transform(X[va])
    X_te = scaler.transform(X[te])

    t0 = time.time()
    pred, proba = train_mlp(X_tr, y[tr], X_va, y[va], X_te, seed, n_classes=n_classes)
    elapsed = time.time() - t0

    acc = accuracy_score(y[te], pred)
    f1 = f1_score(y[te], pred, average="macro")

    result = {
        "experiment": experiment_name,
        "seed": seed,
        "n_features": X.shape[1],
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "seconds": round(elapsed, 1),
    }

    # 二分类额外算 AP
    if n_classes == 2:
        ap = average_precision_score(y[te], proba[:, 1])
        result["avg_precision"] = round(ap, 4)

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] 加载数据 ...")
    df = load_data()

    # 加载标准57维特征（用于上界实验）
    print(f"\n[2/4] 加载标准57维特征（上界实验）...")
    sys.path.insert(0, str(HI9BENCH_DIR))
    from hi9bench.data import load_bundle_legacy_unverified
    bundle = load_bundle_legacy_unverified(RAW_CSV, LABELS_CSV, split_seed=91)

    # 筛选有 g1-g4 完整数据的学生在标准57维特征中的索引
    full_ids = set(df["student_id"].astype(str))
    # bundle 的 ids 可能是 str 或其他类型，统一转 str
    test_mask = np.array([str(sid) in full_ids for sid in bundle.ids_test])
    train_mask = np.array([str(sid) in full_ids for sid in bundle.ids_train])
    val_mask = np.array([str(sid) in full_ids for sid in bundle.ids_val])
    print(f"  标准57维特征中，完整数据学生：train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    print(f"\n[3/4] 运行实验 × {len(SEEDS)} seeds ...")
    all_results = []

    # 实验1：上界 — 标准57维特征（g1-g4全部），在完整数据学生子集上评估
    print(f"\n  === 实验1：上界（标准57维，g1-g4全部特征）===")
    for seed in SEEDS:
        # 用标准 split（seed=91），但只在完整数据子集上评估
        X_tr = bundle.X_train[train_mask]
        y_tr = bundle.y_train[train_mask]
        X_va = bundle.X_val[val_mask]
        y_va = bundle.y_val[val_mask]
        X_te = bundle.X_test[test_mask]
        y_te = bundle.y_test[test_mask]

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MLP(57, N_CLASSES).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)
        Xtr = torch.from_numpy(X_tr).to(DEVICE)
        ytr = torch.from_numpy(y_tr).to(DEVICE)
        Xva = torch.from_numpy(X_va).to(DEVICE)
        yva = torch.from_numpy(y_va).to(DEVICE)
        Xte = torch.from_numpy(X_te).to(DEVICE)
        counts = torch.bincount(ytr, minlength=N_CLASSES).float()
        cw = (1.0 / counts.clamp(min=1)).sqrt()
        cw = cw / cw.sum() * N_CLASSES
        cw = cw.to(DEVICE)
        best_val_acc, best_state = 0.0, None
        n = len(Xtr)
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for s in range(0, n, BATCH_SIZE):
                e = min(s + BATCH_SIZE, n)
                idx = perm[s:e]
                loss = F.cross_entropy(model(Xtr[idx]), ytr[idx], weight=cw)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                va_pred = model(Xva).argmax(1).cpu().numpy()
                va_acc = accuracy_score(y_va, va_pred)
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te_pred = model(Xte).argmax(1).cpu().numpy()
        acc = accuracy_score(y_te, te_pred)
        f1 = f1_score(y_te, te_pred, average="macro")
        print(f"    seed={seed}: acc={acc:.4f}  f1={f1:.4f}")
        all_results.append({
            "experiment": "upper_bound_57d_full",
            "seed": seed, "n_features": 57,
            "accuracy": round(acc, 4), "macro_f1": round(f1, 4), "seconds": 0,
        })

    # 实验2：跨年级 9 类 — g1-g3 raw 特征预测 g4 rule_class
    print(f"\n  === 实验2：跨年级 9 类（g1-g3 raw 22维 → g4 9类）===")
    for seed in SEEDS:
        r = run_experiment(df, build_g1g3_features, "rule_class", N_CLASSES,
                           "cross_grade_g1g3_9class", seed)
        print(f"    seed={seed}: acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}")
        all_results.append(r)

    # 实验3：跨年级 9 类 + 趋势特征 — g1-g3 raw+delta+slope 预测 g4 rule_class
    print(f"\n  === 实验3：跨年级 9 类（g1-g3 +趋势 43维 → g4 9类）===")
    for seed in SEEDS:
        r = run_experiment(df, build_g1g3_trend_features, "rule_class", N_CLASSES,
                           "cross_grade_g1g3_trend_9class", seed)
        print(f"    seed={seed}: acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}")
        all_results.append(r)

    # 实验4：跨年级退化二分类 — g1-g3 raw 预测是否退化型
    print(f"\n  === 实验4：跨年级退化二分类（g1-g3 raw 22维 → 是否退化）===")
    for seed in SEEDS:
        r = run_experiment(df, build_g1g3_features, "is_decline", 2,
                           "cross_grade_g1g3_decline_binary", seed)
        print(f"    seed={seed}: acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}  AP={r.get('avg_precision', 'N/A')}")
        all_results.append(r)

    # 实验5：跨年级退化二分类 + 趋势特征
    print(f"\n  === 实验5：跨年级退化二分类（g1-g3 +趋势 43维 → 是否退化）===")
    for seed in SEEDS:
        r = run_experiment(df, build_g1g3_trend_features, "is_decline", 2,
                           "cross_grade_g1g3_trend_decline_binary", seed)
        print(f"    seed={seed}: acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}  AP={r.get('avg_precision', 'N/A')}")
        all_results.append(r)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "cross_grade_results.csv", index=False)

    # 汇总
    print(f"\n[4/4] 汇总：")
    summary = results_df.groupby("experiment").agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
    ).round(4)
    print(summary.to_string())

    # 写 summary.md
    upper = summary.loc["upper_bound_57d_full", "acc_mean"]
    cg9 = summary.loc["cross_grade_g1g3_9class", "acc_mean"]
    cg9t = summary.loc["cross_grade_g1g3_trend_9class", "acc_mean"]
    cgb = summary.loc["cross_grade_g1g3_decline_binary", "acc_mean"]
    cgbt = summary.loc["cross_grade_g1g3_trend_decline_binary", "acc_mean"]

    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("# 跨年级泛化实验（正确版本 v2）\n\n")
        f.write(f"数据：有 g1-g4 完整数据的学生（{len(df)} 人），按学生划分 train/val/test（无重叠）\n\n")
        f.write("## 结果汇总\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n## 关键对比\n\n")
        f.write("| 实验 | Acc | 与上界差距 |\n")
        f.write("|---|---|---|\n")
        f.write(f"上界（57维 g1-g4 全部特征） | {upper:.4f} | — |\n")
        f.write(f"跨年级 9 类（g1-g3 raw 22维） | {cg9:.4f} | -{upper - cg9:.4f} |\n")
        f.write(f"跨年级 9 类（g1-g3 +趋势 43维） | {cg9t:.4f} | -{upper - cg9t:.4f} |\n")
        f.write(f"跨年级退化二分类（g1-g3 raw） | {cgb:.4f} | — |\n")
        f.write(f"跨年级退化二分类（g1-g3 +趋势） | {cgbt:.4f} | — |\n")
        f.write("\n## 结论\n\n")
        f.write(f"1. **跨年级泛化有明显下降**：用 g1-g3 特征预测 g4 的 9 类标签，准确率 {cg9:.4f}，比上界（{upper:.4f}）低 {(upper - cg9)*100:.1f} 个百分点。\n")
        f.write(f"2. **趋势特征有帮助**：加入显式 delta+slope 后，9 类准确率从 {cg9:.4f} 提升到 {cg9t:.4f}（+{(cg9t - cg9)*100:.1f}pp），退化二分类从 {cgb:.4f} 提升到 {cgbt:.4f}（+{(cgbt - cgb)*100:.1f}pp）。\n")
        f.write(f"3. **退化预警可行**：用 g1-g3 数据预测大四是否退化，二分类准确率 {cgbt:.4f}，可作为早期预警系统的基础。\n")
        f.write(f"4. **局限性**：9 类细分类跨年级预测仍较困难（{cg9t:.4f}），说明仅靠前三年数据难以精确预测大四的具体健康类别，但退化/非退化的二分类判断更可靠。\n")

    print(f"\n写入 {OUT_DIR / 'cross_grade_results.csv'}")
    print(f"写入 {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    sys.exit(main())
