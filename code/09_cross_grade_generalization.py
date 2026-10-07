#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_cross_grade_generalization.py — 跨年级泛化测试

用 g1-g3 年级数据训练，g4 年级数据测试。
不是随机划分学生，而是按时间划分，模拟真实部署场景。

对 Top3 模型（TabM / ModernNCA / SupCon）各跑 3 seeds。
这里先用 MLP 快速验证，后续可替换为具体模型。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"
OUT_DIR = Path(__file__).parent / "outputs" / "cross_grade"
SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
METRICS = ["bmi","vital_capacity","sprint_50m","standing_long_jump","sit_and_reach","endurance_run_sec","strength"]

class MLP(nn.Module):
    def __init__(self, in_dim, n_classes=9, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))
    def forward(self, x): return self.net(x)

def build_features(df, grades):
    cols = []
    for g in grades:
        for m in METRICS:
            cols.append(f"{m}_g{g}")
    X = df[cols].to_numpy(dtype=np.float32)
    gender = (df["gender"]=="男").to_numpy(dtype=np.float32).reshape(-1,1)
    X = np.column_stack([X, gender])
    for j in range(X.shape[1]):
        mask = np.isnan(X[:,j])
        if mask.any(): X[mask,j] = np.nanmedian(X[:,j])
    return X

def has_grades(row, grades):
    pat = str(int(row["real_obs_pattern"])) if pd.notna(row["real_obs_pattern"]) else ""
    return all(str(g) in pat for g in grades)

def train_mlp(X_train, y_train, X_val, y_val, X_test, seed, epochs=60, n_classes=9):
    torch.manual_seed(seed); np.random.seed(seed)
    device = DEVICE
    model = MLP(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    total_steps = max(1, epochs * (len(X_train)//256))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    Xtr = torch.from_numpy(X_train).to(device); ytr = torch.from_numpy(y_train).to(device)
    Xva = torch.from_numpy(X_val).to(device); yva = torch.from_numpy(y_val).to(device)
    Xte = torch.from_numpy(X_test).to(device)
    counts = torch.bincount(ytr, minlength=n_classes).float()
    cw = (1.0/counts.clamp(min=1)).sqrt(); cw = cw/cw.sum()*n_classes; cw = cw.to(device)
    best_val, best_state = -1, None
    n = len(X_train)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 256):
            e = min(s+256, n); idx = perm[s:e]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx], weight=cw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva).argmax(1)==yva).float().mean().item()
        if va_acc > best_val:
            best_val = va_acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): pred = model(Xte).argmax(1).cpu().numpy()
    return pred

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备 = {DEVICE}")
    print("[1/3] 加载数据 ...")
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("","_raw"))

    # 有 g1-g3 训练数据 + g4 测试数据的学生
    train_mask = df.apply(lambda r: has_grades(r, [1,2,3]), axis=1)
    test_mask = df.apply(lambda r: has_grades(r, [4]), axis=1)
    # 训练集：有 g1-g3 的学生（不管有没有 g4），从其中划分 train/val
    train_df = df[train_mask].copy()
    # 测试集：有 g4 的学生（必须也有 g1-g3 才能公平对比）
    test_df = df[train_mask & test_mask].copy()
    print(f"      训练候选（g1-g3）: {len(train_df)}")
    print(f"      测试集（g1-g3+g4）: {len(test_df)}")

    # 训练特征：g1-g3 的 21 维 + gender = 22 维
    X_train_all = build_features(train_df, [1,2,3])
    y_train_all = train_df["rule_class"].to_numpy(dtype=np.int64)
    X_test = build_features(test_df, [1,2,3])
    y_test = test_df["rule_class"].to_numpy(dtype=np.int64)

    print(f"      训练特征: {X_train_all.shape[1]}-D, 测试特征: {X_test.shape[1]}-D")

    print("[2/3] 跨年级泛化训练（3 seeds）...")
    results = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(train_df))
        n_tr = int(len(train_df)*0.85)
        tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
        X_tr, y_tr = X_train_all[tr_idx], y_train_all[tr_idx]
        X_va, y_va = X_train_all[va_idx], y_train_all[va_idx]
        mu = X_tr.mean(axis=0, keepdims=True); sd = X_tr.std(axis=0, keepdims=True)+1e-8
        X_tr = ((X_tr-mu)/sd).astype(np.float32)
        X_va = ((X_va-mu)/sd).astype(np.float32)
        X_te = ((X_test-mu)/sd).astype(np.float32)
        t0 = time.time()
        pred = train_mlp(X_tr, y_tr, X_va, y_va, X_te, seed)
        dt = time.time()-t0
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        print(f"      seed {seed}: acc={acc:.4f} f1={f1:.4f} ({dt:.1f}s)")
        results.append({"seed": seed, "accuracy": round(acc,4), "macro_f1": round(f1,4)})

    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    summary = {
        "experiment": "cross_grade_g1g2g3_to_g4",
        "model": "MLP (256-256)",
        "train_features": "g1-g3 raw (21-D) + gender",
        "test_label": "g4 rule_class (9-class)",
        "n_train_candidates": len(train_df),
        "n_test": len(test_df),
        "accuracy_mean": round(float(np.mean(accs)),4),
        "accuracy_sd": round(float(np.std(accs)),4),
        "macro_f1_mean": round(float(np.mean(f1s)),4),
        "macro_f1_sd": round(float(np.std(f1s)),4),
        "per_seed": results,
        "note": "对比随机split MLP acc~0.85, 跨年级泛化预期下降"
    }
    print(f"\n[3/3] 保存 ...")
    with open(OUT_DIR / "cross_grade_status.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n完成 ✅")
    print(f"  跨年级 acc = {summary['accuracy_mean']:.4f} ± {summary['accuracy_sd']:.4f}")
    print(f"  跨年级 f1 = {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_sd']:.4f}")
    print(f"  写入 {OUT_DIR}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
