#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_raw_vs_hi_features.py — 原始特征 vs HI 管道特征 对比实验

核心问题（来自研究设计）：
  3×3 标签由 HI + slope + 阈值规则算出，但 HI 本身就是 7 指标的加权和。
  问：7 个原始指标能不能不经过 HI 管道，直接学到 9 类？

  - 如果能 → 说明 9 类结构是数据内禀的，HI 只是其中一种编码方式
  - 如果不能完全复现 → 说明 HI 的权重+方向校验+分层 Z-score 确实提取了
    原始指标无法直接表达的信息

对比三组特征（同一模型 MLP、同一 split、3 seeds）：
  A. 7-D 原始：latest available 的 7 项体测 raw 值（未标准化）
  B. 14-D 扩展：7 latest + 7 slope_per_year
  C. 57-D HI 管道：28 z-score(按性别 g1 参考) + 28 validity mask + 1 gender

输出：
  - outputs/raw_vs_hi/comparison_table.csv
  - outputs/raw_vs_hi/summary.md
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
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
MANIFEST_CSV = ROOT / "experiments" / "review_20260910" / "hi9_12models" / "split_manifest.csv"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"
OUT_DIR = Path(__file__).parent / "outputs" / "raw_vs_hi"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

METRICS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data():
    """加载三种特征集，保证同一批学生、同一 split。"""
    sys.path.insert(0, str(HI9BENCH_DIR))
    from hi9bench.data import load_bundle_legacy_unverified

    # 57-D HI 管道特征
    bundle = load_bundle_legacy_unverified(str(RAW_CSV), str(LABELS_CSV), split_seed=91)

    # 验证 split（统一转 int 比较）
    manifest = pd.read_csv(MANIFEST_CSV, dtype={"student_id": str})
    train_ids_bundle = set(int(s) for s in bundle.ids_train)
    train_ids_manifest = set(int(s) for s in manifest[manifest.split == "train"].student_id)
    assert train_ids_bundle == train_ids_manifest, f"split mismatch"

    # 构造 7-D 和 14-D 原始特征
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("", "_raw"))

    # latest available 7 项 + slope
    def latest_val(row, metric):
        pat = str(int(row["real_obs_pattern"])) if pd.notna(row["real_obs_pattern"]) else ""
        for g in [4, 3, 2, 1]:
            if str(g) in pat and pd.notna(row.get(f"{metric}_g{g}")):
                return float(row[f"{metric}_g{g}"])
        return np.nan

    # 对齐到 bundle 的 id 顺序（bundle id 是 int，labels 是零填充字符串）
    id_to_idx = {sid: i for i, sid in enumerate(df["student_id"].astype(str))}
    all_ids = np.concatenate([bundle.ids_train, bundle.ids_val, bundle.ids_test])
    all_ids_str = np.array([f"{int(sid):05d}" for sid in all_ids])
    idx_order = [id_to_idx[sid] for sid in all_ids_str]
    df_aligned = df.iloc[idx_order].reset_index(drop=True)

    # 7-D: latest raw values
    X_7d = np.column_stack([
        df_aligned.apply(lambda r: latest_val(r, m), axis=1).to_numpy(dtype=np.float32)
        for m in METRICS
    ])
    # 填充 NaN（用该列中位数）
    for j in range(X_7d.shape[1]):
        mask = np.isnan(X_7d[:, j])
        if mask.any():
            X_7d[mask, j] = np.nanmedian(X_7d[:, j])

    # 14-D: 7 latest + 7 slope（用 (g4-g1)/3 近似，或用 labels 里的 HI_slope 不行——那是 HI 的 slope）
    # 这里用每项指标的 (latest - g1) / n_years 近似 slope
    slopes = []
    for m in METRICS:
        g1 = df_aligned[f"{m}_g1"].to_numpy(dtype=np.float32)
        latest = np.array([latest_val(r, m) for _, r in df_aligned.iterrows()], dtype=np.float32)
        n_years = df_aligned["real_obs_pattern"].apply(
            lambda p: len(str(int(p))) if pd.notna(p) else 1
        ).to_numpy(dtype=np.float32)
        slope = (latest - g1) / np.maximum(n_years - 1, 1)
        slope = np.nan_to_num(slope, nan=0.0)
        slopes.append(slope)
    X_14d = np.column_stack([X_7d, np.column_stack(slopes)]).astype(np.float32)

    # 57-D: HI 管道特征（直接用 bundle）
    X_57d = np.concatenate([bundle.X_train, bundle.X_val, bundle.X_test], axis=0).astype(np.float32)
    y_all = np.concatenate([bundle.y_train, bundle.y_val, bundle.y_test]).astype(np.int64)

    # split 索引
    n_tr, n_va = len(bundle.X_train), len(bundle.X_val)
    tr_idx = np.arange(n_tr)
    va_idx = np.arange(n_tr, n_tr + n_va)
    te_idx = np.arange(n_tr + n_va, len(y_all))

    return {
        "7d_raw": (X_7d, y_all, tr_idx, va_idx, te_idx),
        "14d_raw_slope": (X_14d, y_all, tr_idx, va_idx, te_idx),
        "57d_hi_pipeline": (X_57d, y_all, tr_idx, va_idx, te_idx),
    }


# ---------------------------------------------------------------------------
# MLP 模型（与 ModernNCA 相同 backbone，无 NCA loss）
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, n_classes=9, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, seed, epochs=60, batch_size=256, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = DEVICE
    n_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1

    model = SimpleMLP(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    total_steps = max(1, epochs * (len(X_train) // batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    yva = torch.from_numpy(y_val).to(device)
    Xte = torch.from_numpy(X_test).to(device)

    # sqrt-inverse class weights
    counts = torch.bincount(ytr, minlength=n_classes).float()
    cw = (1.0 / counts.clamp(min=1)).sqrt()
    cw = cw / cw.sum() * n_classes
    cw = cw.to(device)

    best_val = -1.0
    best_state = None
    n = len(X_train)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            idx = perm[s:e]
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(Xte).argmax(1).cpu().numpy()
    return test_pred, accuracy_score(y_test, test_pred), f1_score(y_test, test_pred, average="macro")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备 = {DEVICE}")
    print("[1/3] 加载三种特征集 ...")
    datasets = load_data()
    for name, (X, y, tr, va, te) in datasets.items():
        print(f"      {name:20s}: {X.shape[1]}-D, train={len(tr)}, val={len(va)}, test={len(te)}")

    print("[2/3] 训练 MLP（每种特征 × 3 seeds）...")
    results = {}
    for feat_name, (X_all, y_all, tr, va, te) in datasets.items():
        X_train, y_train = X_all[tr], y_all[tr]
        X_val, y_val = X_all[va], y_all[va]
        X_test, y_test = X_all[te], y_all[te]

        # 标准化（用 train 的 mean/std）
        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = ((X_train - mu) / sd).astype(np.float32)
        X_val = ((X_val - mu) / sd).astype(np.float32)
        X_test = ((X_test - mu) / sd).astype(np.float32)

        accs, f1s = [], []
        for seed in SEEDS:
            t0 = time.time()
            _, acc, f1 = fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, seed)
            dt = time.time() - t0
            accs.append(acc)
            f1s.append(f1)
            print(f"      {feat_name:20s} seed {seed}: acc={acc:.4f} f1={f1:.4f} ({dt:.1f}s)")
        results[feat_name] = {
            "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
            "dim": X_all.shape[1],
        }

    print("[3/3] 生成对比报告 ...")
    rows = []
    for name, r in results.items():
        rows.append({
            "feature_set": name, "dim": r["dim"],
            "accuracy": round(r["acc_mean"], 4), "accuracy_sd": round(r["acc_sd"], 4),
            "macro_f1": round(r["f1_mean"], 4), "macro_f1_sd": round(r["f1_sd"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "comparison_table.csv", index=False)

    # 解读
    acc_7d = results["7d_raw"]["acc_mean"]
    acc_14d = results["14d_raw_slope"]["acc_mean"]
    acc_57d = results["57d_hi_pipeline"]["acc_mean"]
    gap = acc_57d - acc_7d

    s = []
    s.append("# 原始特征 vs HI 管道特征 对比实验\n")
    s.append("## 实验设计")
    s.append("- 同一 MLP 模型、同一 split、3 seeds（42/43/44）")
    s.append("- 三组特征：")
    s.append("  - **7-D 原始**：latest available 的 7 项体测 raw 值")
    s.append("  - **14-D 扩展**：7 latest + 7 项 slope_per_year")
    s.append("  - **57-D HI 管道**：28 z-score(按性别 g1 参考分布) + 28 validity mask + 1 gender\n")
    s.append("## 结果")
    s.append("| 特征集 | 维度 | accuracy | macro_f1 |")
    s.append("|---|---|---|---|")
    for _, r in df.iterrows():
        s.append(f"| {r['feature_set']} | {r['dim']} | {r['accuracy']:.4f} ± {r['accuracy_sd']:.4f} | {r['macro_f1']:.4f} ± {r['macro_f1_sd']:.4f} |")
    s.append("")
    s.append("## 解读")
    s.append(f"- 7-D 原始特征准确率 = **{acc_7d:.4f}**")
    s.append(f"- 14-D（+slope）准确率 = **{acc_14d:.4f}**")
    s.append(f"- 57-D HI 管道准确率 = **{acc_57d:.4f}**")
    s.append(f"- HI 管道相对原始 7-D 的提升 = **{gap:.4f}（{gap*100:.2f} 个百分点）**")
    s.append("")
    if gap < 0.02:
        s.append("**结论：9 类结构高度内禀于原始 7 项体测数据。**")
        s.append("HI 管道（加权和+分层 z-score+validity mask）带来的提升有限，")
        s.append("说明 9 类标签主要由原始体测指标的水平和趋势决定，HI 只是一种编码方式。")
    elif gap < 0.05:
        s.append("**结论：9 类结构部分内禀于原始数据，HI 管道有中等提升。**")
        s.append("原始 7 项已能学到大部分结构，但 HI 的分层 z-score 和 validity mask")
        s.append("仍提取了额外信息（尤其是跨年级趋势和缺测模式）。")
    else:
        s.append("**结论：HI 管道确实提取了原始指标无法直接表达的信息。**")
        s.append(f"HI 的权重+方向校验+分层 Z-score+validity mask 带来 {gap*100:.1f} 个百分点的显著提升，")
        s.append("说明 9 类结构不完全内禀于原始 7 项，HI 管道的特征工程是必要的。")
    s.append("")
    s.append(f"- 加入 slope 后（14-D）相对 7-D 的提升 = {acc_14d - acc_7d:.4f}")
    s.append(f"- 57-D 相对 14-D 的提升 = {acc_57d - acc_14d:.4f}（主要来自跨年级 z-score + validity mask）")
    (OUT_DIR / "summary.md").write_text("\n".join(s), encoding="utf-8")

    print(f"\n完成 ✅")
    print(f"  7-D  raw:         {acc_7d:.4f} ± {results['7d_raw']['acc_sd']:.4f}")
    print(f"  14-D raw+slope:   {acc_14d:.4f} ± {results['14d_raw_slope']['acc_sd']:.4f}")
    print(f"  57-D HI pipeline: {acc_57d:.4f} ± {results['57d_hi_pipeline']['acc_sd']:.4f}")
    print(f"  写入 {OUT_DIR}/comparison_table.csv")
    print(f"  写入 {OUT_DIR}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
