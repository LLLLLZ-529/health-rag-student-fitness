#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_m2_decline_remaining_models.py — M2-3 补充：剩余 6 个模型的退化二分类

补充模型：SupCon, GRANDE, TabICLv2, TabNet, FT-Transformer, SAINT
与 13_m2_decline_multimodel.py 的结果合并，形成完整 15 模型对比。
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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
OUT_DIR = Path(__file__).parent / "outputs" / "m2_decline_multimodel"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
METRICS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]
GRADES = [1, 2]
N_CLASSES = 2
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3


# ---------------------------------------------------------------------------
# 数据加载（与 13 一致）
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


def standardize(X, train_idx):
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    return scaler.transform(X)


# ---------------------------------------------------------------------------
# 通用 torch 训练器
# ---------------------------------------------------------------------------
def train_torch(model, X, y, seed, epochs=EPOCHS, class_weight=True):
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = split_data(X, y, seed)
    X_s = standardize(X, train_idx)
    X_tr = torch.tensor(X_s[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[train_idx], dtype=torch.long)
    X_va = torch.tensor(X_s[val_idx], dtype=torch.float32)
    y_va = torch.tensor(y[val_idx], dtype=torch.long)
    X_te = torch.tensor(X_s[test_idx], dtype=torch.float32)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    weight = None
    if class_weight:
        n_pos = y[train_idx].sum()
        n_neg = len(y[train_idx]) - n_pos
        weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    best_val_acc, best_state = 0, None
    for epoch in range(epochs):
        model.train()
        rng = np.random.RandomState(seed + epoch)
        perm = rng.permutation(len(train_idx))
        for i in range(0, len(perm), BATCH_SIZE):
            bi = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_tr[bi].to(DEVICE))
            loss = F.cross_entropy(logits, y_tr[bi].to(DEVICE), weight=weight)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            va_pred = model(X_va.to(DEVICE)).argmax(1).cpu().numpy()
            va_acc = accuracy_score(y[val_idx], va_pred)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(X_te.to(DEVICE))
        te_proba = F.softmax(te_logits, dim=1)[:, 1].cpu().numpy()
        te_pred = te_logits.argmax(1).cpu().numpy()
    acc = accuracy_score(y[test_idx], te_pred)
    f1 = f1_score(y[test_idx], te_pred, average="macro")
    ap = average_precision_score(y[test_idx], te_proba)
    return acc, f1, ap


# ---------------------------------------------------------------------------
# 1. SupCon（对比学习预训练 + 线性分类头）
# ---------------------------------------------------------------------------
class SupConEncoder(nn.Module):
    def __init__(self, n_features=15, proj_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, proj_dim))

    def forward(self, x):
        h = self.encoder(x)
        z = self.proj(h)
        return F.normalize(z, dim=1), h


class SupConClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.encoder
        self.classifier = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


def supcon_loss(z1, z2, temperature=0.07):
    """NT-Xent loss for two views."""
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(len(z), device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    labels = torch.cat([torch.arange(len(z1)) + len(z1), torch.arange(len(z1))]).to(z.device)
    return F.cross_entropy(sim, labels)


def run_supcon(X, y, seed):
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = split_data(X, y, seed)
    X_s = standardize(X, train_idx)
    X_tr = torch.tensor(X_s[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[train_idx], dtype=torch.long)

    # 预训练 30 epochs 对比学习
    encoder = SupConEncoder().to(DEVICE)
    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(30):
        encoder.train()
        rng = np.random.RandomState(seed + epoch)
        perm = rng.permutation(len(train_idx))
        for i in range(0, len(perm), BATCH_SIZE):
            bi = perm[i:i + BATCH_SIZE]
            xb = X_tr[bi].to(DEVICE)
            # 两种数据增强：加噪声
            z1, _ = encoder(xb + torch.randn_like(xb) * 0.1)
            z2, _ = encoder(xb + torch.randn_like(xb) * 0.1)
            loss = supcon_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 微调分类头
    clf = SupConClassifier(encoder).to(DEVICE)
    return train_torch(clf, X, y, seed, epochs=30)


# ---------------------------------------------------------------------------
# 2. GRANDE
# ---------------------------------------------------------------------------
def run_grande(X, y, seed):
    sys.path.insert(0, str(HI9BENCH_DIR))
    from hi9bench.models import GRANDEModel
    train_idx, val_idx, test_idx = split_data(X, y, seed)
    X_s = standardize(X, train_idx)
    cfg = dict(depth=5, n_estimators=256, dropout=0.2, selected_variables=0.8,
               epochs=100, batch_size=256)
    model = GRANDEModel(cfg, seed=seed)
    t0 = time.time()
    model.fit(X_s[train_idx], y[train_idx], X_s[val_idx], y[val_idx])
    proba = model.predict_proba(X_s[test_idx])
    pred = proba.argmax(1)
    acc = accuracy_score(y[test_idx], pred)
    f1 = f1_score(y[test_idx], pred, average="macro")
    ap = average_precision_score(y[test_idx], proba[:, 1])
    return acc, f1, ap, time.time() - t0


# ---------------------------------------------------------------------------
# 3. TabICLv2
# ---------------------------------------------------------------------------
def run_tabiclv2(X, y, seed, max_ctx=10000):
    from tabicl import TabICLClassifier
    train_idx, val_idx, test_idx = split_data(X, y, seed)
    X_s = standardize(X, train_idx)
    X_ctx = np.vstack([X_s[train_idx], X_s[val_idx]])
    y_ctx = np.concatenate([y[train_idx], y[val_idx]])
    if len(X_ctx) > max_ctx:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_ctx), max_ctx, replace=False)
        X_ctx, y_ctx = X_ctx[idx], y_ctx[idx]
    t0 = time.time()
    clf = TabICLClassifier(n_estimators=4, device="mps", batch_size=32, random_state=seed)
    clf.fit(X_ctx, y_ctx)
    proba = clf.predict_proba(X_s[test_idx])
    pred = proba.argmax(1)
    acc = accuracy_score(y[test_idx], pred)
    f1 = f1_score(y[test_idx], pred, average="macro")
    ap = average_precision_score(y[test_idx], proba[:, 1])
    return acc, f1, ap, time.time() - t0


# ---------------------------------------------------------------------------
# 4. TabNet（自实现简化版）
# ---------------------------------------------------------------------------
class GLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
    def forward(self, x):
        x = self.fc(x)
        return x[:, :x.size(1)//2] * torch.sigmoid(x[:, x.size(1)//2:])


class TabNetSimple(nn.Module):
    def __init__(self, n_features=15, n_steps=3, hidden=64):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)
        self.initial = GLUBlock(n_features, hidden)
        self.steps = nn.ModuleList([GLUBlock(hidden, hidden) for _ in range(n_steps)])
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        x = self.bn(x)
        h = self.initial(x)
        for step in self.steps:
            h = h + step(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# 5. FT-Transformer（自实现）
# ---------------------------------------------------------------------------
class FTTransformer(nn.Module):
    def __init__(self, n_features=15, d_token=64, n_layers=3, n_heads=4):
        super().__init__()
        self.tokenizer = nn.Linear(1, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, n_features + 1, d_token) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=128,
            dropout=0.2, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, N_CLASSES))

    def forward(self, x):
        # x: (B, n_features) → (B, n_features, 1) → tokens
        tokens = self.tokenizer(x.unsqueeze(-1))  # (B, n_features, d_token)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_emb
        out = self.transformer(tokens)
        return self.head(out[:, 0])  # CLS token


# ---------------------------------------------------------------------------
# 6. SAINT（自实现简化版：自注意力 + 跨注意力）
# ---------------------------------------------------------------------------
class SAINTSimple(nn.Module):
    def __init__(self, n_features=15, d_token=32, n_layers=2, n_heads=4):
        super().__init__()
        self.tokenizer = nn.Linear(1, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        # 列自注意力（每列跨样本）+ 行自注意力（每行跨列）
        col_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=64,
            dropout=0.2, batch_first=True, activation='gelu')
        self.col_attn = nn.TransformerEncoder(col_layer, num_layers=n_layers)
        row_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=64,
            dropout=0.2, batch_first=True, activation='gelu')
        self.row_attn = nn.TransformerEncoder(row_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, N_CLASSES))

    def forward(self, x):
        B, D = x.shape
        tokens = self.tokenizer(x.unsqueeze(-1))  # (B, D, d_token)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        # 行注意力（每行跨列）
        tokens = self.row_attn(tokens)
        # 列注意力（每列跨行）— 简化为转置后做行注意力
        tokens_t = tokens.transpose(0, 1)  # (D+1, B, d_token)
        tokens_t = self.col_attn(tokens_t)
        tokens = tokens_t.transpose(0, 1)
        return self.head(tokens[:, 0])


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备 = {DEVICE}")
    print("[1/2] 加载数据 ...")
    df = load_data()
    X = build_features(df)
    y = df["is_decline"].to_numpy(dtype=np.int64)

    print(f"\n[2/2] 跑剩余 6 个模型 × {len(SEEDS)} seeds ...")
    all_results = []

    # --- SupCon ---
    for seed in SEEDS:
        t0 = time.time()
        acc, f1, ap = run_supcon(X, y, seed)
        elapsed = time.time() - t0
        print(f"  SupCon      seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
        all_results.append({"model": "SupCon", "seed": seed, "accuracy": round(acc,4),
                            "macro_f1": round(f1,4), "avg_precision": round(ap,4), "seconds": round(elapsed,1)})

    # --- TabNet ---
    for seed in SEEDS:
        t0 = time.time()
        model = TabNetSimple()
        acc, f1, ap = train_torch(model, X, y, seed)
        elapsed = time.time() - t0
        print(f"  TabNet      seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
        all_results.append({"model": "TabNet", "seed": seed, "accuracy": round(acc,4),
                            "macro_f1": round(f1,4), "avg_precision": round(ap,4), "seconds": round(elapsed,1)})

    # --- FT-Transformer ---
    for seed in SEEDS:
        t0 = time.time()
        model = FTTransformer()
        acc, f1, ap = train_torch(model, X, y, seed)
        elapsed = time.time() - t0
        print(f"  FT-Trans    seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
        all_results.append({"model": "FT_Transformer", "seed": seed, "accuracy": round(acc,4),
                            "macro_f1": round(f1,4), "avg_precision": round(ap,4), "seconds": round(elapsed,1)})

    # --- SAINT ---
    for seed in SEEDS:
        t0 = time.time()
        model = SAINTSimple()
        acc, f1, ap = train_torch(model, X, y, seed)
        elapsed = time.time() - t0
        print(f"  SAINT       seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
        all_results.append({"model": "SAINT", "seed": seed, "accuracy": round(acc,4),
                            "macro_f1": round(f1,4), "avg_precision": round(ap,4), "seconds": round(elapsed,1)})

    # --- GRANDE ---
    for seed in SEEDS:
        try:
            t0 = time.time()
            acc, f1, ap, fit_time = run_grande(X, y, seed)
            elapsed = time.time() - t0
            print(f"  GRANDE      seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
            all_results.append({"model": "GRANDE", "seed": seed, "accuracy": round(acc,4),
                                "macro_f1": round(f1,4), "avg_precision": round(ap,4), "seconds": round(elapsed,1)})
        except Exception as e:
            print(f"  GRANDE      seed={seed}: 失败 - {e}")

    # --- TabICLv2 ---
    for seed in SEEDS:
        try:
            t0 = time.time()
            acc, f1, ap, fit_time = run_tabiclv2(X, y, seed)
            elapsed = time.time() - t0
            print(f"  TabICLv2    seed={seed}: acc={acc:.4f}  f1={f1:.4f}  AP={ap:.4f}  ({elapsed:.1f}s)")
            all_results.append({"model": "TabICLv2", "seed": seed, "accuracy": round(acc,4),
                                "macro_f1": round(f1,4), "avg_precision": round(ap,4), "seconds": round(elapsed,1)})
        except Exception as e:
            print(f"  TabICLv2    seed={seed}: 失败 - {e}")

    # 合并已有结果
    existing_path = OUT_DIR / "m2_decline_results.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        combined = pd.concat([existing, pd.DataFrame(all_results)], ignore_index=True)
    else:
        combined = pd.DataFrame(all_results)
    combined.to_csv(existing_path, index=False)

    summary = combined.groupby("model").agg(
        acc_mean=("accuracy","mean"), acc_std=("accuracy","std"),
        f1_mean=("macro_f1","mean"), f1_std=("macro_f1","std"),
        ap_mean=("avg_precision","mean"), ap_std=("avg_precision","std"),
    ).round(4).sort_values("acc_mean", ascending=False)

    print(f"\n=== 完整 {len(summary)} 模型汇总（按 accuracy 排序）===")
    print(summary.to_string())
    print(f"\n写入 {existing_path}")


if __name__ == "__main__":
    sys.exit(main())
