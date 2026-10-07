#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_backfill_test_predictions.py — 补跑缺失的 test_predictions.csv

补跑 SupCon / RBF_SVM / Logistic / RandomForest 四个模型的逐样本预测，
保存为标准格式 test_predictions.csv（seed, student_id, pred_class, prob_0...prob_8）。

使用标准57维特征 + 标准 split（seed=91），3 seeds（42, 43, 44）。
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"
OUT_BASE = ROOT / "experiments" / "review_20260910" / "hi9_12models_bonus" / "outputs"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_CLASSES = 9


# ---------------------------------------------------------------------------
# SupCon 模型
# ---------------------------------------------------------------------------
class SupConEncoder(nn.Module):
    def __init__(self, n_features=57, proj_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, proj_dim),
        )

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
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(z.size(0), device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    labels = torch.cat([torch.arange(z1.size(0), device=z.device) + z1.size(0),
                        torch.arange(z1.size(0), device=z.device)])
    return F.cross_entropy(sim, labels)


def train_supcon(X_train, y_train, X_val, y_val, X_test, seed, epochs_pretrain=30, epochs_finetune=30):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.from_numpy(X_train).to(DEVICE)
    ytr = torch.from_numpy(y_train).to(DEVICE)
    Xva = torch.from_numpy(X_val).to(DEVICE)
    yva = torch.from_numpy(y_val).to(DEVICE)
    Xte = torch.from_numpy(X_test).to(DEVICE)

    # 预训练对比学习
    encoder = SupConEncoder().to(DEVICE)
    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 256
    n = len(Xtr)
    for epoch in range(epochs_pretrain):
        encoder.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            idx = perm[s:e]
            xb = Xtr[idx]
            z1, _ = encoder(xb + torch.randn_like(xb) * 0.1)
            z2, _ = encoder(xb + torch.randn_like(xb) * 0.1)
            loss = supcon_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 微调分类头
    clf = SupConClassifier(encoder).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(ytr, minlength=N_CLASSES).float()
    cw = (1.0 / counts.clamp(min=1)).sqrt()
    cw = cw / cw.sum() * N_CLASSES
    cw = cw.to(DEVICE)

    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs_finetune):
        clf.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            idx = perm[s:e]
            logits = clf(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            va_pred = clf(Xva).argmax(1).cpu().numpy()
            va_acc = accuracy_score(y_val, va_pred)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

    clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        te_logits = clf(Xte)
        te_pred = te_logits.argmax(1).cpu().numpy()
        te_proba = F.softmax(te_logits, dim=1).cpu().numpy()
    return te_pred, te_proba


# ---------------------------------------------------------------------------
# sklearn 模型
# ---------------------------------------------------------------------------
def train_sklearn(model_cls, model_params, X_train, y_train, X_test, seed):
    model = model_cls(random_state=seed, **model_params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        # 确保所有类别都有概率列
        if proba.shape[1] < N_CLASSES:
            full_proba = np.zeros((len(X_test), N_CLASSES))
            for i, c in enumerate(model.classes_):
                full_proba[:, c] = proba[:, i]
            proba = full_proba
    else:
        # SVM 默认没有 predict_proba，用 decision_function 转概率
        decision = model.decision_function(X_test)
        if decision.ndim == 1:
            # 二分类
            proba = np.zeros((len(X_test), N_CLASSES))
            proba[:, 1] = 1 / (1 + np.exp(-decision))
            proba[:, 0] = 1 - proba[:, 1]
        else:
            proba = np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)
    return pred, proba


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
MODELS = {
    "SupCon": {"type": "torch", "fn": train_supcon},
    "RBF_SVM": {"type": "sklearn", "cls": SVC, "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": "balanced"}},
    "Logistic": {"type": "sklearn", "cls": LogisticRegression, "params": {"max_iter": 1000, "class_weight": "balanced"}},
    "RandomForest": {"type": "sklearn", "cls": RandomForestClassifier, "params": {"n_estimators": 200, "class_weight": "balanced", "n_jobs": -1}},
}


def main():
    sys.path.insert(0, str(HI9BENCH_DIR))
    from hi9bench.data import load_bundle_legacy_unverified

    print(f"[1/3] 加载标准57维特征 ...")
    bundle = load_bundle_legacy_unverified(RAW_CSV, LABELS_CSV, split_seed=91)
    print(f"  train={len(bundle.y_train)}, val={len(bundle.y_val)}, test={len(bundle.y_test)}")

    # 标准化（用训练集统计量）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(bundle.X_train)
    X_val = scaler.transform(bundle.X_val)
    X_test = scaler.transform(bundle.X_test)
    y_train = bundle.y_train
    y_val = bundle.y_val
    y_test = bundle.y_test
    ids_test = bundle.ids_test

    print(f"\n[2/3] 训练 4 个模型 × {len(SEEDS)} seeds ...")

    for model_name, model_config in MODELS.items():
        print(f"\n  === {model_name} ===")
        out_dir = OUT_BASE / model_name.lower()
        out_dir.mkdir(parents=True, exist_ok=True)

        all_preds = []
        for seed in SEEDS:
            t0 = time.time()
            if model_config["type"] == "torch":
                pred, proba = model_config["fn"](X_train, y_train, X_val, y_val, X_test, seed)
            else:
                pred, proba = train_sklearn(
                    model_config["cls"], model_config["params"],
                    X_train, y_train, X_test, seed
                )
            elapsed = time.time() - t0
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="macro")
            print(f"    seed={seed}: acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)")

            # 保存逐样本预测
            for i in range(len(pred)):
                row = {
                    "seed": seed,
                    "student_id": ids_test[i],
                    "pred_class": int(pred[i]),
                }
                for c in range(N_CLASSES):
                    row[f"prob_{c}"] = float(proba[i, c])
                all_preds.append(row)

        pred_df = pd.DataFrame(all_preds)
        pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

        # 写 status.json
        accs = [accuracy_score(y_test, pred_df[pred_df["seed"] == s]["pred_class"].to_numpy()) for s in SEEDS]
        f1s = [f1_score(y_test, pred_df[pred_df["seed"] == s]["pred_class"].to_numpy(), average="macro") for s in SEEDS]
        status = {
            "model": model_name,
            "n_seeds": len(SEEDS),
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_sd": round(float(np.std(accs)), 4),
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "macro_f1_sd": round(float(np.std(f1s)), 4),
            "per_seed": [
                {"seed": s, "accuracy": round(a, 4), "macro_f1": round(f, 4)}
                for s, a, f in zip(SEEDS, accs, f1s)
            ],
            "features": "57-D (28 z-score + 28 mask + gender)",
        }
        with open(out_dir / f"{model_name}_status.json", "w") as f:
            import json
            json.dump(status, f, indent=2, ensure_ascii=False)

        print(f"    写入 {out_dir / 'test_predictions.csv'}")
        print(f"    写入 {out_dir / f'{model_name}_status.json'}")

    print(f"\n[3/3] 全部完成 ✅")


if __name__ == "__main__":
    sys.exit(main())
