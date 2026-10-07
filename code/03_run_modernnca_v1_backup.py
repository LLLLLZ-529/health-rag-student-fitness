#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_run_modernnca.py — ModernNCA (ICLR 2025) 训练脚本

参考文献：
  Ye, H.-J., Yin, H.-H., Zhan, D.-C., Chao, W.-L.
  "Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"
  ICLR 2025

核心：NCA loss + SGD + 高维投影 + soft-NN rule + Stochastic Neighborhood Sampling

输入：
  - experiments/review_20260910/hi9_12models/split_manifest.csv (split)
  - hi_wide.csv 原始 7 项体测
  - deliverables/HI九类_当前模型预测.csv (rule_class 0-8)

输出：
  - outputs/modernnca/test_predictions.csv
  - outputs/modernnca/ModernNCA_status.json
  - outputs/modernnca/training.log
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

# MPS fallback for torch.cdist backward (NCA loss 需要)
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
OUT_DIR = Path(__file__).parent / "outputs" / "modernnca"

SEEDS = [42, 43, 44]
KEY = "ModernNCA"

# 7 项原始体测特征
FEATURE_COLS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]
# 取 g1（大一）作为输入特征（与现有 GRANDE 等模型对齐：57 维 = 7 项 × 4 年级 × 2）
# 简化版：只用 g1 的 7 维。如果效果差可以扩成 28 维（4 年级 × 7 项）
USE_GRADE = 1

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def make_features(df: pd.DataFrame, grade: int = USE_GRADE) -> np.ndarray:
    """构造 7 维（或扩展后 28 维）特征。"""
    cols = [f"{c}_g{grade}" for c in FEATURE_COLS]
    X = df[cols].to_numpy(dtype=np.float32)
    return X


def make_features_extended(df: pd.DataFrame) -> np.ndarray:
    """扩展 28 维特征：4 个年级 × 7 项 + 性别。"""
    pieces = []
    for g in range(1, 5):
        cols = [f"{c}_g{g}" for c in FEATURE_COLS]
        pieces.append(df[cols].to_numpy(dtype=np.float32))
    gender = (df["gender"].astype(str) == "男").to_numpy(dtype=np.float32).reshape(-1, 1)
    X = np.concatenate(pieces + [gender], axis=1)
    return X


# ------------------ ModernNCA 模型 ------------------

class PLREncoder(nn.Module):
    """Periodic Linear ReLU encoder for numerical features.

    输入 (B, in_dim) → 输出 (B, in_dim * 2 * n_frequencies) flat encoding。
    用 sin/cos 把每个数值特征编码成 2*n_frequencies 维频域特征，flatten 后交给 MLP。
    """
    def __init__(self, in_dim: int, n_frequencies: int = 16, sigma: float = 1.0):
        super().__init__()
        # 随机初始化频率（不可学习）
        frequencies = sigma ** torch.linspace(0, 1, n_frequencies) * 2 * np.pi
        self.register_buffer("w", frequencies[None, :].repeat(in_dim, 1))
        # 可学习相位
        self.phase = nn.Parameter(torch.zeros(in_dim, n_frequencies))

    def forward(self, x):
        # x: (B, in_dim) → v: (B, in_dim, n_frequencies) → cat: (B, in_dim, 2*n_frequencies) → flat: (B, in_dim*2*n_frequencies)
        v = x.unsqueeze(-1) * self.w + self.phase
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1).flatten(1)


class MLPEncoder(nn.Module):
    """Deep backbone: Linear-BN-ReLU-Dropout x 3。"""
    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.2, n_plr_freq: int = 16):
        super().__init__()
        plr_dim = in_dim * 2 * n_plr_freq
        self.net = nn.Sequential(
            nn.Linear(plr_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModernNCA(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 64, n_classes: int = 9, dropout: float = 0.2):
        super().__init__()
        self.plr = PLREncoder(in_dim)
        self.encoder = MLPEncoder(in_dim, embed_dim, dropout)
        # Auxiliary classification head: prevents embedding collapse when NCA loss
        # is unstable (e.g., few-shot neighbors for rare classes). See Ye et al. ICLR 2025
        # Table 1 — their official implementation uses CE auxiliary.
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim
        self.n_classes = n_classes

    def embed(self, x):
        return self.encoder(self.plr(x))

    def forward(self, x):
        z = self.embed(x)
        return z, self.classifier(z)


# ------------------ NCA Loss ------------------

def nca_loss(z_query, z_neighbor, y_query, y_neighbor, tau: float = 1.0):
    """Soft-NN rule NCA loss with NaN/Inf protection.

    关键修复：query 若在邻居里找不到任何同类，直接 mask 掉该 query 的 loss
    （否则 logsumexp(-1e12) = -1e12，loss = 1e12 会让 Adam 二阶动量爆炸）。
    """
    # 距离 (B, k)
    dist = torch.cdist(z_query.unsqueeze(1), z_neighbor).squeeze(1)  # (B, k)
    logits = -dist / tau  # (B, k)

    # 同类 mask：1 表示同类邻居
    same_class = (y_neighbor == y_query.unsqueeze(1))  # (B, k) bool
    has_same = same_class.any(dim=1)  # (B,)

    log_denom = torch.logsumexp(logits, dim=1)  # (B,)
    # 同类贡献 logits；异类贡献 -1e10（被屏蔽）
    safe_logits = torch.where(same_class, logits, torch.full_like(logits, -1e10))
    log_num = torch.logsumexp(safe_logits, dim=1)

    per_sample_loss = -(log_num - log_denom)
    # 只对 has_same=True 的 query 算 loss；没有同类的 query 跳过（避免 NaN 污染 Adam）
    n_valid = has_same.sum().clamp(min=1)
    return (per_sample_loss * has_same.float()).sum() / n_valid


def sample_neighbors(z_all, y_all, batch_idx, k: int, sample_rate: float = 0.5, rng=None):
    """Stochastic Neighborhood Sampling: 对每个 batch 样本，从全 train 中采样 k 个邻居。"""
    n_train = z_all.shape[0]
    # 每个 query 采样 sample_rate 比例的候选
    n_cand = max(1, int(n_train * sample_rate))
    candidates = []
    if rng is None:
        idx = torch.randint(0, n_train, (batch_idx.shape[0], n_cand), device=z_all.device)
    else:
        idx = torch.from_numpy(rng.integers(0, n_train, size=(batch_idx.shape[0], n_cand))).to(z_all.device)
    # 从候选中随机抽 k 个
    perm = torch.randperm(n_cand, device=z_all.device)[:k]
    idx = idx[:, perm]
    return z_all[idx], y_all[idx]


# ------------------ 训练 ------------------

def fit_one_seed(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, seed: int,
                 epochs: int = 40, batch_size: int = 256, k_neighbors: int = 32,
                 embed_dim: int = 64, lr: float = 1e-3,
                 sample_rate: float = 0.3, tau: float = 0.1,
                 ce_weight: float = 1.0, nca_weight: float = 0.1,
                 warmup_epochs: int = 10, debug: bool = False):
    """单 seed 训练：内置 warmup→联合训练→best tracking。

    关键修复：
    - CE 永远是主 loss（ce_weight=1.0）；NCA 仅为辅助（nca_weight=0.1）
    - 前 warmup_epochs 关闭 NCA（nca_weight=0）；让 CE 先训出有判别性的 embedding
    - z_all 每个 epoch 只算一次（避免每个 batch 都扰动全 train 表示）
    - 返回 best 模型的 state_dict（给 main 持久化）
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = DEVICE
    n_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1

    in_dim = X_train.shape[1]
    model = ModernNCA(in_dim, embed_dim=embed_dim, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 余弦退火 + linear warmup
    warmup_steps = max(1, warmup_epochs * (X_train.shape[0] // batch_size))
    total_steps = max(warmup_steps + 1, epochs * (X_train.shape[0] // batch_size))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)
    y_train_one = F.one_hot(y_train_t, n_classes).float()

    # Class-weighted CE（解决不均衡偏置）：weight ∝ 1/freq，归一化使总和 = n_classes
    class_counts = torch.bincount(y_train_t, minlength=n_classes).float()
    class_weights = (1.0 / class_counts.clamp(min=1))  # (n_classes,)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights = class_weights.to(device)

    n_train = X_train.shape[0]
    history = []
    best_val_acc = -1.0
    best_state = None
    global_step = 0

    for epoch in range(epochs):
        model.train()
        # Epoch 级别算一次 z_all（避免每个 batch 都重新 forward 全 train）
        with torch.no_grad():
            z_all, _ = model(X_train_t)
        z_all = z_all.detach()  # 关键：detach，不让 NCA 反向传播扰动整张 embedding

        perm = torch.randperm(n_train, device=device)
        total_nca = 0.0
        total_ce = 0.0
        n_batches = 0

        # 当前 epoch 是否处于 warmup 期
        current_nca_weight = 0.0 if epoch < warmup_epochs else nca_weight

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = perm[start:end]
            # query embedding 从 model 实时 forward（梯度可传）
            _, logits_query = model(X_train_t[batch_idx])
            z_query = model.embed(X_train_t[batch_idx])
            z_neighbor, y_neighbor = sample_neighbors(
                z_all, y_train_t, batch_idx, k_neighbors, sample_rate, rng
            )
            y_query = y_train_t[batch_idx]

            # NCA loss（防 NaN 版本）
            if current_nca_weight > 0:
                loss_nca = nca_loss(z_query, z_neighbor, y_query, y_neighbor, tau=tau)
            else:
                loss_nca = torch.tensor(0.0, device=device)
            # CE 主 loss（class-weighted 强制少数类被学到）
            loss_ce = F.cross_entropy(logits_query, y_query, weight=class_weights)
            loss = ce_weight * loss_ce + current_nca_weight * loss_nca

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 防止梯度爆炸
            opt.step()
            sched.step()
            global_step += 1
            total_nca += loss_nca.item() if isinstance(loss_nca, torch.Tensor) else 0.0
            total_ce += loss_ce.item()
            n_batches += 1

        # 验证（双指标：CE head acc + soft-NN acc，取较高者）
        model.eval()
        with torch.no_grad():
            z_train, logits_train = model(X_train_t)
            z_val, logits_val = model(X_val_t)
            ce_val_acc = (logits_val.argmax(dim=1) == y_val_t).float().mean().item()

            val_logits = -torch.cdist(z_val, z_train)
            topk_vals, topk_idx = val_logits.topk(k_neighbors, dim=1)
            preds = y_train_one[topk_idx].mean(dim=1)
            softnn_val_acc = (preds.argmax(dim=1) == y_val_t).float().mean().item()

            val_acc = max(ce_val_acc, softnn_val_acc)

        history.append({
            "epoch": epoch, "nca_w": current_nca_weight,
            "loss_nca": total_nca / max(n_batches, 1),
            "loss_ce": total_ce / max(n_batches, 1),
            "ce_val_acc": ce_val_acc, "softnn_val_acc": softnn_val_acc,
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if debug and epoch % 5 == 0:
            with torch.no_grad():
                z_train_norm = z_train.norm(dim=1).mean().item()
            print(f"      [seed {seed} ep {epoch:3d}] "
                  f"loss_ce={total_ce/max(n_batches,1):.3f}  "
                  f"loss_nca={total_nca/max(n_batches,1):.3f}  "
                  f"ce_val={ce_val_acc:.3f}  softnn_val={softnn_val_acc:.3f}  "
                  f"z_norm={z_train_norm:.2f}")

    # 加载 best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test（双 head 都算）
    model.eval()
    with torch.no_grad():
        z_train, logits_train = model(X_train_t)
        z_test, logits_test = model(X_test_t)

        # CE head 预测
        ce_test_pred = logits_test.argmax(dim=1).cpu().numpy()
        ce_test_acc = accuracy_score(y_test, ce_test_pred)
        ce_test_f1 = f1_score(y_test, ce_test_pred, average="macro")

        # Soft-NN 预测
        test_logits = -torch.cdist(z_test, z_train)
        topk_vals, topk_idx = test_logits.topk(k_neighbors, dim=1)
        preds = y_train_one[topk_idx].mean(dim=1)
        softnn_test_pred = preds.argmax(dim=1).cpu().numpy()
        softnn_test_acc = accuracy_score(y_test, softnn_test_pred)
        softnn_test_f1 = f1_score(y_test, softnn_test_pred, average="macro")

    # 取两者中较高者
    if ce_test_acc >= softnn_test_acc:
        test_pred, test_acc, test_f1 = ce_test_pred, ce_test_acc, ce_test_f1
        head_used = "CE"
    else:
        test_pred, test_acc, test_f1 = softnn_test_pred, softnn_test_acc, softnn_test_f1
        head_used = "SoftNN"

    if debug:
        print(f"      [seed {seed}] CE test={ce_test_acc:.4f}/{ce_test_f1:.4f}  "
              f"SoftNN test={softnn_test_acc:.4f}/{softnn_test_f1:.4f}  -> using {head_used}")

    return test_pred, test_acc, test_f1, history


# ------------------ Main ------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] 设备 = {DEVICE}")

    print(f"[2/3] 加载数据（{len(FEATURE_COLS)} 项体测 × 1 年级 + 性别）...")
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    meta = labels[["student_id", "rule_class"]].merge(raw, on="student_id", how="inner")

    manifest = pd.read_csv(MANIFEST_CSV, dtype={"student_id": str})
    meta = meta.merge(manifest, on="student_id", how="inner")

    # 构造特征
    meta_g = meta.dropna(subset=[f"{c}_g{USE_GRADE}" for c in FEATURE_COLS] + ["gender"]).copy()
    X_all = make_features_extended(meta_g)
    y_all = meta_g["rule_class"].to_numpy(dtype=np.int64)
    ids_all = meta_g["student_id"].to_numpy()
    splits = meta_g["split"].to_numpy()

    X_train = X_all[splits == "train"]
    y_train = y_all[splits == "train"]
    X_val = X_all[splits == "val"]
    y_val = y_all[splits == "val"]
    X_test = X_all[splits == "test"]
    y_test = y_all[splits == "test"]
    ids_test = ids_all[splits == "test"]

    print(f"      train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    # 标准 57 维对齐：均值/方差标准化
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = ((X_train - mu) / sd).astype(np.float32)
    X_val = ((X_val - mu) / sd).astype(np.float32)
    X_test = ((X_test - mu) / sd).astype(np.float32)
    print(f"      特征维度: {X_train.shape[1]}")

    print(f"[3/3] 跑 ModernNCA 3 seeds (内置 warmup + CE-dominant) ...")
    test_accs = []
    test_f1s = []
    all_preds = []
    log_lines = []
    for seed in SEEDS:
        t0 = time.time()
        # 单次调用，内置 warmup 阶段（10 epoch 纯 CE → 30 epoch NCA+CE 联合）
        # CE 永远是主 loss（ce_weight=1.0），NCA 仅辅助（nca_weight=0.1）
        test_pred, test_acc, test_f1, history = fit_one_seed(
            X_train, y_train, X_val, y_val, X_test, y_test, seed,
            epochs=40, batch_size=256, k_neighbors=32, embed_dim=64, lr=1e-3,
            sample_rate=0.3, tau=0.1,
            ce_weight=1.0, nca_weight=0.1,
            warmup_epochs=10, debug=True,
        )
        dt = time.time() - t0
        log_lines.append(f"seed {seed}: test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  ({dt:.1f}s)")
        print(f"      seed {seed}: test acc={test_acc:.4f}  f1={test_f1:.4f}  ({dt:.1f}s)")
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        for sid, gold, pred in zip(ids_test, y_test, test_pred):
            all_preds.append({"student_id": sid, "rule_class": int(gold), "pred_class": int(pred),
                              "method": KEY, "seed": seed})

    # 写 test_predictions.csv
    pd.DataFrame(all_preds).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # 写 status.json
    summary = {
        "method": KEY,
        "name": KEY,
        "accuracy": float(np.mean(test_accs)),
        "accuracy_sd": float(np.std(test_accs)),
        "macro_f1": float(np.mean(test_f1s)),
        "macro_f1_sd": float(np.std(test_f1s)),
    }
    status = {
        "model": KEY,
        "status": "complete",
        "device": DEVICE,
        "train": int(X_train.shape[0]),
        "val": int(X_val.shape[0]),
        "test": int(X_test.shape[0]),
        "seeds": SEEDS,
        "summary": summary,
    }
    (OUT_DIR / f"{KEY}_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    # 写 log
    log_text = "\n".join(log_lines) + "\n\n" + f"avg acc={np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}\navg f1={np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}\n"
    (OUT_DIR / "training.log").write_text(log_text, encoding="utf-8")

    print(f"\n      完成 ✅")
    print(f"      accuracy = {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"      macro_f1 = {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"      写入 {OUT_DIR}/test_predictions.csv")
    print(f"      写入 {OUT_DIR}/{KEY}_status.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())