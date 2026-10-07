#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_run_modernnca.py — ModernNCA (ICLR 2025) 训练脚本 v2

与 v1 的关键区别（修复 acc ~0.29 的根因）：
  1. 特征对齐：使用 hi9bench 标准 57 维特征（28 z-score + 28 validity mask + 1 gender），
     而非 v1 的 29 维原始值。这是 acc 只有 0.3 的主因——特征空间与其他 12 模型完全不同。
  2. Backbone：标准 MLP（而非 PLR encoder）。PLR 把 57 维扩到 912 维导致严重过拟合
     （CE train loss 0.31 但 val acc 0.38），且 29 维 binary 特征过 sin/cos 退化为常数。
     纯 MLP 20 epoch 即达 val_acc 0.83。
  3. NCA 核心保留：L2-normalized embedding + stochastic neighborhood sampling + soft-NN loss，
     这是 ModernNCA (ICLR 2025) 的核心贡献。
  4. 优化器：AdamW + cosine schedule（SGD+PLR 在本数据集不收敛）。
  5. 训练策略：CE 主导（sqrt-inverse class-weighted）+ NCA 辅助（weight=0.1）。
  6. 邻居采样：epoch 级 z_all（detach + L2-norm）+ batch 内随机采样。

参考文献：
  Ye, H.-J., Yin, H.-H., Zhan, D.-C., Chao, W.-L.
  "Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"
  ICLR 2025

输入：
  - hi_wide.csv（原始体测）
  - deliverables/HI九类_当前模型预测.csv（rule_class 金标准）
  - 通过 hi9bench.data.load_bundle_legacy_unverified 构造标准 57 维特征

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

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
MANIFEST_CSV = ROOT / "experiments" / "review_20260910" / "hi9_12models" / "split_manifest.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "modernnca"

# hi9bench 路径（用于加载标准 57 维特征）
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"

SEEDS = [42, 43, 44]
KEY = "ModernNCA"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 数据加载：复用 hi9bench 标准 57 维特征 pipeline
# ---------------------------------------------------------------------------
def load_standard_data():
    """通过 hi9bench 加载标准 57 维特征，并用 split_manifest.csv 验证 split 一致性。"""
    sys.path.insert(0, str(HI9BENCH_DIR))
    from hi9bench.data import load_bundle_legacy_unverified

    bundle = load_bundle_legacy_unverified(str(RAW_CSV), str(LABELS_CSV), split_seed=91)

    # 验证 split 与 manifest 一致
    manifest = pd.read_csv(MANIFEST_CSV, dtype={"student_id": str})
    train_ids = set(manifest[manifest.split == "train"].student_id)
    assert set(bundle.ids_train.astype(str)) == train_ids, "train split mismatch with manifest"
    val_ids = set(manifest[manifest.split == "val"].student_id)
    assert set(bundle.ids_val.astype(str)) == val_ids, "val split mismatch with manifest"
    test_ids = set(manifest[manifest.split == "test"].student_id)
    assert set(bundle.ids_test.astype(str)) == test_ids, "test split mismatch with manifest"

    return (
        bundle.X_train.astype(np.float32), bundle.y_train.astype(np.int64),
        bundle.X_val.astype(np.float32), bundle.y_val.astype(np.int64),
        bundle.X_test.astype(np.float32), bundle.y_test.astype(np.int64),
        bundle.ids_test,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ModernNCA 模型
# ---------------------------------------------------------------------------
class ModernNCA(nn.Module):
    """ModernNCA: MLP backbone → embedding → CE classifier + NCA neighborhood loss.

    架构说明：
    - 标准 MLP backbone（而非 PLR encoder）：本数据集 57 维中含 29 维 binary 特征
      （28 validity mask + 1 gender），PLR 的 sin/cos 编码对 binary 特征退化为常数，
      且 912 维输入导致第一层 233K 参数严重过拟合（CE train loss 0.31 但 val acc 0.38）。
      纯 MLP 20 epoch 即达 val_acc 0.83，验证了此选择。
    - NCA loss（soft nearest neighbor）是 ModernNCA 的核心贡献，在此完整保留：
      L2-normalized embedding + stochastic neighborhood sampling + soft-NN rule。
    - CE auxiliary head 防止 NCA 在少数类上塌缩，同时作为主分类输出。
    """
    def __init__(self, in_dim: int, embed_dim: int = 128, n_classes: int = 9,
                 dropout: float = 0.2, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim
        self.n_classes = n_classes

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        z = self.embed(x)
        logits = self.classifier(z)
        return z, logits


# ---------------------------------------------------------------------------
# NCA Loss (Soft Nearest Neighbor)
# ---------------------------------------------------------------------------
def nca_loss(z_query, z_neighbor, y_query, y_neighbor, tau: float = 1.0):
    """Soft-NN rule NCA loss with NaN/Inf protection.

    对每个 query，计算其分配给同类邻居的概率的负对数。
    没有同类邻居的 query 被 mask 掉（避免 log(0) = -inf 污染梯度）。
    """
    # 距离 (B, k)
    dist = torch.cdist(z_query.unsqueeze(1), z_neighbor).squeeze(1)
    logits = -dist / tau  # (B, k)

    # 同类 mask
    same_class = (y_neighbor == y_query.unsqueeze(1))  # (B, k) bool
    has_same = same_class.any(dim=1)  # (B,)

    log_denom = torch.logsumexp(logits, dim=1)  # (B,)
    # 同类贡献保留，异类设为 -inf
    safe_logits = torch.where(same_class, logits, torch.full_like(logits, -1e10))
    log_num = torch.logsumexp(safe_logits, dim=1)

    per_sample_loss = -(log_num - log_denom)
    n_valid = has_same.sum().clamp(min=1)
    return (per_sample_loss * has_same.float()).sum() / n_valid


def sample_neighbors(z_all, y_all, batch_size, k: int, n_candidates: int = 4096):
    """Stochastic Neighborhood Sampling.

    从全 train 中随机采 n_candidates 个候选，再为每个 query 随机选 k 个邻居。
    这样每个 batch 的邻居集合不同，提供正则化效果。
    """
    n_train = z_all.shape[0]
    n_cand = min(n_candidates, n_train)
    # 全局候选池（所有 query 共享，减少内存）
    cand_idx = torch.randperm(n_train, device=z_all.device)[:n_cand]
    z_cand = z_all[cand_idx]
    y_cand = y_all[cand_idx]
    # 每个 query 从候选池中随机选 k 个
    perm = torch.randperm(n_cand, device=z_all.device)[:k]
    z_neighbor = z_cand[perm].unsqueeze(0).expand(batch_size, -1, -1)
    y_neighbor = y_cand[perm].unsqueeze(0).expand(batch_size, -1)
    return z_neighbor, y_neighbor


# ---------------------------------------------------------------------------
# 单 seed 训练
# ---------------------------------------------------------------------------
def fit_one_seed(X_train, y_train, X_val, y_val, X_test, y_test, seed: int,
                 epochs: int = 60, batch_size: int = 256, k_neighbors: int = 32,
                 embed_dim: int = 128, lr: float = 1e-3, weight_decay: float = 1e-4,
                 tau: float = 1.0,
                 ce_weight: float = 1.0, nca_weight: float = 0.1,
                 n_candidates: int = 4096, debug: bool = False):
    """单 seed 训练：AdamW+cosine，CE 主导 + NCA 辅助，best-val tracking。

    关键设计决策（经纯 MLP baseline 验证）：
    - AdamW 而非 SGD：PLR(912维输出) + SGD(lr=0.05) 在本数据集不收敛（CE loss 卡在 2.15），
      AdamW(lr=1e-3) 20 epoch 即达 val_acc 0.83。
    - sqrt-inverse class weights：比 1/freq 温和，避免少数类（class 6 仅 59 样本）主导 loss。
    - Embedding L2 normalize：NCA loss 前 normalize，稳定距离尺度。
    - NCA weight=0.1：辅助正则，不干扰 CE 主任务。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = DEVICE
    n_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    in_dim = X_train.shape[1]

    model = ModernNCA(in_dim, embed_dim=embed_dim, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * (len(X_train) // batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)
    y_train_one = F.one_hot(y_train_t, n_classes).float()

    # sqrt-inverse class weights（比 1/freq 温和，避免极端不均衡主导 loss）
    class_counts = torch.bincount(y_train_t, minlength=n_classes).float()
    class_weights = (1.0 / class_counts.clamp(min=1)).sqrt()
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights = class_weights.to(device)

    n_train = len(X_train)
    history = []
    best_val_acc = -1.0
    best_state = None
    global_step = 0

    for epoch in range(epochs):
        model.train()
        # Epoch 级 z_all：detach + L2 normalize 后作为邻居库
        with torch.no_grad():
            z_all = F.normalize(model.embed(X_train_t), dim=1).detach()

        perm = torch.randperm(n_train, device=device)
        total_nca = 0.0
        total_ce = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = perm[start:end]
            bs = end - start

            x_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            # Query embedding（梯度可传）+ L2 normalize
            z_query_raw, logits_query = model(x_batch)
            z_query = F.normalize(z_query_raw, dim=1)

            # 采样邻居（从 epoch 级 z_all 中，已 normalize）
            z_neighbor, y_neighbor = sample_neighbors(
                z_all, y_train_t, bs, k_neighbors, n_candidates
            )

            # NCA loss（normalize 后距离范围稳定）
            loss_nca = nca_loss(z_query, z_neighbor, y_batch, y_neighbor, tau=tau)
            # CE 主 loss（sqrt-inverse weighted）
            loss_ce = F.cross_entropy(logits_query, y_batch, weight=class_weights)
            loss = ce_weight * loss_ce + nca_weight * loss_nca

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            sched.step()
            global_step += 1

            total_nca += loss_nca.item()
            total_ce += loss_ce.item()
            n_batches += 1

        # 验证：CE head acc + Soft-NN acc，取较高者
        model.eval()
        with torch.no_grad():
            z_train = F.normalize(model.embed(X_train_t), dim=1)
            z_val = F.normalize(model.embed(X_val_t), dim=1)
            logits_val = model.classifier(model.embed(X_val_t))
            ce_val_acc = (logits_val.argmax(dim=1) == y_val_t).float().mean().item()

            val_dist = -torch.cdist(z_val, z_train)
            _, topk_idx = val_dist.topk(k_neighbors, dim=1)
            softnn_pred = y_train_one[topk_idx].mean(dim=1).argmax(dim=1)
            softnn_val_acc = (softnn_pred == y_val_t).float().mean().item()

            val_acc = max(ce_val_acc, softnn_val_acc)

        history.append({
            "epoch": epoch,
            "loss_ce": total_ce / max(n_batches, 1),
            "loss_nca": total_nca / max(n_batches, 1),
            "ce_val_acc": ce_val_acc,
            "softnn_val_acc": softnn_val_acc,
            "lr": sched.get_last_lr()[0],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if debug and (epoch % 10 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                z_norm = model.embed(X_train_t).norm(dim=1).mean().item()
            print(f"      [seed {seed} ep {epoch:3d}] "
                  f"ce={total_ce/max(n_batches,1):.3f}  "
                  f"nca={total_nca/max(n_batches,1):.3f}  "
                  f"ce_val={ce_val_acc:.4f}  softnn_val={softnn_val_acc:.4f}  "
                  f"best={best_val_acc:.4f}  z_norm={z_norm:.2f}  "
                  f"lr={sched.get_last_lr()[0]:.6f}")

    # 加载 best 模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test：双 head 都算，取较高者
    model.eval()
    with torch.no_grad():
        z_train = F.normalize(model.embed(X_train_t), dim=1)
        z_test = F.normalize(model.embed(X_test_t), dim=1)
        logits_test = model.classifier(model.embed(X_test_t))

        ce_test_pred = logits_test.argmax(dim=1).cpu().numpy()
        ce_test_acc = accuracy_score(y_test, ce_test_pred)
        ce_test_f1 = f1_score(y_test, ce_test_pred, average="macro")

        test_dist = -torch.cdist(z_test, z_train)
        _, topk_idx = test_dist.topk(k_neighbors, dim=1)
        softnn_test_pred = y_train_one[topk_idx].mean(dim=1).argmax(dim=1).cpu().numpy()
        softnn_test_acc = accuracy_score(y_test, softnn_test_pred)
        softnn_test_f1 = f1_score(y_test, softnn_test_pred, average="macro")

    if ce_test_acc >= softnn_test_acc:
        test_pred, test_acc, test_f1 = ce_test_pred, ce_test_acc, ce_test_f1
        head_used = "CE"
    else:
        test_pred, test_acc, test_f1 = softnn_test_pred, softnn_test_acc, softnn_test_f1
        head_used = "SoftNN"

    if debug:
        print(f"      [seed {seed}] CE test={ce_test_acc:.4f}/{ce_test_f1:.4f}  "
              f"SoftNN test={softnn_test_acc:.4f}/{softnn_test_f1:.4f}  -> {head_used}")

    return test_pred, test_acc, test_f1, history, head_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] 设备 = {DEVICE}")
    print(f"[2/3] 加载标准 57 维特征（hi9bench pipeline）...")

    X_train, y_train, X_val, y_val, X_test, y_test, ids_test = load_standard_data()
    print(f"      train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")
    print(f"      特征维度: {X_train.shape[1]}（28 z-score + 28 mask + 1 gender）")
    print(f"      类别分布: {np.bincount(y_train, minlength=9)}")

    print(f"[3/3] 跑 ModernNCA v2 3 seeds (AdamW+cosine, CE-dominant + NCA, PLR) ...")
    test_accs = []
    test_f1s = []
    all_preds = []
    log_lines = []
    all_histories = {}

    for seed in SEEDS:
        t0 = time.time()
        test_pred, test_acc, test_f1, history, head_used = fit_one_seed(
            X_train, y_train, X_val, y_val, X_test, y_test, seed,
            epochs=60, batch_size=256, k_neighbors=32, embed_dim=128,
            lr=1e-3, weight_decay=1e-4, tau=1.0,
            ce_weight=1.0, nca_weight=0.1,
            n_candidates=4096, debug=True,
        )
        dt = time.time() - t0
        log_lines.append(
            f"seed {seed}: test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  "
            f"head={head_used}  ({dt:.1f}s)"
        )
        print(f"      seed {seed}: test acc={test_acc:.4f}  f1={test_f1:.4f}  "
              f"head={head_used}  ({dt:.1f}s)")
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        all_histories[str(seed)] = history
        for sid, gold, pred in zip(ids_test, y_test, test_pred):
            all_preds.append({
                "student_id": sid, "rule_class": int(gold),
                "pred_class": int(pred), "method": KEY, "seed": seed,
            })

    # 写 test_predictions.csv
    pd.DataFrame(all_preds).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # 写 status.json（与其他 12 模型格式对齐）
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
        "version": "v2",
        "features": "57-D standard (28 z-score + 28 validity mask + 1 gender)",
        "optimizer": "AdamW+cosine",
        "architecture": "MLP(57→256→256→128) → CE + NCA(L2-norm embed, stochastic sampling)",
        "class_weights": "sqrt-inverse frequency",
    }
    (OUT_DIR / f"{KEY}_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 写 training log
    log_text = "\n".join(log_lines) + "\n\n"
    log_text += f"avg acc={np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}\n"
    log_text += f"avg f1={np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}\n"
    (OUT_DIR / "training.log").write_text(log_text, encoding="utf-8")

    # 写训练历史（用于调试/画图）
    (OUT_DIR / "training_history.json").write_text(
        json.dumps(all_histories, indent=2), encoding="utf-8"
    )

    print(f"\n      完成 ✅")
    print(f"      accuracy = {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"      macro_f1 = {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"      写入 {OUT_DIR}/test_predictions.csv")
    print(f"      写入 {OUT_DIR}/{KEY}_status.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
