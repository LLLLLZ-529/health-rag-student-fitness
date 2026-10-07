#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_run_tabm.py — TabM (ICLR 2025 Oral) 训练脚本

TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling
  Gorishniy, Kotelnikov, Babenko — ICLR 2025 (Oral)

核心思想：
  一个模型高效模拟 k 个 MLP 的集成。通过 BatchEnsemble 实现权重共享：
  每个线性层有一个共享权重矩阵 W，加上 k 组 per-member 的输入缩放 r_i
  和输出缩放 s_i。k 个成员并行训练，最终预测取 k 个成员的平均。

架构（遵循官方 tabm 包的默认配置）：
  Input (B, 57)
    → EnsembleView (B, k=32, 57)          # 复制 k 份
    → LinearBatchEnsemble(57→256)          # 第一层 r~N(0,1) 创建多样性
    → ReLU → Dropout
    → LinearBatchEnsemble(256→256)         # 后续层 r=ones
    → ReLU → Dropout
    → LinearBatchEnsemble(256→256)
    → ReLU → Dropout
    → LinearEnsemble(256→9)               # k 个独立输出头
    → 平均 k 个预测 → (B, 9)

输入：
  - hi_wide.csv + deliverables/HI九类_当前模型预测.csv
  - 通过 hi9bench.data.load_bundle_legacy_unverified 构造标准 57 维特征

输出：
  - outputs/tabm/test_predictions.csv
  - outputs/tabm/TabM_status.json
  - outputs/tabm/training.log
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

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
OUT_DIR = Path(__file__).parent / "outputs" / "tabm"
HI9BENCH_DIR = ROOT / "HI9_12models_code_bundle_v2" / "hi9_extended_models"

SEEDS = [42, 43, 44]
KEY = "TabM"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 数据加载（与 ModernNCA 脚本一致）
# ---------------------------------------------------------------------------
def load_standard_data():
    """通过 hi9bench 加载标准 57 维特征，验证 split 与 manifest 一致。"""
    sys.path.insert(0, str(HI9BENCH_DIR))
    from hi9bench.data import load_bundle_legacy_unverified

    bundle = load_bundle_legacy_unverified(str(RAW_CSV), str(LABELS_CSV), split_seed=91)

    manifest = pd.read_csv(MANIFEST_CSV, dtype={"student_id": str})
    for split_name, ids_attr in [("train", "ids_train"), ("val", "ids_val"), ("test", "ids_test")]:
        expected = set(manifest[manifest.split == split_name].student_id)
        actual = set(getattr(bundle, ids_attr).astype(str))
        assert expected == actual, f"{split_name} split mismatch with manifest"

    return (
        bundle.X_train.astype(np.float32), bundle.y_train.astype(np.int64),
        bundle.X_val.astype(np.float32), bundle.y_val.astype(np.int64),
        bundle.X_test.astype(np.float32), bundle.y_test.astype(np.int64),
        bundle.ids_test,
    )


# ---------------------------------------------------------------------------
# TabM 核心层
# ---------------------------------------------------------------------------
class EnsembleView(nn.Module):
    """将 (B, D) 输入复制为 (B, k, D) 的 k 个视图（无额外参数）。"""
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) → (B, k, D)
        return x.unsqueeze(1).expand(-1, self.k, -1)


class LinearBatchEnsemble(nn.Module):
    """BatchEnsemble 线性层：共享权重 W + per-member 缩放 r, s + per-member bias。

    对成员 i：output_i = (input_i * r_i) @ W * s_i + b_i

    参数：
      W: (d_in, d_out)  共享权重
      r: (k, d_in)      per-member 输入缩放
      s: (k, d_out)     per-member 输出缩放
      b: (k, d_out)     per-member 偏置

    初始化遵循 TabM 官方：
      - 第一层：r ~ N(0,1), s = ones（创建 k 个不同的表示）
      - 后续层：r = ones, s = ones（仅靠共享 W + 第一层的多样性）
    """
    def __init__(self, d_in: int, d_out: int, k: int,
                 scaling_init: str = "ones", bias: bool = True):
        super().__init__()
        self.k = k
        self.d_in = d_in
        self.d_out = d_out

        # 共享权重
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # per-member 输入缩放 r
        if scaling_init == "normal":
            self.r = nn.Parameter(torch.randn(k, d_in) * 0.5)
        elif scaling_init == "random-signs":
            self.r = nn.Parameter(torch.ones(k, d_in) * (2 * torch.randint(0, 2, (k, d_in)).float() - 1))
        else:  # "ones"
            self.r = nn.Parameter(torch.ones(k, d_in))

        # per-member 输出缩放 s（始终初始化为 ones）
        self.s = nn.Parameter(torch.ones(k, d_out))

        if bias:
            self.bias = nn.Parameter(torch.zeros(k, d_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, d_in)
        # 输入缩放
        h = x * self.r.unsqueeze(0)  # (B, k, d_in)
        # 共享线性变换（einsum 避免 reshape）
        h = torch.einsum("bki,io->bko", h, self.weight)  # (B, k, d_out)
        # 输出缩放
        h = h * self.s.unsqueeze(0)  # (B, k, d_out)
        if self.bias is not None:
            h = h + self.bias.unsqueeze(0)
        return h


class LinearEnsemble(nn.Module):
    """k 个完全独立的线性层（无权重共享），用于输出头。"""
    def __init__(self, d_in: int, d_out: int, k: int, bias: bool = True):
        super().__init__()
        self.k = k
        # (k, d_in, d_out) — 每个成员独立的权重
        self.weight = nn.Parameter(torch.empty(k, d_in, d_out))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(k, d_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, d_in) → (B, k, d_out)
        h = torch.einsum("bki,kio->bko", x, self.weight)
        if self.bias is not None:
            h = h + self.bias.unsqueeze(0)
        return h


class SharedBatchNorm1d(nn.Module):
    """共享 BatchNorm1d，作用于 (B, k, D) → reshape 为 (B*k, D) 后归一化。"""
    def __init__(self, d: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, k, D = x.shape
        h = x.reshape(B * k, D)
        h = self.bn(h)
        return h.reshape(B, k, D)


# ---------------------------------------------------------------------------
# TabM 模型
# ---------------------------------------------------------------------------
class TabM(nn.Module):
    """TabM: Parameter-Efficient Ensemble of MLPs (ICLR 2025 Oral).

    架构：
      EnsembleView → [LinearBatchEnsemble → BN → ReLU → Dropout] × n_blocks
      → LinearEnsemble (output head) → 平均 k 个成员预测

    关键设计：
      - 第一层 LinearBatchEnsemble 用 scaling_init='normal'，在特征混合前
        创建 k 个不同的对象表示（TabM 论文 Section 3.3 的核心要求）
      - 后续层用 scaling_init='ones'，多样性来自第一层 + 共享权重的正则化
      - 输出头 LinearEnsemble 完全独立，每个成员有自己的分类边界
      - 最终预测 = k 个成员 logits 的平均
    """
    def __init__(self, d_in: int, d_out: int, k: int = 32,
                 n_blocks: int = 3, d_block: int = 256, dropout: float = 0.2):
        super().__init__()
        self.k = k
        self.d_out = d_out

        layers = [EnsembleView(k)]
        for i in range(n_blocks):
            in_dim = d_in if i == 0 else d_block
            # 第一层用 normal scaling 创建多样性，后续用 ones
            scaling = "normal" if i == 0 else "ones"
            layers.extend([
                LinearBatchEnsemble(in_dim, d_block, k, scaling_init=scaling),
                SharedBatchNorm1d(d_block),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.backbone = nn.Sequential(*layers)
        # 输出头：k 个独立线性层
        self.head = LinearEnsemble(d_block, d_out, k)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x: (B, d_in)
            return_all: 若 True，返回 (B, k, d_out) 所有成员预测；
                        若 False，返回 (B, d_out) 平均后的预测
        """
        h = self.backbone(x)  # (B, k, d_block)
        logits_all = self.head(h)  # (B, k, d_out)
        if return_all:
            return logits_all
        return logits_all.mean(dim=1)  # (B, d_out)


# ---------------------------------------------------------------------------
# 单 seed 训练
# ---------------------------------------------------------------------------
def fit_one_seed(X_train, y_train, X_val, y_val, X_test, y_test, seed: int,
                 epochs: int = 80, batch_size: int = 256, k: int = 32,
                 n_blocks: int = 3, d_block: int = 256, dropout: float = 0.2,
                 lr: float = 2e-3, weight_decay: float = 3e-4,
                 debug: bool = False):
    """单 seed 训练：AdamW + cosine，class-weighted CE，best-val tracking。

    官方默认超参：AdamW lr=0.002, weight_decay=0.0003。
    因数据集严重不均衡（class 6 仅 0.3%），使用 sqrt-inverse class weights。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = DEVICE
    n_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    d_in = X_train.shape[1]

    model = TabM(d_in, n_classes, k=k, n_blocks=n_blocks,
                 d_block=d_block, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * (len(X_train) // batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    # sqrt-inverse class weights
    class_counts = torch.bincount(y_train_t, minlength=n_classes).float()
    class_weights = (1.0 / class_counts.clamp(min=1)).sqrt()
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights = class_weights.to(device)

    n_train = len(X_train)
    history = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = perm[start:end]

            x_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            # 前向：返回所有 k 个成员的 logits，对每个成员算 CE 后平均
            logits_all = model(x_batch, return_all=True)  # (B, k, n_classes)
            # 对 k 个成员分别算 CE，然后平均（等价于对平均 logits 算 CE，但梯度更丰富）
            B, k_, C = logits_all.shape
            loss = F.cross_entropy(
                logits_all.reshape(B * k_, C),
                y_batch.unsqueeze(1).expand(-1, k_).reshape(B * k_),
                weight=class_weights,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            sched.step()

            total_loss += loss.item()
            n_batches += 1

        # 验证
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)  # (B, n_classes) 已平均
            val_acc = (logits_val.argmax(dim=1) == y_val_t).float().mean().item()
            val_f1 = f1_score(y_val, logits_val.argmax(dim=1).cpu().numpy(), average="macro")

        history.append({
            "epoch": epoch,
            "loss": total_loss / max(n_batches, 1),
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": sched.get_last_lr()[0],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {kk: v.cpu().clone() for kk, v in model.state_dict().items()}

        if debug and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"      [seed {seed} ep {epoch:3d}] "
                  f"loss={total_loss/max(n_batches,1):.3f}  "
                  f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
                  f"best={best_val_acc:.4f}  lr={sched.get_last_lr()[0]:.6f}")

    # 加载 best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t)
        test_pred = logits_test.argmax(dim=1).cpu().numpy()
        # 也计算每个成员的单独准确率（诊断用）
        logits_all_test = model(X_test_t, return_all=True)  # (B, k, C)
        member_accs = []
        for ki in range(k):
            member_pred = logits_all_test[:, ki, :].argmax(dim=1).cpu().numpy()
            member_accs.append(accuracy_score(y_test, member_pred))

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    if debug:
        print(f"      [seed {seed}] test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
        print(f"      member acc range: [{min(member_accs):.4f}, {max(member_accs):.4f}], "
              f"mean={np.mean(member_accs):.4f}")

    return test_pred, test_acc, test_f1, history


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

    print(f"[3/3] 跑 TabM 3 seeds (k=32 BatchEnsemble, AdamW+cosine) ...")
    test_accs = []
    test_f1s = []
    all_preds = []
    log_lines = []
    all_histories = {}

    for seed in SEEDS:
        t0 = time.time()
        test_pred, test_acc, test_f1, history = fit_one_seed(
            X_train, y_train, X_val, y_val, X_test, y_test, seed,
            epochs=80, batch_size=256, k=32,
            n_blocks=3, d_block=256, dropout=0.2,
            lr=2e-3, weight_decay=3e-4,
            debug=True,
        )
        dt = time.time() - t0
        log_lines.append(f"seed {seed}: test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  ({dt:.1f}s)")
        print(f"      seed {seed}: test acc={test_acc:.4f}  f1={test_f1:.4f}  ({dt:.1f}s)")
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
        "features": "57-D standard (28 z-score + 28 validity mask + 1 gender)",
        "optimizer": "AdamW+cosine (lr=2e-3, wd=3e-4)",
        "architecture": f"TabM k=32, 3 blocks, d_block=256, dropout=0.2, BatchEnsemble",
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

    # 写训练历史
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
