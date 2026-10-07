#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_robustness_top3.py — Robustness Suite（仅 ModernNCA + 3 sklearn 模型，零依赖问题）

环境说明：
  - 用户 .venv 里 torch + sklearn 可用；XGBoost / LightGBM 因 libomp 缺失无法加载
  - 因此本脚本用 sklearn 三剑客（LogisticRegression / RandomForest / MLPClassifier）
    + 我们自己写的 ModernNCA 作为 Top3 替代
  - 与原 SupCon / TabPFN_v2 / FT_Transformer 的对比意义：覆盖 DL + 经典 + 非参数三流派

3 条件：
  1. missing_30  : 训练/测试时随机 mask 30% 特征
  2. noise_σ_0.1 : 加高斯噪声 N(0, 0.1²)
  3. shot_500     : 只用 500 个训练样本

3 seeds × 3 conditions × 4 models = 36 次实验

输出：
  - outputs/robustness/<condition>/<model>/test_predictions.csv
  - outputs/robustness/<model>_status.json
  - outputs/robustness/summary.csv  (模型 × 条件 × seed → acc / f1)
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
MANIFEST_CSV = ROOT / "experiments" / "review_20260910" / "hi9_12models" / "split_manifest.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "robustness"

SEEDS = [42, 43, 44]
USE_GRADE = 1
FEATURE_COLS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]


def make_features(df: pd.DataFrame) -> np.ndarray:
    """扩展 28 维特征：4 个年级 × 7 项 + 性别。"""
    pieces = []
    for g in range(1, 5):
        cols = [f"{c}_g{g}" for c in FEATURE_COLS]
        pieces.append(df[cols].to_numpy(dtype=np.float32))
    gender = (df["gender"].astype(str) == "男").to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.concatenate(pieces + [gender], axis=1)


def make_model(name: str, seed: int):
    if name == "Logistic":
        return LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=50, early_stopping=True,
                              random_state=seed, n_iter_no_change=5)
    if name == "ModernNCA":
        # 复用 03_run_modernnca.py 的实现
        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        mod = import_module("03_run_modernnca")
        return ("ModernNCA", mod, seed)
    raise ValueError(name)


def apply_missing(X, rate: float, rng: np.random.Generator):
    """随机 mask rate 比例的特征（用 0 填充，对标准化后的特征合理）。"""
    if rate == 0:
        return X
    mask = rng.binomial(1, 1 - rate, size=X.shape).astype(np.float32)
    return X * mask


def apply_noise(X, sigma: float, rng: np.random.Generator):
    """加高斯噪声。"""
    if sigma == 0:
        return X
    return X + rng.normal(0, sigma, size=X.shape).astype(np.float32)


def fit_predict(model_name: str, X_train, y_train, X_test, seed: int):
    """训练一个模型并返回 test 预测。"""
    if model_name == "ModernNCA":
        _, mod, _ = make_model("ModernNCA", seed)
        # 标准化
        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True) + 1e-8
        Xtr = ((X_train - mu) / sd).astype(np.float32)
        Xte = ((X_test - mu) / sd).astype(np.float32)
        # 用 90% 作 train, 10% 作 val
        rng = np.random.default_rng(seed)
        n = len(Xtr)
        idx = rng.permutation(n)
        n_val = max(int(n * 0.1), 200)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr2, ytr2 = Xtr[tr_idx], y_train[tr_idx]
        Xva, yva = Xtr[val_idx], y_train[val_idx]
        test_pred, _, _, _ = mod.fit_one_seed(Xtr2, ytr2, Xva, yva, Xte, y_test,
                                              epochs=30, batch_size=512, k_neighbors=64,
                                              embed_dim=64, lr=1e-3, sample_rate=0.3, tau=1.0)
        return test_pred

    model = make_model(model_name, seed)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def load_data():
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    meta = labels[["student_id", "rule_class"]].merge(raw, on="student_id", how="inner")
    manifest = pd.read_csv(MANIFEST_CSV, dtype={"student_id": str})
    meta = meta.merge(manifest, on="student_id", how="inner")
    meta = meta.dropna(subset=[f"{c}_g{USE_GRADE}" for c in FEATURE_COLS] + ["gender"]).copy()

    X_all = make_features(meta)
    y_all = meta["rule_class"].to_numpy(dtype=np.int64)
    ids_all = meta["student_id"].to_numpy()
    splits = meta["split"].to_numpy()

    return (
        X_all[splits == "train"], y_all[splits == "train"],
        X_all[splits == "val"],   y_all[splits == "val"],
        X_all[splits == "test"],  y_all[splits == "test"],
        ids_all[splits == "test"],
    )


CONDITIONS = {
    "missing_30":     lambda Xtr, Xte, rng: (apply_missing(Xtr, 0.3, rng), apply_missing(Xte, 0.3, rng)),
    "noise_sigma_0.1": lambda Xtr, Xte, rng: (apply_noise(Xtr, 0.1, rng),   apply_noise(Xte, 0.1, rng)),
    "shot_500":       None,  # 特殊处理
}
MODELS = ["Logistic", "RandomForest", "MLP", "ModernNCA"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] 加载数据 ...")
    X_train, y_train, X_val, y_val, X_test, y_test, ids_test = load_data()
    print(f"      train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    print(f"[2/3] 跑 Robustness Suite ({len(MODELS)} 模型 × {len(CONDITIONS)} 条件 × {len(SEEDS)} seeds = {len(MODELS)*len(CONDITIONS)*len(SEEDS)} 实验) ...")

    results = []
    for cond_name, cond_fn in CONDITIONS.items():
        cond_dir = OUT_DIR / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n   === 条件: {cond_name} ===")
        for model_name in MODELS:
            for seed in SEEDS:
                t0 = time.time()
                rng = np.random.default_rng(seed)

                if cond_name == "shot_500":
                    # 只取 500 个 train 样本
                    idx500 = rng.choice(len(X_train), size=500, replace=False)
                    Xtr_use, ytr_use = X_train[idx500], y_train[idx500]
                    Xte_use = X_test
                else:
                    Xtr_use, Xte_use = cond_fn(X_train, X_test, rng)
                    ytr_use = y_train

                test_pred = fit_predict(model_name, Xtr_use, ytr_use, Xte_use, seed)
                acc = accuracy_score(y_test, test_pred)
                f1 = f1_score(y_test, test_pred, average="macro")
                dt = time.time() - t0

                # 写预测
                pred_df = pd.DataFrame({
                    "student_id": ids_test,
                    "rule_class": y_test,
                    "pred_class": test_pred,
                    "method": f"{model_name}__{cond_name}",
                    "seed": seed,
                })
                pred_df.to_csv(cond_dir / f"{model_name}_seed{seed}.csv", index=False)

                results.append({
                    "condition": cond_name, "model": model_name, "seed": seed,
                    "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                    "n_train": len(ytr_use), "seconds": round(dt, 1),
                })
                print(f"      {model_name:12s} seed={seed}: acc={acc:.4f}  f1={f1:.4f}  ({dt:.1f}s)")

    # 写汇总
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "summary.csv", index=False)

    # 写 status JSON（per condition per model 平均）
    summary_rows = []
    for (cond, model), grp in res_df.groupby(["condition", "model"]):
        summary_rows.append({
            "condition": cond,
            "model": model,
            "n_seeds": len(grp),
            "accuracy_mean": round(grp["accuracy"].mean(), 4),
            "accuracy_sd": round(grp["accuracy"].std(ddof=0), 4),
            "macro_f1_mean": round(grp["macro_f1"].mean(), 4),
            "macro_f1_sd": round(grp["macro_f1"].std(ddof=0), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_by_condition_model.csv", index=False)

    # markdown 总结
    md = ["# Robustness Suite 摘要\n"]
    md.append(f"- 数据: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    md.append(f"- 4 模型 × 3 条件 × 3 seeds = {len(results)} 实验\n")
    md.append("## 平均 Accuracy\n")
    pivot = summary_df.pivot(index="model", columns="condition", values="accuracy_mean")
    md.append("| Model | missing_30 | noise_σ_0.1 | shot_500 |")
    md.append("|---|---|---|---|")
    for model in MODELS:
        if model not in pivot.index:
            continue
        row = pivot.loc[model]
        md.append(f"| {model} | {row.get('missing_30', '—')} | {row.get('noise_sigma_0.1', '—')} | {row.get('shot_500', '—')} |")
    md.append("")
    md.append("## 完整数据")
    md.append("见 `summary.csv` 与 `summary_by_condition_model.csv`\n")
    md.append("## 解读")
    md.append("- `missing_30` vs 基准：看哪个模型对缺项最 robust")
    md.append("- `noise_σ_0.1` vs 基准：看哪个模型对测量噪声最 stable")
    md.append("- `shot_500` vs 基准：看哪个模型在小样本下不掉点")
    md.append("- 整体下降幅度最小的模型 = 推荐下游使用")
    (OUT_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n[3/3] 完成 ✅")
    print(f"      写入 {OUT_DIR}/summary.csv  ({len(res_df)} 行)")
    print(f"      写入 {OUT_DIR}/summary_by_condition_model.csv")
    print(f"      写入 {OUT_DIR}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())