#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_m3_neural_bandit_comparison.py — M3 第二阶段：Neural Contextual Bandit 深度对比

实现 3 种 Neural Bandit 算法，增加步数让神经网络充分收敛：
  1. NeuralUCB（改进版：每步训练，梯度范数不确定性）
  2. NeuralTS（Neural Thompson Sampling：贝叶斯神经网络，后验采样）
  3. EE-Net（Exploitation-Exploration Network：双网络，2024新方法）

与 LinUCB（线性基线）对比。

输出：
  - outputs/m3_neural_bandit/neural_bandit_results.csv
  - outputs/m3_neural_bandit/neural_bandit_summary.md
  - outputs/m3_neural_bandit/neural_bandit_curves.csv
"""
from __future__ import annotations
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "m3_neural_bandit"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_ARMS = 4
ARM_NAMES = ["运动干预为主", "膳食干预为主", "运动+膳食综合", "维持现状+监测"]
CLASS_NAMES = [
    "低水平-退化型", "低水平-平稳型", "低水平-改善型",
    "中水平-退化型", "中水平-平稳型", "中水平-改善型",
    "高水平-退化型", "高水平-平稳型", "高水平-改善型",
]
OPTIMAL_ARM = {0: 2, 1: 0, 2: 3, 3: 2, 4: 0, 5: 3, 6: 1, 7: 3, 8: 3}
METRICS = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
           "sit_and_reach", "endurance_run_sec", "strength"]


def load_data():
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("", "_raw"))
    return df


def build_context(df, n_samples=900, seed=42):
    rng = np.random.RandomState(seed)
    contexts, classes, optimal_arms = [], [], []
    for cls in range(9):
        cls_df = df[df["rule_class"] == cls]
        if len(cls_df) == 0:
            continue
        n_cls = min(n_samples // 9, len(cls_df))
        sample = cls_df.sample(n=n_cls, random_state=seed)
        for _, row in sample.iterrows():
            cls_onehot = np.zeros(9)
            cls_onehot[cls] = 1.0
            feat = []
            for m in METRICS:
                val = row.get(f"{m}_g4")
                feat.append(0.0 if pd.isna(val) else float(val))
            feat = np.array(feat, dtype=np.float32)
            feat = (feat - feat.mean()) / (feat.std() + 1e-8)
            ctx = np.concatenate([cls_onehot, feat])
            contexts.append(ctx)
            classes.append(cls)
            optimal_arms.append(OPTIMAL_ARM[cls])
    return np.array(contexts, dtype=np.float32), np.array(classes), np.array(optimal_arms)


class StudentEnv:
    def __init__(self, contexts, classes, optimal_arms, noise=0.3, seed=42):
        self.contexts = contexts
        self.classes = classes
        self.optimal_arms = optimal_arms
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self.n = len(contexts)
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self.contexts[self.idx]

    def step(self, action):
        cls = self.classes[self.idx]
        optimal = self.optimal_arms[self.idx]
        base_reward = 0.8 if action == optimal else 0.3
        if cls in [0, 3, 6] and action == 2:
            base_reward = min(0.9, base_reward + 0.1)
        if cls in [6, 7, 8] and action == 3:
            base_reward = min(0.9, base_reward + 0.1)
        reward = np.clip(base_reward + self.rng.normal(0, self.noise), 0, 1)
        self.idx = (self.idx + 1) % self.n
        return self.contexts[self.idx], reward, (action == optimal)


# ---------------------------------------------------------------------------
# LinUCB（线性基线）
# ---------------------------------------------------------------------------
class LinUCB:
    def __init__(self, n_arms, context_dim, alpha=1.0, seed=42):
        self.n_arms = n_arms
        self.alpha = alpha
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def select(self, context):
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        return int(np.argmax(p))

    def update(self, context, action, reward):
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


# ---------------------------------------------------------------------------
# NeuralUCB（改进版）
# ---------------------------------------------------------------------------
class NeuralUCBNet(nn.Module):
    def __init__(self, context_dim, hidden_dim, n_arms):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_arms),
        )
    def forward(self, x):
        return self.net(x)


class NeuralUCB:
    """
    NeuralUCB：用神经网络估计奖励，梯度范数作为不确定性。
    改进：每步都训练，使用SGD更新。
    """
    def __init__(self, n_arms, context_dim, hidden_dim=64, alpha=0.5, lr=1e-3, seed=42):
        self.n_arms = n_arms
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.model = NeuralUCBNet(context_dim, hidden_dim, n_arms).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def select(self, context):
        self.model.eval()
        ctx = torch.tensor(context, dtype=torch.float32, requires_grad=True).to(DEVICE)
        pred = self.model(ctx)
        uncertainties = []
        for a in range(self.n_arms):
            self.model.zero_grad()
            if ctx.grad is not None:
                ctx.grad.zero_()
            pred[a].backward(retain_graph=True)
            grad_norm = ctx.grad.norm().item() if ctx.grad is not None else 0
            uncertainties.append(grad_norm)
        ucb = pred.detach().cpu().numpy() + self.alpha * np.array(uncertainties)
        return int(np.argmax(ucb))

    def update(self, context, action, reward):
        self.memory.append((context, action, reward))
        # 每步都做一次SGD更新
        self.model.train()
        ctx = torch.tensor(context, dtype=torch.float32).to(DEVICE)
        pred = self.model(ctx)
        loss = F.mse_loss(pred[action], torch.tensor(reward, dtype=torch.float32).to(DEVICE))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ---------------------------------------------------------------------------
# NeuralTS（Neural Thompson Sampling）
# ---------------------------------------------------------------------------
class NeuralTS:
    """
    Neural Thompson Sampling：用贝叶斯神经网络（MC Dropout）估计后验，
    从后验采样选择臂。
    """
    def __init__(self, n_arms, context_dim, hidden_dim=64, n_samples=10, lr=1e-3, seed=42):
        self.n_arms = n_arms
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        # 带Dropout的网络，推理时保持dropout开启做MC采样
        self.model = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_arms),
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def select(self, context):
        self.model.train()  # 保持dropout开启
        ctx = torch.tensor(context, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            samples = torch.stack([self.model(ctx) for _ in range(self.n_samples)])
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
        # Thompson Sampling：从后验采样
        sampled = mean + self.rng.normal(0, 1, size=self.n_arms) * std
        return int(np.argmax(sampled))

    def update(self, context, action, reward):
        self.memory.append((context, action, reward))
        self.model.train()
        ctx = torch.tensor(context, dtype=torch.float32).to(DEVICE)
        pred = self.model(ctx)
        loss = F.mse_loss(pred[action], torch.tensor(reward, dtype=torch.float32).to(DEVICE))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ---------------------------------------------------------------------------
# EE-Net（Exploitation-Exploration Network，2024）
# ---------------------------------------------------------------------------
class EENet(nn.Module):
    """双网络：利用网络估计奖励，探索网络估计潜在收益"""
    def __init__(self, context_dim, hidden_dim, n_arms):
        super().__init__()
        self.exploit_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_arms),
        )
        self.explore_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_arms),
        )
    def forward(self, x):
        return self.exploit_net(x), self.explore_net(x)


class EENetBandit:
    """
    EE-Net：利用网络估计当前奖励，探索网络估计与当前估计相比的潜在收益。
    决策 = 利用 + beta * 探索。
    """
    def __init__(self, n_arms, context_dim, hidden_dim=64, beta=0.5, lr=1e-3, seed=42):
        self.n_arms = n_arms
        self.beta = beta
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.model = EENet(context_dim, hidden_dim, n_arms).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def select(self, context):
        self.model.eval()
        ctx = torch.tensor(context, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            exploit, explore = self.model(ctx)
            # 探索网络输出的是潜在收益（非负）
            explore = F.softplus(explore)
            score = exploit + self.beta * explore
        return int(torch.argmax(score).item())

    def update(self, context, action, reward):
        self.memory.append((context, action, reward))
        self.model.train()
        ctx = torch.tensor(context, dtype=torch.float32).to(DEVICE)
        exploit, explore = self.model(ctx)
        # 利用网络：预测实际奖励
        loss_exploit = F.mse_loss(exploit[action], torch.tensor(reward, dtype=torch.float32).to(DEVICE))
        # 探索网络：预测后悔（最优奖励 - 实际奖励），鼓励探索未被充分利用的臂
        optimal_reward = 0.8
        regret = max(0, optimal_reward - reward)
        loss_explore = F.mse_loss(F.softplus(explore[action]), torch.tensor(regret, dtype=torch.float32).to(DEVICE))
        loss = loss_exploit + 0.5 * loss_explore
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ---------------------------------------------------------------------------
# 运行实验
# ---------------------------------------------------------------------------
def run_bandit(algo_name, env, n_steps=3000, seed=42):
    context_dim = env.contexts.shape[1]
    if algo_name == "linucb":
        algo = LinUCB(N_ARMS, context_dim, alpha=1.0, seed=seed)
    elif algo_name == "neuralucb":
        algo = NeuralUCB(N_ARMS, context_dim, hidden_dim=64, alpha=0.3, lr=1e-3, seed=seed)
    elif algo_name == "neuralt":
        algo = NeuralTS(N_ARMS, context_dim, hidden_dim=64, n_samples=10, lr=1e-3, seed=seed)
    elif algo_name == "eenet":
        algo = EENetBandit(N_ARMS, context_dim, hidden_dim=64, beta=0.5, lr=1e-3, seed=seed)
    else:
        raise ValueError(f"Unknown: {algo_name}")

    regrets, rewards = [], []
    action_counts = np.zeros(N_ARMS)
    optimal_count = 0
    cumulative_regret = 0
    ctx = env.reset()

    for t in range(n_steps):
        action = algo.select(ctx)
        next_ctx, reward, is_optimal = env.step(action)
        algo.update(ctx, action, reward)
        cumulative_regret += 0.8 - reward
        regrets.append(cumulative_regret)
        rewards.append(reward)
        action_counts[action] += 1
        if is_optimal:
            optimal_count += 1
        ctx = next_ctx

    return {
        "final_regret": cumulative_regret,
        "avg_reward_last500": np.mean(rewards[-500:]),
        "regrets": regrets,
        "action_distribution": action_counts / n_steps,
        "optimal_rate": optimal_count / n_steps,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] 加载数据 ...")
    df = load_data()
    contexts, classes, optimal_arms = build_context(df, n_samples=900, seed=42)
    print(f"  上下文维度: {contexts.shape[1]}, 样本数: {len(contexts)}")

    algos = ["linucb", "neuralucb", "neuralt", "eenet"]
    n_steps = 3000

    print(f"\n[2/3] 运行 {len(algos)} 种算法 × {len(SEEDS)} seeds × {n_steps} 步 ...")
    all_results = []
    all_curves = {}

    for algo_name in algos:
        print(f"\n  === {algo_name} ===")
        for seed in SEEDS:
            t0 = time.time()
            env = StudentEnv(contexts, classes, optimal_arms, noise=0.3, seed=seed)
            result = run_bandit(algo_name, env, n_steps=n_steps, seed=seed)
            elapsed = time.time() - t0
            all_results.append({
                "algorithm": algo_name,
                "seed": seed,
                "final_regret": result["final_regret"],
                "avg_reward_last500": result["avg_reward_last500"],
                "optimal_rate": result["optimal_rate"],
                "action_dist": json.dumps(result["action_distribution"].tolist()),
                "seconds": round(elapsed, 1),
            })
            print(f"    seed={seed}: regret={result['final_regret']:.1f}  "
                  f"reward={result['avg_reward_last500']:.4f}  "
                  f"optimal={result['optimal_rate']:.4f}  ({elapsed:.1f}s)")
            if seed == 42:
                all_curves[algo_name] = result["regrets"]

    print(f"\n[3/3] 汇总 ...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "neural_bandit_results.csv", index=False)

    summary = results_df.groupby("algorithm").agg(
        regret_mean=("final_regret", "mean"), regret_std=("final_regret", "std"),
        reward_mean=("avg_reward_last500", "mean"), reward_std=("avg_reward_last500", "std"),
        optimal_rate_mean=("optimal_rate", "mean"), optimal_rate_std=("optimal_rate", "std"),
    ).reset_index().sort_values("regret_mean")
    summary.to_csv(OUT_DIR / "neural_bandit_summary.csv", index=False)

    pd.DataFrame(all_curves).to_csv(OUT_DIR / "neural_bandit_curves.csv", index=False)

    with open(OUT_DIR / "neural_bandit_summary.md", "w") as f:
        f.write("# M3 Neural Contextual Bandit 对比（3000步）\n\n")
        f.write("| 算法 | 累积遗憾↓ | 最后500步奖励↑ | 最优臂率↑ |\n")
        f.write("|---|---|---|---|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['algorithm']} | {row['regret_mean']:.1f}±{row['regret_std']:.1f} "
                    f"| {row['reward_mean']:.4f}±{row['reward_std']:.4f} "
                    f"| {row['optimal_rate_mean']:.4f}±{row['optimal_rate_std']:.4f} |\n")
        f.write("\n## 算法说明\n\n")
        f.write("- **LinUCB**: 线性上下文老虎机（基线）\n")
        f.write("- **NeuralUCB**: 神经网络奖励估计 + 梯度范数不确定性\n")
        f.write("- **NeuralTS**: 贝叶斯神经网络（MC Dropout）+ Thompson Sampling\n")
        f.write("- **EE-Net**: 双网络（利用+探索），2024新方法\n")

    print(f"\n完成 ✅")
    print(f"\n  按累积遗憾排名：")
    for _, row in summary.iterrows():
        print(f"    {row['algorithm']:15s}  regret={row['regret_mean']:.1f}±{row['regret_std']:.1f}  "
              f"reward={row['reward_mean']:.4f}  optimal={row['optimal_rate_mean']:.4f}")
    print(f"\n  写入 {OUT_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
