#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_m3_bandit_comparison.py — M3 推荐系统：Contextual Bandit 算法对比

实现 4 种 contextual bandit 算法，在学生健康推荐场景下对比：
  1. Epsilon-Greedy（基线，原M3用的）
  2. LinUCB（线性上下文老虎机，标准算法）
  3. Thompson Sampling（贝叶斯方法）
  4. NeuralUCB（神经网络上下文老虎机，2024+主流）

场景：
  - 上下文：学生健康特征（9类 one-hot + 7项体测指标 = 16维）
  - 臂：4种推荐策略（运动/膳食/综合/维持）
  - 奖励：模拟干预效果（基于学生状态与策略的匹配度）
  - 评估：累积遗憾、收敛速度、最终策略准确率

输出：
  - outputs/m3_bandit/m3_bandit_results.csv
  - outputs/m3_bandit/m3_bandit_summary.md
  - outputs/m3_bandit/m3_bandit_curves.png（收敛曲线）
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
OUT_DIR = Path(__file__).parent / "outputs" / "m3_bandit"

SEEDS = [42, 43, 44]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_ARMS = 4
ARM_NAMES = ["运动干预为主", "膳食干预为主", "运动+膳食综合", "维持现状+监测"]
CLASS_NAMES = [
    "低水平-退化型", "低水平-平稳型", "低水平-改善型",
    "中水平-退化型", "中水平-平稳型", "中水平-改善型",
    "高水平-退化型", "高水平-平稳型", "高水平-改善型",
]

# 每类学生的最优策略（先验，用于模拟奖励）
# 0:运动, 1:膳食, 2:综合, 3:维持
OPTIMAL_ARM = {
    0: 2,  # 低水平-退化：综合
    1: 0,  # 低水平-平稳：运动
    2: 3,  # 低水平-改善：维持
    3: 2,  # 中水平-退化：综合
    4: 0,  # 中水平-平稳：运动
    5: 3,  # 中水平-改善：维持
    6: 1,  # 高水平-退化：膳食
    7: 3,  # 高水平-平稳：维持
    8: 3,  # 高水平-改善：维持
}

METRICS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run_sec", "strength",
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data():
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("", "_raw"))
    print(f"  学生总数: {len(df)}")
    return df


def build_context(df, n_samples=500, seed=42):
    """
    构建上下文向量：9类 one-hot + 7项体测指标(g4) = 16维
    按类别均衡采样。
    """
    rng = np.random.RandomState(seed)
    contexts = []
    classes = []
    optimal_arms = []

    for cls in range(9):
        cls_df = df[df["rule_class"] == cls]
        if len(cls_df) == 0:
            continue
        n_cls = min(n_samples // 9, len(cls_df))
        sample = cls_df.sample(n=n_cls, random_state=seed)

        for _, row in sample.iterrows():
            # 9类 one-hot
            cls_onehot = np.zeros(9)
            cls_onehot[cls] = 1.0
            # 7项体测指标
            feat = []
            for m in METRICS:
                val = row.get(f"{m}_g4")
                feat.append(0.0 if pd.isna(val) else float(val))
            # 标准化体测指标（简单归一化）
            feat = np.array(feat, dtype=np.float32)
            feat = (feat - feat.mean()) / (feat.std() + 1e-8)
            ctx = np.concatenate([cls_onehot, feat])
            contexts.append(ctx)
            classes.append(cls)
            optimal_arms.append(OPTIMAL_ARM[cls])

    return np.array(contexts, dtype=np.float32), np.array(classes), np.array(optimal_arms)


# ---------------------------------------------------------------------------
# 模拟环境
# ---------------------------------------------------------------------------
class StudentEnv:
    """
    模拟学生健康推荐环境。
    奖励 = 策略匹配度 + 噪声。
    最优策略奖励高，非最优策略奖励低。
    """
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
        ctx = self.contexts[self.idx]
        cls = self.classes[self.idx]
        optimal = self.optimal_arms[self.idx]

        # 基础奖励：最优臂0.8，其他0.3
        if action == optimal:
            base_reward = 0.8
        else:
            base_reward = 0.3

        # 类别特定调整
        # 退化型学生对综合干预更敏感
        if cls in [0, 3, 6] and action == 2:  # 退化型 + 综合
            base_reward = min(0.9, base_reward + 0.1)
        # 高水平学生对维持策略更满意
        if cls in [6, 7, 8] and action == 3:  # 高水平 + 维持
            base_reward = min(0.9, base_reward + 0.1)

        reward = base_reward + self.rng.normal(0, self.noise)
        reward = np.clip(reward, 0, 1)

        self.idx = (self.idx + 1) % self.n
        next_ctx = self.contexts[self.idx]
        return next_ctx, reward, (action == optimal)


# ---------------------------------------------------------------------------
# Bandit 算法
# ---------------------------------------------------------------------------
class EpsilonGreedy:
    """Epsilon-Greedy（基线）"""
    def __init__(self, n_arms, context_dim, epsilon=0.1, seed=42):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select(self, context):
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, context, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] = (self.values[action] * (n - 1) + reward) / n


class LinUCB:
    """
    LinUCB：线性上下文老虎机。
    每个臂维护一个线性模型，用UCB探索。
    标准算法，有理论保证。
    """
    def __init__(self, n_arms, context_dim, alpha=1.0, seed=42):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        # 每个臂的线性模型参数
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


class ThompsonSampling:
    """
    Thompson Sampling：贝叶斯上下文老虎机。
    用贝叶斯线性回归，每个臂维护后验分布，采样选择。
    """
    def __init__(self, n_arms, context_dim, v=1.0, seed=42):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.v = v  # 先验方差
        self.rng = np.random.RandomState(seed)
        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]
        self.cov = [np.eye(context_dim) for _ in range(n_arms)]
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def select(self, context):
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            # 从后验采样theta
            cov_inv = np.linalg.inv(self.cov[a])
            theta = self.rng.multivariate_normal(self.mu[a], self.v**2 * np.linalg.inv(cov_inv))
            p[a] = theta @ context
        return int(np.argmax(p))

    def update(self, context, action, reward):
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
        self.cov[action] = np.linalg.inv(self.A[action])
        self.mu[action] = self.cov[action] @ self.b[action]


class NeuralUCB:
    """
    NeuralUCB：神经网络上下文老虎机。
    用MLP建模奖励函数，用梯度的范数作为不确定性估计。
    2024+主流方法。
    """
    def __init__(self, n_arms, context_dim, hidden_dim=64, alpha=1.0, lr=1e-3, seed=42):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        # 共享的奖励预测网络（输入context，输出每个臂的奖励）
        self.model = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_arms),
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []  # (context, action, reward)

    def select(self, context):
        self.model.eval()
        ctx_tensor = torch.tensor(context, dtype=torch.float32).to(DEVICE)
        ctx_tensor.requires_grad_(True)

        pred = self.model(ctx_tensor)
        # 用梯度范数作为不确定性
        uncertainties = []
        for a in range(self.n_arms):
            self.model.zero_grad()
            pred[a].backward(retain_graph=True)
            grad_norm = ctx_tensor.grad.norm().item() if ctx_tensor.grad is not None else 0
            uncertainties.append(grad_norm)
            ctx_tensor.grad.zero_()

        ucb = pred.detach().cpu().numpy() + self.alpha * np.array(uncertainties)
        return int(np.argmax(ucb))

    def update(self, context, action, reward):
        self.memory.append((context, action, reward))
        # 每10步训练一次
        if len(self.memory) % 10 == 0:
            self._train()

    def _train(self, epochs=5):
        if len(self.memory) < 32:
            return
        self.model.train()
        batch = self.rng.choice(len(self.memory), min(64, len(self.memory)), replace=False)
        for _ in range(epochs):
            total_loss = 0
            for idx in batch:
                ctx, act, rew = self.memory[idx]
                ctx_t = torch.tensor(ctx, dtype=torch.float32).to(DEVICE)
                pred = self.model(ctx_t)
                loss = F.mse_loss(pred[act], torch.tensor(rew, dtype=torch.float32).to(DEVICE))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()


# ---------------------------------------------------------------------------
# 运行实验
# ---------------------------------------------------------------------------
def run_bandit(algo_name, env, n_steps=1000, seed=42):
    """运行一个bandit算法，返回累积遗憾和每步奖励"""
    if algo_name == "epsilon_greedy":
        algo = EpsilonGreedy(N_ARMS, env.contexts.shape[1], epsilon=0.1, seed=seed)
    elif algo_name == "linucb":
        algo = LinUCB(N_ARMS, env.contexts.shape[1], alpha=1.0, seed=seed)
    elif algo_name == "thompson":
        algo = ThompsonSampling(N_ARMS, env.contexts.shape[1], v=1.0, seed=seed)
    elif algo_name == "neuralucb":
        algo = NeuralUCB(N_ARMS, env.contexts.shape[1], hidden_dim=64, alpha=0.5, lr=1e-3, seed=seed)
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    regrets = []
    rewards = []
    optimal_counts = np.zeros(N_ARMS)
    action_counts = np.zeros(N_ARMS)
    cumulative_regret = 0

    ctx = env.reset()
    for t in range(n_steps):
        action = algo.select(ctx)
        next_ctx, reward, is_optimal = env.step(action)
        algo.update(ctx, action, reward)

        # 计算遗憾（最优奖励 - 实际奖励）
        optimal_reward = 0.8  # 最优臂的期望奖励
        regret = optimal_reward - reward
        cumulative_regret += regret

        regrets.append(cumulative_regret)
        rewards.append(reward)
        action_counts[action] += 1
        if is_optimal:
            optimal_counts[action] += 1

        ctx = next_ctx

    return {
        "final_regret": cumulative_regret,
        "avg_reward": np.mean(rewards[-100:]),  # 最后100步的平均奖励
        "regrets": regrets,
        "rewards": rewards,
        "action_distribution": action_counts / n_steps,
        "optimal_rate": np.sum(optimal_counts) / n_steps,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载数据并构建上下文 ...")
    df = load_data()
    contexts, classes, optimal_arms = build_context(df, n_samples=900, seed=42)
    print(f"  上下文维度: {contexts.shape[1]}, 样本数: {len(contexts)}")
    print(f"  类别分布: {[np.sum(classes == i) for i in range(9)]}")

    algos = ["epsilon_greedy", "linucb", "thompson", "neuralucb"]
    n_steps = 1000

    print(f"\n[2/4] 运行 4 种算法 × {len(SEEDS)} seeds × {n_steps} 步 ...")
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
                "avg_reward_last100": result["avg_reward"],
                "optimal_rate": result["optimal_rate"],
                "action_dist": json.dumps(result["action_distribution"].tolist()),
                "seconds": round(elapsed, 1),
            })
            print(f"    seed={seed}: regret={result['final_regret']:.1f}  "
                  f"avg_reward={result['avg_reward']:.4f}  "
                  f"optimal_rate={result['optimal_rate']:.4f}  "
                  f"({elapsed:.1f}s)")

            if seed == 42:
                all_curves[algo_name] = result["regrets"]

    print(f"\n[3/4] 汇总结果 ...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "m3_bandit_results.csv", index=False)

    summary = results_df.groupby("algorithm").agg(
        regret_mean=("final_regret", "mean"),
        regret_std=("final_regret", "std"),
        reward_mean=("avg_reward_last100", "mean"),
        reward_std=("avg_reward_last100", "std"),
        optimal_rate_mean=("optimal_rate", "mean"),
        optimal_rate_std=("optimal_rate", "std"),
    ).reset_index()
    summary = summary.sort_values("regret_mean")
    summary.to_csv(OUT_DIR / "m3_bandit_summary.csv", index=False)

    # 保存收敛曲线
    curves_df = pd.DataFrame(all_curves)
    curves_df.to_csv(OUT_DIR / "m3_bandit_curves.csv", index=False)

    # 生成 summary.md
    with open(OUT_DIR / "m3_bandit_summary.md", "w") as f:
        f.write("# M3 Contextual Bandit 算法对比\n\n")
        f.write(f"上下文维度: {contexts.shape[1]} (9类one-hot + 7项体测)\n")
        f.write(f"臂数: {N_ARMS} ({', '.join(ARM_NAMES)})\n")
        f.write(f"步数: {n_steps}\n\n")
        f.write("## 结果汇总（按累积遗憾升序，越低越好）\n\n")
        f.write("| 算法 | 累积遗憾 | 最后100步平均奖励 | 最优臂选择率 |\n")
        f.write("|---|---|---|---|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['algorithm']} | {row['regret_mean']:.1f}±{row['regret_std']:.1f} "
                    f"| {row['reward_mean']:.4f}±{row['reward_std']:.4f} "
                    f"| {row['optimal_rate_mean']:.4f}±{row['optimal_rate_std']:.4f} |\n")
        f.write("\n## 算法说明\n\n")
        f.write("- **Epsilon-Greedy**: 最简单的基线，以epsilon概率随机探索\n")
        f.write("- **LinUCB**: 线性上下文老虎机，每个臂维护线性模型，用UCB探索（标准算法）\n")
        f.write("- **Thompson Sampling**: 贝叶斯方法，从后验分布采样选择臂\n")
        f.write("- **NeuralUCB**: 神经网络上下文老虎机，用MLP建模奖励+梯度范数估计不确定性（2024+主流）\n")

    print(f"\n[4/4] 完成 ✅")
    print(f"\n  按累积遗憾排名（越低越好）：")
    for _, row in summary.iterrows():
        print(f"    {row['algorithm']:20s}  regret={row['regret_mean']:.1f}±{row['regret_std']:.1f}  "
              f"reward={row['reward_mean']:.4f}  optimal_rate={row['optimal_rate_mean']:.4f}")

    print(f"\n  写入 {OUT_DIR / 'm3_bandit_results.csv'}")
    print(f"  写入 {OUT_DIR / 'm3_bandit_summary.csv'}")
    print(f"  写入 {OUT_DIR / 'm3_bandit_curves.csv'}")
    print(f"  写入 {OUT_DIR / 'm3_bandit_summary.md'}")


if __name__ == "__main__":
    sys.exit(main())
