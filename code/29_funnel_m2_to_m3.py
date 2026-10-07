#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29_funnel_m2_to_m3.py — M2→M3 漏斗：用 M2 预测驱动 PPO 奖励

漏斗逻辑：
  M2 在高风险学生上预测 P(继续退化)
  M3 PPO 在 P(退化) > threshold 的学生上选干预
  奖励 = M2 预测的退化概率下降量（不是规则算的 ΔHI）

这样 M3 的优化目标是"降低 M2 预测的退化风险"，不是硬编码的运动效果。
"""
from __future__ import annotations
import os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
OUT_DIR = Path(__file__).parent / "outputs" / "funnel_m2_m3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
           "sit_and_reach", "endurance_run_sec", "strength"]
HIGHER_BETTER = {"bmi": False, "vital_capacity": True, "sprint_50m": False,
                 "standing_long_jump": True, "sit_and_reach": True,
                 "endurance_run_sec": False, "strength": True}
ACTIONS = ["运动干预为主", "膳食干预为主", "运动+膳食综合", "维持现状"]
N_ACTIONS = 4

# 干预效果矩阵（文献依据版，12周z-score改善量）
# 来源: Chase&Conn 2015 (SMD=0.48), Wang 2024 (BMI -0.9), Lin 2015 (SMD=0.65)
# 顺序: bmi↓ vc↑ sprint↓ jump↑ reach↑ endurance↓ strength↑
EFFECT_MATRIX = np.array([
    [-0.26, 0.65, -0.15, 0.30, 0.10, -0.50, 0.15],  # 运动
    [-0.35, 0.05, -0.02, 0.05, 0.05, -0.05, 0.02],  # 膳食
    [-0.45, 0.80, -0.20, 0.40, 0.15, -0.65, 0.35],  # 综合
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],     # 维持
])


def load_data():
    df = pd.read_csv(RAW_CSV)
    labels = pd.read_csv(LABELS_CSV)
    labels = labels[labels["model"] == labels["model"].iloc[0]]
    df = df.merge(labels[["student_id", "rule_class_name", "HI_slope"]], on="student_id")
    rule = df["rule_class_name"].fillna("")
    high_risk = df[rule.str.contains("低水平") | rule.str.contains("退化")].copy()
    high_risk = high_risk[high_risk["HI_g4"].notna()].copy()
    high_risk["g4_decline"] = (high_risk["HI_g4"] < high_risk["HI_g2"]).astype(int)
    return high_risk


def build_features(df):
    feats = {}
    for m in METRICS:
        feats[f"{m}_g2"] = df[f"{m}_g2"]
        feats[f"{m}_slope"] = df[f"{m}_g2"] - df[f"{m}_g1"]
        feats[f"{m}_avg"] = (df[f"{m}_g1"] + df[f"{m}_g2"]) / 2
    feats["gender"] = (df["gender"] == "男").astype(int)
    for m in METRICS:
        feats[f"{m}_sx"] = feats[f"{m}_slope"] * feats[f"{m}_avg"]
    decline_count = np.zeros(len(df))
    for m in METRICS:
        if HIGHER_BETTER[m]:
            decline_count += (feats[f"{m}_slope"] < 0).astype(int)
        else:
            decline_count += (feats[f"{m}_slope"] > 0).astype(int)
    feats["n_decline"] = decline_count
    feats["hi_slope"] = df["HI_slope"]
    return pd.DataFrame(feats).fillna(pd.DataFrame(feats).median())


def train_m2(X, y):
    """训练 M2 退化预测器"""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    model.fit(X_s, y)
    return model, scaler


def predict_decline_prob(model, scaler, features_std):
    """M2 预测退化概率（输入已是标准化特征）"""
    return model.predict_proba(features_std.reshape(1, -1))[0, 1]


class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        return self.actor(h), self.critic(h)


def run_ppo(m2_model, scaler, X_train_raw, n_episodes=500):
    """PPO 训练：奖励 = M2 预测的退化概率下降。特征先标准化。"""
    X_train = scaler.transform(X_train_raw)
    state_dim = X_train.shape[1]
    policy = PPOPolicy(state_dim).float()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    X_tensor = torch.FloatTensor(X_train)

    history = {"reward": [], "action_counts": [0,0,0,0]}

    for ep in range(n_episodes):
        idx = np.random.randint(len(X_tensor))
        state = X_tensor[idx]

        with torch.no_grad():
            init_prob = predict_decline_prob(m2_model, scaler, state.numpy())

        logits, _ = policy(state.unsqueeze(0))
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        new_state = state.clone()
        action_idx = action.item()
        effect = EFFECT_MATRIX[action_idx]
        # 作用在标准化后的前7维特征上
        for i in range(7):
            new_state[i] += effect[i]

        new_prob = predict_decline_prob(m2_model, scaler, new_state.numpy())
        reward = init_prob - new_prob

        # PPO loss
        _, value = policy(state.unsqueeze(0))
        advantage = reward - value.item()
        loss = -log_prob * advantage + 0.5 * advantage ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["reward"].append(reward)
        history["action_counts"][action_idx] += 1

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(history["reward"][-100:])
            print(f"  Episode {ep+1}/{n_episodes}: avg reward={avg_r:.4f}")

    return policy, history


def main():
    print("=" * 60)
    print("M2→M3 漏斗：M2 预测驱动 PPO")
    print("=" * 60)

    df = load_data()
    X = build_features(df)
    y = df["g4_decline"].values
    print(f"\n高风险学生: {len(df)}, 特征: {X.shape[1]}维")

    # 训练 M2
    print("\n[1/3] 训练 M2 退化预测器...")
    m2_model, scaler = train_m2(X.values, y)
    print(f"  M2 train acc: {m2_model.score(scaler.transform(X.values), y):.3f}")

    # 用 M2 筛选"高概率退化"学生（P>0.7）
    all_probs = m2_model.predict_proba(scaler.transform(X.values))[:, 1]
    very_high = all_probs > 0.7
    print(f"\n[2/3] M2 筛出极高风险 (P>0.7): {very_high.sum()} 人")

    # PPO 在极高风险学生上训练
    print("\n[3/3] PPO 训练（奖励=M2退化概率下降）...")
    policy, hist = run_ppo(m2_model, scaler, X.values[very_high], n_episodes=500)

    # 评估：对比 4 种动作的平均奖励
    print("\n=== 4 种动作平均奖励 ===")
    rewards_per_action = {i: [] for i in range(4)}
    very_high_X_std = scaler.transform(X.values[very_high])
    for i in range(min(200, len(very_high_X_std))):
        state = very_high_X_std[i]
        init_p = predict_decline_prob(m2_model, scaler, state)
        for a in range(4):
            ns = state.copy()
            for j in range(7):
                ns[j] += EFFECT_MATRIX[a][j]
            new_p = predict_decline_prob(m2_model, scaler, ns)
            rewards_per_action[a].append(init_p - new_p)

    for a in range(4):
        r = np.mean(rewards_per_action[a])
        print(f"  {ACTIONS[a]}: reward = {r:.4f}")

    best_action = max(range(4), key=lambda a: np.mean(rewards_per_action[a]))
    print(f"\n最优动作: {ACTIONS[best_action]} (reward={np.mean(rewards_per_action[best_action]):.4f})")

    # 保存
    result = {
        "high_risk_n": int(len(df)),
        "very_high_risk_n": int(very_high.sum()),
        "m2_train_acc": float(m2_model.score(scaler.transform(X.values), y)),
        "action_rewards": {ACTIONS[a]: float(np.mean(rewards_per_action[a])) for a in range(4)},
        "best_action": ACTIONS[best_action],
    }
    (OUT_DIR / "m3_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    summary = f"""# M2→M3 漏斗结果

## M2 筛选
- 高风险学生: {len(df)}
- 极高风险 (P(退化)>0.7): {very_high.sum()}

## M3 PPO（奖励 = M2预测退化概率下降）
| 动作 | 平均奖励 |
|------|---------|
"""
    for a in range(4):
        summary += f"| {ACTIONS[a]} | {np.mean(rewards_per_action[a]):.4f} |\n"
    summary += f"\n最优动作: **{ACTIONS[best_action]}**\n"
    (OUT_DIR / "summary.md").write_text(summary)
    print(f"\n保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
