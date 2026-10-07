#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_agentic_rl.py — M3 Agentic RL 主实验

1. 训练 PPO Agent
2. 对比 5 种 Agent（Random / Rule / Bandit / RL / Oracle）
3. 生成案例分析
4. 保存结果
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# 添加项目路径
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from environment.simulator import (
    HealthInterventionEnv, sample_student_from_data,
    ACTIONS, INDICATORS, WEIGHTS, EFFECT_MATRIX,
)
from agents.base import RandomAgent, RuleAgent, BanditAgent, GreedyBestActionAgent
from agents.rl_agent import RLAgent
from training.trainer import train_ppo, evaluate_agent, compare_agents, run_episode

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if __import__("torch").backends.mps.is_available() else "cpu"


def main():
    print("=" * 70)
    print("M3 Agentic RL — 多轮健康干预推荐系统")
    print("=" * 70)

    # 创建环境
    env = HealthInterventionEnv(max_steps=5, noise_std=0.02)
    print(f"\n环境: max_steps={env.max_steps}, noise_std={env.noise_std}")
    print(f"动作空间: {len(ACTIONS)} 种推荐策略")
    print(f"状态空间: {len(INDICATORS)} 维指标 + 轮数")

    # ========== 阶段1: 训练 PPO Agent ==========
    print("\n" + "=" * 70)
    print("阶段 1: 训练 PPO Agent")
    print("=" * 70)

    rl_agent = RLAgent(
        state_dim=8,
        n_actions=8,
        hidden_dim=128,
        n_layers=2,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        device=DEVICE,
    )
    print(f"设备: {DEVICE}")
    print(f"策略网络: 8 → 128 → 128 → 8")
    print(f"总训练步数: 100000")
    print(f"熵衰减: 0.01 → 0.001")

    train_history = train_ppo(
        agent=rl_agent,
        env=env,
        total_timesteps=100000,
        eval_interval=10000,
        n_eval_episodes=100,
        save_dir=str(OUTPUT_DIR / "models"),
        verbose=True,
    )

    # ========== 阶段2: 多 Agent 对比 ==========
    print("\n" + "=" * 70)
    print("阶段 2: 多 Agent 对比评估")
    print("=" * 70)

    agents = [
        RandomAgent(),
        RuleAgent(rest_every=3),
        BanditAgent(n_actions=8, context_dim=8, alpha=1.0),
        rl_agent,
        GreedyBestActionAgent(),
    ]

    comparison = compare_agents(agents, env, n_episodes=200, seed=42)

    # 保存对比结果
    comparison_df = pd.DataFrame([
        {
            "agent": name,
            "avg_reward": res["avg_reward"],
            "avg_delta_hi": res["avg_delta_hi"],
            "std_delta_hi": res["std_delta_hi"],
            "success_rate": res["success_rate"],
            **{f"delta_{k}": v for k, v in res["indicator_deltas"].items()},
        }
        for name, res in comparison.items()
    ]).sort_values("avg_delta_hi", ascending=False)

    comparison_df.to_csv(OUTPUT_DIR / "agent_comparison.csv", index=False)
    print(f"\n对比结果已保存: {OUTPUT_DIR / 'agent_comparison.csv'}")

    # 动作分布
    action_dist_df = pd.DataFrame([
        {"agent": name, **res["action_distribution"]}
        for name, res in comparison.items()
    ])
    action_dist_df.to_csv(OUTPUT_DIR / "action_distribution.csv", index=False)

    # ========== 阶段3: 案例分析 ==========
    print("\n" + "=" * 70)
    print("阶段 3: 典型案例分析")
    print("=" * 70)

    # 选3个典型学生：低水平、中水平、高水平
    case_students = [
        ("低水平学生", np.array([-1.0, -0.8, -0.9, -0.7, -0.6, -1.0, -0.8]), "男", 1),
        ("中水平学生", np.array([0.0, 0.1, -0.2, 0.1, 0.0, -0.3, 0.1]), "女", 2),
        ("高水平学生", np.array([0.8, 0.9, 0.7, 1.0, 0.8, 0.6, 0.9]), "男", 3),
    ]

    case_results = []
    for case_name, init_state, gender, grade in case_students:
        print(f"\n--- {case_name} (初始HI={init_state @ WEIGHTS:.4f}) ---")
        for agent in [RuleAgent(rest_every=3), rl_agent, GreedyBestActionAgent()]:
            agent.reset()
            if hasattr(agent, 'eval'):
                agent.eval()
            result = run_episode(env, agent, init_state.copy(), gender, grade, render=True)
            case_results.append({
                "case": case_name,
                "agent": agent.name,
                "initial_hi": result["initial_hi"],
                "final_hi": result["final_hi"],
                "delta_hi": result["delta_hi_total"],
                "total_reward": result["total_reward"],
                "actions": " → ".join([s["action"] for s in result["steps"]]),
            })
            print(f"  {agent.name}: ΔHI={result['delta_hi_total']:+.4f}, reward={result['total_reward']:.4f}")

    case_df = pd.DataFrame(case_results)
    case_df.to_csv(OUTPUT_DIR / "case_studies.csv", index=False)

    # 生成案例分析报告
    with open(OUTPUT_DIR / "case_studies_report.md", "w") as f:
        f.write("# Agentic RL 案例分析报告\n\n")
        f.write("## 对比方法\n\n")
        f.write("- **Rule**: 针对最弱指标选择对应训练，每3轮休息一次\n")
        f.write("- **RL (PPO)**: 强化学习训练的策略网络\n")
        f.write("- **Oracle**: 贪心最优（理论上界，知道真实效果矩阵）\n\n")

        for case_name, _, _, _ in case_students:
            f.write(f"## {case_name}\n\n")
            case_data = case_df[case_df["case"] == case_name]
            for _, row in case_data.iterrows():
                f.write(f"### {row['agent']}\n\n")
                f.write(f"- 初始HI: {row['initial_hi']:.4f}\n")
                f.write(f"- 最终HI: {row['final_hi']:.4f}\n")
                f.write(f"- ΔHI: **{row['delta_hi']:+.4f}**\n")
                f.write(f"- 总奖励: {row['total_reward']:.4f}\n")
                f.write(f"- 动作序列: {row['actions']}\n\n")

    # ========== 阶段4: 训练曲线 ==========
    print("\n" + "=" * 70)
    print("阶段 4: 保存训练曲线")
    print("=" * 70)

    train_df = pd.DataFrame({
        "timesteps": train_history["timesteps"],
        "eval_reward": train_history["eval_reward"],
        "eval_delta_hi": train_history["eval_delta_hi"],
    })
    train_df.to_csv(OUTPUT_DIR / "training_curve.csv", index=False)

    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("实验总结")
    print("=" * 70)

    print("\n按 ΔHI 排名:")
    for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
        print(f"  {i}. {row['agent']:20s}  ΔHI={row['avg_delta_hi']:+.4f}±{row['std_delta_hi']:.4f}  "
              f"reward={row['avg_reward']:.4f}  success={row['success_rate']:.2%}")

    # RL vs Bandit 提升
    rl_delta = comparison_df[comparison_df["agent"] == "RL_PPO"]["avg_delta_hi"].values[0]
    bandit_delta = comparison_df[comparison_df["agent"] == "Bandit_LinUCB"]["avg_delta_hi"].values[0]
    rule_delta = comparison_df[comparison_df["agent"] == "Rule"]["avg_delta_hi"].values[0]
    random_delta = comparison_df[comparison_df["agent"] == "Random"]["avg_delta_hi"].values[0]

    print(f"\n关键对比:")
    print(f"  RL vs Random: {(rl_delta - random_delta):+.4f} ({(rl_delta/random_delta - 1)*100:+.1f}%)")
    print(f"  RL vs Rule:   {(rl_delta - rule_delta):+.4f} ({(rl_delta/rule_delta - 1)*100:+.1f}%)")
    print(f"  RL vs Bandit: {(rl_delta - bandit_delta):+.4f} ({(rl_delta/bandit_delta - 1)*100:+.1f}%)")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print(f"  - agent_comparison.csv (多Agent对比)")
    print(f"  - action_distribution.csv (动作分布)")
    print(f"  - case_studies.csv (案例分析)")
    print(f"  - case_studies_report.md (案例报告)")
    print(f"  - training_curve.csv (训练曲线)")
    print(f"  - models/ (PPO模型checkpoint)")


if __name__ == "__main__":
    main()
