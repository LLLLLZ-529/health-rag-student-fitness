"""
training/trainer.py — PPO 训练循环与评估
"""
from __future__ import annotations
import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from environment.simulator import HealthInterventionEnv, sample_student_from_data, StudentState, ACTIONS, INDICATORS
from agents.base import BaseAgent, RandomAgent, RuleAgent, BanditAgent, GreedyBestActionAgent
from agents.rl_agent import RLAgent


def run_episode(
    env: HealthInterventionEnv,
    agent: BaseAgent,
    initial_state: Optional[np.ndarray] = None,
    gender: str = "男",
    grade: int = 1,
    render: bool = False,
) -> Dict:
    """
    运行一个episode

    Returns:
        包含总奖励、每步信息、最终状态的字典
    """
    state = env.reset(initial_state, gender, grade)
    agent.reset()

    total_reward = 0.0
    step_infos = []
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)

        total_reward += reward
        step_infos.append(info)
        state = next_state

        if render:
            print(f"  Step {len(step_infos)}: {info['action']:25s} "
                  f"ΔHI={info['delta_hi']:+.4f} reward={reward:+.4f} "
                  f"HI={info['hi_total']:.4f} weakest={info['weakest']}")

    return {
        "total_reward": total_reward,
        "steps": step_infos,
        "final_state": state,
        "initial_hi": state.history[0] @ np.array([0.15, 0.15, 0.20, 0.10, 0.10, 0.20, 0.10]),
        "final_hi": state.hi_total,
        "delta_hi_total": state.hi_total - (state.history[0] @ np.array([0.15, 0.15, 0.20, 0.10, 0.10, 0.20, 0.10])),
    }


def evaluate_agent(
    agent: BaseAgent,
    env: HealthInterventionEnv,
    n_episodes: int = 100,
    seed: int = 42,
) -> Dict:
    """
    评估Agent性能

    Returns:
        平均总奖励、平均ΔHI、各指标改善、动作分布等
    """
    was_training = getattr(agent, 'training', False)
    if hasattr(agent, 'eval'):
        agent.eval()

    np.random.seed(seed)
    results = []
    action_counts = np.zeros(len(ACTIONS))

    for i in range(n_episodes):
        students = sample_student_from_data(1, seed=seed + i)
        init_state, gender, grade = students[0]
        result = run_episode(env, agent, init_state, gender, grade)
        results.append(result)
        for step in result["steps"]:
            action_counts[step["action_idx"]] += 1

    if hasattr(agent, 'train') and was_training:
        agent.train()

    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_delta_hi = np.mean([r["delta_hi_total"] for r in results])
    std_delta_hi = np.std([r["delta_hi_total"] for r in results])

    # 计算各指标平均改善
    indicator_deltas = np.zeros(7)
    for r in results:
        indicator_deltas += r["final_state"].indicators - r["final_state"].history[0]
    indicator_deltas /= len(results)

    return {
        "avg_reward": avg_reward,
        "avg_delta_hi": avg_delta_hi,
        "std_delta_hi": std_delta_hi,
        "indicator_deltas": dict(zip(INDICATORS, indicator_deltas)),
        "action_distribution": dict(zip(ACTIONS, action_counts / action_counts.sum())),
        "n_episodes": n_episodes,
        "success_rate": float(np.mean([r["delta_hi_total"] > 0 for r in results])),
    }


def train_ppo(
    agent: RLAgent,
    env: HealthInterventionEnv,
    total_timesteps: int = 50000,
    eval_interval: int = 5000,
    n_eval_episodes: int = 100,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    PPO 训练循环

    Args:
        agent: RL Agent
        env: 环境
        total_timesteps: 总训练步数
        eval_interval: 评估间隔
        n_eval_episodes: 评估episode数
        save_dir: 保存目录
        verbose: 是否打印日志

    Returns:
        训练历史
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    history = {
        "timesteps": [],
        "eval_reward": [],
        "eval_delta_hi": [],
        "train_policy_loss": [],
        "train_value_loss": [],
        "train_entropy": [],
    }

    agent.train()
    timesteps = 0
    episode_count = 0
    start_time = time.time()
    initial_entropy = agent.entropy_coef

    while timesteps < total_timesteps:
        # 熵衰减：从initial_entropy线性衰减到0.001
        progress = min(timesteps / total_timesteps, 1.0)
        current_entropy = initial_entropy * (1 - progress) + 0.001 * progress
        agent.set_entropy_coef(current_entropy)

        # 收集一个episode
        students = sample_student_from_data(1, seed=episode_count)
        init_state, gender, grade = students[0]
        result = run_episode(env, agent, init_state, gender, grade)
        timesteps += len(result["steps"])
        episode_count += 1

        # 每收集一定步数后更新
        if len(agent.states) >= 2048 or timesteps >= total_timesteps:
            train_metrics = agent.train_step(n_epochs=4, batch_size=64)
            if verbose and train_metrics:
                elapsed = time.time() - start_time
                print(f"[Step {timesteps:6d}/{total_timesteps}] "
                      f"policy_loss={train_metrics['policy_loss']:.4f} "
                      f"value_loss={train_metrics['value_loss']:.4f} "
                      f"entropy={train_metrics['entropy']:.4f} "
                      f"({elapsed:.0f}s)")
                history["train_policy_loss"].append(train_metrics["policy_loss"])
                history["train_value_loss"].append(train_metrics["value_loss"])
                history["train_entropy"].append(train_metrics["entropy"])

        # 定期评估
        if timesteps >= eval_interval * (len(history["timesteps"]) + 1):
            eval_result = evaluate_agent(agent, env, n_eval_episodes)
            history["timesteps"].append(timesteps)
            history["eval_reward"].append(eval_result["avg_reward"])
            history["eval_delta_hi"].append(eval_result["avg_delta_hi"])

            if verbose:
                print(f"  [Eval @ {timesteps}] avg_reward={eval_result['avg_reward']:.4f} "
                      f"avg_delta_hi={eval_result['avg_delta_hi']:+.4f} "
                      f"success_rate={eval_result['success_rate']:.2%}")

            if save_dir:
                agent.save(f"{save_dir}/ppo_step_{timesteps}.pt")

    # 最终评估
    final_eval = evaluate_agent(agent, env, n_eval_episodes * 2)
    history["final_eval"] = final_eval

    if save_dir:
        agent.save(f"{save_dir}/ppo_final.pt")
        with open(f"{save_dir}/training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

    if verbose:
        print(f"\n训练完成！总耗时 {time.time()-start_time:.0f}s")
        print(f"最终评估: avg_reward={final_eval['avg_reward']:.4f} "
              f"avg_delta_hi={final_eval['avg_delta_hi']:+.4f} "
              f"success_rate={final_eval['success_rate']:.2%}")

    return history


def compare_agents(
    agents: List[BaseAgent],
    env: HealthInterventionEnv,
    n_episodes: int = 200,
    seed: int = 42,
) -> Dict:
    """
    对比多个Agent的性能

    Returns:
        各Agent的评估结果字典
    """
    results = {}
    for agent in agents:
        print(f"\n评估 {agent.name} ...")
        result = evaluate_agent(agent, env, n_episodes, seed)
        results[agent.name] = result
        print(f"  avg_reward={result['avg_reward']:.4f}  "
              f"avg_delta_hi={result['avg_delta_hi']:+.4f}±{result['std_delta_hi']:.4f}  "
              f"success_rate={result['success_rate']:.2%}")

    return results
