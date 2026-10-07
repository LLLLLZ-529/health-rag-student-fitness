#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_m3_agentic_rl.py — M3 L3: Agentic RL 多轮健康推荐系统

完整架构：
  1. 干预效果模拟器（环境）：基于运动科学规则，预测推荐对学生HI的影响
  2. RAG 检索工具：从2674 chunk知识库检索相关建议
  3. Agent 框架：神经网络策略（可RL优化）+ 规则基线 + LLM Agent接口
  4. PPO/GRPO 训练：最大化多轮累积HI改善
  5. 对比实验：Agentic RL vs 单步Bandit vs 规则 vs 随机

奖励设计（用HI分数，不需要学生反馈）：
  reward = w1 * ΔHI_total + w2 * Δ最弱指标 - w3 * 轮数惩罚
"""
from __future__ import annotations
import json, os, sys, time, warnings, math
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
CHUNKS_DIR = ROOT / "chunks" / "chunks"
OUT_DIR = Path(__file__).parent / "outputs" / "m3_agentic_rl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
METRICS = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
           "sit_and_reach", "endurance_run_sec", "strength"]
METRIC_NAMES = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
    "endurance_run_sec": "耐力跑", "strength": "力量",
}
# 国标权重
METRIC_WEIGHTS = np.array([0.15, 0.15, 0.20, 0.10, 0.10, 0.20, 0.10])
# 4种干预策略
ACTIONS = ["运动干预为主", "膳食干预为主", "运动+膳食综合", "维持现状+监测"]
N_ACTIONS = len(ACTIONS)

# ============================================================
# 1. 干预效果模拟器
# ============================================================

class InterventionSimulator:
    """
    基于运动科学规则的干预效果模拟器。
    
    每个干预策略对7个体测指标有不同的改善效果：
    - 运动干预：主要改善耐力、力量、肺活量、跳远
    - 膳食干预：主要改善BMI、体前屈（柔韧性间接相关）
    - 综合干预：均衡改善所有指标
    - 维持现状：自然变化（微小波动）
    
    效果受以下因素调节：
    - 基线水平：越差的指标改善空间越大（天花板效应反向）
    - 边际递减：连续相同干预效果递减
    - 个体差异：每人有随机的干预响应系数
    - 噪声：随机波动
    """
    
    # 干预效果矩阵 [action, metric]：每轮的基础改善幅度（Z-score单位）
    # 正值=改善，负值=恶化
    EFFECT_MATRIX = np.array([
        # 运动干预为主
        [0.02, 0.08, 0.06, 0.07, 0.03, 0.10, 0.09],
        # 膳食干预为主
        [0.10, 0.02, 0.01, 0.02, 0.04, 0.02, 0.01],
        # 运动+膳食综合
        [0.06, 0.06, 0.04, 0.05, 0.04, 0.07, 0.06],
        # 维持现状+监测
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ])
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.max_rounds = 5
        
    def reset(self, student_metrics, student_id=None):
        """重置环境，返回初始状态"""
        self.state = student_metrics.copy()  # 7维Z-score
        self.student_id = student_id
        self.round = 0
        self.last_action = -1
        self.consecutive_same = 0
        # 个体干预响应系数（0.5-1.5）
        self.responsiveness = self.rng.uniform(0.5, 1.5, size=7)
        # 自然衰退趋势（不干预时的微小变化）
        self.natural_trend = self.rng.normal(0, 0.005, size=7)
        self.history = []
        return self._get_obs()
    
    def _get_obs(self):
        """获取观测：7指标 + 轮次 + 上一轮动作one-hot"""
        obs = np.zeros(7 + 1 + N_ACTIONS, dtype=np.float32)
        obs[:7] = self.state
        obs[7] = self.round / self.max_rounds
        if self.last_action >= 0:
            obs[8 + self.last_action] = 1.0
        return obs
    
    def _compute_hi(self, metrics):
        """计算综合HI（加权和）"""
        # BMI是倒U型，这里简化为Z-score加权
        return float(np.sum(metrics * METRIC_WEIGHTS))
    
    def step(self, action):
        """执行一步，返回 (obs, reward, done, info)"""
        prev_hi = self._compute_hi(self.state)
        prev_weakest = np.argmin(self.state)
        prev_weakest_val = self.state[prev_weakest]
        
        # 基础效果
        effect = self.EFFECT_MATRIX[action].copy()
        
        # 个体响应
        effect *= self.responsiveness
        
        # 边际递减：连续相同干预，效果打折扣
        if action == self.last_action:
            self.consecutive_same += 1
            discount = 0.7 ** self.consecutive_same
            effect *= discount
        else:
            self.consecutive_same = 0
        
        # 基线调节：指标越差，改善空间越大
        # state越低（越负），改善系数越大
        baseline_factor = np.clip(1.0 - self.state * 0.3, 0.3, 2.0)
        effect *= baseline_factor
        
        # 自然趋势（维持现状时只有自然趋势）
        if action == 3:  # 维持现状
            effect = self.natural_trend
        else:
            effect += self.natural_trend * 0.5
        
        # 噪声
        noise = self.rng.normal(0, 0.01, size=7)
        effect += noise
        
        # 更新状态
        self.state = np.clip(self.state + effect, -3.0, 3.0)
        self.round += 1
        self.last_action = action
        
        # 计算奖励
        new_hi = self._compute_hi(self.state)
        delta_hi = new_hi - prev_hi
        
        # 最弱指标改善
        new_weakest_val = self.state[prev_weakest]
        delta_weakest = new_weakest_val - prev_weakest_val
        
        # 综合奖励
        # 主奖励：HI改善
        reward = delta_hi * 5.0
        # 额外奖励：最弱指标改善（鼓励针对性干预）
        reward += delta_weakest * 2.0
        # 小惩罚：每轮消耗（鼓励高效）
        reward -= 0.02
        
        done = self.round >= self.max_rounds
        
        info = {
            "delta_hi": delta_hi,
            "delta_weakest": delta_weakest,
            "prev_hi": prev_hi,
            "new_hi": new_hi,
            "action": ACTIONS[action],
            "round": self.round,
            "weakest_metric": METRICS[prev_weakest],
        }
        self.history.append(info)
        
        return self._get_obs(), reward, done, info


# ============================================================
# 2. RAG 检索工具
# ============================================================

class RAGRetriever:
    """TF-IDF 检索器，复用现有知识库"""
    
    def __init__(self):
        self.chunks = []
        self.categories = []
        self._load_chunks()
        if self.chunks:
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
    
    def _load_chunks(self):
        for cat_dir in CHUNKS_DIR.iterdir():
            if cat_dir.is_dir():
                for f in list(cat_dir.glob("*.md")) + list(cat_dir.glob("*.txt")):
                    try:
                        text = f.read_text(encoding="utf-8")[:500]
                        if len(text.strip()) > 10:
                            self.chunks.append(text)
                            self.categories.append(cat_dir.name)
                    except:
                        pass
        print(f"  RAG加载 {len(self.chunks)} 个chunk")
    
    def retrieve(self, query, top_k=3):
        if not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({
                "text": self.chunks[idx],
                "category": self.categories[idx],
                "score": float(sims[idx]),
            })
        return results


# ============================================================
# 3. Agent 策略网络
# ============================================================

class PolicyNetwork(nn.Module):
    """神经网络策略：状态 -> 动作分布"""
    
    def __init__(self, obs_dim=12, hidden_dim=128, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits = self(torch.tensor(obs, dtype=torch.float32).to(DEVICE))
            if deterministic:
                action = torch.argmax(logits).item()
                return action, 0.0
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()


class ValueNetwork(nn.Module):
    """价值网络：状态 -> 状态价值"""
    
    def __init__(self, obs_dim=12, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# 4. PPO 训练
# ============================================================

class PPOTrainer:
    def __init__(self, obs_dim=12, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, epochs=4, batch_size=64):
        self.policy = PolicyNetwork(obs_dim).to(DEVICE)
        self.value = ValueNetwork(obs_dim).to(DEVICE)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
    
    def collect_rollout(self, simulator, students, n_episodes=50):
        """收集一批轨迹"""
        all_obs, all_actions, all_logprobs = [], [], []
        all_rewards, all_values, all_dones = [], [], []
        
        for ep in range(n_episodes):
            student = students[ep % len(students)]
            obs = simulator.reset(student)
            ep_obs, ep_acts, ep_lps, ep_rews, ep_vals, ep_dones = [], [], [], [], [], []
            
            done = False
            while not done:
                action, logprob = self.policy.get_action(obs)
                with torch.no_grad():
                    val = self.value(torch.tensor(obs, dtype=torch.float32).to(DEVICE)).item()
                next_obs, reward, done, info = simulator.step(action)
                
                ep_obs.append(obs)
                ep_acts.append(action)
                ep_lps.append(logprob)
                ep_rews.append(reward)
                ep_vals.append(val)
                ep_dones.append(done)
                obs = next_obs
            
            all_obs.extend(ep_obs)
            all_actions.extend(ep_acts)
            all_logprobs.extend(ep_lps)
            all_rewards.extend(ep_rews)
            all_values.extend(ep_vals)
            all_dones.extend(ep_dones)
        
        return (np.array(all_obs), np.array(all_actions), np.array(all_logprobs),
                np.array(all_rewards), np.array(all_values), np.array(all_dones))
    
    def compute_gae(self, rewards, values, dones):
        """计算 GAE 优势"""
        advantages = np.zeros(len(rewards))
        last_adv = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or dones[t]:
                next_val = 0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_adv = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_adv
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    def update(self, obs, actions, old_logprobs, advantages, returns):
        """PPO 更新"""
        obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        acts_t = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        old_lp_t = torch.tensor(old_logprobs, dtype=torch.float32).to(DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        ret_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        
        n = len(obs)
        indices = np.arange(n)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                b_obs, b_acts, b_old_lp = obs_t[batch_idx], acts_t[batch_idx], old_lp_t[batch_idx]
                b_adv, b_ret = adv_t[batch_idx], ret_t[batch_idx]
                
                # Policy loss
                logits = self.policy(b_obs)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_lp = dist.log_prob(b_acts)
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                self.policy_opt.zero_grad()
                policy_loss.backward()
                self.policy_opt.step()
                
                # Value loss
                value_pred = self.value(b_obs)
                value_loss = F.mse_loss(value_pred, b_ret)
                self.value_opt.zero_grad()
                value_loss.backward()
                self.value_opt.step()
        
        return policy_loss.item(), value_loss.item()


# ============================================================
# 5. 基线 Agent
# ============================================================

def random_agent(obs):
    return np.random.randint(N_ACTIONS)

def rule_agent(obs):
    """规则Agent：根据最弱指标选择干预"""
    metrics = obs[:7]
    weakest = np.argmin(metrics)
    weakest_name = METRICS[weakest]
    
    # BMI差 -> 膳食
    if weakest_name == "bmi":
        return 1  # 膳食
    # 耐力/力量/肺活量差 -> 运动
    elif weakest_name in ["endurance_run_sec", "strength", "vital_capacity", "standing_long_jump"]:
        return 0  # 运动
    # 其他 -> 综合
    else:
        return 2  # 综合

def single_step_bandit_agent(obs, class_idx):
    """单步Bandit：根据学生类别选预设最优策略"""
    OPTIMAL_ARM = {0: 2, 1: 0, 2: 3, 3: 2, 4: 0, 5: 3, 6: 1, 7: 3, 8: 3}
    return OPTIMAL_ARM.get(class_idx, 2)


# ============================================================
# 6. 学生数据加载
# ============================================================

def load_students(n=500, seed=42):
    """加载学生初始状态（7维Z-score，按性别×年级标准化）"""
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("", "_raw"))
    
    # 按性别×年级组内标准化（大一）
    for m in METRICS:
        col = f"{m}_g1"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # 组内标准化
        df[f"{col}_z"] = df.groupby("gender")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8))
        # BMI倒U型处理：距正常区间距离取负
        if m == "bmi":
            df[f"{col}_z"] = df[col].apply(
                lambda x: -abs(x - 21.2) / 5.0 if pd.notna(x) else 0.0)
    
    rng = np.random.RandomState(seed)
    df = df.sample(min(n, len(df)), random_state=seed)
    
    students = []
    for _, row in df.iterrows():
        metrics = []
        for m in METRICS:
            col = f"{m}_g1_z"
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            metrics.append(float(np.clip(val, -3, 3)))
        students.append(np.array(metrics, dtype=np.float32))
    
    print(f"  加载 {len(students)} 名学生初始状态（Z-score）")
    print(f"  平均HI: {np.mean([np.sum(s * METRIC_WEIGHTS) for s in students]):.4f}")
    return students


# ============================================================
# 7. 评估函数
# ============================================================

def evaluate_agent(agent_fn, simulator, students, n_episodes=100, label=""):
    """评估Agent，返回平均累积奖励和HI改善"""
    total_rewards = []
    total_hi_changes = []
    action_counts = defaultdict(int)
    
    for ep in range(n_episodes):
        student = students[ep % len(students)]
        obs = simulator.reset(student)
        ep_reward = 0
        done = False
        while not done:
            if label == "ppo":
                action, _ = agent_fn(obs, deterministic=True)
            elif label == "bandit":
                # bandit需要class_idx，这里用HI水平近似
                hi = np.sum(obs[:7] * METRIC_WEIGHTS)
                if hi < -0.3:
                    cls = 0
                elif hi < 0.3:
                    cls = 3
                else:
                    cls = 6
                action = agent_fn(obs, cls)
            else:
                action = agent_fn(obs)
            obs, reward, done, info = simulator.step(action)
            ep_reward += reward
            action_counts[info["action"]] += 1
        
        total_rewards.append(ep_reward)
        total_hi_changes.append(info["new_hi"] - info["prev_hi"] if info["round"] == simulator.max_rounds else 0)
    
    # 计算最终HI变化
    final_hi_changes = []
    for ep in range(min(n_episodes, len(students))):
        student = students[ep % len(students)]
        obs = simulator.reset(student)
        done = False
        init_hi = np.sum(obs[:7] * METRIC_WEIGHTS)
        while not done:
            if label == "ppo":
                action, _ = agent_fn(obs, deterministic=True)
            elif label == "bandit":
                hi = np.sum(obs[:7] * METRIC_WEIGHTS)
                cls = 0 if hi < -0.3 else (3 if hi < 0.3 else 6)
                action = agent_fn(obs, cls)
            else:
                action = agent_fn(obs)
            obs, _, done, _ = simulator.step(action)
        final_hi = np.sum(obs[:7] * METRIC_WEIGHTS)
        final_hi_changes.append(final_hi - init_hi)
    
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_hi_change": np.mean(final_hi_changes),
        "std_hi_change": np.std(final_hi_changes),
        "action_dist": dict(action_counts),
    }


# ============================================================
# 8. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("M3 L3: Agentic RL 多轮健康推荐系统")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/5] 加载学生数据 ...")
    students = load_students(n=500, seed=42)
    simulator = InterventionSimulator(seed=42)
    
    # 初始化RAG
    print("\n[2/5] 初始化RAG检索器 ...")
    rag = RAGRetriever()
    
    # PPO训练
    print("\n[3/5] PPO 训练 ...")
    trainer = PPOTrainer(obs_dim=12, lr=3e-4, gamma=0.99)
    n_iterations = 30
    train_rewards = []
    
    for it in range(n_iterations):
        obs, acts, old_lps, rews, vals, dones = trainer.collect_rollout(
            simulator, students, n_episodes=50)
        advantages, returns = trainer.compute_gae(rews, vals, dones)
        p_loss, v_loss = trainer.update(obs, acts, old_lps, advantages, returns)
        avg_reward = np.mean(rews) * 5  # 每episode平均5步
        train_rewards.append(avg_reward)
        if (it + 1) % 5 == 0:
            print(f"  Iter {it+1:3d}/{n_iterations}: avg_reward={avg_reward:.4f}  "
                  f"p_loss={p_loss:.4f}  v_loss={v_loss:.4f}")
    
    # 保存模型
    torch.save(trainer.policy.state_dict(), OUT_DIR / "ppo_policy.pt")
    torch.save(trainer.value.state_dict(), OUT_DIR / "ppo_value.pt")
    print(f"  模型已保存到 {OUT_DIR}")
    
    # 对比评估
    print("\n[4/5] 对比评估（4种方法 × 100 episodes）...")
    
    results = {}
    
    print("\n  --- 随机Agent ---")
    results["随机"] = evaluate_agent(random_agent, simulator, students, n_episodes=100, label="random")
    print(f"    累积奖励: {results['随机']['mean_reward']:.4f} ± {results['随机']['std_reward']:.4f}")
    print(f"    HI改善: {results['随机']['mean_hi_change']:.4f} ± {results['随机']['std_hi_change']:.4f}")
    
    print("\n  --- 规则Agent ---")
    results["规则"] = evaluate_agent(rule_agent, simulator, students, n_episodes=100, label="rule")
    print(f"    累积奖励: {results['规则']['mean_reward']:.4f} ± {results['规则']['std_reward']:.4f}")
    print(f"    HI改善: {results['规则']['mean_hi_change']:.4f} ± {results['规则']['std_hi_change']:.4f}")
    
    print("\n  --- 单步Bandit（当前M3）---")
    results["单步Bandit"] = evaluate_agent(single_step_bandit_agent, simulator, students, n_episodes=100, label="bandit")
    print(f"    累积奖励: {results['单步Bandit']['mean_reward']:.4f} ± {results['单步Bandit']['std_reward']:.4f}")
    print(f"    HI改善: {results['单步Bandit']['mean_hi_change']:.4f} ± {results['单步Bandit']['std_hi_change']:.4f}")
    
    print("\n  --- Agentic RL (PPO) ---")
    results["Agentic_RL"] = evaluate_agent(trainer.policy.get_action, simulator, students, n_episodes=100, label="ppo")
    print(f"    累积奖励: {results['Agentic_RL']['mean_reward']:.4f} ± {results['Agentic_RL']['std_reward']:.4f}")
    print(f"    HI改善: {results['Agentic_RL']['mean_hi_change']:.4f} ± {results['Agentic_RL']['std_hi_change']:.4f}")
    
    # 保存结果
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "method": name,
            "mean_reward": res["mean_reward"],
            "std_reward": res["std_reward"],
            "mean_hi_change": res["mean_hi_change"],
            "std_hi_change": res["std_hi_change"],
            "action_dist": json.dumps(res["action_dist"], ensure_ascii=False),
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "comparison_results.csv", index=False)
    
    # 训练曲线
    pd.DataFrame({"iteration": range(1, n_iterations+1), "avg_reward": train_rewards}).to_csv(
        OUT_DIR / "training_curve.csv", index=False)
    
    # 案例分析
    print("\n[5/5] 生成案例分析 ...")
    case_studies = []
    for case_idx in range(5):
        student = students[case_idx]
        obs = simulator.reset(student, student_id=f"student_{case_idx}")
        init_hi = np.sum(obs[:7] * METRIC_WEIGHTS)
        weakest = METRICS[np.argmin(obs[:7])]
        
        rounds_log = []
        done = False
        while not done:
            action, _ = trainer.policy.get_action(obs, deterministic=True)
            obs, reward, done, info = simulator.step(action)
            # RAG检索
            query = f"{weakest} {ACTIONS[action]} 大学生体测"
            retrieved = rag.retrieve(query, top_k=1)
            rounds_log.append({
                "round": info["round"],
                "action": info["action"],
                "reward": round(reward, 4),
                "delta_hi": round(info["delta_hi"], 4),
                "hi": round(info["new_hi"], 4),
                "rag_top": retrieved[0]["text"][:80] if retrieved else "无",
            })
        
        final_hi = np.sum(obs[:7] * METRIC_WEIGHTS)
        case_studies.append({
            "case": case_idx + 1,
            "init_hi": round(init_hi, 4),
            "final_hi": round(final_hi, 4),
            "hi_change": round(final_hi - init_hi, 4),
            "weakest": weakest,
            "rounds": rounds_log,
        })
    
    with open(OUT_DIR / "case_studies.json", "w", encoding="utf-8") as f:
        json.dump(case_studies, f, ensure_ascii=False, indent=2)
    
    # 生成Markdown报告
    md = ["# M3 L3: Agentic RL 多轮健康推荐系统 - 实验结果\n"]
    md.append("## 1. 对比实验结果\n")
    md.append("| 方法 | 累积奖励 | HI改善 | 动作分布 |")
    md.append("|---|---|---|---|")
    for name, res in results.items():
        ad = res["action_dist"]
        ad_str = ", ".join([f"{k}:{v}" for k, v in sorted(ad.items())])
        md.append(f"| {name} | {res['mean_reward']:.4f} ± {res['std_reward']:.4f} | "
                  f"{res['mean_hi_change']:.4f} ± {res['std_hi_change']:.4f} | {ad_str} |")
    
    md.append("\n## 2. 训练曲线\n")
    md.append(f"- 训练迭代: {n_iterations}")
    md.append(f"- 最终平均奖励: {train_rewards[-1]:.4f}")
    md.append(f"- 初始平均奖励: {train_rewards[0]:.4f}")
    md.append(f"- 提升幅度: {(train_rewards[-1] - train_rewards[0]) / abs(train_rewards[0]) * 100:.1f}%")
    
    md.append("\n## 3. 案例分析\n")
    for cs in case_studies:
        md.append(f"### 案例 {cs['case']}（最弱指标: {cs['weakest']}）")
        md.append(f"- 初始HI: {cs['init_hi']}, 最终HI: {cs['final_hi']}, 改善: {cs['hi_change']}")
        md.append("| 轮次 | 动作 | 奖励 | ΔHI | HI | RAG检索Top1 |")
        md.append("|---|---|---|---|---|---|")
        for r in cs["rounds"]:
            md.append(f"| {r['round']} | {r['action']} | {r['reward']} | {r['delta_hi']} | {r['hi']} | {r['rag_top']} |")
        md.append("")
    
    with open(OUT_DIR / "agentic_rl_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    
    print(f"\n完成 ✅ 结果已保存到 {OUT_DIR}")
    print(f"  - comparison_results.csv")
    print(f"  - training_curve.csv")
    print(f"  - case_studies.json")
    print(f"  - agentic_rl_report.md")
    print(f"  - ppo_policy.pt / ppo_value.pt")


if __name__ == "__main__":
    sys.exit(main())
