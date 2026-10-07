"""
agents/rl_agent.py — PPO 强化学习 Agent

策略网络：MLP，输入学生状态(8维)，输出8个动作的概率
价值网络：MLP，输入学生状态，输出状态价值
使用PPO算法训练，支持GAE优势估计
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Dict, List, Tuple
from environment.simulator import StudentState, ACTIONS
from agents.base import BaseAgent


class PolicyNetwork(nn.Module):
    """策略网络：状态 → 动作概率"""

    def __init__(self, state_dim: int = 8, n_actions: int = 8,
                 hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return logits


class ValueNetwork(nn.Module):
    """价值网络：状态 → 状态价值"""

    def __init__(self, state_dim: int = 8, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class RLAgent(BaseAgent):
    """
    PPO 强化学习 Agent

    支持：
    - 在线动作选择（带探索）
    - 离线策略更新（PPO clip）
    - GAE 优势估计
    - 学习率调度
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 8,
        hidden_dim: int = 128,
        n_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__("RL_PPO")
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.policy = PolicyNetwork(state_dim, n_actions, hidden_dim, n_layers).to(device)
        self.value = ValueNetwork(state_dim, hidden_dim, n_layers).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr,
        )

        # 轨迹存储
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

        self.training = True

    def _state_to_tensor(self, state: StudentState) -> torch.Tensor:
        """将学生状态转为tensor：7维指标 + 1维轮数"""
        step_norm = len(state.action_history) / 5.0
        arr = np.concatenate([state.indicators, [step_norm]]).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def select_action(self, state: StudentState, context=None) -> int:
        """选择动作（训练时带探索，评估时贪心）"""
        state_t = self._state_to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_t)
            dist = Categorical(logits=logits)
            if self.training:
                action = dist.sample()
            else:
                action = logits.argmax(dim=-1)
            log_prob = dist.log_prob(action)
            value = self.value(state_t)

        action_idx = int(action.item())

        if self.training:
            self.states.append(state_t.squeeze(0).cpu().numpy())
            self.actions.append(action_idx)
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())

        return action_idx

    def update(self, state, action, reward, next_state, done):
        """记录奖励和done标志"""
        if self.training:
            self.rewards.append(reward)
            self.dones.append(done)

    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """计算GAE优势和回报"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + np.array(self.values)
        return advantages, returns

    def train_step(self, n_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """
        PPO 更新步骤

        Returns:
            训练指标字典
        """
        if len(self.states) == 0:
            return {}

        # 计算GAE
        with torch.no_grad():
            last_state = torch.tensor(self.states[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
            next_value = self.value(last_state).item()

        advantages, returns = self.compute_gae(next_value)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转为tensor
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        n_samples = len(states_t)
        indices = np.arange(n_samples)

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                # 新策略
                logits = self.policy(b_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # PPO clip
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                values = self.value(b_states)
                value_loss = F.mse_loss(values, b_returns)

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "n_samples": n_samples,
        }

    def set_entropy_coef(self, coef: float):
        """设置熵系数（用于熵衰减）"""
        self.entropy_coef = coef

    def eval(self):
        """切换到评估模式（贪心选择）"""
        self.training = False
        self.policy.eval()
        self.value.eval()

    def train(self):
        """切换到训练模式"""
        self.training = True
        self.policy.train()
        self.value.train()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
