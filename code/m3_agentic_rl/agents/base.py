"""
agents/base.py — Agent 基类与基线 Agent
"""
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from environment.simulator import StudentState, ACTIONS, INDICATORS, EFFECT_MATRIX, WEIGHTS


class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, state: StudentState, context: Optional[Dict] = None) -> int:
        """选择动作"""
        pass

    def update(self, state: StudentState, action: int, reward: float,
               next_state: StudentState, done: bool):
        """更新策略（在线学习用，默认不做任何事）"""
        pass

    def reset(self):
        """重置Agent状态"""
        pass


class RandomAgent(BaseAgent):
    """随机选择动作（基线）"""

    def __init__(self):
        super().__init__("Random")

    def select_action(self, state: StudentState, context=None) -> int:
        return np.random.randint(len(ACTIONS))


class RuleAgent(BaseAgent):
    """
    规则Agent（基线）：针对最弱指标选择对应训练
    指标→动作映射：
    - bmi → bmi_management
    - vital_capacity → cardio_training
    - sprint_50m → power_training
    - standing_long_jump → power_training
    - sit_and_reach → flexibility_training
    - endurance_run → endurance_training
    - strength → strength_training
    """

    INDICATOR_TO_ACTION = {
        0: 5,  # bmi → bmi_management
        1: 4,  # vital_capacity → cardio_training
        2: 3,  # sprint_50m → power_training
        3: 3,  # standing_long_jump → power_training
        4: 2,  # sit_and_reach → flexibility_training
        5: 0,  # endurance_run → endurance_training
        6: 1,  # strength → strength_training
    }

    def __init__(self, rest_every: int = 3):
        super().__init__("Rule")
        self.rest_every = rest_every
        self.step_count = 0

    def select_action(self, state: StudentState, context=None) -> int:
        self.step_count += 1
        # 每N轮休息一次，防止过度训练
        if self.step_count % self.rest_every == 0:
            return 7  # rest_recovery
        # 针对最弱指标
        weakest = state.weakest_idx
        return self.INDICATOR_TO_ACTION[weakest]

    def reset(self):
        self.step_count = 0


class BanditAgent(BaseAgent):
    """
    LinUCB Contextual Bandit（当前M3方法）
    上下文：学生状态（7维指标）+ 轮数
    """

    def __init__(self, n_actions: int = 8, context_dim: int = 8, alpha: float = 1.0):
        super().__init__("Bandit_LinUCB")
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = [np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]

    def _get_context(self, state: StudentState) -> np.ndarray:
        ctx = np.concatenate([state.indicators, [self.step_count / 5.0]])
        return ctx

    def select_action(self, state: StudentState, context=None) -> int:
        ctx = self._get_context(state)
        ucb_values = []
        for a in range(self.n_actions):
            theta = np.linalg.solve(self.A[a], self.b[a])
            uncertainty = self.alpha * np.sqrt(ctx @ np.linalg.solve(self.A[a], ctx))
            ucb = ctx @ theta + uncertainty
            ucb_values.append(ucb)
        return int(np.argmax(ucb_values))

    def update(self, state, action, reward, next_state, done):
        ctx = self._get_context(state)
        self.A[action] += np.outer(ctx, ctx)
        self.b[action] += reward * ctx

    def reset(self):
        self.A = [np.eye(self.context_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_actions)]
        self.step_count = 0


class GreedyBestActionAgent(BaseAgent):
    """
    贪心最优Agent（上界）：直接选对当前状态预期综合HI改善最大的动作
    知道真实效果矩阵和个体响应系数，是理论上界
    """

    def __init__(self):
        super().__init__("Greedy_Oracle")

    def select_action(self, state: StudentState, context=None) -> int:
        best_action = 0
        best_score = -float("inf")
        for a in range(len(ACTIONS)):
            # 计算该动作的预期效果（考虑个体响应）
            base_effect = EFFECT_MATRIX[a]
            response_factor = 1.0 - state.indicators * 0.15
            response_factor = np.clip(response_factor, 0.5, 1.5)
            expected_delta = base_effect * response_factor
            # 综合HI改善 = 加权和
            expected_hi_improvement = np.dot(expected_delta, WEIGHTS)
            # 最弱指标改善
            weakest_idx = state.weakest_idx
            weakest_improvement = expected_delta[weakest_idx]
            # 总分
            score = expected_hi_improvement + 0.5 * weakest_improvement
            if score > best_score:
                best_score = score
                best_action = a
        return best_action
