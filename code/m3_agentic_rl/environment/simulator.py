"""
environment/simulator.py — 干预效果模拟器

基于运动科学常识构建的规则驱动模拟器，模拟学生在接受不同推荐策略后的HI变化。
学生状态：7项体测指标的标准化值
动作空间：8种推荐策略
奖励：HI综合改善 + 最弱指标改善 - 效率惩罚
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# 7项指标（与HI计算一致，已做方向校正，越大越好）
INDICATORS = [
    "bmi",            # BMI（倒U型，距[18.5,23.9]距离取负）
    "vital_capacity", # 肺活量
    "sprint_50m",     # 50米（取负，越快越好）
    "standing_long_jump",  # 立定跳远
    "sit_and_reach",  # 坐位体前屈
    "endurance_run",  # 耐力跑（取负，越快越好）
    "strength",       # 力量
]

# 国标权重
WEIGHTS = np.array([0.15, 0.15, 0.20, 0.10, 0.10, 0.20, 0.10])

# 8种推荐策略
ACTIONS = [
    "endurance_training",    # 耐力训练：跑步、游泳、骑行
    "strength_training",     # 力量训练：引体向上、俯卧撑、举重
    "flexibility_training",  # 柔韧训练：拉伸、瑜伽、体前屈
    "power_training",        # 爆发力训练：跳远、短跑、跳高
    "cardio_training",       # 心肺功能：肺活量训练、有氧间歇
    "bmi_management",        # BMI管理：饮食控制+有氧
    "comprehensive_training", # 综合训练：全身循环
    "rest_recovery",         # 休息恢复：避免过度训练
]

# 干预效果矩阵：每个动作对7个指标的基础效果（每轮ΔHI）
# 行=动作，列=指标
# 正值=改善，负值=恶化，基于运动科学常识
EFFECT_MATRIX = np.array([
    # bmi   vital  sprint  jump   reach  endur  strength
    [ 0.08,  0.12,  0.03,  0.05,  0.02,  0.15,  0.04],  # endurance_training
    [ 0.03,  0.04,  0.06,  0.10,  0.01,  0.02,  0.18],  # strength_training
    [ 0.02,  0.01,  0.00,  0.02,  0.18,  0.01,  0.00],  # flexibility_training
    [ 0.01,  0.03,  0.15,  0.16,  0.00,  0.04,  0.08],  # power_training
    [ 0.04,  0.18,  0.02,  0.03,  0.01,  0.10,  0.02],  # cardio_training
    [ 0.15,  0.05,  0.02,  0.03,  0.02,  0.08,  0.01],  # bmi_management
    [ 0.06,  0.08,  0.06,  0.07,  0.06,  0.08,  0.08],  # comprehensive_training
    [-0.02, -0.03, -0.02, -0.02, -0.02, -0.03, -0.02],  # rest_recovery（短期下降，长期防止过度训练）
])

# 个体差异：不同基础水平的学生对同一干预的响应不同
# 基础差的学生改善空间大，基础好的学生边际收益递减
def individual_response_factor(state: np.ndarray, action_idx: int) -> np.ndarray:
    """计算个体响应系数：基础越差，改善空间越大"""
    # state范围约[-2, 2]，映射到[0.5, 1.5]
    # 指标值越低（越差），改善系数越高
    base_factor = 1.0 - state * 0.15  # state=-2 → 1.3, state=2 → 0.7
    base_factor = np.clip(base_factor, 0.5, 1.5)
    return base_factor


@dataclass
class StudentState:
    """学生状态"""
    indicators: np.ndarray  # 7项指标的标准化HI值
    gender: str = "男"
    grade: int = 1
    history: List[np.ndarray] = field(default_factory=list)  # 历史状态
    action_history: List[int] = field(default_factory=list)  # 历史动作

    @property
    def hi_total(self) -> float:
        """综合HI总分"""
        return float(np.dot(self.indicators, WEIGHTS))

    @property
    def weakest_idx(self) -> int:
        """最弱指标索引"""
        return int(np.argmin(self.indicators))

    @property
    def weakest_name(self) -> str:
        return INDICATORS[self.weakest_idx]

    def copy(self) -> "StudentState":
        return StudentState(
            indicators=self.indicators.copy(),
            gender=self.gender,
            grade=self.grade,
            history=self.history.copy(),
            action_history=self.action_history.copy(),
        )


class HealthInterventionEnv:
    """
    健康干预模拟环境

    状态空间：7维指标向量 + 性别 + 年级
    动作空间：8种推荐策略（离散）
    奖励：ΔHI综合 + Δ最弱指标 - 轮数惩罚
    """

    def __init__(
        self,
        max_steps: int = 5,
        noise_std: float = 0.02,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.reward_weights = reward_weights or {
            "hi_total": 1.0,
            "weakest": 0.5,
            "step_penalty": 0.01,
            "overtraining_penalty": 0.15,
        }
        self.state: Optional[StudentState] = None
        self.step_count = 0

    def reset(self, initial_state: Optional[np.ndarray] = None,
              gender: str = "男", grade: int = 1) -> StudentState:
        """重置环境"""
        if initial_state is None:
            # 随机初始化：均值0，标准差0.5
            initial_state = np.random.randn(7) * 0.5
        self.state = StudentState(
            indicators=initial_state.copy(),
            gender=gender,
            grade=grade,
            history=[initial_state.copy()],
        )
        self.step_count = 0
        return self.state

    def step(self, action_idx: int) -> Tuple[StudentState, float, bool, Dict]:
        """
        执行一步干预

        Args:
            action_idx: 动作索引（0-7）

        Returns:
            new_state: 新状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        assert self.state is not None, "Call reset() first"
        assert 0 <= action_idx < len(ACTIONS), f"Invalid action: {action_idx}"

        old_hi = self.state.hi_total
        old_weakest = self.state.indicators[self.state.weakest_idx]

        # 计算干预效果
        base_effect = EFFECT_MATRIX[action_idx].copy()
        response_factor = individual_response_factor(self.state.indicators, action_idx)
        effect = base_effect * response_factor

        # 过度训练惩罚：连续3轮以上高强度训练（非休息），效果递减
        if len(self.state.action_history) >= 3:
            recent = self.state.action_history[-3:]
            if all(a != 7 for a in recent):  # 7 = rest_recovery
                effect *= 0.7  # 疲劳导致效果下降

        # 添加噪声
        noise = np.random.randn(7) * self.noise_std
        new_indicators = self.state.indicators + effect + noise

        # 限制范围
        new_indicators = np.clip(new_indicators, -2.0, 2.0)

        # 更新状态
        self.state.indicators = new_indicators
        self.state.history.append(new_indicators.copy())
        self.state.action_history.append(action_idx)
        self.step_count += 1

        # 计算奖励
        new_hi = self.state.hi_total
        new_weakest = self.state.indicators[self.state.weakest_idx]

        reward = (
            self.reward_weights["hi_total"] * (new_hi - old_hi)
            + self.reward_weights["weakest"] * (new_weakest - old_weakest)
            - self.reward_weights["step_penalty"]
        )

        # 过度训练额外惩罚
        if len(self.state.action_history) >= 4:
            recent = self.state.action_history[-4:]
            if all(a != 7 for a in recent):
                reward -= self.reward_weights["overtraining_penalty"]

        # 判断是否结束
        done = self.step_count >= self.max_steps

        info = {
            "action": ACTIONS[action_idx],
            "action_idx": action_idx,
            "delta_hi": new_hi - old_hi,
            "delta_weakest": new_weakest - old_weakest,
            "hi_total": new_hi,
            "weakest": self.state.weakest_name,
            "overtraining": len(self.state.action_history) >= 4 and all(a != 7 for a in self.state.action_history[-4:]),
        }

        return self.state, reward, done, info

    def get_action_names(self) -> List[str]:
        return ACTIONS

    def get_indicator_names(self) -> List[str]:
        return INDICATORS


def sample_student_from_data(n: int = 1, seed: int = None) -> List[Tuple[np.ndarray, str, int]]:
    """
    从真实数据分布中采样学生初始状态

    Returns:
        List of (indicators, gender, grade)
    """
    if seed is not None:
        np.random.seed(seed)

    # 基于真实数据的统计特征（从hi_wide.csv估计）
    # 均值和标准差（近似）
    means = np.array([-0.1, 0.05, -0.05, 0.0, 0.02, -0.08, 0.03])
    stds = np.array([0.4, 0.5, 0.45, 0.5, 0.45, 0.55, 0.5])

    students = []
    for _ in range(n):
        indicators = np.random.randn(7) * stds + means
        indicators = np.clip(indicators, -2.0, 2.0)
        gender = np.random.choice(["男", "女"], p=[0.55, 0.45])
        grade = np.random.randint(1, 4)
        students.append((indicators, gender, grade))
    return students
