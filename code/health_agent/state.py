"""HealthAgent 状态定义（LangGraph State）"""
from typing import TypedDict, List, Optional, Dict, Any

class AgentState(TypedDict, total=False):
    # 输入
    user_query: str               # 用户原始查询
    student_data: Dict[str, Any]  # 学生体测数据
    user_history: Optional[List[Dict[str, Any]]]  # 历年数据（预测用）

    # Router 输出
    task_type: str                # diagnose / predict / recommend / audit
    user_focus: Optional[str]     # 用户问的指标（力量/肺活量/...）

    # Diagnosis 输出
    hi_score: Optional[float]     # 健康指数
    weak_indicator: Optional[str]  # 最弱指标
    level_label: Optional[str]    # 9类标签
    trend: Optional[str]          # 退化/平稳/改善

    # Prediction 输出
    decline_risk: Optional[bool]
    decline_prob: Optional[float]
    risk_level: Optional[str]

    # Retriever 输出
    retrieved_docs: List[str]     # top-k 检索到的处方

    # Strategy 输出（M3 PPO 决策）
    strategy: Optional[str]       # 运动/膳食/综合/维持

    # Writer 输出
    draft_prescription: str
    revision_count: int           # 已重写次数

    # Critic 输出
    critic_pass: bool
    critic_errors: List[str]

    # 最终输出
    final_response: str
    trace: List[Dict[str, Any]]   # 执行轨迹（供 reward/日志用）
