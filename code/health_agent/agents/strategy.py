"""Strategy Agent: M3 PPO 规则版 + GraphRAG 图推理

1. 根据学生状态选干预大类（运动/膳食/综合/维持）；
2. 调用 GraphRAG 做细粒度多约束推理：弱指标→效果边选动作→禁忌边过滤→来源边，
   输出动作排序与自然语言推理链，供 writer 组织可解释的四步回复。
"""
from state import AgentState
from agents.graph_reasoning import graph_reason

ACTIONS = ["运动干预为主", "膳食干预为主", "运动+膳食综合", "维持现状"]


def strategy_node(state: AgentState) -> dict:
    """根据学生数据选干预策略，并做 GraphRAG 图推理。"""
    d = state.get("student_data", {})
    trace_entry = {"node": "strategy"}

    bmi = d.get("bmi", 22)
    vc = d.get("vc", 3000)
    endurance = d.get("endurance", 270)
    strength = d.get("str", 15)

    # 规则（对应 M3 PPO 学到的策略）
    bmi_bad = bmi > 25 or bmi < 18
    cardio_bad = vc < 2800 or endurance > 280
    strength_bad = strength < 10

    if bmi_bad and cardio_bad:
        action = "运动+膳食综合"
    elif bmi_bad:
        action = "膳食干预为主"
    elif cardio_bad or strength_bad:
        action = "运动干预为主"
    else:
        action = "维持现状"

    # GraphRAG 图推理：弱指标 + BMI + 用户问题中的禁忌约束
    weak_cn = state.get("weak_indicator") or "肺活量"
    user_query = state.get("user_query", "")
    bmi_val = bmi if isinstance(bmi, (int, float)) else None
    graph = graph_reason(weak_cn, bmi=bmi_val, query=user_query)

    trace_entry["action"] = action
    trace_entry["graph_primary"] = graph["primary"]
    trace_entry["graph_excluded"] = [
        r["action"] for r in graph["ranked"] if r["excluded"]]
    trace = state.get("trace", []) + [trace_entry]
    return {
        "strategy": action,
        "graph_reasoning": graph,
        "reasoning_chain": graph["reasoning_chain"],
        "graph_citations": graph["citations"],
        "trace": trace,
    }
