"""Strategy Agent: M3 PPO 规则版，根据学生状态选干预方案"""
from state import AgentState

ACTIONS = ["运动干预为主", "膳食干预为主", "运动+膳食综合", "维持现状"]

def strategy_node(state: AgentState) -> dict:
    """根据学生数据选干预策略（模拟 PPO 决策）"""
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

    trace_entry["action"] = action
    trace = state.get("trace", []) + [trace_entry]
    return {"strategy": action, "trace": trace}
