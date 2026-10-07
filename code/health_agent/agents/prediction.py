"""Prediction Agent: 基于历史数据预测大四退化风险"""
from state import AgentState

def prediction_node(state: AgentState) -> dict:
    history = state.get("user_history") or []
    trace_entry = {"node": "prediction", "history_len": len(history)}

    # 简单规则：如果大一到大二 HI 下降 > 0.15，判为高风险
    decline_prob = 0.5
    if len(history) >= 2:
        hi_first = history[0].get("hi", 50)
        hi_last = history[-1].get("hi", 50)
        delta = hi_last - hi_first
        if delta < -0.2:
            decline_prob = 0.82
            risk = "高风险"
        elif delta < -0.1:
            decline_prob = 0.65
            risk = "中风险"
        else:
            decline_prob = 0.30
            risk = "低风险"
    else:
        risk = "数据不足"

    trace_entry.update({"prob": decline_prob, "risk": risk})
    trace = state.get("trace", []) + [trace_entry]
    return {
        "decline_risk": decline_prob >= 0.5,
        "decline_prob": decline_prob,
        "risk_level": risk,
        "trace": trace,
    }
