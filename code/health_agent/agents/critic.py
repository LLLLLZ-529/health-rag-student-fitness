"""Critic Agent: 规则校验处方（指标匹配 + 安全剂量 + 禁忌症）"""
from state import AgentState
from config import INDICATOR_KEYWORDS, MAX_REVISION_ROUNDS

def critic_node(state: AgentState) -> dict:
    weak = state.get("weak_indicator", "")
    prescription = state.get("draft_prescription", "")
    bmi = state.get("student_data", {}).get("bmi", 22)
    revision_count = state.get("revision_count", 0)
    errors = []

    # 1. 指标匹配：处方必须提到与最弱指标相关的训练词
    keywords = INDICATOR_KEYWORDS.get(weak, [])
    if keywords and weak:
        matched = any(kw in prescription for kw in keywords)
        if not matched:
            errors.append(f"张冠李戴：最弱指标是{weak}，但处方未提及相关训练方法")

    # 2. 安全剂量：不能推荐极端剂量
    unsafe_patterns = ["每天100个", "每天10公里", "练到力竭", "屏气1分钟", "越多越好", "不需要恢复"]
    for pat in unsafe_patterns:
        if pat in prescription:
            errors.append(f"不安全：处方包含极端建议（{pat}）")

    # 3. 禁忌症：BMI 偏高不应大重量
    if bmi > 26 and any(kw in prescription for kw in ["大重量", "硬拉", "深蹲", "卧推"]):
        errors.append("禁忌症：BMI偏高不应推荐大重量力量训练")

    # 4. 长度检查：不能太短（敷衍）
    if len(prescription) < 30:
        errors.append("处方过短，建议不够具体")

    passed = len(errors) == 0
    trace_entry = {
        "node": "critic", "passed": passed,
        "errors": errors, "revision": revision_count,
    }
    trace = state.get("trace", []) + [trace_entry]

    return {
        "critic_pass": passed,
        "critic_errors": errors,
        "trace": trace,
    }

def should_revise(state: AgentState) -> str:
    """条件边：Critic 不通过且未超轮次 → 回 Writer；否则结束"""
    if not state.get("critic_pass", False) and state.get("revision_count", 0) < MAX_REVISION_ROUNDS:
        return "revise"
    return "accept"
