"""Diagnosis Agent: 解析体测数据，计算 HI，输出 9 类标签和最弱指标"""
from state import AgentState

# 女生体测阈值（用于判断最弱指标）
THRESHOLDS_FEMALE = {
    "vc":      {"weak": 2800, "good": 3400},    # 肺活量 ml
    "str":     {"weak": 10,   "good": 25},       # 力量（仰卧起坐/引体）
    "endurance": {"weak": 270, "good": 220},     # 耐力跑秒（越少越好）
    "speed":   {"weak": 9.5,  "good": 8.0},      # 50m 秒
    "jump":    {"weak": 150,  "good": 190},      # 立定跳远 cm
    "flex":    {"weak": 5,    "good": 15},       # 坐位体前屈 cm
}

def _norm_score(value, weak, good, lower_better=False):
    """归一化到 0-100"""
    if lower_better:
        if value <= good: return 100
        if value >= weak: return 0
        return (weak - value) / (weak - good) * 100
    else:
        if value >= good: return 100
        if value <= weak: return 0
        return (value - weak) / (good - weak) * 100

def diagnosis_node(state: AgentState) -> dict:
    data = state["student_data"]
    trace_entry = {"node": "diagnosis", "input_keys": list(data.keys())}

    # 归一化各指标得分
    scores = {}
    if "vc" in data:
        t = THRESHOLDS_FEMALE["vc"]; scores["肺活量"] = _norm_score(data["vc"], t["weak"], t["good"])
    if "str" in data:
        t = THRESHOLDS_FEMALE["str"]; scores["力量"] = _norm_score(data["str"], t["weak"], t["good"])
    if "endurance" in data:
        t = THRESHOLDS_FEMALE["endurance"]; scores["耐力"] = _norm_score(data["endurance"], t["weak"], t["good"], lower_better=True)
    if "speed" in data:
        t = THRESHOLDS_FEMALE["speed"]; scores["速度"] = _norm_score(data["speed"], t["weak"], t["good"], lower_better=True)
    if "jump" in data:
        t = THRESHOLDS_FEMALE["jump"]; scores["跳远"] = _norm_score(data["jump"], t["weak"], t["good"])
    if "flex" in data:
        t = THRESHOLDS_FEMALE["flex"]; scores["柔韧"] = _norm_score(data["flex"], t["weak"], t["good"])

    # BMI 特殊处理：倒 U 型
    bmi = data.get("bmi", 22)
    if 18.5 <= bmi <= 23.9:
        bmi_score = 100
    elif bmi < 18.5:
        bmi_score = max(0, 100 - (18.5 - bmi) * 15)
    else:
        bmi_score = max(0, 100 - (bmi - 23.9) * 15)
    scores["BMI"] = bmi_score

    # 国标加权
    weights = {"BMI": .15, "肺活量": .15, "速度": .20, "跳远": .10, "柔韧": .10, "耐力": .20, "力量": .10}
    hi = sum(scores.get(k, 50) * w for k, w in weights.items())

    # 最弱指标（自动检测）
    auto_weak = min(scores, key=scores.get)

    # 如果用户明确问了某个指标，优先用用户问的
    user_focus = state.get("user_focus")
    weak = user_focus if user_focus else auto_weak

    # 9 类标签：水平分高/中/低，趋势分退化/平稳/改善
    if hi >= 80: level = "高水平"
    elif hi >= 60: level = "中水平"
    else: level = "低水平"

    # 趋势：从 history 算（如果有）
    history = state.get("user_history") or []
    trend = "平稳型"
    if len(history) >= 2:
        first = history[0].get("hi", hi)
        last = history[-1].get("hi", hi)
        delta = last - first
        if delta < -0.1: trend = "退化型"
        elif delta > 0.1: trend = "改善型"

    label = f"{level}-{trend}"
    trace_entry.update({"hi": round(hi, 1), "weak": weak, "label": label})
    trace = state.get("trace", []) + [trace_entry]

    return {
        "hi_score": round(hi, 1),
        "weak_indicator": weak,
        "level_label": label,
        "trend": trend,
        "trace": trace,
    }
