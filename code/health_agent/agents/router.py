"""Router Agent: 根据用户查询判断任务类型和关心的指标"""
from state import AgentState

# 指标关键词
INDICATOR_KEYWORDS = {
    "肺活量": ["肺活量", "心肺", "呼吸", "喘"],
    "力量":   ["力量", "肌肉", "力气", "引体", "俯卧撑", "深蹲", "增肌"],
    "耐力":   ["耐力", "跑不动", "长跑", "坚持不下来"],
    "柔韧":   ["柔韧", "拉伸", "劈叉", "压腿", "弯腰"],
    "速度":   ["速度", "跑不快", "50米", "短跑", "冲刺"],
    "BMI":    ["体重", "胖", "减肥", "bmi", "瘦", "增重"],
}

def router_node(state: AgentState) -> dict:
    q = state["user_query"].lower()
    trace_entry = {"node": "router", "input": state["user_query"][:80]}

    # 任务类型
    if any(k in q for k in ["预测", "大四", "未来", "会不会", "退化", "发展"]):
        task = "predict"
    elif any(k in q for k in ["分类", "属于", "哪类", "什么水平", "几级", "等级"]):
        task = "diagnose"
    else:
        task = "recommend"

    # 识别用户关心的指标
    focus = None
    for ind, kws in INDICATOR_KEYWORDS.items():
        if any(kw in q for kw in kws):
            focus = ind
            break

    trace_entry["output"] = task
    if focus:
        trace_entry["focus"] = focus
    trace = state.get("trace", []) + [trace_entry]
    return {"task_type": task, "user_focus": focus, "trace": trace}
