"""Graph Reasoning: 体测健康知识图谱多约束推理（轻量 GraphRAG）

在 strategy 节点前提供结构化先验：
  弱指标 → 遍历『动作→指标』效果边选动作 → 禁忌边过滤 → 来源边取引用，
  并输出自然语言推理链（graph-grounded CoT），供 writer 组织四步回复。

效果矩阵来自运动科学先验（与 m3_agentic_rl 模拟器一致），零 GPU 依赖。
"""
from typing import List, Dict, Optional

INDICATOR_CN = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
    "endurance_run": "耐力跑", "strength": "力量",
}
CN_TO_IND = {
    "bmi": "bmi", "BMI": "bmi", "体重": "bmi",
    "肺活量": "vital_capacity", "心肺": "vital_capacity",
    "50米": "sprint_50m", "50米跑": "sprint_50m", "速度": "sprint_50m", "冲刺": "sprint_50m",
    "立定跳远": "standing_long_jump", "跳远": "standing_long_jump",
    "坐位体前屈": "sit_and_reach", "体前屈": "sit_and_reach", "柔韧": "sit_and_reach",
    "耐力": "endurance_run", "耐力跑": "endurance_run",
    "力量": "strength",
}

ACTIONS = ["endurance_training", "strength_training", "flexibility_training",
           "power_training", "cardio_training", "bmi_management",
           "comprehensive_training", "rest_recovery"]
# 行=动作，列=指标，正值=改善
EFFECT_MATRIX = [
    [0.08, 0.12, 0.03, 0.05, 0.02, 0.15, 0.04],
    [0.03, 0.04, 0.06, 0.10, 0.01, 0.02, 0.18],
    [0.02, 0.01, 0.00, 0.02, 0.18, 0.01, 0.00],
    [0.01, 0.03, 0.15, 0.16, 0.00, 0.04, 0.08],
    [0.04, 0.18, 0.02, 0.03, 0.01, 0.10, 0.02],
    [0.15, 0.05, 0.02, 0.03, 0.02, 0.08, 0.01],
    [0.06, 0.08, 0.06, 0.07, 0.06, 0.08, 0.08],
    [-0.02, -0.03, -0.02, -0.02, -0.02, -0.03, -0.02],
]
ACTION_CN = {
    "endurance_training": "耐力训练", "strength_training": "力量训练",
    "flexibility_training": "柔韧训练", "power_training": "爆发力训练",
    "cardio_training": "心肺训练", "bmi_management": "体重管理",
    "comprehensive_training": "综合训练", "rest_recovery": "休息恢复",
}
ACTION_FORM = {
    "endurance_training": ("匀速慢跑、游泳、骑行", "游泳、骑行"),
    "strength_training": ("深蹲、俯卧撑、哑铃、平板支撑", "弹力带、自重慢速、水中训练"),
    "flexibility_training": ("静态拉伸、瑜伽、髋部拉伸", "静态拉伸、瑜伽"),
    "power_training": ("冲刺、蛙跳、跳箱、高抬腿", "低冲击有氧（替代）"),
    "cardio_training": ("慢跑、跳绳、间歇跑、有氧操", "游泳、快走、骑行、水中行走"),
    "bmi_management": ("快走、游泳、骑行配合饮食控制", "游泳、快走、水中有氧"),
    "comprehensive_training": ("全身循环、有氧+力量交替", "水中综合、低冲击循环"),
    "rest_recovery": ("主动恢复、拉伸、睡眠管理", "主动恢复、拉伸"),
}
ACTION_SOURCE = {
    "endurance_training": "运动处方/_全民健身指南__国家体育总局_OCR__chunk001.md",
    "strength_training": "运动处方/_全民健身指南__国家体育总局_OCR__chunk005.md",
    "flexibility_training": "运动处方/_中青年科学运动指南_解读__chunk001.md",
    "power_training": "运动处方/_全民健身指南__国家体育总局_OCR__chunk007.md",
    "cardio_training": "运动处方/_全民健身指南__国家体育总局_OCR__chunk001.md",
    "bmi_management": "膳食营养/成人肥胖食养指南_2024年版___chunk010.md",
    "comprehensive_training": "运动处方/_全年龄段人群科学健身指南来了_国家体育总局_OCR__chunk004.md",
    "rest_recovery": "运动处方/_中青年科学运动指南_解读__chunk001.md",
}


def _normalize_weak(weak_cn: str) -> List[str]:
    """中文弱指标 → 英文 key 列表，支持多个。"""
    out = []
    for k, v in CN_TO_IND.items():
        if k in weak_cn and v not in out:
            out.append(v)
    return out or ["vital_capacity"]


def _contraindications(bmi: Optional[float], query: str):
    avoid, reasons = {}, []
    if bmi is not None and bmi >= 26:
        avoid["power_training"] = "高冲击跳跃在体重偏大时增加膝关节负荷"
        reasons.append("BMI偏高，避免高冲击跳跃和大重量")
    if bmi is not None and bmi < 18.5:
        avoid["bmi_management"] = "BMI偏低不应再减重"
        reasons.append("BMI偏低，避免进一步减重")
    if any(k in query for k in ["膝", "关节", "膝盖"]):
        avoid["power_training"] = "跳跃/冲刺对膝关节冲击大"
        avoid["endurance_training"] = "长时间跑步对膝关节冲击大"
        reasons.append("膝部不适，避免跑跳，优先游泳/骑行/快走")
    if any(k in query for k in ["腰", "背"]):
        avoid["strength_training"] = "负重可能加重腰背负担"
        reasons.append("腰背问题，避免大重量负重")
    if any(k in query for k in ["心脏", "心血管", "胸闷", "高血压"]):
        avoid["power_training"] = "高强度爆发对心血管压力大"
        reasons.append("心血管风险，避免高强度爆发，遵医嘱")
    return avoid, reasons


def graph_reason(weak_cn: str, bmi: Optional[float] = None,
                 query: str = "") -> Dict:
    """返回 {primary, secondary, forms, reasoning_chain, citations, ranked}。"""
    weak = _normalize_weak(weak_cn or "")
    weak_set = set(weak)
    avoid, reasons = _contraindications(bmi, query or "")

    # 动作对弱指标的综合改善
    scores = {}
    for ai, action in enumerate(ACTIONS):
        score = sum(max(EFFECT_MATRIX[ai][list(INDICATOR_CN).index(w)], 0)
                    for w in weak)
        if score > 0:
            scores[action] = score
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    # 禁忌过滤
    kept = [a for a, _ in ranked if a not in avoid]
    primary = kept[0] if kept else "comprehensive_training"
    secondary = kept[1] if len(kept) > 1 else None
    need_low = bool(avoid)
    forms = ACTION_FORM[primary][1] if need_low else ACTION_FORM[primary][0]

    # 推理链
    weak_names = "、".join(INDICATOR_CN[w] for w in weak)
    top3 = "、".join(f"{ACTION_CN[a]}({s:.2f})" for a, s in ranked[:3])
    chain = [
        f"【诊断】弱指标为{weak_names}。",
        f"【图谱匹配】查『动作→指标』效果矩阵，对{weak_names}改善最高的是：{top3}。",
    ]
    if reasons:
        dropped = "、".join(f"{ACTION_CN[a]}（{r}）" for a, r in avoid.items())
        chain.append(f"【禁忌过滤】{'；'.join(reasons)}；排除：{dropped}。")
        chain.append(f"【方案】首选{ACTION_CN[primary]}，受禁忌限制采用低冲击形式：{forms}。")
    else:
        chain.append("【禁忌过滤】无 BMI/伤病禁忌。")
        chain.append(f"【方案】首选{ACTION_CN[primary]}：{forms}。")
    if secondary:
        chain.append(f"辅以{ACTION_CN[secondary]}：{ACTION_FORM[secondary][1] if need_low else ACTION_FORM[secondary][0]}。")
    citations = [ACTION_SOURCE[primary]]
    if secondary:
        citations.append(ACTION_SOURCE[secondary])

    return {
        "primary": ACTION_CN[primary],
        "secondary": ACTION_CN[secondary] if secondary else None,
        "forms": forms,
        "reasoning_chain": chain,
        "citations": citations,
        "ranked": [{"action": ACTION_CN[a], "score": round(s, 3),
                    "excluded": a in avoid,
                    "reason": avoid.get(a)} for a, s in ranked],
    }
