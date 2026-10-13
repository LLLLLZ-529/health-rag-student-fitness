#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45_graph_rag.py — 轻量 GraphRAG：体测健康知识图谱 + 多约束图推理链

与普通向量 RAG 的区别：
  普通 RAG 只按语义相似度找"看起来相关"的片段，无法做"排除禁忌、选效果最大动作、
  多约束权衡"。本模块把领域知识显式建模为图：

      [体测指标] --效果权重--> [训练动作] --具体方式--> [训练动作实例]
                                  |
                            [禁忌条件] --排除--> [训练动作]
                            [训练动作] --依据--> [知识库来源]

  查询时沿图多跳遍历：弱指标 → 效果矩阵选动作 → 禁忌边过滤 → 来源边取引用，
  并把每一步转成自然语言推理链（graph-grounded CoT）。

说明：这是"手工本体 + 规则图"的轻量 GraphRAG，效果矩阵来自 m3_agentic_rl
环境模拟器的运动科学先验，不是 LLM 自动抽取；垂直场景下更准、可解释、零 GPU。
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# 图谱本体（复用 m3_agentic_rl/environment/simulator.py 的运动科学先验）
# ---------------------------------------------------------------------------
INDICATORS = [
    "bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
    "sit_and_reach", "endurance_run", "strength",
]
INDICATOR_CN = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
    "endurance_run": "耐力跑", "strength": "力量",
}
# 弱指标 → 该指标所在的矩阵列索引（与 EFFECT_MATRIX 列一致）
IND_COL = {name: i for i, name in enumerate(INDICATORS)}

ACTIONS = [
    "endurance_training", "strength_training", "flexibility_training",
    "power_training", "cardio_training", "bmi_management",
    "comprehensive_training", "rest_recovery",
]

# 行=动作，列=指标，正值=改善，负值=恶化（每轮ΔHI，运动科学先验）
EFFECT_MATRIX = [
    [0.08, 0.12, 0.03, 0.05, 0.02, 0.15, 0.04],  # endurance
    [0.03, 0.04, 0.06, 0.10, 0.01, 0.02, 0.18],  # strength
    [0.02, 0.01, 0.00, 0.02, 0.18, 0.01, 0.00],  # flexibility
    [0.01, 0.03, 0.15, 0.16, 0.00, 0.04, 0.08],  # power
    [0.04, 0.18, 0.02, 0.03, 0.01, 0.10, 0.02],  # cardio
    [0.15, 0.05, 0.02, 0.03, 0.02, 0.08, 0.01],  # bmi_management
    [0.06, 0.08, 0.06, 0.07, 0.06, 0.08, 0.08],  # comprehensive
    [-0.02, -0.03, -0.02, -0.02, -0.02, -0.03, -0.02],  # rest
]

# 动作属性：中文名、冲击等级、默认训练方式、低冲击替代、知识来源
ACTION_META = {
    "endurance_training": {
        "cn": "耐力训练", "impact": "mid",
        "forms": "匀速慢跑、游泳、骑行",
        "low_impact": "游泳、骑行", "impact_note": "跑步对下肢冲击较大",
        "source": "运动处方/_全民健身指南__国家体育总局_OCR__chunk001.md"},
    "strength_training": {
        "cn": "力量训练", "impact": "mid",
        "forms": "深蹲、俯卧撑、哑铃、平板支撑",
        "low_impact": "水中训练、弹力带、自重慢速训练",
        "impact_note": "大重量负荷对关节/心肺压力大",
        "source": "运动处方/_全民健身指南__国家体育总局_OCR__chunk005.md"},
    "flexibility_training": {
        "cn": "柔韧训练", "impact": "low",
        "forms": "静态拉伸、瑜伽、髋部拉伸",
        "low_impact": "静态拉伸、瑜伽（本身低冲击）",
        "impact_note": "低冲击，多数人群适用",
        "source": "运动处方/_中青年科学运动指南_解读__chunk001.md"},
    "power_training": {
        "cn": "爆发力训练", "impact": "high",
        "forms": "冲刺、蛙跳、跳箱、高抬腿",
        "low_impact": "（爆发力训练难以低冲击，建议改用低冲击有氧）",
        "impact_note": "跳跃/冲刺为高冲击，对膝关节和体重要求高",
        "source": "运动处方/_全民健身指南__国家体育总局_OCR__chunk007.md"},
    "cardio_training": {
        "cn": "心肺训练", "impact": "mid",
        "forms": "慢跑、跳绳、间歇跑、有氧操",
        "low_impact": "游泳、快走、骑行、水中行走",
        "impact_note": "跳绳/跑步为中高冲击，游泳/快走为低冲击",
        "source": "运动处方/_全民健身指南__国家体育总局_OCR__chunk001.md"},
    "bmi_management": {
        "cn": "体重管理", "impact": "low",
        "forms": "快走、游泳、骑行配合饮食控制",
        "low_impact": "游泳、快走、水中有氧（本身低冲击）",
        "impact_note": "低冲击有氧 + 饮食控制",
        "source": "膳食营养/成人肥胖食养指南_2024年版___chunk010.md"},
    "comprehensive_training": {
        "cn": "综合训练", "impact": "mid",
        "forms": "全身循环、有氧+力量交替",
        "low_impact": "水中综合训练、低冲击循环",
        "impact_note": "多素质均衡发展",
        "source": "运动处方/_全年龄段人群科学健身指南来了_国家体育总局_OCR__chunk004.md"},
    "rest_recovery": {
        "cn": "休息恢复", "impact": "low",
        "forms": "主动恢复、拉伸、睡眠管理",
        "low_impact": "主动恢复、拉伸（本身低冲击）",
        "impact_note": "短期指标下降，长期防止过度训练",
        "source": "运动处方/_中青年科学运动指南_解读__chunk001.md"},
}

# 禁忌规则：条件 →（需排除的动作，原因）
def derive_contraindications(bmi: Optional[float], constraints: List[str]):
    avoid, reasons = {}, []
    if bmi is not None and bmi >= 26:
        avoid["power_training"] = "高冲击跳跃，体重过大会增加膝关节负荷"
        avoid["bmi_management"] = None  # 体重管理反而应作为首选，不排除；占位
        avoid.pop("bmi_management")
        reasons.append(f"BMI={bmi:.1f} 偏高：避免高冲击跳跃和大重量")
    if bmi is not None and bmi < 18.5:
        avoid["bmi_management"] = "BMI偏低不应再减重"
        reasons.append(f"BMI={bmi:.1f} 偏低：避免进一步减重和过度有氧")
    if any(k in "".join(constraints) for k in ["膝", "关节", "膝盖"]):
        avoid["power_training"] = "跳跃/冲刺对膝关节冲击大"
        avoid["endurance_training"] = "长时间跑步对膝关节冲击大"
        reasons.append("膝部不适：避免跑跳，优先游泳/骑行/快走")
    if any(k in "".join(constraints) for k in ["腰", "背"]):
        avoid["strength_training"] = "负重训练可能加重腰背负担"
        reasons.append("腰背问题：避免大重量负重，先做核心稳定")
    if any(k in "".join(constraints) for k in ["心血管", "心脏", "胸闷", "高血压"]):
        avoid["power_training"] = "高强度爆发对心血管压力大"
        reasons.append("心血管风险：避免高强度爆发，采用中等强度并遵医嘱")
    return avoid, reasons


# ---------------------------------------------------------------------------
# 图谱查询 + 推理
# ---------------------------------------------------------------------------
@dataclass
class ReasoningStep:
    step: str
    detail: str


@dataclass
class GraphResult:
    weak: List[str]
    ranked_actions: List[Dict]
    reasoning: List[ReasoningStep]
    final_plan: str
    citations: List[str] = field(default_factory=list)


class HealthKnowledgeGraph:
    def __init__(self):
        self.edges = []  # 效果边 (action, indicator, weight)
        for ai, action in enumerate(ACTIONS):
            for ii, ind in enumerate(INDICATORS):
                w = EFFECT_MATRIX[ai][ii]
                if abs(w) > 0.001:
                    self.edges.append((action, ind, w))

    def reason(self, weak: List[str], bmi: Optional[float] = None,
               constraints: Optional[List[str]] = None,
               weak_pcts: Optional[Dict[str, float]] = None) -> GraphResult:
        """多约束图推理：弱指标 → 效果选动作 → 禁忌过滤 → 来源。"""
        constraints = constraints or []
        weak_pcts = weak_pcts or {}
        reasoning = []

        # 步骤1：弱指标定位
        weak_names = "、".join(INDICATOR_CN.get(w, w) for w in weak)
        lvl = []
        for w in weak:
            p = weak_pcts.get(w, 30)
            lvl.append(f"{INDICATOR_CN.get(w, w)}（同性别人群后{p:.0f}%）" if p < 50
                       else INDICATOR_CN.get(w, w))
        reasoning.append(ReasoningStep("① 弱指标定位", "、".join(lvl) + "，作为干预主目标。"))

        # 步骤2：沿效果边，计算每个动作对弱指标的综合改善
        scores = {}
        contrib = {}
        for ai, action in enumerate(ACTIONS):
            total = 0.0
            parts = []
            for w in weak:
                wcol = IND_COL[w]
                val = EFFECT_MATRIX[ai][wcol]
                if val > 0:
                    # 越弱的指标权重越高（百分位越低越优先）
                    pwt = 1.5 - (weak_pcts.get(w, 30) / 100)
                    val *= max(0.6, pwt)
                    total += val
                    parts.append((INDICATOR_CN.get(w, w), val))
            if total > 0:
                scores[action] = total
                contrib[action] = parts
        top = sorted(scores.items(), key=lambda x: -x[1])
        top_desc = "；".join(
            f"{ACTION_META[a]['cn']}（{'+'.join(f'{n}{v:.2f}' for n,v in contrib[a][:2])}）"
            for a, s in top[:3])
        reasoning.append(ReasoningStep("② 效果边匹配", f"遍历『动作→指标』效果边，综合改善最高：{top_desc}。"))

        # 步骤3：禁忌过滤
        avoid, reasons = derive_contraindications(bmi, constraints)
        kept, dropped = [], []
        for action, score in top:
            if action in avoid:
                reason = avoid[action] or "受当前条件限制"
                dropped.append((ACTION_META[action]["cn"], reason))
            else:
                kept.append((action, score))
        if reasons:
            line = "；".join(reasons)
            if dropped:
                line += "。排除：" + "、".join(f"{n}（{r}）" for n, r in dropped)
            reasoning.append(ReasoningStep("③ 禁忌边过滤", line))
        else:
            reasoning.append(ReasoningStep("③ 禁忌边过滤", "无 BMI/伤病禁忌，无需排除动作。"))

        # 步骤4：选首选/次选，确定冲击形式与来源
        primary = kept[0][0] if kept else "comprehensive_training"
        secondary = kept[1][0] if len(kept) > 1 else None
        meta = ACTION_META[primary]
        need_low = bool(avoid)  # 有禁忌则优先低冲击形式
        form = meta["low_impact"] if need_low else meta["forms"]
        reasoning.append(ReasoningStep(
            "④ 确定方案与冲击形式",
            f"首选【{meta['cn']}】，{'受禁忌限制改用低冲击形式：' + form if need_low else '采用：' + form}。"
            + (f"辅以【{ACTION_META[secondary]['cn']}】。" if secondary else "")))

        citations = [meta["source"]]
        if secondary:
            citations.append(ACTION_META[secondary]["source"])
        reasoning.append(ReasoningStep("⑤ 沿来源边取引用", "依据：" + "、".join(citations)))

        # 最终方案文本
        plan = (
            f"针对{weak_names}，首选【{meta['cn']}】：{form}；"
            + (f"辅以【{ACTION_META[secondary]['cn']}】。" if secondary else "")
            + "遵循循序渐进、运动前热身、出现不适即停的原则。"
        )

        ranked = [
            {"action": a, "cn": ACTION_META[a]["cn"], "score": round(s, 3),
             "impact": ACTION_META[a]["impact"], "excluded": a in avoid,
             "exclude_reason": avoid.get(a)}
            for a, s in top
        ]
        return GraphResult(weak, ranked, reasoning, plan, citations)


def render(result: GraphResult):
    """把图推理结果渲染成 markdown 风格的推理链。"""
    lines = ["# GraphRAG 多约束推理结果\n", "**弱指标**：" + "、".join(
        INDICATOR_CN.get(w, w) for w in result.weak) + "\n"]
    lines.append("## 推理链")
    for r in result.reasoning:
        lines.append(f"- {r.step}：{r.detail}")
    lines.append("\n## 动作排序（效果分，含禁忌标记）")
    for item in result.ranked_actions:
        mark = "❌排除" if item["excluded"] else "✓"
        extra = f"（{item['exclude_reason']}）" if item["exclude_reason"] else ""
        lines.append(f"- {mark} {item['cn']} [{item['impact']}冲击] 效果分 {item['score']}{extra}")
    lines.append("\n## 最终方案")
    lines.append(result.final_plan)
    lines.append("\n## 引用")
    for c in result.citations:
        lines.append(f"- [来源: {c}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 演示：多约束场景（普通 RAG 难以处理）
# ---------------------------------------------------------------------------
def main():
    g = HealthKnowledgeGraph()

    cases = [
        {
            "name": "案例1：单弱指标（基线）",
            "weak": ["vital_capacity"], "bmi": 22.0, "constraints": [],
            "weak_pcts": {"vital_capacity": 12},
        },
        {
            "name": "案例2：肺活量弱 + BMI偏高（多约束）",
            "weak": ["vital_capacity"], "bmi": 27.5, "constraints": [],
            "weak_pcts": {"vital_capacity": 15},
        },
        {
            "name": "案例3：耐力弱 + 膝盖不适 + BMI偏高（复杂多约束）",
            "weak": ["endurance_run"], "bmi": 28.0, "constraints": ["膝盖不适"],
            "weak_pcts": {"endurance_run": 20},
        },
        {
            "name": "案例4：力量与耐力双弱（多指标权衡）",
            "weak": ["strength", "endurance_run"], "bmi": 21.0, "constraints": [],
            "weak_pcts": {"strength": 10, "endurance_run": 25},
        },
    ]

    for case in cases:
        print("=" * 72)
        print(case["name"])
        print("=" * 72)
        res = g.reason(case["weak"], case["bmi"], case["constraints"],
                       case["weak_pcts"])
        print(render(res))
        print()

    # 导出一份图结构（边列表），供可视化/前端使用
    out = __file__.replace("45_graph_rag.py", "health_graph_edges.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"nodes": {
            "indicators": INDICATOR_CN,
            "actions": {a: ACTION_META[a]["cn"] for a in ACTIONS},
        }, "effect_edges": [
            {"action": a, "indicator": INDICATOR_CN[i], "weight": w}
            for a, i, w in g.edges]
        }, f, ensure_ascii=False, indent=2)
    print(f"图结构已导出: {out}")


if __name__ == "__main__":
    main()
