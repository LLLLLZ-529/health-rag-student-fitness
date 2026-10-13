#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
46_e2e_demo.py — 端到端实测 demo（无需 GPU）

串起完整链路（除 7B 语言润色外全部真实运行）：
  真实学生数据 → 诊断弱指标 → 分类路由检索知识库 → GraphRAG 多约束图推理
  → 组装带引用的四步处方（诊断/分析/方案/禁忌）

有真实 hi_wide 数据时用真实学生，否则用内置示例。
"""
import sys, json, importlib.util
from pathlib import Path
import pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# 加载 44 检索器、45 图推理
rmod = _load("r44", CODE / "44_improved_retriever.py")
gmod = _load("g45", CODE / "45_graph_rag.py")

# 知识库路径（发布仓库公开库 / 原始包完整库）
PUB = ROOT / "knowledge_base" / "chunks"
FULL = ROOT / "data" / "chunks" / "chunks"
CHUNKS = PUB if PUB.exists() else FULL
retriever = rmod.RoutedRetriever(CHUNKS)
graph = gmod.HealthKnowledgeGraph()

IND_CN = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
    "endurance_run": "耐力跑", "strength": "力量",
}
QUERY_BY_WEAK = {
    "vital_capacity": "肺活量低怎么训练提升",
    "endurance_run": "耐力跑差怎么提升",
    "strength": "力量薄弱怎么练",
    "sprint_50m": "50米跑怎么提高速度",
    "standing_long_jump": "立定跳远怎么练",
    "sit_and_reach": "坐位体前屈柔韧性差怎么拉伸",
    "bmi": "BMI高怎么减脂运动",
}

# 数据表列名 → 图推理 key 命名统一
DATA_TO_GRAPH = {"endurance_run_sec": "endurance_run"}
def to_graph_key(k):
    return DATA_TO_GRAPH.get(k, k)


def build_prescription(student, extra_constraints=None):
    """完整链路：诊断 → 检索 → 图推理 → 四步处方。"""
    weak_en = student["weak"]
    weak_cn = IND_CN[weak_en]
    bmi = student["bmi"]
    constraints = extra_constraints or []
    query = QUERY_BY_WEAK.get(weak_en, "怎么训练") + "".join(constraints)

    # 1) 检索（分类路由 + jieba）
    docs = retriever.retrieve(query, top_k=3, weak=weak_cn)
    # 2) 图推理
    res = graph.reason([weak_en], bmi=bmi, constraints=constraints,
                       weak_pcts={weak_en: student.get("pct", 15)})

    # 3) 组装四步处方
    from_map = {v["cn"]: k for k, v in gmod.ACTION_META.items()}
    first_cn = res.final_plan.split("首选【")[1].split("】")[0] if "首选【" in res.final_plan else res.ranked_actions[0]["cn"]
    meta = gmod.ACTION_META[from_map[first_cn]]
    need_low = bool(extra_constraints) or (bmi and bmi >= 26) or (bmi and bmi < 18.5)
    forms = meta["low_impact"] if need_low else meta["forms"]

    diagnosis = (f"【诊断】该学生最弱指标为{weak_cn}，处于同性别人群后{student.get('pct',15)}%，"
                 f"四年趋势为{student['trend']}，BMI={bmi:.1f}。")
    # 分析：图谱效果匹配（为什么选这些动作）
    analysis = "【分析】" + res.reasoning[1].detail
    # 方案：图推理的方案步骤（首选/次选/冲击形式）+ 检索引用
    cite = docs[0]["source"] if docs else meta["source"]
    plan = "【方案】" + res.reasoning[3].detail + " 循序渐进、每2周复测一次。"
    plan += f" [来源: {cite}]"
    # 禁忌：图推理禁忌步骤；无禁忌时给安全提示
    contra_detail = res.reasoning[2].detail
    if contra_detail.startswith("无"):
        contra = "【禁忌】无明确 BMI/伤病禁忌；运动中如出现胸痛、关节锐痛、头晕应立即停止。"
    else:
        contra = "【禁忌】" + contra_detail

    return {
        "profile": student["profile"],
        "query": query,
        "routed": docs[0]["category"] if docs else "?",
        "docs": [(d["source"], d["score"]) for d in docs],
        "prescription": "\n".join([diagnosis, analysis, plan, contra]),
    }


def load_real_students():
    """从真实数据挑 3 个代表性学生。"""
    wide = ROOT / "data" / "hi_wide.csv"
    fixed = ROOT / "data" / "weak_indicator_fixed.csv"
    if not (wide.exists() and fixed.exists()):
        return None
    df = pd.read_csv(wide)
    fx = pd.read_csv(fixed)
    df = df.merge(fx[["student_id", "weakest_indicator", "weakest_pct"]], on="student_id")
    dec = df[df["HI_g4"] < df["HI_g1"]].copy()

    def mk(row, extra=None):
        weak = to_graph_key(row["weakest_indicator"])
        def fv(ind, g="g4"):
            v = row.get(f"{ind}_{g}")
            return "?" if pd.isna(v) else (f"{v:.0f}" if ind != "bmi" else f"{v:.1f}")
        profile = f"BMI={fv('bmi')}，肺活量={fv('vital_capacity')}ml，50米={fv('sprint_50m')}秒，跳远={fv('standing_long_jump')}，体前屈={fv('sit_and_reach')}，耐力跑={fv('endurance_run_sec')}秒，力量={fv('strength')}"
        return {"weak": weak, "bmi": row["bmi_g4"], "pct": int(row["weakest_pct"]),
                "trend": "退化", "profile": profile, "extra": extra or []}

    out = []
    # 肺活量弱 + BMI正常（过滤极端脏数据）
    s1 = dec[(dec["weakest_indicator"] == "vital_capacity")
             & (dec["bmi_g4"].between(20, 24))
             & (dec["vital_capacity_g4"].between(1800, 4500))]
    if len(s1): out.append(("真实学生A：肺活量弱、退化、BMI正常", mk(s1.iloc[0])))
    # 耐力弱 + BMI高（列名是 endurance_run_sec）
    s2 = dec[(dec["weakest_indicator"] == "endurance_run_sec") & (dec["bmi_g4"] >= 24)]
    if len(s2): out.append(("真实学生B：耐力弱 + BMI偏高", mk(s2.iloc[0])))
    # 力量弱 + BMI低
    s3 = dec[(dec["weakest_indicator"] == "strength") & (dec["bmi_g4"] < 19)
             & (dec["vital_capacity_g4"].between(1800, 4500))]
    if len(s3): out.append(("真实学生C：力量弱 + BMI偏低", mk(s3.iloc[0])))
    # 构造的复杂多约束案例
    if len(s2):
        base = mk(s2.iloc[0])
        base["extra"] = ["膝盖不适"]
        out.append(("构造案例D：同B但额外有膝盖不适（看GraphRAG如何改推荐）", base))
    return out


def main():
    cases = load_real_students()
    if not cases:
        print("未找到真实数据，使用内置示例。")
        cases = [
            ("示例：肺活量弱", {"weak": "vital_capacity", "bmi": 22.0, "pct": 12,
                              "trend": "退化", "profile": "BMI=22.0，肺活量=2600ml ...", "extra": []}),
            ("示例：耐力弱+BMI高+膝盖不适", {"weak": "endurance_run", "bmi": 28.0, "pct": 18,
                              "trend": "退化", "profile": "BMI=28.0，耐力跑=4分20秒 ...", "extra": ["膝盖不适"]}),
        ]

    for title, st in cases:
        print("\n" + "█" * 76)
        print("■ " + title)
        print("█" * 76)
        print("【学生大四数据】" + st["profile"])
        r = build_prescription(st, st.get("extra"))
        print(f"\n[检索] 问题「{r['query']}」→ 路由分类：{r['routed']}")
        for src, score in r["docs"]:
            print(f"        {score:.3f}  {src[:52]}")
        print("\n[最终四步处方]")
        print(r["prescription"])


if __name__ == "__main__":
    main()
