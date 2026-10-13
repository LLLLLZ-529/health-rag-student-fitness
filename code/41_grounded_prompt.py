#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
41_grounded_prompt.py — 引用溯源（Grounding）+ 推理链（CoT）Prompt 模板

不用重训模型，只改推理时的 prompt，就能让模型：
  1. 先诊断、再分析、再开方、最后过禁忌（CoT 推理链）
  2. 每条建议都标注知识库来源（[来源: 文件名]）

用法：
  # 看 prompt 长什么样（不依赖模型）
  python code/41_grounded_prompt.py demo

  # 真正集成到 HealthAgent：
  #   把 writer.py 里的 sys_msg / user_msg 替换为本文件的 build_messages()
  #   retriever.py 用 retrieve_with_sources() 替换 retriever_node()
"""
import os, json
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 新 System Prompt：强制四步推理 + 引用溯源
# ---------------------------------------------------------------------------
GROUNDED_SYSTEM_PROMPT = """你是一名专业的大学生体测健康顾问，严格基于知识库和学生数据输出建议。

你必须按以下四步结构组织回复，且每一步都不可省略：

【诊断】指出该学生的最弱指标（如肺活量/耐力/力量），结合四年趋势说明判断依据。
【分析】基于运动健康知识，解释弱指标成因、应优先改善的方向，以及为什么不选其他训练类型。
【方案】给出具体可执行的运动建议，必须包含：训练类型、频率（每周几次）、单次时长、强度、持续周数。
【禁忌】根据 BMI/既往风险提示禁忌动作；若无明确禁忌，也要说明出现何种症状应停止。

【引用要求】
- 方案中的每条建议后，用 [来源: 知识库文件名] 标注依据。
- 若知识库中没有对应依据，不要编造，明确标注 [无知识库依据，需人工确认]。
- 禁止输出知识库以外的药物、剂量、疾病诊断建议。
"""


# ---------------------------------------------------------------------------
# 构造带引用的参考资料块
# ---------------------------------------------------------------------------
def format_retrieved_context(docs: List[Dict[str, Any]]) -> str:
    """docs: [{"source": "运动处方/xxx.md", "category": "运动处方", "text": "..."}]"""
    if not docs:
        return "（本次未检索到知识库内容，请基于常识谨慎作答，并标注 [无知识库依据]）"
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.get("source", "unknown")
        text = d.get("text", "")[:400]
        blocks.append(f"[资料{i}] 来源文件：{src}\n{text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 构造完整 messages
# ---------------------------------------------------------------------------
def build_messages(student_data: dict, hi_score: float, level_label: str,
                   weak_indicator: str, strategy: str,
                   user_query: str, retrieved_docs: List[Dict[str, Any]],
                   critic_errors: List[str] = None) -> List[dict]:
    context = format_retrieved_context(retrieved_docs)
    d = student_data
    user_msg = (
        f"【学生体测数据】\n"
        f"BMI={d.get('bmi','?')}，肺活量={d.get('vc','?')}ml，"
        f"50米={d.get('speed','?')}秒，跳远={d.get('jump','?')}cm，"
        f"坐位体前屈={d.get('flex','?')}cm，耐力跑={d.get('endurance','?')}秒，"
        f"力量={d.get('str','?')}。\n"
        f"归一化健康指数 HI={hi_score}/100，综合分类={level_label}，"
        f"最弱指标={weak_indicator}，策略={strategy}。\n\n"
        f"【用户问题】\n{user_query}\n\n"
        f"【检索到的知识库内容】\n{context}\n"
    )
    if critic_errors:
        user_msg += (
            f"\n【审核驳回，需修正】\n上次建议被驳回，原因：{'；'.join(critic_errors)}。"
            "请严格按四步结构重新输出，并确保每条建议都有来源。\n"
        )
    user_msg += "\n请严格按【诊断】【分析】【方案】【禁忌】四步输出，并在每条建议后标注来源。"
    return [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# 带来源的检索函数（替代 retriever.py 里的 retriever_node）
# ---------------------------------------------------------------------------
def retrieve_with_sources(query: str, chunks_dir: Path = None, top_k: int = 3) -> List[Dict[str, Any]]:
    """检索知识库并返回带来源的结果。兼容两个路径。"""
    if chunks_dir is None:
        for c in [ROOT / "knowledge_base" / "chunks",
                  ROOT / "data" / "chunks" / "chunks"]:
            if c.exists():
                chunks_dir = c
                break
    if chunks_dir is None or not chunks_dir.exists():
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    docs, metas = [], []
    for md in chunks_dir.rglob("*.md"):
        try:
            docs.append(md.read_text(encoding="utf-8", errors="ignore"))
            metas.append(str(md.relative_to(chunks_dir)))
        except Exception:
            continue
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    mat = vec.fit_transform(docs)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    results = []
    for i in top_idx:
        if sims[i] > 0.01:
            source = metas[i]
            results.append({
                "source": source,
                "category": source.split("/")[0] if "/" in source else "",
                "text": docs[i][:400],
                "score": round(float(sims[i]), 3),
            })
    return results


# ---------------------------------------------------------------------------
# 演示：展示新 prompt 长什么样（不加载模型）
# ---------------------------------------------------------------------------
def demo():
    print("=" * 70)
    print("演示：推理链 + 引用溯源的 Prompt")
    print("=" * 70)

    # 一个假学生：肺活量弱
    student = {"bmi": 22.6, "vc": 2800, "speed": 9.2, "jump": 165,
               "flex": 15.0, "endurance": "5分20秒", "str": 45}
    docs = retrieve_with_sources("肺活量 有氧训练 慢跑")
    print(f"\n检索到 {len(docs)} 条带来源的知识库内容：")
    for d in docs:
        print(f"  [{d['score']}] {d['source']}")

    messages = build_messages(
        student_data=student,
        hi_score=58,
        level_label="中水平-退化型",
        weak_indicator="肺活量",
        strategy="有氧+呼吸训练",
        user_query="我肺活量太低了，怎么训练？",
        retrieved_docs=docs,
    )
    print("\n" + "=" * 70)
    print("【SYSTEM】")
    print(messages[0]["content"])
    print("=" * 70)
    print("【USER】")
    print(messages[1]["content"])
    print("=" * 70)
    print("\n模型在这个 prompt 下，预期输出结构：")
    print("""
【诊断】该学生肺活量 2800ml（女生大一标准下偏后15%），四年趋势连续下降，
       BMI 22.6 正常，主弱指标为心肺功能，综合分类为中水平-退化型。
【分析】肺活量低多与有氧能力不足相关，应优先中等强度有氧 + 呼吸训练，
       而非力量训练（力量指标 45 已正常）。
【方案】1. 慢跑：每周3次，每次30分钟，心率控制在最大心率60-70% [来源: 运动处方/xxx.md]
       2. 游泳：每周1次，每次40分钟 [来源: 运动处方/xxx.md]
       3. 腹式呼吸训练：每天2组，每组20次 [来源: 运动处方/xxx.md]
       持续8周，每2周复测一次。
【禁忌】BMI 正常，无心血管禁忌；若运动中出现胸闷、头晕应立即停止并就医。
    """)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("用法: python code/41_grounded_prompt.py demo")
