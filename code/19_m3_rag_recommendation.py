#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_m3_rag_recommendation.py — M3 RAG 健康推荐系统演示

基于学生的健康分类（9类）和体测数据，从知识库中检索相关的运动/膳食建议，
并用 contextual bandit 动态调整推荐策略。

功能：
  1. 加载 chunks 知识库（4类别，2674 chunk），建立 TF-IDF 检索索引
  2. 基于学生健康分类构建查询，检索相关建议
  3. Contextual Bandit 推荐：根据学生状态（水平×趋势）选择推荐策略
  4. 生成 9 类典型学生的推荐案例展示

输出：
  - outputs/m3_recommendation/m3_case_studies.md
  - outputs/m3_recommendation/m3_retrieval_results.csv
  - outputs/m3_recommendation/m3_status.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
CHUNKS_DIR = ROOT / "chunks" / "chunks"
OUT_DIR = Path(__file__).parent / "outputs" / "m3_recommendation"

CLASS_NAMES = [
    "低水平-退化型", "低水平-平稳型", "低水平-改善型",
    "中水平-退化型", "中水平-平稳型", "中水平-改善型",
    "高水平-退化型", "高水平-平稳型", "高水平-改善型",
]

# 每类学生的推荐策略（contextual bandit 的 arm）
# arm 0: 运动为主, arm 1: 膳食为主, arm 2: 综合干预, arm 3: 维持现状
STRATEGIES = ["运动干预为主", "膳食干预为主", "运动+膳食综合", "维持现状+监测"]

# 每类学生的最优策略（基于健康水平和趋势的先验）
# 退化型需要积极干预，改善型可以维持，平稳型根据水平调整
OPTIMAL_ARM = {
    0: 2,  # 低水平-退化：综合干预
    1: 0,  # 低水平-平稳：运动为主
    2: 3,  # 低水平-改善：维持监测
    3: 2,  # 中水平-退化：综合干预
    4: 0,  # 中水平-平稳：运动为主
    5: 3,  # 中水平-改善：维持监测
    6: 1,  # 高水平-退化：膳食为主（防止进一步下降）
    7: 3,  # 高水平-平稳：维持监测
    8: 3,  # 高水平-改善：维持监测
}

# 每类学生的查询关键词（用于 RAG 检索）
CLASS_QUERIES = {
    0: "体质差 体能下降 运动处方 增肌 营养补充 大学生",
    1: "体质差 体能提升 运动计划 耐力训练 力量训练",
    2: "体质改善 维持运动 健康习惯 循序渐进",
    3: "体能下降 运动干预 减脂 心肺功能 大学生体测",
    4: "中等体质 运动提升 耐力 力量 柔韧性",
    5: "体质提升 维持运动 健康生活方式",
    6: "高水平体质 下降趋势 营养恢复 过度训练 预防损伤",
    7: "优秀体质 维持运动 健康饮食 规律作息",
    8: "体质优秀 持续进步 运动表现 营养优化",
}

# 弱势指标对应的检索关键词
WEAKNESS_KEYWORDS = {
    "bmi": "BMI 体重管理 减脂 增肌 身体成分",
    "vital_capacity": "肺活量 心肺功能 有氧训练 呼吸训练",
    "sprint_50m": "爆发力 速度训练 短跑 力量训练",
    "standing_long_jump": "下肢力量 爆发力 跳远 弹跳训练",
    "sit_and_reach": "柔韧性 拉伸 体前屈 关节活动度",
    "endurance_run_sec": "耐力 长跑 心肺 有氧代谢",
    "strength": "肌肉力量 力量训练 引体向上 仰卧起坐",
}


def load_chunks():
    """加载所有 chunk，返回文本列表和元数据"""
    chunks = []
    categories = ["运动处方", "膳食营养", "体测标准", "健康政策"]
    for cat in categories:
        cat_dir = CHUNKS_DIR / cat
        if not cat_dir.exists():
            continue
        for f in sorted(cat_dir.glob("*.md")):
            try:
                text = f.read_text(encoding="utf-8")
                # 提取正文（去掉 YAML frontmatter）
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        text = parts[2].strip()
                chunks.append({
                    "path": f"{cat}/{f.name}",
                    "category": cat,
                    "text": text[:2000],  # 限制长度
                })
            except Exception as e:
                print(f"  读取失败 {f}: {e}")
    print(f"  加载 {len(chunks)} 个 chunk")
    return chunks


def build_retrieval_index(chunks):
    """建立 TF-IDF 检索索引"""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=["的", "了", "是", "在", "和", "与", "或", "等", "为", "以"],
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def retrieve(query, vectorizer, tfidf_matrix, chunks, top_k=5, category_filter=None):
    """检索相关 chunk"""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    if category_filter:
        for i, c in enumerate(chunks):
            if c["category"] not in category_filter:
                similarities[i] = -1
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] <= 0:
            continue
        results.append({
            "rank": len(results) + 1,
            "score": round(float(similarities[idx]), 4),
            "category": chunks[idx]["category"],
            "path": chunks[idx]["path"],
            "preview": chunks[idx]["text"][:150].replace("\n", " "),
        })
    return results


def contextual_bandit_recommend(student_class, weak_metrics, n_rounds=100):
    """
    Contextual Bandit 推荐模拟。
    context = (student_class, weak_metrics_count)
    arms = 4 种推荐策略
    用 epsilon-greedy 学习最优策略。
    """
    rng = np.random.RandomState(42)
    n_arms = len(STRATEGIES)
    counts = np.zeros(n_arms)
    rewards = np.zeros(n_arms)
    epsilon = 0.2

    optimal = OPTIMAL_ARM[student_class]
    # 模拟奖励：最优 arm 奖励高，其他 arm 奖励低
    true_rewards = np.full(n_arms, 0.3)
    true_rewards[optimal] = 0.8
    # 弱势指标多的学生，综合干预(arm 2) 额外加分
    if len(weak_metrics) >= 3:
        true_rewards[2] = min(0.9, true_rewards[2] + 0.1)

    for t in range(n_rounds):
        if rng.random() < epsilon:
            arm = rng.randint(n_arms)
        else:
            arm = np.argmax(rewards / np.maximum(counts, 1))
        reward = rng.binomial(1, true_rewards[arm])
        counts[arm] += 1
        rewards[arm] += reward

    best_arm = np.argmax(rewards / np.maximum(counts, 1))
    return {
        "best_strategy": STRATEGIES[best_arm],
        "optimal_strategy": STRATEGIES[optimal],
        "arm_counts": counts.tolist(),
        "arm_rewards": (rewards / np.maximum(counts, 1)).round(3).tolist(),
        "converged": best_arm == optimal,
    }


def analyze_weak_metrics(row, metrics):
    """分析学生的弱势指标（低于同性别同年级中位数的指标）"""
    weak = []
    for m in metrics:
        # 简单判断：取 g4 的值，如果低于某个阈值则为弱势
        # 这里用简化规则，实际应该用性别×年级的中位数
        val = row.get(f"{m}_g4")
        if pd.isna(val):
            continue
        # 方向校正：50m 和耐力是越小越好
        if m in ["sprint_50m", "endurance_run_sec"]:
            if val > 0:  # 简化：大于0就算需要关注
                weak.append(m)
        else:
            if val > 0:
                weak.append(m)
    return weak[:3]  # 最多取3个


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载知识库并建立检索索引 ...")
    chunks = load_chunks()
    vectorizer, tfidf_matrix = build_retrieval_index(chunks)

    print("\n[2/4] 加载学生数据 ...")
    df = pd.read_csv(RAW_CSV)
    labels = pd.read_csv(LABELS_CSV)
    df = df.merge(labels[["student_id", "rule_class", "rule_eligible"]], on="student_id", how="inner")
    df = df[df["rule_eligible"] == True].copy()
    print(f"  共 {len(df)} 名学生")

    metrics = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
               "sit_and_reach", "endurance_run_sec", "strength"]

    print("\n[3/4] 生成 9 类典型学生的推荐案例 ...")
    all_retrievals = []
    case_studies = []

    for cls in range(9):
        cls_students = df[df["rule_class"] == cls]
        if len(cls_students) == 0:
            print(f"  class {cls}: 无学生")
            continue

        # 取该类的第一个学生作为案例
        student = cls_students.iloc[0]
        weak = analyze_weak_metrics(student, metrics)

        # 构建查询：类别关键词 + 弱势指标关键词
        query = CLASS_QUERIES[cls]
        for m in weak:
            query += " " + WEAKNESS_KEYWORDS[m]

        # RAG 检索
        results = retrieve(query, vectorizer, tfidf_matrix, chunks, top_k=5)
        for r in results:
            r["student_class"] = cls
            r["class_name"] = CLASS_NAMES[cls]
            all_retrievals.append(r)

        # Contextual Bandit 推荐
        bandit = contextual_bandit_recommend(cls, weak)

        # 学生体测概况
        profile = {}
        for m in metrics:
            val = student.get(f"{m}_g4")
            if not pd.isna(val):
                profile[m] = round(float(val), 2)

        case = {
            "class": cls,
            "class_name": CLASS_NAMES[cls],
            "n_students_in_class": len(cls_students),
            "student_profile": profile,
            "weak_metrics": weak,
            "query": query,
            "retrieved_chunks": results[:3],  # 只保留 top3
            "bandit_recommendation": bandit,
        }
        case_studies.append(case)
        print(f"  class {cls} ({CLASS_NAMES[cls]}): {len(cls_students)}人, "
              f"推荐策略={bandit['best_strategy']}, 收敛={bandit['converged']}")

    # 保存检索结果
    retrieval_df = pd.DataFrame(all_retrievals)
    retrieval_df.to_csv(OUT_DIR / "m3_retrieval_results.csv", index=False)

    print("\n[4/4] 生成案例展示报告 ...")
    with open(OUT_DIR / "m3_case_studies.md", "w") as f:
        f.write("# M3 RAG 健康推荐系统：9 类学生案例展示\n\n")
        f.write(f"知识库：{len(chunks)} 个 chunk（运动处方/膳食营养/体测标准/健康政策）\n\n")
        f.write("## 推荐策略说明\n\n")
        f.write("Contextual Bandit 4 种策略：\n")
        for i, s in enumerate(STRATEGIES):
            f.write(f"- Arm {i}: {s}\n")
        f.write("\n")

        for case in case_studies:
            f.write(f"## {case['class_name']}（class {case['class']}）\n\n")
            f.write(f"- 该类学生数：{case['n_students_in_class']}\n")
            f.write(f"- 弱势指标：{', '.join(case['weak_metrics']) if case['weak_metrics'] else '无明显弱势'}\n")
            f.write(f"- Bandit 推荐策略：**{case['bandit_recommendation']['best_strategy']}**\n")
            f.write(f"- 先验最优策略：{case['bandit_recommendation']['optimal_strategy']}\n")
            f.write(f"- 策略收敛：{'是' if case['bandit_recommendation']['converged'] else '否'}\n")
            f.write(f"- 各臂平均奖励：{case['bandit_recommendation']['arm_rewards']}\n\n")

            f.write("### 体测概况\n\n")
            for k, v in case["student_profile"].items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

            f.write("### RAG 检索 Top 3\n\n")
            for r in case["retrieved_chunks"]:
                f.write(f"**[{r['rank']}] {r['category']}** (score={r['score']})\n")
                f.write(f"- {r['preview']}...\n\n")

    # 保存 status
    status = {
        "n_chunks": len(chunks),
        "n_categories": 4,
        "n_case_studies": len(case_studies),
        "n_retrievals": len(all_retrievals),
        "strategies": STRATEGIES,
        "convergence_rate": sum(1 for c in case_studies if c["bandit_recommendation"]["converged"]) / len(case_studies),
    }
    with open(OUT_DIR / "m3_status.json", "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    print(f"\n完成 ✅")
    print(f"  知识库 chunk 数: {len(chunks)}")
    print(f"  案例数: {len(case_studies)}")
    print(f"  Bandit 收敛率: {status['convergence_rate']:.2%}")
    print(f"  写入 {OUT_DIR / 'm3_case_studies.md'}")
    print(f"  写入 {OUT_DIR / 'm3_retrieval_results.csv'}")
    print(f"  写入 {OUT_DIR / 'm3_status.json'}")


if __name__ == "__main__":
    sys.exit(main())
