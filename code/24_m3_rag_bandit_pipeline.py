#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24_m3_rag_bandit_pipeline.py — M3 第三阶段：RAG + 推荐理由 + Bandit 完整pipeline

端到端推荐系统：
  1. 学生健康分类输入（9类）
  2. RAG检索：从2674 chunk知识库检索相关运动/膳食建议
  3. LinUCB选策略：4种推荐策略中选最优
  4. 推荐理由生成：基于检索结果和策略，生成自然语言推荐理由
  5. 案例展示：9类学生的完整推荐流程

输出：
  - outputs/m3_pipeline/m3_pipeline_case_studies.md
  - outputs/m3_pipeline/m3_pipeline_results.csv
"""
from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "hi_wide.csv"
LABELS_CSV = ROOT / "deliverables" / "HI九类_当前模型预测.csv"
CHUNKS_DIR = ROOT / "chunks" / "chunks"
OUT_DIR = Path(__file__).parent / "outputs" / "m3_pipeline"

CLASS_NAMES = [
    "低水平-退化型", "低水平-平稳型", "低水平-改善型",
    "中水平-退化型", "中水平-平稳型", "中水平-改善型",
    "高水平-退化型", "高水平-平稳型", "高水平-改善型",
]
ARM_NAMES = ["运动干预为主", "膳食干预为主", "运动+膳食综合", "维持现状+监测"]
OPTIMAL_ARM = {0: 2, 1: 0, 2: 3, 3: 2, 4: 0, 5: 3, 6: 1, 7: 3, 8: 3}

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

# 推荐理由模板
REASON_TEMPLATES = {
    0: {  # 低水平-退化型
        0: "您的体质处于较低水平且呈下降趋势，建议以运动干预为主。重点提升心肺功能和基础体能，从低强度有氧开始，逐步增加力量训练。",
        1: "您的体质处于较低水平且呈下降趋势，膳食调整是关键。建议增加蛋白质摄入，保证营养均衡，为运动恢复提供基础。",
        2: "您的体质处于较低水平且呈下降趋势，需要运动和膳食的综合干预。建议在专业指导下制定个性化训练计划，同时调整饮食结构。",
        3: "您的体质处于较低水平，虽然目前在下降，但建议先建立健康监测习惯，定期体测，观察变化趋势后再制定干预方案。",
    },
    1: {  # 低水平-平稳型
        0: "您的体质处于较低水平但保持稳定，运动干预是提升的关键。建议从有氧运动入手，每周3-4次，逐步提升体能。",
        1: "您的体质处于较低水平，膳食调整可以为运动提升提供营养支持。建议增加优质蛋白和复合碳水的摄入。",
        2: "您的体质处于较低水平，综合干预效果最佳。建议结合有氧运动和力量训练，同时保证营养摄入。",
        3: "您的体质稳定在较低水平，可以维持当前状态，但建议定期监测，避免出现下降趋势。",
    },
    2: {  # 低水平-改善型
        0: "您的体质正在改善，继续保持运动习惯。建议适当增加训练强度，巩固改善趋势。",
        1: "您的体质正在改善，膳食方面继续保持均衡营养，为持续提升提供支持。",
        2: "您的体质正在改善，综合干预可以加速提升。建议在现有基础上增加力量训练比例。",
        3: "您的体质正在改善，建议维持当前的运动和饮食习惯，定期监测体测指标。",
    },
    3: {  # 中水平-退化型
        0: "您的体质中等但呈下降趋势，运动干预可以有效逆转。建议恢复规律运动，重点关注耐力和力量训练。",
        1: "您的体质中等但呈下降趋势，膳食调整可能是原因之一。建议检查饮食结构，减少高糖高脂食物。",
        2: "您的体质中等但呈下降趋势，需要综合干预来逆转。建议制定系统的训练计划，同时调整饮食和作息。",
        3: "您的体质中等但在下降，建议先密切监测，找出下降原因，再决定干预方式。",
    },
    4: {  # 中水平-平稳型
        0: "您的体质中等且稳定，运动干预可以帮助突破平台期。建议增加训练强度或尝试新的运动方式。",
        1: "您的体质中等且稳定，膳食优化可以进一步提升运动表现。建议关注宏量营养素比例。",
        2: "您的体质中等且稳定，综合干预有助于向高水平迈进。建议结合力量训练和营养调整。",
        3: "您的体质稳定在中等水平，可以维持当前状态，但建议设定提升目标。",
    },
    5: {  # 中水平-改善型
        0: "您的体质正在提升，继续保持运动习惯。建议增加力量训练，进一步提高体能。",
        1: "您的体质正在提升，膳食方面继续保持良好习惯，为持续进步提供营养。",
        2: "您的体质正在提升，综合干预可以加速进步。建议在现有基础上优化训练和饮食。",
        3: "您的体质正在改善，建议维持当前的健康生活方式，定期监测体测变化。",
    },
    6: {  # 高水平-退化型
        0: "您的体质优秀但呈下降趋势，运动方面建议调整训练量，避免过度训练，注重恢复。",
        1: "您的体质优秀但呈下降趋势，膳食调整可能是关键。建议关注营养摄入是否充足，特别是蛋白质和微量元素。",
        2: "您的体质优秀但在下降，需要综合评估。建议检查训练负荷、营养和睡眠，找出下降原因。",
        3: "您的体质优秀但在下降，建议密切监测，必要时寻求专业指导，防止进一步下滑。",
    },
    7: {  # 高水平-平稳型
        0: "您的体质优秀且稳定，继续保持运动习惯。建议尝试多样化训练，保持运动兴趣。",
        1: "您的体质优秀且稳定，膳食方面继续保持均衡营养，支持高水平运动表现。",
        2: "您的体质优秀且稳定，综合维护可以保持状态。建议定期评估训练和饮食的合理性。",
        3: "您的体质优秀且稳定，建议维持当前的健康生活方式，这是最佳状态。",
    },
    8: {  # 高水平-改善型
        0: "您的体质优秀且持续进步，继续保持高效训练。建议关注运动表现的进一步提升。",
        1: "您的体质优秀且持续进步，膳食优化可以支持更高水平的表现。建议关注运动营养。",
        2: "您的体质优秀且持续进步，综合优化可以突破极限。建议在专业指导下精细化训练和饮食。",
        3: "您的体质优秀且持续进步，建议维持当前状态，这是非常理想的健康趋势。",
    },
}


def load_chunks():
    chunks = []
    for cat in ["运动处方", "膳食营养", "体测标准", "健康政策"]:
        cat_dir = CHUNKS_DIR / cat
        if not cat_dir.exists():
            continue
        for f in sorted(cat_dir.glob("*.md")):
            try:
                text = f.read_text(encoding="utf-8")
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        text = parts[2].strip()
                chunks.append({"path": f"{cat}/{f.name}", "category": cat, "text": text[:2000]})
            except Exception:
                pass
    return chunks


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=["的", "了", "是", "在", "和", "与", "或", "等", "为", "以"], ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(query, vectorizer, matrix, chunks, top_k=3, category_filter=None):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    if category_filter:
        for i, c in enumerate(chunks):
            if c["category"] not in category_filter:
                sims[i] = -1
    top_idx = sims.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        if sims[idx] <= 0:
            continue
        results.append({"category": chunks[idx]["category"], "path": chunks[idx]["path"],
                        "score": round(float(sims[idx]), 4), "preview": chunks[idx]["text"][:120].replace("\n", " ")})
    return results


class LinUCB:
    def __init__(self, n_arms, context_dim, alpha=1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def select(self, context):
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        return int(np.argmax(p))

    def update(self, context, action, reward):
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


def train_linucb(contexts, classes, optimal_arms, n_steps=2000, seed=42):
    """预训练LinUCB，使其收敛到最优策略"""
    rng = np.random.RandomState(seed)
    algo = LinUCB(n_arms=4, context_dim=contexts.shape[1], alpha=0.5)
    idx = 0
    for t in range(n_steps):
        ctx = contexts[idx % len(contexts)]
        cls = classes[idx % len(contexts)]
        action = algo.select(ctx)
        optimal = optimal_arms[idx % len(contexts)]
        reward = 0.8 + rng.normal(0, 0.1) if action == optimal else 0.3 + rng.normal(0, 0.1)
        algo.update(ctx, action, np.clip(reward, 0, 1))
        idx += 1
    return algo


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载知识库 ...")
    chunks = load_chunks()
    vectorizer, matrix = build_index(chunks)
    print(f"  {len(chunks)} 个 chunk")

    print("[2/4] 加载学生数据并预训练 LinUCB ...")
    raw = pd.read_csv(RAW_CSV, dtype={"student_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype={"student_id": str})
    labels = labels[labels["rule_eligible"].astype(bool)].copy()
    df = labels.merge(raw, on="student_id", how="inner", suffixes=("", "_raw"))

    # 构建上下文
    contexts, classes, optimal_arms = [], [], []
    metrics = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
               "sit_and_reach", "endurance_run_sec", "strength"]
    for cls in range(9):
        cls_df = df[df["rule_class"] == cls]
        if len(cls_df) == 0:
            continue
        sample = cls_df.sample(n=min(100, len(cls_df)), random_state=42)
        for _, row in sample.iterrows():
            cls_oh = np.zeros(9); cls_oh[cls] = 1
            feat = []
            for m in metrics:
                val = row.get(f"{m}_g4")
                feat.append(0.0 if pd.isna(val) else float(val))
            feat = np.array(feat, dtype=np.float32)
            feat = (feat - feat.mean()) / (feat.std() + 1e-8)
            contexts.append(np.concatenate([cls_oh, feat]))
            classes.append(cls)
            optimal_arms.append(OPTIMAL_ARM[cls])
    contexts = np.array(contexts, dtype=np.float32)
    classes = np.array(classes)
    optimal_arms = np.array(optimal_arms)

    bandit = train_linucb(contexts, classes, optimal_arms, n_steps=2000, seed=42)
    print(f"  LinUCB 预训练完成")

    print(f"\n[3/4] 生成 9 类学生完整推荐案例 ...")
    all_results = []
    case_studies = []

    for cls in range(9):
        cls_df = df[df["rule_class"] == cls]
        if len(cls_df) == 0:
            continue
        student = cls_df.iloc[0]

        # 构建上下文
        cls_oh = np.zeros(9); cls_oh[cls] = 1
        feat = []
        for m in metrics:
            val = student.get(f"{m}_g4")
            feat.append(0.0 if pd.isna(val) else float(val))
        feat = np.array(feat, dtype=np.float32)
        feat = (feat - feat.mean()) / (feat.std() + 1e-8)
        ctx = np.concatenate([cls_oh, feat])

        # Bandit选策略
        selected_arm = bandit.select(ctx)
        strategy = ARM_NAMES[selected_arm]

        # RAG检索
        query = CLASS_QUERIES[cls]
        # 根据选择的策略过滤检索类别
        cat_filter = None
        if selected_arm == 0:  # 运动为主
            cat_filter = ["运动处方", "体测标准"]
        elif selected_arm == 1:  # 膳食为主
            cat_filter = ["膳食营养", "健康政策"]
        elif selected_arm == 2:  # 综合
            cat_filter = ["运动处方", "膳食营养"]
        retrieved = retrieve(query, vectorizer, matrix, chunks, top_k=3, category_filter=cat_filter)

        # 生成推荐理由
        reason = REASON_TEMPLATES[cls][selected_arm]

        # 体测概况
        profile = {}
        for m in metrics:
            val = student.get(f"{m}_g4")
            if not pd.isna(val):
                profile[m] = round(float(val), 2)

        case = {
            "class": cls, "class_name": CLASS_NAMES[cls],
            "n_students": len(cls_df),
            "selected_strategy": strategy,
            "optimal_strategy": ARM_NAMES[OPTIMAL_ARM[cls]],
            "strategy_correct": selected_arm == OPTIMAL_ARM[cls],
            "reason": reason,
            "profile": profile,
            "retrieved": retrieved,
        }
        case_studies.append(case)
        all_results.append({
            "class": cls, "class_name": CLASS_NAMES[cls],
            "n_students": len(cls_df),
            "selected_strategy": strategy,
            "optimal_strategy": ARM_NAMES[OPTIMAL_ARM[cls]],
            "strategy_correct": selected_arm == OPTIMAL_ARM[cls],
            "n_retrieved": len(retrieved),
        })
        print(f"  class {cls} ({CLASS_NAMES[cls]}): 策略={strategy}, "
              f"最优={ARM_NAMES[OPTIMAL_ARM[cls]]}, 正确={selected_arm == OPTIMAL_ARM[cls]}")

    print(f"\n[4/4] 生成报告 ...")
    pd.DataFrame(all_results).to_csv(OUT_DIR / "m3_pipeline_results.csv", index=False)

    with open(OUT_DIR / "m3_pipeline_case_studies.md", "w") as f:
        f.write("# M3 完整推荐pipeline：9类学生案例\n\n")
        f.write("流程：学生健康分类 → RAG检索知识库 → LinUCB选策略 → 生成推荐理由\n\n")
        correct = sum(1 for c in case_studies if c["strategy_correct"])
        f.write(f"**策略选择准确率：{correct}/{len(case_studies)} = {correct/len(case_studies):.1%}**\n\n")

        for case in case_studies:
            f.write(f"## {case['class_name']}（class {case['class']}）\n\n")
            f.write(f"- 该类学生数：{case['n_students']}\n")
            f.write(f"- **推荐策略：{case['selected_strategy']}**\n")
            f.write(f"- 先验最优策略：{case['optimal_strategy']}\n")
            f.write(f"- 策略匹配：{'✅ 正确' if case['strategy_correct'] else '❌ 需调整'}\n\n")
            f.write("### 推荐理由\n\n")
            f.write(f"{case['reason']}\n\n")
            f.write("### 体测概况\n\n")
            for k, v in case["profile"].items():
                f.write(f"- {k}: {v}\n")
            f.write("\n### RAG检索结果\n\n")
            for i, r in enumerate(case["retrieved"]):
                f.write(f"**[{i+1}] {r['category']}** (相似度={r['score']})\n")
                f.write(f"- {r['preview']}...\n\n")

    print(f"\n完成 ✅")
    print(f"  策略选择准确率：{correct}/{len(case_studies)} = {correct/len(case_studies):.1%}")
    print(f"  写入 {OUT_DIR / 'm3_pipeline_case_studies.md'}")
    print(f"  写入 {OUT_DIR / 'm3_pipeline_results.csv'}")


if __name__ == "__main__":
    sys.exit(main())
