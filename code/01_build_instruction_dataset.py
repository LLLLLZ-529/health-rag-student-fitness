#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_instruction_dataset.py — 构造成体测健康领域指令数据集

4种任务类型：
1. 分类：给定体测数据 → 9类健康标签
2. 预测：给定大一大二 → 预测大四是否退化
3. 推荐：给定学生状态 → 运动/饮食推荐
4. 知识问答：从知识库提取的健康问答
"""
import os, sys, json, random
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).parents[4]
HI_WIDE = BASE / "hi_wide.csv"
LABELS = BASE / "deliverables" / "HI九类_当前模型预测.csv"
OUTPUT = Path(__file__).parent / "instruction_dataset.jsonl"

# 9类标签名称
CLASS_NAMES = [
    "低水平-退化型", "低水平-平稳型", "低水平-改善型",
    "中水平-退化型", "中水平-平稳型", "中水平-改善型",
    "高水平-退化型", "高水平-平稳型", "高水平-改善型",
]

INDICATOR_CN = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
    "endurance_run_sec": "耐力跑", "strength": "力量",
}

def fmt_value(ind, val):
    """格式化指标值"""
    if pd.isna(val):
        return "缺失"
    if ind == "bmi":
        return f"{val:.1f}"
    if ind == "vital_capacity":
        return f"{val:.0f}ml"
    if ind == "sprint_50m":
        return f"{val:.1f}秒"
    if ind == "standing_long_jump":
        return f"{val:.0f}cm"
    if ind == "sit_and_reach":
        return f"{val:.1f}cm"
    if ind == "endurance_run_sec":
        m, s = divmod(int(val), 60)
        return f"{m}分{s}秒"
    if ind == "strength":
        return f"{val:.1f}"
    return str(val)

def student_profile_str(row, year="g4"):
    """生成学生体测概况字符串"""
    parts = []
    for ind in ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
                "sit_and_reach", "endurance_run_sec", "strength"]:
        col = f"{ind}_{year}"
        if col in row and not pd.isna(row[col]):
            parts.append(f"{INDICATOR_CN[ind]}={fmt_value(ind, row[col])}")
    return "，".join(parts)

def build_classification_tasks(df, labels_df, n=3000):
    """任务1：分类任务"""
    tasks = []
    merged = df.merge(labels_df[["student_id", "rule_class"]], on="student_id", how="inner")
    merged = merged.dropna(subset=[f"bmi_g4", f"vital_capacity_g4"])
    samples = merged.sample(n=min(n, len(merged)), random_state=42)

    for _, row in samples.iterrows():
        profile = student_profile_str(row, "g4")
        gender = "男" if row.get("gender", 1) == 1 else "女"
        grade = f"{int(row.get('enrollment_year', 2020))}级"
        cls = int(row["rule_class"])
        cls_name = CLASS_NAMES[cls]

        instruction = f"以下是一名{gender}大学生（{grade}）的最新体测数据：{profile}。请判断该学生的健康状况属于以下哪一类：{'、'.join(CLASS_NAMES)}。"
        output = f"该学生属于「{cls_name}」。依据：最新体测总分约{row.get('total_score_g4', 0):.0f}分，HI趋势为{'退化' if cls % 3 == 0 else '平稳' if cls % 3 == 1 else '改善'}。"

        tasks.append({"instruction": instruction, "output": output, "task_type": "classification"})
    return tasks

def build_prediction_tasks(df, n=2000):
    """任务2：预测任务（大一大二→大四是否退化）"""
    tasks = []
    valid = df.dropna(subset=["bmi_g1", "bmi_g2", "bmi_g4", "HI_g1", "HI_g2", "HI_g4"])
    samples = valid.sample(n=min(n, len(valid)), random_state=43)

    for _, row in samples.iterrows():
        g1_profile = student_profile_str(row, "g1")
        g2_profile = student_profile_str(row, "g2")
        gender = "男" if row.get("gender", 1) == 1 else "女"

        hi_g1, hi_g2, hi_g4 = row["HI_g1"], row["HI_g2"], row["HI_g4"]
        is_decline = hi_g4 < hi_g2 - 0.1
        trend = "退化" if is_decline else "未退化"

        instruction = f"以下是一名{gender}大学生大一和大二的体测数据：\n大一：{g1_profile}\n大二：{g2_profile}\n请预测该学生大四时健康状况是否会退化（HI显著下降）。"
        output = f"预测：该学生大四时{trend}。依据：大一HI={hi_g1:.3f}，大二HI={hi_g2:.3f}，趋势为{'下降' if hi_g2 < hi_g1 else '上升' if hi_g2 > hi_g1 else '平稳'}，大四实际HI={hi_g4:.3f}。"

        tasks.append({"instruction": instruction, "output": output, "task_type": "prediction"})
    return tasks

def build_recommendation_tasks(df, n=2000):
    """任务3：推荐任务"""
    tasks = []
    valid = df.dropna(subset=["bmi_g4", "vital_capacity_g4"])
    samples = valid.sample(n=min(n, len(valid)), random_state=44)

    rec_rules = [
        ("bmi", "BMI偏高", "建议进行有氧运动（跑步、游泳）配合饮食控制，每周4-5次，每次40分钟以上。"),
        ("vital_capacity", "肺活量偏低", "建议进行心肺功能训练，如慢跑、游泳、深呼吸练习，逐步提升有氧能力。"),
        ("sprint_50m", "50米跑偏慢", "建议进行爆发力训练，如短距离冲刺、蛙跳、高抬腿，每周2-3次。"),
        ("standing_long_jump", "立定跳远偏近", "建议进行下肢力量训练，如深蹲、弓步蹲、跳箱，配合爆发力练习。"),
        ("sit_and_reach", "坐位体前屈偏小", "建议进行柔韧训练，如静态拉伸、瑜伽、腘绳肌拉伸，每天10-15分钟。"),
        ("endurance_run_sec", "耐力跑偏慢", "建议进行耐力训练，如长跑、间歇跑，每周3次，逐步增加跑量。"),
        ("strength", "力量偏弱", "建议进行力量训练，如引体向上、俯卧撑、哑铃训练，每周2-3次。"),
    ]

    for _, row in samples.iterrows():
        profile = student_profile_str(row, "g4")
        gender = "男" if row.get("gender", 1) == 1 else "女"

        # 找最弱指标（简单用z-score思路，这里直接选最低分的）
        weakest = None
        weakest_score = float("inf")
        for ind, _, _ in rec_rules:
            col = f"{ind}_g4"
            if col in row and not pd.isna(row[col]):
                # 简单归一化比较
                val = row[col]
                if ind in ["sprint_50m", "endurance_run_sec"]:  # 越小越好
                    score = val
                else:  # 越大越好，取负
                    score = -val
                if score < weakest_score:
                    weakest_score = score
                    weakest = ind

        if weakest:
            for ind, problem, advice in rec_rules:
                if ind == weakest:
                    instruction = f"以下是一名{gender}大学生的体测数据：{profile}。请针对该学生的薄弱环节给出个性化运动建议。"
                    output = f"该学生主要薄弱环节是{problem}。{advice}同时建议保持均衡饮食和充足睡眠，训练前做好热身，避免运动损伤。"
                    tasks.append({"instruction": instruction, "output": output, "task_type": "recommendation"})
                    break
    return tasks

def build_knowledge_qa(n=1000):
    """任务4：知识问答（从知识库提取）"""
    # 这里用一些通用健康知识问答，后续可以从chunks提取
    qa_pairs = [
        ("大学生体测包括哪些项目？", "大学生体测通常包括：BMI（身体质量指数）、肺活量、50米跑、坐位体前屈、立定跳远、耐力跑（男生1000米/女生800米）、引体向上（男）/仰卧起坐（女）等项目。"),
        ("BMI的正常范围是多少？", "我国健康成年人的BMI正常范围为18.5-23.9 kg/m²。BMI<18.5为偏瘦，24.0-27.9为超重，≥28为肥胖。"),
        ("如何提高肺活量？", "提高肺活量的方法包括：1）有氧运动如慢跑、游泳、骑行；2）深呼吸练习和腹式呼吸；3）吹奏乐器；4）坚持每周3-4次、每次30分钟以上的中等强度有氧运动。"),
        ("运动前为什么要热身？", "运动前热身可以：1）提高肌肉温度，预防肌肉拉伤；2）增加关节活动度；3）提高心率和血液循环；4）激活神经系统，提升运动表现。建议热身5-10分钟。"),
        ("如何科学增加肌肉力量？", "科学增肌的要点：1）渐进超负荷训练，逐步增加重量；2）复合动作为主（深蹲、硬拉、卧推）；3）每组8-12次，3-4组；4）保证蛋白质摄入（1.6-2.2g/kg体重）；5）充足睡眠和恢复。"),
        ("耐力跑的训练方法有哪些？", "耐力跑训练方法包括：1）持续慢跑（心率控制在最大心率的60-70%）；2）间歇跑（快跑+慢跑交替）；3）法特莱克跑（变速跑）；4）长距离慢跑（LSD）。建议每周3次，逐步增加跑量不超过10%。"),
        ("坐位体前屈如何提升？", "提升坐位体前屈的方法：1）静态拉伸腘绳肌和腰背，每次20-30秒；2）瑜伽中的前屈式；3）每天坚持拉伸10-15分钟；4）配合热水浴后拉伸效果更好。"),
        ("运动后如何正确恢复？", "运动后恢复要点：1）静态拉伸5-10分钟；2）补充水分和电解质；3）运动后30分钟内补充蛋白质和碳水；4）保证7-9小时睡眠；5）高强度训练后安排休息日。"),
        ("肥胖学生如何开始运动？", "肥胖学生开始运动建议：1）从低冲击运动开始（快走、游泳、骑行）；2）每周3-4次，每次20-30分钟，逐步增加；3）配合饮食控制，制造热量缺口；4）避免高强度跳跃运动保护关节；5）定期监测体重和体脂。"),
        ("大学生每周建议运动多少次？", "根据《中国人群身体活动指南》，成年人每周应进行至少150分钟中等强度有氧运动，或75分钟高强度有氧运动，相当于每周3-5次、每次30分钟以上。同时建议每周2次力量训练。"),
    ]

    tasks = []
    random.seed(45)
    for _ in range(n):
        q, a = random.choice(qa_pairs)
        tasks.append({"instruction": q, "output": a, "task_type": "knowledge_qa"})
    return tasks

def main():
    print("加载数据...")
    df = pd.read_csv(HI_WIDE)
    labels_df = pd.read_csv(LABELS)
    print(f"  学生数据: {len(df)} 行")
    print(f"  标签数据: {len(labels_df)} 行")

    print("\n构造分类任务...")
    cls_tasks = build_classification_tasks(df, labels_df, n=10000)
    print(f"  生成 {len(cls_tasks)} 条")

    print("构造预测任务...")
    pred_tasks = build_prediction_tasks(df, n=6000)
    print(f"  生成 {len(pred_tasks)} 条")

    print("构造推荐任务...")
    rec_tasks = build_recommendation_tasks(df, n=6000)
    print(f"  生成 {len(rec_tasks)} 条")

    print("构造知识问答...")
    qa_tasks = build_knowledge_qa(n=1000)
    print(f"  生成 {len(qa_tasks)} 条")

    all_tasks = cls_tasks + pred_tasks + rec_tasks + qa_tasks
    random.shuffle(all_tasks)

    # 划分训练/验证
    n_val = int(len(all_tasks) * 0.05)
    val_tasks = all_tasks[:n_val]
    train_tasks = all_tasks[n_val:]

    # 保存
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for t in train_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    val_path = OUTPUT.parent / "instruction_dataset_val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for t in val_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_tasks)} 条")
    print(f"  验证集: {len(val_tasks)} 条")
    print(f"  任务类型分布:")
    from collections import Counter
    type_counts = Counter(t["task_type"] for t in all_tasks)
    for t, c in type_counts.most_common():
        print(f"    {t}: {c} ({c/len(all_tasks)*100:.1f}%)")
    print(f"\n保存到: {OUTPUT}")

if __name__ == "__main__":
    main()
