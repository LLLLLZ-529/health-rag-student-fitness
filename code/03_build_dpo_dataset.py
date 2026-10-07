#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_dpo_dataset.py — 构造DPO偏好数据集

为每个学生生成：
- chosen: 针对最弱指标的精准推荐（好）
- rejected: 随机/通用推荐（差）
"""
import json, random
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).parents[4]
HI_WIDE = BASE / "hi_wide.csv"
OUTPUT = Path(__file__).parent / "dpo_dataset.jsonl"

INDICATOR_CN = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
    "endurance_run_sec": "耐力跑", "strength": "力量",
}

GOOD_ADVICE = {
    "bmi": "建议进行有氧运动（跑步、游泳）配合饮食控制，每周4-5次，每次40分钟以上，同时减少高糖高脂食物摄入。",
    "vital_capacity": "建议进行心肺功能训练，如慢跑、游泳、深呼吸练习，逐步提升有氧能力，每周3-4次。",
    "sprint_50m": "建议进行爆发力训练，如短距离冲刺、蛙跳、高抬腿，每周2-3次，注意充分热身。",
    "standing_long_jump": "建议进行下肢力量训练，如深蹲、弓步蹲、跳箱，配合爆发力练习，每周2-3次。",
    "sit_and_reach": "建议进行柔韧训练，如静态拉伸、瑜伽、腘绳肌拉伸，每天10-15分钟，运动后拉伸效果更佳。",
    "endurance_run_sec": "建议进行耐力训练，如长跑、间歇跑，每周3次，逐步增加跑量不超过10%。",
    "strength": "建议进行力量训练，如引体向上、俯卧撑、哑铃训练，每周2-3次，保证蛋白质摄入。",
}

BAD_ADVICE = [
    "建议多运动，保持健康。",
    "建议每天跑步，越多越好。",
    "建议去健身房办卡，跟着教练练。",
    "建议少吃点，自然就瘦了。",
    "建议多吃蛋白粉，增肌很快。",
    "建议每天运动2小时，坚持就是胜利。",
    "建议随便练练，体测不重要。",
    "建议只做仰卧起坐，全身都能练到。",
]

def fmt_value(ind, val):
    if pd.isna(val): return "缺失"
    if ind == "bmi": return f"{val:.1f}"
    if ind == "vital_capacity": return f"{val:.0f}ml"
    if ind == "sprint_50m": return f"{val:.1f}秒"
    if ind == "standing_long_jump": return f"{val:.0f}cm"
    if ind == "sit_and_reach": return f"{val:.1f}cm"
    if ind == "endurance_run_sec":
        m, s = divmod(int(val), 60); return f"{m}分{s}秒"
    if ind == "strength": return f"{val:.1f}"
    return str(val)

def main():
    print("加载数据...")
    df = pd.read_csv(HI_WIDE)
    valid = df.dropna(subset=["bmi_g4", "vital_capacity_g4"])
    samples = valid.sample(n=min(10000, len(valid)), random_state=42)
    print(f"采样 {len(samples)} 名学生")

    dataset = []
    for _, row in samples.iterrows():
        # 找最弱指标
        weakest = None
        weakest_score = float("inf")
        for ind in INDICATOR_CN:
            col = f"{ind}_g4"
            if col in row and not pd.isna(row[col]):
                val = row[col]
                score = val if ind in ["sprint_50m", "endurance_run_sec"] else -val
                if score < weakest_score:
                    weakest_score = score; weakest = ind

        if not weakest: continue

        # 生成学生概况
        gender = "男" if row.get("gender", 1) == 1 else "女"
        parts = []
        for ind in INDICATOR_CN:
            col = f"{ind}_g4"
            if col in row and not pd.isna(row[col]):
                parts.append(f"{INDICATOR_CN[ind]}={fmt_value(ind, row[col])}")
        profile = "，".join(parts)

        prompt = f"以下是一名{gender}大学生的体测数据：{profile}。请针对该学生的薄弱环节给出个性化运动建议。"

        # chosen: 精准推荐
        chosen = f"该学生主要薄弱环节是{INDICATOR_CN[weakest]}。{GOOD_ADVICE[weakest]}同时建议保持均衡饮食和充足睡眠，训练前做好热身，避免运动损伤。"

        # rejected: 随机差推荐
        rejected = random.choice(BAD_ADVICE)

        dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "weakest_indicator": weakest,
        })

    random.shuffle(dataset)
    n_val = int(len(dataset) * 0.05)
    val_data = dataset[:n_val]
    train_data = dataset[n_val:]

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(Path(__file__).parent / "dpo_dataset_val.jsonl", "w", encoding="utf-8") as f:
        for d in val_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nDPO数据集构造完成:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    print(f"  保存到: {OUTPUT}")

if __name__ == "__main__":
    main()
