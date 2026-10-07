#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03b_build_dpo_hardneg.py — 构造hard negative DPO数据集

三类hard negative（长度~70字，看似专业但犯具体错误）：
A. 张冠李戴：把薄弱指标识别错（肺活量弱→说力量弱，推荐抗阻）
B. 错误方法：识别对了指标，但推荐了不匹配的训练方式
C. 方向矛盾：与学生身体条件矛盾的建议（BMI过胖推荐增肌等）
"""
import json, re, random
from pathlib import Path

BASE = Path(__file__).parent
random.seed(42)

# 读原始数据
with open(BASE / "dpo_dataset.jsonl") as f:
    raw = [json.loads(l) for l in f]

def parse_prompt(prompt):
    """从prompt提取关键指标"""
    info = {}
    m = re.search(r"BMI=([\d.]+)", prompt); info["bmi"] = float(m.group(1)) if m else 22
    m = re.search(r"肺活量=(\d+)ml", prompt); info["vc"] = int(m.group(1)) if m else 3000
    m = re.search(r"力量=([\d.]+)", prompt); info["str"] = float(m.group(1)) if m else 15
    m = re.search(r"50米跑=([\d.]+)秒", prompt); info["speed"] = float(m.group(1)) if m else 9
    return info

# 三类hard negative模板
HARD_TEMPLATES = {
    "A_misidentify": [
        "该学生主要薄弱环节是力量。建议进行抗阻训练，如深蹲、硬拉、卧推，每周3次逐步加重，同时配合高蛋白饮食促进肌肉生长。",
        "该学生主要薄弱环节是柔韧性。建议以静态拉伸和瑜伽为主，每日拉伸大腿前后侧与肩部，逐步增加关节活动度。",
        "该学生主要薄弱环节是速度。建议进行高强度间歇冲刺训练，40米全速冲刺×8组，组间充分休息，提升爆发力。",
    ],
    "B_wrong_method": [
        "该学生主要薄弱环节是肺活量。建议进行大重量深蹲与硬拉训练，每组6-8次做到力竭，通过无氧代谢刺激呼吸肌生长。",
        "该学生主要薄弱环节是肺活量。建议进行高强度间歇训练（HIIT），全力冲刺30秒休息30秒循环20组，快速提升心肺极限。",
        "该学生主要薄弱环节是肺活量。建议进行大负荷负重呼吸训练，每天屏气1分钟以上，强行扩张胸腔容量。",
    ],
    "C_contradict": [
        # BMI 偏高 → 推荐增肌增重
        lambda info: f"该学生当前体重基数合理，建议以增肌为目标：每天多吃一餐，蛋白质摄入提升到每公斤体重2克，进行大重量抗阻训练增加肌肉量。",
        # 肺活量低 → 不建议有氧反而推荐静养
        "该学生主要薄弱环节是肺活量，但考虑到心肺负荷，建议以静养恢复为主，减少户外活动，避免缺氧带来的运动风险。",
        # 通用错误：建议"越多越好"式极端训练
        "该学生主要薄弱环节是肺活量。建议每天跑步10公里以上，训练量越大进步越快，无需考虑身体反应。",
    ],
}

new_pairs = []
for d in raw:
    prompt = d["prompt"]
    chosen = d["chosen"]
    info = parse_prompt(prompt)

    # 根据BMI选更有针对性的C类
    cat = random.choice(["A", "A", "B", "B", "C"])
    if cat == "A":
        rejected = random.choice(HARD_TEMPLATES["A_misidentify"])
    elif cat == "B":
        rejected = random.choice(HARD_TEMPLATES["B_wrong_method"])
    else:
        if info["bmi"] > 24:
            # BMI偏高 → 增肌矛盾建议
            rejected = HARD_TEMPLATES["C_contradict"][0](info)
        else:
            rejected = random.choice(HARD_TEMPLATES["C_contradict"][1:])

    new_pairs.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    })

# 打乱
random.shuffle(new_pairs)
n_train = int(len(new_pairs) * 0.95)
train = new_pairs[:n_train]
val = new_pairs[n_train:]

# 写文件
with open(BASE / "dpo_hardneg.jsonl", "w", encoding="utf-8") as f:
    for p in train:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(BASE / "dpo_hardneg_val.jsonl", "w", encoding="utf-8") as f:
    for p in val:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"训练集: {len(train)} 条")
print(f"验证集: {len(val)} 条")
print()
print("=== 样本预览 ===")
for i in [0, 100, 500]:
    p = new_pairs[i]
    print(f"[{i}] PROMPT: {p['prompt'][:90]}...")
    print(f"    CHOSEN:   {p['chosen'][:90]}")
    print(f"    REJECTED: {p['rejected'][:90]}")
    print()
