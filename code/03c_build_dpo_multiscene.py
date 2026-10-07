#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03c_build_dpo_multiscene.py — 多场景 hard negative DPO 数据集

5个场景：肺活量弱 / 力量弱 / 耐力弱 / 柔韧弱 / BMI偏胖
每个场景 chosen=正确处方，rejected=hard negative（张冠李戴/错误方法/方向矛盾）
"""
import json, re, random
from pathlib import Path

BASE = Path(__file__).parent
random.seed(42)

# 读原始 prompt 池
with open(BASE / "dpo_dataset.jsonl") as f:
    raw = [json.loads(l) for l in f]

def parse_prompt(prompt):
    info = {}
    m = re.search(r"BMI=([\d.]+)", prompt); info["bmi"] = float(m.group(1)) if m else 22
    m = re.search(r"肺活量=(\d+)ml", prompt); info["vc"] = int(m.group(1)) if m else 3000
    m = re.search(r"50米跑=([\d.]+)秒", prompt); info["speed"] = float(m.group(1)) if m else 9
    m = re.search(r"立定跳远=(\d+)cm", prompt); info["jump"] = int(m.group(1)) if m else 180
    m = re.search(r"坐位体前屈=([\d.]+)cm", prompt); info["flex"] = float(m.group(1)) if m else 10
    m = re.search(r"耐力跑=(\d+)分(\d+)秒", prompt)
    if m: info["endurance"] = int(m.group(1))*60 + int(m.group(2))
    else: info["endurance"] = 300
    m = re.search(r"力量=([\d.]+)", prompt); info["str"] = float(m.group(1)) if m else 15
    return info

def detect_scene(info):
    """根据指标判断主要薄弱场景（女生标准）"""
    scores = []
    if info["vc"] < 2800: scores.append(("vc", 2800 - info["vc"]))
    if info["str"] < 10: scores.append(("str", 10 - info["str"]))
    if info["endurance"] > 270: scores.append(("end", info["endurance"] - 270))
    if info["flex"] < 5: scores.append(("flex", 5 - info["flex"]))
    if info["bmi"] > 24: scores.append(("bmi", info["bmi"] - 24))
    if not scores:
        # 没有明显弱项，归到肺活量（兜底）
        return "vc"
    scores.sort(key=lambda x: -x[1])
    return scores[0][0]

# 5个场景的正确处方（chosen）
CHOSEN = {
    "vc": "该学生主要薄弱环节是肺活量偏低。建议进行心肺功能训练，如慢跑、游泳、深呼吸练习，逐步提升有氧能力，每周3-4次。同时建议保持均衡饮食和充足睡眠，训练前做好热身，避免运动损伤。",
    "str": "该学生主要薄弱环节是力量不足。建议进行抗阻训练，如深蹲、硬拉、卧推，每周3次逐步增加重量，同时配合高蛋白饮食促进肌肉生长。训练前充分热身，避免关节损伤。",
    "end": "该学生主要薄弱环节是耐力不足。建议进行有氧耐力训练，如慢跑、骑行、游泳，从低强度开始逐步延长时间，每周3-4次。同时注意呼吸节奏，循序渐进避免过度疲劳。",
    "flex": "该学生主要薄弱环节是柔韧性不足。建议每日进行静态拉伸，重点拉伸大腿前后侧、肩部和腰部，每个动作保持30秒，配合瑜伽练习逐步增加关节活动度。训练前充分热身。",
    "bmi": "该学生BMI偏高，建议以有氧运动为主（快走、慢跑、游泳），每周4次每次40分钟以上，配合饮食控制减少高热量摄入，逐步降低体重。不建议大重量力量训练以免增加关节负担。",
}

# 5个场景的 hard negative（rejected）
HARD_REJ = {
    # 肺活量弱：推荐力量/错误方法
    "vc": [
        "该学生主要薄弱环节是力量。建议进行大重量深蹲与硬拉训练，每组6-8次做到力竭，通过无氧代谢提升整体体能，配合高蛋白饮食。",
        "该学生主要薄弱环节是肺活量。建议每天屏气1分钟以上强行扩张胸腔，并进行大负荷负重呼吸训练，快速提升肺活量。",
        "该学生主要薄弱环节是柔韧性。建议以静态拉伸和瑜伽为主，每日拉伸全身，逐步增加关节活动度。",
    ],
    # 力量弱：推荐有氧/错误方法
    "str": [
        "该学生主要薄弱环节是肺活量。建议进行长时间有氧慢跑，每天10公里以上，通过大量燃脂改善体型。",
        "该学生主要薄弱环节是耐力。建议进行高强度间歇冲刺训练，全力冲刺30秒休息30秒循环20组，提升心肺能力。",
        "该学生主要薄弱环节是力量。建议每天做100个俯卧撑做到力竭，不需要渐进负重，自然就能长肌肉。",
    ],
    # 耐力弱：推荐力量/错误方法
    "end": [
        "该学生主要薄弱环节是力量。建议进行大重量深蹲硬拉，每组3次做到力竭，通过绝对力量提升整体运动表现。",
        "该学生主要薄弱环节是耐力。建议每天全力冲刺跑10组，每组400米，不考虑恢复时间，练得越狠进步越快。",
        "该学生主要薄弱环节是柔韧性。建议以静态拉伸和瑜伽为主，不做有氧训练，避免心肺负担。",
    ],
    # 柔韧弱：推荐力量/错误方法
    "flex": [
        "该学生主要薄弱环节是力量。建议进行大重量抗阻训练，深蹲硬拉卧推，不需要做拉伸，肌肉自然会变软。",
        "该学生主要薄弱环节是肺活量。建议进行高强度间歇训练，全力冲刺30秒循环20组，通过出汗改善柔韧性。",
        "该学生主要薄弱环节是柔韧性。建议做弹震式拉伸，用力压到极限弹动，快速拉开肌肉。",
    ],
    # BMI偏胖：推荐增肌/错误方法
    "bmi": [
        "该学生当前体重基数合理，建议以增肌为目标：每天多吃一餐，蛋白质摄入每公斤体重2克，进行大重量抗阻训练增加肌肉量。",
        "该学生BMI偏高。建议每天节食只吃500大卡，配合高强度间歇训练，两周快速减重10公斤。",
        "该学生主要薄弱环节是力量。建议进行大重量深蹲硬拉卧推，大重量训练能快速燃脂，不需要控制饮食。",
    ],
}

# 为每个 prompt 生成多场景对
all_pairs = []
scene_count = {"vc":0, "str":0, "end":0, "flex":0, "bmi":0}

for d in raw:
    prompt = d["prompt"]
    info = parse_prompt(prompt)
    scene = detect_scene(info)
    scene_count[scene] += 1
    chosen = CHOSEN[scene]
    rejected = random.choice(HARD_REJ[scene])
    all_pairs.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "scene": scene,
    })

print("各场景分布:")
for k,v in scene_count.items():
    print(f"  {k}: {v}")

# 按场景均衡采样：每场景 280 条，共 1400 条
random.shuffle(all_pairs)
balanced = []
per_scene_target = 280
scene_taken = {"vc":0, "str":0, "end":0, "flex":0, "bmi":0}
for p in all_pairs:
    s = p["scene"]
    if scene_taken[s] < per_scene_target:
        balanced.append(p)
        scene_taken[s] += 1
    if len(balanced) >= 1400:
        break

# 去掉 scene 字段（训练不需要）
final = [{"prompt":p["prompt"], "chosen":p["chosen"], "rejected":p["rejected"]} for p in balanced]
random.shuffle(final)

n_val = int(len(final) * 0.05)
train = final[n_val:]
val = final[:n_val]

with open(BASE / "dpo_multiscene.jsonl", "w", encoding="utf-8") as f:
    for p in train:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(BASE / "dpo_multiscene_val.jsonl", "w", encoding="utf-8") as f:
    for p in val:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\n训练集: {len(train)}, 验证集: {len(val)}")
print(f"实际采样: {scene_taken}")

# 预览每个场景1个
print("\n=== 各场景样本预览 ===")
seen = set()
for p in balanced:
    s = p.get("scene", "?")
    if s not in seen:
        seen.add(s)
        print(f"\n--- 场景: {s} ---")
        print(f"PROMPT: {p['prompt'][:100]}...")
        print(f"CHOSEN: {p['chosen'][:80]}")
        print(f"REJECT: {p['rejected'][:80]}")
