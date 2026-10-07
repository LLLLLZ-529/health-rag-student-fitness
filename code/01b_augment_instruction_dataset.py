#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01b_augment_instruction_dataset.py — 数据多样性增强
为每条 instruction 生成多种提问方式和输出表述，避免模板单一。
"""
import json, random, re
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE / "instruction_dataset.jsonl"
VAL_INPUT = BASE / "instruction_dataset_val.jsonl"
OUTPUT = BASE / "instruction_dataset_aug.jsonl"
VAL_OUTPUT = BASE / "instruction_dataset_aug_val.jsonl"

random.seed(42)

# ============ 分类任务：多种提问方式 ============
CLASS_PROMPTS = [
    "以下是一名{gender}大学生（{grade}）的最新体测数据：{profile}。请判断该学生的健康状况属于以下哪一类：{classes}。",
    "{gender}大学生（{grade}）体测结果如下：{profile}。请分析其健康状态并归类（选项：{classes}）。",
    "根据这份体测报告——{profile}，这名{gender}{grade}学生属于哪种健康类型？可选：{classes}。",
    "这名{gender}学生（{grade}）的体测数据为{profile}，请评估其健康水平并从以下类别中选择：{classes}。",
    "体测数据分析：{gender}，{grade}，各项指标{profile}。请判断健康分类（{classes}）。",
]

CLASS_OUTPUTS = [
    "该学生属于「{cls_name}」。依据：最新体测总分约{score:.0f}分，HI趋势为{trend}。",
    "健康分类：{cls_name}。体测总分约{score:.0f}分，整体趋势{trend}。",
    "判定结果为{cls_name}。综合各项指标，总分约{score:.0f}分，趋势{trend}。",
    "这名学生是{cls_name}。主要依据：体测总分{score:.0f}分左右，HI呈{trend}态势。",
]

# ============ 预测任务：多种提问方式 ============
PRED_PROMPTS = [
    "以下是一名{gender}大学生大一和大二的体测数据：\n大一：{g1}\n大二：{g2}\n请预测该学生大四时健康状况是否会退化（HI显著下降）。",
    "{gender}学生前两年体测：大一[{g1}]，大二[{g2}]。基于这些数据，预测大四是否会出现健康退化？",
    "根据大一（{g1}）和大二（{g2}）的体测记录，这名{gender}学生大四时HI会显著下降吗？",
    "纵向数据分析：{gender}学生大一{g1}，大二{g2}。请预测大四健康走向——是否退化？",
    "这名{gender}大学生大一数据：{g1}；大二数据：{g2}。请判断大四时健康状况是否会恶化。",
]

PRED_OUTPUTS = [
    "预测：该学生大四时{trend}。依据：大一HI={hi1:.3f}，大二HI={hi2:.3f}，趋势为{direction}，大四实际HI={hi4:.3f}。",
    "结论：大四{trend}。大一HI={hi1:.3f}→大二HI={hi2:.3f}（{direction}），大四HI={hi4:.3f}。",
    "预测结果为{trend}。从HI变化看：大一{hi1:.3f}，大二{hi2:.3f}，{direction}，大四{hi4:.3f}。",
    "该学生大四将{trend}。HI轨迹：大一{hi1:.3f}→大二{hi2:.3f}（{direction}）→大四{hi4:.3f}。",
]

# ============ 推荐任务：多种提问方式 ============
REC_PROMPTS = [
    "以下是一名{gender}大学生的体测数据：{profile}。请针对该学生的薄弱环节给出个性化运动建议。",
    "{gender}学生体测报告：{profile}。请根据其弱项制定运动方案。",
    "这名{gender}大学生的体测结果为{profile}，请给出有针对性的锻炼建议。",
    "体测数据分析：{gender}，{profile}。请指出薄弱环节并推荐相应的训练方法。",
    "根据以下体测数据——{profile}，为这名{gender}学生设计个性化运动处方。",
]

REC_OUTPUTS = [
    "该学生主要薄弱环节是{problem}。{advice}同时建议保持均衡饮食和充足睡眠，训练前做好热身，避免运动损伤。",
    "薄弱项：{problem}。建议：{advice}此外注意饮食均衡、保证睡眠，运动前充分热身。",
    "针对{problem}，推荐：{advice}配合合理饮食和休息，训练前热身预防损伤。",
    "核心问题是{problem}。{advice}日常还需注意营养均衡、作息规律，做好运动防护。",
]

# ============ 知识问答：扩充题库 ============
EXTRA_QA = [
    ("体测不及格怎么办？", "体测不及格首先不要焦虑，应针对薄弱项目制定训练计划：1）每周安排3-4次专项训练；2）循序渐进，从低强度开始；3）记录每次成绩观察进步；4）必要时请教体育老师；5）保证营养和休息。大多数学生通过1-2学期系统训练都能达标。"),
    ("立定跳远怎么练？", "立定跳远提升方法：1）下肢力量训练——深蹲、弓步蹲、跳箱，每周3次；2）爆发力训练——蛙跳、原地纵跳、短距离冲刺；3）技术动作——预摆协调、蹬地有力、空中收腿、落地缓冲；4）核心稳定性——平板支撑、卷腹。建议每周3-4次，每次30-45分钟。"),
    ("引体向上拉不上去怎么办？", "引体向上进阶路径：1）从悬挂开始——每次悬挂30-60秒，增强握力和背肌耐力；2）弹力带辅助引体——逐步减助力；3）离心训练——跳上去后缓慢放下（3-5秒）；4）澳式引体（低杠水平拉）；5）配合哑铃划船、高位下拉。每周3次，坚持8-12周可见明显进步。"),
    ("800/1000米跑怎么提高？", "中长跑训练方案：1）耐力基础——每周2次持续慢跑（20-30分钟，心率60-70%）；2）间歇训练——400米快跑+200米慢走，重复6-8组，每周1次；3）节奏跑——以比目标配速慢5-10秒的速度跑15-20分钟；4）力量训练——深蹲、弓步、核心；5）跑前热身、跑后拉伸。每周3-4次，8周为一个周期。"),
    ("坐位体前屈很差怎么改善？", "柔韧性提升计划：1）每天静态拉伸腘绳肌、腰背、小腿，每个动作保持20-30秒，做3组；2）瑜伽中的前屈式、坐角式；3）运动后（体温升高时）拉伸效果最佳；4）配合热水浴后拉伸；5）避免弹震式拉伸。坚持每天10-15分钟，4-6周可见改善。"),
    ("BMI偏高如何健康减重？", "科学减重要点：1）制造热量缺口——每天减少300-500大卡，不低于基础代谢；2）饮食结构——增加蛋白质（1.2-1.6g/kg）和蔬菜，减少精制碳水和含糖饮料；3）运动——每周3-5次有氧运动（快走、游泳、骑行），每次30-45分钟，配合2次力量训练；4）睡眠——保证7-8小时；5）每周减重0.5-1kg为宜，不追求快速减重。"),
    ("运动后肌肉酸痛正常吗？", "运动后24-72小时出现的肌肉酸痛称为延迟性肌肉酸痛（DOMS），是正常现象，说明肌肉在适应新的运动负荷。缓解方法：1）轻度活动促进血液循环；2）静态拉伸；3）热敷或温水浴；4）保证蛋白质摄入；5）充足休息。如果疼痛剧烈伴肿胀或无力，可能是肌肉拉伤，需就医。"),
    ("如何制定一周运动计划？", "大学生一周运动计划示例：周一——力量训练（上肢+核心）45分钟；周二——有氧运动（慢跑/骑行）30分钟；周三——休息或轻度拉伸；周四——力量训练（下肢+核心）45分钟；周五——有氧运动（游泳/跳绳）30分钟；周六——球类运动或户外徒步1小时；周日——休息。每周至少150分钟中等强度有氧+2次力量训练。"),
    ("体测前需要注意什么？", "体测前准备：1）提前2-3周开始针对性训练，不要临时抱佛脚；2）测试前一天保证充足睡眠（7-9小时）；3）测试前2小时进食易消化食物（如香蕉、面包），不要空腹或过饱；4）测试前15-20分钟充分热身（慢跑+动态拉伸）；5）穿着合适的运动服和运动鞋；6）保持良好心态，按平时训练节奏发挥。"),
    ("肺活量低说明什么？", "肺活量低可能反映：1）心肺功能较弱——缺乏有氧运动；2）呼吸肌力量不足；3）体型因素（身高体重影响）；4）测试方法不当（未充分吸气或呼气）。提升方法：有氧运动（慢跑、游泳、骑行）每周3-4次；深呼吸练习和腹式呼吸每天5-10分钟；吹奏乐器也有帮助。坚持3-6个月可明显提升。"),
    ("女生仰卧起坐怎么练？", "仰卧起坐提升方法：1）核心基础——平板支撑（30-60秒×3组）、死虫式、鸟狗式；2）卷腹训练——注意用腹部发力而非手拉脖子；3）俄罗斯转体——增强腹斜肌；4）腿部抬高卷腹——针对下腹部；5）每周3-4次，每次15-20分钟。注意：仰卧起坐对腰椎有一定压力，卷腹是更安全的替代动作。"),
    ("运动减肥多久能看到效果？", "运动减肥的见效时间因人而异：1）体重变化——通常坚持4-6周后开始下降，每周0.5-1kg；2）体型变化——2-3周后可能感觉衣服变松（体脂下降先于体重）；3）体能提升——1-2周即可感觉运动能力提高；4）关键是坚持——3个月为一个评估周期。注意配合饮食控制，单纯运动不控制饮食效果有限。"),
]


def augment_classification(item):
    """增强分类任务"""
    results = []
    # 解析原始 instruction 中的信息
    instr = item["instruction"]
    out = item["output"]

    # 提取 gender, grade, profile
    m = re.search(r'一名(\S)大学生（(\S+)）的最新体测数据：(.+?)。请判断', instr)
    if not m:
        return [item]
    gender, grade, profile = m.group(1), m.group(2), m.group(3)

    # 提取 output 中的信息
    m2 = re.search(r'总分约(\d+)分，HI趋势为(\S+)', out)
    if not m2:
        return [item]
    score = int(m2.group(1))
    trend = m2.group(2)
    cls_name = re.search(r'「(.+?)」', out).group(1)

    classes = "、".join([
        "低水平-退化型", "低水平-平稳型", "低水平-改善型",
        "中水平-退化型", "中水平-平稳型", "中水平-改善型",
        "高水平-退化型", "高水平-平稳型", "高水平-改善型",
    ])

    # 生成多种组合
    n_prompts = min(3, len(CLASS_PROMPTS))
    n_outputs = min(2, len(CLASS_OUTPUTS))
    prompt_idxs = random.sample(range(len(CLASS_PROMPTS)), n_prompts)
    output_idxs = random.sample(range(len(CLASS_OUTPUTS)), n_outputs)

    for pi in prompt_idxs:
        for oi in output_idxs:
            new_instr = CLASS_PROMPTS[pi].format(
                gender=gender, grade=grade, profile=profile, classes=classes
            )
            new_out = CLASS_OUTPUTS[oi].format(cls_name=cls_name, score=score, trend=trend)
            results.append({"instruction": new_instr, "output": new_out, "task_type": "classification"})
    return results


def augment_prediction(item):
    """增强预测任务"""
    results = []
    instr = item["instruction"]
    out = item["output"]

    m = re.search(r'一名(\S)大学生大一和大二的体测数据：\n大一：(.+?)\n大二：(.+?)\n请预测', instr)
    if not m:
        return [item]
    gender, g1, g2 = m.group(1), m.group(2), m.group(3)

    m2 = re.search(r'大一HI=([\d.]+)，大二HI=([\d.]+)，趋势为(\S+?)，大四实际HI=([\d.]+)', out)
    if not m2:
        return [item]
    hi1, hi2, direction, hi4 = float(m2.group(1)), float(m2.group(2)), m2.group(3), float(m2.group(4))
    trend = "退化" if "退化" in out else "未退化"

    n_prompts = min(3, len(PRED_PROMPTS))
    n_outputs = min(2, len(PRED_OUTPUTS))
    prompt_idxs = random.sample(range(len(PRED_PROMPTS)), n_prompts)
    output_idxs = random.sample(range(len(PRED_OUTPUTS)), n_outputs)

    for pi in prompt_idxs:
        for oi in output_idxs:
            new_instr = PRED_PROMPTS[pi].format(gender=gender, g1=g1, g2=g2)
            new_out = PRED_OUTPUTS[oi].format(trend=trend, hi1=hi1, hi2=hi2, direction=direction, hi4=hi4)
            results.append({"instruction": new_instr, "output": new_out, "task_type": "prediction"})
    return results


def augment_recommendation(item):
    """增强推荐任务"""
    results = []
    instr = item["instruction"]
    out = item["output"]

    m = re.search(r'一名(\S)大学生的体测数据：(.+?)。请针对', instr)
    if not m:
        return [item]
    gender, profile = m.group(1), m.group(2)

    m2 = re.search(r'主要薄弱环节是(\S+?)。(.+?)同时建议', out)
    if not m2:
        return [item]
    problem, advice = m2.group(1), m2.group(2)

    n_prompts = min(3, len(REC_PROMPTS))
    n_outputs = min(2, len(REC_OUTPUTS))
    prompt_idxs = random.sample(range(len(REC_PROMPTS)), n_prompts)
    output_idxs = random.sample(range(len(REC_OUTPUTS)), n_outputs)

    for pi in prompt_idxs:
        for oi in output_idxs:
            new_instr = REC_PROMPTS[pi].format(gender=gender, profile=profile)
            new_out = REC_OUTPUTS[oi].format(problem=problem, advice=advice)
            results.append({"instruction": new_instr, "output": new_out, "task_type": "recommendation"})
    return results


def augment_knowledge_qa():
    """扩充知识问答"""
    results = []
    for q, a in EXTRA_QA:
        for _ in range(3):  # 每个问答重复3次（随机顺序）
            results.append({"instruction": q, "output": a, "task_type": "knowledge_qa"})
    return results


def main():
    print("加载原始数据集...")
    with open(INPUT, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"  原始: {len(data)} 条")

    augmented = []
    stats = {"classification": 0, "prediction": 0, "recommendation": 0, "knowledge_qa": 0}

    for item in data:
        t = item.get("task_type", "default")
        if t == "classification":
            new_items = augment_classification(item)
        elif t == "prediction":
            new_items = augment_prediction(item)
        elif t == "recommendation":
            new_items = augment_recommendation(item)
        else:
            new_items = [item]
        augmented.extend(new_items)
        stats[t] = stats.get(t, 0) + len(new_items)

    # 扩充知识问答
    extra_qa = augment_knowledge_qa()
    augmented.extend(extra_qa)
    stats["knowledge_qa"] += len(extra_qa)

    random.shuffle(augmented)

    # 划分验证集 (5%)
    n_val = int(len(augmented) * 0.05)
    val = augmented[:n_val]
    train = augmented[n_val:]

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n增强完成:")
    print(f"  训练集: {len(train)} 条")
    print(f"  验证集: {len(val)} 条")
    print(f"  任务类型分布:")
    for t, c in sorted(stats.items()):
        print(f"    {t}: {c} ({c/len(augmented)*100:.1f}%)")
    print(f"\n保存到: {OUTPUT}")


if __name__ == "__main__":
    main()
