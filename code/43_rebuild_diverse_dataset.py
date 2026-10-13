#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
43_rebuild_diverse_dataset.py — 重建多样化 SFT/DPO 数据集

解决旧数据 99.8% 模板化问题：
  1. 用 42_fix_weak_indicator 修正后的弱指标标签（不再全是肺活量）
  2. 每个弱指标建"动作池 × 频率 × 时长 × 强度"组合，个性化采样
  3. assistant 输出四步推理：诊断→分析→方案→禁忌
  4. 每条方案标注知识库来源文件
  5. 强度随弱指标百分位/BMI 调整

输出：
  data/sft_recommend_diverse.jsonl
  data/dpo_diverse.jsonl
"""
import json, random, argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
random.seed(42)

# 知识库真实引用文件
SRC_SPORT = "运动处方/_全民健身指南__国家体育总局_OCR__chunk001.md"
SRC_SPORT2 = "运动处方/_中青年科学运动指南_解读__chunk001.md"
SRC_DIET = "膳食营养/中国居民膳食指南__chunk001.md"
SRC_STANDARD = "体测标准/大学士国家体质健康测试评分标准__chunk001.md"

# ---------------------------------------------------------------------------
# 动作池：每个弱指标 → 多个动作，每个动作含参数范围
# ---------------------------------------------------------------------------
ACTION_POOL = {
    "vital_capacity": {
        "cn": "肺活量",
        "analysis": "肺活量偏低多与心肺有氧能力不足、呼吸肌力量欠缺有关，应优先发展中等强度有氧运动，配合呼吸训练，而非单纯力量训练。",
        "actions": [
            ("慢跑", ["每次25-30分钟，心率约最大心率60%-70%，每周3次",
                       "每次30-40分钟，配速以能正常说话为准，每周4次",
                       "每次35分钟，慢跑与快走交替，每周3次"]),
            ("游泳", ["每次30-40分钟，蛙泳自由泳交替，每周2次",
                     "每次40-45分钟匀速游，每周2-3次"]),
            ("跳绳", ["每组3分钟、组间休息1分钟，共5-6组，每周3次",
                     "每组2分钟、组间休息90秒，共6组，每周4次"]),
            ("快走", ["每次40-50分钟，步频约120步/分钟，每周4次",
                     "每次50-60分钟，每周5次"]),
            ("骑行", ["每次30-40分钟，中等阻力，每周2-3次",
                     "每次45分钟，平路为主，每周3次"]),
            ("间歇跑", ["快跑1分钟+慢走2分钟交替，共6-8组，每周2次",
                       "快跑400米+慢走200米交替，共6组，每周3次"]),
            ("有氧操", ["每次30-40分钟，每周3次",
                       "每次30分钟，每周4次"]),
        ],
    },
    "endurance_run_sec": {
        "cn": "耐力跑",
        "analysis": "耐力跑成绩差主要反映有氧耐力与速度耐力不足，应以持续慢跑打底，辅以间歇跑提升速度耐力，循序渐进，每周跑量增幅不超过10%。",
        "actions": [
            ("匀速长跑", ["每次20-30分钟，配速以能正常说话为准，每周3次",
                         "每次25-35分钟，每周4次"]),
            ("间歇跑", ["400米快跑+200米慢走交替，共6组，每周2次",
                       "200米快跑+200米慢走交替，共8组，每周3次"]),
            ("法特莱克跑", ["快慢交替自由跑20-30分钟，每周2-3次",
                           "根据地形变速跑30分钟，每周3次"]),
            ("游泳", ["每次30-40分钟连续游，作为交叉训练，每周2次",
                     "每次40分钟，每周2次"]),
            ("跳绳", ["每组3-5分钟，共4组，每周3次",
                     "每组3分钟，共5组，每周4次"]),
            ("骑行", ["每次30-40分钟，每周3次",
                     "每次45分钟，每周2次"]),
        ],
    },
    "strength": {
        "cn": "力量",
        "analysis": "力量偏弱（核心/上肢或下肢）应进行抗阻训练，以自重与轻负荷多次数为主，先掌握动作模式再逐步增加负荷，保证蛋白质摄入。",
        "actions": [
            ("深蹲", ["每组12-15次，做3-4组，组间休息60秒，每周3次",
                     "每组15-20次，做3组，每周3次"]),
            ("俯卧撑", ["每组8-12次，做3-4组（可跪姿起步），每周3次",
                       "每组10-15次，做3组，每周4次"]),
            ("哑铃训练", ["每组12-15次，做3组，轻负荷，每周3次",
                         "每组15次，做3组，循序渐进加重，每周3次"]),
            ("平板支撑", ["每次保持30-45秒，做3-4组，每周4次",
                         "每次保持45-60秒，做3组，每周3次"]),
            ("弓步蹲", ["每侧10-15次，做3组，每周3次",
                       "每侧12次，做3组，每周4次"]),
            ("臀桥", ["每组15-20次，做3组，每周3次",
                     "每组20次，做3组，每周4次"]),
        ],
    },
    "sprint_50m": {
        "cn": "50米跑",
        "analysis": "50米跑成绩差多与爆发力、步频和起跑反应不足有关，应做短距离冲刺与爆发性练习，注意充分热身，避免疲劳状态下练习。",
        "actions": [
            ("短距离冲刺", ["30米全力冲刺，组间休息2分钟，共6-8组，每周2次",
                           "60米冲刺，组间休息3分钟，共4-6组，每周3次"]),
            ("高抬腿", ["每组20-30秒，做4组，每周3次",
                       "每组30秒，做4组，每周4次"]),
            ("蛙跳", ["每组10-15次，做3组，每周2次",
                     "每组10次，做3组，每周3次"]),
            ("起跑练习", ["蹲踞式起跑6-8次，做2组，每周3次",
                         "听信号起跑反应练习8次，做2组，每周3次"]),
            ("跳绳", ["快速单摇2-3分钟，做4组，每周3次",
                     "快速单摇2分钟，做5组，每周4次"]),
            ("台阶跳", ["每组10-15次，做3组，每周2-3次",
                       "每组10次，做3组，每周3次"]),
        ],
    },
    "standing_long_jump": {
        "cn": "立定跳远",
        "analysis": "立定跳远依赖下肢爆发力与蹬伸协调性，应结合下肢力量训练与跳跃练习，并纠正摆臂、蹬地、落地缓冲动作。",
        "actions": [
            ("深蹲跳", ["每组10-15次，做3组，每周3次",
                       "每组10-12次，做3组，每周4次"]),
            ("弓步蹲", ["每侧10-12次，做3组，每周3次",
                       "每侧15次，做3组，每周4次"]),
            ("跳箱/跳台阶", ["每组8-10次，做3组，每周3次",
                           "每组10次，做3组，每周2次"]),
            ("提踵", ["每组20-30次，做3组，每周4次",
                     "每组30次，做3组，每周3次"]),
            ("摆臂练习", ["原地体会摆臂蹬地配合15-20次，做2组，每周3次",
                         "摆臂起跳分解练习15次，做2组，每周4次"]),
            ("蛙跳", ["连续8-10次为一组，做2-3组，每周2次",
                     "连续10次为一组，做3组，每周3次"]),
        ],
    },
    "sit_and_reach": {
        "cn": "坐位体前屈",
        "analysis": "坐位体前屈反映下肢后侧链与腰背柔韧性，应在运动后肌肉发热时进行静态拉伸，配合髋部灵活性练习，循序渐进避免弹震。",
        "actions": [
            ("腘绳肌静态拉伸", ["保持20-30秒，每侧3次，每天进行",
                               "保持30-40秒，每侧3次，每天1次"]),
            ("坐位体前屈", ["缓慢前伸保持15-20秒，做3组，每天2次",
                           "保持20-30秒，做3组，每天1次"]),
            ("猫牛式", ["每组10-15次，做4组，每周4次",
                       "每组10次，做4组，每周5次"]),
            ("髋部拉伸", ["每侧保持30-40秒，做3组，每周4次",
                         "每侧保持30秒，做3组，每周5次"]),
            ("瑜伽", ["跟练20-30分钟，每周3次",
                     "跟练30分钟，每周4次"]),
            ("动态拉伸", ["运动前进行8-10分钟，每周4次",
                         "热身时动态拉伸10分钟，每周5次"]),
        ],
    },
    "bmi": {
        "cn": "BMI（体重管理）",
        "analysis": "BMI超出正常范围，应以规律有氧运动为主配合饮食热量控制，避免大重量和高强度冲击，循序渐进，每周减重不超过0.5-1kg。",
        "actions": [
            ("快走", ["每次40-50分钟，每周4-5次",
                     "每次50-60分钟，每周5次"]),
            ("慢跑", ["每次25-30分钟，可走跑结合，每周3-4次",
                     "每次30-40分钟，每周4次"]),
            ("游泳", ["每次30-40分钟，每周3次",
                     "每次40分钟，每周3次"]),
            ("骑行", ["每次30-40分钟，每周3次",
                     "每次45分钟，每周3次"]),
            ("有氧操", ["每次30-40分钟，每周3次",
                       "每次40分钟，每周4次"]),
        ],
    },
}

# 禁忌规则
def get_contraindication(bmi, weak):
    tips = []
    if bmi is not None and bmi >= 26:
        tips.append("BMI偏高，避免大重量深蹲/硬拉和高冲击跳跃，先以低冲击有氧控制体重")
    if bmi is not None and bmi < 18.5:
        tips.append("BMI偏低，避免过度有氧消耗，应增加力量训练与营养摄入")
    if weak in ("endurance_run_sec", "sprint_50m"):
        tips.append("高强度间歇前充分热身，若出现胸闷、头晕、心悸应立即停止")
    if weak == "strength":
        tips.append("抗阻训练保持腰背中立、动作规范，不要盲目追求重量")
    if not tips:
        tips.append("运动中如出现胸痛、关节锐痛、头晕应立即停止并就医")
    return "；".join(tips)


def sample_action(weak, rng):
    """从动作池随机选 2 个动作，每个动作选一条具体表述。"""
    pool = ACTION_POOL[weak]["actions"]
    chosen = rng.sample(pool, k=min(2, len(pool)))
    return [f"{name}：{rng.choice(variants)}" for name, variants in chosen]


def build_prescription(row, weak_pct, rng):
    """生成四步推理的完整 output。"""
    weak = row["weakest_indicator"]
    pool = ACTION_POOL[weak]
    bmi = row.get("bmi_g4")
    gender = "男" if row.get("gender", 1) == 1 else "女"

    # 弱指标百分位 → 表述
    pct = weak_pct
    if pct < 15:
        level = "处于同性别人群后10%-15%，属于明显短板"
    elif pct < 30:
        level = "处于同性别人群后30%，有较大提升空间"
    else:
        level = "在同性别人群中相对偏低"

    # 诊断
    diagnosis = (
        f"【诊断】该{gender}学生最弱指标为{pool['cn']}，{level}。"
        f"综合四年趋势为{'退化' if row['HI_g4'] < row['HI_g1'] else '平稳'}，"
        f"大四BMI={bmi:.1f}。"
    )
    # 分析
    analysis = f"【分析】{pool['analysis']}"
    # 方案
    actions = sample_action(weak, rng)
    weeks = rng.choice([6, 8, 10, 12])
    plan = "【方案】" + "；".join(actions) + f"。建议持续{weeks}周，每2周复测一次。"
    # 引用（按弱指标选来源）
    src = SRC_DIET if weak == "bmi" else rng.choice([SRC_SPORT, SRC_SPORT2])
    plan += f" [来源: {src}]"
    # 禁忌
    contra = f"【禁忌】{get_contraindication(bmi, weak)}。"

    return diagnosis + "\n" + analysis + "\n" + plan + "\n" + contra


def build_profile(row):
    cn = {"bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
          "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
          "endurance_run_sec": "耐力跑", "strength": "力量"}
    parts = []
    for ind in cn:
        v = row.get(f"{ind}_g4")
        if pd.notna(v):
            if ind == "vital_capacity":
                parts.append(f"{cn[ind]}={v:.0f}ml")
            elif ind in ("sprint_50m",):
                parts.append(f"{cn[ind]}={v:.1f}秒")
            elif ind in ("endurance_run_sec",):
                m, s = divmod(int(v), 60)
                parts.append(f"{cn[ind]}={m}分{s}秒")
            elif ind == "bmi":
                parts.append(f"{cn[ind]}={v:.1f}")
            else:
                parts.append(f"{cn[ind]}={v:.0f}")
    return "，".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="生成样本数")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "hi_wide.csv")
    fixed = pd.read_csv(ROOT / "data" / "weak_indicator_fixed.csv")
    df = df.merge(fixed[["student_id", "weakest_indicator", "weakest_pct"]], on="student_id")
    # 只针对退化学生
    declined = df[df["HI_g4"] < df["HI_g1"]].dropna(subset=["bmi_g4", "weakest_indicator"])
    declined = declined.sample(n=min(args.n, len(declined)), random_state=42)
    print(f"退化学生 {len(declined)} 名，开始生成多样化数据...")

    rng = random.Random(42)
    sft_out, dpo_out = [], []
    cn = {w: ACTION_POOL[w]["cn"] for w in ACTION_POOL}

    for _, row in declined.iterrows():
        weak = row["weakest_indicator"]
        profile = build_profile(row)
        gender = "男" if row.get("gender", 1) == 1 else "女"
        instruction = f"以下是一名{gender}大学生的体测数据：{profile}。请针对该学生的薄弱环节给出个性化运动建议。"
        output = build_prescription(row, row["weakest_pct"], rng)

        sft_out.append({"instruction": instruction, "output": output,
                        "task_type": "recommendation"})

        # DPO：chosen=四步个性化；rejected=通用模板或错误弱指标
        rejected = rng.choice([
            "建议多运动，保持健康。",
            "建议每天跑步，越多越好。",
            "建议多吃蛋白粉，增肌很快。",
            "建议随便练练，体测不重要。",
        ])
        # 50% 用错误弱指标的方案作为 hard negative
        if rng.random() < 0.5:
            other = rng.choice([w for w in ACTION_POOL if w != weak])
            rejected = f"该学生主要薄弱环节是{cn[other]}。{ACTION_POOL[other]['analysis']}"
        dpo_out.append({"prompt": instruction, "chosen": output,
                        "rejected": rejected, "weakest_indicator": weak})

    # 9:1
    n_val = int(len(sft_out) * 0.1)
    sft_train, sft_val = sft_out[n_val:], sft_out[:n_val]
    dpo_train, dpo_val = dpo_out[n_val:], dpo_out[:n_val]

    for name, data in [("sft_recommend_diverse_train.jsonl", sft_train),
                        ("sft_recommend_diverse_val.jsonl", sft_val),
                        ("dpo_diverse_train.jsonl", dpo_train),
                        ("dpo_diverse_val.jsonl", dpo_val)]:
        with open(ROOT / "data" / name, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  写出 {name}: {len(data)} 条")

    # 快速统计模板化率
    unique = len(set(d["output"] for d in sft_out))
    print(f"\n独特输出数: {unique}/{len(sft_out)}")
    print(f"模板化率: {1 - unique/len(sft_out):.1%}")


if __name__ == "__main__":
    main()
