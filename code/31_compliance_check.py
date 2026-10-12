#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31_compliance_check.py — 干预方案合规率自动检查 + 数据质量诊断

对每条「体测数据 prompt + 方案 response」做 4 项检查：
  C1 弱指标识别正确：response 点名的薄弱指标 = 学生真实最弱指标
  C2 无禁忌词：不出现"越多越好/屏气/蛋白粉/静养/随便练"等
  C3 处方类型匹配：弱指标 → 正确的训练类型（肺活量弱→有氧，不能是力量）
  C4 方案可执行：包含频率（每周X次）或时长（每次X分钟）或具体动作
合规 = 4 项全过。

两种"真实最弱指标"口径：
  field  ：数据自带 weakest_indicator 字段（数据构造时打的标签）
  science：按分性别固定阈值（国标近似）重新判定 ← 用于检验数据标签是否正确

用法：
  python 31_compliance_check.py --test-on-dpo
  python 31_compliance_check.py --jsonl <文件> [--resp-key chosen]
"""
import json, re, argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent

INDICATOR_CN = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
    "endurance_run_sec": "耐力", "strength": "力量",
}
INDICATOR_MENTIONS = {
    "bmi": ["BMI", "体重", "肥胖", "超重"],
    "vital_capacity": ["肺活量", "心肺"],
    "sprint_50m": ["50米", "速度", "爆发力"],
    "standing_long_jump": ["立定跳远", "跳远", "下肢"],
    "sit_and_reach": ["柔韧", "坐位体前屈", "拉伸"],
    "endurance_run_sec": ["耐力", "长跑", "有氧"],
    "strength": ["力量", "肌肉", "握力", "引体", "仰卧起坐"],
}
FORBIDDEN = [
    "越多越好", "屏气", "蛋白粉", "随便练", "体测不重要", "静养", "办卡",
    "每天10公里", "力竭", "500大卡", "快速减重", "跟着教练", "只吃",
    "节食", "弹震", "强行", "自然就",
]
CORRECT_TRAINING = {
    "bmi": ["有氧", "慢跑", "游泳", "快走", "饮食控制", "骑行"],
    "vital_capacity": ["慢跑", "游泳", "心肺", "有氧", "深呼吸", "呼吸"],
    "sprint_50m": ["冲刺", "爆发力", "高抬腿", "蛙跳"],
    "standing_long_jump": ["深蹲", "下肢力量", "跳箱", "爆发力", "弓步"],
    "sit_and_reach": ["拉伸", "瑜伽", "柔韧", "静态"],
    "endurance_run_sec": ["耐力", "长跑", "间歇", "有氧", "慢跑"],
    "strength": ["力量", "抗阻", "深蹲", "俯卧撑", "哑铃", "引体"],
}

# 分性别阈值（国标近似；"偏弱"线，比及格线略高）
THRESHOLDS = {
    "女": {"bmi_high": 24, "vc_low": 2800, "sprint_high": 9.5, "jump_low": 170,
           "reach_low": 8, "endur_high": 270, "str_low": 20},
    "男": {"bmi_high": 25, "vc_low": 3200, "sprint_high": 8.5, "jump_low": 210,
           "reach_low": 8, "endur_high": 255, "str_low": 6},
}

def parse_prompt(prompt):
    info = {}
    gm = re.search(r"一名(\S)大学生", prompt)
    info["gender"] = gm.group(1) if gm else "女"
    m = re.search(r"BMI=([\d.]+)", prompt); info["bmi"] = float(m.group(1)) if m else 22
    m = re.search(r"肺活量=(\d+)ml", prompt); info["vital_capacity"] = int(m.group(1)) if m else 3000
    m = re.search(r"50米跑=([\d.]+)秒", prompt); info["sprint_50m"] = float(m.group(1)) if m else 9
    m = re.search(r"立定跳远=(\d+)cm", prompt); info["standing_long_jump"] = int(m.group(1)) if m else 180
    m = re.search(r"坐位体前屈=([\d.]+)cm", prompt); info["sit_and_reach"] = float(m.group(1)) if m else 10
    m = re.search(r"耐力跑=(\d+)分(\d+)秒", prompt)
    info["endurance_run_sec"] = (int(m.group(1))*60 + int(m.group(2))) if m else 300
    m = re.search(r"力量=([\d.]+)", prompt); info["strength"] = float(m.group(1)) if m else 15
    return info

def weakest_by_science(info):
    """按分性别阈值，找超出阈值最严重的指标"""
    t = THRESHOLDS.get(info["gender"], THRESHOLDS["女"])
    gaps = []
    if info["bmi"] > t["bmi_high"]: gaps.append(("bmi", info["bmi"] - t["bmi_high"]))
    if info["vital_capacity"] < t["vc_low"]: gaps.append(("vital_capacity", t["vc_low"] - info["vital_capacity"]))
    if info["sprint_50m"] > t["sprint_high"]: gaps.append(("sprint_50m", info["sprint_50m"] - t["sprint_high"]))
    if info["standing_long_jump"] < t["jump_low"]: gaps.append(("standing_long_jump", t["jump_low"] - info["standing_long_jump"]))
    if info["sit_and_reach"] < t["reach_low"]: gaps.append(("sit_and_reach", t["reach_low"] - info["sit_and_reach"]))
    if info["endurance_run_sec"] > t["endur_high"]: gaps.append(("endurance_run_sec", info["endurance_run_sec"] - t["endur_high"]))
    if info["strength"] < t["str_low"]: gaps.append(("strength", t["str_low"] - info["strength"]))
    if not gaps:
        return None
    gaps.sort(key=lambda x: -x[1])
    return gaps[0][0]

# 明确诊断陈述："薄弱环节是X / 最弱是X / 主要问题是X"
DIAG_PATTERNS = [
    r"薄弱环节是([^。，.；;]+)",
    r"最弱[^。，.；;]{0,5}?是([^。，.；;]+)",
    r"主要问题是([^。，.；;]+)",
    r"短板是([^。，.；;]+)",
]

def map_text_to_indicators(text):
    """把一段诊断文本映射到它提到的指标"""
    return [ind for ind, words in INDICATOR_MENTIONS.items()
            if any(w in text for w in words)]

def extract_diagnosis(response):
    """提取模型明确诊断的薄弱指标"""
    for pat in DIAG_PATTERNS:
        m = re.search(pat, response)
        if m:
            inds = map_text_to_indicators(m.group(1))
            if inds:
                return inds
    return []

def check_response(prompt, response, weak=None):
    info = parse_prompt(prompt)
    if weak is None:
        weak = weakest_by_science(info)
    reasons, checks = [], {}

    # C1 弱指标识别正确：优先看明确诊断陈述对不对
    if weak is None:
        checks["C1_识别"] = True
    else:
        diag = extract_diagnosis(response)
        if diag:
            c1 = weak in diag
        else:  # 没有明确诊断陈述，退而看是否点名该指标
            c1 = any(m in response for m in INDICATOR_MENTIONS[weak])
        checks["C1_识别"] = c1
        if not c1:
            reasons.append(f"诊断错误：应识别为「{INDICATOR_CN[weak]}」，实际诊断{[INDICATOR_CN[i] for i in diag] if diag else '未提及'}")

    hit = [w for w in FORBIDDEN if w in response]
    checks["C2_禁忌"] = len(hit) == 0
    if hit:
        reasons.append(f"含禁忌词 {hit}")

    if weak is not None:
        c3 = any(k in response for k in CORRECT_TRAINING[weak])
        checks["C3_处方"] = c3
        if not c3:
            reasons.append(f"处方与「{INDICATOR_CN[weak]}」弱不匹配")
    else:
        checks["C3_处方"] = True

    has_freq = bool(re.search(r"每周\s*\d|一周\s*\d|\d+\s*次", response))
    has_dur = bool(re.search(r"\d+\s*分钟|\d+\s*小时", response))
    has_action = bool(re.search(r"慢跑|游泳|深蹲|拉伸|俯卧撑|瑜伽|骑行|快走|冲刺|引体", response))
    checks["C4_可执行"] = has_freq or has_dur or has_action
    if not checks["C4_可执行"]:
        reasons.append("方案不具体（无频率/时长/动作）")

    return all(checks.values()), checks, reasons, weak

def evaluate(pairs, resp_key, weak_mode="science"):
    total = len(pairs)
    passed = {"C1_识别": 0, "C2_禁忌": 0, "C3_处方": 0, "C4_可执行": 0}
    n_ok = 0
    mismatches = []
    for p in pairs:
        info = parse_prompt(p["prompt"])
        sci_weak = weakest_by_science(info)
        field_weak = p.get("weakest_indicator")
        weak = field_weak if weak_mode == "field" else sci_weak
        ok, checks, reasons, used_weak = check_response(p["prompt"], p[resp_key], weak)
        for k, v in checks.items():
            passed[k] += int(v)
        n_ok += int(ok)
        # 记录"数据标签 vs 科学判定"不一致
        if field_weak and sci_weak and field_weak != sci_weak and resp_key == "chosen":
            mismatches.append({"field": field_weak, "science": sci_weak,
                               "prompt": p["prompt"][:120]})
    return {
        "总条数": total, "合规率": round(n_ok/total, 4),
        "各项通过率": {k: round(v/total, 4) for k, v in passed.items()},
        "标签不一致数": len(mismatches), "标签不一致样例": mismatches[:5],
    }

def test_on_dpo():
    dpo = [json.loads(l) for l in open(BASE_DIR / "dpo_merged.jsonl")]
    print("=" * 64)
    print("【1】chosen 按『数据标签』口径（应接近100%，说明标签自洽）")
    print("=" * 64)
    r1 = evaluate(dpo, "chosen", "field")
    print(f"合规率: {r1['合规率']*100:.1f}%  各项: {r1['各项通过率']}")

    print()
    print("=" * 64)
    print("【2】chosen 按『科学阈值』口径（低则说明数据标签错误）")
    print("=" * 64)
    r2 = evaluate(dpo, "chosen", "science")
    print(f"合规率: {r2['合规率']*100:.1f}%  各项: {r2['各项通过率']}")
    print(f"数据标签与科学判定不一致: {r2['标签不一致数']} 条")
    for m in r2["标签不一致样例"][:5]:
        print(f"  数据标签={m['field']} → 科学判定={m['science']}")

    print()
    print("=" * 64)
    print("【3】rejected 按『科学阈值』口径（应接近0%，坏方案确实差）")
    print("=" * 64)
    r3 = evaluate(dpo, "rejected", "science")
    print(f"合规率: {r3['合规率']*100:.1f}%  各项: {r3['各项通过率']}")

    out = {"chosen_field": r1, "chosen_science": r2, "rejected_science": r3}
    (BASE_DIR / "compliance_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print("\n报告已保存: compliance_report.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-on-dpo", action="store_true")
    ap.add_argument("--jsonl", type=str)
    ap.add_argument("--resp-key", default="response")
    ap.add_argument("--weak-mode", default="science", choices=["science", "field"])
    args = ap.parse_args()
    if args.test_on_dpo:
        test_on_dpo()
    elif args.jsonl:
        pairs = [json.loads(l) for l in open(args.jsonl)]
        print(json.dumps(evaluate(pairs, args.resp_key, args.weak_mode),
                         ensure_ascii=False, indent=2))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
