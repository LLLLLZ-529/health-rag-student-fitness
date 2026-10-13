#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
42_fix_weak_indicator.py — 修复弱指标判定 bug

旧逻辑（错误）：在原始值上直接比较，肺活量数值上千，取负后永远最小 → 71% 被错标肺活量。
新逻辑（正确）：按性别分组，对每个指标做百分位标准化，统一量纲后再找最弱指标。
  - 越大越好（肺活量/跳远/体前屈/力量）：百分位越低越弱
  - 越小越好（50米/耐力跑秒）：百分位越高越弱（取反）
  - BMI：偏离正常区间[18.5,23.9]越远越弱（两头都差）

输出：
  data/weak_indicator_fixed.csv —— 每个学生修正后的最弱指标
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HI_WIDE = ROOT / "data" / "hi_wide.csv"
OUT = ROOT / "data" / "weak_indicator_fixed.csv"

# 各指标的方向：True=越大越好，False=越小越好
HIGHER_BETTER = {
    "vital_capacity": True,
    "sprint_50m": False,       # 秒，越小越好
    "standing_long_jump": True,
    "sit_and_reach": True,
    "endurance_run_sec": False,  # 秒，越小越好
    "strength": True,
}
INDICATOR_CN = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
    "endurance_run_sec": "耐力跑", "strength": "力量",
}


def bmi_health_score(bmi):
    """BMI 偏离正常区间越远，分数越低（0-100）。中国标准：正常 18.5-23.9。"""
    lo, hi = 18.5, 23.9
    if lo <= bmi <= hi:
        return 100.0
    deviation = (lo - bmi) if bmi < lo else (bmi - hi)
    return max(0.0, 100.0 - deviation * 20)  # 每偏离1，扣20分


def compute_weak_indicators(df, grade="g4"):
    """按性别百分位法，返回每个学生最弱指标 + 各指标百分位。"""
    scores = pd.DataFrame(index=df.index)
    scores["student_id"] = df["student_id"]
    scores["gender"] = df["gender"]

    # 6 个常规指标：按性别分组做百分位
    for ind, higher_better in HIGHER_BETTER.items():
        col = f"{ind}_{grade}"
        vals = df[col].copy()
        # 同性别百分位
        pct = vals.groupby(df["gender"]).rank(pct=True) * 100
        if not higher_better:
            pct = 100 - pct  # 秒数越大越弱，取反
        scores[f"{ind}_pct"] = pct

    # BMI 单独算健康分（区间最优）
    bmi = df[f"bmi_{grade}"]
    scores["bmi_pct"] = bmi.apply(bmi_health_score)

    pct_cols = [f"{ind}_pct" for ind in list(HIGHER_BETTER) + ["bmi"]]
    # 最弱 = 百分位最低
    scores["weakest_indicator"] = scores[pct_cols].idxmin(axis=1).str.replace("_pct", "")
    scores["weakest_pct"] = scores[pct_cols].min(axis=1)
    return scores


def main():
    print("加载 hi_wide.csv ...")
    df = pd.read_csv(HI_WIDE)
    print(f"总学生数: {len(df)}")

    valid = df.dropna(subset=[f"vital_capacity_g4", f"endurance_run_sec_g4"]).copy()
    print(f"大四数据完整: {len(valid)}")

    scores = compute_weak_indicators(valid, "g4")

    print("\n" + "=" * 50)
    print("修正后的弱指标分布（全体大四学生）")
    print("=" * 50)
    dist = scores["weakest_indicator"].value_counts()
    for ind, cnt in dist.items():
        name = INDICATOR_CN.get(ind, ind)
        print(f"  {name:8s}: {cnt:6d} ({cnt/len(scores):.1%})")

    # 退化学生子集
    print("\n" + "=" * 50)
    print("退化趋势学生（HI g1 > g4）的弱指标分布")
    print("=" * 50)
    merged = scores.merge(valid[["student_id", "HI_g1", "HI_g4"]], on="student_id")
    declined = merged[merged["HI_g4"] < merged["HI_g1"]]
    print(f"退化学生数: {len(declined)}")
    dist_d = declined["weakest_indicator"].value_counts()
    for ind, cnt in dist_d.items():
        name = INDICATOR_CN.get(ind, ind)
        print(f"  {name:8s}: {cnt:6d} ({cnt/len(declined):.1%})")

    scores.to_csv(OUT, index=False)
    print(f"\n已保存: {OUT}")

    # 与旧逻辑对比
    print("\n" + "=" * 50)
    print("旧逻辑 vs 新逻辑对比")
    print("=" * 50)
    # 旧逻辑
    old_weak = []
    for _, row in valid.iterrows():
        weakest, ws = None, float("inf")
        for ind in INDICATOR_CN:
            col = f"{ind}_g4"
            if col in row and not pd.isna(row[col]):
                val = row[col]
                s = val if ind in ["sprint_50m", "endurance_run_sec"] else -val
                if s < ws:
                    ws, weakest = s, ind
        old_weak.append(weakest)
    old_dist = pd.Series(old_weak).value_counts()
    print("旧逻辑（原始值比较）:")
    for ind, cnt in old_dist.items():
        print(f"  {INDICATOR_CN.get(ind, ind):8s}: {cnt/len(valid):.1%}")
    print("新逻辑（同性别百分位）:")
    for ind, cnt in dist.items():
        print(f"  {INDICATOR_CN.get(ind, ind):8s}: {cnt/len(valid):.1%}")


if __name__ == "__main__":
    main()
