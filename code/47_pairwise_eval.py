#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
47_pairwise_eval.py — 成对比较评测（Pairwise LLM-as-Judge）

不打 1-5 绝对分（位置/长度偏差大），改成 MT-Bench 式成对比较：
同一道题，微调后模型(A) vs 基线模型(B)各一份回答，judge 逐维度判 A 胜 / B 胜 / 平，
最后汇总胜率。

两种数据源：
  1) dpo_pairs  —— 直接用 DPO 验证集的 chosen/rejected 当 A/B（无需 GPU/API 生成，
                   用来检验 DPO 偏好对本身是否"chosen 确实更好"）
  2) model_compare —— 微调后模型 vs 基线模型在同一批测试题上各自生成回答后比较
                   （需先把两模型输出落盘为 {question, answer_sft, answer_base}，
                    或配置 OpenAI 兼容接口调用两个 model）

用法：
  python code/47_pairwise_eval.py dpo_pairs            # 离线检查 DPO 数据
  python code/47_pairwise_eval.py model_compare --file data/pairwise_input.jsonl
  export OPENAI_API_KEY=sk-xxx
  python code/47_pairwise_eval.py model_compare --judge-model qwen-max
"""
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

# ---------------------------------------------------------------------------
# 成对比较 Prompt：逐维度判 A/B/tie，要求 JSON
# ---------------------------------------------------------------------------
PAIRWISE_PROMPT = """你是大学生体测健康领域的评审专家。下面是同一个学生问题，
模型 A（微调后）和模型 B（未微调基线）各给了一份干预方案。
请针对每个维度，比较 A 和 B 哪份更好，或持平。**不要被篇幅、措辞风格影响，只看事实质量。**

【学生问题与数据】
{question}

【模型 A：微调后】
{answer_a}

【模型 B：基线 Qwen2.5-7B】
{answer_b}

请逐维度判断（严格输出 JSON）：
{{
  "weak_indicator_accuracy": "A" / "B" / "tie",
  "prescription_relevance":  "A" / "B" / "tie",
  "safety":                  "A" / "B" / "tie",
  "specificity":             "A" / "B" / "tie",
  "personalization":         "A" / "B" / "tie",
  "reason": "一句话说明 A 为何胜出/落败"
}}

维度标准：
- weak_indicator_accuracy：谁更准确地识别了学生真正薄弱的指标；
- prescription_relevance：谁推荐的训练更对症该弱指标；
- safety：谁没有禁忌/危险建议（膝盖不推跑跳、BMI 高不推大重量等）；
- specificity：谁给出了具体可执行的频率/时长/强度；
- personalization：谁更针对该生数据，而不是通用套话。
"""

DIMENSIONS = ["weak_indicator_accuracy", "prescription_relevance",
              "safety", "specificity", "personalization"]
DIM_CN = {
    "weak_indicator_accuracy": "弱指标准确",
    "prescription_relevance": "处方对症",
    "safety": "安全合规",
    "specificity": "具体可执行",
    "personalization": "个性化",
}


def load_jsonl(path):
    if not Path(path).exists():
        print(f"[warn] 文件不存在: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def judge_one(client, judge_model, question, answer_a, answer_b):
    prompt = PAIRWISE_PROMPT.format(
        question=question[:600], answer_a=answer_a[:800], answer_b=answer_b[:800])
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, response_format={"type": "json_object"})
    return json.loads(resp.choices[0].message.content)


def summarize(results):
    """results: list of dict，逐维度统计 A/B/tie。"""
    stat = {d: {"A": 0, "B": 0, "tie": 0} for d in DIMENSIONS}
    for r in results:
        for d in DIMENSIONS:
            v = r.get(d, "tie")
            stat[d][v if v in ("A", "B") else "tie"] += 1

    print("\n" + "=" * 64)
    print("成对比较结果（A=微调后，B=基线）")
    print("=" * 64)
    print(f"{'维度':<14}{'A胜':>6}{'平':>6}{'B胜':>6}{'胜率':>10}")
    print("-" * 64)
    total_a = total_b = 0
    for d in DIMENSIONS:
        a, b, t = stat[d]["A"], stat[d]["B"], stat[d]["tie"]
        total = a + b
        wr = a / total if total else 0
        total_a += a; total_b += b
        print(f"{DIM_CN[d]:<12}{a:>6}{t:>6}{b:>6}{wr:>9.1%}")
    print("-" * 64)
    overall = total_a / (total_a + total_b) if (total_a + total_b) else 0
    print(f"{'综合胜率':<12}{total_a:>6}{'':>6}{total_b:>6}{overall:>9.1%}")
    print(f"\n解读：胜率 > 50% 表示微调后方案整体优于基线；"
          f"安全/对症维度若接近 100% 说明对齐有效。")
    return {"per_dim": stat, "overall_win_rate": round(overall, 3),
            "n": len(results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dpo_pairs", "model_compare"])
    ap.add_argument("--file", default=str(ROOT / "data" / "pairwise_input.jsonl"))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "qwen-max"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    # 组装 (question, A, B) 三元组
    pairs = []
    if args.mode == "dpo_pairs":
        # A=chosen（偏好好答案），B=rejected（偏好坏答案）
        dpo_val = load_jsonl(ROOT / "data" / "dpo_merged_val.jsonl")
        for d in dpo_val[:args.limit]:
            q = d.get("instruction") or d.get("question") or d.get("prompt", "")
            pairs.append((q, d.get("chosen", ""), d.get("rejected", "")))
    else:
        for d in load_jsonl(args.file)[:args.limit]:
            pairs.append((d.get("question", ""),
                           d.get("answer_sft") or d.get("answer_a", ""),
                           d.get("answer_base") or d.get("answer_b", "")))

    print(f"共 {len(pairs)} 组成对样本，judge={args.judge_model}")

    if not args.api_key:
        print("\n[提示] 未配置 OPENAI_API_KEY，仅展示评测流程，不实际调用 judge。")
        print("       配置后将逐维度输出 A/B/tie 并汇总胜率。")
        # 离线演示：打印一条样例
        if pairs:
            q, a, b = pairs[0]
            print("\n--- 样例输入（第 1 对）---")
            print("Q:", q[:120])
            print("A(微调后)前120字:", a[:120])
            print("B(基线)前120字:", b[:120])
        return

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    results = []
    for i, (q, a, b) in enumerate(pairs):
        try:
            r = judge_one(client, args.judge_model, q, a, b)
            results.append(r)
        except Exception as e:
            print(f"[err] 第{i}对: {e}")
        if (i + 1) % 10 == 0:
            print(f"已评 {i+1}/{len(pairs)}")

    report = summarize(results)
    out = ROOT / "code" / "pairwise_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
