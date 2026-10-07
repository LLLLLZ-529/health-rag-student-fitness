#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_evaluate_models_1.5b.py — 三方模型生成质量对比评估（1.5B版）

对比：Base模型 vs SFT模型 vs SFT+DPO模型
任务：分类、预测、推荐
输出：JSON报告 + 可读文本对比
"""
import os, sys, json, glob
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BASE_DIR = Path(__file__).parent
_base_candidates = glob.glob(str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))
BASE_MODEL_PATH = _base_candidates[0] if _base_candidates else "Qwen/Qwen2.5-1.5B-Instruct"
SFT_MODEL = BASE_DIR / "outputs_1.5b" / "best_model"
DPO_MODEL = BASE_DIR / "outputs_1.5b" / "dpo_model_multiscene"
OUTPUT_DIR = BASE_DIR / "outputs_1.5b" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = "你是一个大学生体测健康助手，擅长分析体测数据并给出运动建议。"


def load_model(model_path, is_lora=False, base_path=None):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path) if not is_lora else str(base_path),
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_lora:
        base = AutoModelForCausalLM.from_pretrained(
            base_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype=torch.float16
        )
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,  # 用greedy解码，小模型更稳定
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# 测试样本（与训练数据格式保持一致：9类用中文名称，不要求编号）
LABEL_LIST = "低水平-退化型、低水平-平稳型、低水平-改善型、中水平-退化型、中水平-平稳型、中水平-改善型、高水平-退化型、高水平-平稳型、高水平-改善型"
CLASSIFICATION_CASES = [
    {
        "name": "案例1-高分上升型",
        "prompt": f"""请根据以下体测数据判断学生的健康状况属于以下哪一类：{LABEL_LIST}
性别：男，年级：大二
BMI：22.1，肺活量：4500ml，50米跑：7.2秒，立定跳远：235cm，坐位体前屈：18cm，耐力跑：1000米3分50秒，力量：引体向上12个
历年总分：大一78，大二85
请输出类别名称和依据。""",
    },
    {
        "name": "案例2-及格下降型",
        "prompt": f"""请根据以下体测数据判断学生的健康状况属于以下哪一类：{LABEL_LIST}
性别：女，年级：大三
BMI：21.5，肺活量：2800ml，50米跑：9.1秒，立定跳远：165cm，坐位体前屈：12cm，耐力跑：800米4分30秒，力量：仰卧起坐28个
历年总分：大一72，大二65，大三58
请输出类别名称和依据。""",
    },
    {
        "name": "案例3-不及格平稳型",
        "prompt": f"""请根据以下体测数据判断学生的健康状况属于以下哪一类：{LABEL_LIST}
性别：男，年级：大四
BMI：26.8，肺活量：3200ml，50米跑：8.5秒，立定跳远：195cm，坐位体前屈：5cm，耐力跑：1000米4分50秒，力量：引体向上3个
历年总分：大一55，大二53，大三56，大四54
请输出类别名称和依据。""",
    },
]

PREDICTION_CASES = [
    {
        "name": "预测1-退化风险",
        "prompt": """根据大一大二数据，预测该学生大四是否会出现体能退化：
性别：男，大一BMI=23.5肺活量=4200，大二BMI=24.2肺活量=4000
大一50米=7.5秒，大二50米=7.8秒
大一耐力=1000米4分00秒，大二耐力=1000米4分15秒
请判断：大四是否会退化？给出理由。""",
    },
    {
        "name": "预测2-上升趋势",
        "prompt": """根据大一大二数据，预测该学生大四是否会出现体能退化：
性别：女，大一BMI=20.1肺活量=2600，大二BMI=20.5肺活量=2900
大一50米=8.8秒，大二50米=8.5秒
大一耐力=800米4分10秒，大二耐力=800米3分55秒
请判断：大四是否会退化？给出理由。""",
    },
]

RECOMMENDATION_CASES = [
    {
        "name": "推荐1-肺活量弱",
        "prompt": """该学生最弱的指标是肺活量（仅2500ml，低于及格线），其他指标中等。
请给出针对性的运动建议，要具体、可执行。""",
    },
    {
        "name": "推荐2-耐力弱",
        "prompt": """该学生最弱的指标是耐力（800米4分50秒，不及格），BMI正常，其他指标及格。
请给出针对性的运动建议，要具体、可执行。""",
    },
    {
        "name": "推荐3-力量弱",
        "prompt": """该学生最弱的指标是力量（引体向上只能做1个），BMI偏瘦，其他指标及格。
请给出针对性的运动建议，要具体、可执行。""",
    },
]


def evaluate_model(model_name, model, tokenizer):
    results = {"model": model_name, "classification": [], "prediction": [], "recommendation": []}
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")

    for case in CLASSIFICATION_CASES:
        print(f"\n--- {case['name']} ---")
        output = generate(model, tokenizer, case["prompt"])
        print(f"输出: {output[:200]}")
        results["classification"].append({"case": case["name"], "output": output})

    for case in PREDICTION_CASES:
        print(f"\n--- {case['name']} ---")
        output = generate(model, tokenizer, case["prompt"])
        print(f"输出: {output[:200]}")
        results["prediction"].append({"case": case["name"], "output": output})

    for case in RECOMMENDATION_CASES:
        print(f"\n--- {case['name']} ---")
        output = generate(model, tokenizer, case["prompt"])
        print(f"输出: {output[:200]}")
        results["recommendation"].append({"case": case["name"], "output": output})

    return results


def main():
    print("=" * 60)
    print("三方模型生成质量对比评估 (1.5B)")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"基座模型: {BASE_MODEL_PATH}")

    all_results = []

    # 1. Base模型
    print("\n加载 Base 模型...")
    base_model, base_tokenizer = load_model(BASE_MODEL_PATH, is_lora=False)
    all_results.append(evaluate_model("Base (Qwen2.5-1.5B)", base_model, base_tokenizer))
    del base_model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # 2. SFT模型
    print("\n加载 SFT 模型...")
    sft_model, sft_tokenizer = load_model(SFT_MODEL, is_lora=True, base_path=BASE_MODEL_PATH)
    all_results.append(evaluate_model("SFT (LoRA r=16)", sft_model, sft_tokenizer))
    del sft_model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # 3. SFT+DPO模型
    if DPO_MODEL.exists():
        print("\n加载 SFT+DPO 模型...")
        dpo_model, dpo_tokenizer = load_model(DPO_MODEL, is_lora=True, base_path=BASE_MODEL_PATH)
        all_results.append(evaluate_model("SFT+DPO (β=0.1)", dpo_model, dpo_tokenizer))
        del dpo_model
        if DEVICE == "mps":
            torch.mps.empty_cache()
    else:
        print(f"\n⚠️ DPO模型不存在: {DPO_MODEL}，跳过")

    # 保存结果
    output_file = OUTPUT_DIR / "model_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n结果已保存: {output_file}")

    # 生成可读对比报告
    report_file = OUTPUT_DIR / "comparison_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# 模型生成质量对比报告 (1.5B)\n\n")
        for task_name, task_key in [("分类任务", "classification"), ("预测任务", "prediction"), ("推荐任务", "recommendation")]:
            f.write(f"## {task_name}\n\n")
            cases = all_results[0][task_key]
            for i, case in enumerate(cases):
                f.write(f"### {case['case']}\n\n")
                for result in all_results:
                    f.write(f"**{result['model']}**:\n")
                    f.write(f"```\n{result[task_key][i]['output']}\n```\n\n")
                f.write("---\n\n")

    print(f"对比报告已保存: {report_file}")
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
