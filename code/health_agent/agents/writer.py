"""Writer Agent: 加载 DPO 7B adapter，生成运动处方"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from state import AgentState
from config import BASE_LLM_PATH, DPO_ADAPTER_PATH, MAX_NEW_TOKENS, MAX_REVISION_ROUNDS

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    _tokenizer = AutoTokenizer.from_pretrained(str(BASE_LLM_PATH))
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_LLM_PATH), torch_dtype=torch.float16, device_map="mps"
    )
    _model = PeftModel.from_pretrained(base, str(DPO_ADAPTER_PATH))
    _model.eval()
    return _model, _tokenizer

def writer_node(state: AgentState) -> dict:
    weak = state.get("weak_indicator", "肺活量")
    hi = state.get("hi_score", 60)
    label = state.get("level_label", "中水平-平稳型")
    strategy = state.get("strategy", "运动+膳食综合")
    docs = state.get("retrieved_docs", [])
    revision_count = state.get("revision_count", 0)
    errors = state.get("critic_errors", [])

    model, tok = _load_model()

    # 构造 prompt
    d = state.get("student_data", {})
    user_query = state.get("user_query", "")
    sys_msg = "你是一名专业的大学生体测健康顾问，根据学生数据和用户的问题给出精准、安全、可执行的运动建议。"
    user_msg = (
        f"学生体测原始数据：BMI={d.get('bmi','?')}，"
        f"肺活量={d.get('vc','?')}ml，"
        f"50米={d.get('speed','?')}秒，"
        f"跳远={d.get('jump','?')}cm，"
        f"坐位体前屈={d.get('flex','?')}cm，"
        f"耐力跑={d.get('endurance','?')}秒，"
        f"力量={d.get('str','?')}。"
        f"\n该学生归一化健康指数HI={hi}/100，综合分类={label}。"
        f"用户问题：{user_query}"
        f"\n推荐干预策略（由RL策略选择）：{strategy}。"
        f"\n请针对{weak}给出{strategy}的运动建议。"
    )
    if docs:
        user_msg += f"\n参考资料：{docs[0][:200]}"
    if revision_count > 0 and errors:
        user_msg += f"\n注意：上次建议被审核驳回，原因：{'；'.join(errors)}。请修正后重新给出建议。"
    user_msg += "\n请给出个性化运动建议，不要重复罗列学生数据。"

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=0.1, top_p=0.9,
        )
    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    trace_entry = {"node": "writer", "revision": revision_count, "len": len(response)}
    trace = state.get("trace", []) + [trace_entry]
    return {
        "draft_prescription": response,
        "revision_count": revision_count + 1,
        "trace": trace,
    }
