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
    sys_msg = (
        "你是一名专业的大学生体测健康顾问，严格基于学生数据、知识库和图谱推理给出"
        "安全、可执行、个性化的运动建议。必须按【诊断】【分析】【方案】【禁忌】四步"
        "组织，不可省略：\n"
        "【诊断】指出最弱指标，结合趋势说明依据。\n"
        "【分析】解释成因和优先方向，说明为什么排除其他训练类型。\n"
        "【方案】给出训练类型、频率、单次时长、强度、持续周数，每条建议后用"
        "[来源: 文件名] 标注依据；知识库无依据的标注[无知识库依据，需人工确认]。\n"
        "【禁忌】根据 BMI/伤病提示禁忌动作；无明确禁忌时说明出现何种症状应停止。\n"
        "禁止编造知识库外的药物、剂量、疾病诊断。"
    )
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
        f"\n干预策略（RL策略）：{strategy}。"
    )
    # 注入 GraphRAG 推理链作为结构化先验
    chain = state.get("reasoning_chain")
    if chain:
        user_msg += "\n\n【GraphRAG 图谱推理结论（请据此组织四步回复）】\n" + "\n".join(chain)
    if docs:
        user_msg += "\n\n【知识库检索资料】\n" + "\n".join(docs[:3])
    if revision_count > 0 and errors:
        user_msg += f"\n\n注意：上次建议被审核驳回，原因：{'；'.join(errors)}。请修正后重新给出。"
    user_msg += "\n\n请严格按【诊断】【分析】【方案】【禁忌】四步输出，并标注来源。"

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
