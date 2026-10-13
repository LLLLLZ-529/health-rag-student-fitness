"""Writer Agent: 加载 DPO 7B adapter 生成运动处方；无 GPU 时自动降级为图谱模板。

两种模式：
1. 有模型（torch + transformers + peft 可用且权重存在）：
   加载 Qwen2.5-7B-Instruct + DPO LoRA，结合 GraphRAG 推理链与检索资料生成。
2. 无模型（演示/无 GPU）：_template_writer() 用 GraphRAG 推理链 + 检索结果
   确定性组装四步处方（诊断/分析/方案/禁忌），完整保留可解释性与引用。
"""
from state import AgentState
from config import BASE_LLM_PATH, DPO_ADAPTER_PATH, MAX_NEW_TOKENS, MAX_REVISION_ROUNDS

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None

_model = None
_tokenizer = None
_model_failed = False


def _load_model():
    """惰性加载模型；任何失败（无 torch / 无权重 / 无 GPU）都降级为模板模式。"""
    global _model, _tokenizer, _model_failed
    if _model is not None:
        return _model, _tokenizer
    if _model_failed or not _TORCH_AVAILABLE:
        return None, None
    try:
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if not BASE_LLM_PATH.exists():
            raise FileNotFoundError(f"基座模型不存在: {BASE_LLM_PATH}")
        _tokenizer = AutoTokenizer.from_pretrained(str(BASE_LLM_PATH))
        base = AutoModelForCausalLM.from_pretrained(
            str(BASE_LLM_PATH), torch_dtype=torch.float16, device_map="mps"
        )
        if DPO_ADAPTER_PATH.exists():
            _model = PeftModel.from_pretrained(base, str(DPO_ADAPTER_PATH))
        else:
            _model = base
        _model.eval()
        return _model, _tokenizer
    except Exception as e:
        print(f"[writer] 模型不可用，降级为图谱模板生成：{e}")
        _model_failed = True
        return None, None


def _generic_draft():
    """模拟通用大模型未对齐时的千人一面草稿，用于演示 Critic 驳回。"""
    return (
        "【诊断】您的身体状况总体正常，不用太担心。\n"
        "【分析】大学生平时学习忙、运动少，体质下降是常见现象。\n"
        "【方案】建议每周运动3-4次、每次30分钟左右，可以跑跑步、打打球，"
        "同时注意饮食均衡、多喝水、少吃外卖、保证充足睡眠，循序渐进。\n"
        "【禁忌】运动前做好热身，避免突然剧烈运动导致受伤。"
    )


def _template_writer(state: AgentState) -> str:
    """无模型模式：用 GraphRAG 推理链 + 检索结果确定性组装四步处方。"""
    d = state.get("student_data", {})
    weak = state.get("weak_indicator", "肺活量")
    hi = state.get("hi_score", 60)
    label = state.get("level_label", "中水平-平稳型")
    bmi = d.get("bmi", "?")
    chain = state.get("reasoning_chain") or []
    docs = state.get("retrieved_docs") or []

    def pick(prefix):
        for line in chain:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
        return ""

    match = pick("【图谱匹配】")
    contra = pick("【禁忌过滤】")
    plan_line = pick("【方案】")
    assist = next((l for l in chain if l.startswith("辅以")), "")

    # 诊断：补 HI 与分类
    diag = (f"【诊断】该学生最弱指标为{weak}，归一化健康指数 HI={hi}/100，"
            f"综合分类为{label}，BMI={bmi}。")
    # 分析：图谱效果匹配
    analysis = "【分析】" + (match or f"结合体质健康知识图谱，针对{weak}优先匹配相关训练。")
    # 方案：图推理动作 + 真实检索来源
    plan = "【方案】" + plan_line
    if assist:
        plan += assist
    sources = []
    for doc in docs[:2]:
        first = doc.split("\n")[0]
        if first.startswith("[来源"):
            name = first.replace("[来源:", "").replace("]", "").strip()
            sources.append(name)
    if sources:
        plan += " 依据：" + "；".join(sources) + "。"
    plan += "建议循序渐进、每周3-4次、每2周复测一次。"
    # 禁忌：图谱禁忌步骤；无禁忌给安全提示
    if contra.startswith("无"):
        contra_text = "无明确 BMI/伤病禁忌；运动中如出现胸痛、关节锐痛、头晕应立即停止。"
    else:
        contra_text = contra + "；运动中如出现胸痛、关节锐痛、头晕应立即停止。"
    return "\n".join([diag, analysis, plan, "【禁忌】" + contra_text])


def writer_node(state: AgentState) -> dict:
    weak = state.get("weak_indicator", "肺活量")
    revision_count = state.get("revision_count", 0)
    errors = state.get("critic_errors", [])

    model, tok = _load_model()

    # 演示模式：首轮刻意给通用泛化草稿，让 Critic 驳回，第二轮再给精准方案
    force_reject = state.get("demo_force_reject", False)
    if model is None:
        if force_reject and revision_count == 0:
            response = _generic_draft()
        else:
            response = _template_writer(state)
        trace = state.get("trace", []) + [
            {"node": "writer", "revision": revision_count, "len": len(response),
             "mode": "graph-template"}]
        return {
            "draft_prescription": response,
            "revision_count": revision_count + 1,
            "trace": trace,
        }

    # ===== 有模型：原有 LLM 生成路径 =====
    hi = state.get("hi_score", 60)
    label = state.get("level_label", "中水平-平稳型")
    strategy = state.get("strategy", "运动+膳食综合")
    docs = state.get("retrieved_docs", [])
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

    trace_entry = {"node": "writer", "revision": revision_count, "len": len(response),
                   "mode": "llm"}
    trace = state.get("trace", []) + [trace_entry]
    return {
        "draft_prescription": response,
        "revision_count": revision_count + 1,
        "trace": trace,
    }
