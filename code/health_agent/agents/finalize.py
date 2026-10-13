"""Finalize Agent: 汇总结果，生成最终回复"""
from state import AgentState

def finalize_node(state: AgentState) -> dict:
    parts = []
    # 顶部健康画像一行（分类/HI/最弱指标）
    head = []
    if state.get("level_label"):
        head.append(f"{state['level_label']}（HI={state.get('hi_score','?')}/100）")
    if state.get("risk_level"):
        head.append(f"{state.get('risk_level')}（风险概率={state.get('decline_prob','?')}）")
    if state.get("weak_indicator"):
        head.append(f"最弱指标：{state['weak_indicator']}")
    if head:
        parts.append("【健康画像】" + " · ".join(head))
    # 四步处方直接给出，不再套【建议】
    if state.get("draft_prescription"):
        parts.append(state["draft_prescription"])
    if not state.get("critic_pass", True):
        parts.append(f"【注意】本次建议经 {state.get('revision_count',0)} 轮修正仍存在问题：{';'.join(state.get('critic_errors',[]))}")

    final = "\n\n".join(parts)
    trace = state.get("trace", []) + [{"node": "finalize", "len": len(final)}]
    return {"final_response": final, "trace": trace}
