"""Finalize Agent: 汇总结果，生成最终回复"""
from state import AgentState

def finalize_node(state: AgentState) -> dict:
    parts = []
    if state.get("level_label"):
        parts.append(f"【分类】{state['level_label']}（HI={state.get('hi_score','?')}）")
    if state.get("risk_level"):
        parts.append(f"【风险】{state['risk_level']}（概率={state.get('decline_prob','?')}）")
    if state.get("weak_indicator"):
        parts.append(f"【最弱指标】{state['weak_indicator']}")
    if state.get("draft_prescription"):
        parts.append(f"【建议】{state['draft_prescription']}")
    if not state.get("critic_pass", True):
        parts.append(f"【注意】本次建议经 {state.get('revision_count',0)} 轮修正仍存在问题：{';'.join(state.get('critic_errors',[]))}")

    final = "\n".join(parts)
    trace = state.get("trace", []) + [{"node": "finalize", "len": len(final)}]
    return {"final_response": final, "trace": trace}
