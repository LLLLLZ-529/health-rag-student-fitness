"""HealthAgent LangGraph 定义"""
from langgraph.graph import StateGraph, START, END
from state import AgentState
from agents.router import router_node
from agents.diagnosis import diagnosis_node
from agents.prediction import prediction_node
from agents.retriever import retriever_node
from agents.strategy import strategy_node
from agents.writer import writer_node
from agents.critic import critic_node, should_revise
from agents.finalize import finalize_node

def route_after_router(state: AgentState) -> str:
    """Router 后条件路由"""
    t = state.get("task_type", "recommend")
    if t == "predict":
        return "prediction"
    elif t == "diagnose":
        return "diagnosis"
    else:
        return "recommend_flow"

def build_graph():
    g = StateGraph(AgentState)

    # 节点
    g.add_node("router", router_node)
    g.add_node("diagnosis", diagnosis_node)
    g.add_node("prediction", prediction_node)
    g.add_node("retriever", retriever_node)
    g.add_node("strategy", strategy_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    # 边
    g.add_edge(START, "router")

    # Router 分流
    g.add_conditional_edges("router", route_after_router, {
        "diagnosis": "diagnosis",
        "prediction": "prediction",
        "recommend_flow": "diagnosis",  # 推荐也要先诊断
    })

    # diagnosis → retriever（推荐流程）
    g.add_edge("diagnosis", "retriever")
    # retriever → strategy → writer
    g.add_edge("retriever", "strategy")
    g.add_edge("strategy", "writer")
    # writer → critic
    g.add_edge("writer", "critic")
    # critic 条件：通过→finalize，不通过→回 writer
    g.add_conditional_edges("critic", should_revise, {
        "revise": "writer",
        "accept": "finalize",
    })

    # prediction → finalize（预测不需要写处方）
    g.add_edge("prediction", "finalize")

    g.add_edge("finalize", END)
    return g.compile()

# 单例
_app = None
def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
