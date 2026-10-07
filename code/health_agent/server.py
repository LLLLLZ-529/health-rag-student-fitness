#!/usr/bin/env python3
"""HealthAgent 后端：Flask + SSE，流式推送每步状态"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from graph import get_app

app = Flask(__name__, static_folder='.')
CORS(app)

@app.after_request
def no_cache(resp):
    """禁用浏览器缓存，保证每次刷新拿到最新 demo.html"""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# 预加载模型（首次请求时加载，避免首次等待太久）
_app = None
def get_graph():
    global _app
    if _app is None:
        print("加载模型中...")
        _app = get_app()
        print("模型加载完成")
    return _app

@app.route('/')
def index():
    return send_from_directory('.', 'demo.html')

@app.route('/run', methods=['POST'])
def run():
    data = request.json
    student = {
        "bmi": data.get("bmi", 22),
        "vc": data.get("vc", 2800),
        "str": data.get("str", 15),
        "endurance": data.get("endurance", 270),
        "speed": data.get("speed", 9.0),
        "jump": data.get("jump", 170),
        "flex": data.get("flex", 10),
    }
    query = data.get("query", "这个学生该怎么练")

    def generate():
        graph = get_graph()
        state = {
            "user_query": query,
            "student_data": student,
            "user_history": [],
            "revision_count": 0,
            "critic_errors": [],
            "trace": [],
        }

        # SSE 格式
        def sse(event, data):
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            final_output = {}
            # 用 stream 模式逐步推送
            for chunk in graph.stream(state, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    final_output = node_output  # 最后一轮是 finalize 的输出
                    # 推送节点开始
                    yield sse("node_start", {"node": node_name})

                    # 根据节点输出推送具体数据
                    if node_name == "router":
                        yield sse("log", {"msg": f"Router → task_type={node_output.get('task_type')}"})
                    elif node_name == "diagnosis":
                        yield sse("log", {"msg": f"Diagnosis → HI={node_output.get('hi_score')}, 最弱={node_output.get('weak_indicator')}, 分类={node_output.get('level_label')}"})
                    elif node_name == "retriever":
                        n = len(node_output.get("retrieved_docs", []))
                        yield sse("log", {"msg": f"Retriever → 检索{n}篇相关处方"})
                    elif node_name == "writer":
                        rev = node_output.get("revision_count", 1)
                        presc = node_output.get("draft_prescription", "")
                        yield sse("log", {"msg": f"Writer [轮{rev}] → {presc[:50]}..."})
                    elif node_name == "critic":
                        passed = node_output.get("critic_pass")
                        errs = node_output.get("critic_errors", [])
                        if passed:
                            yield sse("log", {"msg": "Critic → ✅ 通过（指标匹配）", "color": "green"})
                        else:
                            yield sse("log", {"msg": f"Critic → ❌ {';'.join(errs)}", "color": "red"})
                            yield sse("reject", {})
                    elif node_name == "finalize":
                        yield sse("log", {"msg": "Finalize → 输出完成", "color": "blue"})

                    # 节点完成
                    yield sse("node_done", {"node": node_name, "output": node_output})

            # 最终结果
            final_text = final_output.get("final_response", "")
            yield sse("final", {"text": final_text})

        except Exception as e:
            yield sse("error", {"msg": str(e)})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })

if __name__ == "__main__":
    # 预热模型
    get_graph()
    app.run(host="127.0.0.1", port=5099, debug=False, threaded=True)
