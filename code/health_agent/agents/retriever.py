"""Retriever Agent: TF-IDF 检索体测健康知识库"""
import os, json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from state import AgentState
from config import CHUNKS_DIR, TOP_K_RETRIEVAL

_index = None
_vectorizer = None

def _build_index():
    """加载全部 chunk 并建 TF-IDF 索引（只建一次）"""
    global _index, _vectorizer
    if _index is not None:
        return _index

    docs, metas = [], []
    for root, _, files in os.walk(CHUNKS_DIR):
        for fn in files:
            if fn.endswith(".md"):
                fp = Path(root) / fn
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    docs.append(text)
                    rel = fp.relative_to(CHUNKS_DIR)
                    metas.append(str(rel))
                except Exception:
                    continue

    _vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = _vectorizer.fit_transform(docs)
    _index = {"docs": docs, "metas": metas, "matrix": matrix}
    return _index

def retriever_node(state: AgentState) -> dict:
    weak = state.get("weak_indicator", "健康")
    query = f"{weak} 运动处方 训练方法 建议"
    trace_entry = {"node": "retriever", "query": query}

    idx = _build_index()
    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, idx["matrix"]).flatten()
    top_idx = np.argsort(sims)[-TOP_K_RETRIEVAL:][::-1]

    retrieved = []
    for i in top_idx:
        if sims[i] > 0.01:
            snippet = idx["docs"][i][:300]
            retrieved.append(snippet)

    trace_entry.update({"n_docs": len(retrieved), "top_score": float(sims[top_idx[0]]) if len(top_idx) else 0})
    trace = state.get("trace", []) + [trace_entry]
    return {"retrieved_docs": retrieved, "trace": trace}
