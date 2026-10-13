"""Retriever Agent: 分类路由 + jieba 中文分词的 TF-IDF 检索

改进点（相对旧版纯 TF-IDF）：
1. jieba 分词：旧版 sklearn 默认不切中文，整句被当成一个 token，召回几乎为 0；
2. 分类路由：按 query 关键词先判定意图（运动处方/膳食营养/体测标准/健康政策），
   在目标分类内检索，避免跨类高频词干扰；
3. 全局兜底：分类内结果不足时回退全局检索；
4. 返回带来源标注的文本，配合引用溯源（[来源: 分类/文件名]）。
"""
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba

from state import AgentState
from config import CHUNKS_DIR, TOP_K_RETRIEVAL

jieba.setLogLevel(20)

# 分类关键词（词, 权重）：强区分名词 2 分，泛词 1 分
_CATEGORY_KEYWORDS = {
    "运动处方": [("训练", 1), ("运动", 1), ("怎么练", 2), ("锻炼", 2), ("跑步", 2),
               ("游泳", 2), ("跳绳", 2), ("力量", 2), ("肌肉", 2), ("有氧", 2),
               ("拉伸", 2), ("瑜伽", 2), ("深蹲", 2), ("俯卧撑", 2), ("引体", 2),
               ("肺活量", 2), ("耐力", 2), ("跳远", 2), ("健身", 2), ("热身", 2),
               ("心率", 2), ("间歇", 2), ("骑行", 2), ("蛙跳", 2), ("冲刺", 2),
               ("体能", 2), ("减脂", 2), ("练", 1), ("负荷", 2)],
    "膳食营养": [("吃", 2), ("饮食", 2), ("营养", 2), ("热量", 2), ("蛋白", 2),
               ("维生素", 2), ("餐", 2), ("食谱", 2), ("膳食", 2), ("食物", 2),
               ("水果", 2), ("蔬菜", 2), ("碳水", 2), ("脂肪", 2), ("早餐", 2),
               ("午餐", 2), ("晚餐", 2), ("补钙", 2), ("补铁", 2), ("牛奶", 2),
               ("鸡蛋", 2), ("瘦肉", 2), ("粗粮", 2), ("低糖", 2)],
    "体测标准": [("标准", 2), ("及格", 2), ("满分", 2), ("评分", 2), ("几分", 2),
               ("合格", 2), ("分数线", 2), ("怎么测", 2), ("测什么", 2),
               ("分数", 2), ("评级", 2), ("优秀", 2), ("良好", 2), ("权重", 2),
               ("占比", 2), ("加分", 2), ("扣分", 2)],
    "健康政策": [("政策", 2), ("文件", 2), ("规定", 2), ("通知", 2), ("制度", 2),
               ("法规", 2), ("意见", 2), ("纲要", 2), ("教育部", 2),
               ("体育总局", 2), ("健康中国", 2)],
}

_index = None


def _tokenize(text):
    return [w for w in jieba.cut(text) if len(w.strip()) > 1]


def _make_vectorizer(max_features):
    return TfidfVectorizer(
        tokenizer=_tokenize, token_pattern=None,
        max_features=max_features, ngram_range=(1, 2), min_df=1,
    )


def _route(query, weak=None):
    """按加权关键词判定主分类。弱指标明确时强制路由到运动处方。"""
    if weak and weak not in ("BMI", "健康"):
        # 体测指标类问题默认进运动处方
        pass
    scores = {cat: sum(w for kw, w in kws if kw in query)
              for cat, kws in _CATEGORY_KEYWORDS.items()}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if ranked[0][1] > 0:
        return ranked[0][0], [c for c, s in ranked if s > 0]
    return None, []


def _build_index():
    """加载全部 chunk，分别建全局索引和分类索引（只建一次）。"""
    global _index
    if _index is not None:
        return _index

    global_docs, global_meta, cat_docs = [], [], {}
    for root, _, files in os.walk(CHUNKS_DIR):
        for fn in sorted(files):
            if not fn.endswith(".md"):
                continue
            fp = Path(root) / fn
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = str(fp.relative_to(CHUNKS_DIR))
            cat = rel.split("/")[0] if "/" in rel else "未分类"
            global_docs.append(text)
            global_meta.append(rel)
            cat_docs.setdefault(cat, []).append((text, rel))

    global_vec = _make_vectorizer(12000)
    global_mat = global_vec.fit_transform(global_docs)
    cat_vec, cat_mat = {}, {}
    for cat, items in cat_docs.items():
        vec = _make_vectorizer(8000)
        cat_vec[cat] = vec
        cat_mat[cat] = vec.fit_transform([t for t, _ in items])

    _index = {
        "global_docs": global_docs, "global_meta": global_meta,
        "global_vec": global_vec, "global_mat": global_mat,
        "cat_docs": cat_docs, "cat_vec": cat_vec, "cat_mat": cat_mat,
    }
    return _index


def _search(vec, mat, query, top_k):
    sims = cosine_similarity(vec.transform([query]), mat).flatten()
    order = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in order if sims[i] > 0.01]


def retriever_node(state: AgentState) -> dict:
    weak = state.get("weak_indicator") or "健康"
    user_query = state.get("user_query", "")
    # 用真实用户问题 + 弱指标构造检索 query，而不是硬编码
    query = f"{weak} {user_query}".strip()
    trace_entry = {"node": "retriever", "query": query}

    idx = _build_index()
    main_cat, candidates = _route(query, weak)

    retrieved, seen = [], set()

    def _add(cat, items, hits):
        for i, score in hits:
            if cat:
                text, rel = items[i]
            else:
                text, rel = idx["global_docs"][i], idx["global_meta"][i]
            if rel in seen:
                continue
            seen.add(rel)
            # 带来源标注的文本，供 writer 引用
            retrieved.append(f"[来源: {rel}]\n{text}")

    # 1) 主分类内检索
    if main_cat and main_cat in idx["cat_vec"]:
        hits = _search(idx["cat_vec"][main_cat], idx["cat_mat"][main_cat],
                       query, TOP_K_RETRIEVAL)
        _add(main_cat, idx["cat_docs"][main_cat], hits)
    # 2) 候选分类补充
    for cat in candidates[1:]:
        if cat in idx["cat_vec"] and len(retrieved) < TOP_K_RETRIEVAL:
            hits = _search(idx["cat_vec"][cat], idx["cat_mat"][cat],
                           query, TOP_K_RETRIEVAL - len(retrieved))
            _add(cat, idx["cat_docs"][cat], hits)
    # 3) 全局兜底
    if len(retrieved) < TOP_K_RETRIEVAL:
        hits = _search(idx["global_vec"], idx["global_mat"], query, TOP_K_RETRIEVAL)
        _add(None, None, hits)

    retrieved = retrieved[:TOP_K_RETRIEVAL]
    trace_entry.update({"n_docs": len(retrieved), "routed_category": main_cat})
    trace = state.get("trace", []) + [trace_entry]
    return {"retrieved_docs": retrieved, "trace": trace}
