#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
44_improved_retriever.py — 分类路由检索器（替代纯 TF-IDF）

旧检索问题：全局 TF-IDF 纯字面匹配，问"肺活量怎么训练"会被"体测标准/政策"里的
高频词带偏，召回 0/5。

改进：
  1. 分类路由：按 query 关键词先判定意图（运动处方/膳食营养/体测标准/健康政策）
  2. 分类内检索：在目标分类目录内做 TF-IDF，避免跨类干扰
  3. 全局兜底：分类内结果不足时回退全局检索补齐
  4. 带来源：返回 source/category，配合引用溯源 prompt

可直接替换 health_agent/agents/retriever.py 的 retriever_node。
"""
import os, re, json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba

# 关闭 jieba 初始化日志
jieba.setLogLevel(20)


def _tokenize(text):
    """jieba 中文分词：过滤单字和空白，解决 sklearn 默认不切中文导致召回为 0 的问题。"""
    return [w for w in jieba.cut(text) if len(w.strip()) > 1]


def _make_vectorizer(max_features):
    return TfidfVectorizer(
        tokenizer=_tokenize, token_pattern=None,
        max_features=max_features, ngram_range=(1, 2), min_df=1,
    )

# 分类关键词：(词, 权重)。强区分名词给 2 分，泛词给 1 分
CATEGORY_KEYWORDS = {
    "运动处方": [("训练", 1), ("运动", 1), ("怎么练", 2), ("锻炼", 2), ("提升", 1),
               ("方案", 1), ("计划", 1), ("跑步", 2), ("慢跑", 2), ("游泳", 2),
               ("跳绳", 2), ("力量", 2), ("肌肉", 2), ("有氧", 2), ("无氧", 2),
               ("拉伸", 2), ("瑜伽", 2), ("深蹲", 2), ("俯卧撑", 2), ("引体", 2),
               ("仰卧起坐", 2), ("肺活量", 2), ("耐力", 2), ("跳远", 2),
               ("健身", 2), ("组数", 2), ("频率", 2), ("热身", 2), ("心率", 2),
               ("间歇", 2), ("骑行", 2), ("高抬腿", 2), ("蛙跳", 2), ("冲刺", 2),
               ("体能", 2), ("减脂", 2), ("练", 1), ("负荷", 2)],
    "膳食营养": [("吃", 2), ("饮食", 2), ("营养", 2), ("热量", 2), ("蛋白", 2),
               ("维生素", 2), ("餐", 2), ("食谱", 2), ("膳食", 2), ("食物", 2),
               ("水果", 2), ("蔬菜", 2), ("碳水", 2), ("脂肪", 2), ("早餐", 2),
               ("午餐", 2), ("晚餐", 2), ("补钙", 2), ("补铁", 2), ("电解质", 2),
               ("牛奶", 2), ("鸡蛋", 2), ("瘦肉", 2), ("粗粮", 2), ("低盐", 2),
               ("低糖", 2)],
    "体测标准": [("标准", 2), ("及格", 2), ("满分", 2), ("评分", 2), ("几分", 2),
               ("多少分", 2), ("合格", 2), ("分数线", 2), ("怎么测", 2),
               ("测什么", 2), ("成绩", 1), ("分数", 2), ("评级", 2),
               ("优秀", 2), ("良好", 2), ("及格线", 2), ("权重", 2),
               ("占比", 2), ("加分", 2), ("扣分", 2)],
    "健康政策": [("政策", 2), ("文件", 2), ("规定", 2), ("通知", 2), ("制度", 2),
               ("法规", 2), ("意见", 2), ("纲要", 2), ("教育部", 2),
               ("体育总局", 2), ("健康中国", 2), ("学校", 1)],
}

# 弱指标 → 默认分类
WEAK_CATEGORY = {
    "肺活量": "运动处方", "耐力": "运动处方", "耐力跑": "运动处方",
    "力量": "运动处方", "立定跳远": "运动处方", "50米": "运动处方",
    "坐位体前屈": "运动处方", "柔韧": "运动处方", "BMI": "运动处方",
}


def route_query(query):
    """返回 (主分类, 候选分类列表)，按加权命中数排序。"""
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(w for kw, w in kws if kw in query)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if ranked[0][1] == 0:
        return None, []  # 无法判定
    return ranked[0][0], [c for c, s in ranked if s > 0]


# 口语修饰词：检索时归一化（query rewriting）
_COLLOQUIAL = re.compile(r"薄弱|很差|太差|不行|偏低|太低|比较差|怎么办|怎么练|"
                         r"怎么提升|怎么训练|如何提升|想提升|怎么提高|如何提高|"
                         r"怎么$|的$|低$|差$")


def rewrite_query(query, weak=None):
    """把口语查询改写为贴近文档措辞的检索串。
    如 '力量薄弱怎么练' + weak='力量' → '力量 训练 方法'。
    """
    q = _COLLOQUIAL.sub(" ", query)
    q = re.sub(r"[，。？?！!、]+", " ", q)
    prefix = weak or ""
    # 训练类意图补"训练方法"，标准类补"标准"
    main, _ = route_query(query)
    tail = "训练 方法" if main in (None, "运动处方") else ("标准" if main == "体测标准" else "")
    q = re.sub(r"\s+", " ", f"{prefix} {q} {tail}").strip()
    return q


class RoutedRetriever:
    def __init__(self, chunks_dir):
        self.chunks_dir = Path(chunks_dir)
        self.global_docs, self.global_meta = [], []
        self.cat_docs = {}  # cat -> [(text, rel)]
        self.cat_vec, self.cat_mat = {}, {}
        self.global_vec = self.global_mat = None
        self._build()

    def _build(self):
        for root, _, files in os.walk(self.chunks_dir):
            for fn in sorted(files):
                if not fn.endswith(".md"):
                    continue
                fp = Path(root) / fn
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                rel = str(fp.relative_to(self.chunks_dir))
                cat = rel.split("/")[0] if "/" in rel else "未分类"
                self.global_docs.append(text)
                self.global_meta.append(rel)
                self.cat_docs.setdefault(cat, []).append((text, rel))

        # 全局索引
        self.global_vec = _make_vectorizer(12000)
        self.global_mat = self.global_vec.fit_transform(self.global_docs)

        # 各分类索引
        for cat, items in self.cat_docs.items():
            vec = _make_vectorizer(8000)
            mat = vec.fit_transform([t for t, _ in items])
            self.cat_vec[cat] = vec
            self.cat_mat[cat] = mat

    def _search(self, vec, mat, docs, query, top_k):
        q = vec.transform([query])
        sims = cosine_similarity(q, mat).flatten()
        order = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0.01]

    def retrieve(self, query, top_k=3, weak=None):
        # 路由用原始 query；检索用改写后的 query（口语→文档措辞）
        search_query = rewrite_query(query, weak)
        # 先尝试弱指标指定分类
        main_cat, candidates = route_query(query)
        if weak and weak in WEAK_CATEGORY:
            main_cat = WEAK_CATEGORY[weak]
            if main_cat not in candidates:
                candidates.insert(0, main_cat)

        results, seen = [], set()

        def collect(cat, vec, mat, items, query, k):
            hits = self._search(vec, mat, items, query, k)
            for i, score in hits:
                rel = items[i][1] if cat else self.global_meta[i]
                if rel not in seen:
                    seen.add(rel)
                    text = items[i][0] if cat else self.global_docs[i]
                    results.append({
                        "source": rel,
                        "category": cat or rel.split("/")[0],
                        "text": text,
                        "score": round(score, 4),
                    })

        # 1) 主分类内检索
        if main_cat and main_cat in self.cat_vec:
            items = self.cat_docs[main_cat]
            collect(main_cat, self.cat_vec[main_cat], self.cat_mat[main_cat],
                    items, search_query, top_k)

        # 2) 候选分类补充
        for cat in candidates[1:]:
            if cat in self.cat_vec and len(results) < top_k:
                items = self.cat_docs[cat]
                collect(cat, self.cat_vec[cat], self.cat_mat[cat],
                        items, search_query, top_k - len(results))

        # 3) 全局兜底
        if len(results) < top_k:
            collect(None, self.global_vec, self.global_mat, None,
                    search_query, top_k)

        return results[:top_k]


def evaluate(retriever):
    """用 40 脚本同款测试集评估召回率。"""
    test = [
        ("肺活量低怎么训练", "运动处方"),
        ("BMI高怎么减脂", "运动处方"),
        ("耐力跑的及格标准", "体测标准"),
        ("想提升力量应该怎么练", "运动处方"),
        ("运动后饮食注意事项", "膳食营养"),
    ]
    print(f"{'问题':<20} {'路由分类':<8} {'命中':<6} 检索到的分类")
    print("-" * 70)
    hit = 0
    for q, expect in test:
        main, _ = route_query(q)
        res = retriever.retrieve(q, top_k=3)
        cats = [r["category"] for r in res]
        ok = expect in cats
        hit += ok
        flag = "✅" if ok else "❌"
        print(f"{q:<20} {str(main):<8} {flag}     {cats}")
        for r in res[:2]:
            print(f"      [{r['score']:.3f}] {r['source'][:50]}")
    print(f"\n召回率: {hit}/{len(test)}")


if __name__ == "__main__":
    import sys
    # 优先用公开知识库（453块），其次完整知识库
    pub = Path(__file__).resolve().parents[1] / "knowledge_base" / "chunks"
    full = Path(__file__).resolve().parents[2] / "data" / "chunks" / "chunks"
    chunks_dir = pub if pub.exists() else full
    print(f"知识库目录: {chunks_dir}\n")
    r = RoutedRetriever(chunks_dir)
    print(f"索引文档: {len(r.global_docs)} 个，分类: {list(r.cat_docs)}\n")
    evaluate(r)
