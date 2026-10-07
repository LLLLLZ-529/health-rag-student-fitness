"""
utils/rag_retriever.py — 基于 TF-IDF 的 RAG 检索器

加载2678个知识chunk，构建TF-IDF索引，支持按query检索top-k相关知识。
不依赖外部向量数据库，纯sklearn实现。
"""
from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGRetriever:
    """基于TF-IDF的知识检索器"""

    def __init__(self, chunks_dir: str, top_k: int = 3):
        self.chunks_dir = Path(chunks_dir)
        self.top_k = top_k
        self.chunks: List[Dict] = []
        self.texts: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._loaded = False

    def load(self):
        """加载所有chunk并构建TF-IDF索引"""
        if self._loaded:
            return

        index_file = self.chunks_dir / "INDEX.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
        else:
            index = []

        for item in index:
            chunk_path = self.chunks_dir / item["path"]
            if chunk_path.exists():
                text = self._read_chunk(chunk_path)
                if text:
                    self.chunks.append({
                        "path": item["path"],
                        "category": item.get("category", ""),
                        "preview": item.get("preview", ""),
                        "text": text,
                    })
                    self.texts.append(text)

        print(f"RAG检索器: 加载 {len(self.chunks)} 个chunk")

        # 构建TF-IDF索引
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        self._loaded = True
        print(f"RAG检索器: TF-IDF索引构建完成 ({self.tfidf_matrix.shape})")

    def _read_chunk(self, path: Path) -> str:
        """读取chunk文件，提取正文（去掉YAML frontmatter）"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # 去掉YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2]
            # 清理多余空白
            content = re.sub(r"\s+", " ", content).strip()
            return content[:2000]  # 限制长度
        except:
            return ""

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 category: Optional[str] = None) -> List[Dict]:
        """
        检索相关知识

        Args:
            query: 检索查询
            top_k: 返回数量
            category: 限定类别（运动处方/膳食营养/体测标准/健康政策）

        Returns:
            相关chunk列表，每个包含text/category/score
        """
        if not self._loaded:
            self.load()

        k = top_k or self.top_k

        # 计算query的TF-IDF向量
        query_vec = self.vectorizer.transform([query])

        # 计算相似度
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 如果限定类别，先过滤
        if category:
            cat_mask = np.array([1 if c["category"] == category else 0 for c in self.chunks])
            similarities = similarities * cat_mask

        # 取top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "text": self.chunks[idx]["text"],
                    "category": self.chunks[idx]["category"],
                    "score": float(similarities[idx]),
                    "preview": self.chunks[idx]["preview"],
                })
        return results

    def retrieve_for_action(self, action_idx: int, state: "StudentState") -> List[Dict]:
        """
        根据Agent选择的动作和学生状态，检索相关知识

        Args:
            action_idx: 动作索引（0-7）
            state: 学生状态

        Returns:
            相关知识列表
        """
        action_queries = {
            0: "耐力训练 跑步 游泳 有氧运动 心肺功能",
            1: "力量训练 引体向上 俯卧撑 举重 肌肉",
            2: "柔韧训练 拉伸 瑜伽 坐位体前屈 灵活性",
            3: "爆发力训练 跳远 短跑 跳高 速度",
            4: "心肺功能 肺活量 有氧间歇 呼吸训练",
            5: "BMI 体重管理 饮食控制 减脂 有氧运动",
            6: "综合训练 全身运动 循环训练 健身",
            7: "休息 恢复 过度训练 运动损伤 疲劳",
        }

        # 加上最弱指标的关键词
        weakest_names = ["BMI", "肺活量", "50米跑", "立定跳远", "坐位体前屈", "耐力跑", "力量"]
        weakest_query = weakest_names[state.weakest_idx]

        query = f"{action_queries.get(action_idx, '')} {weakest_query}"
        return self.retrieve(query, top_k=2)


# 动作中文名称
ACTION_NAMES_CN = [
    "耐力训练",
    "力量训练",
    "柔韧训练",
    "爆发力训练",
    "心肺功能训练",
    "BMI管理",
    "综合训练",
    "休息恢复",
]


def generate_recommendation(action_idx: int, state: "StudentState",
                            retrieved_knowledge: List[Dict]) -> str:
    """
    基于检索到的知识生成推荐理由（模板化，无LLM）

    Args:
        action_idx: 推荐策略
        state: 学生状态
        retrieved_knowledge: 检索到的知识

    Returns:
        推荐理由文本
    """
    action_name = ACTION_NAMES_CN[action_idx]
    weakest = state.weakest_name
    weakest_names_cn = {
        "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50米跑",
        "standing_long_jump": "立定跳远", "sit_and_reach": "坐位体前屈",
        "endurance_run": "耐力跑", "strength": "力量",
    }
    weakest_cn = weakest_names_cn.get(weakest, weakest)

    # 基础推荐
    rec = f"针对您当前最弱的{weakest_cn}指标，建议进行{action_name}。"

    # 加入检索到的知识
    if retrieved_knowledge:
        top_knowledge = retrieved_knowledge[0]
        # 提取知识中的关键句子（前100字）
        knowledge_text = top_knowledge["text"][:100]
        if knowledge_text:
            rec += f"参考《{top_knowledge.get('preview', '健康指南')}》：{knowledge_text}..."

    # 个性化建议
    if state.hi_total < -0.3:
        rec += "您的整体健康水平偏低，建议循序渐进，避免过度训练。"
    elif state.hi_total > 0.5:
        rec += "您的基础较好，可以适当增加训练强度。"
    else:
        rec += "建议保持规律训练，每周3-4次，每次30-45分钟。"

    return rec
