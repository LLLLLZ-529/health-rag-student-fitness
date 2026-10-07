"""HealthAgent 全局配置

路径说明：
- 本文件位于 code/health_agent/config.py
- 项目根 = code/health_agent 的上两级目录
- 知识库 chunks 位于项目根下 knowledge_base/chunks/
- 模型权重默认从 ModelScope 下载到 ~/.cache/modelscope/，也可通过环境变量覆盖
"""
from pathlib import Path
import os, glob

# 项目根（code/health_agent → 上溯 2 级 = 项目根）
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

# LLM 基础模型（Qwen2.5-7B-Instruct）
# 优先使用环境变量 BASE_LLM_PATH，否则自动探测 ~/.cache/modelscope 下的快照
_llm_cands = sorted(glob.glob(str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/*")))
DEFAULT_LLM = _llm_cands[-1] if _llm_cands else str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master")
BASE_LLM_PATH = Path(os.environ.get("BASE_LLM_PATH", DEFAULT_LLM))

# SFT / DPO 适配器路径（可通过环境变量覆盖；默认指向项目根下 models/）
SFT_ADAPTER_PATH = Path(os.environ.get("SFT_ADAPTER_PATH", str(PROJECT_ROOT / "models" / "sft_best_model")))
DPO_ADAPTER_PATH = Path(os.environ.get("DPO_ADAPTER_PATH", str(PROJECT_ROOT / "models" / "dpo_model")))

# RAG 知识库（公开来源 chunks）
CHUNKS_DIR = Path(os.environ.get("CHUNKS_DIR", str(PROJECT_ROOT / "knowledge_base" / "chunks")))
TOP_K_RETRIEVAL = 3

# 训练/推理参数
MAX_NEW_TOKENS = 256
MAX_REVISION_ROUNDS = 2  # Critic 打回最多轮次

# 9 类标签
LABELS_9 = [
    "低水平-退化型", "低水平-平稳型", "低水平-改善型",
    "中水平-退化型", "中水平-平稳型", "中水平-改善型",
    "高水平-退化型", "高水平-平稳型", "高水平-改善型",
]

# 指标关键词映射（Critic 用）
INDICATOR_KEYWORDS = {
    "肺活量": ["心肺", "有氧", "慢跑", "游泳", "深呼吸", "呼吸", "骑行", "快走", "耐力跑", "心肺功能", "腹式呼吸"],
    "力量":   ["力量", "抗阻", "深蹲", "硬拉", "卧推", "哑铃", "杠铃", "引体", "肌肉", "蛋白质", "抗阻训练", "举重"],
    "耐力":   ["耐力", "有氧", "慢跑", "骑行", "长距离", "间歇", "心肺", "持续跑", "匀速跑"],
    "柔韧":   ["柔韧", "拉伸", "瑜伽", "关节", "活动度", "静态拉伸", "压腿", "伸展"],
    "速度":   ["速度", "冲刺", "爆发力", "起跑", "短冲", "反应"],
    "BMI":    ["减重", "控制饮食", "热量", "减脂", "有氧运动", "体重", "饮食控制", "低热量"],
}
