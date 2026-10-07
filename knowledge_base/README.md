# Knowledge Base

This directory contains the public-source subset of the health knowledge base used for RAG retrieval in M3.

## Included Chunks

**453 chunks** from public government and WHO publications, organized into 4 categories:

### 体测标准 (Fitness Standards) — 16 chunks

| Source | Type |
|--------|------|
| 国民体质测定标准手册（成年人部分） | Government publication |
| 大学生国家体质健康测试评分标准 | Government standard |

### 健康政策 (Health Policy) — 35 chunks

| Source | Type |
|--------|------|
| "健康中国2030"规划纲要 | Government policy |

### 膳食营养 (Diet & Nutrition) — 142 chunks

| Source | Type |
|--------|------|
| 成人肥胖食养指南（2024年版） | Government guideline |
| 中国居民平衡膳食宝塔 | Government guideline |
| 主要食物营养成分表 | Government reference |

### 运动处方 (Exercise Prescription) — 260 chunks

| Source | Type |
|--------|------|
| WHO 2018-2030 全球身体活动行动计划 | WHO publication |
| WHO 关于身体活动有益健康的全球建议（2010） | WHO publication |
| WHO 关于身体活动和久坐行为的指南（2020） | WHO publication |
| 全民健身指南（国家体育总局） | Government publication |
| 全年龄段人群科学健身指南（国家体育总局） | Government publication |
| 慢性疾病运动干预中心运动风险防控指南（征求意见稿） | Government draft |
| BROG 改良 CR10 量表 | Clinical reference |

## INDEX.json

`chunks/INDEX.json` contains metadata for all 453 chunks:
- `chunk_id`: Unique identifier
- `source`: Source document name
- `category`: One of the 4 categories above
- `file_path`: Relative path to the chunk markdown file
- `char_count`: Character count of the chunk

## Not Included (Full Knowledge Base)

The complete knowledge base used in the original experiments contained **2,674 chunks**. The following sources are **excluded** from this public repository due to copyright restrictions:

| Source | Chunks | Reason for Exclusion |
|--------|--------|----------------------|
| 膳食营养素参考摄入量（2023） | ~1,071 | Z-Library pirated source |
| 中国营养科学全书 | ~550 | Z-Library pirated source |
| 实用运动营养学 | ~381 | Z-Library pirated source |
| ACSM 运动测试与处方指南 | ~156 | Copyrighted textbook |
| 中国居民膳食指南科学依据 | ~62 | Copyrighted companion |
| 中青年科学运动指南解读 | ~1 | Copyrighted |

### Rebuilding the Full Knowledge Base

If you have legal access to these books, you can rebuild the full knowledge base:

1. Place the source PDFs in a directory
2. Run the chunking script:
   ```bash
   python scripts/chunk_knowledge_v3.py --input-dir /path/to/pdfs --output-dir knowledge_base/chunks
   ```
3. The script will generate markdown chunks and update `INDEX.json`

## Chunk Format

Each chunk is a markdown file with:
- YAML front matter (source, category, page numbers)
- Extracted text content
- Character count typically 500–2000 characters per chunk

## Retrieval

M3 uses TF-IDF vectorization over chunk content for retrieval. The retrieval code is in `code/19_m3_rag_recommendation.py` and `code/health_agent/tools.py`.
