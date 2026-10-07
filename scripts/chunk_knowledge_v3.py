"""
知识库切块脚本 V3
核心改进：
- 修复段落合并逻辑：按字符数合并，不依赖分隔符
- 扫描 PDF 标注清晰，不强行切块
- 输出 md 格式，YAML frontmatter
"""
import sys, os, re, json, shutil
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ===== 依赖 =====
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ===== 配置 =====
KB_ROOT      = Path(r"d:\大创\知识库\知识库")
OUTPUT_ROOT  = Path(r"d:\大创\知识库\chunks")

CHUNK_SIZE    = 800    # 目标 chunk 字符数
CHUNK_OVERLAP = 150   # 相邻 chunk 重叠
MIN_CHUNK     = 250    # 小于此才合并
MAX_CHUNK     = 2000   # 大于此再切

CATEGORIES = {
    "体测标准": {"category": "体测标准", "topic": "大学生体测评分与标准"},
    "运动处方": {"category": "运动处方", "topic": "运动处方指南与干预"},
    "膳食营养": {"category": "膳食营养", "topic": "膳食营养与健康"},
    "健康政策": {"category": "健康政策", "topic": "国家健康政策法规"},
    "数据集":   {"category": "数据集",   "topic": "结构化统计数据"},
}

# ===== 文本提取 =====

def extract_pdf(filepath):
    text = ""
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(str(filepath))
            parts = [page.get_text().strip() for page in doc]
            doc.close()
            text = "\n".join(p for p in parts if p)
        except Exception as e:
            print(f"    [MuPDF ERR] {e}")
    if not text and HAS_PDF:
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(str(filepath)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t.strip())
            text = "\n".join(parts)
        except Exception as e:
            print(f"    [pdfplumber ERR] {e}")
    return text.strip() if text else None


def extract_docx(filepath):
    try:
        doc = DocxDocument(str(filepath))
        lines = []
        for para in doc.paragraphs:
            t = para.text.strip()
            if not t:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            # 检测标题
            is_h = False
            if para.style and para.style.name:
                sn = para.style.name.lower()
                if 'heading' in sn or 'title' in sn or '标题' in sn:
                    is_h = True
            if is_h:
                lines.append(f"## {t}")
            else:
                lines.append(t)
        return "\n".join(lines)
    except Exception as e:
        print(f"    [DOCX ERR] {e}")
        return None


def extract_md(filepath):
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312"):
        try:
            return filepath.read_text(encoding=enc)
        except:
            continue
    return None


def extract_text(filepath):
    suf = filepath.suffix.lower()
    if suf == ".md":
        return extract_md(filepath)
    elif suf == ".pdf":
        return extract_pdf(filepath)
    elif suf == ".docx":
        if not HAS_DOCX:
            print("    [SKIP] python-docx 不可用")
            return None
        return extract_docx(filepath)
    elif suf in (".xlsx", ".xls", ".csv", ".doc"):
        print(f"    [SKIP] 结构化/旧格式")
        return None
    else:
        print(f"    [SKIP] 未知格式")
        return None


# ===== 文本清理 =====

def clean_text(text):
    # 去掉页眉页脚数字（孤立数字行）
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        s = line.strip()
        # 跳过纯页码
        if re.match(r'^[0-9]+$', s) or re.match(r'^第[一二三四五六七八九十]+章', s):
            continue
        # 跳过太短的乱码行（大量孤立字符）
        if len(s) <= 2 and s and not re.match(r'^[。，、；：]/s*$', s):
            continue
        cleaned.append(line)
    text = '\n'.join(cleaned)
    # 合并多余空行
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


# ===== 切块核心 =====

def split_to_sentences(text):
    """
    把文本切成句子（按中英文句末标点）
    """
    # 中文句末：。！？；   英文句末：. ! ?
    # 注意英文缩写中的 . 不分割
    # 简化版：按标点+空格分割
    parts = re.split(r'(?<=[。！？；\.!?]\s)', text)
    result = []
    for p in parts:
        p = p.strip()
        if p:
            result.append(p)
    return result


def chunk_text_smart(text):
    """
    智能切块：
    1. 先按 \n\n 分大段落
    2. 把大段落切成句子
    3. 按 CHUNK_SIZE 合并句子
    4. 保留 CHUNK_OVERLAP 重叠
    """
    text = clean_text(text)
    if not text or len(text) < MIN_CHUNK:
        return []

    # 第一步：按双换行分大段落
    raw_paras = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip().replace('\n', '') for p in raw_paras if p.strip()]

    # 第二步：把大段落（>1000字）切成句子
    final_paras = []
    for para in paragraphs:
        if len(para) > 1000:
            sentences = split_to_sentences(para)
            final_paras.extend(sentences)
        else:
            final_paras.append(para)

    # 第三步：合并成 chunk
    chunks = []
    current = ""
    for para in final_paras:
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + para) if current else para
        else:
            if current and len(current) >= MIN_CHUNK:
                chunks.append(current)
            # 重叠：把当前 para 的前 CHUNK_OVERLAP 字符作为下一个 chunk 开头
            if len(para) > CHUNK_OVERLAP:
                current = para[:CHUNK_OVERLAP] + "…\n\n" + para
            else:
                current = para

    if current and len(current) >= MIN_CHUNK:
        chunks.append(current)

    # 第四步：把太大的 chunk 再切
    final = []
    for c in chunks:
        if len(c) > MAX_CHUNK:
            # 按句子切
            sentences = split_to_sentences(c)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) + 2 <= MAX_CHUNK:
                    buf = (buf + "\n\n" + s) if buf else s
                else:
                    if buf:
                        final.append(buf)
                    buf = s
            if buf:
                final.append(buf)
        else:
            final.append(c)

    return final


def chunk_markdown(text):
    """
    Markdown 文件专用切块：按标题分节，每节再按大小切
    """
    text = clean_text(text)
    if not text or len(text) < MIN_CHUNK:
        return []

    # 按 ### / ## / # 标题分割
    sections = re.split(r'\n(?=#{1,3}\s)', text)
    chunks = []
    current_section = ""

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # 如果当前节 + 新节 < CHUNK_SIZE * 2，合并
        if len(current_section) + len(sec) < CHUNK_SIZE * 2:
            current_section = (current_section + "\n\n" + sec) if current_section else sec
        else:
            if current_section and len(current_section) >= MIN_CHUNK:
                # 如果当前节太大，再用智能切块细分
                if len(current_section) > CHUNK_SIZE:
                    sub = chunk_text_smart(current_section)
                    chunks.extend(sub)
                else:
                    chunks.append(current_section)
            current_section = sec

    if current_section:
        if len(current_section) > CHUNK_SIZE:
            sub = chunk_text_smart(current_section)
            chunks.extend(sub)
        elif len(current_section) >= MIN_CHUNK:
            chunks.append(current_section)

    return chunks if chunks else chunk_text_smart(text)


def chunk_text(text, is_markdown=False):
    """统一切块入口"""
    if is_markdown and re.search(r'\n#{1,4}\s', text):
        return chunk_markdown(text)
    return chunk_text_smart(text)


# ===== 元数据 =====

def guess_title(filepath):
    name = filepath.stem
    name = re.sub(r'^\d+\s+', '', name)
    name = re.sub(r'\s*\(.*?\)\s*$', '', name)
    return name[:80] if len(name) > 80 else name


def make_metadata(filepath, cat_info, idx, total, content):
    title = guess_title(filepath)
    preview = content[:80].replace('\n', ' ').strip()
    if len(content) > 80:
        preview += "..."
    return {
        "source_file":  filepath.name,
        "source_title": title,
        "category":    cat_info["category"],
        "topic":       cat_info["topic"],
        "chunk_index":  idx,
        "total_chunks": total,
        "chunk_size_chars": len(content),
        "preview": preview,
        "format":      filepath.suffix.lower().lstrip('.'),
    }


def meta_to_yaml(meta):
    lines = ["---"]
    for k, v in meta.items():
        v_str = str(v)
        if any(c in v_str for c in ['"', ':', '\n']):
            lines.append(f'{k}: "{v_str}"')
        else:
            lines.append(f'{k}: {v_str}')
    lines.append("---")
    return "\n".join(lines)


# ===== 主流程 =====

def process_category(cat_dir, cat_info):
    out_dir = OUTPUT_ROOT / cat_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in ("*.md", "*.pdf", "*.docx"):
        files.extend(cat_dir.glob(ext))
    for sub in cat_dir.iterdir():
        if sub.is_dir():
            for ext in ("*.md", "*.pdf", "*.docx"):
                files.extend(sub.glob(ext))

    files = sorted(files, key=lambda f: f.name)
    stats = {"files": 0, "chunks": 0, "skipped": 0, "scan": 0}

    for fp in files:
        if fp.name.startswith("~$"):
            continue
        print(f"\n  📄 {fp.name}")
        text = extract_text(fp)
        if text is None:
            stats["skipped"] += 1
            continue

        # 检测扫描件
        if len(text.strip()) < 200:
            print(f"    ⚠ 疑似扫描件（{len(text.strip())}字符），生成占位标记")
            stats["scan"] += 1
            meta = make_metadata(fp, cat_info, 1, 1,
                               f"[SCAN] 需OCR。文件大小: {fp.stat().st_size//1024}KB")
            safe = re.sub(r'[^\w\s\-.并]', '_', fp.stem)[:60]
            out = out_dir / f"{safe}__NEEDS_OCR.md"
            out.write_text(
                f"{meta_to_yaml(meta)}\n\n"
                f"⚠ 此文件为扫描版PDF，无可用文字层。\n"
                f"源文件路径: {fp}\n"
                f"文件大小: {fp.stat().st_size//1024}KB\n",
                encoding="utf-8"
            )
            stats["chunks"] += 1
            continue

        is_md = fp.suffix.lower() == ".md"
        chunks = chunk_text(text, is_markdown=is_md)
        if not chunks:
            print("    → 无可切块内容")
            stats["skipped"] += 1
            continue

        safe = re.sub(r'[^\w\s\-.秋]', '_', fp.stem)[:60]
        for i, c in enumerate(chunks):
            meta = make_metadata(fp, cat_info, i+1, len(chunks), c)
            out = out_dir / f"{safe}__chunk{i+1:03d}.md"
            out.write_text(f"{meta_to_yaml(meta)}\n\n{c}", encoding="utf-8")

        stats["files"] += 1
        stats["chunks"] += len(chunks)
        avg = sum(len(c) for c in chunks) // max(len(chunks), 1)
        print(f"    → {len(chunks)} chunks  (平均每块 {avg} 字符)")

    return stats


def generate_index():
    records = []
    for cat_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        for cf in sorted(cat_dir.glob("*.md")):
            try:
                t = cf.read_text(encoding="utf-8")
                ls = t.split('\n')
                fm_end = 0
                in_fm = False
                for i, line in enumerate(ls):
                    if line.strip() == "---":
                        if not in_fm:
                            in_fm = True
                        else:
                            fm_end = i
                            break
                body = ls[fm_end+1:] if fm_end else ls
                first = next((l.strip() for l in body if l.strip()), "")
            except:
                first = ""
            records.append({
                "path": str(cf.relative_to(OUTPUT_ROOT)).replace("\\", "/"),
                "category": cat_dir.name,
                "preview": first[:120],
            })
    idx = OUTPUT_ROOT / "INDEX.json"
    idx.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📋 索引: {idx} ({len(records)} 条)")


def main():
    print("=" * 60)
    print("知识库切块工具 V3")
    print("=" * 60)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    total = {"files": 0, "chunks": 0, "skipped": 0, "scan": 0}
    for cat_name, cat_info in CATEGORIES.items():
        cat_dir = KB_ROOT / cat_name
        if not cat_dir.exists() or not cat_dir.is_dir():
            continue
        print(f"\n📁 处理: {cat_name}")
        s = process_category(cat_dir, cat_info)
        for k in s:
            total[k] += s[k]
        print(f"  ✅ {cat_name}: {s['files']}文件→{s['chunks']}chunks  跳过{s['skipped']}  扫描{s['scan']}")

    generate_index()

    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"  文件: {total['files']}   chunks: {total['chunks']}")
    print(f"  跳过: {total['skipped']}   扫描件(需OCR): {total['scan']}")
    print(f"  输出: {OUTPUT_ROOT}")
    print(f"{'='*60}")

    if total['scan'] > 0:
        print(f"\n⚠ 需OCR文件（已生成占位chunk）：")
        for cat_dir in OUTPUT_ROOT.iterdir():
            if cat_dir.is_dir():
                for f in cat_dir.glob("*NEEDS_OCR*"):
                    print(f"  - {cat_dir.name}/{f.name}")


if __name__ == "__main__":
    main()
