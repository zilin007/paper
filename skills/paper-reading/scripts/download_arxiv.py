#!/usr/bin/env python3
"""
arXiv 论文搜索与下载脚本。
用法: python download_arxiv.py "<论文名称或arXiv ID>" "<保存目录>"
输出: JSON 格式的论文元信息，PDF 保存到指定目录。
"""

import sys
import os
import re
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def search_arxiv(query, max_results=5):
    """通过 arXiv API 搜索论文，返回结果列表。"""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PaperReadingSkill/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"arXiv API 请求失败: {e}")

    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    results = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        if title_el is None:
            continue

        title = " ".join(title_el.text.strip().split())
        summary = entry.find("atom:summary", ns)
        summary_text = " ".join(summary.text.strip().split()) if summary is not None else ""

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None:
                authors.append(name.text.strip())

        published = entry.find("atom:published", ns)
        published_text = published.text.strip()[:10] if published is not None else ""

        pdf_url = ""
        arxiv_id = ""
        for link in entry.findall("atom:link", ns):
            href = link.get("href", "")
            if link.get("title") == "pdf":
                pdf_url = href
            if "/abs/" in href:
                arxiv_id = href.split("/abs/")[-1]

        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term", "")
            if term:
                categories.append(term)

        results.append({
            "title": title,
            "authors": authors,
            "abstract": summary_text,
            "published": published_text,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "categories": categories,
        })

    return results


def get_by_id(arxiv_id):
    """通过 arXiv ID 直接获取论文信息。"""
    clean_id = arxiv_id.strip().replace("arXiv:", "")
    params = urllib.parse.urlencode({"id_list": clean_id})
    url = f"{ARXIV_API}?{params}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PaperReadingSkill/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"arXiv API 请求失败 (ID查询): {e}")

    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entry = root.find("atom:entry", ns)
    if entry is None:
        return None

    title_el = entry.find("atom:title", ns)
    if title_el is None or title_el.text is None:
        return None

    title = " ".join(title_el.text.strip().split())
    summary = entry.find("atom:summary", ns)
    summary_text = " ".join(summary.text.strip().split()) if summary is not None and summary.text else ""

    authors = []
    for author in entry.findall("atom:author", ns):
        name = author.find("atom:name", ns)
        if name is not None:
            authors.append(name.text.strip())

    published = entry.find("atom:published", ns)
    published_text = published.text.strip()[:10] if published is not None else ""

    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"

    categories = []
    for cat in entry.findall("atom:category", ns):
        term = cat.get("term", "")
        if term:
            categories.append(term)

    return {
        "title": title,
        "authors": authors,
        "abstract": summary_text,
        "published": published_text,
        "arxiv_id": clean_id,
        "pdf_url": pdf_url,
        "categories": categories,
    }


def download_pdf(pdf_url, save_path):
    """下载 PDF 文件到指定路径。"""
    for attempt in range(3):
        try:
            req = urllib.request.Request(pdf_url, headers={"User-Agent": "PaperReadingSkill/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()

            with open(save_path, "wb") as f:
                f.write(content)

            size_mb = len(content) / (1024 * 1024)
            return size_mb
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"PDF 下载失败: {e}")


def sanitize_filename(name):
    """将论文标题转为安全的文件名。"""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    if len(name) > 100:
        name = name[:100]
    return name


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "用法: python download_arxiv.py <论文名称或ID> <保存目录>"}))
        sys.exit(1)

    query = sys.argv[1].strip()
    save_dir = sys.argv[2].strip()

    os.makedirs(save_dir, exist_ok=True)

    # 判断是 arXiv ID 还是论文名称
    paper = None
    if ARXIV_ID_PATTERN.match(query):
        paper = get_by_id(query)
        if paper is None:
            print(json.dumps({"error": f"未找到 arXiv ID: {query}"}, ensure_ascii=False))
            sys.exit(1)
    else:
        results = search_arxiv(query)
        if not results:
            print(json.dumps({"error": f"未找到匹配论文: {query}"}, ensure_ascii=False))
            sys.exit(1)
        paper = results[0]

    # 下载 PDF
    filename = sanitize_filename(paper["title"]) + ".pdf"
    save_path = os.path.join(save_dir, filename)
    size_mb = download_pdf(paper["pdf_url"], save_path)

    # 输出结果
    result = {
        "status": "success",
        "title": paper["title"],
        "authors": paper["authors"],
        "abstract": paper["abstract"],
        "published": paper["published"],
        "arxiv_id": paper["arxiv_id"],
        "pdf_url": paper["pdf_url"],
        "categories": paper["categories"],
        "saved_path": save_path,
        "file_size_mb": round(size_mb, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
