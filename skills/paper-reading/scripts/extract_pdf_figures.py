"""
从论文 PDF 中提取 Figure 图片。
策略: 嵌入图片提取 → 页面渲染+自动裁剪 → 页面渲染+手动裁剪

用法:
  python extract_pdf_figures.py <pdf_path> <output_dir> [--dpi 300]
  python extract_pdf_figures.py <pdf_path> <output_dir> --crop <page>:<x0>,<y0>,<x1>,<y1> --name <figure_name>
"""

import fitz
import os
import re
import sys
import json


def extract_embedded_images(doc, out_dir, min_size=100):
    """方式1: 提取嵌入的位图图片（适用于包含位图 figure 的论文）"""
    results = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        images = page.get_images(full=True)
        for i, img_info in enumerate(images):
            xref = img_info[0]
            base_img = doc.extract_image(xref)
            w, h = base_img["width"], base_img["height"]
            if w >= min_size and h >= min_size:
                ext = base_img["ext"]
                fname = f"page{page_num+1}_img{i+1}_{w}x{h}.{ext}"
                path = os.path.join(out_dir, fname)
                with open(path, "wb") as f:
                    f.write(base_img["image"])
                results.append({"page": page_num+1, "file": fname, "size": f"{w}x{h}", "method": "embedded"})
    return results


def find_figure_regions(doc):
    """自动检测 Figure 标题的位置，推断 figure 区域"""
    figures = []
    pattern = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_rect = page.rect

        for block in blocks:
            if block["type"] != 0:  # 只看文本块
                continue
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"])
                match = pattern.search(text)
                if match:
                    fig_num = match.group(1)
                    # 标题所在行的 bbox
                    caption_bbox = fitz.Rect(line["bbox"])
                    figures.append({
                        "fig_num": fig_num,
                        "page": page_num,
                        "caption_bbox": caption_bbox,
                        "page_rect": page_rect,
                    })
    return figures


def crop_figure_region(doc, page_num, rect, out_path, dpi=300):
    """从指定页面裁剪指定区域，渲染为图片"""
    page = doc[page_num]
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    clip = fitz.Rect(rect)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    pix.save(out_path)
    return pix.width, pix.height


def render_full_page(doc, page_num, out_path, dpi=300):
    """渲染整页为图片"""
    page = doc[page_num]
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    pix.save(out_path)
    return pix.width, pix.height


def auto_extract_figures(pdf_path, out_dir, dpi=300):
    """自动提取流程: 先尝试嵌入图片，再尝试 figure 区域检测+页面截图"""
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    all_results = []

    # Step 1: 提取嵌入图片
    embedded = extract_embedded_images(doc, out_dir)
    all_results.extend(embedded)
    if embedded:
        print(f"[嵌入图片] 提取到 {len(embedded)} 张有效图片")
    else:
        print("[嵌入图片] 未找到有效的嵌入位图（figure 可能是矢量绘制的）")

    # Step 2: 检测 figure 标题位置
    figures = find_figure_regions(doc)
    if figures:
        print(f"[Figure检测] 检测到 {len(figures)} 个 Figure 标题:")
        for fig in figures:
            print(f"  Figure {fig['fig_num']} → 第{fig['page']+1}页 (caption at y={fig['caption_bbox'].y0:.0f})")

    # Step 3: 对包含 figure 的页面做全页截图（供后续手动裁剪或 AI 辅助裁剪）
    figure_pages = set(fig["page"] for fig in figures)
    for pg in sorted(figure_pages):
        fname = f"temp_page{pg+1}_full_{dpi}dpi.png"
        path = os.path.join(out_dir, fname)
        w, h = render_full_page(doc, pg, path, dpi)
        all_results.append({"page": pg+1, "file": fname, "size": f"{w}x{h}", "method": "page_render"})
        print(f"[页面截图] 第{pg+1}页 → {fname} ({w}x{h})")

    # 在关闭文档前，先收集需要的信息
    total_pages = doc.page_count
    detected_figures = [{"fig_num": f["fig_num"], "page": f["page"]+1,
                         "caption_y": round(f["caption_bbox"].y0, 1)} for f in figures]
    doc.close()

    # 输出摘要 JSON
    summary_path = os.path.join(out_dir, "extraction_summary.json")
    summary = {
        "pdf": pdf_path,
        "total_pages": total_pages,
        "embedded_images": len(embedded),
        "detected_figures": detected_figures,
        "files": all_results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n摘要已保存到 {summary_path}")
    return summary


def manual_crop(pdf_path, out_dir, page_num, x0, y0, x1, y1, name, dpi=300):
    """手动指定区域裁剪"""
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    fname = f"{name}.png"
    path = os.path.join(out_dir, fname)
    w, h = crop_figure_region(doc, page_num - 1, (x0, y0, x1, y1), path, dpi)
    doc.close()
    print(f"已裁剪: {fname} ({w}x{h})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法:")
        print("  自动提取: python extract_pdf_figures.py <pdf> <out_dir> [--dpi 300]")
        print("  手动裁剪: python extract_pdf_figures.py <pdf> <out_dir> --crop <page>:<x0>,<y0>,<x1>,<y1> --name <name>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    out_dir = sys.argv[2]

    if "--crop" in sys.argv:
        idx = sys.argv.index("--crop")
        crop_spec = sys.argv[idx + 1]  # e.g. "1:300,50,600,350"
        page_str, coords_str = crop_spec.split(":")
        page_num = int(page_str)
        x0, y0, x1, y1 = [float(c) for c in coords_str.split(",")]
        name = "cropped_figure"
        if "--name" in sys.argv:
            name = sys.argv[sys.argv.index("--name") + 1]
        dpi = 300
        if "--dpi" in sys.argv:
            dpi = int(sys.argv[sys.argv.index("--dpi") + 1])
        manual_crop(pdf_path, out_dir, page_num, x0, y0, x1, y1, name, dpi)
    else:
        dpi = 300
        if "--dpi" in sys.argv:
            dpi = int(sys.argv[sys.argv.index("--dpi") + 1])
        auto_extract_figures(pdf_path, out_dir, dpi)
