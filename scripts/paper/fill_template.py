"""Pour results/paper/submission.md into a Word template, keeping the template's styles.

Usage. python scripts/paper/fill_template.py TEMPLATE.docx OUT.docx [--md results/paper/submission.md] [--keep-body]

The template's styles, page setup, headers and footers are kept. By default every body paragraph and table of the
template is removed and the markdown is appended with the template's own style names (Title, Heading 1, Heading 2,
Heading 3, Normal, Caption, List Bullet, and the first table style found). With --keep-body the template's existing
paragraphs stay and the content is appended after them. Images referenced as ![..](path) are inserted at 6.3 inches wide.
Markdown handled. # headings, paragraphs, **bold** and *italic* runs, - bullets, | tables |, ![img](path).
"""
import argparse, re, sys
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def style_or(doc, *names):
    have = {s.name for s in doc.styles}
    for n in names:
        if n in have:
            return n
    return None


def add_runs(par, text):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("*") and part.endswith("*"):
            r = par.add_run(part[1:-1]); r.italic = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Courier New"
        else:
            par.add_run(part)


def clear_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("}sectPr"):
            continue
        body.remove(child)


def add_table(doc, rows, table_style):
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    data = []
    for r in rows[2:]:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        data.append(cells[: len(header)])
    t = doc.add_table(rows=1 + len(data), cols=len(header))
    if table_style:
        try:
            t.style = table_style
        except Exception:
            pass
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]; cell.text = ""
        p = cell.paragraphs[0]; add_runs(p, h)
        for r in p.runs:
            r.bold = True; r.font.size = Pt(8.5)
    for i, row in enumerate(data, start=1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]; cell.text = ""
            p = cell.paragraphs[0]; add_runs(p, v)
            for r in p.runs:
                r.font.size = Pt(8.5)
    doc.add_paragraph()


def fill(template, out, md, keep_body):
    doc = Document(template)
    s_title = style_or(doc, "Title")
    s_h = {1: style_or(doc, "Heading 1"), 2: style_or(doc, "Heading 2"), 3: style_or(doc, "Heading 3")}
    s_normal = style_or(doc, "Normal") or "Normal"
    s_caption = style_or(doc, "Caption")
    s_bullet = style_or(doc, "List Bullet", "List Paragraph")
    table_style = None
    for st in doc.styles:
        if st.type == 3 and st.name not in ("Normal Table",):  # WD_STYLE_TYPE.TABLE
            table_style = st.name; break
    if not keep_body:
        clear_body(doc)
    lines = Path(md).read_text().splitlines()
    i = 0
    base = Path(md).parent
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1; continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            if len(block) >= 2:
                add_table(doc, block, table_style)
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            level = len(m.group(1)); text = m.group(2).strip()
            if level == 1 and s_title:
                doc.add_paragraph(text, style=s_title)
            else:
                st = s_h.get(level) or s_normal
                doc.add_paragraph(text, style=st)
            i += 1; continue
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            img = (base / m.group(1)).resolve()
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if img.exists():
                p.add_run().add_picture(str(img), width=Inches(6.3))
            else:
                p.add_run(f"[missing image {img}]")
            i += 1; continue
        if line.startswith("- "):
            while i < len(lines) and lines[i].startswith("- "):
                p = doc.add_paragraph(style=s_bullet or s_normal)
                add_runs(p, lines[i][2:].strip()); i += 1
            continue
        # paragraph, possibly spanning several lines until a blank
        buf = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "- ", "![")):
            buf.append(lines[i].strip()); i += 1
        text = " ".join(buf)
        is_caption = re.match(r"^(Figure|Table) \d+\.", text) is not None
        p = doc.add_paragraph(style=(s_caption if (is_caption and s_caption) else s_normal))
        add_runs(p, text)
    doc.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("template"); ap.add_argument("out")
    ap.add_argument("--md", default="results/paper/submission.md")
    ap.add_argument("--keep-body", action="store_true")
    a = ap.parse_args()
    print(fill(a.template, a.out, a.md, a.keep_body))
