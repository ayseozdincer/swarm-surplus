"""Fill the Apart Research hackathon submission template from a markdown draft that may use numbered lists, LaTeX display and
inline math, markdown links and a bold author line. A generalisation of scripts/paper/fill_apart.py, same template handling.

Usage. python scripts/paper/fill_apart_md.py TEMPLATE.docx OUT.docx --md DRAFT.md

Markdown handled. "# title" (first line), an optional "### subtitle" right after it, an author line that is either
"AUTHORS: A. B." or a bold "**A and B**" line, "## Abstract" up to the next "##", then ## and ### headings, paragraphs with
**bold**, *italic* and `code`, "- " bullets, "1. " numbered items (number kept as text, hanging indent), | tables |, ![img](path),
display math between "\\[" and "\\]" lines (rendered as a centred italic line of plain text), inline "\\( ... \\)" (rendered inline),
and [text](url) links (citation-shaped text becomes "(text)", references keep "text, url", everything else keeps the text).
"""
import argparse, copy, re
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.table import Table
from docx.text.paragraph import Paragraph

FONT = "Old Standard TT"
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
INLINE_MATH = re.compile(r"\\\((.+?)\\\)")

SUBS = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "t": "ₜ", "i": "ᵢ", "j": "ⱼ", "n": "ₙ", "a": "ₐ", "e": "ₑ", "o": "ₒ", "x": "ₓ", "s": "ₛ", "l": "ₗ", "k": "ₖ", "m": "ₘ", "p": "ₚ"}


def tex_to_text(t: str) -> str:
    """Small LaTeX to plain-text mapping for the formulas a methods section uses. Order matters, subscripts before fractions."""
    t = t.strip()
    t = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", t)
    t = t.replace("\\ell", "ℓ").replace("\\in", "∈").replace("\\delta", "δ")
    def sub(m):
        inner = m.group(1).replace("\\", "")
        if "∈" in inner:
            return "_(" + inner.replace(" ", "") + ")"
        if inner.isalpha() and len(inner) > 1:
            return "_" + inner
        return "".join(SUBS.get(c, c) for c in inner)
    t = re.sub(r"_\{([^{}]*)\}", sub, t)
    t = re.sub(r"_([A-Za-z0-9])(?![A-Za-z0-9])", lambda m: SUBS.get(m.group(1), "_" + m.group(1)), t)
    t = re.sub(r"\^\{-P\}", "⁽⁻ᴾ⁾", t)
    t = re.sub(r"\\\|([^|]+?)\\\|₂", r"‖\1‖₂", t)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", lambda m: "(" + m.group(1) + ")/(" + m.group(2) + ")" if (" " in m.group(1) + m.group(2) or "|" in m.group(2)) else m.group(1) + "/" + m.group(2), t)
    t = t.replace("\\geq", " ≥ ").replace("\\leq", " ≤ ").replace("\\iff", " ⟺ ").replace("\\sum", "Σ")
    t = t.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ").replace("\\quad", "    ").replace("\\qquad", "      ")
    t = t.replace("{", "").replace("}", "").replace("\\", "")
    t = re.sub(r"\s+", " ", t).replace(" ,", ",")
    return t


def set_font(run, size=None, bold=None, italic=None, font=FONT):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def resolve_links(text: str, in_refs: bool) -> str:
    def rep(m):
        label, url = m.group(1), m.group(2)
        if in_refs:
            return f"{label}, {url}"
        if re.search(r"\b(19|20)\d\d[a-z]?$", label.strip()):
            return f"({label})"
        return label
    return LINK.sub(rep, text)


def add_runs(par, text, size=None, italic=None):
    text = INLINE_MATH.sub(lambda m: tex_to_text(m.group(1)), text)
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2]); set_font(r, size, bold=True, italic=italic)
        elif part.startswith("*") and part.endswith("*"):
            r = par.add_run(part[1:-1]); set_font(r, size, italic=True)
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); set_font(r, size or 10, font="Courier New")
        else:
            r = par.add_run(part); set_font(r, size, italic=italic)


def borders(table):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:space"), "0"); el.set(qn("w:color"), "999999")
        b.append(el)
    tblPr.append(b)


def add_table(doc, rows, in_refs=False):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, data = cells[0], [r for r in cells[1:] if not all(set(c) <= set("-: ") for c in r)]
    t = doc.add_table(rows=1 + len(data), cols=len(header))
    borders(t)
    for j, h in enumerate(header):
        p = t.rows[0].cells[j].paragraphs[0]; add_runs(p, resolve_links(h, in_refs), size=8.5)
        for r in p.runs: r.bold = True
    for i, row in enumerate(data, start=1):
        for j in range(len(header)):
            p = t.rows[i].cells[j].paragraphs[0]
            add_runs(p, resolve_links(row[j] if j < len(row) else "", in_refs), size=8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template"); ap.add_argument("out"); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    doc = Document(a.template)
    body = doc.element.body
    blocks = list(body.iterchildren())
    outer = Table(blocks[1], doc)

    md = Path(a.md).read_text(encoding="utf-8").splitlines()
    title_i = next(i for i, l in enumerate(md) if l.startswith("# "))
    title = md[title_i][2:].strip()
    subtitle = None
    for l in md[title_i + 1:title_i + 4]:
        if l.startswith("### "):
            subtitle = l[4:].strip(); break
    authors = []
    for l in md[:md.index("## Abstract")]:
        if l.startswith("AUTHORS:"):
            for chunk in l[len("AUTHORS:"):].split(". "):
                chunk = chunk.strip().rstrip(".")
                if chunk:
                    parts = [x.strip() for x in chunk.split(",")]
                    authors.append((parts[0], ", ".join(parts[1:])))
            break
        m = re.match(r"^\*\*(.+)\*\*\s*$", l.strip())
        if m and " and " in m.group(1):
            authors = [(n.strip().rstrip(","), "") for n in re.split(r",\s*|\s+and\s+", m.group(1)) if n.strip()]
            break
    ai = md.index("## Abstract")
    aj = next(i for i in range(ai + 1, len(md)) if md[i].startswith("## "))
    abstract = " ".join(l.strip() for l in md[ai + 1:aj] if l.strip())
    abstract = resolve_links(abstract, False)

    # title and subtitle
    title_par = next(p for p in outer.rows[0].cells[0].paragraphs if p.style.name == "Title")
    runs = title_par.runs
    fmt = copy.deepcopy(runs[0]._element.find(qn("w:rPr"))) if runs else None
    for r in runs:
        r._element.getparent().remove(r._element)
    r = title_par.add_run(title)
    if fmt is not None:
        r._element.insert(0, fmt)
    if subtitle:
        new_p = OxmlElement("w:p"); title_par._element.addnext(new_p)
        sp = Paragraph(new_p, title_par._parent); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sr = sp.add_run(subtitle); set_font(sr, 13, italic=True)
        sp.paragraph_format.space_after = Pt(6)

    # authors, centred lines before the abstract table, as fill_apart.py does
    second = outer.rows[1].cells[0]
    author_tbl = second.tables[0]._tbl
    abstract_tbl = second.tables[1]._tbl
    author_tbl.getparent().remove(author_tbl)

    def insert_par_before(anchor, text, size=11, bold=False, italic=False, space_after=0):
        new_p = OxmlElement("w:p"); anchor.addprevious(new_p)
        par = Paragraph(new_p, second); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(space_after)
        rr = par.add_run(text); set_font(rr, size, bold=bold, italic=italic)
        return par
    label = next(par._element for par in second.paragraphs if par.text.strip() == "Abstract")
    for name, aff in authors:
        insert_par_before(label, name, size=12, bold=True, space_after=(8 if not aff else 0))
        if aff:
            insert_par_before(label, aff, size=11, space_after=8)
    insert_par_before(label, "With", size=11, bold=True)
    insert_par_before(label, "Apart Research", size=11, space_after=14)

    abs_cell = Table(abstract_tbl, second).rows[0].cells[0]
    p0 = abs_cell.paragraphs[0]
    for extra in abs_cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)
    for rr in p0.runs[1:]:
        rr._element.getparent().remove(rr._element)
    if p0.runs:
        p0.runs[0].text = abstract; p0.runs[0].italic = False
    else:
        add_runs(p0, abstract)
    p0.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    bullet_ppr = None
    for child in blocks[4:]:
        if child.tag == qn("w:p"):
            ppr = child.find(qn("w:pPr"))
            if ppr is not None and ppr.find(qn("w:numPr")) is not None:
                bullet_ppr = copy.deepcopy(ppr); break
    for child in blocks[3:]:
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)
    sect = body.find(qn("w:sectPr"))

    def add_par(style=None):
        return doc.add_paragraph(style=style) if style else doc.add_paragraph()

    def heading(text, level):
        p = add_par("Heading 2" if level == 2 else "Heading 3")
        rr = p.add_run(text); set_font(rr, 14 if level == 2 else 13, bold=True)
        rpr = rr._element.get_or_add_rPr(); col = OxmlElement("w:color"); col.set(qn("w:val"), "434343"); rpr.append(col)
        p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.5

    base = Path(a.md).parent
    in_refs = False
    i = aj
    n_img = n_math = n_num = 0
    while i < len(md):
        line = md[i]
        if not line.strip() or line.startswith("AUTHORS:") or line.startswith("<!--"):
            i += 1; continue
        if line.startswith("|"):
            block = []
            while i < len(md) and md[i].startswith("|"):
                block.append(md[i]); i += 1
            add_table(doc, block, in_refs); continue
        m = re.match(r"^(#{2,3})\s+(.*)$", line)
        if m:
            text = m.group(2).strip(); heading(text, len(m.group(1)))
            in_refs = text.lower().startswith("references")
            i += 1; continue
        if line.startswith("# "):
            i += 1; continue
        if line.strip() == "\\[":
            buf = []; i += 1
            while i < len(md) and md[i].strip() != "\\]":
                buf.append(md[i].strip()); i += 1
            i += 1
            p = add_par(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = p.add_run(tex_to_text(" ".join(buf))); set_font(rr, 11, italic=True, font="Cambria Math")
            p.paragraph_format.space_after = Pt(6); n_math += 1
            continue
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            img = Path(m.group(1))
            img = img if img.is_absolute() else (base / img).resolve()
            p = add_par(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(img), width=Inches(6.2)); n_img += 1
            i += 1; continue
        if line.startswith("- "):
            while i < len(md) and md[i].startswith("- "):
                p = add_par()
                if bullet_ppr is not None:
                    old = p._element.find(qn("w:pPr"))
                    if old is not None:
                        p._element.remove(old)
                    p._element.insert(0, copy.deepcopy(bullet_ppr))
                add_runs(p, resolve_links(md[i][2:].strip(), in_refs)); p.paragraph_format.space_after = Pt(4)
                i += 1
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if m:
            buf = [line.strip()]; i += 1
            while i < len(md) and md[i].strip() and not re.match(r"^(\d+)\.\s+", md[i]) and not md[i].startswith(("|", "#", "- ", "![", "\\[")):
                buf.append(md[i].strip()); i += 1
            p = add_par(); p.paragraph_format.left_indent = Inches(0.3); p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.space_after = Pt(4)
            add_runs(p, resolve_links(" ".join(buf), in_refs), size=(9.5 if in_refs else None)); n_num += 1
            continue
        buf = [line.strip()]; i += 1
        while i < len(md) and md[i].strip() and not md[i].startswith(("|", "#", "- ", "![", "AUTHORS:", "\\[")) and not re.match(r"^\d+\.\s+", md[i]):
            buf.append(md[i].strip()); i += 1
        text = resolve_links(" ".join(buf), in_refs)
        is_caption = re.match(r"^\**(Figure|Table) [A-Z]?\d+\.", text) is not None
        p = add_par()
        add_runs(p, text, size=(9.5 if is_caption else None), italic=(True if is_caption else None))
        p.paragraph_format.space_after = Pt(6); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    body.remove(sect); body.append(sect)
    doc.save(a.out)
    print(a.out, "images", n_img, "math blocks", n_math, "numbered items", n_num, "authors", [n for n, _ in authors], "subtitle", bool(subtitle))


if __name__ == "__main__":
    main()
