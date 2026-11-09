"""Fill the Apart Research hackathon submission template from results/paper/submission_template.md.

Usage. python scripts/paper/fill_apart.py TEMPLATE.docx OUT.docx [--md results/paper/submission_template.md]

Template layout (as inspected). Block 1 is an outer table whose first cell holds the Title paragraph and whose second cell
holds the word Abstract, a nested author grid (2 x 3 cells, each "Author name N" + line break + "Affiliation") and a nested
one-cell table with the abstract text. Block 3 is a yellow instructions table. Everything after it is guidance text.
We keep the outer table, fill title, authors and abstract, delete the yellow table and the guidance, then append the
markdown with the template's styles (Heading 2 for ## sections, Heading 3 for ###, normal body in Old Standard TT), bullets
copied from the template's own list paragraph, bordered tables at 8.5 pt, captions in italics, and the figure at 6.3 inches.
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


def set_font(run, size=None, bold=None, italic=None):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_runs(par, text, size=None, italic=None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2]); set_font(r, size, bold=True, italic=italic)
        elif part.startswith("*") and part.endswith("*"):
            r = par.add_run(part[1:-1]); set_font(r, size, italic=True)
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Courier New"; r.font.size = Pt(size or 10)
        else:
            r = par.add_run(part); set_font(r, size, italic=italic)


def replace_text_keep_format(par, text, breaks=None):
    """Keep the first run's formatting, set its text, drop other runs. breaks splits text on \\n into line breaks."""
    runs = par.runs
    if not runs:
        r = par.add_run(); runs = [r]
    first = runs[0]
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    first.text = ""
    parts = text.split("\n")
    first.text = parts[0]
    for extra in parts[1:]:
        first.add_break()
        first.add_text(extra) if hasattr(first, "add_text") else None
    if len(parts) > 1 and not hasattr(first, "add_text"):
        # python-docx Run has add_break but add_text lives on the element; append a new text node
        pass


def set_cell_lines(cell, lines):
    """Replace the cell's single paragraph content with lines joined by line breaks, keeping run formatting."""
    p = cell.paragraphs[0]
    for extra in cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)
    runs = p.runs
    if runs:
        fmt = copy.deepcopy(runs[0]._element.find(qn("w:rPr")))
        for r in runs:
            r._element.getparent().remove(r._element)
    else:
        fmt = None
    for i, line in enumerate(lines):
        r = p.add_run(line)
        if fmt is not None:
            r._element.insert(0, copy.deepcopy(fmt))
        if i < len(lines) - 1:
            r.add_break()


def borders(table):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "808080")
        b.append(e)
    tblPr.append(b)


def add_table(doc, rows):
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    data = []
    for r in rows[2:]:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        cells += [""] * (len(header) - len(cells))
        data.append(cells[: len(header)])
    t = doc.add_table(rows=1 + len(data), cols=len(header))
    borders(t)
    t.autofit = True
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]; p = cell.paragraphs[0]
        r = p.add_run(h); set_font(r, 8.5, bold=True)
    for i, row in enumerate(data, start=1):
        for j, v in enumerate(row):
            p = t.rows[i].cells[j].paragraphs[0]
            add_runs(p, v, size=8.5)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
    spacer = doc.add_paragraph(); spacer.paragraph_format.space_after = Pt(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template"); ap.add_argument("out")
    ap.add_argument("--md", default="results/paper/submission_template.md")
    a = ap.parse_args()
    doc = Document(a.template)
    body = doc.element.body
    blocks = list(body.iterchildren())
    outer = Table(blocks[1], doc)

    md = Path(a.md).read_text().splitlines()
    title = next(l[2:].strip() for l in md if l.startswith("# "))
    authors_line = next(l for l in md if l.startswith("AUTHORS:"))[len("AUTHORS:"):].strip()
    authors = []
    for chunk in authors_line.split(". "):
        chunk = chunk.strip().rstrip(".")
        if not chunk:
            continue
        parts = [x.strip() for x in chunk.split(",")]
        authors.append((parts[0], ", ".join(parts[1:]) if len(parts) > 1 else ""))
    ai = md.index("## Abstract")
    aj = next(i for i in range(ai + 1, len(md)) if md[i].startswith("## "))
    abstract = " ".join(l.strip() for l in md[ai + 1:aj] if l.strip())

    # title
    title_par = next(p for p in outer.rows[0].cells[0].paragraphs if p.style.name == "Title")
    runs = title_par.runs
    fmt = copy.deepcopy(runs[0]._element.find(qn("w:rPr"))) if runs else None
    for r in runs:
        r._element.getparent().remove(r._element)
    r = title_par.add_run(title)
    if fmt is not None:
        r._element.insert(0, fmt)
    # authors: remove the nested author table (narrow columns) and write centred lines before the abstract table
    second = outer.rows[1].cells[0]
    author_tbl = second.tables[0]._tbl
    abstract_tbl = second.tables[1]._tbl
    author_tbl.getparent().remove(author_tbl)
    # the cell's own paragraphs: keep the first (blank) one, then insert author lines before the abstract table
    def insert_par_before(anchor, text, size=11, bold=False, italic=False, space_after=0):
        new_p = OxmlElement("w:p")
        anchor.addprevious(new_p)
        par = Paragraph(new_p, second)
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(space_after)
        r = par.add_run(text); set_font(r, size, bold=bold, italic=italic)
        return par
    label = next(par._element for par in second.paragraphs if par.text.strip() == "Abstract")
    for name, aff in authors:
        insert_par_before(label, name, size=12, bold=True, space_after=(8 if not aff else 0))
        if aff:
            insert_par_before(label, aff, size=11, space_after=8)
    insert_par_before(label, "With", size=11, bold=True)
    insert_par_before(label, "Apart Research", size=11, space_after=14)
    # remove the template's 'Abstract' label duplicates? keep the existing label paragraph, which sits before the tables
    abs_cell = Table(abstract_tbl, second).rows[0].cells[0]
    set_cell_lines(abs_cell, [abstract])
    for r in abs_cell.paragraphs[0].runs:
        r.italic = False
    abs_cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # capture the bullet paragraph properties before deleting guidance
    bullet_ppr = None
    for child in blocks[4:]:
        if child.tag == qn("w:p"):
            ppr = child.find(qn("w:pPr"))
            if ppr is not None and ppr.find(qn("w:numPr")) is not None:
                bullet_ppr = copy.deepcopy(ppr); break
    # delete yellow box and all guidance paragraphs (keep sectPr)
    for child in blocks[3:]:
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)

    sect = body.find(qn("w:sectPr"))

    def add_par(style=None):
        p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
        return p

    base = Path(a.md).parent
    i = aj  # start after the abstract section
    while i < len(md):
        line = md[i]
        if not line.strip() or line.startswith("AUTHORS:"):
            i += 1; continue
        if line.startswith("|"):
            block = []
            while i < len(md) and md[i].startswith("|"):
                block.append(md[i]); i += 1
            add_table(doc, block); continue
        m = re.match(r"^(#{2,3})\s+(.*)$", line)
        if m:
            level = len(m.group(1)); text = m.group(2).strip()
            p = add_par("Heading 2" if level == 2 else "Heading 3")
            r = p.add_run(text); set_font(r, 14 if level == 2 else 13, bold=True)
            rpr = r._element.get_or_add_rPr(); col = OxmlElement("w:color"); col.set(qn("w:val"), "434343"); rpr.append(col)
            p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.5
            i += 1; continue
        if line.startswith("# "):
            i += 1; continue
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            img = (base / m.group(1)).resolve()
            p = add_par(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(img), width=Inches(6.2))
            i += 1; continue
        if line.startswith("- "):
            while i < len(md) and md[i].startswith("- "):
                p = add_par()
                if bullet_ppr is not None:
                    old = p._element.find(qn("w:pPr"))
                    if old is not None:
                        p._element.remove(old)
                    p._element.insert(0, copy.deepcopy(bullet_ppr))
                add_runs(p, md[i][2:].strip())
                p.paragraph_format.space_after = Pt(4)
                i += 1
            continue
        buf = [line.strip()]; i += 1
        while i < len(md) and md[i].strip() and not md[i].startswith(("|", "#", "- ", "![", "AUTHORS:")):
            buf.append(md[i].strip()); i += 1
        text = " ".join(buf)
        is_caption = re.match(r"^(Figure|Table) [A-Z]?\d+\.", text) is not None
        p = add_par()
        add_runs(p, text, size=(9.5 if is_caption else None), italic=(True if is_caption else None))
        p.paragraph_format.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # python-docx appends new elements before sectPr already; ensure order
    body.remove(sect); body.append(sect)
    doc.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
