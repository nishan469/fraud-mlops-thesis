"""Build the P1 as a PDF (reportlab), following the layout of the BRAC University CSE thesis
template: unnumbered title page, roman-numbered front matter, 'Chapter N' headings, numbered
figures and tables, IEEE-numbered bibliography.

  python build_pdf.py                ->  P1_Drift_Aware_Continuous_Learning_MLOps.pdf (ch. 1-2)
  python build_pdf.py --p2           ->  P2_Methodology_and_Design.pdf (Chapter 4)
  python build_pdf.py --chapters 4   ->  Thesis_Draft_Ch1-4.pdf (full draft so far)
  python build_pdf.py --report       ->  Thesis_Report.pdf (full report, BRAC template structure)
  python build_pdf.py --report-empty ->  Thesis_Report_Ch1-3_Empty.pdf (same, Chapters 1-3 empty)
"""

import os
import re
import sys

N_CHAPTERS = int(sys.argv[sys.argv.index("--chapters") + 1]) if "--chapters" in sys.argv else 2
MODE = ("report_empty" if "--report-empty" in sys.argv
        else "report" if "--report" in sys.argv else "p2" if "--p2" in sys.argv
        else "draft" if "--chapters" in sys.argv else "p1")
os.environ["THESIS_CHAPTERS"] = str(N_CHAPTERS)      # read by content.py
os.environ["THESIS_MODE"] = MODE

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

import content as C
from markup import numbering, resolve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, {"p1": "P1_Drift_Aware_Continuous_Learning_MLOps.pdf",
                         "p2": "P2_Methodology_and_Design.pdf",
                         "report": "Thesis_Report.pdf",
                         "report_empty": "Thesis_Report_Ch1-3_Empty.pdf",
                         "draft": f"Thesis_Draft_Ch1-{N_CHAPTERS}.pdf"}[MODE])
MARGIN = 1.0 * inch
TEXT_W = A4[0] - 2 * MARGIN
CITES, FIGS, TABS = numbering()

# ------------------------------------------------------------------------------ styles
BASE = "Times-Roman"
BOLD = "Times-Bold"
S = {
    "body": ParagraphStyle("body", fontName=BASE, fontSize=12, leading=17, alignment=TA_JUSTIFY,
                           spaceAfter=9),
    "item": ParagraphStyle("item", fontName=BASE, fontSize=12, leading=17, alignment=TA_JUSTIFY,
                           leftIndent=26, firstLineIndent=0, spaceAfter=5),
    "chapnum": ParagraphStyle("chapnum", fontName=BOLD, fontSize=20, leading=24, spaceAfter=14),
    "chaptitle": ParagraphStyle("chaptitle", fontName=BOLD, fontSize=25, leading=30,
                                spaceAfter=26),
    "section": ParagraphStyle("section", fontName=BOLD, fontSize=14.5, leading=19,
                              spaceBefore=12, spaceAfter=8, keepWithNext=1),
    "subsection": ParagraphStyle("subsection", fontName=BOLD, fontSize=12.5, leading=17,
                                 spaceBefore=8, spaceAfter=6, keepWithNext=1),
    "caption": ParagraphStyle("caption", fontName=BASE, fontSize=11, leading=14,
                              alignment=TA_CENTER, spaceBefore=6, spaceAfter=14),
    "tcaption": ParagraphStyle("tcaption", fontName=BASE, fontSize=11, leading=14,
                               alignment=TA_CENTER, spaceBefore=6, spaceAfter=8, keepWithNext=1),
    "cell": ParagraphStyle("cell", fontName=BASE, fontSize=9, leading=11, alignment=TA_LEFT),
    "cellh": ParagraphStyle("cellh", fontName=BOLD, fontSize=9, leading=11, alignment=TA_LEFT),
    "ref": ParagraphStyle("ref", fontName=BASE, fontSize=11, leading=14.5, leftIndent=30,
                          firstLineIndent=-30, spaceAfter=7, alignment=TA_LEFT),
    "eq": ParagraphStyle("eq", fontName=BASE, fontSize=12.5, leading=18, alignment=TA_CENTER),
    "eqnum": ParagraphStyle("eqnum", fontName=BASE, fontSize=12, leading=18,
                            alignment=TA_RIGHT),
    "center": ParagraphStyle("center", fontName=BASE, fontSize=12, leading=17,
                             alignment=TA_CENTER),
    "left": ParagraphStyle("left", fontName=BASE, fontSize=12, leading=17, alignment=TA_LEFT),
}


def esc(text):
    """Escape & for reportlab's XML while keeping <b>/<i> markup."""
    return re.sub(r"&(?!amp;|lt;|gt;)", "&amp;", text)


def cite_fmt(keys):
    return ", ".join(f"[{CITES[k]}]" for k in keys)


def ref_fmt(kind, key, label):
    return label


def rich(text):
    return esc(resolve(text, CITES, FIGS, TABS, cite_fmt, ref_fmt))


def roman(n):
    out = ""
    for v, s in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out += s
            n -= v
    return out


# ------------------------------------------------------------------------------ document
class MainStart(Flowable):
    """Zero-size marker: arabic page numbering starts on the page where it is drawn."""
    def wrap(self, *args):
        return 0, 0

    def draw(self):
        pass


class Entry(Paragraph):
    """A paragraph that also registers a contents entry (TOC, figures or tables)."""
    def __init__(self, text, style, kind="TOCEntry", level=0, entry=None):
        super().__init__(text, style)
        self.entry = (kind, level, entry if entry is not None else re.sub(r"<[^>]+>", "", text))


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN, title=C.TITLE,
                         author=", ".join(a for a, _ in C.AUTHORS))
        frame = Frame(MARGIN, MARGIN, TEXT_W, A4[1] - 2 * MARGIN, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPageEnd=self.footer)])
        self.main_start = None

    def label(self, page):
        if self.main_start is not None and page >= self.main_start:
            return str(page - self.main_start + 1)
        return roman(page - 1) if page > 1 else ""

    def footer(self, canvas, doc):
        text = self.label(doc.page)
        if text:
            canvas.setFont(BASE, 11)
            canvas.drawCentredString(A4[0] / 2, 0.6 * inch, text)

    def afterFlowable(self, f):
        if isinstance(f, MainStart):
            # front-matter listings are drawn before this point, so they use the value
            # from the previous layout pass; later pages use the current one
            self.main_start = self.page
        entry = getattr(f, "entry", None)
        if entry:
            kind, level, text = entry
            self.notify(kind, (level, text, self.page))   # listings format it via label()

class Listing(TableOfContents):
    """TableOfContents that listens to its own entry kind (figures, tables)."""
    def __init__(self, kind, **kw):
        super().__init__(**kw)
        self.kind = kind

    def notify(self, kind, stuff):
        if kind == self.kind:
            self.addEntry(*stuff)


def toc_styles(size=12):
    return [ParagraphStyle("t0", fontName=BOLD, fontSize=size, leading=size + 6, leftIndent=0,
                           firstLineIndent=0, spaceBefore=5),
            ParagraphStyle("t1", fontName=BASE, fontSize=size, leading=size + 5, leftIndent=22,
                           firstLineIndent=0),
            ParagraphStyle("t2", fontName=BASE, fontSize=size - 1, leading=size + 4,
                           leftIndent=50, firstLineIndent=0)]


def front_heading(title, story):
    story.append(Entry(title, S["chaptitle"], entry=title))


# ------------------------------------------------------------------------------ pages
def title_page(story):
    story.append(Spacer(1, 40))
    story.append(Paragraph(C.TITLE, ParagraphStyle("title", fontName=BOLD, fontSize=21,
                                                   leading=28, alignment=TA_CENTER)))
    story.append(Spacer(1, 26))
    story.append(Paragraph("by", S["center"]))
    story.append(Spacer(1, 14))
    for name, sid in C.AUTHORS:
        story.append(Paragraph(f"{name}<br/>{sid}", S["center"]))
        story.append(Spacer(1, 8))
    story.append(Spacer(1, 22))
    story.append(Paragraph(
        f"A thesis submitted to the {C.META['department']}<br/>in partial fulfillment of the "
        f"requirements for the degree of<br/>{C.META['degree']}", S["center"]))
    story.append(Spacer(1, 22))
    story.append(Paragraph(f"{C.META['department']}<br/>{C.META['university']}<br/>"
                           f"{C.META['date']}", S["center"]))
    story.append(Spacer(1, 50))
    story.append(Paragraph(f"&#169; {C.META['copyright_year']}. {C.META['university']}<br/>"
                           "All rights reserved.", S["center"]))
    story.append(PageBreak())


def declaration(story):
    front_heading("Declaration", story)
    story.append(Paragraph("It is hereby declared that", S["body"]))
    for i, t in enumerate([
            f"The thesis submitted is our own original work while completing degree at "
            f"{C.META['university']}.",
            "The thesis does not contain material previously published or written by a third "
            "party, except where this is appropriately cited through full and accurate "
            "referencing.",
            "The thesis does not contain material which has been accepted, or submitted, for "
            "any other degree or diploma at a university or other institution.",
            "We have acknowledged all main sources of help."], start=1):
        story.append(Paragraph(f"{i}. {t}", S["item"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph("<b>Student's Full Name &amp; Signature:</b>", S["left"]))
    story.append(Spacer(1, 10))
    cells = [[Paragraph(f"_______________________<br/>{n}<br/>{i}", S["center"])
              for n, i in C.AUTHORS[k:k + 2]] for k in range(0, len(C.AUTHORS), 2)]
    cells[-1] += [""] * (2 - len(cells[-1]))
    t = Table(cells, colWidths=[TEXT_W / 2] * 2)
    t.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 16)]))
    story += [t, PageBreak()]


def approval(story):
    front_heading("Approval", story)
    names = "<br/>".join(f"{i}. {n} ({sid})" for i, (n, sid) in enumerate(C.AUTHORS, start=1))
    story.append(Paragraph(f"The thesis titled “{C.TITLE}” submitted by", S["body"]))
    story.append(Paragraph(names, S["item"]))
    story.append(Paragraph(
        f"of {C.META['semester']} has been accepted as satisfactory in partial fulfillment of "
        f"the requirement for the degree of {C.META['degree']} in {C.META['degree_term']}.",
        S["body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Examining Committee:</b>", S["left"]))
    for role, member, (name, title) in [
            ("Supervisor:", "(Member)", C.META["supervisor"]),
            ("Co-Supervisor:", "(Member)", C.META["cosupervisor"]),
            ("Thesis Coordinator:", "(Member)", C.META["coordinator"]),
            ("Head of Department:", "(Chair)", C.META["head"])]:
        block = Table([[Paragraph(f"<b>{role}</b><br/>{member}", S["left"]),
                        Paragraph(f"_______________________________<br/>{name}<br/>{title}<br/>"
                                  f"{C.META['department']}<br/>{C.META['university']}",
                                  S["left"])]], colWidths=[1.7 * inch, TEXT_W - 1.7 * inch])
        block.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 14)]))
        story.append(KeepTogether(block))
    story.append(PageBreak())


def abstract_ack(story):
    front_heading("Abstract", story)
    for p in C.ABSTRACT:
        story.append(Paragraph(esc(p), S["body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Keywords:</b> {esc(C.KEYWORDS)}", S["body"]))
    story.append(PageBreak())
    front_heading("Acknowledgement", story)
    for p in C.ACKNOWLEDGEMENT:
        story.append(Paragraph(esc(p), S["body"]))
    story.append(PageBreak())


LISTINGS = []


def listings(story):
    story.append(Paragraph("Table of Contents", S["chaptitle"]))
    toc = TableOfContents(levelStyles=toc_styles(), dotsMinLevel=0)
    lof = Listing("LOFEntry", levelStyles=[toc_styles()[1].clone("f", leftIndent=0)],
                  dotsMinLevel=0)
    lot = Listing("LOTEntry", levelStyles=[toc_styles()[1].clone("t", leftIndent=0)],
                  dotsMinLevel=0)
    LISTINGS.extend([toc, lof, lot])
    story += [toc, PageBreak()]
    front_heading("List of Figures", story)
    story += [lof, PageBreak()]
    front_heading("List of Tables", story)
    story += [lot, PageBreak()]
    front_heading("Nomenclature", story)
    story.append(Paragraph("The next list describes several symbols and abbreviations that "
                           "will be used later within the body of the document.", S["body"]))
    rows = [[Paragraph(f"<b>{a}</b>", S["left"]), Paragraph(esc(d), S["left"])]
            for a, d in C.ABBREVIATIONS]
    t = Table(rows, colWidths=[1.3 * inch, TEXT_W - 1.3 * inch])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2)]))
    story += [t, PageBreak()]


def figure(story, key):
    path, caption = C.FIGURES[key]
    img = Image(os.path.join(HERE, path))
    scale = TEXT_W / img.imageWidth
    img.drawWidth, img.drawHeight = TEXT_W, img.imageHeight * scale
    label = FIGS[key]
    story.append(KeepTogether([
        Spacer(1, 4), img,
        Entry(f"Figure {label}: {rich(caption)}", S["caption"], kind="LOFEntry",
              entry=f"{label}&nbsp;&nbsp;&nbsp;{rich(caption)}")]))


def table(story, key):
    spec = C.TABLES[key]
    label = TABS[key]
    rows = [[Paragraph(esc(h), S["cellh"]) for h in spec["header"]]]
    rows += [[Paragraph(rich(c), S["cell"]) for c in r] for r in spec["rows"]]
    t = Table(rows, colWidths=[w * TEXT_W for w in spec["widths"]], repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#BBBBBB")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(Entry(f"Table {label}: {esc(spec['caption'])}", S["tcaption"], kind="LOTEntry",
                       entry=f"{label}&nbsp;&nbsp;&nbsp;{esc(spec['caption'])}"))
    story.append(t)
    story.append(Spacer(1, 10))


def equation(story, key):
    html = C.EQUATIONS[key][1]
    num = TABS["eq:" + key]
    t = Table([[Paragraph(html, S["eq"]), Paragraph(f"({num})", S["eqnum"])]],
              colWidths=[TEXT_W - 0.8 * inch, 0.8 * inch])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (0, 0), 0.8 * inch)]))
    story += [Spacer(1, 2), t, Spacer(1, 8)]


def blocks(story, num, ch):
    for s, sec in enumerate(ch["sections"], start=1):
        story.append(Entry(f"{num}.{s}&nbsp;&nbsp;&nbsp;{sec['title']}", S["section"], level=1,
                           entry=f"{num}.{s}&nbsp;&nbsp;{sec['title']}"))
        nsub = 0
        for block in sec["blocks"]:
            kind = block[0]
            if kind == "sub":
                nsub += 1
                label = f"{num}.{s}.{nsub}"
                story.append(Entry(f"{label}&nbsp;&nbsp;&nbsp;{block[1]}", S["subsection"], level=2,
                                   entry=f"{label}&nbsp;&nbsp;{block[1]}"))
            elif kind == "p":
                story.append(Paragraph(rich(block[1]), S["body"]))
            elif kind == "list":
                items, numbered = block[1], block[2]
                for i, item in enumerate(items, start=1):
                    prefix = f"{i}.&nbsp;&nbsp;" if numbered else ""
                    story.append(Paragraph(prefix + rich(item), S["item"]))
                story.append(Spacer(1, 4))
            elif kind == "fig":
                figure(story, block[1])
            elif kind == "table":
                table(story, block[1])
            elif kind == "eq":
                equation(story, block[1])


def chapter(story, num, ch, page_break=True):
    """Chapter 1 starts on the page after the front matter (no extra page break)."""
    if page_break:
        story.append(PageBreak())
    story.append(Paragraph(f"Chapter {num}", S["chapnum"]))
    story.append(Entry(ch["title"], S["chaptitle"], entry=f"{num}&nbsp;&nbsp;{ch['title']}"))
    blocks(story, num, ch)


def bibliography(story):
    story.append(PageBreak())
    story.append(Entry("Bibliography", S["chaptitle"], entry="Bibliography"))
    for key, n in sorted(CITES.items(), key=lambda kv: kv[1]):
        story.append(Paragraph(f"[{n}]&nbsp;&nbsp;{esc(C.REFERENCES[key])}", S["ref"]))


def main():
    story = []
    title_page(story)
    declaration(story)
    approval(story)
    abstract_ack(story)
    listings(story)
    story.append(MainStart())
    for i, ch in enumerate(C.CHAPTERS, start=C.FIRST_CHAPTER):
        chapter(story, i, ch, page_break=i > C.FIRST_CHAPTER)
    bibliography(story)
    doc = Doc(OUT)
    for listing in LISTINGS:
        listing.formatter = doc.label     # absolute page -> roman/arabic label
    doc.multiBuild(story)
    print(f"Wrote {OUT} ({doc.page} pages)")


if __name__ == "__main__":
    main()
