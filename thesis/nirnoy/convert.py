"""Convert Nirnoy's Google-Docs thesis PDF into a LaTeX project (for Overleaf).

Keeps the document's own wording, headings, numbering, tables and captions; replaces every
figure with the original high-resolution image from thesis/p1/figures (identified by image
similarity, or by caption where the Google Docs export lost the image, e.g. Figures 1.1 and
1.2), and writes the equations, which the export fragments, as proper LaTeX.

  python convert.py "path/to/Nirnoy_thesis.pdf"   ->  latex/main.tex, latex/figures/
"""

import os
import re
import shutil
import sys

import numpy as np
import pymupdf
from PIL import Image

from manual import PASSAGES

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_SRC = os.path.normpath(os.path.join(HERE, "..", "p1", "figures"))
OUT = os.path.join(HERE, "latex")

# Figures the export lost (grey placeholder boxes): caption number -> original image
MISSING = {"1.1": "methodology.png", "1.2": "framework_loop.png"}
# Images chosen explicitly, by page (in order of appearance on the page)
PAGE_IMAGE = {16: ["methodology_3ds.png"],   # caption says three datasets: the 3-dataset version
              32: ["protocol.png"],          # cropped evaluation-protocol diagram
              # the document had the two images the wrong way round for their captions
              # (4.4: continuous-learning loop, 4.5: components of the framework)
              25: ["framework_loop.png", "architecture.png"]}

# ------------------------------------------------------------------ corrections to the document
CHAPTER_TITLES = {1: "Introduction"}         # Chapter 1 had no title
CAPTION_FIX = {
    # Figure 4.2 shows the evaluation protocol but repeated the caption of Figure 1.1
    "Figure 4.2": "Evaluation protocol: each dataset is split in time into a history and a stream "
                  "that is replayed week by week; settings are tuned walk-forward in 4-week folds",
    # the text before Table 4.1 introduces the retraining strategies, not the datasets
    "Table 4.1": "Retraining strategies compared in the simulation (default settings)",
}
MISSING_FIG_CAPTION = {   # page -> caption for an image that had none
    46: "Figure 5.3: Equal-budget comparison on IEEE-CIS (top) and BAF (bottom): the best mean "
        "weekly PR-AUC each family reaches with at most a given number of retrains (dots: "
        "individual settings)",
}
MISSING_TABLE_CAPTION = {   # page -> caption for a table that had none
    40: "Table 5.2: Gain in mean weekly PR-AUC over the static model by label delay "
        "(retraining every 14 days; 95\\% intervals)",
}
# Table 4.1 is replaced by the retraining-strategies table its text describes (the datasets
# table it showed is Table 4.8)
_P = r">{\raggedright\arraybackslash}p"
TABLE_OVERRIDE = {
    18: "{\\small\n\\begin{longtable}{|" + _P + "{0.17\\textwidth}|" + _P + "{0.29\\textwidth}|"
        + _P + "{0.21\\textwidth}|" + _P + "{0.17\\textwidth}|}\n\\hline\n"
        "Strategy & When it retrains & Training data & Needs labels to decide? \\\\ \\hline\n"
        "Static & Never & History before the stream & - \\\\ \\hline\n"
        "Periodic expanding & Every 14 days & All labelled data so far & No \\\\ \\hline\n"
        "Periodic sliding & Every 14 days & Labelled data of the last 60 days & No \\\\ \\hline\n"
        "Drift: score PSI & PSI of the model's scores over the last week, against its "
        "validation scores, exceeds 0.1 (at least 14 days apart) & All labelled data so far "
        "& No \\\\ \\hline\n"
        "Drift: performance & PR-AUC on newly matured labels falls more than 15\\% below the "
        "reference (Equation 4.2; at least 14 days apart) & All labelled data so far (or the "
        "last 60 days in tuned variants) & Yes (delayed) \\\\ \\hline\n"
        "\\end{longtable}\n}\n",
}

# Display equations by number (the document's numbering, same as the thesis report)
EQUATIONS = {
    "4.1": r"\mathcal{L}(t) = \{\, x_j : t_j \le t - d \,\}",
    "4.2": r"\mathrm{AP}_{\mathrm{live}}(t) < (1 - \tau)\, \mathrm{AP}_{\mathrm{ref}}",
    "4.3": r"\mathrm{PSI} = \sum_{i=1}^{B} (c_i - r_i)\,\ln\frac{c_i}{r_i}",
    "4.4": r"\mathrm{AP} = \sum_{n} (R_n - R_{n-1})\,P_n",
    "4.5": r"\overline{\mathrm{AP}} = \frac{1}{W} \sum_{w=1}^{W} \mathrm{AP}_w",
    "4.6": r"J(c) = \overline{\mathrm{AP}}(c) - \kappa \cdot "
           r"\frac{N_{\mathrm{retrain}}(c)}{\text{number of 4-week periods}}",
    "4.7": r"\mathrm{AP}_{\mathrm{live}} < (1 - \tau)\,\max\big(\mathrm{AP}_{\mathrm{val}},\ "
           r"\mathrm{median}_{4}(\mathrm{AP}_{\mathrm{live}})\big) \quad\text{or}\quad "
           r"\mathrm{AP}_{\mathrm{live}} < \lambda\,\pi",
    "4.8": r"w^{*} = \arg\max_{w \in \mathcal{W}} \mathrm{AP}_{E}(m_w), \qquad "
           r"\mathcal{W} = \{\text{60 days},\ \text{all history}\}",
    "4.9": r"\mathrm{AP}(\text{challenger}) \ge \mathrm{AP}(\text{champion}) - \delta",
}

CHAR = {"\u2019": "'", "\u2018": "`", "\u201c": "``", "\u201d": "''", "\u2014": "---",
        "\u2013": "--", "\u2212": "-", "\u200b": "", "\u00a0": " ", "\u2264": r"$\le$",
        "\u2265": r"$\ge$", "\u03c4": r"$\tau$", "\u03c0": r"$\pi$", "\u03bb": r"$\lambda$",
        "\u03b4": r"$\delta$", "\u03ba": r"$\kappa$", "\u2022": r"\textbullet{}"}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


def esc(text):
    out = []
    for ch in text:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in CHAR:
            out.append(CHAR[ch])
        elif ord(ch) > 0x1D400:                          # mathematical italic letters
            out.append(pymupdf_unmath(ch))
        else:
            out.append(ch)
    return "".join(out)


def pymupdf_unmath(ch):
    o = ord(ch)
    if 0x1D434 <= o <= 0x1D44D:
        return chr(ord("A") + o - 0x1D434)
    if 0x1D44E <= o <= 0x1D467:
        return chr(ord("a") + o - 0x1D44E)
    return "?"


def span_tex(span):
    t = span["text"]
    if not t.strip():
        return esc(t)
    body = esc(t)
    lead = " " if t[:1] == " " else ""
    trail = " " if t[-1:] == " " else ""
    core = body.strip()
    if "Bold" in span["font"]:
        core = r"\textbf{" + core + "}"
    if "Italic" in span["font"] or "Oblique" in span["font"]:
        core = r"\textit{" + core + "}"
    return lead + core + trail


# ------------------------------------------------------------------------------ images
def fingerprint(img):
    g = img.convert("L").resize((24, 16))
    a = np.asarray(g, dtype=float)
    return (a - a.mean()) / (a.std() + 1e-6)


REFS = {}


def identify(pix_bytes):
    if not REFS:
        for f in os.listdir(FIG_SRC):
            if f.endswith(".png"):
                REFS[f] = fingerprint(Image.open(os.path.join(FIG_SRC, f)))
    import io
    fp = fingerprint(Image.open(io.BytesIO(pix_bytes)))
    best = min(REFS, key=lambda k: float(((REFS[k] - fp) ** 2).mean()))
    return best, float(((REFS[best] - fp) ** 2).mean())


# ------------------------------------------------------------------------------ extraction
def elements(doc):
    """Page elements in reading order: ('text', page, block) / ('img', page, bytes) /
    ('table', page, rows, widths)."""
    out = []
    for pno, page in enumerate(doc):
        tables = []
        for t in page.find_tables().tables:
            rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in t.extract()]
            xs = sorted({round(c[0]) for c in t.cells if c} | {round(t.bbox[2])})
            widths = np.diff(xs) / max(1, (xs[-1] - xs[0]))
            tables.append((pymupdf.Rect(t.bbox), rows, list(widths)))
        items = []
        for b in split_blocks(page.get_text("dict", sort=True)["blocks"]):
            r = pymupdf.Rect(b["bbox"])
            if b["type"] == 1:
                if r.width > 80 and r.height > 40:
                    items.append((r.y0, ("img", pno, b["image"])))
                continue
            if any(tr.contains(r) or (tr & r).get_area() > 0.6 * r.get_area() for tr, _, _ in tables):
                continue
            items.append((r.y0, ("text", pno, b)))
        for tr, rows, widths in tables:
            items.append((tr.y0, ("table", pno, rows, widths)))
        out += [it for _, it in sorted(items, key=lambda x: x[0])]
    return out


def _line_bold(l):
    spans = [s for s in l["spans"] if s["text"].strip()]
    return bool(spans) and all("Bold" in s["font"] for s in spans)


def split_blocks(blocks):
    """The export sometimes puts a whole page in one block. Split text blocks into paragraphs:
    a new paragraph starts after a vertical gap larger than the normal line step, at a switch
    between all-bold and normal lines (headings), and at a line that starts a numbered item,
    a bibliography entry, a figure/table caption or 'Chapter N'."""
    out = []
    starts = re.compile(r"^(\[\d+\]|\d+\.\s|Figure \d+\.\d+:|Table \d+\.\d+:|Chapter \d+\s*$)")
    for b in blocks:
        if b["type"] != 0:
            out.append(b)
            continue
        cur = []
        for l in b["lines"]:
            text = "".join(s["text"] for s in l["spans"]).replace("​", " ").strip()
            if not text:                       # an empty line ends the paragraph
                if cur:
                    out.append(_mk(b, cur))
                    cur = []
                continue
            if cur:
                prev = cur[-1]
                gap = l["bbox"][1] - prev["bbox"][3]
                math = any("Cambria" in s["font"] for s in l["spans"] + prev["spans"])
                new = (not math and gap > 6) or \
                      (text and _line_bold(l) != _line_bold(prev)) or \
                      (text and not math and starts.match(text))
                if new:
                    out.append(_mk(b, cur))
                    cur = []
            cur.append(l)
        if cur:
            out.append(_mk(b, cur))
    return out


def _mk(b, lines):
    x0 = min(l["bbox"][0] for l in lines)
    y0 = min(l["bbox"][1] for l in lines)
    x1 = max(l["bbox"][2] for l in lines)
    y1 = max(l["bbox"][3] for l in lines)
    return {"type": 0, "bbox": (x0, y0, x1, y1), "lines": lines, "number": b.get("number", 0)}


def block_text(b):
    text = " ".join("".join(s["text"] for s in l["spans"]) for l in b["lines"])
    return " ".join(text.replace("\u200b", " ").split())


def block_tex(b):
    lines = []
    for l in b["lines"]:
        lines.append("".join(span_tex(s) for s in l["spans"]).strip())
    return " ".join(x for x in lines if x)


def has_math(b):
    return any("Cambria" in s["font"] for l in b["lines"] for s in l["spans"])


def all_bold(b):
    spans = [s for l in b["lines"] for s in l["spans"] if s["text"].strip()]
    return bool(spans) and all("Bold" in s["font"] for s in spans)


HEAD = re.compile(r"^(\d+(?:\.\d+){1,2})\.?\s+(\S.*)$")
CAP = re.compile(r"^(Figure|Table)\s+(\d+\.\d+)\s*:\s*(.*)$", re.S)


def convert(pdf_path):
    doc = pymupdf.open(pdf_path)
    els = elements(doc)
    body, notes = [], []
    pending_chapter = None
    last_kind = None          # for merging paragraphs across pages
    in_bib = False
    pending_img = None        # image waiting for its caption
    seen_caps = set()
    used_eqs = set()
    skip_until = None
    done_passages = set()

    page_imgs = {}            # images seen so far on each page (for PAGE_IMAGE order)
    pending_img_page = None
    tables_done = set()

    def flush_img(caption=None):
        nonlocal pending_img
        if pending_img is None:
            return
        name = pending_img
        pending_img = None
        if caption is None and pending_img_page in MISSING_FIG_CAPTION:
            caption = MISSING_FIG_CAPTION[pending_img_page]
        cap = f"\n  \\caption*{{{caption}}}" if caption else ""
        body.append("\\begin{figure}[H]\n  \\centering\n"
                    f"  \\includegraphics[width=0.95\\textwidth]{{figures/{name}}}{cap}\n"
                    "\\end{figure}\n")

    def emit_par(tex, merge_ok=True):
        """A paragraph that continues from the previous page (previous paragraph on an earlier
        page and not ending a sentence) is joined to it; otherwise a new paragraph starts."""
        nonlocal last_kind, last_par_page
        joined = (merge_ok and last_kind == "par" and body and body[-1].endswith("\n")
                  and last_par_page is not None and cur_page > last_par_page
                  and not re.search(r"[.:?!)\]]\s*$", body[-1].strip()))
        last_par_page = cur_page
        if joined:
            body[-1] = body[-1].rstrip("\n") + " " + tex + "\n"
        else:
            body.append(tex + "\n")
        last_kind = "par"

    last_par_page = None
    cur_page = 0
    for el in els:
        cur_page = el[1]
        if el[0] == "img":
            flush_img()
            name, dist = identify(el[2])
            k = page_imgs.get(el[1] + 1, 0)
            page_imgs[el[1] + 1] = k + 1
            if el[1] + 1 in PAGE_IMAGE and k < len(PAGE_IMAGE[el[1] + 1]):
                name, dist = PAGE_IMAGE[el[1] + 1][k], 0.0
            if dist > 0.6:
                notes.append(f"image on page {el[1] + 1} matched {name} only loosely ({dist:.2f})")
            pending_img = name
            pending_img_page = el[1] + 1
            last_kind = "img"
            continue
        if el[0] == "table":
            flush_img()
            page = el[1] + 1
            if page in TABLE_OVERRIDE:
                if page not in tables_done:
                    tables_done.add(page)
                    body.append(TABLE_OVERRIDE[page])
                    notes.append(f"table on page {page} replaced (see TABLE_OVERRIDE)")
                last_kind = "table"
                continue
            if page in MISSING_TABLE_CAPTION and last_kind != "table" and page not in tables_done:
                tables_done.add(page)
                body.append(f"\\begin{{center}}\\small {MISSING_TABLE_CAPTION[page]}\\end{{center}}\n"
                            "\\vspace{-1.2em}\n")
                last_kind = "cap"
            rows, widths = el[2], el[3]
            ncol = max(len(r) for r in rows)
            if len(widths) != ncol:
                widths = [1 / ncol] * ncol
            cols = "|" + "|".join(f">{{\\raggedright\\arraybackslash}}p{{{max(w, 0.06) * 0.88:.3f}\\textwidth}}"
                                  for w in widths) + "|"
            lines = [" & ".join(esc(c) for c in (r + [""] * (ncol - len(r)))) + r" \\ \hline"
                     for r in rows]
            if last_kind == "table":          # continuation of the table on the previous page
                body[-1] = body[-1].replace("\\end{longtable}\n}\n", "\n".join(lines) +
                                            "\n\\end{longtable}\n}\n")
            else:
                body.append("{\\small\n\\begin{longtable}{" + cols + "}\n\\hline\n"
                            + "\n".join(lines) + "\n\\end{longtable}\n}\n")
            last_kind = "table"
            continue

        b = el[2]
        raw = block_text(b)
        if not raw:
            continue
        if skip_until is not None:
            if raw.startswith(skip_until):
                skip_until = None
            else:
                continue
        passage = next((ps for ps in PASSAGES if ps["page"] == el[1] + 1
                        and raw.startswith(ps["start"]) and ps["start"] not in done_passages), None)
        if passage:
            flush_img()
            done_passages.add(passage["start"])
            body.append(passage["tex"])
            used_eqs.update(re.findall(r"\\tag\{(4\.\d)\}", passage["tex"]))
            skip_until = passage["end"]
            last_kind = "eq"
            continue
        tex = block_tex(b)
        m_cap = CAP.match(raw)
        # a caption's second line can come out as its own block: join it back
        if last_kind == "cap" and raw[:1].islower() and body:
            for end in ("}\n\\end{figure}\n", "\\end{center}\n\\vspace{-1.2em}\n", "\\end{center}\n"):
                if body[-1].endswith(end):
                    body[-1] = body[-1][: -len(end)] + " " + esc(raw) + end
                    break
            continue

        m_ch = re.fullmatch(r"(\d+)\. CHAPTER", raw.strip())     # "2. CHAPTER" in the document
        if m_ch:
            flush_img()
            pending_chapter = int(m_ch.group(1))
            last_kind = "head"
            continue
        if raw.startswith("Chapter ") and re.fullmatch(r"Chapter \d+", raw.strip()):
            flush_img()
            pending_chapter = int(raw.split()[1])
            last_kind = "head"
            continue
        if pending_chapter is not None:
            num = pending_chapter
            pending_chapter = None
            if all_bold(b) and not HEAD.match(raw):
                title = esc(raw)
                body.append(f"\\setcounter{{chapter}}{{{num - 1}}}\n\\chapter{{{title}}}\n")
                last_kind = "head"
                continue
            body.append(f"\\setcounter{{chapter}}{{{num - 1}}}\n\\chapter{{{CHAPTER_TITLES.get(num, '')}}}\n")
        if raw in ("Abstract", "Acknowledgement", "Bibliography"):
            flush_img()
            body.append(f"\\chapter*{{{raw}}}\n\\addcontentsline{{toc}}{{chapter}}{{{raw}}}\n")
            in_bib = raw == "Bibliography"
            last_kind = "head"
            continue
        m = HEAD.match(raw)
        if m and len(raw) < 90 and len(b["lines"]) <= 2 and not raw.endswith("."):
            flush_img()
            number, title = m.group(1), esc(m.group(2).strip())
            depth = number.count(".")
            cmd = "section" if depth == 1 else "subsection"
            body.append(f"\\{cmd}*{{{number}\\quad {title}}}\n"
                        f"\\addcontentsline{{toc}}{{{cmd}}}{{{number}\\quad {title}}}\n")
            last_kind = "head"
            continue
        if m_cap:
            kind, num, text = m_cap.groups()
            key = (kind, num)
            if key in seen_caps:
                notes.append(f"duplicate caption '{kind} {num}' on page {el[1] + 1} dropped: {raw[:60]}")
                # text that runs on after the duplicate caption (a new sentence) is kept
                rest = re.search(r"[a-z] ([A-Z][a-z]+ [a-z].*)$", raw[12:])
                if rest:
                    emit_par(esc(rest.group(1)), merge_ok=False)
                continue
            seen_caps.add(key)
            cap = f"{kind} {num}: " + esc(" ".join(text.split()))
            if f"{kind} {num}" in CAPTION_FIX:
                cap = f"{kind} {num}: " + CAPTION_FIX[f"{kind} {num}"]
                notes.append(f"{kind} {num}: caption corrected")
            if kind == "Figure":
                if pending_img is None and num in MISSING:
                    pending_img = MISSING[num]
                    notes.append(f"Figure {num}: image restored from {MISSING[num]}")
                if pending_img is None:
                    notes.append(f"Figure {num} has no image in the PDF")
                    body.append(f"\\begin{{center}}\\small {cap}\\end{{center}}\n")
                else:
                    flush_img(cap)
            else:
                flush_img()
                body.append(f"\\begin{{center}}\\small {cap}\\end{{center}}\n\\vspace{{-1.2em}}\n")
            last_kind = "cap"
            continue
        if has_math(b):
            flush_img()
            nums = re.findall(r"\((4\.\d)\)", raw)
            pre = re.split(r"[\U0001D400-\U0001D7FF\u2211]", raw)[0]
            body.append(f"% --- equation block from page {el[1] + 1}: {raw[:120]!r}\n")
            if pre.strip() and len(pre.strip()) > 3:
                emit_par(esc(pre.strip()))
            for n in nums:
                if n in EQUATIONS and n not in used_eqs:
                    used_eqs.add(n)
                    body.append(f"\\begin{{equation*}}\n  {EQUATIONS[n]} \\tag{{{n}}}\n\\end{{equation*}}\n")
            if not nums:
                notes.append(f"math without equation number on page {el[1] + 1}: {raw[:80]!r}")
                body.append(f"% MATH-REVIEW page {el[1] + 1}\n{tex}\n")
            last_kind = "eq"
            continue
        flush_img()
        if in_bib and re.match(r"^\[\d+\]", raw):
            n, rest = re.match(r"^\[(\d+)\]\s*(.*)$", tex, re.S).groups()
            body.append(f"\\bibentry{{{n}}}{{{rest}}}\n")
            last_kind = "bib"
            continue
        if re.match(r"^(\d+\.|\u2022|-)\s", raw):
            body.append("\\listitem{" + tex + "}\n")
            last_kind = "item"
            continue
        if all_bold(b) and len(raw) < 120:
            body.append("\\noindent\\textbf{" + esc(raw) + "}\n")
            last_kind = "bold"
            continue
        emit_par(tex, merge_ok=(last_kind == "par"))
    flush_img()
    missing = sorted(set(EQUATIONS) - used_eqs)
    if missing:
        notes.append(f"equations not placed: {missing}")
    return "\n".join(body), notes


# Literal clean-ups of export artefacts (not wording changes)
FIXES = [
    ("3 RQ1:", "RQ1:"),                                   # stray list number
    ("Md. To Sabbir Ahmed", "Md. Sabbir Ahmed"),          # supervisor's name
    ("idea drift", "concept drift"),                      # mistranslated term
    ("because drift indi.",                               # sentence cut off mid-word
     "because drift indicators alone can both miss real degradation and raise alarms about drift "
     "that does no harm."),
    ("The student shows understanding of drift and performance measurements.",   # lost heading
     "\\subsection*{4.1.4\\quad Drift and Performance Measures}\n"
     "\\addcontentsline{toc}{subsection}{4.1.4\\quad Drift and Performance Measures}"),
    ("Learning Rate & 0.03", "Learning Rate & 0.05"),     # the model's actual learning rate
    ("0.751/0.0943", "0.751/0.943"),                      # Table 5.5 typo
    (" & IEEE-CIS (12 weeks) &  &  \\\\",                  # Table 5.5 header missing two columns
     " & IEEE-CIS (12 weeks) & Sparkov (52 weeks) & BAF (19 weeks) \\\\"),
    ("Paris", "Pairs"),                                   # Table 5.4 header
]


def front_matter():
    """Title page, declaration and approval (as in the P1), from ../p1/content.py."""
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "p1")))
    import content as C
    m = C.META
    names = "\n".join(f"  {n}\\\\ {i}\\\\[0.7em]" for n, i in C.AUTHORS)
    sig = []
    for k in range(0, len(C.AUTHORS), 2):
        pair = C.AUTHORS[k:k + 2]
        cells = [f"\\rule{{0.38\\textwidth}}{{0.4pt}}\\newline {n}\\newline {i}" for n, i in pair]
        sig.append(" & ".join(cells + [""] * (2 - len(cells))) + " \\\\[1.2em]")
    approvers = [("Supervisor:", "(Member)", m["supervisor"]),
                 ("Co-Supervisor:", "(Member)", m["cosupervisor"]),
                 ("Thesis Coordinator:", "(Member)", m["coordinator"]),
                 ("Head of Department:", "(Chair)", m["head"])]
    committee = "\n".join(
        f"\\textbf{{{role}}}\\newline {mem} & \\rule{{0.5\\textwidth}}{{0.4pt}}\\newline {name}"
        f"\\newline {title}\\newline {m['department']}\\newline {m['university']} \\\\[1.4em]"
        for role, mem, (name, title) in approvers)
    listed = "\n".join(f"  \\item {n} ({i})" for n, i in C.AUTHORS)
    return (
        "\\pagenumbering{gobble}\n\\begin{titlepage}\n\\centering\n\\vspace*{1cm}\n"
        f"{{\\LARGE\\bfseries {C.TITLE}\\par}}\n\\vspace{{2em}}\nby\\par\\vspace{{1.5em}}\n"
        f"{names}\n\\vfill\n"
        f"A thesis submitted to the {m['department']}\\\\ in partial fulfillment of the "
        f"requirements for the degree of\\\\ {m['degree']}\\par\n\\vspace{{2.5em}}\n"
        f"{m['department']}\\\\ {m['university']}\\\\ {m['date']}\\par\n\\vfill\n"
        f"\\copyright\\ {m['copyright_year']}. {m['university']}\\\\ All rights reserved.\n"
        "\\end{titlepage}\n\n"
        "\\pagenumbering{roman}\n"
        "\\chapter*{Declaration}\n\\addcontentsline{toc}{chapter}{Declaration}\n"
        "It is hereby declared that\n\\begin{enumerate}\n"
        f"  \\item The thesis submitted is our own original work while completing degree at {m['university']}.\n"
        "  \\item The thesis does not contain material previously published or written by a third "
        "party, except where this is appropriately cited through full and accurate referencing.\n"
        "  \\item The thesis does not contain material which has been accepted, or submitted, for "
        "any other degree or diploma at a university or other institution.\n"
        "  \\item We have acknowledged all main sources of help.\n\\end{enumerate}\n\n"
        "\\textbf{Student's Full Name \\& Signature:}\\par\\vspace{2.5em}\n"
        "\\begin{center}\n\\begin{tabular}{>{\\centering\\arraybackslash}p{0.42\\textwidth}"
        ">{\\centering\\arraybackslash}p{0.42\\textwidth}}\n" + "\n".join(sig) +
        "\n\\end{tabular}\n\\end{center}\n\n"
        "\\chapter*{Approval}\n\\addcontentsline{toc}{chapter}{Approval}\n"
        f"The thesis titled ``{C.TITLE}'' submitted by\n\\begin{{enumerate}}\n{listed}\n"
        f"\\end{{enumerate}}\nof {m['semester']} has been accepted as satisfactory in partial "
        f"fulfillment of the requirement for the degree of {m['degree']} in {m['degree_term']}.\n\n"
        "\\textbf{Examining Committee:}\\par\\vspace{1em}\n"
        "\\noindent\\begin{tabular}{@{}p{0.3\\textwidth}p{0.65\\textwidth}@{}}\n"
        + committee + "\n\\end{tabular}\n\n")


TOC = ("\\renewcommand{\\contentsname}{Table of Contents}\n\\tableofcontents\n"
       "\\clearpage\n\\pagenumbering{arabic}\n\n")

PREAMBLE = r"""%% Nirnoy's thesis, converted from the Google Docs PDF to LaTeX (compile with pdfLaTeX).
%% Figures are the original high-resolution images; Figures 1.1 and 1.2 restored.
\documentclass[11pt,letterpaper]{report}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{graphicx,float,longtable,array,amsmath,parskip}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelformat=empty}
\setlength{\parskip}{0.7em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.25}
% chapter headings in the document's style: "Chapter N" then the title, left aligned
\makeatletter
\renewcommand{\@makechapterhead}[1]{{\parindent 0pt \raggedright\normalfont\bfseries
  \large\@chapapp\space\thechapter\par\vspace{0.6em}\large #1\par\vspace{1.2em}}}
\renewcommand{\@makeschapterhead}[1]{{\parindent 0pt \raggedright\normalfont\bfseries
  \LARGE #1\par\vspace{1.2em}}}
\makeatother
\usepackage{titlesec}
\titleformat*{\section}{\normalsize\bfseries}
\titleformat*{\subsection}{\normalsize\bfseries}
\titlespacing*{\section}{0pt}{1.2em}{0.5em}
\titlespacing*{\subsection}{0pt}{1.0em}{0.4em}
\newcommand{\listitem}[1]{\par\hangindent=1.5em\hangafter=1 #1\par}
\newcommand{\bibentry}[2]{\par\hangindent=2.2em\hangafter=1 [#1]\ #2\par}

\begin{document}
"""


def main():
    pdf = sys.argv[1]
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    body, notes = convert(pdf)
    for old, new in FIXES:
        if old not in body:
            print(f"fix not applied (text not found): {old!r}")
        body = body.replace(old, new)
    # the table of contents goes after the front matter, where Chapter 1 begins
    first = body.index("\\setcounter{chapter}{0}")
    body = body[:first] + TOC + body[first:]
    with open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + front_matter() + body + "\n\\end{document}\n")
    for name in set(re.findall(r"figures/([\w.\-]+\.png)", body)):
        shutil.copy(os.path.join(FIG_SRC, name), os.path.join(OUT, "figures", name))
    print("\n".join(notes))
    print(f"Wrote {OUT}\\main.tex")


if __name__ == "__main__":
    main()
