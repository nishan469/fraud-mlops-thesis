"""Write the P1 as LaTeX for Overleaf (paste the chapters into the BRAC template, or compile
latex/main.tex on its own with pdflatex).

  python build_latex.py   ->  latex/main.tex, latex/chapters/*.tex, latex/figures/*.png
"""

import os
import re
import shutil

import content as C
from markup import numbering

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "latex")
CITES, FIGS, TABS = numbering()
TOKEN = re.compile(r"(<b>|</b>|<i>|</i>|\[@[^\]]+\]|https?://\S+)")
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


def plain(text):
    text = "".join(SPECIAL.get(ch, ch) for ch in text)
    text = text.replace("\u2013", "--").replace("\u201c", "``").replace("\u201d", "''")
    # straight double quotes: alternate opening and closing
    parts = text.split('"')
    return "".join(p + (("``" if i % 2 == 0 else "''") if i < len(parts) - 1 else "")
                   for i, p in enumerate(parts))


def tex(text):
    out = []
    for tok in TOKEN.split(text):
        if not tok:
            continue
        if tok == "<b>":
            out.append(r"\textbf{")
        elif tok == "<i>":
            out.append(r"\textit{")
        elif tok in ("</b>", "</i>"):
            out.append("}")
        elif tok.startswith("[@"):
            keys = [k.strip().lstrip("@") for k in tok[2:-1].split(",")]
            if keys[0].startswith(("fig:", "tab:")):
                out.append(r"\ref{" + keys[0] + "}")
            else:
                out.append(r"\cite{" + ",".join(keys) + "}")
        elif tok.startswith("http"):
            out.append(r"\url{" + tok + "}")
        else:
            out.append(plain(tok))
    return "".join(out)


def block_tex(block):
    kind = block[0]
    if kind == "p":
        return tex(block[1]) + "\n"
    if kind == "list":
        env = "enumerate" if block[2] else "itemize"
        opts = "" if block[2] else "[label={}, leftmargin=2em]"
        items = "\n".join(r"  \item " + tex(i) for i in block[1])
        return f"\\begin{{{env}}}{opts}\n{items}\n\\end{{{env}}}\n"
    if kind == "fig":
        path, caption = C.FIGURES[block[1]]
        return ("\\begin{figure}[htbp]\n  \\centering\n"
                f"  \\includegraphics[width=\\textwidth]{{{path}}}\n"
                f"  \\caption{{{tex(caption)}}}\n  \\label{{fig:{block[1]}}}\n\\end{{figure}}\n")
    if kind == "table":
        spec = C.TABLES[block[1]]
        cols = "".join(f">{{\\raggedright\\arraybackslash}}p{{{w * 0.94:.2f}\\textwidth}}"
                       for w in spec["widths"])
        head = " & ".join(r"\textbf{" + tex(h) + "}" for h in spec["header"]) + r" \\"
        rows = "\n".join(" & ".join(tex(c) for c in r) + r" \\ \midrule" for r in spec["rows"])
        rows = rows[: -len(r" \midrule")] + r" \bottomrule"
        return ("{\\footnotesize\n"
                f"\\begin{{longtable}}{{{cols}}}\n"
                f"\\caption{{{tex(spec['caption'])}}}\\label{{tab:{block[1]}}}\\\\\n"
                f"\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
                f"\\toprule\n{head}\n\\midrule\n\\endhead\n{rows}\n\\end{{longtable}}\n}}\n")
    raise ValueError(kind)


def chapter_tex(ch):
    lines = [f"\\chapter{{{tex(ch['title'])}}}", ""]
    for sec in ch["sections"]:
        lines += [f"\\section{{{tex(sec['title'])}}}", ""]
        lines += [block_tex(b) for b in sec["blocks"]]
    return "\n".join(lines)


def bibliography_tex():
    items = []
    for key, _ in sorted(CITES.items(), key=lambda kv: kv[1]):
        entry = re.sub(r"(pp\. \d+)-(\d+)", r"\1--\2", tex(C.REFERENCES[key]))   # page ranges only
        items.append(f"\\bibitem{{{key}}}\n{entry}\n")
    return ("% IEEE style, numbered in order of first citation\n"
            "\\begin{thebibliography}{99}\n" + "\n".join(items) + "\\end{thebibliography}\n")


def main_tex():
    m = C.META
    authors = "\\\\[0.4em]\n".join(f"{n}\\\\ {i}" for n, i in C.AUTHORS)
    names = "\n".join(f"  \\item {n} ({i})" for n, i in C.AUTHORS)
    sigs = "\n\\hfill\n".join(
        f"\\begin{{minipage}}[t]{{0.45\\textwidth}}\\centering\\vspace{{2.2em}}"
        f"\\rule{{0.8\\textwidth}}{{0.4pt}}\\\\ {n}\\\\ {i}\\end{{minipage}}" for n, i in C.AUTHORS)
    committee = "\n".join(
        f"\\noindent\\begin{{minipage}}[t]{{0.3\\textwidth}}\\textbf{{{role}}}\\\\ {mem}"
        f"\\end{{minipage}}\\begin{{minipage}}[t]{{0.65\\textwidth}}\\vspace{{1.5em}}"
        f"\\rule{{0.7\\textwidth}}{{0.4pt}}\\\\ {name}\\\\ {title}\\\\ {m['department']}\\\\ "
        f"{m['university']}\\end{{minipage}}\\\\[1.5em]"
        for role, mem, (name, title) in [("Supervisor:", "(Member)", m["supervisor"]),
                                          ("Co-Supervisor:", "(Member)", m["cosupervisor"]),
                                          ("Thesis Coordinator:", "(Member)", m["coordinator"]),
                                          ("Head of Department:", "(Chair)", m["head"])])
    abbrev = "\n".join(f"\\textbf{{{a}}} & {tex(d)} \\\\" for a, d in C.ABBREVIATIONS)
    return rf"""% P1: {C.TITLE}
% Generated by build_latex.py from content.py. Compiles on Overleaf with pdflatex.
% To use the official BRAC template instead, paste chapters/*.tex and the bibliography there.
\documentclass[12pt,a4paper]{{report}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx,longtable,booktabs,array,enumitem,setspace}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{url}}
\onehalfspacing
\setlist[enumerate]{{itemsep=2pt}}

\begin{{document}}
\pagenumbering{{gobble}}
\begin{{titlepage}}
\centering
{{\LARGE\bfseries {tex(C.TITLE)}\par}}
\vspace{{1.5em}} by\par\vspace{{1em}}
{authors}\par
\vfill
A thesis submitted to the {m['department']}\\ in partial fulfillment of the requirements for the degree of\\ {m['degree']}\par
\vspace{{1.5em}}
{m['department']}\\ {m['university']}\\ {m['date']}\par
\vfill
\copyright {m['copyright_year']}. {m['university']}\\ All rights reserved.
\end{{titlepage}}

\pagenumbering{{roman}}
\chapter*{{Declaration}}\addcontentsline{{toc}}{{chapter}}{{Declaration}}
It is hereby declared that
\begin{{enumerate}}
  \item The thesis submitted is our own original work while completing degree at {m['university']}.
  \item The thesis does not contain material previously published or written by a third party, except where this is appropriately cited through full and accurate referencing.
  \item The thesis does not contain material which has been accepted, or submitted, for any other degree or diploma at a university or other institution.
  \item We have acknowledged all main sources of help.
\end{{enumerate}}
\vspace{{1em}}\noindent\textbf{{Student's Full Name \& Signature:}}\par
{sigs}

\chapter*{{Approval}}\addcontentsline{{toc}}{{chapter}}{{Approval}}
The thesis titled ``{tex(C.TITLE)}'' submitted by
\begin{{enumerate}}
{names}
\end{{enumerate}}
of {m['semester']} has been accepted as satisfactory in partial fulfillment of the requirement for the degree of {m['degree']} in {m['degree_term']}.

\vspace{{1em}}\noindent\textbf{{Examining Committee:}}\par\vspace{{0.5em}}
{committee}

\chapter*{{Abstract}}\addcontentsline{{toc}}{{chapter}}{{Abstract}}
{chr(10).join(tex(p) + chr(10) for p in C.ABSTRACT)}
\noindent\textbf{{Keywords:}} {tex(C.KEYWORDS)}

\chapter*{{Acknowledgement}}\addcontentsline{{toc}}{{chapter}}{{Acknowledgement}}
{chr(10).join(tex(p) + chr(10) for p in C.ACKNOWLEDGEMENT)}
\tableofcontents
\listoffigures\addcontentsline{{toc}}{{chapter}}{{List of Figures}}
\listoftables\addcontentsline{{toc}}{{chapter}}{{List of Tables}}
\chapter*{{Nomenclature}}\addcontentsline{{toc}}{{chapter}}{{Nomenclature}}
The next list describes several symbols and abbreviations that will be used later within the body of the document.\par\vspace{{0.5em}}
\noindent\begin{{tabular}}{{@{{}}p{{0.18\textwidth}}p{{0.75\textwidth}}@{{}}}}
{abbrev}
\end{{tabular}}

\clearpage
\pagenumbering{{arabic}}
\input{{chapters/chapter1}}
\input{{chapters/chapter2}}

\clearpage
\addcontentsline{{toc}}{{chapter}}{{Bibliography}}
\input{{chapters/bibliography}}
\end{{document}}
"""


def main():
    os.makedirs(os.path.join(OUT, "chapters"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    for i, ch in enumerate(C.CHAPTERS, start=1):
        with open(os.path.join(OUT, "chapters", f"chapter{i}.tex"), "w", encoding="utf-8") as f:
            f.write(chapter_tex(ch))
    with open(os.path.join(OUT, "chapters", "bibliography.tex"), "w", encoding="utf-8") as f:
        f.write(bibliography_tex())
    with open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8") as f:
        f.write(main_tex())
    for path, _ in C.FIGURES.values():
        shutil.copy(os.path.join(HERE, path), os.path.join(OUT, path))
    print(f"Wrote LaTeX project to {OUT}")


if __name__ == "__main__":
    main()
