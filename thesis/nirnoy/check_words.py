"""Compare the words of the PDF with the words of latex/main.tex, to show that the conversion
dropped or added nothing (apart from equation fragments and LaTeX markup)."""

import re
import sys
from collections import Counter

import pymupdf

pdf = sys.argv[1]
doc = pymupdf.open(pdf)
pdf_text = " ".join(p.get_text() for p in doc)
pdf_text = re.sub(r"[\U0001D400-\U0001D7FF\u2211\u03c4\u03c0\u03bb\u03b4\u03ba\u03f5\u210e]", " ", pdf_text)

tex = open("latex/main.tex", encoding="utf-8").read()
tex = tex[tex.index(r"\begin{document}"):]
tex = re.sub(r"(?m)^%.*$", " ", tex)                                  # comments
tex = re.sub(r"(?m)^\\addcontentsline.*$", " ", tex)                # hidden TOC copies of headings
tex = re.sub(r"\\begin\{equation\*\}.*?\\end\{equation\*\}", " ", tex, flags=re.S)
tex = re.sub(r"\$[^$]*\$", " ", tex)                                  # inline math
tex = re.sub(r"\\(textbf|textit|caption\*|chapter\*?|section\*|subsection\*|addcontentsline|listitem|bibentry)", " ", tex)
tex = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?", " ", tex)                   # other commands
tex = tex.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_").replace("``", '"').replace("''", '"')


def words(t):
    return Counter(w.lower() for w in re.findall(r"[A-Za-z]{3,}", t))


a, b = words(pdf_text), words(tex)
missing, extra = a - b, b - a
ignore = {"toc", "chapter", "section", "subsection", "quad", "figure", "hline", "small", "center",
          "textwidth", "png", "figures", "raggedright", "arraybackslash", "longtable", "includegraphics",
          "centering", "vspace", "noindent", "setcounter", "tag", "begin", "end"}
extra = Counter({k: v for k, v in extra.items() if k not in ignore})
print("words in PDF:", sum(a.values()), "| in LaTeX:", sum(b.values()))
print("missing from LaTeX:", sum(missing.values()), dict(missing.most_common(25)))
print("extra in LaTeX:", sum(extra.values()), dict(extra.most_common(25)))
