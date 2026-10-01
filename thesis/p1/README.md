# Thesis documents: P1 and the full draft

The P1 (front matter, Chapter 1 Introduction, Chapter 2 Literature Review) and the full draft
so far (adds Chapter 3 Methodology), following the
BRAC University CSE thesis layout.

| File | What |
|---|---|
| `content.py` | Front matter, Chapters 1-2, abbreviations and references. **Edit here**, then rebuild. |
| `chapter3.py` | Chapter 3 (Methodology): text, figures, tables and equations |
| `Thesis_Draft_Ch1-3.pdf`, `latex_draft/` | Full draft so far (Chapters 1-3), PDF and Overleaf project |
| `P1_Drift_Aware_Continuous_Learning_MLOps.pdf` | Ready-to-read PDF |
| `latex/` | Overleaf project: `main.tex`, `chapters/`, `figures/` (compiles with pdflatex) |
| `figures.py`, `figures/` | The diagrams (Figures 1.1, 1.2, 2.1, 2.2, 3.1 and 3.2) |
| `build_pdf.py`, `build_latex.py`, `markup.py` | Builders (citations numbered in IEEE order of first use) |

Rebuild after editing `content.py`:

```bash
python figures.py && python build_pdf.py && python build_latex.py              # P1 (chapters 1-2)
python build_pdf.py --chapters 3 && python build_latex.py --chapters 3           # full draft
```

Citations are written as `[@key]` in `content.py`; a key must exist in `REFERENCES`, and every
reference must be cited, or the build stops with an error.

To use the official BRAC Overleaf template instead of `latex/main.tex`, paste
`latex/chapters/chapter1.tex`, `chapter2.tex` and `bibliography.tex` into it and copy the
`figures/` folder (the chapters need the `graphicx`, `longtable`, `booktabs`, `array`,
`enumitem`, `url` and, from Chapter 3 on, `amsmath` packages).
