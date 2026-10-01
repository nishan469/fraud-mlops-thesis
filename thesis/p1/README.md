# P1: A Drift-Aware Continuous Learning MLOps Framework for Financial Fraud Detection

Front matter, Chapter 1 (Introduction) and Chapter 2 (Literature Review), following the
BRAC University CSE thesis layout.

| File | What |
|---|---|
| `content.py` | All text, tables, abbreviations and references. **Edit here**, then rebuild. |
| `P1_Drift_Aware_Continuous_Learning_MLOps.pdf` | Ready-to-read PDF |
| `latex/` | Overleaf project: `main.tex`, `chapters/`, `figures/` (compiles with pdflatex) |
| `figures.py`, `figures/` | The four diagrams (Figures 1.1, 1.2, 2.1 and 2.2) |
| `build_pdf.py`, `build_latex.py`, `markup.py` | Builders (citations numbered in IEEE order of first use) |

Rebuild after editing `content.py`:

```bash
python figures.py && python build_pdf.py && python build_latex.py
```

Citations are written as `[@key]` in `content.py`; a key must exist in `REFERENCES`, and every
reference must be cited, or the build stops with an error.

To use the official BRAC Overleaf template instead of `latex/main.tex`, paste
`latex/chapters/chapter1.tex`, `chapter2.tex` and `bibliography.tex` into it and copy the
`figures/` folder (the chapters need the `graphicx`, `longtable`, `booktabs`, `array`,
`enumitem` and `url` packages).
