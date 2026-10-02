# Thesis documents: P1 and the full draft

The P1 (front matter, Chapter 1 Introduction, Chapter 2 Literature Review) and the full draft
so far (adds Chapter 3 Methodology), following the
BRAC University CSE thesis layout.

| File | What |
|---|---|
| `content.py` | Front matter, Chapters 1-2, abbreviations and references. **Edit here**, then rebuild. |
| `chapter3.py`, `chapter4.py` | Methodology and Design (the P2 chapter) and Results: text, figures, tables, equations, subsections (`("sub", title, key)`, referenced as `[@sec:key]`) |
| `results_figures.py` | Chapter 4 figures, drawn from the experiment outputs in `../../outputs` |
| `Thesis_Draft_Ch1-4.pdf`, `latex_draft/` | Full draft so far (Chapters 1-4), PDF and Overleaf project |
| `P1_Drift_Aware_Continuous_Learning_MLOps.pdf` | The P1 (Chapters 1-2) |
| `Thesis_Report.pdf`, `latex_report/`, `report.py` | The full report in the BRAC report-template structure (6 chapters, Chapter 3 left empty): reuses the chapters above and adds Data Collection, Dataset Overview, Project Plan, Implementation, Final Design Adjustments, Discussions and Conclusion |
| `P2_Methodology_and_Design.pdf`, `latex_p2/` | The P2: Chapter 4 (4.1 Design Process or Methodology Overview, 4.2 Preliminary Design or Design (Model) Specification) |
| `latex/` | Overleaf project: `main.tex`, `chapters/`, `figures/` (compiles with pdflatex) |
| `figures.py`, `figures/` | The diagrams (Figures 1.1-3.2; results figures 4.1-4.4 come from results_figures.py) |
| `build_pdf.py`, `build_latex.py`, `markup.py` | Builders (citations numbered in IEEE order of first use) |

Rebuild after editing `content.py`:

```bash
python figures.py && python build_pdf.py && python build_latex.py              # P1 (chapters 1-2)
python build_pdf.py --p2 && python build_latex.py --p2                          # P2 (Chapter 4)
python build_pdf.py --report && python build_latex.py --report                  # full report
python results_figures.py && python build_pdf.py --chapters 4 && python build_latex.py --chapters 4   # full draft
```

Citations are written as `[@key]` in `content.py`; a key must exist in `REFERENCES`, and every
reference must be cited, or the build stops with an error.

To use the official BRAC Overleaf template instead of `latex/main.tex`, paste
`latex/chapters/chapter1.tex`, `chapter2.tex` and `bibliography.tex` into it and copy the
`figures/` folder (the chapters need the `graphicx`, `longtable`, `booktabs`, `array`,
`enumitem`, `url` and, from Chapter 3 on, `amsmath` packages).
