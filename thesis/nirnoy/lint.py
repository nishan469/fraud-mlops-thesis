"""Static checks of latex/main.tex (no LaTeX compiler needed): brace balance, unescaped special
characters outside math and tables, non-ASCII characters, and element counts."""

import re

s = open("latex/main.tex", encoding="utf-8").read()
body = s[s.index(r"\begin{document}"):]

t = re.sub(r"\\[{}]", "", body)
depth = 0
for i, ch in enumerate(t):
    depth += (ch == "{") - (ch == "}")
    if depth < 0:
        print("closing brace without opening near:", t[max(0, i - 80):i + 20])
        break
print("final brace depth:", depth)

issues = []
in_tab = False
for n, line in enumerate(body.splitlines(), 1):
    if line.startswith(r"\begin{longtable}"):
        in_tab = True
    if line.startswith(r"\end{longtable}"):
        in_tab = False
    if line.lstrip().startswith("%"):
        continue
    plain = re.sub(r"\$[^$]*\$", "", line)
    if "\\tag{" in line or "\\mathrm" in line or "\\mathcal" in line:
        continue
    for ch in "&%#_^~":
        if re.search(r"(?<!\\)" + re.escape(ch), plain) and not (ch == "&" and in_tab) \
                and not (ch == "_" and "includegraphics" in line) \
                and not (ch == "~" and "textasciitilde" in line):
            issues.append((n, ch, line[:100]))
    if any(ord(c) > 127 for c in line):
        issues.append((n, "non-ascii", [c for c in line if ord(c) > 127][:5]))
for x in issues[:40]:
    print(x)
print("issues:", len(issues))
print("longtables:", body.count(r"\begin{longtable}"), "| figures:", body.count(r"\begin{figure}"),
      "| table captions:", len(re.findall(r"small Table \d", body)),
      "| equations:", body.count(r"\tag{"), "| references:", body.count(r"\bibentry{"))
