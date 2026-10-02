"""Shared helpers: citation numbering (IEEE order of first appearance), figure/table numbering
and cross-reference resolution, used by both the PDF and the LaTeX builders."""

import re

import content as C

CITE = re.compile(r"\[@([^\]]+)\]")


def _texts_in_order():
    for ch in C.CHAPTERS:
        for sec in ch["sections"]:
            for block in sec["blocks"]:
                kind = block[0]
                if kind == "p":
                    yield block[1]
                elif kind == "list":
                    yield from block[1]
                elif kind == "fig":
                    yield C.FIGURES[block[1]][1]      # captions can cite too
                elif kind == "table":
                    for row in C.TABLES[block[1]]["rows"]:
                        yield from row


def numbering():
    """Citation numbers, and labels like '1.1' for figures, tables, equations ('eq:key') and
    sections ('sec:key': a section's "key", or the key of a ("sub", title, key) block)."""
    cites = {}
    for text in _texts_in_order():
        for group in CITE.findall(text):
            for key in group.split(","):
                key = key.strip().lstrip("@")
                if key.startswith(("fig:", "tab:", "eq:", "sec:")):
                    continue
                if key not in C.REFERENCES:
                    raise KeyError(f"Unknown reference: {key}")
                cites.setdefault(key, len(cites) + 1)
    figs, tabs = {}, {}
    for c, ch in enumerate(C.CHAPTERS, start=C.FIRST_CHAPTER):
        nf = nt = ne = 0
        for s, sec in enumerate(ch["sections"], start=1):
            if sec.get("key"):
                tabs["sec:" + sec["key"]] = f"{c}.{s}"
            nsub = 0
            for block in sec["blocks"]:
                if block[0] == "sub":
                    nsub += 1
                    tabs["sec:" + block[2]] = f"{c}.{s}.{nsub}"   # sections share the label map
                elif block[0] == "fig":
                    nf += 1
                    figs[block[1]] = f"{c}.{nf}"
                elif block[0] == "table":
                    nt += 1
                    tabs[block[1]] = f"{c}.{nt}"
                elif block[0] == "eq":
                    ne += 1
                    tabs["eq:" + block[1]] = f"{c}.{ne}"     # equations share the label map
    unused = set(C.REFERENCES) - set(cites)
    if unused and C.REQUIRE_ALL_CITED:
        raise ValueError(f"References never cited: {sorted(unused)}")
    return cites, figs, tabs


def resolve(text, cites, figs, tabs, cite_fmt, ref_fmt):
    """Replace [@key,...] with cite_fmt([numbers or keys]) and [@fig:x]/[@tab:x]/[@eq:x] with
    ref_fmt."""
    def sub(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(",")]
        if keys[0].startswith("fig:"):
            return ref_fmt("fig", keys[0][4:], figs[keys[0][4:]])
        if keys[0].startswith("tab:"):
            return ref_fmt("tab", keys[0][4:], tabs[keys[0][4:]])
        if keys[0].startswith(("eq:", "sec:")):
            return ref_fmt(keys[0].split(":")[0], keys[0].split(":", 1)[1], tabs[keys[0]])
        return cite_fmt(keys)
    return CITE.sub(sub, text)
