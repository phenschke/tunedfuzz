"""Turns the Overproof datasets into pairs.

Dataset 1 (Sydney Morning Herald, volunteer corrections) becomes "trove", without the articles
dataset 2 re-corrects. Dataset 2 becomes "trove_clean" and dataset 3 (Chronicling America)
"chronam"; both are test corpora.

Usage: pairs_overproof.py <dir with d1/ d2/ d3/> <out.parquet>
"""
import glob
import sys

from common import diffed_words, write


def articles(path):
    """Yields (article id, [(ocr line, truth line)]). Column 2 holds the human correction."""
    doc, lines = None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("*$*OVERPROOF*$*"):
            if doc:
                yield doc, lines
            doc, lines = line.split()[1], []
        elif "||@@||" in line:
            cols = line.rstrip("\n").split("||@@||")
            lines.append((cols[0], cols[1]))
    if doc:
        yield doc, lines


root, out = sys.argv[1:3]
sources = [("trove_clean", f"{root}/d2/allArticles.txt"), ("chronam", f"{root}/d3/allArticles.txt")]
sources += [("trove", p) for p in sorted(glob.glob(f"{root}/d1/*.txt"))]
clean, rows = set(), []
for corpus, path in sources:
    for doc, lines in articles(path):
        if corpus == "trove_clean":
            clean.add(doc)
        if corpus == "trove" and doc in clean:
            continue
        for ocr, gt in lines:
            rows += [(corpus, f"{corpus}_{doc}", g, o) for g, o in diffed_words(ocr, gt)]
write(rows, out)
