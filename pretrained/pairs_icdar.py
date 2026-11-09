"""Turns the ICDAR 2019 post-OCR corpus into pairs, one corpus per subset (icdar_DE1, icdar_EN1, ...).

Usage: pairs_icdar.py <full_22M_without_Finnish dir> <out.parquet>
"""
import glob
import sys

from common import aligned_words, write

root, out = sys.argv[1:3]
rows = []
for d in sorted(glob.glob(f"{root}/*/*/")):
    sub = d.rstrip("/").rsplit("/", 1)[1]
    for path in sorted(glob.glob(f"{d}*.txt")):
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
        ocr = next((l[14:] for l in lines if l.startswith("[OCR_aligned]")), None)
        gs = next((l[14:] for l in lines if l.startswith("[ GS_aligned]")), None)
        if ocr is None or gs is None or len(ocr) != len(gs):
            continue
        doc = f"{sub}_{path.rsplit('/', 1)[1]}"
        rows += [(f"icdar_{sub}", doc, g, o) for g, o in aligned_words(ocr, gs)]
write(rows, out)
