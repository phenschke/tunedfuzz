"""Turns BLN600 (CC BY-NC) into pairs, for testing only.

Usage: pairs_bln600.py <dataset.csv from Shef-AIRE/vlms_post-ocr_correction> <out.parquet>
"""
import sys

import polars as pl

from common import diffed_words, write

src, out = sys.argv[1:3]
rows = []
for sample, ocr, gt in pl.read_csv(src).select("Sample", "OCR Text", "Ground Truth").iter_rows():
    if ocr and gt:
        rows += [("BLN600", f"bln_{sample}", g, o) for g, o in diffed_words(ocr, gt)]
write(rows, out)
