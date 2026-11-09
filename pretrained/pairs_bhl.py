"""Turns IMPACT-BHL ground truth and the BHL OCR leaderboard outputs into pairs, for testing only.

Writes the corpora bhl_tesseract, bhl_kraken and bhl_vlm, which pools every permissively licensed
VLM. It leaves out Datalab's models because their licence restricts use.

Usage: pairs_bhl.py <downloads/bhl> <out.parquet>
"""
import glob
import re
import sys
import unicodedata

import polars as pl

from common import diffed_words, write

src, out = sys.argv[1:3]

DATALAB = {"surya-ocr25", "chandra-ocr2", "indra-ocr"}
CORPUS = {"tesseract": "bhl_tesseract", "kraken-ppocrv6": "bhl_kraken"}
TAG = re.compile(r"<[^>]+>")
HYPHEN = re.compile(r"(\w)[-¬]\s*\n\s*(\w)")


def clean(text: str) -> str:
    """Drops HTML tags, joins hyphenated line breaks, and applies NFKC, which also turns ſ into s."""
    return unicodedata.normalize("NFKC", HYPHEN.sub(r"\1\2", TAG.sub(" ", text)))


def failed(row: dict) -> bool:
    return any(isinstance(v, str) and v.startswith("__ERR__") for v in row.values())


gt = {int(p): t for p, t in pl.read_parquet(f"{src}/gt/metadata.parquet", columns=["PageID", "text"]).iter_rows()}
rows = []
for path in sorted(glob.glob(f"{src}/outputs/rows/*")):
    system = path.split("/")[-1]
    if system in DATALAB:
        continue
    d = pl.read_parquet(f"{path}/normalized.parquet")
    sealed = {int(r["PageID"]) for r in pl.read_parquet(f"{path}/raw.parquet").iter_rows(named=True) if failed(r)}
    errors = d["error"] if "error" in d.columns else pl.Series([None] * len(d))
    for page, md, err in zip(d["PageID"].cast(pl.Int64), d["markdown"], errors):
        if (err and err != "None") or not md or md.startswith("__ERR__") or page in sealed or not gt.get(page):
            continue

        # A hyphen left inside a word is a line break the truth joined

        rows += [(CORPUS.get(system, "bhl_vlm"), f"{system}_{page}", g, o)
                 for g, o in diffed_words(clean(md), clean(gt[page])) if "-" not in o]
write(rows, out)
