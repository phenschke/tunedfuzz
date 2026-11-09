"""Turns RISE humanities data benchmark payloads into pairs of truth and LLM reading, for testing only.

Writes the corpora rise_fraktur (Basler Avisblatt adverts, 1731–1762) and rise_medieval (Rot pilgrimage
accounts, 15th c.), each read by about a hundred LLMs.

Usage: pairs_rise.py <downloads/rise> <out.parquet>
"""
import glob
import gzip
import json
import sys

from common import diffed_words, write

src, out = sys.argv[1:3]

CORPUS = {"fraktur_adverts": "rise_fraktur", "medieval_manuscripts": "rise_medieval"}


def fields(score):
    """Yields (truth, reading) of every scored text field."""
    if isinstance(score, dict):
        g, r = score.get("ground_truth"), score.get("response")
        if isinstance(g, str) and isinstance(r, str):
            yield g, r
            return
        for v in score.values():
            yield from fields(v)
    elif isinstance(score, list):
        for v in score:
            yield from fields(v)


rows = []
for bench, corpus in CORPUS.items():
    for f in sorted(glob.glob(f"{src}/payloads/{bench}/*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            rec = json.loads(line)
            score = (rec.get("source_record") or {}).get("score") or {}
            for g, r in fields(score.get("field_scores") or {}):
                rows += [(corpus, rec["object_id"], gt, o) for gt, o in diffed_words(r, g) if "-" not in o]
write(rows, out)
