"""Word pairs from the kurrent-vlm benchmark, one corpus per model, on German handwriting in Kurrent.

Reads the bench-v4 runs of the six strongest models and their references through kurrent-vlm's
scorer, so both sides get the bench folds (long s, hyphens, gap marks) before pairing. Keeps suites
A, ID and B, drops pages in Latin script and pages not in German, and keeps Titan to the pages
Transkribus cannot have trained on.

Usage: PYTHONPATH=<kurrent-vlm>/src pairs_kurrent_bench.py <out parquet>
"""
import sys

from kurrent_vlm import benchscore as bs

from common import diffed_words, write

RUNS = {
    "kbench_argon4": "argon4-full-v2-b4",
    "kbench_fable51": "fable51-core-v2-b4",
    "kbench_opus55": "opus55-core-v2-b4",
    "kbench_flash38": "flash38-full-v2-b4",
    "kbench_titan1": "titan1-full-b4",
    "kbench_titan2": "titan2-full-b4",
}
SUITES = {"A", "ID", "B"}

# Diplomatic references abbreviate (d, u, dr, dß) where models expand, so differing pairs with a side
# of two letters or fewer are mostly convention. Case-only pairs stay, because Kurrent D and d look alike.
MIN_LETTERS = 3


def informative(g: str, o: str) -> bool:
    return g == o or g.lower() == o.lower() or min(len(g), len(o)) >= MIN_LETTERS


def keep(row: dict, titan: bool) -> bool:
    return (row["bench"]["suite"] in SUITES
            and row["script_class"] != "latin"
            and row["language"] in ("de", None)
            and (not titan or row.get("in_baseline_training") in bs.TITAN_CLEAN))


rows = []
for corpus, run in RUNS.items():
    cfg, answers, bench = bs.run_rows(run)
    titan = cfg["model"].startswith("transkribus")
    for row in bench:
        a = answers.get(row["id"])
        if not keep(row, titan) or not (a and a.get("ok")):
            continue
        pred = a["text"]
        if any("page/folio number" in m for m in row["bench"]["scoring"].get("mask_omits", [])):
            pred = bs.mask_page_numbers(pred)
        doc = f"{row['source']}|{row['bench'].get('doc') or row['id']}"
        rows += [(corpus, doc, g, o) for g, o in diffed_words(bs.fold(pred), bs.fold(row["text"] or ""))
                 if informative(g, o)]

write(rows, sys.argv[1])
