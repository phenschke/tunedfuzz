"""Tests the tables by leave-one-corpus-out retrieval, or with --prior on small samples learned with
the general table as prior.

For each corpus, the general table learned from all other corpora ranks candidates from the corpus
lexicon for each misread word. Top-1 is the share of misread words whose cheapest candidate is the
truth, with ties split evenly. A corpus whose type tables.SCRIPTS lists is also ranked by that
type's table, learned without it. The Munich address books test on the surnames of every third year.

With --kurrent, each source of the kurrent-vlm bench is held out in turn. Its misread words are ranked
by the kurrent table learned from the other sources, so no page trains the table that ranks it.

Usage: evaluate.py <pairs dir> [--prior | --kurrent]
"""
import random
import sys
from collections import Counter

import numpy as np
import polars as pl
import tunedfuzz as tfz
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from common import ratio
from tables import KURRENT_ONLY, SCRIPT_OF, SCRIPTS, TEST_ONLY, costs, learn, load

MIN_LEXICON = 3
TOP_K = 30
N_TEST = 2000
SEED = 1

# Held-out corpus -> corpora from the same source, left out with it
SIBLINGS = {"kurrent_titan1": ["kurrent_titan2"], "kurrent_titan2": ["kurrent_titan1"],
            "trove_clean": ["trove"]}


def munich_test_docs(d: pl.DataFrame) -> set[str]:
    years = sorted({doc.split("_")[0] for doc in d["doc"].unique()})
    return {f"{y}_last" for y in years[2::3]}


def test_set(corpora, c):
    d = corpora[c]
    if c == "munich_directories":
        d = d.filter(pl.col("doc").str.ends_with("_last"))
        test = d.filter(pl.col("doc").is_in(munich_test_docs(corpora[c])))
    else:
        test = d
    counts = d.group_by("gt").agg(pl.col("n").sum())
    lexicon = counts.filter(pl.col("n") >= MIN_LEXICON)["gt"].sort().to_list()
    in_lex = set(lexicon)
    variants = test.filter(pl.col("gt") != pl.col("ocr")).select("gt", "ocr").unique().sort("gt", "ocr").rows()
    variants = [(g, o) for g, o in variants if g in in_lex and o not in in_lex and ratio(g, o) >= 0.5]
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(variants), min(N_TEST, len(variants)), replace=False))
    return [variants[k] for k in pick], lexicon


def retrieval(variants, lexicon, tables: dict[str, tfz.Costs]) -> dict[str, float]:
    hits = Counter()
    for gt, ocr in variants:
        cands = {c for c, _, _ in process.extract(ocr, lexicon, scorer=Levenshtein.distance, limit=TOP_K)}
        cands |= {c for c, _, _ in process.extract(ocr, lexicon, scorer=JaroWinkler.distance, limit=TOP_K)}
        cands.add(gt)
        for name, D in tables.items():
            scores = {c: tfz.distance(c, ocr, D) for c in cands}
            best = min(scores.values())
            tied = [c for c, s in scores.items() if s <= best + 1e-9]
            hits[name] += (gt in tied) / len(tied)
    return {name: round(hits[name] / len(variants), 3) for name in tables}


def held_out(corpora, c, script=None):
    """Returns the training corpora other than c and its siblings, limited to one type if given."""
    excl = {c, *SIBLINGS.get(c, []), *(() if script == "kurrent" else KURRENT_ONLY)}
    keep = set(SCRIPTS[script]) if script else set(corpora)
    return [x for x in corpora if x not in TEST_ONLY and x not in excl and x in keep]


_tables = {}


def table_without(corpora, c, script=None) -> tfz.Costs:
    """Returns the table learned from every training corpus but c, cached per training set."""
    names = tuple(sorted(held_out(corpora, c, script)))
    if names not in _tables:
        _tables[names] = costs(learn(corpora, list(names)))
    return _tables[names]


corpora = load(sys.argv[1])
pl.Config.set_tbl_rows(40)

if "--kurrent" in sys.argv:
    bench = {c: corpora[c].with_columns(source=pl.col("doc").str.split("|").list.first())
             for c in sorted(KURRENT_ONLY & set(corpora))}
    fixed = {name: tfz.Costs.pretrained(name) for name in ["general", "fraktur"]}
    rows = []
    for src in sorted({s for d in bench.values() for s in d["source"].unique()}):
        train = {c: d.filter(pl.col("source") != src).drop("source") for c, d in bench.items()}
        train = {c: d for c, d in train.items() if d.height}
        test = {src: pl.concat([d.filter(pl.col("source") == src).drop("source") for d in bench.values()])}
        sample, lexicon = test_set(test, src)
        if len(sample) < 100:
            continue
        tables = {"levenshtein": tfz.Costs(), **fixed, "kurrent": costs(learn(train, list(train)))}
        rows.append({"held_out": src.split(":")[-1], "n": len(sample), **retrieval(sample, lexicon, tables)})
        print(rows[-1], flush=True)
    print(pl.DataFrame(rows))
elif "--prior" not in sys.argv:
    rows = []
    for c in sorted(set(corpora) - KURRENT_ONLY, key=lambda c: (c.startswith("icdar_"), c)):
        sample, lexicon = test_set(corpora, c)
        tables = {"levenshtein": tfz.Costs(), "general": table_without(corpora, c)}
        script = SCRIPT_OF.get(c)
        if script:
            tables[script] = table_without(corpora, c, script)
        r = retrieval(sample, lexicon, tables)
        rows.append({"held_out": c, "n": len(sample), "levenshtein": r["levenshtein"], "general": r["general"],
                     "script": script, "script_table": r.get(script)})
        print(rows[-1], flush=True)
    print(pl.DataFrame(rows))
else:
    # Samples of the corpus's own training data stand in for a user's corrections (Trove for trove_clean)
    for c, own in [("munich_directories", "munich_directories"), ("trove_clean", "trove")]:
        if own not in corpora:
            continue
        prior = costs(learn(corpora, held_out(corpora, c)))
        sample, lexicon = test_set(corpora, c)
        d = corpora[own]
        if c == "munich_directories":
            d = d.filter(pl.col("doc").str.ends_with("_last") & ~pl.col("doc").is_in(munich_test_docs(d)))
        pool = [(g, o) for g, o, n in d.select("gt", "ocr", "n").sort("gt", "ocr").rows() for _ in range(n)]
        tables = {"levenshtein": tfz.Costs(), "general": prior}
        for k in [300, 3000, 30000]:
            small = Counter(random.Random(k).sample(pool, k))
            mine = {own: pl.DataFrame([(None, g, o, n) for (g, o), n in small.items()],
                                      schema=["doc", "gt", "ocr", "n"], orient="row")}
            tables[f"own_{k}"] = costs(learn(mine, [own], weight=False))
            tables[f"prior_{k}"] = costs(learn(mine, [own], prior=prior, weight=False))
        print(c, retrieval(sample, lexicon, tables), flush=True)
