"""Helpers that cut texts into (truth, reading) word pairs for the pairs_*.py scripts."""
import difflib
import re
import unicodedata

import polars as pl

LETTER = re.compile(r"[^\W\d_]")
WORD = re.compile(r"^[^\W\d_]+$")
EDGE = re.compile(r"^[\W_]+|[\W_]+$")
SCHEMA = ["corpus", "doc", "gt", "ocr"]


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def aligned_words(ocr: str, gt: str):
    """Yields (truth, reading) word pairs of two character-aligned strings, where @ marks a gap and # an unalignable span."""
    start = 0
    for k in range(len(gt) + 1):
        if k < len(gt) and not (gt[k] == " " and ocr[k] == " "):
            continue
        o, g = ocr[start:k], gt[start:k]
        start = k + 1

        # Trim edge columns whose truth is not a letter

        i, j = 0, len(g)
        while i < j and not LETTER.match(g[i]):
            i += 1
        while j > i and not LETTER.match(g[j - 1]):
            j -= 1
        o, g = o[i:j], g[i:j]
        if "#" in o or "#" in g:
            continue
        o, g = o.replace("@", ""), unicodedata.normalize("NFKC", g.replace("@", ""))

        # Some ICDAR truths write long-s ligatures as "ff"

        if "ff" in g and "ff" not in o:
            continue
        if WORD.match(g) and o and " " not in o:
            yield g, o


def diffed_words(ocr: str, gt: str):
    """Yields (truth, reading) word pairs of two unaligned texts, pairing words inside equal-length blocks of a word diff."""
    ot, gt_ = ocr.split(), gt.split()
    sm = difflib.SequenceMatcher(None, ot, gt_, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag not in ("equal", "replace") or i2 - i1 != j2 - j1:
            continue
        for o, g in zip(ot[i1:i2], gt_[j1:j2]):
            o, g = EDGE.sub("", o), EDGE.sub("", unicodedata.normalize("NFKC", g))
            if WORD.match(g) and o and (tag == "equal" or ratio(o, g) >= 0.5):
                yield g, o


def write(rows: list[tuple[str, str, str, str]], path: str) -> None:
    """Counts duplicate rows, writes (corpus, doc, gt, ocr, n) and prints a summary per corpus."""
    df = (pl.DataFrame(rows, schema=SCHEMA, orient="row")
          .group_by(SCHEMA).len("n").sort(SCHEMA))
    print(df.group_by("corpus").agg(
        words=pl.col("n").sum(), docs=pl.col("doc").n_unique(),
        misread=(pl.col("n") * (pl.col("gt") != pl.col("ocr"))).sum() / pl.col("n").sum(),
    ).sort("corpus"))
    df.write_parquet(path)
