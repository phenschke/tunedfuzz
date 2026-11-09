"""Corpora, weighting and learning shared by build.py and evaluate.py."""
import glob

import polars as pl
import tunedfuzz as tfz

# Corpora used only for testing. BLN600 is CC BY-NC, trove_clean and chronam re-correct or resemble
# Trove, and BHL and RISE test engines and documents the training corpora do not cover.
TEST_ONLY = {"BLN600", "trove_clean", "chronam", "bhl_tesseract", "bhl_kraken", "bhl_vlm",
             "rise_fraktur", "rise_medieval"}

# Models whose kurrent-vlm bench readings train the kurrent table. Weaker models are left out. The
# bench pages are the eval set of another project, so they train only the kurrent table, as does the
# private corpus kurrent_extra.
KBENCH_MODELS = ["argon4", "fable51", "opus55", "flash38", "titan1", "titan2"]
KURRENT_ONLY = {"kurrent_extra", *(f"kbench_{m}" for m in KBENCH_MODELS)}

# The type each corpus is printed in, judged from its scans and misreadings. Fraktur readings turn k
# into f and V into B, which Antiqua readings almost never do. Cyrillic and the subsets in older
# blackletter or of unclear type train only the general table.
SCRIPTS = {
    "antiqua": ["icdar_DE4", "icdar_EN1", "icdar_ES1", "icdar_FR1", "icdar_FR2", "icdar_FR3", "icdar_SL1",
                "trove", "BLN600", "trove_clean", "chronam", "bhl_tesseract", "bhl_kraken", "bhl_vlm"],
    "fraktur": ["munich_directories", "icdar_CZ1", "icdar_DE1", "icdar_DE2", "icdar_DE3", "icdar_DE7",
                "rise_fraktur"],
    "kurrent": sorted(KURRENT_ONLY),
}
SCRIPT_OF = {c: s for s, cs in SCRIPTS.items() for c in cs}

# Words each corpus is weighted to, so the large ones do not drown the small ones
WEIGHT_TO = 100_000

SEEDS = [("m", "rn"), ("d", "cl"), ("h", "li"), ("w", "vv"), ("ck", "k"), ("tz", "z")]


def load(pairs_dir: str) -> dict[str, pl.DataFrame]:
    """Reads every parquet in pairs_dir into one (doc, gt, ocr, n) frame per corpus."""
    p = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{pairs_dir}/*.parquet"))])
    return {c: d.drop("corpus") for (c,), d in p.group_by("corpus")}


def learn(corpora: dict[str, pl.DataFrame], names: list[str], prior=None, prior_weight=30.0, weight=True):
    """Learns a directed table from the named corpora, each weighted to WEIGHT_TO words unless weight is false."""
    frames = []
    for c in sorted(names):
        d = corpora[c].group_by("gt", "ocr").agg(pl.col("n").sum())
        w = WEIGHT_TO / d["n"].sum() if weight else 1.0
        frames.append(d.with_columns(n=(pl.col("n") * w).round().cast(pl.Int64).clip(lower_bound=1)))
    train = pl.concat(frames).group_by("gt", "ocr").agg(pl.col("n").sum()).sort("gt", "ocr").rows()
    letters = {ch for gt, ocr, _ in train if gt != ocr for ch in gt + ocr}
    seeds = [(ch, ch * 2) for ch in sorted(letters) if ch.islower()] + SEEDS
    return tfz.learn_costs(train, seeds=seeds, symmetric=False, prior=prior, prior_weight=prior_weight)


def costs(table: dict) -> tfz.Costs:
    return tfz.Costs(zip(table["a"], table["b"], table["cost"]), symmetric=False)
