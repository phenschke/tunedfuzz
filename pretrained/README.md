# Building the pretrained tables

These scripts rebuild the tables in `core/data/` and the evaluation in the main README. Two builds from the same pairs give identical tables, byte for byte.

## Steps

```bash
cd pretrained && mkdir -p pairs
./download.sh downloads
```

```bash
uv run --with polars --with ../python python pairs_icdar.py downloads/full_22M_without_Finnish pairs/icdar.parquet
```

```bash
uv run --with polars --with ../python python pairs_overproof.py downloads/overproof pairs/overproof.parquet
```

```bash
uv run --with polars --with ../python python pairs_bln600.py downloads/bln600.csv pairs/bln600.parquet
```

```bash
POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage uv run --with polars --with ../python python pairs_bhl.py downloads/bhl pairs/bhl.parquet
```

```bash
uv run --with polars --with ../python python pairs_rise.py downloads/rise pairs/rise.parquet
```

```bash
PYTHONPATH=../../kurrent-vlm/src uv run --no-project --with polars --with rapidfuzz python pairs_kurrent_bench.py pairs/kurrent_bench.parquet
```

```bash
uv run --with polars --with ../python python build.py pairs ../core/data
```

```bash
uv run --with polars --with numpy --with rapidfuzz --with ../python python evaluate.py pairs
```

```bash
uv run --with polars --with numpy --with rapidfuzz --with ../python python evaluate.py pairs --prior
```

```bash
uv run --with polars --with numpy --with rapidfuzz --with ../python python evaluate.py pairs --kurrent
```

`build.py pairs ../core/data kurrent` rebuilds only the named table.

`--with ../python` builds the Python package from source, which needs Rust. A built wheel works too.

## Files

| File | Does |
|---|---|
| `download.sh` | Fetches ICDAR 2019, Overproof, BLN600, BHL and RISE |
| `common.py` | Splits aligned or diffed texts into (truth, reading) word pairs |
| `pairs_*.py` | One corpus each, written as (corpus, doc, gt, ocr, n) |
| `tables.py` | Test-only corpora, the type of each corpus, corpus weighting, seeds and the call to `learn_costs` |
| `pairs_kurrent_bench.py` | Reads six models' runs on the kurrent-vlm benchmark through its scorer, one corpus per model |
| `build.py` | Learns `general` from every training corpus but the benchmark ones, and `antiqua`, `fraktur` and `kurrent` from the corpora in that type |
| `evaluate.py` | Leave-one-corpus-out retrieval for `general` and the type tables, small samples learned with `general` as prior, and leave-one-source-out retrieval for `kurrent` |

## Corpora

The Munich address books and civil registers are private. The shipped tables include them, but their pairs and the script that builds those pairs are not published. Without them, `build.py` learns from the public corpora alone, and `evaluate.py` skips the Munich tests.

The kurrent-vlm benchmark lives on the storagebox and `/tank/data/kurrent-vlm`, so
`pairs_kurrent_bench.py` runs only where those are mounted. Its pages include the private Munich
certificates, so its pairs are not published either.

The test corpora are BLN600, the clean Overproof set, Chronicling America, BHL and RISE.
`build.py` never trains on them.

- BHL has Tesseract, Kraken and 19 VLMs, pooled, on natural-history books in Antiqua type. The
  three Datalab models are left out because their licence restricts use.
- The BHL outputs bucket has no versions, so a later download may hold more systems.
- RISE is pinned to the dataset release `2026-10-29.6`.
- BHL page outputs and RISE fields are paired with a word diff, and readings that keep a
  line-break hyphen are dropped.
