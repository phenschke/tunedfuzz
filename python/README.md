# tunedfuzz

Basically [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz), but with adjustable edit weights.

tunedfuzz matches strings that OCR, handwriting recognition or a person has misread. Each edit has its own cost, so a likely misreading costs little and an unlikely one costs a lot. The core is written in Rust, with Python and R packages.

OCR read the Munich surname Rötzer as Nößer. Levenshtein ranks Nußer, one edit away, above Rötzer, three edits away. The pretrained table charges little for R read as N and tz read as ß, so Rötzer wins.

```python
import tunedfuzz as tfz

lev = tfz.Costs()                       # no rows, so plain Levenshtein
ocr = tfz.Costs.pretrained()            # learned from 17M words of OCR and HTR output

tfz.distance("Rötzer", "Nößer", lev)    # 3.0
tfz.distance("Nußer", "Nößer", lev)     # 1.0
tfz.distance("Rötzer", "Nößer", ocr)    # 0.53, R read as N and tz read as ß
tfz.distance("Nußer", "Nößer", ocr)     # 1.0
```

## Install

tunedfuzz is not on PyPI yet, so it installs from GitHub.

```bash
uv add "tunedfuzz @ git+https://github.com/phenschke/tunedfuzz#subdirectory=python"
```

The install compiles the Rust core, which needs Cargo and rustc 1.85 or newer (<https://rustup.rs>).

## Usage

```python
import tunedfuzz as tfz

# A table from (a, b, cost) rows, or from a CSV file with columns a, b, cost
costs = tfz.Costs([("R", "N", 0.22), ("tz", "ß", 0.23)])
costs = tfz.Costs.read_csv("costs.csv")
costs.to_csv("costs.csv")

# One pair of strings
tfz.distance("Nußbaum", "Rutzbaum", costs)                   # 0.45
tfz.distance("Nußbaum", "Rutzbaum", costs, normalize=True)   # 0.056
tfz.distance("Huber", "Hauser", costs, score_cutoff=1.0)     # inf, above the cutoff
tfz.align("Nußbaum", "Rutzbaum", costs)                      # [("N", "R", 0.22), ("ß", "tz", 0.23)]

# Search a list of names
ix = tfz.Index(surnames, costs)
ix.extract_one("Rutzbaum", score_cutoff=2.0)                 # ("Nußbaum", 0.45, 17), or None
ix.extract("Rutzbaum", limit=5)                              # the 5 closest as (name, cost, index)
ix.extract("Rutzbaum", limit=None, score_cutoff=1.0)         # every name within 1.0
ix.extract_one_many(queries, workers=-1)                     # many queries on all cores

# Many pairs at once, as numpy arrays
tfz.cdist(queries, surnames, costs)                          # one row per query
tfz.cpdist(df["truth"], df["ocr"], costs)                    # one distance per row
```

- The search follows [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz), but it returns costs, so lower is closer. rapidfuzz returns similarities from 0 to 100.
- Results are sorted by cost, then by index.
- With `normalize=True`, `score_cutoff` applies to the normalized distance.
- `workers=-1` uses all cores in `cdist`, `cpdist` and the `*_many` methods.
- `processor` transforms every string before matching, as in rapidfuzz. `Index(names, costs, processor=str.lower)` ignores case and still returns the names as passed.
- `cdist` and `cpdist` need numpy. Nothing else does.
- `Costs` and `Index` pickle, so they work with multiprocessing and joblib. Searches release the GIL.

## Documentation

The [full README](https://github.com/phenschke/tunedfuzz#readme) also covers the R package and the Rust crate.

- [Cost tables](https://github.com/phenschke/tunedfuzz#cost-tables), the CSV format and directed tables
- [Pretrained tables](https://github.com/phenschke/tunedfuzz#pretrained-tables), which one to use and how they score on held-out corpora
- [Learning costs](https://github.com/phenschke/tunedfuzz#learning-costs), with or without a pretrained table as the prior
- [How the search works](https://github.com/phenschke/tunedfuzz#how-the-search-works)

## License

MIT. The pretrained tables are learned partly from CC-BY data, credited in [core/data/README.md](https://github.com/phenschke/tunedfuzz/blob/main/core/data/README.md).
