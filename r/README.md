# tunedfuzz

Basically [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz), but with adjustable edit weights.

tunedfuzz matches strings that OCR, handwriting recognition or a person has misread. Each edit has its own cost, so a likely misreading costs little and an unlikely one costs a lot. The core is written in Rust, with Python and R packages.

OCR read the Munich surname Rötzer as Nößer. Levenshtein ranks Nußer, one edit away, above Rötzer, three edits away. The pretrained table charges little for R read as N and tz read as ß, so Rötzer wins.

```r
library(tunedfuzz)

lev <- tfz_costs()                                     # no rows, so plain Levenshtein
ocr <- tfz_pretrained_costs()                          # learned from 17M words of OCR and HTR output

tfz_distance(c("Rötzer", "Nußer"), "Nößer", lev)       # 3 1
tfz_distance(c("Rötzer", "Nußer"), "Nößer", ocr)       # 0.53 1
```

## Install

tunedfuzz is not on CRAN or R-universe yet, so it installs from GitHub.

```r
remotes::install_github("phenschke/tunedfuzz", subdir = "r")
```

The install compiles the Rust core, which needs Cargo and rustc 1.85 or newer (<https://rustup.rs>).

## Usage

```r
library(tunedfuzz)

costs <- tfz_costs(data.frame(a = c("R", "tz"), b = c("N", "ß"), cost = c(0.22, 0.23)))
costs <- tfz_read_costs("costs.csv")
tfz_write_costs(costs, "costs.csv")

tfz_distance(c("Nußbaum", "Huber"), "Rutzbaum", costs)   # vectorized and recycled
tfz_distance("Nußbaum", "Rutzbaum", costs, normalize = TRUE)
tfz_align("Nußbaum", "Rutzbaum", costs)                  # a data frame of edits

ix <- tfz_index(surnames, costs)
tfz_extract_one(ix, queries, score_cutoff = 2)           # one row per query: index, name, cost
tfz_extract(ix, queries, limit = 5, workers = -1)        # one row per match: query, index, name, cost
tfz_cdist(queries, surnames, costs)                      # a matrix
```

Tables and indexes can be saved with `saveRDS()`. An index rebuilds its search tree on first use after `readRDS()`.

## Documentation

The [full README](https://github.com/phenschke/tunedfuzz#readme) also covers the Python package and the Rust crate.

- [Cost tables](https://github.com/phenschke/tunedfuzz#cost-tables), the CSV format and directed tables
- [Pretrained tables](https://github.com/phenschke/tunedfuzz#pretrained-tables), which one to use and how they score on held-out corpora
- [Learning costs](https://github.com/phenschke/tunedfuzz#learning-costs), with or without a pretrained table as the prior
- [How the search works](https://github.com/phenschke/tunedfuzz#how-the-search-works)

## License

MIT. The pretrained tables are learned partly from CC-BY data, credited in [core/data/README.md](https://github.com/phenschke/tunedfuzz/blob/main/core/data/README.md).
