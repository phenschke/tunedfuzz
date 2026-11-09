# tunedfuzz

Basically [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz), but with adjustable edit weights.

tunedfuzz matches strings that OCR, handwriting recognition or a person has misread. Each edit has its own cost, so a likely misreading costs little and an unlikely one costs a lot. The core is written in Rust, with Python and R packages.

## Why

Levenshtein distance charges 1 for every edit, but misreadings follow patterns. In Fraktur type, ℜ and 𝔑 differ by one stroke, so OCR often reads R as N and almost never reads R as a. Some misreadings span several letters, such as tz read as ß or rn read as m, and Levenshtein charges 2 for each of those.

Equal costs pick the wrong match. OCR read the Munich surname Rötzer as Nößer. Levenshtein ranks Nußer, one edit away, above Rötzer, three edits away. The pretrained table charges 0.31 for R read as N and 0.22 for tz read as ß, so Rötzer wins.

```python
import tunedfuzz as tfz

lev = tfz.Costs()                       # no rows, so plain Levenshtein
ocr = tfz.Costs.pretrained()            # learned from 17M words of OCR and HTR output

tfz.distance("Rötzer", "Nößer", lev)    # 3.0
tfz.distance("Nußer", "Nößer", lev)     # 1.0
tfz.distance("Rötzer", "Nößer", ocr)    # 0.53, R read as N and tz read as ß
tfz.distance("Nußer", "Nößer", ocr)     # 1.0
```

The library has four parts.

- A weighted edit distance. An edit can replace several characters (`tz` by `ß`), and its cost can differ by direction.
- An exact nearest-match search. On 110,000 surnames, a search takes about 13 ms, 26 times faster than computing every distance.
- A learner that estimates costs from pairs of true strings and their readings.
- Four pretrained tables, learned from OCR and HTR output in nine languages and Latin.

## Install

tunedfuzz is not on PyPI, CRAN or crates.io yet, so each package installs from GitHub.

```bash
uv add "tunedfuzz @ git+https://github.com/phenschke/tunedfuzz#subdirectory=python"
```

```r
remotes::install_github("phenschke/tunedfuzz", subdir = "r")
```

```bash
cargo add tunedfuzz --git https://github.com/phenschke/tunedfuzz
```

Each install compiles the Rust core, which needs Cargo and rustc 1.85 or newer (<https://rustup.rs>).

## Python

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

## R

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

## Rust

```rust
use tunedfuzz::{Costs, Index, distance};

let costs = Costs::new([("R", "N", 0.22), ("tz", "ß", 0.23)], 1.0)?;
distance("Nußbaum", "Rutzbaum", &costs, None);                   // 0.45

let ix = Index::new(surnames, &costs);
ix.extract_one("Rutzbaum", Some(2.0), false);                     // Some(Match { index, cost })
ix.extract_many(&queries, Some(5), None, false, 0);               // 0 uses all cores

let general = Costs::pretrained("general")?;
```

The [API documentation](https://docs.rs/tunedfuzz) covers `learn_costs`, `align`, `cdist` and the rest.

## Cost tables

A table is a list of rows `a,b,cost`.

| Row | Edit |
|---|---|
| `R,N,0.22` | Substitute one character |
| `n,,0.6` | Delete or insert one character |
| `tz,ß,0.23` | Replace one string with another, each at least one character long |

- A symmetric table, the default, applies each row in both directions.
- A directed table (`symmetric=False`) only replaces `a` in the first string with `b` in the second. So `n,,0.6` deletes an n and `,n,0.6` inserts one.
- `reversed()` swaps `a` and `b` in every row of a directed table.
- An edit missing from the table costs `default`, which is 1, as in Levenshtein. A table with no rows gives plain Levenshtein distance.
- When two rows describe the same edit, the later one wins.
- Costs apply to Unicode code points, are case-sensitive and can't be negative.
- A normalized distance divides by the length of the longer string. It stays between 0 and 1 as long as no edit costs more than 1.
- A CSV file holds only the rows, so pass `read_csv` the same `symmetric` the table was built with.

```python
directed = tfz.Costs([("R", "N", 0.22), ("N", "R", 0.6)], symmetric=False)
tfz.distance("Rauch", "Nauch", directed)                 # 0.22, R read as N
tfz.distance("Nauch", "Rauch", directed.reversed())      # 0.22, the same edit turned around
```

The distance is not a metric. With the pretrained table, K→N costs 0.75, but K→R and R→N together cost 0.60.

## Pretrained tables

`Costs.pretrained()` returns `general`, a directed table learned from 17M words of truth and reading. The text covers printed books, newspapers and handwritten registers from the 15th to the 20th century, in nine languages and Latin. ABBYY FineReader, Gemini, Transkribus and other engines read it. [core/data/README.md](core/data/README.md) lists the sources and the build steps.

The tables edit truth into reading. So pass the truth first to `distance`, and reverse the table for an index of true names.

```python
general = tfz.Costs.pretrained()
tfz.distance(truth, reading, general)
tfz.Index(truths, general.reversed()).extract_one(reading)
```

```r
general <- tfz_pretrained_costs()
tfz_distance(truth, reading, general)
tfz_index(truths, tfz_reverse_costs(general))
```

### Which table to use

| Text | Table |
|---|---|
| 19th- and 20th-century print in Fraktur | `fraktur` |
| 19th- and 20th-century print in Antiqua (roman type) | `antiqua` |
| German handwriting in Kurrent | `kurrent` |
| Older print, mixed type, other handwriting | `general` |

OCR fails differently on the two types. It reads Fraktur V as B and k as f, and it almost never does so in Antiqua. `general` learns from both, so it charges only 0.18 for V read as B, even for names in roman type. `antiqua` charges 0.67. Older Antiqua sets the long s, which OCR reads as f as often as it does in Fraktur, so `general` handles it better.

If you have corrected readings of your own, learn a table from them with a pretrained table as the prior ([Learning with a prior](#learning-with-a-prior)).

### Evaluation

Each test holds out one corpus. The table is learned from the other corpora and scored on the held-out one. For each misread word, the candidates are the most similar words in that corpus's lexicon. Top-1 is the share of misread words whose cheapest candidate is the truth. [pretrained/evaluate.py](pretrained/evaluate.py) reruns every test except the two on private Munich data.

| Held-out corpus | Text | Levenshtein | `general` |
|---|---|---|---|
| Munich address books | Fraktur surnames, Gemini | 0.49 | 0.67 |
| Munich civil registers | Kurrent handwriting, Transkribus | 0.79 | 0.83 |
| Sydney Morning Herald | English newspapers, ABBYY | 0.61 | 0.73 |
| Chronicling America | US newspapers | 0.54 | 0.73 |
| BLN600 | London newspapers | 0.63 | 0.69 |
| IMPACT-BHL | Natural-history books in Antiqua, Tesseract 5 | 0.68 | 0.82 |
| IMPACT-BHL | The same pages, Kraken | 0.67 | 0.74 |
| IMPACT-BHL | The same pages, 19 VLMs | 0.62 | 0.69 |
| RISE Basler Avisblatt | Fraktur adverts, about 100 LLMs | 0.71 | 0.80 |
| RISE Rot accounts | 15th-century handwriting, about 100 LLMs | 0.60 | 0.69 |
| ICDAR 2019, 17 subsets (mean) | Books and newspapers, nine languages and Latin | 0.54 | 0.73 |

`general` beats Levenshtein on every held-out corpus except Bulgarian, the only Cyrillic one, where both score 0.71. The table has seen few Cyrillic edits.

| Held-out corpus | Type | `general` | Type table |
|---|---|---|---|
| Munich address books | Fraktur | 0.67 | 0.74 |
| RISE Basler Avisblatt | Fraktur | 0.80 | 0.80 |
| ICDAR 2019, 5 Fraktur subsets (mean) | Fraktur | 0.79 | 0.79 |
| Sydney Morning Herald | Antiqua, 19th–20th c. | 0.73 | 0.74 |
| Chronicling America | Antiqua, 19th–20th c. | 0.73 | 0.76 |
| BLN600 | Antiqua, 19th c. | 0.69 | 0.70 |
| IMPACT-BHL, Tesseract 5, Kraken and VLMs (mean) | Antiqua, 1708–1913 | 0.75 | 0.74 |
| ICDAR 2019, 5 Antiqua subsets (mean) | Antiqua | 0.66 | 0.65 |
| ICDAR 2019 DE4 and EN1 (mean) | Antiqua with long s, 16th–18th c. | 0.79 | 0.71 |
| kurrent-vlm benchmark, 21 sources (mean) | Kurrent handwriting, Gemini, Claude and Transkribus | 0.68 | 0.71 |

The `kurrent` test holds out one benchmark source at a time, so no page of the held-out source trains the table. `kurrent` beats or ties `general` on all 21 sources. Levenshtein scores 0.60.

## Learning costs

`learn_costs` takes `(truth, reading, count)` triples and repeats two steps for a fixed number of rounds. First it aligns every pair under the current costs and counts each edit. Then it sets the cost of an edit seen `c` times with the formulas below, where `n` counts how often the edited strings occur in all pairs.

```text
p    = max(c - shrink * sqrt(c), 0) / n
cost = min(1, log p / log p_floor), rounded to 4 decimals
```

A common edit costs little, and an edit at `p_floor` or rarer costs 1. `shrink` takes standard errors off each count, so an edit seen a few times by chance stays expensive.

```python
table = tfz.learn_costs(
    pairs,                                           # (truth, reading, count)
    seeds=[("ll", "l"), ("ck", "k"), ("tz", "z")],   # multi-character edits to consider
    fixed=[("ae", "ä", 0.1)],                        # costs set by hand
)
costs = tfz.Costs(zip(table["a"], table["b"], table["cost"]))
```

```r
table <- tfz_learn_costs(truth, reading, count, seeds = data.frame(a = "ll", b = "l"))
costs <- tfz_costs(table)
```

The result has columns `a`, `b`, `count`, `p` and `cost`. It lists the learned edits that cost less than 1, sorted by cost, and then the fixed edits.

| Parameter | Default | Meaning |
|---|---|---|
| `rounds` | 4 | Rounds of aligning and counting |
| `p_floor` | 1e-5 | Edits this rare or rarer cost 1 |
| `min_multi` | 20 | How often a 2:1, 1:2 or 2:2 replacement must occur in the word diffs to become a candidate edit |
| `min_ratio` | 0.5 | Pairs below this difflib ratio count as misaligned and are dropped |
| `shrink` | 2 | Standard errors taken off each count |
| `symmetric` | true | One cost for an edit and its reverse, or one per direction |
| `prior` | none | A table to start from and shrink towards |
| `prior_weight` | 30 | How many occurrences of each string the prior counts as |

### Direction

Misreadings are rarely symmetric. In Munich directory surnames, OCR reads `tz` as `ß` 27 times as often as the reverse, and `V` as `B` 17 times as often. With `symmetric=False`, each direction gets its own cost.

- An edit `(a, b)` replaces `a` in the truth with `b` in the reading.
- `n` counts how often `a` occurs in the truths. For an insertion, it counts the places the insertion could go.
- Score with `distance(truth, reading)`. For an index of true names, use `Index(truths, costs.reversed())`.

### Learning with a prior

With a `prior`, each probability shrinks towards the prior's `p0 = p_floor ** cost`.

```text
p = (max(c - shrink * sqrt(c), 0) + prior_weight * p0) / (n + prior_weight)
```

Where your data has few examples of an edit, the prior's cost wins. Where it has many, your counts win. The prior's costs also align the pairs in the first round.

```python
table = tfz.learn_costs(pairs, symmetric=False, prior=tfz.Costs.pretrained())
```

The table below scores Munich surnames, with `general` learned without the Munich data. A few thousand corrected words with `general` as the prior beat both `general` alone and a table learned from those words alone.

| Munich words | Own only | `general` alone | `general` as prior |
|---|---|---|---|
| 300 | 0.49 | 0.67 | 0.71 |
| 3,000 | 0.66 | 0.67 | 0.77 |
| 30,000 | 0.79 | 0.67 | 0.82 |

## How the search works

The distance is not a metric, so metric indexes such as BK-trees would miss matches. `Index` stores the names in a trie and walks it, filling one row of the edit-distance matrix per trie level.

One edit covers at most as many characters of a name as the longest string in the table. So the smallest entry in that many recent rows is a lower bound on the cost of every name below the current node. The walk skips each subtree whose bound exceeds the best cost found so far. Nothing is approximated, so the results always equal a full scan.

## Development

Code was developed using Claude Opus 5.5 in Claude Code.

## License

MIT. The pretrained tables are learned partly from CC-BY data, credited in [core/data/README.md](core/data/README.md).
