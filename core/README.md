# tunedfuzz

Basically [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz), but with adjustable edit weights.

tunedfuzz matches strings that OCR, handwriting recognition or a person has misread. Each edit has its own cost, so a likely misreading costs little and an unlikely one costs a lot. The core is written in Rust, with Python and R packages.

OCR read the Munich surname Rötzer as Nößer. Levenshtein ranks Nußer, one edit away, above Rötzer, three edits away. The pretrained table charges little for R read as N and tz read as ß, so Rötzer wins.

```rust
use tunedfuzz::{Costs, distance};

let lev = Costs::default();                  // no rows, so plain Levenshtein
let ocr = Costs::pretrained("general")?;     // learned from 17M words of OCR and HTR output

distance("Rötzer", "Nößer", &lev, None);     // 3.0
distance("Nußer", "Nößer", &lev, None);      // 1.0
distance("Rötzer", "Nößer", &ocr, None);     // 0.53, R read as N and tz read as ß
distance("Nußer", "Nößer", &ocr, None);      // 1.0
```

## Install

tunedfuzz is not on crates.io yet, so it installs from GitHub.

```bash
cargo add tunedfuzz --git https://github.com/phenschke/tunedfuzz
```

The crate needs Rust 1.85 or newer. Its only dependencies are `csv` and `rayon`.

## Usage

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

## Documentation

The [full README](https://github.com/phenschke/tunedfuzz#readme) also covers the Python and R packages.

- [Cost tables](https://github.com/phenschke/tunedfuzz#cost-tables), the CSV format and directed tables
- [Pretrained tables](https://github.com/phenschke/tunedfuzz#pretrained-tables), which one to use and how they score on held-out corpora
- [Learning costs](https://github.com/phenschke/tunedfuzz#learning-costs), with or without a pretrained table as the prior
- [How the search works](https://github.com/phenschke/tunedfuzz#how-the-search-works)

## License

MIT. The pretrained tables are learned partly from CC-BY data, credited in [core/data/README.md](https://github.com/phenschke/tunedfuzz/blob/main/core/data/README.md).
