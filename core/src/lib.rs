//! Weighted edit distance with multi-character edits, exact nearest-name search under it,
//! and edit costs learned from (truth, reading) pairs.
//!
//! ```
//! use tunedfuzz::{Costs, Index, distance};
//!
//! let costs = Costs::new([("R", "N", 0.22), ("tz", "ß", 0.23)], 1.0).unwrap();
//! assert_eq!(distance("Nußbaum", "Rutzbaum", &costs, None), 0.22 + 0.23);
//!
//! let ix = Index::new(vec!["Huber".into(), "Nußbaum".into()], &costs);
//! assert_eq!(ix.extract_one("Rutzbaum", Some(1.0), false).unwrap().index, 1);
//! ```

#![warn(missing_docs)]

mod costs;
mod difflib;
mod distance;
mod index;
mod learn;

pub use costs::{Costs, PRETRAINED};
pub use distance::{align, distance, normalized_distance};
pub use index::{Index, Match, cdist, cpdist};
pub use learn::{LearnOptions, LearnedEdit, learn_costs};

/// An invalid cost table, CSV file or learning option, with a message saying which.
#[derive(Clone, Debug, PartialEq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Runs `f` on a pool of `workers` threads, or on all cores for 0.
pub(crate) fn with_workers<R: Send>(workers: usize, f: impl FnOnce() -> R + Send) -> R {
    if workers == 0 {
        return f();
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .expect("thread pool")
        .install(f)
}
