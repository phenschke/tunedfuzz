use std::collections::HashMap;
use std::path::Path;

use crate::Error;

/// Names of the tables `Costs::pretrained` returns.
pub const PRETRAINED: &[&str] = &["general", "antiqua", "fraktur"];

const PRETRAINED_CSV: &[(&str, &str)] = &[
    ("general", include_str!("../data/general.csv")),
    ("antiqua", include_str!("../data/antiqua.csv")),
    ("fraktur", include_str!("../data/fraktur.csv")),
];

/// One direction `u -> v` of an edit with more than one character on a side.
#[derive(Clone, Debug)]
pub(crate) struct Op {
    pub u: Vec<char>,
    pub v: Vec<char>,
    pub cost: f64,

    // Symbol of v's last character
    pub v_sym: usize,
}

/// Edit costs over Unicode code points. Edits missing from the table cost `default`.
///
/// Each row `(a, b, cost)` is one edit:
///
/// - `("R", "N", 0.22)` substitutes one character for another
/// - `("n", "", 0.6)` deletes one character, and `("", "n", 0.6)` inserts it
/// - `("tz", "ß", 0.23)` replaces one string by another, both sides at least one character long
///
/// `Costs::new` makes a symmetric table, where each row also gives the reverse edit, so
/// `("n", "", 0.6)` deletes and inserts. `Costs::directed` makes a directed table, where a row
/// only replaces `a` in the first string of `distance` by `b` in the second.
///
/// A later row for the same edit replaces an earlier one. Costs are case-sensitive.
#[derive(Clone, Debug)]
pub struct Costs {
    rows: Vec<(String, String, f64)>,
    default: f64,
    symmetric: bool,

    // Characters the table names, sorted. Symbol s >= 1 stands for syms[s - 1], symbol 0 for every other character.
    // latin[c] is the symbol of code point c below LATIN, to skip the binary search.
    syms: Vec<char>,
    latin: Vec<u32>,
    pub(crate) sub: Vec<f64>,
    pub(crate) del: Vec<f64>,
    pub(crate) ins: Vec<f64>,

    // Ops grouped by the symbol of u's last character, then by u's length, then by row
    pub(crate) ops: Vec<Op>,
    pub(crate) op_start: Vec<usize>,

    // Longest side of any edit, at least 1. A DP path jumps at most this many rows.
    pub(crate) max_len: usize,
}

impl Default for Costs {
    /// Returns an empty table with default 1, which gives plain Levenshtein distance.
    fn default() -> Self {
        Costs::new(Vec::<(String, String, f64)>::new(), 1.0).unwrap()
    }
}

impl Costs {
    /// Makes a symmetric table, where each row gives an edit and its reverse.
    pub fn new<I, A, B>(rows: I, default: f64) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (A, B, f64)>,
        A: Into<String>,
        B: Into<String>,
    {
        Costs::with_symmetry(rows, default, true)
    }

    /// Makes a directed table, where each row `(a, b, cost)` only replaces `a` in the first string
    /// by `b` in the second.
    pub fn directed<I, A, B>(rows: I, default: f64) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (A, B, f64)>,
        A: Into<String>,
        B: Into<String>,
    {
        Costs::with_symmetry(rows, default, false)
    }

    /// Makes a symmetric table as `Costs::new` does, or a directed one as `Costs::directed` does.
    pub fn with_symmetry<I, A, B>(rows: I, default: f64, symmetric: bool) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (A, B, f64)>,
        A: Into<String>,
        B: Into<String>,
    {
        check_cost(default)?;

        // Keep the last row of each edit (an unordered pair when symmetric), in the order the rows were last set

        let mut kept: Vec<Option<(String, String, f64)>> = Vec::new();
        let mut pos: HashMap<(String, String), usize> = HashMap::new();
        for (a, b, cost) in rows {
            let (a, b) = (a.into(), b.into());
            check_cost(cost)?;
            if a == b {
                continue;
            }
            if (a.is_empty() || b.is_empty()) && a.chars().count() + b.chars().count() > 1 {
                return Err(Error(format!("deletion of more than one character: ({a:?}, {b:?})")));
            }
            let key = if a < b || !symmetric {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            };
            if let Some(old) = pos.insert(key, kept.len()) {
                kept[old] = None;
            }
            kept.push(Some((a, b, cost)));
        }
        let rows: Vec<(String, String, f64)> = kept.into_iter().flatten().collect();

        let mut syms: Vec<char> = rows.iter().flat_map(|(a, b, _)| a.chars().chain(b.chars())).collect();
        syms.sort_unstable();
        syms.dedup();
        let ns = syms.len() + 1;
        let latin = (0..LATIN)
            .map(|c| {
                syms.binary_search(&char::from_u32(c).unwrap())
                    .map_or(0, |k| k as u32 + 1)
            })
            .collect();
        let mut costs = Costs {
            rows: Vec::new(),
            default,
            symmetric,
            syms,
            latin,
            sub: vec![default; ns * ns],
            del: vec![default; ns],
            ins: vec![default; ns],
            ops: Vec::new(),
            op_start: vec![0; ns + 1],
            max_len: 1,
        };

        for (a, b, cost) in &rows {
            let (u, v): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
            costs.max_len = costs.max_len.max(u.len()).max(v.len());
            if u.is_empty() || v.is_empty() {
                let x = costs.sym(*u.first().or(v.first()).unwrap());
                if symmetric || v.is_empty() {
                    costs.del[x] = *cost;
                }
                if symmetric || u.is_empty() {
                    costs.ins[x] = *cost;
                }
            } else if u.len() == 1 && v.len() == 1 {
                let (x, y) = (costs.sym(u[0]), costs.sym(v[0]));
                costs.sub[x * ns + y] = *cost;
                if symmetric {
                    costs.sub[y * ns + x] = *cost;
                }
            } else {
                let (us, vs) = (costs.sym(u[u.len() - 1]), costs.sym(v[v.len() - 1]));
                costs.ops.push(Op {
                    u: u.clone(),
                    v: v.clone(),
                    cost: *cost,
                    v_sym: vs,
                });
                if symmetric {
                    costs.ops.push(Op {
                        u: v,
                        v: u,
                        cost: *cost,
                        v_sym: us,
                    });
                }
            }
        }

        let last = |c: &Costs, op: &Op| c.sym(*op.u.last().unwrap());
        let mut ops = std::mem::take(&mut costs.ops);
        ops.sort_by_key(|op| (last(&costs, op), op.u.len()));
        let mut op_start = vec![0; ns + 1];
        for op in &ops {
            op_start[last(&costs, op) + 1] += 1;
        }
        for s in 0..ns {
            op_start[s + 1] += op_start[s];
        }
        costs.ops = ops;
        costs.op_start = op_start;
        costs.rows = rows;
        Ok(costs)
    }

    /// Reads rows from the `a`, `b` and `cost` columns of a CSV file with a header, ignoring other columns.
    /// The table is directed when `symmetric` is false, as in `Costs::directed`.
    pub fn read_csv(path: impl AsRef<Path>, default: f64, symmetric: bool) -> Result<Self, Error> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Costs::from_csv(file, default, symmetric).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))
    }

    /// Returns a directed table shipped with the library, learned from public (truth, reading) pairs.
    /// Its rows edit truth into reading. `PRETRAINED` lists the names, and `data/README.md` the sources.
    pub fn pretrained(name: &str) -> Result<Self, Error> {
        let (_, csv) = PRETRAINED_CSV.iter().find(|(n, _)| *n == name).ok_or_else(|| {
            Error(format!(
                "no pretrained table {name:?}; there are {}",
                PRETRAINED.join(", ")
            ))
        })?;
        Costs::from_csv(csv.as_bytes(), 1.0, false)
    }

    fn from_csv(source: impl std::io::Read, default: f64, symmetric: bool) -> Result<Self, Error> {
        let err = |e: &dyn std::fmt::Display| Error(e.to_string());
        let mut reader = csv::Reader::from_reader(source);
        let header = reader.headers().map_err(|e| err(&e))?.clone();
        let col = |name: &str| {
            header
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| err(&format!("no column {name:?}")))
        };
        let (a, b, cost) = (col("a")?, col("b")?, col("cost")?);
        let mut rows = Vec::new();
        for record in reader.records() {
            let r = record.map_err(|e| err(&e))?;
            let c: f64 = r[cost]
                .trim()
                .parse()
                .map_err(|_| err(&format!("bad cost {:?}", &r[cost])))?;
            rows.push((r[a].to_string(), r[b].to_string(), c));
        }
        Costs::with_symmetry(rows, default, symmetric)
    }

    /// Writes the rows to a CSV file with columns `a`, `b` and `cost`, which `read_csv` reads back.
    /// The file does not record `default` or whether the table is symmetric.
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        let err = |e: &dyn std::fmt::Display| Error(format!("{}: {e}", path.display()));
        let mut writer = csv::Writer::from_path(path).map_err(|e| err(&e))?;
        writer.write_record(["a", "b", "cost"]).map_err(|e| err(&e))?;
        for (a, b, cost) in &self.rows {
            writer
                .write_record([a.as_str(), b.as_str(), &cost.to_string()])
                .map_err(|e| err(&e))?;
        }
        writer.flush().map_err(|e| err(&e))
    }

    /// Returns the table with every row turned around, which edits the second string into the first.
    pub fn reversed(&self) -> Costs {
        if self.symmetric {
            return self.clone();
        }
        let rows = self.rows.iter().map(|(a, b, c)| (b.clone(), a.clone(), *c));
        Costs::with_symmetry(rows, self.default, false).expect("valid rows stay valid")
    }

    /// Whether each row also gives the reverse edit.
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }

    /// The rows in effect, one per edit, in the order they were last set.
    pub fn rows(&self) -> &[(String, String, f64)] {
        &self.rows
    }

    /// The cost of every edit the table lacks.
    pub fn default_cost(&self) -> f64 {
        self.default
    }

    pub(crate) fn nsym(&self) -> usize {
        self.syms.len() + 1
    }

    pub(crate) fn sym(&self, x: char) -> usize {
        match self.latin.get(x as usize) {
            Some(&s) => s as usize,
            None => self.syms.binary_search(&x).map_or(0, |k| k + 1),
        }
    }

    /// Ops whose `u` ends in the character with symbol `s`, shorter `u` first.
    pub(crate) fn ops_ending(&self, s: usize) -> &[Op] {
        &self.ops[self.op_start[s]..self.op_start[s + 1]]
    }
}

const LATIN: u32 = 0x250;

fn check_cost(c: f64) -> Result<(), Error> {
    if c >= 0.0 {
        Ok(())
    } else {
        Err(Error(format!("cost must be at least 0, got {c}")))
    }
}
