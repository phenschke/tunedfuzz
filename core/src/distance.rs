use crate::Costs;

const NONE: usize = usize::MAX;

/// The column string of a DP and what each row step needs of it.
pub(crate) struct Cols {
    pub chars: Vec<char>,
    syms: Vec<usize>,

    // ins[j] is the cost of inserting chars[j - 1]
    ins: Vec<f64>,

    // head[s] is the first position of symbol s in chars, next[p] the one after p, and NONE ends the list
    head: Vec<usize>,
    next: Vec<usize>,
}

impl Cols {
    pub fn new(b: &str, costs: &Costs) -> Self {
        let chars: Vec<char> = b.chars().collect();
        let syms: Vec<usize> = chars.iter().map(|&c| costs.sym(c)).collect();
        let ins = std::iter::once(0.0).chain(syms.iter().map(|&s| costs.ins[s])).collect();
        let mut head = vec![NONE; costs.nsym()];
        let mut next = vec![NONE; chars.len()];
        for (p, &s) in syms.iter().enumerate().rev() {
            next[p] = head[s];
            head[s] = p;
        }
        Cols {
            chars,
            syms,
            ins,
            head,
            next,
        }
    }

    pub fn len(&self) -> usize {
        self.chars.len()
    }

    /// Row 0 of the DP: inserting b[..j].
    pub fn row0(&self) -> Vec<f64> {
        let mut row = vec![0.0; self.len() + 1];
        for j in 1..row.len() {
            row[j] = row[j - 1] + self.ins[j];
        }
        row
    }
}

/// Computes row i = a.len() of the DP into `row`, where row[j] is the cheapest edit of a[..i] into b[..j].
/// `done` holds rows 0..i back to back, each `cols.len() + 1` long. `extra` is scratch space
/// that holds infinity in each of its `cols.len() + 1` cells between calls. Returns the row minimum.
///
/// Each candidate adds one cost to one earlier cell, so the result is bitwise the same
/// whatever order the candidates are compared in.
pub(crate) fn row_step(
    costs: &Costs,
    a: &[char],
    done: &[f64],
    cols: &Cols,
    extra: &mut Vec<f64>,
    row: &mut [f64],
) -> f64 {
    let (i, m) = (a.len(), cols.len());
    let w = m + 1;
    let x = a[i - 1];
    let xs = costs.sym(x);
    let ns = costs.nsym();
    let del = costs.del[xs];
    let sub = &costs.sub[xs * ns..(xs + 1) * ns];
    extra.resize(w, f64::INFINITY);

    // Multi-character edits start in earlier rows, so they go into `extra` first,
    // visiting only the columns where the last character of v occurs

    let mut any = false;
    for op in costs.ops_ending(xs) {
        let mut p = cols.head[op.v_sym];
        if p == NONE || !ends_with(&a[..i - 1], &op.u[..op.u.len() - 1]) {
            continue;
        }
        let (lv, from) = (op.v.len(), (i - op.u.len()) * w);
        while p != NONE {
            if ends_with(&cols.chars[..p], &op.v[..lv - 1]) {
                let j = p + 1;
                let t = done[from + j - lv] + op.cost;
                if t < extra[j] {
                    extra[j] = t;
                    any = true;
                }
            }
            p = cols.next[p];
        }
    }

    let (b, bs, ins) = (&cols.chars[..m], &cols.syms[..m], &cols.ins[..w]);
    let p1 = &done[(i - 1) * w..i * w];
    let row = &mut row[..w];
    let extra = &mut extra[..w];
    row[0] = p1[0] + del;
    let mut low = row[0];
    for j in 1..w {
        let mut best = if x == b[j - 1] {
            p1[j - 1]
        } else {
            p1[j - 1] + sub[bs[j - 1]]
        };
        let t = p1[j] + del;
        if t < best {
            best = t;
        }
        let t = row[j - 1] + ins[j];
        if t < best {
            best = t;
        }
        if any {
            if extra[j] < best {
                best = extra[j];
            }
            extra[j] = f64::INFINITY;
        }
        row[j] = best;
        if best < low {
            low = best;
        }
    }
    low
}

/// Whether s ends with t, written out so it inlines.
#[inline(always)]
fn ends_with(s: &[char], t: &[char]) -> bool {
    t.len() <= s.len() && s.iter().rev().zip(t.iter().rev()).all(|(x, y)| x == y)
}

/// Cheapest cost of editing `a` into `b`. Never above Levenshtein distance when no edit costs more
/// than 1, and symmetric when the table is.
///
/// With `score_cutoff`, returns infinity for distances above it, and stops as soon as every path is too expensive.
pub fn distance(a: &str, b: &str, costs: &Costs, score_cutoff: Option<f64>) -> f64 {
    if a == b {
        return 0.0;
    }
    let a: Vec<char> = a.chars().collect();
    distance_to(&a, &Cols::new(b, costs), costs, score_cutoff.unwrap_or(f64::INFINITY))
}

/// `distance(a, b)` divided by the length of the longer string, and 0 for two empty strings.
/// At most 1 when no edit costs more than 1. `score_cutoff` applies to the normalized value.
pub fn normalized_distance(a: &str, b: &str, costs: &Costs, score_cutoff: Option<f64>) -> f64 {
    if a == b {
        return 0.0;
    }
    let a: Vec<char> = a.chars().collect();
    normalized_to(&a, &Cols::new(b, costs), costs, score_cutoff.unwrap_or(f64::INFINITY))
}

pub(crate) fn normalized_to(a: &[char], cols: &Cols, costs: &Costs, cutoff: f64) -> f64 {
    let len = a.len().max(cols.len());
    if len == 0 {
        return 0.0;
    }

    // The absolute cutoff only ends the DP early, so it errs on the loose side

    let d = distance_to(a, cols, costs, cutoff * len as f64 * (1.0 + 1e-12));
    let n = d / len as f64;
    if n <= cutoff { n } else { f64::INFINITY }
}

pub(crate) fn distance_to(a: &[char], cols: &Cols, costs: &Costs, cutoff: f64) -> f64 {
    match table(a, cols, costs, cutoff) {
        Some(rows) if rows[rows.len() - 1] <= cutoff => rows[rows.len() - 1],
        _ => f64::INFINITY,
    }
}

/// The DP rows of a against cols back to back, or None as soon as every path costs more than `cutoff`.
fn table(a: &[char], cols: &Cols, costs: &Costs, cutoff: f64) -> Option<Vec<f64>> {
    let w = cols.len() + 1;
    let mut rows = cols.row0();
    rows.resize((a.len() + 1) * w, 0.0);

    // A path jumps at most k rows, so every path to the end passes through one of the last k rows

    let k = costs.max_len;
    let mut mins = vec![0.0; k];
    let mut extra = Vec::new();
    for i in 1..=a.len() {
        let (done, rest) = rows.split_at_mut(i * w);
        mins[i % k] = row_step(costs, &a[..i], done, cols, &mut extra, &mut rest[..w]);
        if mins.iter().all(|&m| m > cutoff) {
            return None;
        }
    }
    Some(rows)
}

/// The edits behind `distance(a, b)` in order, as (part of a, part of b, cost), without matches.
pub fn align(a: &str, b: &str, costs: &Costs) -> Vec<(String, String, f64)> {
    let a: Vec<char> = a.chars().collect();
    let cols = Cols::new(b, costs);
    let b = &cols.chars;
    let w = b.len() + 1;
    let rows = table(&a, &cols, costs, f64::INFINITY).expect("no cutoff");
    let cell = |i: usize, j: usize| rows[i * w + j];

    // Walk back from the end, taking the first step that explains each cell:
    // a substitution or match, a deletion, an insertion, then multi-character edits

    let ns = costs.nsym();
    let (mut i, mut j) = (a.len(), b.len());
    let mut out = Vec::new();
    while i > 0 || j > 0 {
        let mut steps: Vec<(usize, usize, f64)> = Vec::new();
        if i > 0 && j > 0 {
            let same = a[i - 1] == b[j - 1];
            steps.push((
                1,
                1,
                if same {
                    0.0
                } else {
                    costs.sub[costs.sym(a[i - 1]) * ns + costs.sym(b[j - 1])]
                },
            ));
        }
        if i > 0 {
            steps.push((1, 0, costs.del[costs.sym(a[i - 1])]));
        }
        if j > 0 {
            steps.push((0, 1, costs.ins[costs.sym(b[j - 1])]));
        }
        if i > 0 {
            for op in costs.ops_ending(costs.sym(a[i - 1])) {
                let (lu, lv) = (op.u.len(), op.v.len());
                if lu <= i && lv <= j && a[i - lu..i] == op.u[..] && b[j - lv..j] == op.v[..] {
                    steps.push((lu, lv, op.cost));
                }
            }
        }
        let here = cell(i, j);
        let &(di, dj, c) = steps
            .iter()
            .find(|&&(di, dj, c)| {
                let t = cell(i - di, j - dj) + c;
                t == here || (t - here).abs() < 1e-9
            })
            .expect("some step explains every DP cell");
        let matched = di == 1 && dj == 1 && a[i - 1] == b[j - 1];
        if !matched {
            out.push((a[i - di..i].iter().collect(), b[j - dj..j].iter().collect(), c));
        }
        i -= di;
        j -= dj;
    }
    out.reverse();
    out
}
