use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::{Costs, Error, align, difflib};

/// Settings of `learn_costs`, with defaults from `LearnOptions::default()`.
#[derive(Clone, Debug)]
pub struct LearnOptions {
    /// Multi-character edits to consider beyond those found in the data, such as ("ll", "l") or ("tz", "z")
    pub seeds: Vec<(String, String)>,
    /// Edits with a cost set by hand, such as ("ae", "ä", 0.1). They go into the result as given and are not learned.
    pub fixed: Vec<(String, String, f64)>,
    /// Rounds of aligning and counting
    pub rounds: usize,
    /// Edits with a probability at or below this cost 1
    pub p_floor: f64,
    /// Times a 2:1, 1:2 or 2:2 replacement must occur in the diffs to become a candidate edit
    pub min_multi: u64,
    /// Pairs whose difflib ratio falls below this count as misalignments and are left out
    pub min_ratio: f64,
    /// Standard errors taken off each count before it becomes a probability
    pub shrink: f64,
    /// Whether to learn one cost per edit and its reverse, or a cost per direction (truth to reading)
    pub symmetric: bool,
    /// Costs to start from and shrink towards, such as a pretrained table
    pub prior: Option<Costs>,
    /// Occurrences of each string the prior counts as
    pub prior_weight: f64,
}

impl Default for LearnOptions {
    fn default() -> Self {
        LearnOptions {
            seeds: Vec::new(),
            fixed: Vec::new(),
            rounds: 4,
            p_floor: 1e-5,
            min_multi: 20,
            min_ratio: 0.5,
            shrink: 2.0,
            symmetric: true,
            prior: None,
            prior_weight: 30.0,
        }
    }
}

/// One row of a learned cost table. `count` and `p` are `None` for fixed edits.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedEdit {
    /// The string replaced, in the truth for a directed table.
    pub a: String,
    /// The string it is replaced by, in the reading for a directed table.
    pub b: String,
    /// The learned or fixed cost.
    pub cost: f64,
    /// Times the edit was counted in the last round.
    pub count: Option<u64>,
    /// The edit's estimated probability.
    pub p: Option<f64>,
}

/// Cost of a multi-character candidate the last round did not use, just below the default of 1.
const CANDIDATE_COST: f64 = 0.999;

/// Learns edit costs from (truth, reading, count) pairs.
///
/// Each round aligns every misread pair under the current costs, counts the edits, and turns
/// each count c into a probability and a cost:
///
/// ```text
/// p    = max(c - shrink * sqrt(c), 0) / n
/// cost = min(1, ln p / ln p_floor), rounded to 4 decimals
/// ```
///
/// where n counts the occurrences of the edited strings in all pairs: of u for a deletion of u,
/// the mean of u and v for a replacement of equal lengths, the longer side otherwise. The
/// candidate multi-character edits are the seeds and the 2:1, 1:2 and 2:2 replacements difflib
/// finds at least `min_multi` times.
///
/// With a `prior`, the probabilities shrink towards the prior's. Each prior cost becomes
/// p0 = p_floor ^ cost, and with m = `prior_weight`
///
/// ```text
/// p = (max(c - shrink * sqrt(c), 0) + m * p0) / (n + m)
/// ```
///
/// So the prior's cost wins for an edit the data rarely shows, and the data's count wins for an
/// edit it shows often. The prior's costs also align the first round, and its multi-character
/// edits join the candidates. A directed prior needs `symmetric` false.
///
/// With `symmetric` false, each direction gets its own cost. An edit (a, b) replaces a in the
/// truth by b in the reading, and n counts the occurrences of a in the truths. For an insertion,
/// n counts the places in the truths it could go, one more than their length. Candidates and
/// seeds are tried in both directions, and fixed edits apply in the direction given.
///
/// Returns the edits that cost less than 1, by cost, then by a and b, followed by the fixed edits.
pub fn learn_costs(pairs: &[(String, String, u64)], opts: &LearnOptions) -> Result<Vec<LearnedEdit>, Error> {
    let sym = opts.symmetric;
    let key = |u: String, v: String| {
        if sym && (u.is_empty() || v.is_empty()) {
            (u + &v, String::new())
        } else if sym {
            unordered(u, v)
        } else {
            (u, v)
        }
    };
    Costs::with_symmetry(opts.fixed.clone(), 1.0, sym)?;

    // Prior probabilities by edit

    let mut prior: HashMap<(String, String), f64> = HashMap::new();
    let mut prior_rows = Vec::new();
    if let Some(p) = &opts.prior {
        if sym && !p.is_symmetric() {
            return Err(Error("a directed prior needs symmetric=false".into()));
        }
        for (a, b, c) in p.rows() {
            let both = if p.is_symmetric() && !sym {
                vec![(a.clone(), b.clone()), (b.clone(), a.clone())]
            } else {
                vec![(a.clone(), b.clone())]
            };
            for (u, v) in both {
                prior.insert(key(u.clone(), v.clone()), opts.p_floor.powf(c.min(1.0)));
                prior_rows.push((u, v, *c));
            }
        }
    }
    let m = if prior.is_empty() { 0.0 } else { opts.prior_weight };
    let chars: Vec<(Vec<char>, Vec<char>)> = pairs
        .iter()
        .map(|(t, r, _)| (t.chars().collect(), r.chars().collect()))
        .collect();

    // Misread pairs, without misalignments (Stöckl / Rimmel share few letters)

    let mut diff = Vec::new();
    let mut seen: HashMap<(String, String), u64> = HashMap::new();
    for (k, (t, r)) in chars.iter().enumerate() {
        if t == r {
            continue;
        }
        let blocks = difflib::matching_blocks(t, r);
        if difflib::ratio(t, r, &blocks) < opts.min_ratio {
            continue;
        }
        diff.push(k);
        for (x, y) in difflib::replacements(t, r, &blocks) {
            if x.len() <= 2 && y.len() <= 2 && x.len() + y.len() > 2 {
                let (u, v): (String, String) = (t[x].iter().collect(), r[y].iter().collect());
                *seen.entry(unordered(u, v)).or_default() += pairs[k].2;
            }
        }
    }
    let mut multi: Vec<(String, String)> = seen
        .into_iter()
        .filter(|(_, c)| *c >= opts.min_multi)
        .map(|(e, _)| e)
        .collect();
    multi.sort();
    let mut known: HashSet<(String, String)> = multi.iter().cloned().collect();
    let multi_prior = prior_rows
        .iter()
        .filter(|(u, v, _)| u.chars().count().max(v.chars().count()) > 1)
        .map(|(u, v, _)| (u.clone(), v.clone()));
    for (u, v) in opts.seeds.iter().cloned().chain(multi_prior) {
        if u != v && !u.is_empty() && !v.is_empty() && known.insert(unordered(u.clone(), v.clone())) {
            multi.push((u, v));
        }
    }

    // Occurrences of every string an edit can touch

    let longest = multi
        .iter()
        .map(|(u, v)| u.chars().count().max(v.chars().count()))
        .max()
        .unwrap_or(1)
        .max(1);
    let mut occ: HashMap<&[char], u64> = HashMap::new();
    let mut places = 0.0;
    for ((t, r), (_, _, n)) in chars.iter().zip(pairs) {
        places += ((t.len() + 1) as u64 * n) as f64;
        let sides = if sym { vec![t, r] } else { vec![t] };
        for s in sides {
            for len in 1..=longest.min(s.len()) {
                for w in s.windows(len) {
                    *occ.entry(w).or_default() += n;
                }
            }
        }
    }
    let occ_of = |s: &str| occ.get(&s.chars().collect::<Vec<char>>()[..]).copied().unwrap_or(0) as f64;

    // Candidates, in both directions for a directed table

    let tried: Vec<(String, String)> = if sym {
        multi.clone()
    } else {
        multi
            .iter()
            .flat_map(|(u, v)| [(u.clone(), v.clone()), (v.clone(), u.clone())])
            .collect()
    };
    let candidates = |skip: &dyn Fn(&(String, String)) -> bool| {
        tried
            .iter()
            .filter(|e| !skip(e))
            .map(|(u, v)| (u.clone(), v.clone(), CANDIDATE_COST))
            .collect::<Vec<_>>()
    };
    let in_prior = |(u, v): &(String, String)| prior.contains_key(&key(u.clone(), v.clone()));
    let mut start = prior_rows.clone();
    start.extend(candidates(&in_prior));
    let mut costs = Costs::with_symmetry(start, 1.0, sym)?;
    let mut table = Vec::new();
    for _ in 0..opts.rounds {
        // Count the edits of the best alignments, keeping the order they first occur in

        let aligned: Vec<Vec<(String, String, f64)>> = diff
            .par_iter()
            .map(|&k| align(&pairs[k].0, &pairs[k].1, &costs))
            .collect();
        let mut edits: Vec<((String, String), u64)> = Vec::new();
        let mut pos: HashMap<(String, String), usize> = HashMap::new();
        for (&k, row) in diff.iter().zip(aligned) {
            for (u, v, _) in row {
                let e = key(u, v);
                let p = *pos.entry(e.clone()).or_insert_with(|| {
                    edits.push((e, 0));
                    edits.len() - 1
                });
                edits[p].1 += pairs[k].2;
            }
        }

        // Prior edits the alignments did not use still get a cost from the prior alone

        let mut unused: Vec<(String, String)> = prior.keys().filter(|e| !pos.contains_key(*e)).cloned().collect();
        unused.sort();
        for e in unused {
            pos.insert(e.clone(), edits.len());
            edits.push((e, 0));
        }
        let p0_default = opts
            .p_floor
            .powf(opts.prior.as_ref().map_or(1.0, |p| p.default_cost().min(1.0)));

        table.clear();
        let mut rows = Vec::new();
        for ((u, v), c) in edits {
            let (lu, lv) = (u.chars().count(), v.chars().count());
            let n = if !sym {
                if u.is_empty() { places } else { occ_of(&u) }
            } else if v.is_empty() {
                occ_of(&u)
            } else if lu == lv {
                (occ_of(&u) + occ_of(&v)) / 2.0
            } else {
                occ_of(if lu > lv { &u } else { &v })
            };
            let cf = c as f64;
            let p0 = prior.get(&(u.clone(), v.clone())).copied().unwrap_or(p0_default);
            let p = if n + m > 0.0 {
                ((cf - opts.shrink * cf.sqrt()).max(0.0) + m * p0) / (n + m)
            } else {
                0.0
            };
            let cost = if p > 0.0 {
                round4((p.ln() / opts.p_floor.ln()).min(1.0))
            } else {
                1.0
            };
            if cost < 1.0 {
                rows.push((u.clone(), v.clone(), cost));
            }
            table.push(LearnedEdit {
                a: u,
                b: v,
                cost,
                count: Some(c),
                p: Some(p),
            });
        }
        rows.extend(candidates(&|(u, v)| pos.contains_key(&key(u.clone(), v.clone()))));
        costs = Costs::with_symmetry(rows, 1.0, sym)?;
    }

    let fixed: HashSet<(String, String)> = opts.fixed.iter().map(|(a, b, _)| key(a.clone(), b.clone())).collect();
    table.retain(|e| e.cost < 1.0 && !fixed.contains(&key(e.a.clone(), e.b.clone())));
    table.sort_by(|x, y| x.cost.total_cmp(&y.cost).then_with(|| (&x.a, &x.b).cmp(&(&y.a, &y.b))));
    for (a, b, cost) in &opts.fixed {
        table.push(LearnedEdit {
            a: a.clone(),
            b: b.clone(),
            cost: *cost,
            count: None,
            p: None,
        });
    }
    Ok(table)
}

fn unordered(u: String, v: String) -> (String, String) {
    if u <= v { (u, v) } else { (v, u) }
}

/// Rounds to 4 decimals from the exact binary value, as Python's `round(x, 4)` does.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap()
}
