use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::distance::{Cols, distance_to, normalized_to, row_step};
use crate::{Costs, Error, with_workers};

/// A name a search found, with its position in the list and its distance from the query,
/// normalized if the search was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match {
    /// Position of the name in the list the index was built from.
    pub index: usize,
    /// Distance from the query to the name.
    pub cost: f64,
}

impl Eq for Match {}

impl Ord for Match {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.total_cmp(&other.cost).then(self.index.cmp(&other.index))
    }
}

impl PartialOrd for Match {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Exact nearest-name search over a fixed list of names. A name's cost is `distance(query, name)`,
/// or `normalized_distance(query, name)` in a normalized search.
///
/// The names form a trie, and a search walks it with one DP row per depth, so names that
/// share a prefix share rows. A DP path jumps at most `max_len` rows (the longest edit side),
/// so every name below a node costs at least the smallest row minimum among the node's last
/// `max_len` rows. The walk skips every subtree whose bound exceeds the cutoff. It approximates
/// nothing, so it returns what brute force would.
pub struct Index {
    names: Vec<String>,

    // The walk edits names into the query, the reverse of distance(query, name), so it uses the reversed table
    costs: Costs,

    // Trie nodes in depth-first order, node 0 the root. Node v's subtree is v..end[v],
    // and its longest name is deep[v] long.
    ch: Vec<char>,
    depth: Vec<usize>,
    end: Vec<usize>,
    deep: Vec<usize>,

    // Names ending at node v are order[lo[v]..hi[v]], ascending
    order: Vec<usize>,
    lo: Vec<usize>,
    hi: Vec<usize>,
    max_depth: usize,
}

impl Index {
    /// Builds an index of `names` for searches under `costs`. A name's cost is `distance(query, name)`.
    pub fn new(names: Vec<String>, costs: &Costs) -> Self {
        let chars: Vec<Vec<char>> = names.iter().map(|s| s.chars().collect()).collect();
        let mut order: Vec<usize> = (0..names.len()).collect();
        order.sort_by(|&x, &y| chars[x].cmp(&chars[y]));

        // Sorted names build the trie in depth-first order; path holds the nodes of the previous name

        let mut ix = Index {
            names,
            costs: costs.reversed(),
            ch: vec!['\0'],
            depth: vec![0],
            end: vec![0],
            deep: Vec::new(),
            order,
            lo: vec![0],
            hi: vec![0],
            max_depth: 0,
        };
        let mut path = vec![0];
        let mut parent = vec![0];
        let mut prev: &[char] = &[];
        for p in 0..ix.order.len() {
            let s = &chars[ix.order[p]];
            let common = prev.iter().zip(s).take_while(|(x, y)| x == y).count();
            while path.len() > common + 1 {
                ix.end[path.pop().unwrap()] = ix.ch.len();
            }
            for (t, &c) in s.iter().enumerate().skip(common) {
                parent.push(*path.last().unwrap());
                path.push(ix.ch.len());
                ix.ch.push(c);
                ix.depth.push(t + 1);
                ix.end.push(0);
                ix.lo.push(0);
                ix.hi.push(0);
            }
            let v = *path.last().unwrap();
            if ix.hi[v] == ix.lo[v] {
                ix.lo[v] = p;
            }
            ix.hi[v] = p + 1;
            ix.max_depth = ix.max_depth.max(s.len());
            prev = s;
        }
        for v in path {
            ix.end[v] = ix.ch.len();
        }

        // Children come after their parent, so one backward pass carries each depth up

        ix.deep = ix.depth.clone();
        for v in (1..ix.ch.len()).rev() {
            ix.deep[parent[v]] = ix.deep[parent[v]].max(ix.deep[v]);
        }
        ix
    }

    /// The names, in the order they were passed.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The number of names.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the index holds no names.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The cheapest name within `score_cutoff`, ties to the lowest index.
    pub fn extract_one(&self, query: &str, score_cutoff: Option<f64>, normalize: bool) -> Option<Match> {
        self.extract(query, Some(1), score_cutoff, normalize).pop()
    }

    /// The `limit` cheapest names within `score_cutoff` (all of them for `None`), by cost, then index.
    /// With `normalize`, costs are normalized distances, and `score_cutoff` applies to them.
    pub fn extract(&self, query: &str, limit: Option<usize>, score_cutoff: Option<f64>, normalize: bool) -> Vec<Match> {
        let mut hits = Hits {
            limit,
            score_cutoff: score_cutoff.unwrap_or(f64::INFINITY),
            heap: BinaryHeap::new(),
        };
        if limit == Some(0) {
            return Vec::new();
        }
        let costs = &self.costs;
        let cols = Cols::new(query, costs);
        let m = cols.len();
        let k = costs.max_len;
        let w = m + 1;
        let mut rows = cols.row0();
        rows.resize((self.max_depth + 1) * w, 0.0);
        let mut rmin = vec![0.0; self.max_depth + 1];
        let mut path = vec!['\0'; self.max_depth];
        let mut extra = Vec::new();

        // A normalized cost divides by the longer of query and name, so a subtree's bound divides by its longest name

        let score = |cost: f64, len: usize| match (normalize, len.max(m)) {
            (false, _) => cost,
            (true, 0) => 0.0,
            (true, l) => cost / l as f64,
        };
        for p in self.lo[0]..self.hi[0] {
            hits.offer(self.order[p], score(rows[m], 0));
        }
        let mut v = 1;
        while v < self.ch.len() {
            let i = self.depth[v];
            path[i - 1] = self.ch[v];
            let (done, rest) = rows.split_at_mut(i * w);
            rmin[i] = row_step(costs, &path[..i], done, &cols, &mut extra, &mut rest[..w]);
            for p in self.lo[v]..self.hi[v] {
                hits.offer(self.order[p], score(rest[m], i));
            }
            let bound = rmin[(i + 1).saturating_sub(k)..=i]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            v = if score(bound, self.deep[v]) <= hits.cutoff() {
                v + 1
            } else {
                self.end[v]
            };
        }
        hits.heap.into_sorted_vec()
    }

    /// `extract` for each query, on `workers` threads (0 for all cores).
    pub fn extract_many<S: AsRef<str> + Sync>(
        &self,
        queries: &[S],
        limit: Option<usize>,
        score_cutoff: Option<f64>,
        normalize: bool,
        workers: usize,
    ) -> Vec<Vec<Match>> {
        with_workers(workers, || {
            queries
                .par_iter()
                .map(|q| self.extract(q.as_ref(), limit, score_cutoff, normalize))
                .collect()
        })
    }
}

/// The best names so far. The cutoff tightens to the worst kept name once `limit` names are kept.
struct Hits {
    limit: Option<usize>,
    score_cutoff: f64,
    heap: BinaryHeap<Match>,
}

impl Hits {
    fn cutoff(&self) -> f64 {
        match (self.limit, self.heap.peek()) {
            (Some(k), Some(worst)) if self.heap.len() == k => worst.cost,
            _ => self.score_cutoff,
        }
    }

    fn offer(&mut self, index: usize, cost: f64) {
        let hit = Match { index, cost };
        if cost > self.score_cutoff {
            return;
        }
        if self.limit.is_some_and(|k| self.heap.len() == k) {
            if hit >= *self.heap.peek().unwrap() {
                return;
            }
            self.heap.pop();
        }
        self.heap.push(hit);
    }
}

/// Distances from each query to each name, row-major (`queries.len()` rows), on `workers` threads (0 for all cores).
/// With `score_cutoff`, distances above it are infinity. With `normalize`, they are normalized distances.
pub fn cdist<Q, N>(
    queries: &[Q],
    names: &[N],
    costs: &Costs,
    score_cutoff: Option<f64>,
    normalize: bool,
    workers: usize,
) -> Vec<f64>
where
    Q: AsRef<str> + Sync,
    N: AsRef<str> + Sync,
{
    let cutoff = score_cutoff.unwrap_or(f64::INFINITY);
    let cols: Vec<Cols> = names.iter().map(|n| Cols::new(n.as_ref(), costs)).collect();
    with_workers(workers, || {
        queries
            .par_iter()
            .flat_map_iter(|q| {
                let a: Vec<char> = q.as_ref().chars().collect();
                cols.iter()
                    .map(|c| pair(&a, c, costs, cutoff, normalize))
                    .collect::<Vec<f64>>()
            })
            .collect()
    })
}

/// Distances between `a[i]` and `b[i]` for each i, on `workers` threads (0 for all cores).
/// `score_cutoff` and `normalize` work as in `cdist`.
pub fn cpdist<A, B>(
    a: &[A],
    b: &[B],
    costs: &Costs,
    score_cutoff: Option<f64>,
    normalize: bool,
    workers: usize,
) -> Result<Vec<f64>, Error>
where
    A: AsRef<str> + Sync,
    B: AsRef<str> + Sync,
{
    if a.len() != b.len() {
        return Err(Error(format!("lengths differ: {} and {}", a.len(), b.len())));
    }
    let cutoff = score_cutoff.unwrap_or(f64::INFINITY);
    Ok(with_workers(workers, || {
        a.par_iter()
            .zip(b)
            .map(|(x, y)| {
                let x: Vec<char> = x.as_ref().chars().collect();
                pair(&x, &Cols::new(y.as_ref(), costs), costs, cutoff, normalize)
            })
            .collect()
    }))
}

fn pair(a: &[char], cols: &Cols, costs: &Costs, cutoff: f64, normalize: bool) -> f64 {
    if a == cols.chars {
        0.0
    } else if normalize {
        normalized_to(a, cols, costs, cutoff)
    } else {
        distance_to(a, cols, costs, cutoff)
    }
}
