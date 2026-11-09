//! The parts of Python's `difflib.SequenceMatcher(None, a, b, autojunk=False)` the learner uses.

use std::collections::HashMap;

/// Matching blocks (i, j, size), as `get_matching_blocks()` returns them, without the final empty block.
pub(crate) fn matching_blocks(a: &[char], b: &[char]) -> Vec<(usize, usize, usize)> {
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, &c) in b.iter().enumerate() {
        b2j.entry(c).or_default().push(j);
    }
    let mut blocks = Vec::new();
    let mut queue = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest_match(a, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    blocks.sort_unstable();

    // Merge adjacent blocks

    let mut out: Vec<(usize, usize, usize)> = Vec::new();
    for (i, j, k) in blocks {
        match out.last_mut() {
            Some(last) if last.0 + last.2 == i && last.1 + last.2 == j => last.2 += k,
            _ => out.push((i, j, k)),
        }
    }
    out
}

/// Longest block a[i..i+k] == b[j..j+k] within the bounds, earliest in a, then in b.
fn longest_match(
    a: &[char],
    b2j: &HashMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0);
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for (i, c) in a.iter().enumerate().take(ahi).skip(alo) {
        let mut next = HashMap::new();
        for &j in b2j.get(c).into_iter().flatten() {
            if j < blo {
                continue;
            }
            if j >= bhi {
                break;
            }
            let k = if j > 0 {
                j2len.get(&(j - 1)).copied().unwrap_or(0)
            } else {
                0
            } + 1;
            next.insert(j, k);
            if k > bestsize {
                (besti, bestj, bestsize) = (i + 1 - k, j + 1 - k, k);
            }
        }
        j2len = next;
    }
    (besti, bestj, bestsize)
}

/// `ratio()`, twice the matched characters over the total length.
pub(crate) fn ratio(a: &[char], b: &[char], blocks: &[(usize, usize, usize)]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * blocks.iter().map(|x| x.2).sum::<usize>() as f64 / total as f64
}

/// The `replace` opcodes as (a range, b range).
pub(crate) fn replacements(
    a: &[char],
    b: &[char],
    blocks: &[(usize, usize, usize)],
) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    for &(bi, bj, k) in blocks.iter().chain([(a.len(), b.len(), 0)].iter()) {
        if i < bi && j < bj {
            out.push((i..bi, j..bj));
        }
        (i, j) = (bi + k, bj + k);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(a: &str, b: &str) -> (f64, Vec<(usize, usize, usize, usize)>) {
        let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
        let blocks = matching_blocks(&a, &b);
        let reps = replacements(&a, &b, &blocks)
            .into_iter()
            .map(|(x, y)| (x.start, x.end, y.start, y.end))
            .collect();
        (ratio(&a, &b, &blocks), reps)
    }

    /// Expected values from CPython 3.13's difflib
    #[test]
    fn matches_python() {
        assert_eq!(
            run("Nußbaum", "Rutzbaum"),
            (2.0 * 5.0 / 15.0, vec![(0, 1, 0, 1), (2, 3, 2, 4)])
        );
        assert_eq!(run("Stöckl", "Rimmel"), (2.0 / 12.0, vec![(0, 5, 0, 5)]));
        assert_eq!(run("abcabba", "cbabac"), (6.0 / 13.0, vec![]));
        assert_eq!(run("Moritz", "Moriß"), (8.0 / 11.0, vec![(4, 6, 4, 5)]));
        assert_eq!(run("", "ab"), (0.0, vec![]));
        assert_eq!(
            run("Lieffenbacher", "Liessenbacker"),
            (20.0 / 26.0, vec![(3, 5, 3, 5), (10, 11, 10, 11)])
        );
    }
}
