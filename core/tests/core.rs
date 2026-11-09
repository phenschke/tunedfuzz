use tunedfuzz::{Costs, Index, LearnOptions, align, cdist, cpdist, distance, learn_costs, normalized_distance};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures");

fn toy() -> Costs {
    Costs::read_csv(format!("{FIXTURES}/costs.csv"), 1.0, true).unwrap()
}

/// The toy table made directed. Each edit costs more in reverse, and `B` -> `V` goes one way only.
fn directed_toy() -> Costs {
    let mut rows = Vec::new();
    for (a, b, c) in toy().rows() {
        rows.push((a.clone(), b.clone(), *c));
        rows.push((b.clone(), a.clone(), (c * 1.7).min(1.0)));
    }
    rows.push(("B".into(), "V".into(), 0.05));
    Costs::directed(rows, 1.0).unwrap()
}

/// Random names over letters the toy table names, so its edits come up often.
fn random_names(n: usize, seed: u64) -> Vec<String> {
    let letters: Vec<char> = "aeinrstuldbhkzßRNBV".chars().collect();
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = move |k: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % k as u64) as usize
    };
    (0..n)
        .map(|_| (0..next(9)).map(|_| letters[next(letters.len())]).collect())
        .collect()
}

fn levenshtein(a: &str, b: &str) -> f64 {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut diag = row[0];
        row[0] = i;
        for j in 1..=b.len() {
            let up = row[j];
            row[j] = (diag + (a[i - 1] != b[j - 1]) as usize).min(up + 1).min(row[j - 1] + 1);
            diag = up;
        }
    }
    row[b.len()] as f64
}

#[test]
fn fixtures() {
    let costs = toy();
    let text = std::fs::read_to_string(format!("{FIXTURES}/distances.csv")).unwrap();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let want: f64 = f[2].parse().unwrap();
        let got = distance(f[0], f[1], &costs, None);
        assert!((got - want).abs() < 1e-9, "{line}: got {got}");
    }
}

#[test]
fn empty_table_is_levenshtein() {
    let names = random_names(600, 1);
    for p in names.chunks(2) {
        assert_eq!(
            distance(&p[0], &p[1], &Costs::default(), None),
            levenshtein(&p[0], &p[1])
        );
    }
}

#[test]
fn symmetric_and_below_levenshtein() {
    let costs = toy();
    let names = random_names(600, 2);
    for p in names.chunks(2) {
        let d = distance(&p[0], &p[1], &costs, None);
        assert!((d - distance(&p[1], &p[0], &costs, None)).abs() < 1e-9);
        assert!(d <= levenshtein(&p[0], &p[1]));
        assert_eq!(d == 0.0, p[0] == p[1]);
    }
}

#[test]
fn later_rows_override_earlier_ones() {
    let costs = Costs::new(
        [("R", "N", 0.2), ("N", "R", 0.4), ("tz", "ß", 0.3), ("ß", "tz", 0.1)],
        1.0,
    )
    .unwrap();
    assert_eq!(distance("R", "N", &costs, None), 0.4);
    assert_eq!(distance("Seitz", "Seiß", &costs, None), 0.1);
    assert_eq!(costs.rows().len(), 2);
}

#[test]
fn directed_tables() {
    let costs = Costs::directed(
        [("R", "N", 0.2), ("n", "", 0.3), ("", "e", 0.4), ("tz", "ß", 0.25)],
        1.0,
    )
    .unwrap();
    assert_eq!(distance("R", "N", &costs, None), 0.2);
    assert_eq!(distance("N", "R", &costs, None), 1.0);
    assert_eq!(distance("Hans", "Has", &costs, None), 0.3);
    assert_eq!(distance("Has", "Hans", &costs, None), 1.0);
    assert_eq!(distance("Hans", "Hanse", &costs, None), 0.4);
    assert_eq!(distance("Seitz", "Seiß", &costs, None), 0.25);
    assert_eq!(distance("Seiß", "Seitz", &costs, None), 2.0);
    assert!(Costs::directed([("", "ab", 0.5)], 1.0).is_err());

    // Each edit in both directions at one cost gives the symmetric table, and reversed() swaps distance's arguments

    let both: Vec<_> = toy()
        .rows()
        .iter()
        .flat_map(|(a, b, c)| [(a.clone(), b.clone(), *c), (b.clone(), a.clone(), *c)])
        .collect();
    let both = Costs::directed(both, 1.0).unwrap();
    let (directed, reversed) = (directed_toy(), directed_toy().reversed());
    assert!(!directed.is_symmetric() && toy().is_symmetric());
    let names = random_names(800, 9);
    for p in names.chunks(2) {
        assert_eq!(
            distance(&p[0], &p[1], &both, None),
            distance(&p[0], &p[1], &toy(), None)
        );
        assert_eq!(
            distance(&p[0], &p[1], &directed, None),
            distance(&p[1], &p[0], &reversed, None)
        );
        assert!(distance(&p[0], &p[1], &directed, None) <= levenshtein(&p[0], &p[1]));
    }
}

#[test]
fn normalized_distance_divides_by_the_longer_string() {
    let costs = toy();
    assert_eq!(normalized_distance("Nußbaum", "Rutzbaum", &costs, None), 0.45 / 8.0);
    assert_eq!(normalized_distance("", "", &costs, None), 0.0);
    assert_eq!(normalized_distance("", "ab", &Costs::default(), None), 1.0);
    let names = random_names(600, 10);
    for p in names.chunks(2) {
        let d = normalized_distance(&p[0], &p[1], &costs, None);
        assert!((0.0..=1.0).contains(&d));
        for c in [0.0, 0.1, 0.3] {
            let want = if d <= c { d } else { f64::INFINITY };
            assert_eq!(normalized_distance(&p[0], &p[1], &costs, Some(c)), want);
        }
    }
}

#[test]
fn csv_round_trip() {
    let dir = std::env::temp_dir().join(format!("tunedfuzz-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("costs.csv");
    for costs in [toy(), directed_toy()] {
        costs.write_csv(&path).unwrap();
        let back = Costs::read_csv(&path, 1.0, costs.is_symmetric()).unwrap();
        assert_eq!(back.rows(), costs.rows());
    }
    let tricky = Costs::new([("a,b", "\"x\"", 0.1), ("ß", "", 1.0 / 3.0)], 1.0).unwrap();
    tricky.write_csv(&path).unwrap();
    assert_eq!(Costs::read_csv(&path, 1.0, true).unwrap().rows(), tricky.rows());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn rejects_bad_rows() {
    assert!(Costs::new([("a", "b", -0.1)], 1.0).is_err());
    assert!(Costs::new([("a", "b", f64::NAN)], 1.0).is_err());
    assert!(Costs::new([("ab", "", 0.5)], 1.0).is_err());
    assert!(Costs::new(Vec::<(&str, &str, f64)>::new(), -1.0).is_err());
}

#[test]
fn edits_longer_than_two_characters() {
    let costs = Costs::new([("sch", "f", 0.3), ("ß", "ssz", 0.2)], 1.0).unwrap();
    assert_eq!(distance("Mischa", "Mifa", &costs, None), 0.3);
    assert_eq!(
        distance("Mischa", "Mifa", &Costs::new([("ch", "f", 0.3)], 1.0).unwrap(), None),
        1.3
    );
    assert_eq!(distance("Faßl", "Fasszl", &costs, None), 0.2);
    let names = vec!["Misa".to_string(), "Mischa".into(), "Fasszl".into()];
    let ix = Index::new(names, &costs);
    assert_eq!(ix.extract_one("Mifa", None, false).unwrap().index, 1);
    assert_eq!(ix.extract_one("Faßl", Some(0.5), false).unwrap().index, 2);
}

#[test]
fn score_cutoff() {
    let costs = toy();
    assert_eq!(distance("Nußbaum", "Rutzbaum", &costs, Some(0.45)), 0.45);
    assert_eq!(distance("Nußbaum", "Rutzbaum", &costs, Some(0.44)), f64::INFINITY);
    assert_eq!(distance("Huber", "Xaverlmeier", &costs, Some(1.0)), f64::INFINITY);
    let names = random_names(600, 3);
    for p in names.chunks(2) {
        let d = distance(&p[0], &p[1], &costs, None);
        for c in [0.0, 0.5, 1.0, 2.5] {
            let want = if d <= c { d } else { f64::INFINITY };
            assert_eq!(distance(&p[0], &p[1], &costs, Some(c)), want);
        }
    }
}

#[test]
fn align_explains_distance() {
    let costs = toy();
    let edits = align("Nußbaum", "Rutzbaum", &costs);
    assert_eq!(
        edits,
        vec![("N".into(), "R".into(), 0.2), ("ß".into(), "tz".into(), 0.25)]
    );
    let names = random_names(400, 4);
    for costs in [costs, directed_toy()] {
        for p in names.chunks(2) {
            let total: f64 = align(&p[0], &p[1], &costs).iter().map(|e| e.2).sum();
            assert!((total - distance(&p[0], &p[1], &costs, None)).abs() < 1e-9);
        }
    }
}

/// (cost, index) of every name, ascending.
fn brute_force(names: &[String], q: &str, costs: &Costs, normalize: bool) -> Vec<(f64, usize)> {
    let d = if normalize { normalized_distance } else { distance };
    let mut all: Vec<(f64, usize)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (d(q, n, costs, None), i))
        .collect();
    all.sort_by(|x, y| x.partial_cmp(y).unwrap());
    all
}

#[test]
fn index_is_brute_force() {
    let mut names = random_names(400, 5);
    names.extend(names[..20].to_vec());
    names.push(String::new());
    let mut queries = random_names(100, 6);
    queries.push(String::new());
    for costs in [toy(), Costs::default(), directed_toy()] {
        let ix = Index::new(names.clone(), &costs);
        for (normalize, cutoffs) in [(false, [0.5, 2.0, f64::INFINITY]), (true, [0.1, 0.3, f64::INFINITY])] {
            for q in &queries {
                let all = brute_force(&names, q, &costs, normalize);
                for cutoff in cutoffs {
                    let within: Vec<(f64, usize)> = all.iter().copied().filter(|h| h.0 <= cutoff).collect();
                    let got = ix.extract(q, Some(3), Some(cutoff), normalize);
                    assert_eq!(
                        got.iter().map(|m| (m.cost, m.index)).collect::<Vec<_>>(),
                        within[..within.len().min(3)]
                    );
                    let got = ix.extract(q, None, Some(cutoff), normalize);
                    assert_eq!(got.iter().map(|m| (m.cost, m.index)).collect::<Vec<_>>(), within);
                    let one = ix.extract_one(q, Some(cutoff), normalize).map(|m| (m.cost, m.index));
                    assert_eq!(one, within.first().copied());
                }
            }
        }
    }
}

#[test]
fn batches_match_single_queries() {
    let costs = directed_toy();
    let names = random_names(300, 7);
    let queries = random_names(50, 8);
    let ix = Index::new(names.clone(), &costs);
    let many = ix.extract_many(&queries, Some(2), Some(2.0), false, 4);
    for (q, got) in queries.iter().zip(&many) {
        assert_eq!(got, &ix.extract(q, Some(2), Some(2.0), false));
    }
    let m = cdist(&queries, &names, &costs, None, false, 0);
    let mn = cdist(&queries, &names, &costs, Some(0.3), true, 0);
    for (i, q) in queries.iter().enumerate() {
        for (j, n) in names.iter().enumerate() {
            assert_eq!(m[i * names.len() + j], distance(q, n, &costs, None));
            assert_eq!(mn[i * names.len() + j], normalized_distance(q, n, &costs, Some(0.3)));
        }
    }
    let p = cpdist(&queries, &names[..50], &costs, Some(2.0), false, 3).unwrap();
    for (i, q) in queries.iter().enumerate() {
        assert_eq!(p[i], distance(q, &names[i], &costs, Some(2.0)));
    }
    assert!(cpdist(&queries, &names, &costs, None, false, 1).is_err());
}

#[test]
fn learner_recovers_ocr_confusions() {
    // Truth names read with R as N, tz as ß, and nothing else
    let truth = ["Rauch", "Rieger", "Seitz", "Moritz", "Lutz", "Huber", "Rotzer", "Kurz"];
    let mut pairs = Vec::new();
    for t in truth {
        pairs.push((t.to_string(), t.to_string(), 50));
        let read = t.replacen('R', "N", 1).replace("tz", "ß");
        if read != t {
            pairs.push((t.to_string(), read, 10));
        }
    }
    let opts = LearnOptions {
        seeds: vec![("ll".into(), "l".into())],
        fixed: vec![("ae".into(), "ä".into(), 0.1)],
        min_multi: 5,
        ..LearnOptions::default()
    };
    let rows = learn_costs(&pairs, &opts).unwrap();
    let cost = |a: &str, b: &str| rows.iter().find(|r| r.a == a && r.b == b).map(|r| r.cost);
    assert!(cost("N", "R").unwrap() < 0.5);
    assert!(cost("tz", "ß").unwrap() < 0.5);
    assert_eq!(cost("ae", "ä"), Some(0.1));
    assert_eq!(rows.last().unwrap().count, None);
    assert!(rows.iter().all(|r| r.cost < 1.0));
    assert!(rows.windows(2).take(rows.len() - 2).all(|w| w[0].cost <= w[1].cost));
}

#[test]
fn learner_learns_directions() {
    // Truth R read as N and tz as ß, never the reverse
    let truth = [
        "Rauch", "Rieger", "Seitz", "Moritz", "Lutz", "Huber", "Rotzer", "Kurz", "Nagl", "Bauß",
    ];
    let mut pairs = Vec::new();
    for t in truth {
        pairs.push((t.to_string(), t.to_string(), 50));
        let read = t.replacen('R', "N", 1).replace("tz", "ß");
        if read != t {
            pairs.push((t.to_string(), read, 10));
        }
    }
    let opts = LearnOptions {
        min_multi: 5,
        symmetric: false,
        ..LearnOptions::default()
    };
    let rows = learn_costs(&pairs, &opts).unwrap();
    let cost = |a: &str, b: &str| rows.iter().find(|r| r.a == a && r.b == b).map(|r| r.cost);
    assert!(cost("R", "N").unwrap() < 0.5);
    assert!(cost("tz", "ß").unwrap() < 0.5);
    assert_eq!(cost("N", "R"), None);
    assert_eq!(cost("ß", "tz"), None);
}

#[test]
fn prior_shrinks_towards_its_costs() {
    // A little data with R read as N, a prior that also knows e read as c
    let mut pairs = vec![("Rauch".to_string(), "Nauch".to_string(), 3)];
    pairs.extend(["Rieger", "Huber", "Seitz"].map(|t| (t.to_string(), t.to_string(), 30)));
    let prior = Costs::directed([("R", "N", 0.2), ("e", "c", 0.3)], 1.0).unwrap();
    let learn = |prior_weight: f64| {
        let opts = LearnOptions {
            symmetric: false,
            prior: Some(prior.clone()),
            prior_weight,
            ..LearnOptions::default()
        };
        learn_costs(&pairs, &opts).unwrap()
    };
    let cost =
        |rows: &[tunedfuzz::LearnedEdit], a: &str, b: &str| rows.iter().find(|r| r.a == a && r.b == b).map(|r| r.cost);

    // A heavy prior keeps its costs; an unseen edit keeps a cost from the prior alone
    let heavy = learn(1e9);
    assert!((cost(&heavy, "R", "N").unwrap() - 0.2).abs() < 1e-3);
    assert!((cost(&heavy, "e", "c").unwrap() - 0.3).abs() < 1e-3);

    // No data on e, so a light prior lets its cost drift towards 1
    let light = learn(10.0);
    assert!(cost(&light, "e", "c").is_none_or(|c| c > 0.3));

    // A symmetric run rejects a directed prior
    let opts = LearnOptions {
        prior: Some(prior.clone()),
        ..LearnOptions::default()
    };
    assert!(learn_costs(&pairs, &opts).is_err());
}

#[test]
fn pretrained_tables_load() {
    for name in tunedfuzz::PRETRAINED {
        let costs = Costs::pretrained(name).unwrap();
        assert!(!costs.is_symmetric());
        assert!(costs.rows().len() > 100);
    }
    assert!(Costs::pretrained("nonexistent").is_err());
}
