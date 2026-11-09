// Bindings behind the R functions in R/tunedfuzz.R. Indices returned to R are 1-based.

use extendr_api::Result;
use extendr_api::prelude::*;

fn fail(e: tfz::Error) -> Error {
    Error::Other(e.0)
}

// -1 means all cores, which tfz calls 0.
fn workers(n: i32) -> Result<usize> {
    match n {
        -1 => Ok(0),
        n if n >= 1 => Ok(n as usize),
        _ => Err(Error::Other("workers must be -1 (all cores) or at least 1".into())),
    }
}

// (a, b, cost) triples as a list of three columns.
fn triples(rows: &[(String, String, f64)]) -> List {
    list!(
        a = rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
        b = rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
        cost = rows.iter().map(|r| r.2).collect::<Vec<_>>()
    )
}

#[extendr]
fn costs_new(
    a: Vec<String>,
    b: Vec<String>,
    cost: Vec<f64>,
    default: f64,
    symmetric: bool,
) -> Result<ExternalPtr<tfz::Costs>> {
    let rows = a.into_iter().zip(b).zip(cost).map(|((a, b), c)| (a, b, c));
    tfz::Costs::with_symmetry(rows, default, symmetric)
        .map(ExternalPtr::new)
        .map_err(fail)
}

#[extendr]
fn costs_read(path: &str, default: f64, symmetric: bool) -> Result<ExternalPtr<tfz::Costs>> {
    tfz::Costs::read_csv(path, default, symmetric)
        .map(ExternalPtr::new)
        .map_err(fail)
}

#[extendr]
fn costs_pretrained(name: &str) -> Result<ExternalPtr<tfz::Costs>> {
    tfz::Costs::pretrained(name).map(ExternalPtr::new).map_err(fail)
}

#[extendr]
fn costs_reversed(costs: &ExternalPtr<tfz::Costs>) -> ExternalPtr<tfz::Costs> {
    ExternalPtr::new(costs.reversed())
}

// The rows, default and symmetry that rebuild the table.
#[extendr]
fn costs_info(costs: &ExternalPtr<tfz::Costs>) -> List {
    list!(
        rows = triples(costs.rows()),
        default = costs.default_cost(),
        symmetric = costs.is_symmetric()
    )
}

#[extendr]
fn costs_write(costs: &ExternalPtr<tfz::Costs>, path: &str) -> Result<()> {
    costs.write_csv(path).map_err(fail)
}

// Whether an external pointer still has its address. saveRDS() keeps the pointer but not the address.
#[extendr]
fn ptr_alive(ptr: Robj) -> bool {
    ptr.is_external_pointer() && unsafe { !ptr.external_ptr_addr::<u8>().is_null() }
}

#[extendr]
fn distance_pairs(
    a: Vec<String>,
    b: Vec<String>,
    costs: &ExternalPtr<tfz::Costs>,
    score_cutoff: f64,
    normalize: bool,
    workers: i32,
) -> Result<Vec<f64>> {
    tfz::cpdist(&a, &b, costs, Some(score_cutoff), normalize, self::workers(workers)?).map_err(fail)
}

#[extendr]
fn align_pair(a: &str, b: &str, costs: &ExternalPtr<tfz::Costs>) -> List {
    triples(&tfz::align(a, b, costs))
}

#[extendr]
fn index_new(names: Vec<String>, costs: &ExternalPtr<tfz::Costs>) -> ExternalPtr<tfz::Index> {
    ExternalPtr::new(tfz::Index::new(names, costs))
}

// Returns one row per match, with `query` the position of its query. A negative limit returns every match.
#[extendr]
fn index_extract(
    index: &ExternalPtr<tfz::Index>,
    queries: Vec<String>,
    limit: i32,
    score_cutoff: f64,
    normalize: bool,
    workers: i32,
) -> Result<List> {
    let limit = usize::try_from(limit).ok();
    let w = self::workers(workers)?;
    let found = index.extract_many(&queries, limit, Some(score_cutoff), normalize, w);
    let (mut query, mut idx, mut cost) = (Vec::new(), Vec::new(), Vec::new());
    for (q, ms) in found.iter().enumerate() {
        for m in ms {
            query.push(q as i32 + 1);
            idx.push(m.index as i32 + 1);
            cost.push(m.cost);
        }
    }
    Ok(list!(query = query, index = idx, cost = cost))
}

// Row-major distances, which R fills into a matrix by row.
#[extendr]
fn cdist_flat(
    queries: Vec<String>,
    names: Vec<String>,
    costs: &ExternalPtr<tfz::Costs>,
    score_cutoff: f64,
    normalize: bool,
    workers: i32,
) -> Result<Vec<f64>> {
    let w = self::workers(workers)?;
    Ok(tfz::cdist(&queries, &names, costs, Some(score_cutoff), normalize, w))
}

#[extendr]
#[allow(clippy::too_many_arguments)]
fn learn(
    truth: Vec<String>,
    reading: Vec<String>,
    count: Vec<f64>,
    seeds_a: Vec<String>,
    seeds_b: Vec<String>,
    fixed_a: Vec<String>,
    fixed_b: Vec<String>,
    fixed_cost: Vec<f64>,
    rounds: i32,
    p_floor: f64,
    min_multi: f64,
    min_ratio: f64,
    shrink: f64,
    symmetric: bool,
    prior: Robj,
    prior_weight: f64,
) -> Result<List> {
    let pairs: Vec<(String, String, u64)> = truth
        .into_iter()
        .zip(reading)
        .zip(count)
        .map(|((t, r), n)| (t, r, n as u64))
        .collect();
    let opts = tfz::LearnOptions {
        seeds: seeds_a.into_iter().zip(seeds_b).collect(),
        fixed: fixed_a
            .into_iter()
            .zip(fixed_b)
            .zip(fixed_cost)
            .map(|((a, b), c)| (a, b, c))
            .collect(),
        rounds: rounds.max(0) as usize,
        p_floor,
        min_multi: min_multi as u64,
        min_ratio,
        shrink,
        symmetric,
        prior: if prior.is_null() {
            None
        } else {
            let p: ExternalPtr<tfz::Costs> = prior.try_into()?;
            Some((*p).clone())
        },
        prior_weight,
    };
    let table = tfz::learn_costs(&pairs, &opts).map_err(fail)?;
    let na = |x: Option<f64>| x.map_or(Rfloat::na(), Rfloat::from);
    Ok(list!(
        a = table.iter().map(|e| e.a.clone()).collect::<Vec<_>>(),
        b = table.iter().map(|e| e.b.clone()).collect::<Vec<_>>(),
        count = Doubles::from_values(table.iter().map(|e| na(e.count.map(|c| c as f64)))),
        p = Doubles::from_values(table.iter().map(|e| na(e.p))),
        cost = table.iter().map(|e| e.cost).collect::<Vec<_>>()
    ))
}

extendr_module! {
    mod tunedfuzz;
    fn costs_new;
    fn costs_read;
    fn costs_pretrained;
    fn costs_write;
    fn ptr_alive;
    fn costs_reversed;
    fn costs_info;
    fn distance_pairs;
    fn align_pair;
    fn index_new;
    fn index_extract;
    fn cdist_flat;
    fn learn;
}
