use std::path::PathBuf;

use pyo3::exceptions::{PyImportError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict, PyString, PyTuple};

fn value_error(e: tfz::Error) -> PyErr {
    PyValueError::new_err(e.0)
}

/// Collects an iterable of str, each passed through `processor` if given. Refuses a bare str.
fn strings(obj: &Bound<'_, PyAny>, processor: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err("expected an iterable of str, got a str"));
    }
    obj.try_iter()?.map(|x| process(&x?, processor)).collect()
}

fn process(s: &Bound<'_, PyAny>, processor: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    match processor {
        Some(f) => f.call1((s,))?.extract(),
        None => s.extract(),
    }
}

fn tuples<T: for<'py> FromPyObjectOwned<'py>>(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<T>> {
    match obj {
        None => Ok(Vec::new()),
        Some(obj) => obj.try_iter()?.map(|x| x?.extract().map_err(Into::into)).collect(),
    }
}

/// -1 means all cores, which tfz calls 0.
fn workers(n: i64) -> PyResult<usize> {
    match n {
        -1 => Ok(0),
        n if n >= 1 => Ok(n as usize),
        _ => Err(PyValueError::new_err("workers must be -1 (all cores) or at least 1")),
    }
}

/// Builds a float64 numpy array of the given shape. Only this function imports numpy, so it stays optional.
fn ndarray<'py>(py: Python<'py>, values: &[f64], shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let np = py
        .import("numpy")
        .map_err(|_| PyImportError::new_err("cdist and cpdist return numpy arrays; install numpy"))?;
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let buf = PyByteArray::new(py, &bytes);
    np.call_method1("frombuffer", (buf, np.getattr("float64")?))?
        .call_method1("reshape", (PyTuple::new(py, shape)?,))
}

/// A cost table maps edits of Unicode code points to costs. Edits missing from it cost `default`.
///
/// Each row (a, b, cost) is one edit. ("R", "N", 0.22) substitutes, ("n", "", 0.6) deletes one
/// character and ("", "n", 0.6) inserts it, and ("tz", "ß", 0.23) replaces one string
/// with another, both at least one character long.
/// With symmetric=True each row also gives the reverse edit. With symmetric=False a row only
/// replaces a in the first string of distance() by b in the second. A later row for the same
/// edit replaces an earlier one. Costs are case-sensitive.
#[pyclass(frozen, module = "tunedfuzz")]
struct Costs(tfz::Costs);

#[pymethods]
impl Costs {
    #[new]
    #[pyo3(signature = (rows = None, default = 1.0, *, symmetric = true))]
    fn new(rows: Option<&Bound<'_, PyAny>>, default: f64, symmetric: bool) -> PyResult<Self> {
        let rows: Vec<(String, String, f64)> = tuples(rows)?;
        tfz::Costs::with_symmetry(rows, default, symmetric)
            .map(Costs)
            .map_err(value_error)
    }

    /// Reads rows from the a, b and cost columns of a CSV file with a header.
    #[staticmethod]
    #[pyo3(signature = (path, default = 1.0, *, symmetric = true))]
    fn read_csv(path: PathBuf, default: f64, symmetric: bool) -> PyResult<Self> {
        tfz::Costs::read_csv(path, default, symmetric)
            .map(Costs)
            .map_err(value_error)
    }

    /// Loads a directed table shipped with the library, learned from public (truth, reading) pairs.
    /// "general" comes from every training corpus, "antiqua" and "fraktur" from the corpora printed
    /// in that type. The table edits truth into reading, so call distance(truth, reading), or build
    /// an Index of true names with costs.reversed(). Use it as is, or as the prior of learn_costs.
    #[staticmethod]
    #[pyo3(signature = (name = "general"))]
    fn pretrained(name: &str) -> PyResult<Self> {
        tfz::Costs::pretrained(name).map(Costs).map_err(value_error)
    }

    /// Writes the rows to a CSV file with columns a, b and cost, which read_csv reads back.
    /// The file does not record default or symmetric.
    fn to_csv(&self, path: PathBuf) -> PyResult<()> {
        self.0.write_csv(path).map_err(value_error)
    }

    /// Returns the table that edits the second string into the first, with every row turned around.
    fn reversed(&self) -> Self {
        Costs(self.0.reversed())
    }

    /// The rows in effect, one per edit, in the order they were last set.
    fn rows(&self) -> Vec<(String, String, f64)> {
        self.0.rows().to_vec()
    }

    /// Cost of every edit the table lacks.
    #[getter]
    fn default(&self) -> f64 {
        self.0.default_cost()
    }

    /// Whether each row also gives the reverse edit.
    #[getter]
    fn symmetric(&self) -> bool {
        self.0.is_symmetric()
    }

    fn __len__(&self) -> usize {
        self.0.rows().len()
    }

    fn __getnewargs_ex__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("symmetric", self.0.is_symmetric())?;
        Ok(((self.rows(), self.default()).into_pyobject(py)?, kwargs))
    }

    fn __repr__(&self) -> String {
        let sym = if self.0.is_symmetric() { "" } else { ", symmetric=False" };
        format!(
            "Costs(<{} edits>, default={:?}{sym})",
            self.0.rows().len(),
            self.0.default_cost()
        )
    }
}

/// Returns the cheapest cost of editing `a` into `b`. normalize=True divides it by the length
/// of the longer string. A distance above `score_cutoff` comes back as inf.
/// `processor`, if given, runs on both strings first.
#[pyfunction]
#[pyo3(signature = (a, b, costs, *, score_cutoff = None, normalize = false, processor = None))]
fn distance(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    costs: PyRef<'_, Costs>,
    score_cutoff: Option<f64>,
    normalize: bool,
    processor: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (a, b) = (process(a, processor)?, process(b, processor)?);
    Ok(if normalize {
        tfz::normalized_distance(&a, &b, &costs.0, score_cutoff)
    } else {
        tfz::distance(&a, &b, &costs.0, score_cutoff)
    })
}

/// Returns the edits behind distance(a, b) as (from a, to b, cost) tuples in order, without matches.
#[pyfunction]
#[pyo3(signature = (a, b, costs, *, processor = None))]
fn align(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    costs: PyRef<'_, Costs>,
    processor: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<(String, String, f64)>> {
    Ok(tfz::align(&process(a, processor)?, &process(b, processor)?, &costs.0))
}

/// Returns distance(query, name) for each query and name as a float64 numpy array of shape
/// (len(queries), len(names)). Other arguments work as in distance(). `workers` threads
/// compute it, and -1 uses all cores.
#[pyfunction]
#[pyo3(signature = (queries, names, costs, *, score_cutoff = None, normalize = false, processor = None, workers = 1))]
#[allow(clippy::too_many_arguments)]
fn cdist<'py>(
    py: Python<'py>,
    queries: &Bound<'py, PyAny>,
    names: &Bound<'py, PyAny>,
    costs: PyRef<'py, Costs>,
    score_cutoff: Option<f64>,
    normalize: bool,
    processor: Option<&Bound<'py, PyAny>>,
    workers: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let (queries, names) = (strings(queries, processor)?, strings(names, processor)?);
    let (w, costs) = (self::workers(workers)?, &costs.0);
    let flat = py.detach(|| tfz::cdist(&queries, &names, costs, score_cutoff, normalize, w));
    ndarray(py, &flat, &[queries.len(), names.len()])
}

/// Returns distance(queries[i], choices[i]) for each i as a float64 numpy array. Other arguments
/// work as in cdist(). Raises ValueError if the lengths differ.
#[pyfunction]
#[pyo3(signature = (queries, choices, costs, *, score_cutoff = None, normalize = false, processor = None, workers = 1))]
#[allow(clippy::too_many_arguments)]
fn cpdist<'py>(
    py: Python<'py>,
    queries: &Bound<'py, PyAny>,
    choices: &Bound<'py, PyAny>,
    costs: PyRef<'py, Costs>,
    score_cutoff: Option<f64>,
    normalize: bool,
    processor: Option<&Bound<'py, PyAny>>,
    workers: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let (queries, choices) = (strings(queries, processor)?, strings(choices, processor)?);
    let (w, costs) = (self::workers(workers)?, &costs.0);
    let flat = py
        .detach(|| tfz::cpdist(&queries, &choices, costs, score_cutoff, normalize, w))
        .map_err(value_error)?;
    ndarray(py, &flat, &[flat.len()])
}

type Hit = (String, f64, usize);

/// Finds the names in a fixed list nearest to a query. The search is exact, and a name's cost is
/// distance(query, name), so lower is closer. `score_cutoff` and `normalize` work as in distance().
///
/// Results are (name, cost, index) tuples, as in rapidfuzz, ordered by cost, then index.
/// `processor` runs on the names here and on each query, and results give the names as passed.
/// Searches release the GIL, so threads scale. The *_many methods run on `workers` threads, and
/// -1 uses all cores.
#[pyclass(frozen, module = "tunedfuzz")]
struct Index {
    index: tfz::Index,
    names: Vec<String>,
    costs: Py<Costs>,
    processor: Option<Py<PyAny>>,
}

impl Index {
    fn hit(&self, m: tfz::Match) -> Hit {
        (self.names[m.index].clone(), m.cost, m.index)
    }

    fn query(&self, py: Python<'_>, q: &Bound<'_, PyAny>) -> PyResult<String> {
        process(q, self.processor.as_ref().map(|f| f.bind(py)))
    }

    fn many(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyAny>,
        limit: Option<usize>,
        score_cutoff: Option<f64>,
        normalize: bool,
        workers: i64,
    ) -> PyResult<Vec<Vec<tfz::Match>>> {
        let queries = strings(queries, self.processor.as_ref().map(|f| f.bind(py)))?;
        let w = self::workers(workers)?;
        Ok(py.detach(|| self.index.extract_many(&queries, limit, score_cutoff, normalize, w)))
    }
}

#[pymethods]
impl Index {
    #[new]
    #[pyo3(signature = (names, costs, *, processor = None))]
    fn new(names: &Bound<'_, PyAny>, costs: Bound<'_, Costs>, processor: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let names = strings(names, None)?;
        let keys = match &processor {
            Some(f) => names
                .iter()
                .map(|n| f.call1((n,))?.extract())
                .collect::<PyResult<Vec<String>>>()?,
            None => names.clone(),
        };
        Ok(Index {
            index: tfz::Index::new(keys, &costs.get().0),
            names,
            costs: costs.unbind(),
            processor: processor.map(Bound::unbind),
        })
    }

    fn __len__(&self) -> usize {
        self.names.len()
    }

    fn __getnewargs_ex__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("processor", &self.processor)?;
        Ok((
            (self.names.clone(), self.costs.clone_ref(py)).into_pyobject(py)?,
            kwargs,
        ))
    }

    /// Returns the cheapest name within `score_cutoff`, or None. Ties go to the lowest index.
    #[pyo3(signature = (query, *, score_cutoff = None, normalize = false))]
    fn extract_one(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        score_cutoff: Option<f64>,
        normalize: bool,
    ) -> PyResult<Option<Hit>> {
        let q = self.query(py, query)?;
        let found = py.detach(|| self.index.extract_one(&q, score_cutoff, normalize));
        Ok(found.map(|m| self.hit(m)))
    }

    /// Returns the `limit` cheapest names within `score_cutoff`, or all of them for limit=None.
    #[pyo3(signature = (query, *, limit = Some(5), score_cutoff = None, normalize = false))]
    fn extract(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        limit: Option<usize>,
        score_cutoff: Option<f64>,
        normalize: bool,
    ) -> PyResult<Vec<Hit>> {
        let q = self.query(py, query)?;
        let found = py.detach(|| self.index.extract(&q, limit, score_cutoff, normalize));
        Ok(found.into_iter().map(|m| self.hit(m)).collect())
    }

    /// Runs extract_one for each query.
    #[pyo3(signature = (queries, *, score_cutoff = None, normalize = false, workers = 1))]
    fn extract_one_many(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyAny>,
        score_cutoff: Option<f64>,
        normalize: bool,
        workers: i64,
    ) -> PyResult<Vec<Option<Hit>>> {
        let found = self.many(py, queries, Some(1), score_cutoff, normalize, workers)?;
        Ok(found
            .into_iter()
            .map(|ms| ms.into_iter().next().map(|m| self.hit(m)))
            .collect())
    }

    /// Runs extract for each query.
    #[pyo3(signature = (queries, *, limit = Some(5), score_cutoff = None, normalize = false, workers = 1))]
    fn extract_many(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyAny>,
        limit: Option<usize>,
        score_cutoff: Option<f64>,
        normalize: bool,
        workers: i64,
    ) -> PyResult<Vec<Vec<Hit>>> {
        let found = self.many(py, queries, limit, score_cutoff, normalize, workers)?;
        Ok(found
            .into_iter()
            .map(|ms| ms.into_iter().map(|m| self.hit(m)).collect())
            .collect())
    }
}

/// Learns edit costs from (truth, reading, count) pairs.
///
/// Each of the `rounds` rounds aligns every misread pair under the current costs, counts the
/// edits, and sets
///
///     p    = max(c - shrink * sqrt(c), 0) / n
///     cost = min(1, log p / log p_floor), rounded to 4 decimals
///
/// for an edit seen c times, where n counts the occurrences of the edited strings in all pairs.
/// Candidate multi-character edits are the `seeds` and the 2:1, 1:2 and 2:2 replacements difflib
/// finds at least `min_multi` times. Pairs with a difflib ratio below `min_ratio` are left out.
///
/// With symmetric=False, each direction gets its own cost. An edit (a, b) replaces a in the truth
/// by b in the reading, and n counts the occurrences of a in the truths (for an insertion, the
/// places it could go). Build the table with Costs(..., symmetric=False).
///
/// A `prior` table, such as Costs.pretrained("general"), shrinks each probability towards the
/// prior's p0 = p_floor ** cost. The prior counts as `prior_weight` occurrences, which gives
///
///     p = (max(c - shrink * sqrt(c), 0) + prior_weight * p0) / (n + prior_weight)
///
/// So the prior's cost wins for an edit the data rarely shows, and the data's count wins for an
/// edit it shows often. A directed prior needs symmetric=False.
///
/// Returns a dict of lists under a, b, count, p and cost. It holds the edits that cost less
/// than 1, by cost, then the `fixed` (a, b, cost) edits as given, with count and p None.
#[pyfunction]
#[pyo3(signature = (pairs, *, seeds = None, fixed = None, rounds = 4, p_floor = 1e-5, min_multi = 20, min_ratio = 0.5, shrink = 2.0, symmetric = true, prior = None, prior_weight = 30.0))]
#[allow(clippy::too_many_arguments)]
fn learn_costs<'py>(
    py: Python<'py>,
    pairs: &Bound<'py, PyAny>,
    seeds: Option<&Bound<'py, PyAny>>,
    fixed: Option<&Bound<'py, PyAny>>,
    rounds: usize,
    p_floor: f64,
    min_multi: u64,
    min_ratio: f64,
    shrink: f64,
    symmetric: bool,
    prior: Option<PyRef<'py, Costs>>,
    prior_weight: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let pairs: Vec<(String, String, u64)> = tuples(Some(pairs))?;
    let opts = tfz::LearnOptions {
        seeds: tuples(seeds)?,
        fixed: tuples(fixed)?,
        rounds,
        p_floor,
        min_multi,
        min_ratio,
        shrink,
        symmetric,
        prior: prior.map(|p| p.0.clone()),
        prior_weight,
    };
    let table = py.detach(|| tfz::learn_costs(&pairs, &opts)).map_err(value_error)?;
    let out = PyDict::new(py);
    out.set_item("a", table.iter().map(|e| e.a.clone()).collect::<Vec<_>>())?;
    out.set_item("b", table.iter().map(|e| e.b.clone()).collect::<Vec<_>>())?;
    out.set_item("count", table.iter().map(|e| e.count).collect::<Vec<_>>())?;
    out.set_item("p", table.iter().map(|e| e.p).collect::<Vec<_>>())?;
    out.set_item("cost", table.iter().map(|e| e.cost).collect::<Vec<_>>())?;
    Ok(out)
}

/// tunedfuzz measures a weighted edit distance with multi-character edits, finds the nearest
/// names under it exactly, and learns its costs from (truth, reading) pairs.
#[pymodule]
mod tunedfuzz {
    #[pymodule_export]
    use super::{Costs, Index, align, cdist, cpdist, distance, learn_costs};
}
