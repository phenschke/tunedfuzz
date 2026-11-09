import csv
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

import tunedfuzz as tfz

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures"

with open(FIXTURES / "costs.csv", encoding="utf-8", newline="") as f:
    TOY_ROWS = [(r["a"], r["b"], float(r["cost"])) for r in csv.DictReader(f)]
TOY = tfz.Costs(TOY_ROWS)

# DIRECTED charges more for each toy edit in reverse and allows B -> V one way only
DIRECTED_ROWS = [r for a, b, c in TOY_ROWS for r in ((a, b, c), (b, a, min(c * 1.7, 1.0)))] + [("B", "V", 0.05)]
DIRECTED = tfz.Costs(DIRECTED_ROWS, symmetric=False)


def random_names(n, seed):
    rng = random.Random(seed)
    letters = "aeinrstuldbhkzßRNBV"
    return ["".join(rng.choices(letters, k=rng.randint(0, 8))) for _ in range(n)]


def oracle(a, b, rows, default=1.0, symmetric=True):
    """Computes the reference DP. d[i][j] is the cheapest edit of a[:i] into b[:j]."""
    sub, dele, ins, multi = {}, {}, {}, {}
    for x, y, c in rows:
        edits = [(x, y), (y, x)] if symmetric else [(x, y)]
        for u, v in edits:
            if u == v:
                continue
            if not v:
                dele[u] = c
            elif not u:
                ins[v] = c
            elif len(u) == 1 and len(v) == 1:
                sub[u, v] = c
            else:
                multi[u] = [op for op in multi.get(u, []) if op[0] != v] + [(v, c)]
    n, m = len(a), len(b)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        d[0][j] = d[0][j - 1] + ins.get(b[j - 1], default)
    for i in range(1, n + 1):
        d[i][0] = d[i - 1][0] + dele.get(a[i - 1], default)
        for j in range(1, m + 1):
            best = d[i - 1][j - 1] if a[i - 1] == b[j - 1] else d[i - 1][j - 1] + sub.get((a[i - 1], b[j - 1]), default)
            best = min(best, d[i - 1][j] + dele.get(a[i - 1], default), d[i][j - 1] + ins.get(b[j - 1], default))
            for k in range(1, i + 1):
                for v, c in multi.get(a[i - k:i], ()):
                    if len(v) <= j and b[j - len(v):j] == v:
                        best = min(best, d[i - k][j - len(v)] + c)
            d[i][j] = best
    return d[n][m]


def test_fixtures():
    with open(FIXTURES / "distances.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert tfz.distance(r["a"], r["b"], TOY) == pytest.approx(float(r["expected"])), r


@pytest.mark.parametrize("rows", [TOY_ROWS, [], [("sch", "f", 0.3), ("ß", "ssz", 0.2), ("s", "", 0.4)]])
def test_distance_is_bitwise_oracle(rows):
    costs = tfz.Costs(rows)
    names = random_names(2000, seed=1) + ["Mischa", "Mifa", "Faßl", "Fasszl", "Niederhuber" * 5, "𝔄ae", "Aä"]
    for a, b in zip(names, names[1:]):
        assert tfz.distance(a, b, costs) == oracle(a, b, rows), (a, b)


def test_score_cutoff():
    assert tfz.distance("Nußbaum", "Rutzbaum", TOY, score_cutoff=0.45) == 0.45
    assert tfz.distance("Nußbaum", "Rutzbaum", TOY, score_cutoff=0.44) == math.inf


def test_align_explains_distance():
    assert tfz.align("Nußbaum", "Rutzbaum", TOY) == [("N", "R", 0.2), ("ß", "tz", 0.25)]
    names = random_names(200, seed=2)
    for a, b in zip(names, names[1:]):
        assert sum(c for _, _, c in tfz.align(a, b, TOY)) == pytest.approx(tfz.distance(a, b, TOY))


def test_costs_round_trip(tmp_path):
    path = tmp_path / "costs.csv"
    path.write_text("a,b,count,cost\n" + "".join(f'"{a}","{b}",1,{c}\n' for a, b, c in TOY_ROWS), encoding="utf-8")
    assert tfz.Costs.read_csv(path).rows() == TOY.rows() == TOY_ROWS
    assert len(TOY) == len(TOY_ROWS) and TOY.default == 1.0
    with pytest.raises(ValueError):
        tfz.Costs([("ab", "", 0.5)])
    for costs in (TOY, DIRECTED, tfz.Costs([("a,b", '"x"', 1 / 3), ("ß", "", 0.1)])):
        costs.to_csv(path)
        assert tfz.Costs.read_csv(path, symmetric=costs.symmetric).rows() == costs.rows()


def test_normalized_distance():
    assert tfz.distance("Nußbaum", "Rutzbaum", TOY, normalize=True) == 0.45 / 8
    assert tfz.distance("", "", TOY, normalize=True) == 0.0
    assert tfz.distance("Nußbaum", "Rutzbaum", TOY, normalize=True, score_cutoff=0.05) == math.inf


@pytest.mark.parametrize(("normalize", "cutoff"), [(False, 0.5), (False, 2.0), (False, None), (True, 0.2), (True, None)])
@pytest.mark.parametrize("costs", [TOY, DIRECTED], ids=["symmetric", "directed"])
def test_index_is_brute_force(normalize, cutoff, costs):
    names = random_names(400, seed=3)
    names += names[:20] + [""]
    ix = tfz.Index(names, costs)
    queries = random_names(100, seed=4)
    limit = math.inf if cutoff is None else cutoff
    kw = dict(score_cutoff=cutoff, normalize=normalize)
    for q in queries:
        want = sorted((tfz.distance(q, n, costs, normalize=normalize), i) for i, n in enumerate(names))
        want = [(names[i], d, i) for d, i in want if d <= limit]
        assert ix.extract(q, limit=None, **kw) == want
        assert ix.extract(q, limit=3, **kw) == want[:3]
        assert ix.extract_one(q, **kw) == (want[0] if want else None)
    assert ix.extract_one_many(queries, workers=4, **kw) == [ix.extract_one(q, **kw) for q in queries]
    assert ix.extract_many(queries, limit=2, workers=-1, **kw) == [ix.extract(q, limit=2, **kw) for q in queries]


def test_cdist_and_cpdist():
    queries, names = random_names(30, seed=5), random_names(40, seed=6)
    for costs in (TOY, DIRECTED):
        for kw in ({}, {"normalize": True, "score_cutoff": 0.3}):
            m = tfz.cdist(queries, names, costs, workers=2, **kw)
            assert isinstance(m, np.ndarray) and m.dtype == np.float64 and m.shape == (30, 40)
            assert m.tolist() == [[tfz.distance(q, n, costs, **kw) for n in names] for q in queries]
            p = tfz.cpdist(queries, names[:30], costs, workers=2, **kw)
            assert p.shape == (30,)
            assert p.tolist() == [tfz.distance(q, n, costs, **kw) for q, n in zip(queries, names)]
    assert tfz.cdist([], names, TOY).shape == (0, 40)
    with pytest.raises(ValueError):
        tfz.cpdist(queries, names, TOY)


def test_processor():
    names = ["MÜLLER", "Huber", "Nußbaum"]
    ix = tfz.Index(names, TOY, processor=str.lower)
    assert ix.extract_one("müller") == ("MÜLLER", 0.0, 0)
    assert tfz.distance("HUBER", "huber", TOY, processor=str.lower) == 0.0
    assert tfz.cdist(["HUBER"], names, TOY, processor=str.lower)[0, 1] == 0.0
    assert tfz.cpdist(["HUBER"], ["huber"], TOY, processor=str.lower).tolist() == [0.0]
    assert tfz.align("NUẞBAUM", "nußbaum", TOY, processor=str.lower) == []


def test_pickle():
    for costs in (TOY, DIRECTED):
        back = pickle.loads(pickle.dumps(costs))
        assert back.rows() == costs.rows() and back.symmetric == costs.symmetric and back.default == costs.default
    ix = tfz.Index(["MÜLLER", "Huber"], DIRECTED, processor=str.lower)
    back = pickle.loads(pickle.dumps(ix))
    assert len(back) == 2 and back.extract("müller") == ix.extract("müller")


def test_directed_tables():
    names = random_names(1000, seed=7)
    reversed_ = DIRECTED.reversed()
    assert not DIRECTED.symmetric and TOY.symmetric and not reversed_.symmetric
    assert repr(DIRECTED) == f"Costs(<{len(DIRECTED)} edits>, default=1.0, symmetric=False)"
    for a, b in zip(names[::2], names[1::2]):
        d = tfz.distance(a, b, DIRECTED)
        assert d == oracle(a, b, DIRECTED_ROWS, symmetric=False), (a, b)
        assert d == tfz.distance(b, a, reversed_)
    assert tfz.distance("Hans", "Has", tfz.Costs([("n", "", 0.3)], symmetric=False)) == 0.3
    assert tfz.distance("Has", "Hans", tfz.Costs([("n", "", 0.3)], symmetric=False)) == 1.0


def test_refuses_a_bare_string():
    with pytest.raises(TypeError):
        tfz.Index("Huber", TOY)


def test_learn_costs():
    truth = ["Rauch", "Rieger", "Seitz", "Moritz", "Lutz", "Huber", "Rotzer", "Kurz"]
    pairs = [(t, t, 50) for t in truth]
    pairs += [(t, t.replace("R", "N", 1).replace("tz", "ß"), 10) for t in truth if "R" in t or "tz" in t]
    table = tfz.learn_costs(pairs, seeds=[("ll", "l")], fixed=[("ae", "ä", 0.1)], min_multi=5)
    assert list(table) == ["a", "b", "count", "p", "cost"]
    learned = {(a, b): c for a, b, c in zip(table["a"], table["b"], table["cost"])}
    assert learned["N", "R"] < 0.5 and learned["tz", "ß"] < 0.5 and learned["ae", "ä"] == 0.1
    assert table["count"][-1] is None
    costs = tfz.Costs(zip(table["a"], table["b"], table["cost"]))
    assert tfz.distance("Rotzer", "Noßer", costs) == learned["N", "R"] + learned["tz", "ß"]


def test_learn_directed_costs():
    truth = ["Rauch", "Rieger", "Seitz", "Moritz", "Lutz", "Huber", "Rotzer", "Kurz", "Nagl", "Bauß"]
    pairs = [(t, t, 50) for t in truth]
    pairs += [(t, t.replace("R", "N", 1).replace("tz", "ß"), 10) for t in truth if "R" in t or "tz" in t]
    table = tfz.learn_costs(pairs, min_multi=5, symmetric=False)
    learned = {(a, b): c for a, b, c in zip(table["a"], table["b"], table["cost"])}
    assert learned["R", "N"] < 0.5 and learned["tz", "ß"] < 0.5
    assert ("N", "R") not in learned and ("ß", "tz") not in learned
    costs = tfz.Costs(zip(table["a"], table["b"], table["cost"]), symmetric=False)
    assert tfz.distance("Rotzer", "Noßer", costs) == learned["R", "N"] + learned["tz", "ß"]
    assert tfz.distance("Noßer", "Rotzer", costs) == 3.0


def test_pretrained_and_prior():
    general = tfz.Costs.pretrained()
    assert not general.symmetric and len(general) > 100
    assert tfz.distance("Rauch", "Nauch", general) < 1
    with pytest.raises(ValueError):
        tfz.Costs.pretrained("nonexistent")

    # Fraktur readings turn V into B, Antiqua readings rarely do
    fraktur, antiqua = tfz.Costs.pretrained("fraktur"), tfz.Costs.pretrained("antiqua")
    assert tfz.distance("V", "B", fraktur) < 0.2 < tfz.distance("V", "B", antiqua)

    # A heavy prior keeps its costs where the data is silent
    pairs = [("Rauch", "Nauch", 3), ("Huber", "Huber", 30)]
    table = tfz.learn_costs(pairs, symmetric=False, prior=general, prior_weight=1e9)
    learned = {(a, b): c for a, b, c in zip(table["a"], table["b"], table["cost"])}
    prior = {(a, b): c for a, b, c in general.rows()}
    assert abs(learned["e", "c"] - prior["e", "c"]) < 1e-3
    with pytest.raises(ValueError):
        tfz.learn_costs(pairs, prior=general)
