"""Learns the general table from every training corpus, and one table per type from the corpora in that type.

Usage: build.py <pairs dir> <out dir>
"""
import sys

import polars as pl

from tables import SCRIPTS, TEST_ONLY, learn, load

pairs_dir, out = sys.argv[1:3]
corpora = load(pairs_dir)
train = sorted(c for c in corpora if c not in TEST_ONLY)
for name, names in [("general", train), *((s, [c for c in train if c in cs]) for s, cs in SCRIPTS.items())]:
    print(name, "learns from", ", ".join(names))
    t = learn(corpora, names)
    pl.DataFrame({"a": t["a"], "b": t["b"], "cost": t["cost"]}).write_csv(f"{out}/{name}.csv")
    print(f"{len(t['a'])} edits -> {out}/{name}.csv")
