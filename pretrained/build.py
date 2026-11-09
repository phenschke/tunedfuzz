"""Learns the general table from every training corpus, and one table per type from the corpora in that type.

Usage: build.py <pairs dir> <out dir> [table ...]

Naming tables builds only those. A type table with none of its corpora in the pairs dir is skipped.
"""
import sys

import polars as pl

from tables import KURRENT_ONLY, SCRIPTS, TEST_ONLY, learn, load

pairs_dir, out, *only = sys.argv[1:]
corpora = load(pairs_dir)
train = sorted(c for c in corpora if c not in TEST_ONLY)
general = [c for c in train if c not in KURRENT_ONLY]
for name, names in [("general", general), *((s, [c for c in train if c in cs]) for s, cs in SCRIPTS.items())]:
    if (only and name not in only) or not names:
        continue
    print(name, "learns from", ", ".join(names))
    t = learn(corpora, names)
    pl.DataFrame({"a": t["a"], "b": t["b"], "cost": t["cost"]}).write_csv(f"{out}/{name}.csv")
    print(f"{len(t['a'])} edits -> {out}/{name}.csv")
