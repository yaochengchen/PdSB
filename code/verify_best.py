#!/usr/bin/env python3
"""Independent check of the best-colouring archive results/best/ (written by run.py for every (graph, K)).

    python verify_best.py                      # results/best, Gset files in data/gset
    python verify_best.py --best results/best --gset data/gset

For each <graph>_K<K>.npy + .json: reads the raw Gset file with plain Python (no project code), recomputes the cut
edge by edge and compares it with the stored cut and with the best known value.  Non-Gset graphs (rand:...) are
listed but not re-verified.  Exit code 1 if any stored cut does not match.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--best", default="results/best")
ap.add_argument("--gset", default="data/gset")
a = ap.parse_args()

bad = 0
metas = sorted(Path(a.best).glob("*.json"), key=lambda p: (len(p.stem), p.stem))
if not metas:
    print(f"no archive in {a.best}")
for mp in metas:
    meta = json.loads(mp.read_text())
    name, K = meta["graph"], int(meta["K"])
    col = np.load(mp.with_suffix(".npy")).astype(int)
    src = f"{meta['plan']}/{meta['method']} T={meta['T']} {json.dumps(meta['params'])} ({meta['kind']})"
    gfile = Path(a.gset) / name
    if not gfile.exists():
        print(f"{name:<16} K={K}  stored {meta['cut']:.0f}  (not a Gset file here, not re-verified)  {src}")
        continue
    lines = gfile.read_text().split("\n")
    n, m = map(int, lines[0].split()[:2])
    ok_col = len(col) == n and col.min() >= 0 and col.max() < K
    cut = 0.0
    for line in lines[1:m + 1]:
        i, j, w = line.split()[:3]
        if col[int(i) - 1] != col[int(j) - 1]:
            cut += float(w)
    ok = ok_col and abs(cut - meta["cut"]) < 1e-6
    bad += not ok
    ref = meta.get("ref")
    vs = "" if ref is None else f"  BKS {ref}: " + ("NEW BEST" if cut > ref else "= BKS" if cut == ref else f"-{ref - cut:.0f}")
    print(f"{name:<16} K={K}  cut {cut:.0f}  stored {meta['cut']:.0f}  {'OK' if ok else 'MISMATCH'}"
          f"  colours used {len(set(col.tolist()))}{vs}  {src}")
sys.exit(1 if bad else 0)
