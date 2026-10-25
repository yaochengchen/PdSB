#!/usr/bin/env python3
"""Independent check of a saved colouring: reads the Gset file directly (plain Python, no project code).

    python verify_cert.py results/E1/certificates/G1_K4_16804_*.npy  [--gset data/gset]
"""
import argparse
import re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--gset", default="data/gset")
a = ap.parse_args()
for f in a.files:
    name, K = re.match(r"(G\d+)_K(\d+)_", Path(f).name).groups()
    col = np.load(f).astype(int)
    lines = open(Path(a.gset) / name).read().split("\n")
    n, m = map(int, lines[0].split()[:2])
    assert len(col) == n and col.min() >= 0 and col.max() < int(K), "not a valid K-colouring"
    cut = 0.0
    for line in lines[1:m + 1]:
        i, j, w = line.split()[:3]
        if col[int(i) - 1] != col[int(j) - 1]:
            cut += float(w)
    print(f"{Path(f).name}: graph {name} n={n} m={m} K={K} colours used={len(set(col))}  cut = {cut:.0f}")
