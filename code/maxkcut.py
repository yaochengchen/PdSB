#!/usr/bin/env python3
"""Max-k-cut problem: graphs (Gset / random), reference values, exact cost evaluation (numpy + torch).

Convention used by every solver: minimise the same-colour weight  S(c) = sum_{(i,j) in E} w_ij [c_i = c_j];
the cut is  C(c) = W_tot - S(c),  W_tot = sum_E w_ij.  Edges are stored once (i < j).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GSET_URL = "https://web.stanford.edu/~yyye/yyye/Gset/{name}"

# Best known Max-3-Cut / Max-4-Cut of G1-G5 (multiple operator heuristic, Ma & Hao, Ann. Oper. Res. 2017),
# as listed in Frederiksen, De Prins, Bienstman, arXiv:2605.06425, Supp. Table S6.1.
BKS = {
    ("G1", 3): 15165, ("G1", 4): 16803,
    ("G2", 3): 15172, ("G2", 4): 16809,
    ("G3", 3): 15173, ("G3", 4): 16806,
    ("G4", 3): 15184, ("G4", 4): 16814,
    ("G5", 3): 15193, ("G5", 4): 16816,
}

# Published results of arXiv:2605.06425 (Supp. Tables S7.28 / S7.29), relative optimality gap in %:
# (mean, median, min, SD) over 100 runs at the best hyper-parameters, raw dynamics (no local search).
PUBLISHED = {
    "ref-IM (2605.06425)": {
        ("G1", 3): (0.09, 0.09, 0.00, 0.05), ("G2", 3): (0.12, 0.11, 0.05, 0.04), ("G3", 3): (0.08, 0.08, 0.01, 0.04),
        ("G4", 3): (0.14, 0.14, 0.00, 0.05), ("G5", 3): (0.06, 0.03, 0.00, 0.07),
        ("G1", 4): (0.18, 0.19, 0.02, 0.06), ("G2", 4): (0.19, 0.20, 0.04, 0.06), ("G3", 4): (0.17, 0.17, 0.05, 0.06),
        ("G4", 4): (0.18, 0.18, 0.05, 0.05), ("G5", 4): (0.18, 0.18, 0.04, 0.06),
    },
    "NEC PM (2605.06425)": {
        ("G1", 3): (0.25, 0.24, 0.16, 0.05), ("G2", 3): (0.25, 0.23, 0.13, 0.11), ("G3", 3): (0.21, 0.17, 0.13, 0.08),
        ("G4", 3): (0.26, 0.20, 0.14, 0.09), ("G5", 3): (0.26, 0.25, 0.15, 0.05),
        ("G1", 4): (3.32, 3.29, 3.12, 0.11), ("G2", 4): (3.32, 3.28, 3.28, 0.12), ("G3", 4): (3.24, 3.21, 3.17, 0.08),
        ("G4", 4): (3.15, 3.15, 3.15, 0.04), ("G5", 4): (3.35, 3.32, 3.21, 0.12),
    },
}


@dataclass
class Graph:
    name: str
    n: int
    edges: np.ndarray   # (m, 2) int64, i < j
    w: np.ndarray       # (m,) float64

    @property
    def m(self) -> int:
        return len(self.w)

    @property
    def total_w(self) -> float:
        return float(self.w.sum())

    def degree_w(self) -> np.ndarray:
        d = np.zeros(self.n)
        np.add.at(d, self.edges[:, 0], self.w)
        np.add.at(d, self.edges[:, 1], self.w)
        return d

    def csr(self):
        """Symmetric CSR (indptr, indices, data) as numpy arrays."""
        import scipy.sparse as sp
        i, j = self.edges[:, 0], self.edges[:, 1]
        A = sp.coo_matrix((np.r_[self.w, self.w], (np.r_[i, j], np.r_[j, i])), shape=(self.n, self.n)).tocsr()
        A.sum_duplicates()
        A.sort_indices()
        return A


def _canon(edges: np.ndarray, w: np.ndarray, n: int):
    edges = np.asarray(edges, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    keep = edges[:, 0] != edges[:, 1]
    edges, w = edges[keep], w[keep]
    lo, hi = edges.min(1), edges.max(1)
    key = lo * n + hi
    uk, inv = np.unique(key, return_inverse=True)
    ws = np.zeros(len(uk))
    np.add.at(ws, inv, w)                  # merge duplicate edges
    nz = ws != 0
    return np.stack([uk[nz] // n, uk[nz] % n], 1), ws[nz]


def read_gset(path: str | Path, name: str | None = None) -> Graph:
    with open(path) as f:
        n, m = (int(v) for v in f.readline().split()[:2])
        data = np.loadtxt(f, dtype=np.float64, ndmin=2)
    if len(data) != m:
        raise ValueError(f"{path}: header says {m} edges, file has {len(data)}")
    e, w = _canon(data[:, :2].astype(np.int64) - 1, data[:, 2], n)
    return Graph(name or Path(path).name, n, e, w)


def download_gset(names, out_dir: str | Path = "data/gset") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = out / name
        if not p.exists() or p.stat().st_size == 0:
            print(f"downloading {name} ...", flush=True)
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(GSET_URL.format(name=name), tmp)
            tmp.rename(p)
        paths.append(p)
    return paths


def random_graph(n: int, p: float, seed: int = 0, weights: str = "one", name: str | None = None) -> Graph:
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    mask = rng.random(len(iu[0])) < p
    e = np.stack([iu[0][mask], iu[1][mask]], 1).astype(np.int64)
    w = np.ones(len(e)) if weights == "one" else rng.choice([-1.0, 1.0], len(e))
    return Graph(name or f"rand_n{n}_p{p}_s{seed}", n, e, w)


def load_graph(spec: str, data_dir: str | Path = "data/gset", download: bool = True) -> Graph:
    """'G1' (Gset, downloaded if missing), 'rand:n:p:seed[:pm1]', or a path to a Gset-format file."""
    if spec.startswith("rand:"):
        parts = spec.split(":")
        n, p, seed = int(parts[1]), float(parts[2]), int(parts[3])
        return random_graph(n, p, seed, "pm1" if len(parts) > 4 and parts[4] == "pm1" else "one", name=spec)
    if Path(spec).exists():
        return read_gset(spec)
    p = Path(data_dir) / spec
    if not p.exists():
        if not download:
            raise FileNotFoundError(p)
        download_gset([spec], data_dir)
    return read_gset(p, spec)


def same_weight_np(g: Graph, colors: np.ndarray) -> np.ndarray:
    """colors (n,) or (n, R) -> same-colour weight per replica (float64)."""
    c = np.asarray(colors)
    eq = c[g.edges[:, 0]] == c[g.edges[:, 1]]
    return (g.w[:, None] * eq.reshape(len(g.w), -1)).sum(0) if c.ndim > 1 else float((g.w * eq).sum())


def cut_np(g: Graph, colors: np.ndarray):
    return g.total_w - same_weight_np(g, colors)


def reference_cut(name: str, K: int):
    return BKS.get((name, K))
