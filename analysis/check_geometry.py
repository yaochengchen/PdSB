"""Numerical consistency audit for the mathematical statements in the paper; no GPU benchmark.

Checks (CPU only, NumPy):
  1. regular-simplex identities, Euclidean projection onto the simplex hull, normal-cone condition,
     radial identity and the frozen-field vertex-selection threshold, for K = 2, 3, 4, 5, 6, 8;
  2. exact Potts energy and the zero-momentum fixed point <=> 1-flip local optimum equivalence
     on every colouring of a small signed graph at three time steps;
  3. the eight PdSB values of the record table against code/results_pdsb/best/*.json and
     certificates/results.csv (including the G81 K=4 external value 16670).

Usage (from anywhere):  python analysis/check_geometry.py   -> analysis/geometry_validation.txt
Originally written for the 2026-10-15 manuscript revision; paths updated for this repository layout.
"""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent          # analysis/
CODE = ROOT.parent / "code"
RELEASE = ROOT.parent / "certificates"           # Zenodo 2.0.0 release files (results.csv)
rng = np.random.default_rng(314159)
lines = []


def record(text):
    lines.append(text)
    print(text)


def vertices(k):
    a = np.eye(k) - 1 / k
    _, _, vt = np.linalg.svd(a)
    v = a @ vt[:k - 1].T
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def projection(x, v):
    k = len(v)
    b = ((k - 1) * (x @ v.T) + 1) / k
    u = np.sort(b, axis=-1)[..., ::-1]
    cs = np.cumsum(u, axis=-1) - 1
    js = np.arange(1, k + 1)
    rho = (u - cs / js > 0).sum(axis=-1) - 1
    theta = np.take_along_axis(cs, rho[..., None], -1) / (rho[..., None] + 1)
    beta = np.maximum(b - theta, 0)
    return beta @ v, beta


for k in (2, 3, 4, 5, 6, 8):
    v = vertices(k)
    assert np.allclose(v @ v.T, (k * np.eye(k) - 1) / (k - 1))
    x = rng.normal(size=(1000, k - 1))
    y, beta = projection(x, v)
    assert np.min(beta) >= 0 and np.allclose(beta.sum(-1), 1)
    # The residual belongs to the normal cone at the projected point.
    for ec in v:
        assert np.max(np.sum((x - y) * (ec - y), -1)) < 1e-10
    assert np.allclose((y * y).sum(-1), (k * (beta * beta).sum(-1) - 1) / (k - 1))
    # Strict frozen-field selection at the derived finite lambda threshold.
    f = rng.normal(size=(100, k - 1))
    scores = f @ v.T
    best = scores.argmax(-1)
    gaps = np.sort(scores, axis=-1)[:, -1] - np.sort(scores, axis=-1)[:, -2]
    lam = 0.4 * (k - 1) * gaps / k
    selected, _ = projection(f / lam[:, None], v)
    assert np.allclose(selected, v[best], atol=1e-10)
    # Zero-field and tied-field cases need not select a vertex.
    centre, _ = projection(np.zeros((1, k - 1)), v)
    assert np.allclose(centre, 0)
record("PASS: simplex identities, projection normal cones, radial identity and frozen-field threshold (K=2,3,4,5,6,8).")

count = 0
for k in (2, 3, 4, 5):
    v = vertices(k)
    w = np.array([[0, 2, -1, 0], [2, 0, 3, -2], [-1, 3, 0, 1], [0, -2, 1, 0]])
    for colours in itertools.product(range(k), repeat=4):
        c = np.array(colours)
        h = w @ np.eye(k)[c]
        f = -h @ v
        same = sum(w[i, j] for i in range(4) for j in range(i + 1, 4) if c[i] == c[j])
        energy = sum(w[i, j] * (v[c[i]] @ v[c[j]]) for i in range(4) for j in range(i + 1, 4))
        assert np.isclose(energy, (k * same - np.triu(w, 1).sum()) / (k - 1))
        local = np.all(h[np.arange(4), c] <= h.min(-1))
        for dt in (0.07, 0.4, 1.8):
            kicked = dt * f
            drifted = v[c] + dt * kicked
            projected, _ = projection(drifted, v)
            residual = drifted - projected
            nr = np.linalg.norm(residual, axis=-1, keepdims=True)
            nn = residual / np.maximum(nr, 1e-30)
            after = kicked - np.maximum((kicked * nn).sum(-1, keepdims=True), 0) * nn
            fixed = np.allclose(projected, v[c], atol=1e-9) and np.allclose(after, 0, atol=1e-9)
            assert bool(fixed) == bool(local)
        count += 1
record(f"PASS: exact Potts energy and zero-momentum fixed-point/1-flip equivalence on {count} signed-graph colourings at three time steps.")

rows = {(r["instance"], int(r["K"])): r for r in csv.DictReader((RELEASE / "results.csv").open(encoding="utf-8-sig"))}
expected = {("G1", 5):17706, ("G2", 5):17710, ("G3", 5):17705, ("G4", 5):17710,
            ("G22", 4):18787, ("G55", 3):12436, ("G81", 3):16498, ("G81", 4):16654}
for (graph, k), cut in expected.items():
    meta = json.loads((CODE / "results_pdsb/best" / f"{graph}_K{k}.json").read_text(encoding="utf-8"))
    assert meta["cut"] == cut and meta["method"] in ("pdsb", "pdsb_fused")
    assert meta["kind"] == "best"
    assert int(rows[(graph, k)]["cut"]) >= cut
assert int(rows[("G81", 4)]["previous_best"]) == 16670
record("PASS: all eight PdSB table values match raw best-output metadata; release values and G81 K=4 baseline match results.csv.")
record("Scope: these are CPU mathematical and metadata checks. GH200 performance and GPU trajectories were not rerun.")
(ROOT / "geometry_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
