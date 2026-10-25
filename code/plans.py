#!/usr/bin/env python3
"""Experiment plans for the PdSB paper (two GH200; run order = importance).

Framing: PdSB is a new algorithm.  The reference one-hot IM of arXiv:2605.06425 is compared mainly through its
PUBLISHED numbers (Supp. Tables S7.28/29, in maxkcut.PUBLISHED); we run it ourselves only for a reproduction
check (G1) and for the time axis.

Resume is per SETTING (run.py): a planned job only runs the grid points that no finished job with the same
(method, graph, K, T, seed, runs) has run yet.  Widening a grid therefore only runs the new points, and a job
interrupted by a kill is simply run again.

  E1   protocol of arXiv:2605.06425 on G1-G5, K=3,4: every method tuned on its grid, 100 runs per setting,
       metric = best cut seen along each run.  Grids widened 2026-10-14 where E1 optima sat on a grid edge.
       ref_im: their full grid + an extension above their (B/A)n range on G1 (our optimum sat on their edge),
       plus any (graph, K) already finished.
  E1v  the best setting of every (method, graph, K, T) in E1 on a FRESH seed, 1000 runs (unbiased numbers,
       Mann-Whitney tests).  ref_im only on G1.
  E2   (old, 100 runs per budget: GPU under-used; kept for reference, not in --plan all)
  E2L  time-to-quality with LARGE batches (1000 runs per budget, one setting per (method, graph, K) from E1):
       PdSB fused, PdSB PyTorch, one-hot dSB, CPU SA on G1-G5; ref_im on G1.  CPU jobs run after the GPU jobs
       (no contention in the timing).
  E3   K = 5, 6, 8 on G1-G5 (reference = best found), widened grids.
  E3v  best E3 setting of PdSB fused / SA / one-hot dSB at T=1e5 on a fresh seed, 1000 runs (tests + colourings).
  E4   larger Gset graphs (G43, G22, G55, G70, G81), K=3,4: PdSB fused and SA each tuned PER GRAPH on its grid
       (the old top-3-from-E1 settings are part of the grid and are not re-run).  n > 4096 -> sparse PdSB kernel.
  E4L  time-to-target on the E4 graphs: best E4 setting per (method, graph, K), 200 runs, CPU after GPU.
       Budgets (2026-10-15): PdSB 1e3 ... 1e6 (3e5, 1e6 added: equal wall-clock with SA 1e5 needs ~1e6 PdSB
       steps), SA 1e3 ... 3e5 (3e5 not on the proper-colouring instances).

Grid widening of 2026-10-15 (optima of the 2026-10-14 run on a grid edge; only the new points are run):
  SA (E1, E3)   T1 0.3-0.5 above the old upper edge 0.2, T0 0.5 and 6.0
  SA (E4)       colder: T0 0.25-0.5 x T1 0.01-0.05 (optimum sat on T0=0.5, T1=0.05); T1=0.3 for G22/G43
                (not on the proper-colouring instances G55 K4, G70 K3/K4: every setting reaches the optimum)
  one-hot (E1, E3)  xi 0.25-1.0 x dt 0.35-1.0 at the zeta / A chosen for that K (optimum sat on dt=0.5, xi edges)
  PdSB (E3)     xi 1.75, 2.0 (K=8 optimum on xi=1.5);  PdSB (E4) dt 1.25-2.0 (K=3 optimum on dt=1.0)

PdSB record hunt (separate results folder, default results_pdsb, its own best/ archive = PdSB-only solutions):
  H1   PdSB fused, T=1e6, 25 settings x 40 runs per target (tuning at the long horizon)
  H2   top-3 H1 settings per target (2 by mean, 1 by max cut), T=1e6, 1000 runs each
  H3   top-1 H1 setting per target, T=1e7, 200 runs
  targets: G22 K3/4, G43 K3/4, G55 K3, G81 K3/4, G1-G5 K5/6/8 (every instance whose best is not a proper
  colouring).  Best-seen colourings are also 1-opt polished (polish_best), still a pure PdSB(+1-opt) result.

Method names: pdsb (PyTorch), pdsb_fused (fused CUDA kernel), pbsb (ballistic variant), pdsb_sphere (wall
ablation), clock_dsb (planar Potts / embedding ablation), onehot_dsb (same integrator, one-hot encoding),
ref_im (2605.06425 Eq. 9), sa (CPU heat-bath Potts SA).
"""

from __future__ import annotations

import itertools

import numpy as np

G15 = ["G1", "G2", "G3", "G4", "G5"]
REFIM_REPRO = [("G1", 3), ("G1", 4)]          # where we re-run their IM (reproduction check, time axis)
REFIM_T = 1_000_000
TIMING_PLANS = {"E2L", "E4L"}                 # run.py runs their CPU jobs only after the GPU jobs


def grid(**kw):
    names = sorted(kw)
    return [dict(zip(names, v)) for v in itertools.product(*(kw[k] for k in names))]


def rng_(a, b, s):
    return [round(float(v), 4) for v in np.arange(a, b + 1e-9, s)]


def union(*gs):
    out, seen = [], set()
    for g in gs:
        for p in g:
            k = tuple(sorted(p.items()))
            if k not in seen:
                seen.add(k)
                out.append(p)
    return out


# grids: the first-run grid (kept, so finished settings are recognised) + the widening of 2026-10-14 ---------
G_PDSB = union(grid(xi=[0.5, 0.75, 1.0], dt=[0.35, 0.5, 0.75]),
               grid(xi=[0.35, 0.5, 0.75, 1.0, 1.25, 1.5], dt=[0.25, 0.35, 0.5, 0.75, 1.0]))
G_PBSB = union(grid(xi=[0.25, 0.35, 0.5], dt=[0.1, 0.15, 0.25]),
               grid(xi=[0.25, 0.35, 0.5, 0.75, 1.0], dt=[0.05, 0.1, 0.15, 0.25]))
G_SPHERE = union(grid(xi=[0.75, 1.0, 1.5], dt=[0.35, 0.5, 0.75]),
                 grid(xi=[0.75, 1.0, 1.5, 2.0, 3.0], dt=[0.2, 0.25, 0.35, 0.5, 0.75]))
G_CLOCK = union(grid(xi=[0.2, 0.35, 0.5, 1.0], dt=[0.5, 1.0]),
                grid(xi=[0.1, 0.2, 0.35, 0.5, 1.0, 1.5, 2.0], dt=[0.25, 0.35, 0.5, 1.0]))
G_ONEHOT = union(grid(xi=[0.5], dt=[0.5], A=[1, 2, 4], zeta=[0.5, 0.75]),
                 grid(xi=[0.5], dt=[0.5], A=[1, 2, 4], zeta=[0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.2]),
                 grid(xi=[0.5, 0.75], dt=[0.35, 0.5], A=[2], zeta=[0.5, 0.75, 1.0]))
G_SA = union(grid(T0=[2.0, 4.0], T1=[0.2]),
             grid(T0=[0.75, 1.0, 1.5, 2.0, 3.0, 4.0], T1=[0.05, 0.1, 0.2]))
G_SA_E4 = union(grid(T0=[2.0, 4.0], T1=[0.2]), grid(T0=[0.5, 1.0, 2.0, 4.0], T1=[0.05, 0.2]))

# widening of 2026-10-15 ------------------------------------------------------------------------
G_SA_W = union(G_SA,
               grid(T0=[0.75, 1.0, 1.5, 2.0, 3.0, 4.0], T1=[0.3, 0.4, 0.5]),
               grid(T0=[0.5], T1=[0.1, 0.2, 0.3]),
               grid(T0=[6.0], T1=[0.2, 0.3]))
G_SA_E4W = union(G_SA_E4,
                 grid(T0=[0.25, 0.35, 0.5], T1=[0.01, 0.02, 0.035, 0.05]),
                 grid(T0=[1.0, 2.0], T1=[0.3]))
G_PDSB_E3 = union(G_PDSB, grid(xi=[1.75, 2.0], dt=[0.35, 0.5, 0.75, 1.0]))
G_PDSB_E4 = union(G_PDSB, grid(xi=[0.35, 0.5, 0.75, 1.0, 1.25, 1.5], dt=[1.25, 1.5, 2.0]))
ONEHOT_ZETA = {3: 0.5, 4: 0.75, 5: 0.75, 6: 0.9, 8: 0.9}     # chosen in the 2026-10-14 run (always A = 2)
E4_PROPER = {("G55", 4), ("G70", 3), ("G70", 4)}            # best cut = m: proper colourings, every setting hits


def g_onehot(K: int):
    return union(G_ONEHOT, grid(A=[2], zeta=[ONEHOT_ZETA.get(K, 0.75)], xi=[0.25, 0.35, 0.5, 0.75, 1.0],
                                dt=[0.35, 0.5, 0.75, 1.0]))


def g_refim(K: int):
    """2605.06425 Supp. Tables S6.25 / S6.26 for K = 3, 4; a widened version of the K=4 grid for K >= 5."""
    if K == 3:
        return grid(BA_n=rng_(100, 200, 10), zeta=rng_(0.4, 0.7, 0.05))
    if K == 4:
        return grid(BA_n=rng_(150, 250, 10), zeta=rng_(0.5, 0.8, 0.05))
    return grid(BA_n=[200, 250, 300, 350], zeta=[0.7, 0.85, 1.0])


def g_refim_ext(K: int):
    """Our optimum sat on the upper (B/A)n edge of their grid: extension above it (G1 only)."""
    if K == 3:
        return grid(BA_n=[220.0, 240.0, 270.0, 300.0], zeta=[0.5, 0.55, 0.6, 0.65, 0.7])
    return grid(BA_n=[270.0, 290.0, 320.0, 360.0], zeta=[0.65, 0.7, 0.75, 0.8, 0.85])


def job(method, graph, K, T, grid_, seed=0, runs=100, **extra):
    j = {"method": method, "graph": graph, "K": int(K), "T": int(T), "seed": int(seed), "runs": int(runs),
         "grid": grid_}
    j.update(extra)
    return j


# plans ------------------------------------------------------------------------------------------
def plan_smoke():
    gs = ["rand:200:0.1:0"]
    out = []
    for g, K in itertools.product(gs, [3, 4]):
        out += [job("pdsb", g, K, 300, G_PDSB[:2], runs=16), job("pdsb_fused", g, K, 300, G_PDSB[:2], runs=16),
                job("pbsb", g, K, 300, G_PBSB[:1], runs=16), job("pdsb_sphere", g, K, 300, G_SPHERE[:1], runs=16),
                job("clock_dsb", g, K, 300, G_CLOCK[:1], runs=16), job("onehot_dsb", g, K, 300, G_ONEHOT[:1], runs=16),
                job("ref_im", g, K, 3000, g_refim(K)[:2], runs=16), job("sa", g, K, 300, G_SA[:1], runs=16)]
    return out


def plan_E1(refim_done=()):
    """refim_done: (graph, K) with ref_im results in results/E1 (run.py passes it); kept in the plan."""
    refim = set(REFIM_REPRO) | {(g, int(K)) for g, K in refim_done}
    out = []
    for g, K in itertools.product(G15, [3, 4]):
        if (g, K) in refim:
            out.append(job("ref_im", g, K, REFIM_T, g_refim(K)))
        if (g, K) in REFIM_REPRO:
            out.append(job("ref_im", g, K, REFIM_T, g_refim_ext(K)))
        for T in (1_000, 10_000, 100_000):
            out.append(job("pdsb_fused", g, K, T, G_PDSB))
        out.append(job("pdsb", g, K, 10_000, G_PDSB))
        out.append(job("pbsb", g, K, 10_000, G_PBSB))
        out.append(job("pdsb_sphere", g, K, 10_000, G_SPHERE))
        out.append(job("clock_dsb", g, K, 10_000, G_CLOCK))
        for T in (10_000, 100_000):
            out.append(job("onehot_dsb", g, K, T, g_onehot(K)))
            out.append(job("sa", g, K, T, G_SA_W))
    return out


def plan_E3():
    out = []
    for g, K in itertools.product(G15, [5, 6, 8]):
        for T in (10_000, 100_000):
            out.append(job("pdsb_fused", g, K, T, G_PDSB_E3))
        out.append(job("clock_dsb", g, K, 10_000, G_CLOCK))
        out.append(job("onehot_dsb", g, K, 100_000, g_onehot(K)))
        out.append(job("sa", g, K, 100_000, G_SA_W))
    return out


def plan_E1v(best):
    """best[(method, graph, K, T)] = params  (from E1).  ref_im only on REFIM_REPRO."""
    keep = {k: p for k, p in best.items() if k[0] != "ref_im" or (k[1], k[2]) in REFIM_REPRO}
    return [job(m, g, K, T, [p], seed=1, runs=1000) for (m, g, K, T), p in sorted(keep.items(), key=str)]


def plan_E3v(best):
    """best[(method, graph, K, T)] = params (from E3): fresh seed, 1000 runs, the T=1e5 methods."""
    keep = {k: p for k, p in best.items() if k[0] in ("pdsb_fused", "sa", "onehot_dsb") and k[3] == 100_000}
    return [job(m, g, K, T, [p], seed=1, runs=1000) for (m, g, K, T), p in sorted(keep.items(), key=str)]


E2_BUDGETS = {
    "pdsb_fused": [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000],
    "pdsb": [300, 1_000, 3_000, 10_000, 30_000],
    "ref_im": [10_000, 30_000, 100_000, 300_000, 1_000_000],
    "sa": [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000],
}


def plan_E2(best_any_T):
    """(old small-batch timing, 100 runs per budget)"""
    best = dict(best_any_T)
    for K in (3, 4):
        base = best.get(("ref_im", "G1", K))
        if base is None:
            continue
        for g in G15:
            best.setdefault(("ref_im", g, K), base)
    out = []
    for (m, g, K), p in sorted(best.items(), key=str):
        for T in E2_BUDGETS.get(m, []):
            out.append(job(m, g, K, T, [p], seed=2, runs=100))
    return out


E2L_BUDGETS = {
    "pdsb_fused": [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000],
    "pdsb": [1_000, 3_000, 10_000, 30_000],
    "onehot_dsb": [1_000, 3_000, 10_000, 30_000, 100_000],
    "ref_im": [10_000, 30_000, 100_000, 300_000, 1_000_000],
    "sa": [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000],
}
E2L_RUNS = 1000


def plan_E2L(best_any_T):
    """best_any_T[(method, graph, K)] = params (E1, best at the largest budget).  ref_im only on REFIM_REPRO."""
    out = []
    for (m, g, K), p in sorted(best_any_T.items(), key=str):
        if m == "ref_im" and (g, K) not in REFIM_REPRO:
            continue
        for T in E2L_BUDGETS.get(m, []):
            out.append(job(m, g, K, T, [p], seed=5, runs=E2L_RUNS))
    return out


E4_GRAPHS = ["G43", "G22", "G55", "G70", "G81"]


def plan_E4():
    """Per-graph tuning of PdSB fused and SA (static grids; the earlier top-3 settings are inside them)."""
    out = []
    for g, K in itertools.product(E4_GRAPHS, [3, 4]):
        for T in (10_000, 100_000):
            out.append(job("pdsb_fused", g, K, T, G_PDSB_E4))
            out.append(job("sa", g, K, T, G_SA_E4 if (g, K) in E4_PROPER else G_SA_E4W))
    return out


E4L_BUDGETS = {"pdsb_fused": [1_000, 10_000, 100_000, 300_000, 1_000_000],
               "sa": [1_000, 10_000, 100_000, 300_000]}
E4L_RUNS = 200


def plan_E4L(best_any_T):
    """best_any_T[(method, graph, K)] = params (E4, best at the largest budget).  SA 3e5 is skipped on the
    proper-colouring instances (SA finds the optimum in every run at 1e3 sweeps already)."""
    out = []
    for (m, g, K), p in sorted(best_any_T.items(), key=str):
        for T in E4L_BUDGETS.get(m, []):
            if m == "sa" and T > 100_000 and (g, K) in E4_PROPER:
                continue
            out.append(job(m, g, K, T, [p], seed=6, runs=E4L_RUNS))
    return out


# PdSB record hunt (own results folder) ----------------------------------------------------------
HUNT_TARGETS = ([("G22", 3), ("G22", 4), ("G43", 3), ("G43", 4), ("G55", 3), ("G81", 3), ("G81", 4)]
                + [(g, K) for K in (5, 6, 8) for g in G15])
HUNT_T1, HUNT_T2 = 1_000_000, 10_000_000
HUNT_RUNS = {"H1": 40, "H2": 1000, "H3": 200}


def g_hunt(K: int):
    xi = [0.35, 0.5, 0.75, 1.0, 1.25] if K <= 4 else [0.75, 1.0, 1.25, 1.5, 2.0]
    return grid(xi=xi, dt=[0.5, 0.75, 1.0, 1.25, 1.5])


def plan_H1():
    return [job("pdsb_fused", g, K, HUNT_T1, g_hunt(K), seed=10, runs=HUNT_RUNS["H1"], polish_best=True)
            for g, K in HUNT_TARGETS]


def plan_H2(sel):
    """sel[(graph, K)] = list of settings from H1 (best first)."""
    return [job("pdsb_fused", g, K, HUNT_T1, ps[:3], seed=11, runs=HUNT_RUNS["H2"], polish_best=True)
            for (g, K), ps in sorted(sel.items(), key=str) if ps]


def plan_H3(sel):
    return [job("pdsb_fused", g, K, HUNT_T2, ps[:1], seed=12, runs=HUNT_RUNS["H3"], polish_best=True)
            for (g, K), ps in sorted(sel.items(), key=str) if ps]


PLAN_ORDER = ["smoke", "E1", "E1v", "E2", "E2L", "E3", "E3v", "E4", "E4L"]
WAVES_ALL = [["E1", "E3", "E4"], ["E1v", "E3v"], ["E2L", "E4L"]]
HUNT_PLANS = ["H1", "H2", "H3"]
WAVES_HUNT = [["H1"], ["H2", "H3"]]
