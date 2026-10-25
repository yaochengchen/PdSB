#!/usr/bin/env python3
"""CPU baseline: Potts simulated annealing with heat-bath updates (numba, replicas in parallel).

Per replica: random colouring, local fields F[i, c] = sum_j w_ij [c_j = c] kept incrementally; one sweep
visits every vertex once in random order and redraws its colour from the Boltzmann distribution over
all K colours at temperature T (geometric schedule T0 -> T1).  Returns final and best-seen colourings.
Runs on all Grace cores via numba prange (set NUMBA_NUM_THREADS).
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _xs(s):
    s ^= (s << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    s ^= s >> np.uint64(7)
    s ^= (s << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return s


@njit(cache=True, inline="always")
def _u01(s):
    return (s >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(parallel=True, cache=True)
def _sa(indptr, indices, data, n, K, R, sweeps, T0, T1, seed, out_final, out_best, same_final, same_best):
    for r in prange(R):
        s = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(r + 1) * np.uint64(0xBF58476D1CE4E5B9)
        if s == 0:
            s = np.uint64(88172645463325252)
        for _ in range(10):
            s = _xs(s)
        col = np.empty(n, np.int64)
        for i in range(n):
            s = _xs(s)
            col[i] = int(_u01(s) * K) % K
        F = np.zeros((n, K))
        for i in range(n):
            for k in range(indptr[i], indptr[i + 1]):
                F[i, col[indices[k]]] += data[k]
        same = 0.0
        for i in range(n):
            same += F[i, col[i]]
        same *= 0.5
        best = same
        bestcol = col.copy()
        perm = np.arange(n)
        prob = np.empty(K)
        ratio = (T1 / T0) ** (1.0 / max(sweeps - 1, 1))
        T = T0
        for sw in range(sweeps):
            for a in range(n - 1, 0, -1):                      # Fisher-Yates
                s = _xs(s)
                b = int(_u01(s) * (a + 1))
                perm[a], perm[b] = perm[b], perm[a]
            beta = 1.0 / T
            for t in range(n):
                i = perm[t]
                fmin = F[i, 0]
                for c in range(1, K):
                    if F[i, c] < fmin:
                        fmin = F[i, c]
                z = 0.0
                for c in range(K):
                    prob[c] = np.exp(-beta * (F[i, c] - fmin))
                    z += prob[c]
                s = _xs(s)
                u = _u01(s) * z
                new = K - 1
                acc = 0.0
                for c in range(K):
                    acc += prob[c]
                    if u < acc:
                        new = c
                        break
                old = col[i]
                if new != old:
                    same += F[i, new] - F[i, old]
                    col[i] = new
                    for k in range(indptr[i], indptr[i + 1]):
                        j = indices[k]
                        F[j, old] -= data[k]
                        F[j, new] += data[k]
            if same < best - 1e-9:
                best = same
                bestcol[:] = col
            T *= ratio
        out_final[:, r] = col
        out_best[:, r] = bestcol
        same_final[r] = same
        same_best[r] = best


def run_sa(g, K: int, R: int, sweeps: int, T0: float, T1: float, seed: int = 0) -> dict:
    A = g.csr()
    n = g.n
    out_f = np.zeros((n, R), np.int64)
    out_b = np.zeros((n, R), np.int64)
    sf = np.zeros(R)
    sb = np.zeros(R)
    t0 = time.perf_counter()
    _sa(A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data.astype(np.float64), n, K, R, int(sweeps),
        float(T0), float(T1), int(seed), out_f, out_b, sf, sb)
    t = time.perf_counter() - t0
    return {"final_col": out_f, "best_col": out_b, "final_cut": g.total_w - sf, "best_cut": g.total_w - sb,
            "t_dyn": t}
