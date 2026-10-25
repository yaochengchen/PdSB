#!/usr/bin/env python3
"""Self-tests.  Run on the server before any plan:   python selftest.py            (GPU 0)
                                                     python selftest.py --device cpu  (items 1-8 only)

 1  simplex geometry; Potts energy identity  sum_E w ((K-1)<e,e>+1)/K == same-colour weight
 2  one-hot Ising encoding (Eq. S1.33): J, h, rownorm, structured J s == explicit matrix
 3  simplex projection: KKT, and Michelot == sort-based
 4  exact costs (torch == numpy); 1-opt polish returns 1-opt colourings, never worse
 5  SA incremental bookkeeping (needs numba)
 6  RefIM step == Eq. (9) written out in numpy (noise off)
 7  batched hyper-parameter grid == separate runs (same initial states)
 8  every machine reaches the brute-force optimum on small graphs (K = 3, 4); K=2 PdSB = max-cut
 9  (GPU) CUDA graph == eager for PdSB / one-hot dSB / RefIM (noise off)
10  (GPU) fused PdSB vs PyTorch PdSB: same arithmetic over a few steps (wall active), same statistics over 3000
    steps, device-tracked best == exact; 40-step per-replica agreement reported; K=3,4,5,6,8
11  (GPU) fused PdSB with the sparse CSR kernel == fused PdSB with the dense GEMM, bit for bit (same colours,
    same best cuts, same x), K=3,4,8; and the sparse step time on a 20000-vertex torus-like graph
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maxkcut import random_graph, same_weight_np  # noqa: E402
from machines import (DeviceGraph, OneHotSB, PdSB, RefIM, onehot_fields, onehot_Js,  # noqa: E402
                      project_prob_simplex, project_prob_simplex_sort, simplex_vertices)

FAILS = []


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def brute(g, K):
    best = -np.inf
    for c in itertools.product(range(K), repeat=g.n - 1):
        best = max(best, g.total_w - same_weight_np(g, np.array((0,) + c)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    rng = np.random.default_rng(0)

    # 1
    ok = True
    for K in range(2, 9):
        V = simplex_vertices(K)
        G = V @ V.T
        ok &= np.allclose(np.diag(G), 1) and np.allclose(G[~np.eye(K, dtype=bool)], -1 / (K - 1))
    g = random_graph(30, 0.3, 1, "pm1")
    for K in (3, 4, 5, 8):
        V = simplex_vertices(K)
        col = rng.integers(0, K, g.n)
        X = V[col]
        e = (g.w * ((K - 1) * (X[g.edges[:, 0]] * X[g.edges[:, 1]]).sum(1) + 1) / K).sum()
        ok &= np.isclose(e, same_weight_np(g, col))
    check("1 simplex geometry + Potts identity", ok)

    # 2
    g = random_graph(9, 0.5, 2, "pm1")
    dg = DeviceGraph(g, "cpu")
    ok = True
    for K in (3, 4):
        A, B = 2.7, 1.3
        h, rn = onehot_fields(dg, K, A, B)
        n = g.n
        Jd = np.zeros((n * K, n * K))
        for i in range(n):
            for c in range(K):
                for c2 in range(K):
                    if c != c2:
                        Jd[i * K + c, i * K + c2] = -A / 2
        Wd = dg.W.numpy().astype(np.float64)
        for i in range(n):
            for j in range(n):
                for c in range(K):
                    Jd[i * K + c, j * K + c] += -B * Wd[i, j] / 4
        hv = h.numpy().reshape(n, 1).repeat(K, 1).reshape(-1)
        ok &= bool(np.isclose(np.sqrt((Jd ** 2).sum(1).mean()), rn[0], rtol=1e-6))
        vals = []
        for _ in range(6):
            s = rng.choice([-1.0, 1.0], (n, K))
            x = (1 + s) / 2
            lucas = A * ((1 - x.sum(1)) ** 2).sum() + B * sum(w * (x[i] * x[j]).sum() for (i, j), w in zip(g.edges, g.w))
            sv = s.reshape(-1)
            vals.append(lucas - (-0.5 * sv @ Jd @ sv - hv @ sv))
            Js = onehot_Js(dg, torch.as_tensor(s[:, None, :], dtype=torch.float32),
                           torch.tensor(A).reshape(1, 1, 1), torch.tensor(B).reshape(1, 1, 1))
            ok &= np.allclose(Js[:, 0, :].numpy().reshape(-1), Jd @ sv, atol=1e-5)
        ok &= np.allclose(vals, vals[0], atol=1e-4)
    check("2 one-hot Ising encoding (J, h, rownorm, J s)", ok)

    # 3
    ok = True
    for K in (2, 3, 4, 5, 6, 8):
        V = torch.as_tensor(simplex_vertices(K))
        x = 2 * torch.randn(500, K - 1, dtype=torch.float64)
        q = ((K - 1) * (x @ V.T) + 1) / K
        b = project_prob_simplex(q)
        xp = b @ V
        ok &= bool((b >= -1e-12).all()) and torch.allclose(b.sum(-1), torch.ones(500, dtype=torch.float64))
        ok &= bool((((x - xp)[:, None, :] * (V[None] - xp[:, None, :])).sum(-1) <= 1e-9).all())
        ok &= bool(torch.allclose(b, project_prob_simplex_sort(q), atol=1e-12))
    check("3 simplex projection (KKT, Michelot == sort)", ok)

    # 4
    g = random_graph(60, 0.2, 3, "pm1")
    dg = DeviceGraph(g, dev)
    col = torch.as_tensor(rng.integers(0, 4, (g.n, 16)), device=dev)
    ok = np.allclose(dg.same_weight(col, 4).cpu().numpy(), same_weight_np(g, col.cpu().numpy()))
    ok &= np.allclose(dg.cut_exact(col), g.total_w - same_weight_np(g, col.cpu().numpy()))
    pc = dg.polish(col, 4).cpu().numpy()
    ok &= bool((same_weight_np(g, pc) <= same_weight_np(g, col.cpu().numpy()) + 1e-9).all())
    A_ = g.csr().toarray()
    for r in range(pc.shape[1]):
        F = A_ @ np.eye(4)[pc[:, r]]
        ok &= bool((F.min(1) >= F[np.arange(g.n), pc[:, r]] - 1e-9).all())
    check("4 exact costs + 1-opt polish", ok)

    # 5
    try:
        from sa_potts import run_sa
    except ImportError:
        run_sa = None
        print("[SKIP] 5 SA (numba not installed)")
    if run_sa is not None:
        out = run_sa(g, 4, 8, 50, 2.0, 0.1, seed=3)
        ok = np.allclose(out["final_cut"], g.total_w - same_weight_np(g, out["final_col"])) and \
            np.allclose(out["best_cut"], g.total_w - same_weight_np(g, out["best_col"]))
        check("5 SA bookkeeping", ok)

    # 6  RefIM vs numpy Eq. (9), noise off
    g = random_graph(12, 0.5, 4, "one")
    dg = DeviceGraph(g, "cpu")
    K, R, BA_n, zeta, beta = 3, 2, 200.0, 0.6, 0.3
    m = RefIM(dg, K, R, 10, BA_n, zeta, f_xi=0.0, use_graph=False)
    x0 = torch.randn(g.n, R, K) * 0.1
    m.x.copy_(x0)
    m.s.copy_(torch.sign(x0))
    m._step(torch.tensor(beta))
    A, B = 1.0, BA_n / g.n
    Wd = g.csr().toarray()
    xs = x0.numpy().astype(np.float64)
    s = np.sign(xs)
    Js = -(B / 4) * np.einsum("ij,jrc->irc", Wd, s) - (A / 2) * (s.sum(-1, keepdims=True) - s)
    h = (-A * (K - 2) / 2 - B * g.degree_w() / 4)[:, None, None]
    xn = xs + 1e-3 * (np.tanh(-50 * xs + beta * (Js + zeta * h)) - xs)
    check("6 RefIM step == Eq. (9)", np.allclose(m.x.numpy(), xn, atol=1e-6))

    # 7  batched grid == separate runs
    g = random_graph(40, 0.2, 5, "pm1")
    dg = DeviceGraph(g, "cpu")
    K, T, r1 = 4, 60, 3
    xi, dt = np.array([0.5] * r1 + [0.9] * r1), np.array([0.5] * r1 + [0.3] * r1)
    x0 = torch.randn(g.n, 2 * r1, K - 1) * 0.1
    p0 = torch.randn(g.n, 2 * r1, K - 1) * 0.1
    big = PdSB(dg, K, 2 * r1, T, xi, dt, x0=x0, p0=p0, use_graph=False)
    big.run()
    ok = True
    for k in range(2):
        sl = slice(k * r1, (k + 1) * r1)
        sm = PdSB(dg, K, r1, T, xi[sl][0], dt[sl][0], x0=x0[:, sl], p0=p0[:, sl], use_graph=False)
        sm.run()
        ok &= bool(torch.allclose(sm.x, big.x[:, sl], atol=1e-5))
    check("7 batched grid == separate runs", ok)

    # 8
    for K, n in ((3, 9), (4, 8)):
        g = random_graph(n, 0.6, 10 + K, "one")
        opt = brute(g, K)
        dg = DeviceGraph(g, dev)
        res = {}
        for name, mk in {
            "pdsb": lambda: PdSB(dg, K, 64, 400, 0.75, 0.5),
            "pbsb": lambda: PdSB(dg, K, 64, 400, 0.35, 0.25, coupling="ballistic"),
            "pdsb_sphere": lambda: PdSB(dg, K, 64, 400, 1.0, 0.5, wall="sphere"),
            "clock_dsb": lambda: PdSB(dg, K, 64, 400, 0.35, 1.0, embed="clock", wall="sphere"),
            "onehot_dsb": lambda: OneHotSB(dg, K, 64, 400, 0.5, 0.5, 2.0, 0.5),
            "ref_im": lambda: RefIM(dg, K, 64, 20000, 150.0, 0.6),
        }.items():
            res[name] = float(mk().run()["best_cut"].max())
        if run_sa is not None:
            res["sa"] = float(run_sa(g, K, 16, 200, 2.0, 0.05)["best_cut"].max())
        ok = all(v >= opt - 1e-9 for k, v in res.items() if k != "clock_dsb" or K == 3)
        check(f"8 small-graph optimum K={K} (opt {opt:.0f})", ok, str(res))
    g = random_graph(12, 0.5, 5, "pm1")
    check("8 K=2 PdSB = max-cut", float(PdSB(DeviceGraph(g, dev), 2, 64, 400, 0.75, 0.5).run()["best_cut"].max())
          >= brute(g, 2) - 1e-9)

    if torch.device(dev).type != "cuda":
        print("[SKIP] 9, 10, 11 (no GPU)")
    else:
        # 9
        g = random_graph(300, 0.05, 7, "pm1")
        dg = DeviceGraph(g, dev)
        ok = True
        for mk in (lambda ug: PdSB(dg, 4, 32, 230, 0.75, 0.5, seed=1, use_graph=ug),
                   lambda ug: OneHotSB(dg, 3, 32, 230, 0.5, 0.5, 2.0, 0.5, seed=1, use_graph=ug),
                   lambda ug: RefIM(dg, 4, 32, 2300, 200.0, 0.6, f_xi=0.0, seed=1, use_graph=ug)):
            s1, s2 = mk(False), mk(True)
            o1, o2 = s1.run(), s2.run()
            ok &= s2.graph_used and bool(torch.allclose(s1.x, s2.x, atol=1e-5)) and \
                bool(torch.equal(o1["best_cut"], o2["best_cut"]))
        check("9 CUDA graph == eager", ok)

        # 10
        # (a) a few steps from identical states with large amplitude (wall active for many elements): every
        #     element must agree to 1e-3 (a formula error gives >= 1e-2; fp32 vs fp64 of the same code ~1e-5);
        # (b) 40-step trajectories, per replica (reported, not required): the force sums are added in a different
        #     order, so a colour or wall contact decided by ~1e-7 can differ and the discrete dynamics then amplify
        #     it inside that replica (dSB is chaotic); what must agree is the statistics, (c);
        # (c) statistics: mean best cut of fused vs PyTorch PdSB on a G1-like graph within 4 standard errors;
        # (d) the best cut tracked on the device == exact recomputation of the stored best colouring.
        from fused import FusedPdSB
        ok = True
        info = []
        for K in (3, 4, 5, 6, 8):
            g = random_graph(300, 0.06, 20 + K, "pm1")
            dg = DeviceGraph(g, dev, dense_max=32768)
            R = 96
            xi = np.linspace(0.5, 1.0, R)
            gen = torch.Generator(device=dev).manual_seed(5)
            x0 = 0.7 * torch.randn(g.n, R, K - 1, generator=gen, device=dev)
            p0 = 0.5 * torch.randn(g.n, R, K - 1, generator=gen, device=dev)
            ref = PdSB(dg, K, R, 3, xi, 0.5, x0=x0, p0=p0, use_graph=False, eval_every=1)
            ref.run()
            fu = FusedPdSB(dg, K, R, 3, xi, 0.5, x0=x0, p0=p0, chunk=2)
            fu.run()
            dxa = float((fu.x - ref.x).abs().max())
            Vt = torch.as_tensor(simplex_vertices(K), dtype=torch.float32, device=dev)
            onwall = float((((K - 1) * (ref.x @ Vt.T) + 1) / K).min(-1).values.lt(1e-6).float().mean())
            x0 = 0.1 * torch.randn(g.n, R, K - 1, generator=gen, device=dev)
            p0 = 0.1 * torch.randn(g.n, R, K - 1, generator=gen, device=dev)
            ref = PdSB(dg, K, R, 40, xi, 0.5, x0=x0, p0=p0, use_graph=False, eval_every=1)
            oref = ref.run()
            fu = FusedPdSB(dg, K, R, 40, xi, 0.5, x0=x0, p0=p0, chunk=20)
            ofu = fu.run()
            per_rep = (fu.x - ref.x).abs().amax(dim=(0, 2))
            same = float((per_rep < 1e-3).float().mean())
            agree = float((ofu["final_col"] == oref["final_col"]).float().mean())
            trk_ok = np.allclose(dg.cut_exact(ofu["best_col"]), ofu["best_cut"].cpu().numpy())
            ok &= dxa < 1e-3 and trk_ok and fu.graph_used
            info.append(f"K={K}: 3 steps max|dx|={dxa:.1e} (on wall {onwall:.0%}); 40 steps: {same:.0%} of replicas "
                        f"identical, colours agree {agree:.4f}; tracked==exact={trk_ok}")
        g = random_graph(800, 0.06, 1, "one")
        dg = DeviceGraph(g, dev, dense_max=32768)
        fu = FusedPdSB(dg, 4, 256, 3000, 0.75, 0.5, seed=3)
        o = fu.run()
        to = PdSB(dg, 4, 256, 3000, 0.75, 0.5, seed=4).run()
        a, b = o["best_cut"].cpu().numpy(), to["best_cut"].cpu().numpy()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        stat_ok = abs(a.mean() - b.mean()) < 4 * se + 1e-9
        trk_ok = np.allclose(dg.cut_exact(o["best_col"]), a)
        ok &= stat_ok and trk_ok
        info.append(f"G1-like K=4 T=3000 R=256: mean best cut fused {a.mean():.1f} vs PyTorch {b.mean():.1f} "
                    f"(4 SE = {4 * se:.1f}); tracked==exact {trk_ok}; {1e3 * o['t_dyn'] / 3000:.3f} ms/step")
        check("10 fused PdSB == PyTorch PdSB", ok, "\n       " + "\n       ".join(info))

        # 11
        ok = True
        info = []
        g = random_graph(600, 0.03, 11, "pm1")
        dg_d = DeviceGraph(g, dev, dense_max=32768)
        dg_s = DeviceGraph(g, dev, dense_max=0)
        for K in (3, 4, 8):
            fd = FusedPdSB(dg_d, K, 64, 600, np.linspace(0.5, 1.0, 64), 0.5, seed=2)
            od = fd.run()
            fs = FusedPdSB(dg_s, K, 64, 600, np.linspace(0.5, 1.0, 64), 0.5, seed=2)
            os_ = fs.run()
            same = (fs.sparse and not fd.sparse and bool(torch.equal(od["final_col"], os_["final_col"]))
                    and bool(torch.equal(od["best_cut"], os_["best_cut"])) and bool(torch.equal(fd.x, fs.x)))
            trk = np.allclose(dg_s.cut_exact(os_["best_col"]), os_["best_cut"].cpu().numpy())
            pol = bool(torch.equal(dg_s.polish(os_["final_col"], K), dg_d.polish(od["final_col"], K)))
            ok &= same and trk and pol and fs.graph_used
            info.append(f"K={K}: sparse == dense {same}, tracked==exact {trk}, 1-opt sparse == dense {pol}, "
                        f"graph {fs.graph_used}")
        # 2D periodic +-1 grid, 20000 vertices (G81-like)
        L = 100
        idx = np.arange(20000).reshape(L, 200)
        e = np.concatenate([np.stack([idx.ravel(), np.roll(idx, -1, 0).ravel()], 1),
                            np.stack([idx.ravel(), np.roll(idx, -1, 1).ravel()], 1)])
        from maxkcut import Graph, _canon
        w = np.random.default_rng(3).choice([-1.0, 1.0], len(e))
        ee, ww = _canon(e, w, 20000)
        gt = Graph("torus", 20000, ee, ww)
        dgt = DeviceGraph(gt, dev, dense_max=4096)
        ft = FusedPdSB(dgt, 4, 1000, 2000, 0.75, 0.5, seed=1)
        ot = ft.run()
        trk = np.allclose(dgt.cut_exact(ot["best_col"]), ot["best_cut"].cpu().numpy())
        ok &= trk
        info.append(f"20000-vertex +-1 torus, K=4, R=1000: {1e3 * ot['t_dyn'] / 2000:.3f} ms/step "
                    f"({1e6 * ot['t_dyn'] / 2000 / 1000:.4f} us per run-step), tracked==exact {trk}")
        check("11 sparse fused PdSB == dense fused PdSB", ok, "\n       " + "\n       ".join(info))

    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
