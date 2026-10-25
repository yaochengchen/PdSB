#!/usr/bin/env python3
"""Report of the PdSB record hunt (plans H1-H3, run in their own folder).

    python hunt_report.py results_pdsb --main results      # -> results_pdsb/hunt_report.md

Question answered: does PdSB (fused kernel, best-seen colouring, optionally 1-opt polished) itself reach the best
known cut of every target, which in the main run was mostly found by SA?
  target   = max(literature value, best cut in <main>/best/ by any method, best PdSB cut in <main>)
  PdSB     = best cut of a PdSB method anywhere in <main> (E plans) and in the hunt (H1/H2/H3)
The hunt folder's best/ archive holds PdSB solutions only (verify with: python verify_best.py --best results_pdsb/best).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load_runs, md, tlabel  # noqa: E402
from maxkcut import BKS  # noqa: E402
from plans import HUNT_PLANS, HUNT_TARGETS, PLAN_ORDER  # noqa: E402

# literature best-known values (record check of 2026-10-14; MOH = Ma & Hao 2017, Garvardt et al. IJCAI 2023 /
# arXiv 2409.13380, PEAF = arXiv 2607.17158, PAMC = arXiv 2606.25203).  K = 5, 6, 8: none known to us.
LITERATURE = {("G22", 3): (17169, "PEAF/PAMC"), ("G22", 4): (18776, "MOH/PEAF"), ("G43", 3): (8573, "BKS"),
              ("G43", 4): (9377, "Garvardt/PEAF"), ("G55", 3): (12432, "Garvardt ILP/PAMC"),
              ("G81", 3): (16392, "Garvardt LS"), ("G81", 4): (16525, "Garvardt LS")}
for k, v in BKS.items():
    LITERATURE.setdefault(k, (v, "BKS (2605.06425)"))
PDSB = {"pdsb_fused", "pdsb", "pdsb_sphere"}


def cuts(c):
    """all cuts of one setting: best-seen, polished final, polished best-seen (hunt)."""
    v = [np.asarray(c["best"], float), np.asarray(c["polished"], float)]
    if "best_polished" in c:
        v.append(np.asarray(c["best_polished"], float))
    return np.maximum.reduce(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hunt", type=Path, nargs="?", default=Path("results_pdsb"))
    ap.add_argument("--main", type=Path, default=Path("results"))
    a = ap.parse_args()

    archive = {}
    for p in sorted((a.main / "best").glob("*.json")):
        m = json.loads(p.read_text())
        archive[(m["graph"], int(m["K"]))] = m
    main_pdsb = {}
    for plan in PLAN_ORDER:
        for r in load_runs(a.main / plan):
            if r["method"] in PDSB:
                for c in r["configs"]:
                    k = (r["graph"], int(r["K"]))
                    main_pdsb[k] = max(main_pdsb.get(k, -np.inf), float(cuts(c).max()))
    hunt = {p: load_runs(a.hunt / p) for p in HUNT_PLANS}
    hbest = {}
    for p in sorted((a.hunt / "best").glob("*.json")):
        m = json.loads(p.read_text())
        hbest[(m["graph"], int(m["K"]))] = m

    targets = {}
    for k in HUNT_TARGETS:
        lit = LITERATURE.get(k, (None, ""))[0]
        arc = archive.get(k, {}).get("cut")
        targets[k] = max(v for v in (lit, arc, main_pdsb.get(k), -np.inf) if v is not None)

    # 1. summary per target
    rws, n_reach = [], 0
    for k in HUNT_TARGETS:
        g, K = k
        lit, src = LITERATURE.get(k, (None, ""))
        arc = archive.get(k)
        best_h = -np.inf
        where = ""
        for p, recs in hunt.items():
            for r in recs:
                if (r["graph"], int(r["K"])) != k:
                    continue
                for c in r["configs"]:
                    v = float(cuts(c).max())
                    if v > best_h:
                        best_h, where = v, f"{p} {tlabel(r['T'])} {json.dumps(c['params'])}"
        tgt = targets[k]
        if np.isfinite(best_h):
            d = best_h - tgt
            status = ("**new best**" if d > 0 else "**reached**" if d == 0 else f"−{-d:.0f}")
            n_reach += d >= 0
        else:
            status = "not run yet"
        rws.append([f"{g} K={K}", f"{lit} ({src})" if lit else "—",
                    f"{arc['cut']:.0f} ({arc['method']})" if arc else "—",
                    f"{main_pdsb[k]:.0f}" if k in main_pdsb else "—",
                    f"{best_h:.0f}" if np.isfinite(best_h) else "—", status, where])
    out = ["# PdSB record hunt", "",
           f"hunt folder: {a.hunt}   main results: {a.main}   targets reached or beaten by PdSB: {n_reach} / "
           f"{len(HUNT_TARGETS)}", "",
           "target = max(literature, best of any method in the main archive / main runs); PdSB cut = max over best-seen, "
           "1-opt polished final and 1-opt polished best-seen colouring", "",
           md(rws, ["instance", "literature", "main archive (finder)", "PdSB in main run", "PdSB hunt best",
                    "vs target", "where (plan, steps, setting)"]), ""]

    # 2. per plan / budget: the best setting of each (target, plan, T)
    rws = []
    for k in HUNT_TARGETS:
        tgt = targets[k]
        for p, recs in hunt.items():
            for r in recs:
                if (r["graph"], int(r["K"])) != k:
                    continue
                st = []
                for c in r["configs"]:
                    b = np.asarray(c["best"], float)
                    cc = cuts(c)
                    st.append((b.mean(), c["params"], cc.max(), int((cc >= tgt).sum()), len(b)))
                mean, prm, mx, hit, n = max(st, key=lambda t: t[0])
                mx_all = max(t[2] for t in st)
                hit_all = sum(t[3] for t in st)
                n_all = sum(t[4] for t in st)
                rws.append([f"{k[0]} K={k[1]}", p, tlabel(r["T"]), len(st), f"{100 * (tgt - mean) / tgt:.3f}",
                            json.dumps(prm), f"{mx_all:.0f}", f"{hit_all} / {n_all}",
                            f"{1e3 * r['t_dyn'] / r['R_total']:.1f}"])
    out += ["## per plan and budget", "",
            "mean gap % = best setting (by mean best-seen cut) vs target; max cut and hits over all settings of the job",
            "", md(rws, ["instance", "plan", "steps", "settings", "mean gap % (best setting)", "best setting",
                         "max cut", "runs ≥ target", "ms/run"]), ""]

    # 3. archive of the hunt (PdSB-only colourings)
    rws = [[f"{m['graph']} K={m['K']}", f"{m['cut']:.0f}", m["plan"], tlabel(m["T"]), json.dumps(m["params"]),
            m["kind"]] for _, m in sorted(hbest.items(), key=lambda t: (len(t[0][0]), t[0]))]
    out += [f"## {a.hunt}/best (PdSB solutions; verify: python verify_best.py --best {a.hunt}/best)", "",
            md(rws, ["instance", "cut", "plan", "steps", "setting", "kind"]) if rws else "(empty)", ""]
    txt = "\n".join(out)
    (a.hunt / "hunt_report.md").write_text(txt)
    print(txt)
    print(f"\n-> {a.hunt / 'hunt_report.md'}")


if __name__ == "__main__":
    main()
