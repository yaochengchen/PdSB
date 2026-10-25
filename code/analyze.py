#!/usr/bin/env python3
"""Tables and figures for the PdSB experiments.

    python analyze.py results            # reads results/<plan>/runs.jsonl for every plan present
                                         # -> results/report.md, results/summary.csv, results/figures/*.pdf|png

Metric (as in arXiv:2605.06425): relative optimality gap of the best cut seen along each run,
gap = (ref - cut) / ref in %.  ref = best known value (G1-G5, K=3,4) or the best cut found by any method in
any plan ("best found").  Settings are chosen per (method, graph, K, steps) by the mean gap (their protocol);
E1v reports the chosen settings on fresh runs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maxkcut import BKS, PUBLISHED  # noqa: E402

LABEL = {"pdsb_fused": "PdSB (fused CUDA)", "pdsb": "PdSB (PyTorch)", "pbsb": "PbSB (ballistic)",
         "pdsb_sphere": "PdSB, sphere wall", "clock_dsb": "clock-dSB (planar)", "onehot_dsb": "one-hot dSB",
         "ref_im": "ref-IM, Eq. (9) (ours)", "sa": "SA (CPU)"}
ORDER = ["pdsb_fused", "pdsb", "pbsb", "ref_im", "onehot_dsb", "pdsb_sphere", "clock_dsb", "sa"]


def load_runs(path) -> list[dict]:
    p = Path(path)
    p = p / "runs.jsonl" if p.is_dir() else p
    if not p.exists():
        return []
    out, bad, dup, seen = [], 0, 0, set()
    for line in open(p):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1                     # record cut off by a kill; that job is re-run by run.py
                continue
            key = json.dumps({k: r[k] for k in ("method", "graph", "K", "T", "seed", "runs", "grid")}, sort_keys=True)
            if key in seen:                  # same job finished twice (e.g. an orphan worker of a killed run)
                dup += 1
                continue
            seen.add(key)
            out.append(r)
    if bad or dup:
        print(f"[warn] {p}: skipped {bad} truncated and {dup} duplicate line(s)", file=sys.stderr)
    return out


def reference_values(recs) -> dict:
    ref = {}
    for r in recs:
        for c in r["configs"]:
            k = (r["graph"], r["K"])
            ref[k] = max(ref.get(k, -math.inf), max(c["best"]), max(c["polished"]))
    for k, v in BKS.items():
        ref[k] = v
    return ref


def rows(recs, refs, plan="") -> pd.DataFrame:
    out = []
    for r in recs:
        ref = refs[(r["graph"], r["K"])]
        nset = len(r["configs"])
        for k, c in enumerate(r["configs"]):
            b, f, p = (np.asarray(c[x], dtype=float) for x in ("best", "final", "polished"))
            g = 100 * (ref - b) / ref
            out.append(dict(plan=plan, method=r["method"], graph=r["graph"], K=r["K"], T=r["T"], seed=r["seed"],
                            params=json.dumps(c["params"], sort_keys=True), runs=len(b), ref=ref,
                            ref_kind="BKS" if (r["graph"], r["K"]) in BKS else "best found",
                            mean_gap=g.mean(), median_gap=np.median(g), min_gap=g.min(),
                            sd_gap=g.std(ddof=1) if len(g) > 1 else 0.0,
                            mean_gap_final=(100 * (ref - f) / ref).mean(),
                            mean_gap_pol=(100 * (ref - p) / ref).mean(), max_cut=max(b.max(), p.max()),
                            p_hit=float((b >= ref - 1e-9).mean()),
                            t_job=r["t_dyn"], t_per_run_ms=1e3 * r["t_dyn"] / r["R_total"],
                            t_batch_s=r["t_dyn"] / nset, R_total=r["R_total"], device=r.get("device", ""),
                            cuda_graph=r.get("cuda_graph"), mismatch=r.get("tracked_mismatch", 0),
                            total_w=r.get("total_w", float("nan")), sparse=bool(r.get("sparse", False)),
                            gaps=g))
    return pd.DataFrame(out)


def best_per(df, keys=("method", "graph", "K", "T")):
    idx = df.groupby(list(keys))["mean_gap"].idxmin()
    return df.loc[idx]


def best_settings(e1_dir, mode: str, top: int = 3) -> dict:
    """Settings chosen on E1 for the follow-up plans."""
    recs = load_runs(e1_dir)
    df = rows(recs, reference_values(recs))
    if mode == "per_T":
        B = best_per(df)
        return {(r.method, r.graph, int(r.K), int(r.T)): json.loads(r.params) for r in B.itertuples()}
    if mode == "per_graph":
        out = {}
        for (m, g, K), s in df.groupby(["method", "graph", "K"]):
            s = s[s["T"] == s["T"].max()]
            r = s.loc[s["mean_gap"].idxmin()]
            out[(m, g, int(K))] = json.loads(r.params)
        return out
    if mode == "top":
        out = {}
        for (m, K), s in df.groupby(["method", "K"]):
            s = s[s["T"] == s["T"].max()]
            agg = s.groupby("params")["mean_gap"].mean().sort_values()
            out[(m, int(K))] = [json.loads(p) for p in agg.index[:top]]
        return out
    raise ValueError(mode)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def md(rows_, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows_]
    return "\n".join(out)


def tlabel(T):
    return f"{T:.0e}".replace("+0", "").replace("+", "") if T >= 1000 else str(T)


def method_rows(B, K, graphs):
    """rows: (label, method, T) for the methods present, in ORDER, each T separately."""
    out = []
    for m in ORDER:
        for T in sorted(B[(B["method"] == m) & (B["K"] == K)]["T"].unique()):
            out.append((f"{LABEL.get(m, m)}, T={tlabel(T)}", m, T))
    return out


def table_main(df, title, published=True, graphs=None):
    lines = []
    B = best_per(df)
    for K in sorted(B["K"].unique()):
        gs = graphs or sorted(B[B["K"] == K]["graph"].unique(), key=lambda s: (len(s), s))
        lines.append(f"### {title}: max-{K}-cut, mean gap (%) of the best cut per run, best setting per graph")
        rws = []
        for lab, m, T in method_rows(B, K, gs):
            vals = []
            for g in gs:
                s = B[(B.method == m) & (B.K == K) & (B["T"] == T) & (B.graph == g)]
                vals.append(s["mean_gap"].iloc[0] if len(s) else float("nan"))
            tpr = B[(B.method == m) & (B.K == K) & (B["T"] == T)]["t_per_run_ms"].mean()
            rws.append([lab] + [fmt(v) for v in vals] + [fmt(np.nanmean(vals)), fmt(tpr, 2)])
        if published:
            for name, tab in PUBLISHED.items():
                vals = [tab.get((g, K), (float("nan"),))[0] for g in gs]
                if not all(np.isnan(vals)):
                    rws.append([f"*{name}*, 1e5–1e6 steps"] + [fmt(v, 2) for v in vals] + [fmt(np.nanmean(vals), 2), ""])
        lines.append(md(rws, ["method"] + gs + ["mean", "ms/run"]))
        lines.append("")
    return "\n".join(lines)


def table_detail(df, title):
    lines = [f"### {title}: per-graph detail (format of 2605.06425 Tables S7.28/29)"]
    B = best_per(df)
    rws = []
    for (K, g), s in B.groupby(["K", "graph"]):
        s = s.copy()
        s["o"] = s["method"].map({m: i for i, m in enumerate(ORDER)})
        for r in s.sort_values(["o", "T"]).itertuples():
            rws.append([f"{g} K={K}", f"{LABEL.get(r.method, r.method)} T={tlabel(r.T)}", fmt(r.mean_gap),
                        fmt(r.median_gap), fmt(r.min_gap), fmt(r.sd_gap), r.runs, fmt(r.p_hit, 2), r.params])
        for name, tab in PUBLISHED.items():
            if (g, K) in tab:
                a, b_, c, d = tab[(g, K)]
                rws.append([f"{g} K={K}", f"*{name}*", fmt(a, 2), fmt(b_, 2), fmt(c, 2), fmt(d, 2), 100, "", ""])
    lines.append(md(rws, ["instance", "method", "mean", "median", "min", "SD", "runs", "P(BKS)", "setting"]))
    return "\n".join(lines) + "\n"


def table_repro(df):
    """Our ref-IM vs published: best setting inside THEIR grid, and best including our extension (G1)."""
    from plans import g_refim
    tab = PUBLISHED["ref-IM (2605.06425)"]
    d = df[df.method == "ref_im"]
    rws = []
    for (g, K), s in sorted(d.groupby(["graph", "K"]), key=lambda t: (t[0][1], len(t[0][0]), t[0][0])):
        if (g, K) not in tab:
            continue
        theirs = {json.dumps(p, sort_keys=True) for p in g_refim(int(K))}
        a = s[s.params.isin(theirs)]
        ra = a.loc[a["mean_gap"].idxmin()] if len(a) else None
        rb = s.loc[s["mean_gap"].idxmin()]
        p = tab[(g, K)]
        rws.append([f"{g} K={K}", fmt(ra.mean_gap) if ra is not None else "—", ra.params if ra is not None else "",
                    fmt(rb.mean_gap) if len(s) > len(a) else "", rb.params if len(s) > len(a) else "",
                    fmt(p[0], 2), fmt(p[2], 2)])
    if not rws:
        return ""
    return ("### Reproduction check: our ref-IM (Eq. 9, 1e6 steps) vs the published numbers\n" +
            md(rws, ["instance", "ours, their grid", "setting", "ours, + extension", "setting",
                     "published mean", "published min"]) + "\n")


def table_tests(df, title="E1v", others=("ref_im", "sa", "onehot_dsb")):
    """One-sided Mann-Whitney U (gaps of method A < gaps of B), A = PdSB fused at its largest T, B = others."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return ""
    rws = []
    for (g, K), s in df.groupby(["graph", "K"]):
        a = s[s.method == "pdsb_fused"]
        if not len(a):
            continue
        a = a[a["T"] == a["T"].max()]
        a = a.loc[a["mean_gap"].idxmin()]
        for other in others:
            b = s[s.method == other]
            if not len(b):
                continue
            b = b[b["T"] == b["T"].max()]
            b = b.loc[b["mean_gap"].idxmin()]
            u = mannwhitneyu(a.gaps, b.gaps, alternative="less")
            rws.append([f"{g} K={K}", f"T={tlabel(a['T'])}", f"{LABEL[other]} T={tlabel(b['T'])}", fmt(a.mean_gap),
                        fmt(b.mean_gap), f"{u.pvalue:.1e}"])
    if not rws:
        return ""
    return (f"### {title}: PdSB (fused) vs others on fresh runs, one-sided Mann-Whitney U on per-run gaps\n" +
            md(rws, ["instance", "PdSB", "other", "PdSB mean", "other mean", "p (PdSB < other)"]) + "\n")


TTS_TARGETS = (0.1, 0.05, 0.03, 0.0)


def tts(sg, target):
    """TTS99 (ms) of one (method, graph, K) over its budgets/settings: min over rows of tau ln(0.01)/ln(1-p),
    tau = amortised wall-clock per run, p = fraction of runs whose best gap <= target (target 0 = reach ref)."""
    best = math.inf
    for r in sg.itertuples():
        p = float((r.gaps <= target + 1e-12).mean())
        tau = r.t_per_run_ms
        if p >= 1.0:
            best = min(best, tau)
        elif p > 0:
            best = min(best, tau * math.log(0.01) / math.log(1 - p))
    return best


def tts_table(df, targets=TTS_TARGETS, methods=None, title=None):
    """rows (K, target), columns methods: TTS99 in ms, geometric mean over the graphs (inf if any graph never
    reaches the target); last column = SA / PdSB-fused ratio."""
    methods = [m for m in (methods or ORDER) if m in set(df.method)]
    rws = []
    for K in sorted(df.K.unique()):
        for target in targets:
            row, vals = [K, "reach ref" if target == 0 else f"≤ {target}%"], {}
            for m in methods:
                s = df[(df.K == K) & (df.method == m)]
                if not len(s):
                    row.append("")
                    continue
                v = [tts(sg, target) for _, sg in s.groupby("graph")]
                ok = np.all(np.isfinite(v))
                vals[m] = float(np.exp(np.mean(np.log(v)))) if ok else math.inf
                row.append(fmt(vals[m], 1) if ok else f"∞ ({int(np.sum(np.isfinite(v)))}/{len(v)})")
            a, b = vals.get("pdsb_fused"), vals.get("sa")
            row.append(fmt(b / a, 1) + "×" if a and b and np.isfinite(a) and np.isfinite(b) else "")
            rws.append(row)
    head = (f"{title}: " if title else "") + ("TTS99 in ms (time to reach the target with 99 % probability, best "
                                              "budget; geometric mean over graphs; ∞ (k/n) = reached on k of n graphs)")
    return head + "\n" + md(rws, ["K", "target"] + [LABEL.get(m, m) for m in methods] + ["SA / PdSB"]) + "\n"


def table_e2(df, figdir, title="E2", note="one setting per graph from E1, 100 runs per budget (GPU under-used)",
             prefix="e2"):
    lines = [f"### {title}: time to quality (mean gap vs wall-clock; {note})"]
    rws = []
    agg = df.groupby(["K", "method", "T"]).agg(mean_gap=("mean_gap", "mean"), t_run=("t_per_run_ms", "mean"),
                                               t_batch=("t_batch_s", "mean")).reset_index()
    for r in agg.sort_values(["K", "method", "T"]).itertuples():
        rws.append([r.K, LABEL.get(r.method, r.method), r.T, fmt(r.mean_gap), fmt(r.t_run, 3), fmt(r.t_batch, 2)])
    lines.append(md(rws, ["K", "method", "steps/sweeps", "mean gap (avg G1–G5)", "ms per run (amortised)",
                          "s per batch of 100"]))
    # TTS99 to reach a target gap, minimised over budgets: tau * ln(0.01)/ln(1-p), tau = amortised time per run
    lines.append("\n" + tts_table(df))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for K in sorted(agg.K.unique()):
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            for m in ORDER:
                s = agg[(agg.K == K) & (agg.method == m)].sort_values("t_run")
                if len(s):
                    ax.plot(s.t_run / 1e3, s.mean_gap, marker="o", ms=3, label=LABEL.get(m, m))
            for name, tab in PUBLISHED.items():
                v = [tab[(g, K)][0] for g in ("G1", "G2", "G3", "G4", "G5") if (g, K) in tab]
                if v:
                    ax.axhline(np.mean(v), ls="--", lw=0.8, color="gray")
                    ax.text(0.01, np.mean(v), name, fontsize=6, va="bottom", transform=ax.get_yaxis_transform())
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("wall-clock time per run (s, amortised over the batch)")
            ax.set_ylabel("mean gap (%)")
            ax.set_title(f"max-{K}-cut, {title}, " + ", ".join(sorted(df.graph.unique(), key=lambda x: (len(x), x))))
            ax.legend(fontsize=6)
            fig.tight_layout()
            for ext_ in ("pdf", "png"):
                fig.savefig(figdir / f"{prefix}_time_quality_K{K}.{ext_}", dpi=200)
            plt.close(fig)
    except Exception as e:
        lines.append(f"\n(figure skipped: {e})")
    return "\n".join(lines) + "\n"


def table_e3(df, figdir):
    B = best_per(df)
    lines = ["### E3: mean gap (%) vs number of colours K (reference = best cut found by any method; G1–G5 average)"]
    piv = B.groupby(["method", "T", "K"])["mean_gap"].mean().unstack("K")
    rws = [[f"{LABEL.get(m, m)}, T={tlabel(T)}"] + [fmt(v) for v in r.values] for (m, T), r in piv.iterrows()]
    lines.append(md(rws, ["method"] + [f"K={k}" for k in piv.columns]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for (m, T), r in piv.iterrows():
            ax.plot(piv.columns, r.values, marker="o", ms=3, label=f"{LABEL.get(m, m)} T={tlabel(T)}")
        ax.set_yscale("log")
        ax.set_xlabel("K")
        ax.set_ylabel("mean gap (%)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        for ext_ in ("pdf", "png"):
            fig.savefig(figdir / f"e3_gap_vs_K.{ext_}", dpi=200)
        plt.close(fig)
    except Exception as e:
        lines.append(f"\n(figure skipped: {e})")
    return "\n".join(lines) + "\n"


def table_e3v(df):
    B = best_per(df)
    rws = []
    for K in sorted(B.K.unique()):
        for m in ORDER:
            s = B[(B.K == K) & (B.method == m)]
            if len(s):
                rws.append([K, LABEL.get(m, m), tlabel(int(s["T"].iloc[0])), fmt(s.mean_gap.mean()),
                            fmt(s.t_per_run_ms.mean(), 2), fmt(s.p_hit.mean(), 3)])
    out = ("### E3v: K = 5, 6, 8 on fresh runs (1000 per graph; reference = best found by any method/plan)\n" +
           md(rws, ["K", "method", "steps", "mean gap % (G1–G5)", "ms/run", "P(reach ref)"]) + "\n")
    return out + "\n" + table_tests(df, "E3v", others=("sa", "onehot_dsb")) + "\n" + \
        tts_table(df, targets=(0.1, 0.05, 0.0), methods=["pdsb_fused", "sa", "onehot_dsb"],
                  title="E3v (single budget T=1e5)")


def table_e4(df):
    """Per (graph, K): each method at its best setting per budget (tuned per graph).  When the best cut found
    equals the total edge weight the instance is a proper-colouring instance: report the success rate."""
    B = best_per(df)
    rws = []
    for (g, K), s in sorted(B.groupby(["graph", "K"]), key=lambda t: (len(t[0][0]), t[0][0], t[0][1])):
        for r in s.sort_values(["method", "T"]).itertuples():
            proper = abs(r.ref - r.total_w) < 1e-9
            rws.append([f"{g} K={K}", LABEL.get(r.method, r.method), tlabel(r.T), f"{r.max_cut:.0f}", f"{r.ref:.0f}",
                        fmt(r.mean_gap), (f"{r.p_hit:.2f} (proper colouring)" if proper else f"{r.p_hit:.2f}"),
                        r.params])
    return ("### E4: larger Gset graphs, each method tuned per graph (reference = best found; time: see E4L)\n" +
            md(rws, ["instance", "method", "steps", "best cut", "reference", "mean gap %", "P(reach ref)",
                     "setting"]) + "\n")


def table_e4l(df):
    rws = []
    for (g, K), s in sorted(df.groupby(["graph", "K"]), key=lambda t: (len(t[0][0]), t[0][0], t[0][1])):
        for r in s.sort_values(["method", "T"]).itertuples():
            rws.append([f"{g} K={K}", LABEL.get(r.method, r.method), tlabel(r.T), "sparse" if r.sparse else "dense"
                        if r.device != "cpu" else "CPU", fmt(r.mean_gap), fmt(r.p_hit, 3), fmt(r.t_per_run_ms, 2)])
    out = ("### E4L: time on the larger graphs (best E4 setting at 1e5, 200 runs per budget, SA timed alone)\n" +
           md(rws, ["instance", "method", "steps", "path", "mean gap %", "P(reach ref)", "ms/run"]) + "\n")
    rws = []
    for (g, K), s in sorted(df.groupby(["graph", "K"]), key=lambda t: (len(t[0][0]), t[0][0], t[0][1])):
        row = [f"{g} K={K}"]
        vals = {}
        for m in ("pdsb_fused", "sa"):
            for tgt in (0.1, 0.0):
                v = tts(s[s.method == m], tgt) if len(s[s.method == m]) else math.inf
                vals[(m, tgt)] = v
                row.append(fmt(v, 1) if np.isfinite(v) else "∞")
        for tgt in (0.1, 0.0):
            a, b = vals[("pdsb_fused", tgt)], vals[("sa", tgt)]
            row.append(fmt(b / a, 1) + "×" if np.isfinite(a) and np.isfinite(b) else ("PdSB only" if np.isfinite(a)
                                                                                    else ""))
        rws.append(row)
    return out + "\nTTS99 (ms) per instance\n" + md(rws, ["instance", "PdSB ≤0.1%", "PdSB ref", "SA ≤0.1%", "SA ref",
                                                          "SA/PdSB ≤0.1%", "SA/PdSB ref"]) + "\n"


def table_e4l_equal_time(df):
    """Equal wall-clock comparison on the E4 graphs: for every PdSB budget, SA's mean gap at the same amortised
    time per run, interpolated log-log between SA budgets (blank outside SA's measured time range)."""
    rws = []
    for (g, K), s in sorted(df.groupby(["graph", "K"]), key=lambda t: (len(t[0][0]), t[0][0], t[0][1])):
        P = s[s.method == "pdsb_fused"].sort_values("t_per_run_ms")
        S = s[s.method == "sa"].sort_values("t_per_run_ms")
        if not len(P) or len(S) < 2 or float(S.mean_gap.max()) <= 0:
            continue
        lt = np.log(S.t_per_run_ms.to_numpy())
        lg = np.log(np.maximum(S.mean_gap.to_numpy(), 1e-4))
        for r in P.itertuples():
            x = math.log(r.t_per_run_ms)
            if x < lt[0] or x > lt[-1]:
                sa_g, ratio = "", ""
            else:
                v = float(np.exp(np.interp(x, lt, lg)))
                sa_g, ratio = fmt(v), (fmt(v / r.mean_gap, 1) + "×" if r.mean_gap > 0 else "PdSB at ref")
            rws.append([f"{g} K={K}", tlabel(r.T), fmt(r.t_per_run_ms, 2), fmt(r.mean_gap), sa_g, ratio])
    if not rws:
        return ""
    return ("\nEqual wall-clock (mean gap %; SA interpolated log-log at the PdSB time per run; ratio = SA gap / "
            "PdSB gap)\n" + md(rws, ["instance", "PdSB steps", "ms/run", "PdSB gap", "SA gap at same time",
                                     "SA/PdSB"]) + "\n")


def table_edges(df, title):
    """Appendix: how often the chosen setting sits on the edge of the grid that was tried for that
    (method, graph, K, T) -- a check that every method's grid was wide enough."""
    rws = []
    for (m, K), s in df.groupby(["method", "K"]):
        n, hits = 0, {}
        for _, sg in s.groupby(["graph", "T"]):
            ps = [json.loads(x) for x in sg.params.unique()]
            best = json.loads(sg.loc[sg["mean_gap"].idxmin()].params)
            n += 1
            for k in best:
                vals = sorted({p[k] for p in ps})
                if len(vals) > 1 and best[k] in (vals[0], vals[-1]):
                    lab = f"{k}={'lo' if best[k] == vals[0] else 'hi'}"
                    hits[lab] = hits.get(lab, 0) + 1
        rws.append([LABEL.get(m, m), K, n, ", ".join(f"{k}: {v}" for k, v in sorted(hits.items())) or "none"])
    return (f"### {title}: chosen setting on the edge of its grid (count of (graph, T) cells)\n" +
            md(rws, ["method", "K", "cells", "edge hits"]) + "\n")


def planned_filter(df, plan, root):
    """Keep only rows whose (method, graph, K, T, seed, setting) belongs to the CURRENT plan: after the grids were
    widened a selection plan (E1v, E2L, ...) can hold rows of settings that are no longer the selected ones."""
    try:
        import run
        jobs = run.build_plan(plan, root)
    except (SystemExit, Exception) as e:
        print(f"[warn] {plan}: cannot rebuild the plan ({e}); using all rows", file=sys.stderr)
        return df
    keep = {(j["method"], j["graph"], j["K"], j["T"], j["seed"], json.dumps(p, sort_keys=True))
            for j in jobs for p in j["grid"]}
    k = list(zip(df.method, df.graph, df.K, df["T"], df.seed, df.params))
    mask = [x in keep for x in k]
    dropped = len(df) - int(np.sum(mask))
    if dropped:
        print(f"[info] {plan}: {dropped} rows of earlier selections not used", file=sys.stderr)
    return df[mask]


def main():
    from plans import PLAN_ORDER
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    plans_ = [p for p in PLAN_ORDER if (root / p / "runs.jsonl").exists()]
    allrecs = {p: load_runs(root / p) for p in plans_}
    refs = reference_values([r for v in allrecs.values() for r in v])
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    parts, frames, appendix = ["# PdSB experiments — report", ""], [], ["## Appendix: grid-edge check", ""]
    for p in plans_:
        df = rows(allrecs[p], refs, p)
        if df.empty:
            continue
        if p in ("E1v", "E2", "E2L", "E3v", "E4L"):
            df = planned_filter(df, p, root)
        frames.append(df.drop(columns=["gaps"]))
        bad = df[df["mismatch"].fillna(0) > 0]
        if len(bad):
            parts.append(f"**WARNING {p}: tracked best != exact recomputation in {len(bad)} settings**\n")
        nograph = df[(df["cuda_graph"] == False) & (df["device"] != "cpu") & (df["T"] >= 1000)]  # noqa: E712
        if len(nograph):
            parts.append(f"note {p}: {nograph[['method']].drop_duplicates().method.tolist()} ran without CUDA graph\n")
        parts.append(f"## {p}\n")
        if p in ("E1", "E1v", "smoke"):
            parts.append(table_main(df, p, published=p != "smoke"))
            if p == "E1":
                parts.append(table_repro(df))
                parts.append(table_detail(df, p))
                appendix.append(table_edges(df, "E1"))
            if p == "E1v":
                parts.append(table_tests(df))
                parts.append(tts_table(df, title="E1v (1000 runs per setting, budgets = the E1 step counts)"))
                parts.append(table_detail(df, p))
        elif p == "E2":
            parts.append(table_e2(df, figdir))
        elif p == "E2L":
            parts.append(table_e2(df, figdir, title="E2L", prefix="e2L",
                                  note="one setting per graph from E1, 1000 runs per budget, SA timed alone"))
        elif p == "E3":
            parts.append(table_e3(df, figdir))
            appendix.append(table_edges(df, "E3"))
        elif p == "E3v":
            parts.append(table_e3v(df))
        elif p == "E4":
            parts.append(table_e4(df))
            appendix.append(table_edges(df, "E4"))
        elif p == "E4L":
            parts.append(table_e4l(df))
            parts.append(table_e4l_equal_time(df))
    txt = "\n".join(parts + appendix)
    (root / "report.md").write_text(txt)
    if frames:
        pd.concat(frames).to_csv(root / "summary.csv", index=False)
    print(txt)
    print(f"\n-> {root / 'report.md'}, {root / 'summary.csv'}, {figdir}/")


if __name__ == "__main__":
    main()
