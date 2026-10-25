#!/usr/bin/env python3
"""Figures and derived numbers for the PdSB paper, recomputed from the raw runs.jsonl files."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
import analyze as A  # noqa

OUT = HERE / "figs"; OUT.mkdir(exist_ok=True)
NUM = {}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "lines.linewidth": 1.4, "lines.markersize": 4.5, "savefig.bbox": "tight", "pdf.fonttype": 42})
COL = {"pdsb_fused": "#2a78d6", "sa": "#eb6834", "onehot_dsb": "#1baf7a", "ref_im": "#eda100",
       "pdsb": "#e87ba4", "pdsb_sphere": "#6250d6", "clock_dsb": "#e34948"}
MK = {"pdsb_fused": "o", "sa": "s", "onehot_dsb": "^", "ref_im": "D", "pdsb": "v", "pdsb_sphere": "P",
      "clock_dsb": "X"}
NAME = {"pdsb_fused": "PdSB (fused CUDA)", "sa": "Potts SA (136 CPU threads)", "onehot_dsb": "one-hot dSB",
        "ref_im": "reference one-hot IM, our run (G1 only)", "pdsb": "PdSB (PyTorch)"}


def load(plan, folder="results"):
    recs = A.load_runs(CODE / folder / plan)
    return recs


# references: same as analyze.py (BKS for G1-G5 K3/4, otherwise best found in the main run)
ALL = []
for p in ["E1", "E1v", "E2", "E2L", "E3", "E3v", "E4", "E4L"]:
    ALL += load(p)
REFS = A.reference_values(ALL)


def df_of(plan):
    df = A.rows(load(plan), REFS, plan)
    if plan in ("E1v", "E2", "E2L", "E3v", "E4L"):
        # Rebuild the same selections as run.build_plan using the pure plan
        # functions. Importing the Linux runner would require fcntl on Windows.
        import plans
        source = {"E1v": "E1", "E2": "E1", "E2L": "E1", "E3v": "E3", "E4L": "E4"}[plan]
        mode = "per_T" if plan in ("E1v", "E3v") else "per_graph"
        selected = A.best_settings(CODE / "results" / source, mode)
        jobs = getattr(plans, "plan_" + plan)(selected)
        keep = {(j["method"], j["graph"], j["K"], j["T"], j["seed"], json.dumps(p, sort_keys=True))
                for j in jobs for p in j["grid"]}
        keys = zip(df.method, df.graph, df.K, df["T"], df.seed, df.params)
        mask = [key in keep for key in keys]
        n0 = len(df)
        df = df[mask]
        if df.empty:
            raise ValueError(f"{plan}: no archived rows match the selected plan")
        print(f"[filter] {plan}: {n0} -> {len(df)} rows", file=sys.stderr)
    return df


# ------------------------------------------------------------------ Fig. 1: geometry + toy dynamics
def simplex_vertices(K):
    E = np.eye(K) - 1.0 / K
    _, _, Vt = np.linalg.svd(E)
    V = E @ Vt[: K - 1].T
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def proj_prob(q):
    K = q.shape[-1]
    act = np.ones_like(q)
    for _ in range(K):
        th = ((q * act).sum(-1, keepdims=True) - 1) / act.sum(-1, keepdims=True)
        act = (q > th).astype(float)
    return np.clip(q - th, 0, None)


def toy_run(n=60, p=0.25, K=3, T=1500, dt=0.5, xi=0.75, seed=3):
    rng = np.random.default_rng(seed)
    W = np.triu((rng.random((n, n)) < p).astype(float), 1); W = W + W.T
    V = simplex_vertices(K); D = K - 1
    rown = math.sqrt((W ** 2).sum(1).mean()); c0 = xi / rown
    def ball():
        v = rng.normal(size=(n, D)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * 0.1 * rng.random((n, 1)) ** (1 / D)
    x, pm = ball(), ball()
    traj, cuts = [x.copy()], []
    for t in range(T):
        a = t / T
        c = np.argmax(x @ V.T, 1)
        oh = np.eye(K)[c]
        cuts.append(W.sum() / 2 - 0.5 * (oh * (W @ oh)).sum())
        F = -(W @ oh) @ V
        pm += dt * (-(1 - a) * x + c0 * F)
        x += dt * pm
        b = ((K - 1) * (x @ V.T) + 1) / K
        out = (b < -1e-4).any(1)
        if out.any():
            xp = proj_prob(b[out]) @ V
            d = x[out] - xp; nn = d / np.linalg.norm(d, axis=1, keepdims=True)
            pn = np.clip((pm[out] * nn).sum(1, keepdims=True), 0, None)
            pm[out] -= pn * nn; x[out] = xp
        traj.append(x.copy())
    return np.array(traj), V, np.array(cuts), W


def fig_geometry():
    traj, V, cuts, W = toy_run(n=80, p=0.15, T=3000, dt=0.25, xi=0.75, seed=5)
    T = len(traj) - 1
    fin = np.argmax(traj[-1] @ V.T, 1)
    vc = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [1.0, 1.0, 1.45]})
    # Same three discrete states, different absorbing normal cones.
    ax0 = axs[0]
    a = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
    q = np.column_stack((np.cos(a), np.sin(a)))
    theta = np.linspace(0, 2 * np.pi, 240)
    ax0.plot(np.cos(theta), np.sin(theta), color=COL["pdsb_fused"], ls="--", lw=0.9)
    ax0.fill(q[:, 0], q[:, 1], color="#f3f2ee")
    ax0.plot(*np.vstack((q, q[0])).T, color="#52514e", lw=0.9)
    cone = np.array([q[0], q[0] + 0.65 * np.array([-np.sqrt(3)/2, .5]),
                     q[0] + 0.65 * np.array([np.sqrt(3)/2, .5])])
    ax0.fill(*cone.T, color="#eb6834", alpha=0.25, zorder=1)
    ax0.annotate("", xy=(0, 1.62), xytext=q[0], arrowprops=dict(arrowstyle="->", color=COL["pdsb_fused"], lw=1.0))
    ax0.annotate("", xy=(.36, 1.49), xytext=q[0], arrowprops=dict(arrowstyle="->", color="#1b8c60", lw=1.0))
    ax0.text(.39, 1.48, "$F$", color="#1b8c60", fontsize=8)
    ax0.text(-.77, 1.44, "simplex\ncone", fontsize=6.5, ha="right", color="#b24b20")
    ax0.text(.04, 1.74, "disk: normal ray", fontsize=6.5, ha="center", color=COL["pdsb_fused"])
    for j, e in enumerate(q):
        ax0.plot(*e, "o", color=vc[j], ms=4.5, mec="white", mew=.6)
        ax0.text(*(1.2 * e - np.array([0, .04])), f"$e_{j+1}$", fontsize=8, ha="center")
    ax0.set_aspect("equal"); ax0.axis("off")
    ax0.set_xlim(-1.42, 1.28); ax0.set_ylim(-1.1, 2.02)
    ax = axs[1]
    tri = np.vstack([q, q[:1]])
    ax.fill(q[:, 0], q[:, 1], color="#f3f2ee", zorder=0)
    ax.plot(tri[:, 0], tri[:, 1], color="#52514e", lw=.9)
    for j, e in enumerate(q):
        ax.plot(*e, marker="o", ms=5, color=vc[j], mec="white", mew=.6, zorder=5)
        ax.text(*(e * 1.17), f"$e_{j+1}$", fontsize=8, ha="center", va="center")
    xp = .55 * q[0] + .45 * q[2]
    nrm = np.array([np.sqrt(3) / 2, .5])
    xo = xp + .34 * nrm
    tangent = (q[2] - q[0]) / np.linalg.norm(q[2] - q[0])
    previous = xp - .36 * nrm - .30 * tangent
    ax.annotate("", xy=xo, xytext=previous,
                arrowprops=dict(arrowstyle="->", color="#96948e", lw=.9, ls="--"))
    ax.annotate("", xy=xp, xytext=xo,
                arrowprops=dict(arrowstyle="->", color="#e34948", lw=1.1))
    end = xp + .30 * tangent
    ax.annotate("", xy=end, xytext=xp,
                arrowprops=dict(arrowstyle="->", color=COL["pdsb_fused"], lw=1.2))
    ax.plot(*xo, "o", ms=4, mfc="white", mec="#30343a")
    ax.plot(*xp, "o", ms=4, color="#30343a")
    ax.text(*(xo + np.array([.08, .08])), "$x'$", fontsize=9)
    ax.text(*(xp + np.array([-.34, .12])), r"$x^\Pi$", fontsize=9)
    ax.text(*(end + np.array([-.08, -.23])), r"$p_\parallel$", fontsize=9, color=COL["pdsb_fused"])
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.12, 1.2); ax.set_ylim(-.85, 1.42)
    ax2 = axs[2]
    bary = ((V.shape[0] - 1) * np.einsum("tnd,kd->tnk", traj, V) + 1) / V.shape[0]
    tt = np.arange(T + 1) / T
    atv = (bary.max(-1) > 1 - 1e-6).mean(1)
    ax2.plot(tt, atv, color="#2a78d6", label="positions near a vertex")
    oh_final = np.eye(len(V))[fin]
    final_cut = W.sum() / 2 - .5 * (oh_final * (W @ oh_final)).sum()
    cut = np.append(cuts, final_cut)
    run = np.maximum.accumulate(cut)
    ax2.plot(tt, cut / cut.max(), color="#eb6834", lw=1.0, label="current cut / best cut")
    ax2.set_xlabel("normalized time $t/T$")
    ax2.set_ylabel("fraction")
    ax2.set_ylim(0, 1.05); ax2.set_xlim(0, 1)
    ax2.legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, .91))
    for ax_, title in zip(axs, ["(a) same states, different walls", "(b) inelastic simplex collision",
                               "(c) vertex selection during the pump"]):
        box = ax_.get_position()
        fig.text((box.x0 + box.x1) / 2, .96, title, ha="center", va="top", fontsize=8)
    fig.savefig(OUT / "fig_geometry.pdf")
    plt.close(fig)


def fig_onehot():
    """Quality against the published reference, without mixing in our reproduction."""
    d = A.best_per(df_of("E1v"))
    graphs = ["G1", "G2", "G3", "G4", "G5"]
    records = []
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=True)
    for ax, k in zip(axs, (3, 4)):
        p = d[(d.method == "pdsb_fused") & (d.K == k) & (d["T"] == 100000)].set_index("graph").loc[graphs]
        y = p.mean_gap.to_numpy()
        published = np.array([A.PUBLISHED["ref-IM (2605.06425)"][(g, k)][0] for g in graphs])
        assert (y < published).all(), "Headline claim must hold on every plotted instance"
        assert (p.runs == 1000).all()
        x = np.arange(len(graphs))
        ax.bar(x - .18, published, width=.34, color="#eb6834", alpha=.88,
               label="Published one-hot reference IM")
        ax.bar(x + .18, y, width=.34, color=COL["pdsb_fused"], label="PdSB, $10^5$ steps")
        for j, (baseline, value) in enumerate(zip(published, y)):
            ax.text(j, baseline + .006, f"{baseline/value:.1f}" + r"$\times$", ha="center",
                    fontsize=7, color="#30343a")
            records.append({"graph": graphs[j], "K": k, "published_refim_gap": float(baseline),
                            "pdsb_gap": float(value), "ratio": float(baseline / value),
                            "pdsb_runs": int(p.iloc[j].runs)})
        ratio = published.mean() / y.mean()
        ax.set_title(f"({'ab'[k-3]}) Max-{k}-Cut: {ratio:.1f}" + r"$\times$ smaller mean gap",
                     fontsize=8, pad=8)
        ax.set_xticks(x, graphs); ax.set_ylim(0, .235)
        ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("mean reference gap (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, -.015), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1), w_pad=1.2)
    fig.savefig(OUT / "fig_onehot_comparison.pdf")
    plt.close(fig)
    NUM["onehot_comparison"] = records


# ------------------------------------------------------------------ Fig. 2: time to quality, G1-G5
def fig_e2l():
    df = df_of("E2L")
    agg = df.groupby(["K", "method", "T"]).agg(gap=("mean_gap", "mean"), t=("t_per_run_ms", "mean")).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, K in zip(axs, (3, 4)):
        for m in ["pdsb_fused", "pdsb", "sa", "onehot_dsb", "ref_im"]:
            s = agg[(agg.K == K) & (agg.method == m)].sort_values("t")
            if len(s):
                ax.plot(s.t, s.gap, marker=MK[m], color=COL[m], label=NAME[m],
                        mec="white", mew=0.5, ls="-" if m != "pdsb" else "--")
        pub = np.mean([A.PUBLISHED["ref-IM (2605.06425)"][(g, K)][0] for g in A.A_G if (g, K) in A.PUBLISHED["ref-IM (2605.06425)"]]) if hasattr(A, "A_G") else None
        v = [A.PUBLISHED["ref-IM (2605.06425)"][(g, K)][0] for g in ("G1", "G2", "G3", "G4", "G5")]
        ax.axhline(np.mean(v), color="#8a8984", ls=":", lw=1.0)
        ax.text(0.98, np.mean(v) * 1.08, "published reference IM", fontsize=6.5, color="#52514e", ha="right", transform=ax.get_yaxis_transform())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("amortized time per run (ms)")
        ax.set_title(f"({'ab'[K-3]}) Max-{K}-Cut, mean of G1–G5")
    axs[0].set_ylabel("mean reference gap (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_time_quality.pdf")
    plt.close(fig)
    NUM["e2l"] = agg.to_dict("records")


# ------------------------------------------------------------------ Fig. 3: gap vs K and TTS ratio vs K
def fig_vs_K():
    d1 = df_of("E1v"); d3 = df_of("E3v")
    d = pd.concat([d1[d1["T"] == 100000], d3], ignore_index=True)
    d = d[d.method.isin(["pdsb_fused", "sa", "onehot_dsb"])]
    B = A.best_per(d)
    g = B.groupby(["method", "K"]).agg(gap=("mean_gap", "mean"), t=("t_per_run_ms", "mean")).reset_index()
    tts = {}
    for (m, K), s in B.groupby(["method", "K"]):
        v = [A.tts(sg, 0.1) for _, sg in s.groupby("graph")]
        tts[(m, K)] = float(np.exp(np.mean(np.log(v)))) if np.all(np.isfinite(v)) else math.inf
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    Ks = [3, 4, 5, 6, 8]
    for m in ["pdsb_fused", "sa", "onehot_dsb"]:
        s = g[g.method == m].set_index("K").reindex(Ks)
        axs[0].plot(Ks, s.gap, marker=MK[m], color=COL[m], label=NAME[m].replace(" (G1 only)", ""), mec="white", mew=0.5)
    axs[0].set_yscale("log"); axs[0].set_xticks(Ks)
    from matplotlib.ticker import FuncFormatter
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[-1] in "36" else ""))
    axs[0].set_xlabel("number of parts $K$"); axs[0].set_ylabel("mean gap (%) at $10^5$ steps / sweeps")
    axs[0].set_title("(a) quality at equal step count (1000 fresh runs per graph)")
    axs[0].legend(frameon=False, loc="lower right", fontsize=6.5)
    r = [tts[("sa", K)] / tts[("pdsb_fused", K)] for K in Ks]
    tp = [g[(g.method == "pdsb_fused") & (g.K == K)].t.iloc[0] for K in Ks]
    ts = [g[(g.method == "sa") & (g.K == K)].t.iloc[0] for K in Ks]
    axs[1].bar(range(len(Ks)), r, color=COL["pdsb_fused"], width=0.55)
    for i, v in enumerate(r):
        axs[1].text(i, v + 0.12, f"{v:.1f}×", ha="center", fontsize=7)
    axs[1].set_xticks(range(len(Ks))); axs[1].set_xticklabels([f"K={k}" for k in Ks])
    axs[1].set_ylabel("TTS$_{99}$(≤0.1%): SA / PdSB")
    axs[1].set_ylim(0, max(r) * 1.25); axs[1].grid(axis="x", visible=False)
    axs[1].set_title("(b) TTS$_{99}$ ratio at a budget of $10^5$")
    fig.tight_layout()
    fig.savefig(OUT / "fig_vs_K.pdf")
    plt.close(fig)
    NUM["vsK"] = {"gap": g.to_dict("records"), "tts": {f"{m}_{K}": v for (m, K), v in tts.items()},
                  "ratio": dict(zip(Ks, r)), "t_pdsb": dict(zip(Ks, tp)), "t_sa": dict(zip(Ks, ts))}


# ------------------------------------------------------------------ Fig. 4: large graphs, equal wall-clock
def fig_e4l():
    df = df_of("E4L")
    hunt = []
    for p in ("H2", "H3"):
        hunt += A.load_runs(CODE / "results_pdsb" / p)
    H = A.rows(hunt, REFS, "H")
    inst = [("G22", 3), ("G22", 4), ("G43", 3), ("G43", 4), ("G55", 3), ("G81", 3), ("G81", 4)]
    fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.9))
    axs = axs.ravel()
    rec = {}
    for ax, (gname, K) in zip(axs, inst):
        for m in ("pdsb_fused", "sa"):
            s = df[(df.graph == gname) & (df.K == K) & (df.method == m)].sort_values("t_per_run_ms")
            ax.plot(s.t_per_run_ms, s.mean_gap, marker=MK[m], color=COL[m], mec="white", mew=0.5,
                    label=("PdSB, tuned at $10^5$" if m == "pdsb_fused" else "Potts SA"))
            rec[f"{gname}_K{K}_{m}"] = list(zip(s["T"].tolist(), s.t_per_run_ms.round(2).tolist(), s.mean_gap.round(3).tolist()))
        h = H[(H.graph == gname) & (H.K == K)]
        pts = []
        for T in sorted(h["T"].unique()):
            b = h[h["T"] == T]
            b = b.loc[b.mean_gap.idxmin()]
            pts.append((b.t_per_run_ms, b.mean_gap))
        if pts:
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], marker="o", mfc="white", color=COL["pdsb_fused"], ls=":",
                    label="PdSB, dedicated search (retuned at $10^6$)")
            rec[f"{gname}_K{K}_hunt"] = pts.round(3).tolist()
        ax.set_xscale("log"); ax.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(f"{v:g}")[0] in "25" else ""))
        ax.set_title(f"{gname}, $K={K}$", pad=2)
        ax.tick_params(labelsize=6)
    axs[-1].axis("off")
    h_, l_ = axs[0].get_legend_handles_labels()
    axs[-1].legend(h_, l_, loc="center", frameon=False, fontsize=7)
    for ax in axs[4:7]:
        ax.set_xlabel("amortized time per run (ms)")
    for ax in (axs[0], axs[4]):
        ax.set_ylabel("mean gap (%)")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(OUT / "fig_large.pdf")
    plt.close(fig)
    NUM["e4l"] = rec


def fig_tuning():
    """Descriptive sensitivity and search-history audit using tuning data only."""
    import plans
    e1 = A.rows(load("E1"), REFS, "E1")
    pdsb = e1[e1.method == "pdsb_fused"]
    fixed_key = json.dumps({"dt": .5, "xi": .75}, sort_keys=True)
    records = []
    for (g, k, t), cell in pdsb.groupby(["graph", "K", "T"]):
        selected = cell.loc[cell.mean_gap.idxmin()]
        fixed = cell[cell.params == fixed_key].iloc[0]
        initial = cell[cell.params.map(lambda p: json.loads(p)["dt"] in (.35, .5, .75)
                                      and json.loads(p)["xi"] in (.5, .75, 1.))]
        initial_best = initial.loc[initial.mean_gap.idxmin()]
        assert abs(initial_best.mean_gap - selected.mean_gap) < 1e-12
        records.append({"graph": g, "K": int(k), "steps": int(t), "fixed_gap": float(fixed.mean_gap),
                        "selected_gap": float(selected.mean_gap),
                        "extra_gap_pp": float(fixed.mean_gap - selected.mean_gap),
                        "mean_cut_loss": float((fixed.mean_gap - selected.mean_gap) * selected.ref / 100),
                        "selected_params": selected.params, "runs_per_setting": int(fixed.runs)})
    audit = pd.DataFrame(records)
    assert len(audit) == 30 and (audit.runs_per_setting == 100).all()
    exact = int((audit.extra_gap_pp.abs() < 1e-12).sum())
    assert exact == 23
    assert audit.mean_cut_loss.max() < 1
    audit.to_csv(HERE / "parameter_robustness.csv", index=False)

    e4 = A.rows(load("E4"), REFS, "E4")
    e4 = e4[(e4.method == "sa") & (e4["T"] == 100000)]
    initial = plans.grid(T0=[2., 4.], T1=[.2])
    stages = [initial, plans.G_SA_E4, plans.G_SA_E4W]
    stage_labels = ["Initial: 2 settings", "Expanded: 8 settings", "Final: 21 settings"]
    instances = [("G55", 3), ("G81", 3), ("G81", 4)]
    history = []
    for g, k in instances:
        cell = e4[(e4.graph == g) & (e4.K == k)]
        for j, grid in enumerate(stages):
            keep = {json.dumps(p, sort_keys=True) for p in grid}
            trial = cell[cell.params.isin(keep)]
            assert len(trial) == len(keep)
            best = trial.loc[trial.mean_gap.idxmin()]
            history.append({"graph": g, "K": k, "stage": j, "settings": len(keep),
                            "mean_gap": float(best.mean_gap), "params": best.params,
                            "runs_per_setting": int(best.runs)})
    hist = pd.DataFrame(history)
    hist.to_csv(HERE / "sa_tuning_history.csv", index=False)
    initial_pdsb = A.rows(load("E4", folder="results_initial"), REFS, "initial_E4")
    initial_pdsb = initial_pdsb[(initial_pdsb.method == "pdsb_fused") & (initial_pdsb["T"] == 100000)]
    initial_pdsb = A.best_per(initial_pdsb)
    transferred = [float(initial_pdsb[(initial_pdsb.graph == g) & (initial_pdsb.K == k)].mean_gap.iloc[0])
                   for g, k in instances]

    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={"width_ratios": [1, 1.1]})
    for k, color in [(3, COL["pdsb_fused"]), (4, "#6250d6")]:
        agg = audit[audit.K == k].groupby("steps")[["fixed_gap", "selected_gap"]].mean()
        axs[0].plot(agg.index, agg.selected_gap, "-", color=color, marker="o", ms=3,
                    label=f"$K={k}$, cellwise tuned")
        axs[0].plot(agg.index, agg.fixed_gap, "--", color=color, marker="s", mfc="white", ms=4,
                    label=f"$K={k}$, fixed parameters")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("PdSB steps"); axs[0].set_ylabel("mean reference gap (%)")
    axs[0].set_title("(a) PdSB: one setting for G1--G5", fontsize=8, pad=9)
    axs[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    axs[0].text(.03, .05, "Fixed: $\\Delta t=0.5$, $\\xi=0.75$\n"
                f"{exact}/30 grid minima; worst mean cut loss < 1",
                transform=axs[0].transAxes, fontsize=6.5, va="bottom",
                bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=2))
    x = np.arange(len(instances)); colors = ["#fed7aa", "#ef9b69", "#c84e21"]
    for j, label in enumerate(stage_labels):
        values = [hist[(hist.graph == g) & (hist.K == k) & (hist.stage == j)].mean_gap.iloc[0]
                  for g, k in instances]
        axs[1].bar(x + (j - 1) * .23, values, width=.22, color=colors[j], label=label)
    axs[1].set_yscale("log"); axs[1].set_ylim(.035, 2)
    axs[1].set_xticks(x, [f"{g}, $K={k}$" for g, k in instances])
    axs[1].plot(x, transferred, ls="none", marker="_", ms=19, mew=1.8,
                color=COL["pdsb_fused"], label="PdSB, initial transferred grid", zorder=5)
    axs[1].set_ylabel("mean reference gap (%)")
    axs[1].set_title("(b) SA: searching colder schedules", fontsize=8, pad=9)
    axs[1].grid(axis="x", visible=False); axs[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "fig_parameter_robustness.pdf")
    plt.close(fig)
    NUM["tuning"] = {
        "pdsb_fixed": records, "sa_history": history, "pdsb_exact_minima": exact,
        "pdsb_mean_extra_gap_pp": float(audit.extra_gap_pp.mean()),
        "pdsb_max_extra_gap_pp": float(audit.extra_gap_pp.max()),
        "pdsb_max_mean_cut_loss": float(audit.mean_cut_loss.max()),
        "initial_pdsb_e4": [dict(graph=g, K=k, mean_gap=value)
                            for (g, k), value in zip(instances, transferred)],
    }


if __name__ == "__main__":
    fig_geometry()
    fig_onehot()
    fig_e2l()
    fig_vs_K()
    fig_e4l()
    fig_tuning()
    (HERE / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float), encoding="utf-8")
    print("ok")
