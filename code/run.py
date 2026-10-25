#!/usr/bin/env python3
"""Driver for the PdSB experiments on 2x GH200 (one worker per GPU, SA on the Grace cores at the same time).

    python check_results.py                        # before every (re)start
    python run.py --plan smoke --results results_smoke   # ~2 min, pipeline check
    python run.py --plan all                       # wave 1: E1 E3 E4 | wave 2: E1v E3v | wave 3: E2L E4L
    python run.py --plan E1 | E3 | E4              # static grids
    python run.py --plan E1v | E2L                 # need E1 (selection of settings)
    python run.py --plan E3v                       # needs E3
    python run.py --plan E4L                       # needs E4
    python analyze.py results                      # all tables / figures

PdSB record hunt (own folder; its best/ archive then holds PdSB solutions only):
    python run.py --plan hunt --results results_pdsb   # wave 1: H1 (T=1e6 tuning) | wave 2: H2 (1e6 x 1000) + H3 (1e7)
    python run.py --plan H1 --results results_pdsb     # only the tuning wave, then look at it:
    python hunt_report.py results_pdsb --main results

Every plan writes results/<plan>/runs.jsonl (one line per finished job, all its settings inside).  Resume is per
SETTING: a planned job only runs the grid points that no finished job with the same (method, graph, K, T, seed,
runs) has already run, so widened grids add only the new points and killed jobs are simply run again.
A job = (method, graph, K, steps, seed) with its hyper-parameter grid in ONE batch (R = settings x runs).
Timing plans (E2L, E4L): CPU jobs start only after the GPU jobs, so SA is timed without contention.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plans  # noqa: E402
from maxkcut import load_graph, reference_cut  # noqa: E402

GPU_METHODS = {"pdsb", "pdsb_fused", "pbsb", "pdsb_sphere", "clock_dsb", "onehot_dsb", "ref_im"}
POTTS = {"pdsb": ("simplex", "simplex", "discrete"), "pbsb": ("simplex", "simplex", "ballistic"),
         "pdsb_sphere": ("simplex", "sphere", "discrete"), "clock_dsb": ("clock", "sphere", "discrete")}


def job_key(j: dict) -> str:
    return json.dumps({k: j[k] for k in ("method", "graph", "K", "T", "seed", "runs", "grid")}, sort_keys=True)


def expand(grid: list, runs: int):
    """per-replica parameter arrays: setting k occupies replicas [k*runs, (k+1)*runs)."""
    keys = sorted(grid[0])
    return {k: np.repeat(np.array([g[k] for g in grid], dtype=np.float64), runs) for k in keys}


def exact_cuts(g, col, chunk=1024):
    out = []
    c = col.cpu().numpy() if hasattr(col, "cpu") else np.asarray(col)
    e = g.edges
    for r0 in range(0, c.shape[1], chunk):
        cc = c[:, r0:r0 + chunk]
        eq = cc[e[:, 0]] == cc[e[:, 1]]
        out.append(g.total_w - (g.w[:, None] * eq).sum(0))
    return np.concatenate(out)


def save_certificates(g, K, cols, cuts, ref, cert_dir, tag):
    """Every colouring whose exact cut beats the reference (BKS) is saved and re-verified on the CPU in fp64
    from the edge list: results/<plan>/certificates/<graph>_K<K>_<cut>_<tag>.npy (one uint8 colour per vertex)."""
    if ref is None or cert_dir is None:
        return []
    c = cols.cpu().numpy() if hasattr(cols, "cpu") else np.asarray(cols)
    idx = np.flatnonzero(np.asarray(cuts) > ref + 1e-9)
    found = []
    seen = set()
    for r in idx:
        col = c[:, r].astype(np.uint8)
        cut = float(g.total_w - (g.w * (col[g.edges[:, 0]] == col[g.edges[:, 1]])).sum())
        if cut <= ref or col.tobytes() in seen or len(found) >= 5:
            continue
        seen.add(col.tobytes())
        Path(cert_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cert_dir) / f"{g.name}_K{K}_{cut:.0f}_{tag}_r{r}.npy", col)
        found.append(cut)
    return found


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in name)


def save_best(g, K, j, candidates, best_dir, plan):
    """Archive of the best colouring ever found per (graph, K), across all plans / methods / settings, with or
    without a BKS:  results/best/<graph>_K<K>.npy (uint8 colour per vertex) + .json (cut, source).
    candidates: list of (tag, colours (n, R), exact cuts (R,)).  The cut is recomputed in fp64 from the edge list
    before it is accepted; every improvement is also appended to results/best/history.jsonl.  A file lock makes
    the GPU and CPU workers safe against each other.  Returns the new record cut if this job improved it."""
    if best_dir is None:
        return None
    tag, cols, cuts = max(((t, c, np.asarray(x)) for t, c, x in candidates), key=lambda t: t[2].max())
    r = int(np.argmax(cuts))
    c = cols.cpu().numpy() if hasattr(cols, "cpu") else np.asarray(cols)
    col = c[:, r].astype(np.uint8)
    cut = float(g.total_w - (g.w * (col[g.edges[:, 0]] == col[g.edges[:, 1]])).sum())
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe(g.name)}_K{K}"
    with open(best_dir / ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            meta_p = best_dir / f"{stem}.json"
            old = json.loads(meta_p.read_text())["cut"] if meta_p.exists() else -np.inf
            if cut <= old + 1e-9:
                return None
            meta = {"graph": g.name, "K": int(K), "n": int(g.n), "cut": cut, "previous": None if old == -np.inf else old,
                    "ref": reference_cut(g.name, K), "plan": plan, "method": j["method"], "T": j["T"],
                    "seed": j["seed"], "params": j["grid"][r // j["runs"]], "replica": r, "kind": tag,
                    "colours_used": int(len(np.unique(col))), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            tmp = best_dir / f".{stem}.tmp.npy"
            np.save(tmp, col)
            os.replace(tmp, best_dir / f"{stem}.npy")
            tmpj = best_dir / f".{stem}.tmp.json"
            tmpj.write_text(json.dumps(meta, indent=1))
            os.replace(tmpj, meta_p)
            with open(best_dir / "history.jsonl", "a") as h:
                h.write(json.dumps(meta) + "\n")
            return cut
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def run_job(j: dict, cache: dict, device: str, cert_dir=None) -> dict:
    import torch
    from machines import DeviceGraph, OneHotSB, PdSB, RefIM
    g = load_graph(j["graph"])
    K, T, runs, grid = j["K"], j["T"], j["runs"], j["grid"]
    # results/<plan>/certificates -> results/best (shared by all plans)
    best_dir = Path(cert_dir).parent.parent / "best" if cert_dir is not None else None
    plan = Path(cert_dir).parent.name if cert_dir is not None else ""
    R = runs * len(grid)
    P = expand(grid, runs)
    rec = {k: j[k] for k in ("method", "graph", "K", "T", "seed", "runs", "grid")}
    rec.update(n=g.n, m=g.m, total_w=g.total_w, ref=reference_cut(g.name, K), host=socket.gethostname(),
               R_total=R, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    m = j["method"]
    bpol = None
    if m == "sa":
        from sa_potts import run_sa
        fin, bst, cols, bcols, t = [], [], [], [], 0.0
        for k, prm in enumerate(grid):
            o = run_sa(g, K, runs, T, prm["T0"], prm["T1"], seed=j["seed"] * 1000 + k)
            fin.append(o["final_cut"]); bst.append(o["best_cut"]); cols.append(o["final_col"]); t += o["t_dyn"]
            bcols.append(o["best_col"])
        fin, bst = np.concatenate(fin), np.concatenate(bst)
        final_col = torch.as_tensor(np.concatenate(cols, 1))
        best_col_sa = np.concatenate(bcols, 1)
        dkey = ("cpu", j["graph"])
        if dkey not in cache:
            cache[dkey] = DeviceGraph(g, "cpu")
        t0 = time.perf_counter()
        pcol = cache[dkey].polish(final_col, K)
        pol = exact_cuts(g, pcol)
        rec["certificates"] = save_certificates(g, K, pcol, pol, rec["ref"], cert_dir, "sa_pol")
        rec["new_best"] = save_best(g, K, j, [("best", best_col_sa, exact_cuts(g, best_col_sa)), ("polished", pcol, pol)],
                                    best_dir, plan)
        rec.update(t_dyn=t, t_polish=time.perf_counter() - t0, device="cpu",
                   threads=int(os.environ.get("NUMBA_NUM_THREADS", 0)), steps_unit="sweeps")
    else:
        # dense W (FP16 tensor-core GEMM) up to 4096 vertices; above: CSR (fused PdSB: sparse gather kernel)
        dense_max = int(os.environ.get("PDSB_DENSE_MAX", 4096))
        dkey = (device, j["graph"], dense_max)
        if dkey not in cache:
            cache[dkey] = DeviceGraph(g, device, dense_max=dense_max)
        dg = cache[dkey]
        seed = j["seed"]
        if m == "pdsb_fused":
            from fused import FusedPdSB
            mach = FusedPdSB(dg, K, R, T, P["xi"], P["dt"], seed=seed)
        elif m in POTTS:
            embed, wall, coupling = POTTS[m]
            mach = PdSB(dg, K, R, T, P["xi"], P["dt"], embed=embed, wall=wall, coupling=coupling, seed=seed)
        elif m == "onehot_dsb":
            mach = OneHotSB(dg, K, R, T, P["xi"], P["dt"], P["A"], P["zeta"], seed=seed)
        elif m == "ref_im":
            mach = RefIM(dg, K, R, T, P["BA_n"], P["zeta"], seed=seed)
        else:
            raise ValueError(m)
        out = mach.run()
        fin = exact_cuts(g, out["final_col"])
        bst = exact_cuts(g, out["best_col"])
        trk = out["best_cut"].double().cpu().numpy()
        rec["tracked_mismatch"] = int(np.sum(np.abs(trk - bst) > 1e-6))     # must be 0
        if hasattr(mach, "onehot_violation"):
            v = ((mach.x > 0).sum(-1) != 1).float().mean(0).cpu().numpy()
            rec["onehot_violation"] = [float(v[k * runs:(k + 1) * runs].mean()) for k in range(len(grid))]
        t0 = time.perf_counter()
        pcol = dg.polish(out["final_col"], K)
        pol = exact_cuts(g, pcol)
        rec["certificates"] = (save_certificates(g, K, out["best_col"], bst, rec["ref"], cert_dir, f"{m}_best")
                               + save_certificates(g, K, pcol, pol, rec["ref"], cert_dir, f"{m}_pol"))
        cands = [("best", out["best_col"], bst), ("polished", pcol, pol)]
        bpol = None
        if j.get("polish_best"):                     # hunt plans: 1-opt on the best-seen colouring as well
            bcol = dg.polish(out["best_col"], K)
            bpol = exact_cuts(g, bcol)
            rec["certificates"] += save_certificates(g, K, bcol, bpol, rec["ref"], cert_dir, f"{m}_bestpol")
            cands.append(("best_polished", bcol, bpol))
            del bcol
        rec["new_best"] = save_best(g, K, j, cands, best_dir, plan)
        if dg.device.type == "cuda":
            torch.cuda.synchronize(dg.device)
        rec["sparse"] = not dg.dense
        rec.update(t_dyn=out["t_dyn"], t_polish=time.perf_counter() - t0, cuda_graph=bool(mach.graph_used),
                   exact16=bool(dg.exact16), dense=bool(dg.dense),
                   device=torch.cuda.get_device_name(dg.device) if dg.device.type == "cuda" else "cpu")
        del mach, out
    rec["configs"] = [{"params": grid[k],
                       "final": np.round(fin[k * runs:(k + 1) * runs], 6).tolist(),
                       "best": np.round(bst[k * runs:(k + 1) * runs], 6).tolist(),
                       "polished": np.round(pol[k * runs:(k + 1) * runs], 6).tolist()} for k in range(len(grid))]
    if m != "sa" and bpol is not None:
        for k, c in enumerate(rec["configs"]):
            c["best_polished"] = np.round(bpol[k * runs:(k + 1) * runs], 6).tolist()
    return rec


# ----------------------------------------------------------------------------------------------
def append_jsonl(path: Path, rec: dict):
    """One line per finished job.  If a previous write was cut off (Ctrl+C), the file does not end in a newline:
    start a new line first so the truncated record stays isolated (done_keys / load_runs skip it)."""
    with open(path, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        end = f.tell()
        prefix = b""
        if end > 0:
            f.seek(end - 1)
            if f.read(1) != b"\n":
                prefix = b"\n"
        f.seek(0, os.SEEK_END)
        f.write(prefix + (json.dumps(rec) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def pin_to_gpu_numa(gpu: int):
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout
        bus = {int(a): b.strip() for a, b in (l.split(",") for l in q.strip().splitlines())}[gpu]
        dom, rest = bus.split(":", 1)
        node = int(Path(f"/sys/bus/pci/devices/{int(dom, 16):04x}:{rest}".lower() + "/numa_node").read_text())
        if node < 0:
            return None
        cset = set()
        for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
            a, _, b = part.partition("-")
            cset.update(range(int(a), int(b or a) + 1))
        if cset:
            os.sched_setaffinity(0, cset)
            return node
    except Exception:
        return None


def summary_line(rec):
    # progress line only: without a BKS the reference is the best cut of this job (analyze.py uses all plans)
    ref = rec["ref"] or max(max(c["best"]) for c in rec["configs"])
    gaps = [100 * (ref - np.mean(c["best"])) / ref for c in rec["configs"]]
    k = int(np.argmin(gaps))
    return (f"{rec['method']:<11} {rec['graph']:<15} K={rec['K']} T={rec['T']:<8} R={rec['R_total']:<6} "
            f"best mean-gap {gaps[k]:.3f}% {json.dumps(rec['configs'][k]['params'])}  {rec['t_dyn']:.1f}s"
            + (f"  MISMATCH {rec['tracked_mismatch']}" if rec.get("tracked_mismatch") else "")
            + (f"  new best {rec['new_best']:.0f} -> results/best" if rec.get("new_best") else "")
            + (f"  ** BEATS BKS {rec['ref']:.0f}: {sorted(set(rec['certificates']))} (saved) **" if rec.get("certificates") else ""))


def gpu_worker(gpu, q, pin, ext_ok):
    import torch
    node = pin_to_gpu_numa(gpu) if pin else None
    torch.set_num_threads(4)
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    print(f"[gpu{gpu}] {torch.cuda.get_device_name(device)} numa={node}", flush=True)
    cache = {}
    while True:
        item = q.get()
        if item is None:
            break
        j, out_path = item
        try:
            rec = run_job(j, cache, device, Path(out_path).parent / "certificates")
            rec["gpu"] = gpu
            append_jsonl(Path(out_path), rec)
            print(f"[gpu{gpu}] {summary_line(rec)}", flush=True)
        except Exception:
            print(f"[gpu{gpu}] FAILED {job_key(j)[:300]}\n{traceback.format_exc()}", flush=True)
        finally:
            if len(cache) > 6:
                cache.clear()
            torch.cuda.empty_cache()


def cpu_worker(items, threads):
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    import torch
    torch.set_num_threads(min(16, threads))
    cache = {}
    for j, out_path in items:
        try:
            rec = run_job(j, cache, "cpu", Path(out_path).parent / "certificates")
            append_jsonl(Path(out_path), rec)
            print(f"[cpu ] {summary_line(rec)}", flush=True)
        except Exception:
            print(f"[cpu ] FAILED {job_key(j)[:300]}\n{traceback.format_exc()}", flush=True)


def done_keys(path: Path) -> set:
    s = set()
    if path.exists():
        for line in open(path):
            try:
                s.add(job_key(json.loads(line)))
            except Exception:
                pass
    return s


def _cell(j: dict) -> str:
    return json.dumps({k: j[k] for k in ("method", "graph", "K", "T", "seed", "runs")}, sort_keys=True)


def covered_settings(path: Path) -> dict:
    """(method, graph, K, T, seed, runs) -> set of settings already finished in this plan's runs.jsonl."""
    cov = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                cov.setdefault(_cell(r), set()).update(json.dumps(p, sort_keys=True) for p in r["grid"])
            except Exception:
                pass
    return cov


def pending(jobs: list, path: Path) -> list:
    """Per-setting resume: every planned job reduced to the settings not finished yet; empty jobs dropped."""
    cov = covered_settings(path)
    out = []
    for j in jobs:
        done = cov.get(_cell(j), set())
        left = [p for p in j["grid"] if json.dumps(p, sort_keys=True) not in done]
        if left:
            out.append(dict(j, grid=left))
    return out


def refim_finished(e1_dir: Path) -> set:
    """(graph, K) of ref_im jobs already finished in E1 (any line that does not parse is ignored)."""
    out = set()
    p = e1_dir / "runs.jsonl"
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("method") == "ref_im":
                out.add((r["graph"], int(r["K"])))
    return out


def hunt_select(h1_dir: Path) -> dict:
    """Settings for H2/H3 from H1, per (graph, K): the 2 best by mean best-seen cut, then the best by the
    maximum cut over all runs (best-seen or 1-opt polished) if it is not one of them.  Order = mean rank."""
    from analyze import load_runs
    stats = {}
    for r in load_runs(h1_dir):
        if r["method"] != "pdsb_fused":
            continue
        for c in r["configs"]:
            key = (r["graph"], int(r["K"]))
            b = np.asarray(c["best"], dtype=float)
            mx = max(b.max(), max(c["polished"]), max(c.get("best_polished", [-np.inf])))
            stats.setdefault(key, []).append((float(b.mean()), float(mx), c["params"]))
    sel = {}
    for key, v in stats.items():
        by_mean = sorted(v, key=lambda t: -t[0])
        pick = [t[2] for t in by_mean[:2]]
        top_max = max(v, key=lambda t: (t[1], t[0]))[2]
        if top_max not in pick:
            pick.append(top_max)
        sel[key] = pick
    return sel


def build_plan(name: str, results: Path, top: int = 3) -> list:
    from analyze import best_settings
    if name == "smoke":
        return plans.plan_smoke()
    if name == "E1":
        done = refim_finished(results / "E1")
        return plans.plan_E1(done)
    if name == "E3":
        return plans.plan_E3()
    if name == "E4":
        return plans.plan_E4()
    if name == "H1":
        return plans.plan_H1()
    if name in ("H2", "H3"):
        src = results / "H1"
        if not (src / "runs.jsonl").exists():
            raise SystemExit(f"plan {name} needs {src}/runs.jsonl (run H1 first)")
        sel = hunt_select(src)
        return plans.plan_H2(sel) if name == "H2" else plans.plan_H3(sel)
    need = {"E1v": "E1", "E2": "E1", "E2L": "E1", "E3v": "E3", "E4L": "E4"}.get(name)
    if need is None:
        raise SystemExit(f"unknown plan {name}")
    src = results / need
    if not (src / "runs.jsonl").exists():
        raise SystemExit(f"plan {name} needs {src}/runs.jsonl (run {need} first)")
    if name == "E1v":
        return plans.plan_E1v(best_settings(src, "per_T"))
    if name == "E3v":
        return plans.plan_E3v(best_settings(src, "per_T"))
    if name == "E2":
        return plans.plan_E2(best_settings(src, "per_graph"))
    if name == "E2L":
        return plans.plan_E2L(best_settings(src, "per_graph"))
    if name == "E4L":
        return plans.plan_E4L(best_settings(src, "per_graph"))


def execute(jobs_by_plan: dict, results: Path, gpus, cpu_threads, pin, dry):
    import torch.multiprocessing as mp
    todo_gpu, todo_cpu = [], []
    exclusive = False
    for name, jobs in jobs_by_plan.items():
        out = results / name
        out.mkdir(parents=True, exist_ok=True)
        path = out / "runs.jsonl"
        left = pending(jobs, path)
        n_set = sum(len(j["grid"]) for j in jobs)
        n_left = sum(len(j["grid"]) for j in left)
        print(f"plan {name}: {len(jobs)} jobs / {n_set} settings planned; to run: {len(left)} jobs / {n_left} "
              f"settings", flush=True)
        for j in left:
            (todo_gpu if j["method"] in GPU_METHODS else todo_cpu).append((j, str(path)))
        exclusive |= name in plans.TIMING_PLANS
    if todo_cpu:
        import importlib.util
        if importlib.util.find_spec("numba") is None:
            print(f"[warn] numba missing: {len(todo_cpu)} SA jobs skipped", flush=True)
            todo_cpu = []
    if dry:
        for j, p in (todo_gpu + todo_cpu)[:30]:
            print(Path(p).parent.name, j["method"], j["graph"], j["K"], j["T"], len(j["grid"]), "settings x", j["runs"])
        return
    for j, _ in todo_gpu + todo_cpu:
        load_graph(j["graph"])                     # download once
    # heaviest first: steps x replicas x K
    todo_gpu.sort(key=lambda it: -it[0]["T"] * len(it[0]["grid"]) * it[0]["runs"] * it[0]["K"])
    if todo_gpu and not gpus:
        cache = {}
        for j, p in todo_gpu:
            try:
                rec = run_job(j, cache, "cpu", Path(p).parent / "certificates")
                append_jsonl(Path(p), rec)
                print(f"[cpu ] {summary_line(rec)}", flush=True)
            except Exception as e:
                print(f"[cpu ] skipped {j['method']} {j['graph']} K={j['K']}: {type(e).__name__}: {e}", flush=True)
        todo_gpu = []
    ctx = mp.get_context("spawn")
    procs = []
    if todo_gpu:
        q = ctx.Queue()
        for it in todo_gpu:
            q.put(it)
        for _ in gpus:
            q.put(None)
        for gpu in gpus:
            p = ctx.Process(target=gpu_worker, args=(gpu, q, pin, True))
            p.start()
            procs.append(p)
    if todo_cpu and exclusive:
        # timing plans: SA gets the whole CPU, after the GPU workers are done
        for p in procs:
            p.join()
        procs = []
        print(f"timing plan: GPU jobs done, now {len(todo_cpu)} CPU jobs alone", flush=True)
    if todo_cpu:
        p = ctx.Process(target=cpu_worker, args=(todo_cpu, cpu_threads))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="smoke", help="smoke | E1 | E1v | E2 | E2L | E3 | E3v | E4 | E4L | all | H1 | H2 | H3 | hunt")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--gpus", nargs="*", default=["0", "1"], help="GPU ids; 'none' = run GPU methods on the CPU")
    ap.add_argument("--cpu-threads", type=int, default=max(1, (os.cpu_count() or 8) - 8))
    ap.add_argument("--top", type=int, default=3, help="(unused, kept for old command lines)")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gpus = [] if a.gpus in (["none"], []) else [int(x) for x in a.gpus]
    t0 = time.time()
    hunt = a.plan == "hunt" or a.plan in plans.HUNT_PLANS
    if hunt and a.results == Path("results"):
        a.results = Path("results_pdsb")             # keep the PdSB-only archive apart from the main results
        print(f"hunt plans -> {a.results}/", flush=True)
    if not hunt and (a.results / "H1").exists():
        raise SystemExit(f"{a.results} holds hunt results (H1/); use another --results for the E plans")
    if hunt and any((a.results / p).exists() for p in plans.PLAN_ORDER):
        raise SystemExit(f"{a.results} holds E-plan results; the hunt needs its own folder (e.g. results_pdsb)")
    if a.plan == "all":
        waves = plans.WAVES_ALL
    elif a.plan == "hunt":
        waves = plans.WAVES_HUNT
    else:
        waves = [[a.plan]]
    for wave in waves:
        jobs = {name: build_plan(name, a.results, a.top) for name in wave}
        execute(jobs, a.results, gpus, a.cpu_threads, not a.no_pin, a.dry_run)
        if a.dry_run and len(waves) > 1:
            print("(dry run: later waves need the results of this one)")
            break
    print(f"finished in {(time.time() - t0) / 3600:.2f} h.  Next: python analyze.py {a.results}", flush=True)


if __name__ == "__main__":
    main()
