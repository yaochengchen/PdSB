#!/usr/bin/env python3
"""Pre-restart check after killed runs.  Read-only unless --clean.

    python check_results.py                  # results/ ; run this BEFORE  python run.py --plan all
    python check_results.py --clean          # also rewrite each runs.jsonl without truncated / duplicate lines
                                             # (the original is kept as runs.jsonl.bak-<time>)

Checks
  1. leftover processes: run.py / its spawned workers still alive, and compute processes on the GPUs.
     A killed parent can leave workers that keep appending to runs.jsonl -> kill them before restarting.
  2. every results/<plan>/runs.jsonl: lines that do not parse (cut off by a kill), missing final newline,
     the same job recorded more than once.
  3. against the CURRENT plans.py: settings done / still to run per plan (resume is per setting).
  4. certificates (> BKS) and the best-colouring archive results/best/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run  # noqa: E402


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"({cmd[0]} not available: {type(e).__name__})"


def check_processes():
    print("== 1. processes")
    me = os.getpid()
    ps = [l for l in sh(["ps", "-eo", "pid,ppid,etime,args"]).splitlines()
          if ("run.py" in l or "multiprocessing.spawn" in l) and "check_results" not in l and l.split()[0] != str(me)]
    if ps:
        print("  STILL RUNNING (kill these before restarting, e.g. kill <pid>; kill -9 if they survive):")
        for l in ps:
            print("   ", l)
    else:
        print("  no run.py / spawned worker processes")
    gpu = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    print("  GPU compute processes:", gpu if gpu else "none")
    return bool(ps)


def scan(path: Path):
    lines = path.read_bytes().split(b"\n")
    tail_ok = path.read_bytes().endswith(b"\n") or path.stat().st_size == 0
    good, bad, keys = [], 0, Counter()
    for raw in lines:
        if not raw.strip():
            continue
        try:
            r = json.loads(raw)
            k = run.job_key(r)
        except Exception:
            bad += 1
            continue
        keys[k] += 1
        if keys[k] == 1:
            good.append(raw)
    dup = sum(v - 1 for v in keys.values())
    return good, bad, dup, tail_ok, set(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    alive = check_processes()

    print("\n== 2. runs.jsonl integrity")
    done = {}
    for p in sorted(a.results.glob("*/runs.jsonl")):
        good, bad, dup, tail_ok, keys = scan(p)
        done[p.parent.name] = keys
        flag = "OK" if not (bad or dup or not tail_ok) else "needs attention (harmless for run.py; --clean tidies it)"
        print(f"  {p}: {len(good)} jobs, {bad} unreadable line(s), {dup} duplicate(s), "
              f"ends with newline: {tail_ok}  -> {flag}")
        if a.clean and (bad or dup or not tail_ok):
            if alive:
                print("    --clean skipped: processes still running")
                continue
            bak = p.with_name(f"runs.jsonl.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            os.replace(p, bak)
            p.write_bytes(b"".join(g + b"\n" for g in good))
            print(f"    cleaned; original kept as {bak.name}")

    print("\n== 3. current plans.py vs results (per setting)")
    import plans
    hunt = any((a.results / p).exists() for p in plans.HUNT_PLANS) or a.results.name == "results_pdsb"
    for wave_no, wave in enumerate(plans.WAVES_HUNT if hunt else plans.WAVES_ALL, 1):
        for name in wave:
            try:
                jobs = run.build_plan(name, a.results)
            except (SystemExit, Exception) as e:
                print(f"  wave {wave_no} {name}: built after its source plan is complete ({type(e).__name__}: {e})")
                continue
            left = run.pending(jobs, a.results / name / "runs.jsonl")
            n_set = sum(len(j["grid"]) for j in jobs)
            n_left = sum(len(j["grid"]) for j in left)
            by = Counter()
            for j in left:
                by[j["method"]] += len(j["grid"])
            print(f"  wave {wave_no} {name}: {n_set} settings planned, {n_set - n_left} done, {n_left} to run  {dict(by)}")
    print("  (wave 2/3 plans are rebuilt from the finished wave-1/2 results when run.py reaches them; the counts")
    print("   above use the results as they are now)")
    if "E1" in done:
        print("  E1 ref_im finished:", sorted(run.refim_finished(a.results / "E1")))

    print("\n== 4. solutions")
    certs = sorted(a.results.glob("*/certificates/*.npy"))
    print(f"  certificates (> BKS): {len(certs)}" + "".join(f"\n    {c}" for c in certs[:20]))
    best = sorted((a.results / "best").glob("*.json"))
    for b in best:
        m = json.loads(b.read_text())
        print(f"  best {m['graph']} K={m['K']}: {m['cut']:.0f} (BKS {m.get('ref')}) from {m['plan']}/{m['method']}")
    if not best:
        print(f"  {a.results}/best/ empty (filled by the next run)")
    print(f"\nverify solutions: python verify_best.py --best {a.results}/best ; "
          f"python verify_cert.py {a.results}/*/certificates/*.npy")
    if alive:
        print("\n!! processes still running: stop them before  python run.py")
        sys.exit(2)


if __name__ == "__main__":
    main()
