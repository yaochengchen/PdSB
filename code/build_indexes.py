#!/usr/bin/env python3
"""Rebuild the two lookup tables at the repository root from the raw runs.jsonl archives.

    python code/build_indexes.py            # writes archive_index.csv and parameter_grids.csv at the repo root
    python code/build_indexes.py --check    # only compares with the existing files, writes nothing

archive_index.csv   one row per archive: role, number of records (JSONL lines), number of
                    configurations (sum of len(configs)), graphs present, file size in bytes.
parameter_grids.csv one row per configuration: archive path, 1-based JSONL line, method, graph,
                    K, T, seed, runs per configuration, index into `configs`, and the exact params.

Both tables are lookup aids derived from the archives; they add no data.  Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (archive path relative to the repository root, role) in the order used by the tables.
ARCHIVES = [
    ("code/results/E1/runs.jsonl", "small-graph tuning/ablation"),
    ("code/results/E1v/runs.jsonl", "fresh-seed validation"),
    ("code/results/E2/runs.jsonl", "legacy timing retained as figure reference input"),
    ("code/results/E2L/runs.jsonl", "large-batch timing"),
    ("code/results/E3/runs.jsonl", "more-colours tuning"),
    ("code/results/E3v/runs.jsonl", "more-colours validation"),
    ("code/results/E4/runs.jsonl", "larger-graph tuning"),
    ("code/results/E4L/runs.jsonl", "larger-graph timing"),
    ("code/results_initial/E4/runs.jsonl", "initial larger-graph snapshot for Fig. 3"),
    ("code/results_pdsb/H1/runs.jsonl", "dedicated search initial tuning"),
    ("code/results_pdsb/H2/runs.jsonl", "dedicated search selected settings"),
    ("code/results_pdsb/H3/runs.jsonl", "dedicated long search"),
]


def records(path: Path):
    """Yield (1-based line number, record) for every non-empty JSONL line."""
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if line.strip():
                yield i, json.loads(line)


def build():
    index = io.StringIO()
    grids = io.StringIO()
    wi = csv.writer(index)
    wg = csv.writer(grids)
    wi.writerow(["archive", "role", "records", "configurations", "graphs", "bytes"])
    wg.writerow(["archive", "line", "method", "graph", "K", "T", "seed", "runs_per_config",
                 "config_index", "params_json"])
    for rel, role in ARCHIVES:
        path = ROOT / rel
        n_rec = n_cfg = 0
        graphs = set()
        for line, rec in records(path):
            n_rec += 1
            graphs.add(rec["graph"])
            for j, cfg in enumerate(rec["configs"]):
                n_cfg += 1
                wg.writerow([rel, line, rec["method"], rec["graph"], rec["K"], rec["T"], rec["seed"],
                             rec["runs"], j, json.dumps(cfg["params"])])
        wi.writerow([rel, role, n_rec, n_cfg, ";".join(sorted(graphs)), path.stat().st_size])
    return {"archive_index.csv": index.getvalue(), "parameter_grids.csv": grids.getvalue()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the existing files only")
    a = ap.parse_args()
    ok = True
    for name, text in build().items():
        target = ROOT / name
        if a.check:
            old = target.read_bytes().decode("utf-8") if target.exists() else None
            same = old == text
            ok &= same
            print(f"{name}: {'identical' if same else 'DIFFERS'}")
        else:
            target.write_bytes(text.encode("utf-8"))   # csv module writes \r\n line endings
            print(f"wrote {target}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
